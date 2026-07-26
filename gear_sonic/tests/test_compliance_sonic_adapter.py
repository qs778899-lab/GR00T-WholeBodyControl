"""IsaacLab-free contract tests for the thin SONIC compliance adapter."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from gear_sonic.compliance_control.adapters.sonic.event import (
    _set_body_wrench,
    apply_compliance_wrench,
    reset_compliance_wrench,
    transition_compliance_operational_state,
)
from gear_sonic.compliance_control.adapters.sonic.frames import (
    _align_trailing_components,
    _common_to_world_vectors_unchecked,
    _rotate_vectors_wxyz_unchecked,
    _world_to_body_vectors_unchecked,
    _world_to_common_positions_unchecked,
    common_to_world_vectors,
    world_to_common_positions,
)
from gear_sonic.compliance_control.adapters.sonic.mapping import resolve_body_index_map
from gear_sonic.compliance_control.adapters.sonic.state import (
    ComplianceCommandState,
    ComplianceSamplingSpec,
)
from gear_sonic.compliance_control.adapters.sonic.validation import site_body_offsets_tensor
from gear_sonic.compliance_control.adapters.sonic.wrench import ResidualWrenchLimiter
from gear_sonic.compliance_control.core import (
    encode_compliance_condition,
    virtual_force_from_reference_delta,
)
from gear_sonic.compliance_control.core.math import _clamp_vector_norm_unchecked
from gear_sonic.compliance_control.core.reference_modifier import (
    _expanded_current_reference,
    _expanded_enabled_unchecked,
    _expanded_site_data_unchecked,
    _reshape_site_data,
    _virtual_force_from_reference_delta_unchecked,
)


ROOT = Path(__file__).parents[2]
COMPLIANCE_PACKAGE = ROOT / "gear_sonic" / "compliance_control"


def test_reference_and_articulation_indices_resolve_independently_for_full_and_partial_sites():
    reference_names = ["anchor", "right_site", "unused", "left_site"]
    articulation_names = ["left_site", "anchor", "right_site", "extra"]

    full = resolve_body_index_map(
        reference_names,
        articulation_names,
        ["left_site", "right_site"],
        "anchor",
    )
    partial = resolve_body_index_map(
        reference_names,
        articulation_names,
        ["right_site"],
        "anchor",
    )

    assert full.reference_site_indices == (3, 1)
    assert full.articulation_site_indices == (0, 2)
    assert full.reference_anchor_index == 0
    assert full.articulation_anchor_index == 1
    assert partial.reference_site_indices == (1,)
    assert partial.articulation_site_indices == (2,)
    with pytest.raises(ValueError, match="missing"):
        resolve_body_index_map(reference_names, articulation_names, ["missing"], "anchor")


@pytest.mark.parametrize(
    ("reference_names", "articulation_names", "site_names", "anchor_name", "error"),
    [
        ("anchor", ["anchor", "site"], ["site"], "anchor", TypeError),
        (["anchor", "site"], b"anchor", ["site"], "anchor", TypeError),
        ([], ["anchor", "site"], ["site"], "anchor", ValueError),
        (["anchor", ""], ["anchor", "site"], ["site"], "anchor", ValueError),
        (["anchor", 7], ["anchor", "site"], ["site"], "anchor", TypeError),
        (["anchor", "anchor"], ["anchor", "site"], ["site"], "anchor", ValueError),
        (["anchor", "site"], ["anchor", "site"], "site", "anchor", TypeError),
        (["anchor", "site"], ["anchor", "site"], ["site"], "", ValueError),
    ],
)
def test_body_mapping_rejects_ambiguous_or_malformed_names(
    reference_names,
    articulation_names,
    site_names,
    anchor_name,
    error,
):
    with pytest.raises(error):
        resolve_body_index_map(
            reference_names,
            articulation_names,
            site_names,
            anchor_name,
        )


def test_site_body_offsets_validate_exact_shape_and_finiteness_before_runtime():
    valid = site_body_offsets_tensor(
        [[0, 0.1, 0], [0, -0.1, 0]],
        num_sites=2,
        device="cpu",
    )
    assert valid.shape == (2, 3)
    assert valid.dtype == torch.float32

    for malformed in (
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]],
        [[0.0, 0.0, 0.0], [0.0, float("inf"), 0.0]],
        [[False, False, False], [False, False, False]],
        "not-an-offset-array",
    ):
        with pytest.raises((TypeError, ValueError)):
            site_body_offsets_tensor(malformed, num_sites=2, device="cpu")


def test_nonzero_rotation_round_trip_supports_batch_future_site_broadcasting():
    batch_size, num_future, num_sites = 2, 3, 5
    half_sqrt = math.sqrt(0.5)
    quaternion = torch.tensor(
        [
            [half_sqrt, 0.0, 0.0, half_sqrt],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    vectors_common = torch.zeros(
        (batch_size, num_future, num_sites, 3),
        dtype=torch.float64,
    )
    vectors_common[..., 0] = 1.0

    vectors_world = common_to_world_vectors(vectors_common, quaternion)
    vectors_world_fast = _common_to_world_vectors_unchecked(vectors_common, quaternion)
    torch.testing.assert_close(vectors_world_fast, vectors_world, atol=1.0e-12, rtol=0.0)

    torch.testing.assert_close(
        vectors_world[0, ..., :2],
        torch.tensor([0.0, 1.0], dtype=torch.float64).expand(num_future, num_sites, 2),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(vectors_world[1], vectors_common[1])

    origin = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]], dtype=torch.float64)
    positions_world = vectors_world + origin[:, None, None]
    recovered = world_to_common_positions(positions_world, origin, quaternion)
    recovered_fast = _world_to_common_positions_unchecked(
        positions_world,
        origin,
        quaternion,
    )
    torch.testing.assert_close(recovered, vectors_common, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(recovered_fast, recovered, atol=1.0e-12, rtol=0.0)


def test_current_body_quaternion_converts_world_wrench_to_fresh_local_frame():
    half_sqrt = math.sqrt(0.5)
    body_quaternion_world = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
        ],
        dtype=torch.float64,
    )
    force_world = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).repeat(2, 1)
    torque_world = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64).repeat(2, 1)

    force_body = _world_to_body_vectors_unchecked(force_world, body_quaternion_world)
    torque_body = _world_to_body_vectors_unchecked(torque_world, body_quaternion_world)

    torch.testing.assert_close(force_body[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(force_body[1], torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(torque_body[0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(torque_body[1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_seeded_state_sampling_is_deterministic_and_threshold_kp_coupled():
    sampling = ComplianceSamplingSpec(
        enable_probability=1.0,
        site_activation_probability=0.5,
    )
    first = ComplianceCommandState(8, 4, 3, sampling, seed=123)
    second = ComplianceCommandState(8, 4, 3, sampling, seed=123)

    first.reset()
    second.reset()

    for first_value, second_value in (
        (first.enabled, second.enabled),
        (first.active_site_mask, second.active_site_mask),
        (first.force_threshold_n, second.force_threshold_n),
        (first.reference_offset_common, second.reference_offset_common),
    ):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.stiffness_n_per_m,
        first.force_threshold_n / sampling.reference_displacement_m,
    )
    assert first.condition.shape == (8, 3)
    torch.testing.assert_close(
        first.condition,
        encode_compliance_condition(
            first.enabled,
            first.force_threshold_n,
            sampling.reference_displacement_m,
        ),
    )
    cached_condition = first.condition
    assert first.condition is cached_condition


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_command_owned_sampling_never_advances_global_rng(device):
    first = ComplianceCommandState(
        4,
        2,
        3,
        ComplianceSamplingSpec(enable_probability=1.0),
        device=device,
        seed=19,
    )
    second = ComplianceCommandState(
        4,
        2,
        3,
        ComplianceSamplingSpec(enable_probability=1.0),
        device=device,
        seed=19,
    )
    if device == "cpu":
        global_before = torch.random.get_rng_state()
    else:
        global_before = torch.cuda.get_rng_state(device)

    first_duration = first.sample_resampling_time(4, (2.0, 16.0))
    first.reset()
    second_duration = second.sample_resampling_time(4, (2.0, 16.0))
    second.reset()

    if device == "cpu":
        global_after = torch.random.get_rng_state()
    else:
        global_after = torch.cuda.get_rng_state(device)
    assert torch.equal(global_after, global_before)
    torch.testing.assert_close(first_duration, second_duration, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.condition, second.condition, rtol=0.0, atol=0.0)

    owned_before_disable = first.generator.get_state()
    first.disable()
    assert torch.equal(first.generator.get_state(), owned_before_disable)


def test_prevalidated_resample_path_does_not_reenter_checked_id_resolution():
    state = ComplianceCommandState(
        4,
        2,
        3,
        ComplianceSamplingSpec(enable_probability=1.0),
        seed=23,
    )
    ids = torch.tensor([1, 3])

    def checked_resolution_is_forbidden(*args, **kwargs):
        raise AssertionError("prevalidated path re-entered checked ID resolution")

    state._env_ids_tensor = checked_resolution_is_forbidden
    state._resample_prevalidated(ids)
    assert state.enabled[ids].all()
    state._disable_prevalidated(ids)
    assert not state.enabled[ids].any()


@pytest.mark.parametrize("operation_name", ["clear_dynamic", "disable", "resample", "reset"])
def test_public_state_operations_validate_env_ids_exactly_once(operation_name):
    state = ComplianceCommandState(
        4,
        2,
        3,
        ComplianceSamplingSpec(enable_probability=1.0),
        seed=29,
    )
    original_resolver = state._env_ids_tensor
    resolved_inputs = []

    def counting_resolver(env_ids):
        resolved_inputs.append(env_ids)
        return original_resolver(env_ids)

    state._env_ids_tensor = counting_resolver
    ids = torch.tensor([0, 2])
    getattr(state, operation_name)(ids)
    assert len(resolved_inputs) == 1
    assert resolved_inputs[0] is ids


def test_fixed_shape_masked_resample_updates_only_due_rows_with_fixed_rng_cost():
    sampling = ComplianceSamplingSpec(
        enable_probability=1.0,
        site_activation_probability=0.0,
    )
    due_state = ComplianceCommandState(4, 2, 3, sampling, seed=31)
    idle_state = ComplianceCommandState(4, 2, 3, sampling, seed=31)
    due_mask = torch.tensor([False, True, False, True])
    idle_mask = torch.zeros(4, dtype=torch.bool)

    tracked_names = (
        "enabled",
        "active_site_mask",
        "force_threshold_n",
        "stiffness_n_per_m",
        "_condition",
        "reference_offset_common",
        "original_reference_common",
        "compliant_reference_common",
        "current_reference_common",
        "force_common_future",
        "site_force_world",
        "site_torque_world",
        "anchor_force_world",
        "anchor_torque_world",
    )
    for name in tracked_names:
        tensor = getattr(due_state, name)
        if tensor.dtype == torch.bool:
            tensor.fill_(True)
        else:
            tensor.fill_(7.0)
        getattr(idle_state, name).copy_(tensor)
    before = {name: getattr(due_state, name).clone() for name in tracked_names}

    due_state._resample_masked_prevalidated(due_mask)
    idle_state._resample_masked_prevalidated(idle_mask)

    for name in tracked_names:
        torch.testing.assert_close(
            getattr(due_state, name)[~due_mask],
            before[name][~due_mask],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            getattr(idle_state, name),
            before[name],
            rtol=0.0,
            atol=0.0,
        )
    for name in (
        "original_reference_common",
        "compliant_reference_common",
        "current_reference_common",
        "force_common_future",
        "site_force_world",
        "site_torque_world",
        "anchor_force_world",
        "anchor_torque_world",
    ):
        torch.testing.assert_close(
            getattr(due_state, name)[due_mask],
            torch.zeros_like(getattr(due_state, name)[due_mask]),
            rtol=0.0,
            atol=0.0,
        )
    assert due_state.enabled[due_mask].all()
    assert due_state.active_site_mask[due_mask].sum(dim=-1).tolist() == [1, 1]
    assert torch.equal(due_state.generator.get_state(), idle_state.generator.get_state())


def test_command_masked_resample_preserves_non_due_rows_and_has_fixed_rng_cost(monkeypatch):
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.__path__ = []
    fake_assets = types.ModuleType("isaaclab.assets")
    fake_managers = types.ModuleType("isaaclab.managers")
    fake_assets.Articulation = type("Articulation", (), {})
    fake_managers.CommandTerm = type("CommandTerm", (), {})
    fake_isaaclab.assets = fake_assets
    fake_isaaclab.managers = fake_managers
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", fake_assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", fake_managers)

    module_name = "gear_sonic.compliance_control.adapters.sonic._command_pure_test"
    command_path = COMPLIANCE_PACKAGE / "adapters/sonic/command.py"
    spec = importlib.util.spec_from_file_location(module_name, command_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    def make_command(seed):
        command = object.__new__(module.MotionComplianceCommand)
        command.num_envs = 4
        command.cfg = SimpleNamespace(resampling_time_range=(2.0, 3.0))
        command.state = ComplianceCommandState(
            4,
            2,
            3,
            ComplianceSamplingSpec(
                enable_probability=1.0,
                site_activation_probability=0.0,
            ),
            seed=seed,
        )
        for name in (
            "enabled",
            "active_site_mask",
            "force_threshold_n",
            "stiffness_n_per_m",
            "_condition",
            "reference_offset_common",
            "original_reference_common",
            "compliant_reference_common",
            "current_reference_common",
            "force_common_future",
            "site_force_world",
            "site_torque_world",
            "anchor_force_world",
            "anchor_torque_world",
        ):
            tensor = getattr(command.state, name)
            tensor.fill_(True if tensor.dtype == torch.bool else 7.0)
        command.time_left = torch.tensor([11.0, 12.0, 13.0, 14.0])
        command.command_counter = torch.tensor([3, 4, 5, 6], dtype=torch.long)
        application_shape = (4, 3, 3)
        command._application_force_world = torch.full(application_shape, 21.0)
        command._application_torque_world = torch.full(application_shape, 22.0)
        command._application_force_body = torch.full(application_shape, 23.0)
        command._application_torque_body = torch.full(application_shape, 24.0)
        return command

    mixed = make_command(seed=37)
    alternate = make_command(seed=37)
    due_mask = torch.tensor([True, False, True, False])
    alternate_due_mask = ~due_mask
    state_names = (
        "enabled",
        "active_site_mask",
        "force_threshold_n",
        "stiffness_n_per_m",
        "_condition",
        "reference_offset_common",
        "original_reference_common",
        "compliant_reference_common",
        "current_reference_common",
        "force_common_future",
        "site_force_world",
        "site_torque_world",
        "anchor_force_world",
        "anchor_torque_world",
    )
    application_names = (
        "_application_force_world",
        "_application_torque_world",
        "_application_force_body",
        "_application_torque_body",
    )
    state_before = {name: getattr(mixed.state, name).clone() for name in state_names}
    application_before = {
        name: getattr(mixed, name).clone() for name in application_names
    }
    time_before = mixed.time_left.clone()
    counter_before = mixed.command_counter.clone()

    mixed._resample_masked_prevalidated(due_mask)
    alternate._resample_masked_prevalidated(alternate_due_mask)

    torch.testing.assert_close(mixed.time_left[~due_mask], time_before[~due_mask])
    assert ((mixed.time_left[due_mask] >= 2.0) & (mixed.time_left[due_mask] <= 3.0)).all()
    torch.testing.assert_close(
        mixed.command_counter,
        counter_before + due_mask.to(torch.long),
        rtol=0.0,
        atol=0.0,
    )
    for name in application_names:
        actual = getattr(mixed, name)
        torch.testing.assert_close(
            actual[~due_mask],
            application_before[name][~due_mask],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual[due_mask],
            torch.zeros_like(actual[due_mask]),
            rtol=0.0,
            atol=0.0,
        )
    for name in state_names:
        torch.testing.assert_close(
            getattr(mixed.state, name)[~due_mask],
            state_before[name][~due_mask],
            rtol=0.0,
            atol=0.0,
        )
    for name in (
        "original_reference_common",
        "compliant_reference_common",
        "current_reference_common",
        "force_common_future",
        "site_force_world",
        "site_torque_world",
        "anchor_force_world",
        "anchor_torque_world",
    ):
        actual = getattr(mixed.state, name)[due_mask]
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0.0, atol=0.0)
    assert mixed.state.enabled[due_mask].all()
    assert mixed.state.active_site_mask[due_mask].sum(dim=-1).tolist() == [1, 1]
    assert torch.equal(
        mixed.state.generator.get_state(),
        alternate.state.generator.get_state(),
    )


def test_independent_sampling_supports_simultaneous_single_and_disabled_masks():
    simultaneous = ComplianceCommandState(
        3,
        4,
        2,
        ComplianceSamplingSpec(
            enable_probability=1.0,
            site_activation_probability=1.0,
        ),
        seed=1,
    )
    single = ComplianceCommandState(
        3,
        4,
        2,
        ComplianceSamplingSpec(
            enable_probability=1.0,
            site_activation_probability=0.0,
        ),
        seed=1,
    )
    disabled = ComplianceCommandState(
        3,
        4,
        2,
        ComplianceSamplingSpec(
            enable_probability=0.0,
            site_activation_probability=1.0,
        ),
        seed=1,
    )

    simultaneous.reset()
    single.reset()
    disabled.reset()

    assert simultaneous.active_site_mask.sum(dim=-1).tolist() == [4, 4, 4]
    assert single.active_site_mask.sum(dim=-1).tolist() == [1, 1, 1]
    assert not disabled.active_site_mask.any()
    torch.testing.assert_close(disabled.condition, torch.zeros((3, 3)), rtol=0.0, atol=0.0)


def test_partial_reset_clears_every_dynamic_buffer_without_touching_other_envs():
    state = ComplianceCommandState(
        4,
        2,
        3,
        ComplianceSamplingSpec(enable_probability=1.0),
        seed=5,
    )
    state.reset()
    dynamic = (
        state.original_reference_common,
        state.compliant_reference_common,
        state.current_reference_common,
        state.force_common_future,
        state.site_force_world,
        state.site_torque_world,
        state.anchor_force_world,
        state.anchor_torque_world,
    )
    for tensor in dynamic:
        tensor[:] = 7.0

    state.reset(torch.tensor([1, 3]))

    for tensor in dynamic:
        torch.testing.assert_close(tensor[[1, 3]], torch.zeros_like(tensor[[1, 3]]))
        torch.testing.assert_close(tensor[[0, 2]], torch.full_like(tensor[[0, 2]], 7.0))


def test_per_site_full_formula_is_separately_capped_with_multi_future_state():
    batch_size, num_future, num_sites = 2, 4, 3
    original = torch.zeros((batch_size, num_future, num_sites, 3))
    compliant = original.clone()
    compliant[..., 0] = 0.05
    current = torch.zeros((batch_size, num_sites, 3))
    active = torch.tensor([[True, False, True], [False, True, False]])
    threshold = torch.tensor([10.0, 20.0])

    force = virtual_force_from_reference_delta(
        original,
        compliant,
        active,
        threshold,
        current_reference=current,
    )
    force_fast = _virtual_force_from_reference_delta_unchecked(
        original,
        compliant,
        active,
        threshold,
        current_reference=current,
    )
    norms = torch.linalg.vector_norm(force, dim=-1)

    torch.testing.assert_close(norms[0, :, 0], torch.full((num_future,), 15.0))
    torch.testing.assert_close(norms[0, :, 2], torch.full((num_future,), 15.0))
    torch.testing.assert_close(norms[1, :, 1], torch.full((num_future,), 25.0))
    assert torch.count_nonzero(norms.masked_select(~active[:, None].expand_as(norms))) == 0
    torch.testing.assert_close(force_fast, force)


@pytest.mark.parametrize("num_sites", [1, 2, 5])
def test_replaceable_net_wrench_limiter_preserves_sites_and_caps_residual(num_sites):
    limiter = ResidualWrenchLimiter(max_force_n=20.0, max_torque_nm=10.0)
    positions = torch.zeros((1, num_sites, 3))
    positions[..., 1] = 2.0
    forces = torch.zeros_like(positions)
    forces[..., 0] = 15.0
    torques = torch.zeros_like(positions)
    anchor = torch.zeros((1, 3))

    result = limiter(positions, anchor, forces, torques)
    fast_result = limiter._limit_unchecked(positions, anchor, forces, torques)

    torch.testing.assert_close(result.site_force_world, forces)
    assert torch.linalg.vector_norm(result.residual_force_world, dim=-1).item() <= 20.0 + 1.0e-5
    assert torch.linalg.vector_norm(result.residual_torque_world, dim=-1).item() <= 10.0 + 1.0e-5
    reconstructed_force = forces.sum(dim=1) + result.anchor_force_world
    reconstructed_torque = (
        torch.cross(positions - anchor[:, None], forces, dim=-1) + torques
    ).sum(dim=1) + result.anchor_torque_world
    torch.testing.assert_close(reconstructed_force, result.residual_force_world)
    torch.testing.assert_close(reconstructed_torque, result.residual_torque_world)
    for name in vars(result):
        torch.testing.assert_close(getattr(fast_result, name), getattr(result, name))


class _FakeComposer:
    def __init__(self):
        self.set_calls = []
        self.reset_calls = []
        self.force_rows = torch.full((3, 6, 3), 9.0)
        self.torque_rows = torch.full((3, 6, 3), -9.0)

    def set_forces_and_torques(self, **kwargs):
        self.set_calls.append(kwargs)
        env_ids = kwargs["env_ids"]
        if env_ids is None or isinstance(env_ids, slice):
            resolved_env_ids = range(self.force_rows.shape[0])
        else:
            resolved_env_ids = env_ids.tolist()
        body_ids = kwargs["body_ids"].tolist()
        for source_index, env_id in enumerate(resolved_env_ids):
            self.force_rows[env_id, body_ids] = kwargs["forces"][source_index]
            self.torque_rows[env_id, body_ids] = kwargs["torques"][source_index]

    def reset(self, env_ids):
        self.reset_calls.append(env_ids)
        index = slice(None) if env_ids is None else env_ids
        self.force_rows[index] = 0.0
        self.torque_rows[index] = 0.0


class _FakeAsset:
    def __init__(self):
        self.permanent_wrench_composer = _FakeComposer()


class _FakeCommand:
    def __init__(self):
        self.num_envs = 3
        self.robot = _FakeAsset()
        self.application_body_ids = torch.tensor([1, 4, 2])
        self.force = torch.full((3, 3, 3), 7.0)
        self.torque = torch.full((3, 3, 3), -2.0)
        self.operational_enabled = True
        self._wrench_dirty = False

    @property
    def wrench_dirty(self):
        return self._wrench_dirty

    def mark_wrench_applied(self):
        self._wrench_dirty = True

    def mark_wrench_cleared(self):
        self._wrench_dirty = False

    def _resample(self, env_ids):
        self.force.zero_()
        self.torque.zero_()

    def set_operational_enabled(self, enabled):
        transition_compliance_operational_state(self, enabled)

    def body_wrench_for_envs(self, env_ids):
        if env_ids is None:
            return self.force, self.torque, None
        return self.force[env_ids], self.torque[env_ids], env_ids

    def clear_wrench(self, env_ids):
        index = slice(None) if env_ids is None else env_ids
        self.force[index] = 0.0
        self.torque[index] = 0.0


class _FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "motion_compliance"
        return self.command


class _FakeEnv:
    def __init__(self, command):
        self.command_manager = _FakeCommandManager(command)


def test_event_only_writes_physx_buffers_and_reset_clears_composer_and_stale_state():
    command = _FakeCommand()
    env = _FakeEnv(command)
    env_ids = torch.tensor([0, 2])

    apply_compliance_wrench(env, env_ids)

    call = command.robot.permanent_wrench_composer.set_calls[-1]
    torch.testing.assert_close(call["forces"], torch.full((2, 3, 3), 7.0))
    torch.testing.assert_close(call["torques"], torch.full((2, 3, 3), -2.0))
    assert call["is_global"] is False

    reset_compliance_wrench(env, env_ids)
    torch.testing.assert_close(command.force[env_ids], torch.zeros((2, 3, 3)))
    torch.testing.assert_close(command.torque[env_ids], torch.zeros((2, 3, 3)))
    assert len(command.robot.permanent_wrench_composer.reset_calls) == 1
    assert command.wrench_dirty

    command.operational_enabled = False
    reset_compliance_wrench(env, None)
    assert len(command.robot.permanent_wrench_composer.reset_calls) == 2
    assert not command.wrench_dirty

    apply_compliance_wrench(env, None)
    assert len(command.robot.permanent_wrench_composer.set_calls) == 1


def test_disable_setter_immediately_clears_only_owned_rows_then_becomes_inert():
    command = _FakeCommand()
    env = _FakeEnv(command)
    env_ids = torch.tensor([0, 1, 2])

    apply_compliance_wrench(env, env_ids)
    composer = command.robot.permanent_wrench_composer
    assert len(composer.set_calls) == 1
    assert command.wrench_dirty

    command.set_operational_enabled(False)

    assert len(composer.set_calls) == 2
    assert composer.reset_calls == []
    torch.testing.assert_close(
        composer.force_rows[:, command.application_body_ids],
        torch.zeros((3, 3, 3)),
    )
    torch.testing.assert_close(
        composer.force_rows[:, [0, 3, 5]],
        torch.full((3, 3, 3), 9.0),
    )
    torch.testing.assert_close(
        composer.torque_rows[:, command.application_body_ids],
        torch.zeros((3, 3, 3)),
    )
    torch.testing.assert_close(
        composer.torque_rows[:, [0, 3, 5]],
        torch.full((3, 3, 3), -9.0),
    )
    assert not command.wrench_dirty

    apply_compliance_wrench(env, env_ids)
    assert len(composer.set_calls) == 2
    assert composer.reset_calls == []


def test_partial_disabled_clear_conservatively_retains_global_dirty_ownership():
    command = _FakeCommand()
    env = _FakeEnv(command)
    apply_compliance_wrench(env, None)
    composer = command.robot.permanent_wrench_composer

    command.operational_enabled = False
    apply_compliance_wrench(env, torch.tensor([0, 2]))

    assert command.wrench_dirty
    torch.testing.assert_close(
        composer.force_rows[[0, 2]][:, command.application_body_ids],
        torch.zeros((2, 3, 3)),
    )
    torch.testing.assert_close(
        composer.force_rows[1, command.application_body_ids],
        torch.full((3, 3), 7.0),
    )

    apply_compliance_wrench(env, None)
    assert not command.wrench_dirty
    torch.testing.assert_close(
        composer.force_rows[:, command.application_body_ids],
        torch.zeros((3, 3, 3)),
    )


def test_deprecated_wrench_setter_is_centralized_fallback_only():
    class LegacyAsset:
        def __init__(self):
            self.calls = []

        def set_external_force_and_torque(self, **kwargs):
            self.calls.append(kwargs)

    asset = LegacyAsset()
    forces = torch.zeros((1, 2, 3))
    torques = torch.zeros_like(forces)
    _set_body_wrench(asset, forces, torques, torch.tensor([1, 2]), torch.tensor([0]))
    assert len(asset.calls) == 1
    assert asset.calls[0]["is_global"] is False


def test_simulator_hot_path_uses_internal_no_sync_functions_and_cached_condition():
    command_source = (COMPLIANCE_PACKAGE / "adapters/sonic/command.py").read_text(
        encoding="utf-8"
    )
    for checked_call in (
        "virtual_force_from_reference_delta(",
        "rotate_vectors_wxyz(",
        "world_to_common_positions(",
        "common_to_world_vectors(",
    ):
        assert checked_call not in command_source
    for synchronizing_call in (".item(", "torch.isfinite(", "torch.allclose(", ".all()"):
        assert synchronizing_call not in command_source
    assert "_virtual_force_from_reference_delta_unchecked(" in command_source
    assert "_world_to_body_vectors_unchecked(" in command_source
    assert "._limit_unchecked(" in command_source
    assert "write_compliance_command_wrench(self)" in command_source

    command_tree = ast.parse(command_source)
    resample_node = next(
        node
        for node in ast.walk(command_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_resample"
    )
    resample_source = ast.get_source_segment(command_source, resample_node)
    assert "sample_resampling_time" in resample_source
    assert "_env_ids_tensor_prevalidated" in resample_source
    assert "_env_ids_tensor(" not in resample_source
    assert "torch.rand" not in resample_source
    assert ".uniform_" not in resample_source

    resample_command_node = next(
        node
        for node in ast.walk(command_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_resample_command"
    )
    resample_command_source = ast.get_source_segment(command_source, resample_command_node)
    assert "_resample_prevalidated" in resample_command_source
    assert "_disable_prevalidated" in resample_command_source
    assert "_env_ids_tensor" not in resample_command_source

    compute_node = next(
        node
        for node in ast.walk(command_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute"
    )
    compute_source = ast.get_source_segment(command_source, compute_node)
    assert "_resample_masked_prevalidated" in compute_source
    assert "self.time_left <= 0.0" in compute_source
    assert ".nonzero(" not in compute_source
    assert "super().compute" not in compute_source

    masked_resample_source = inspect.getsource(
        ComplianceCommandState._resample_masked_prevalidated
    )
    assert ".nonzero(" not in masked_resample_source
    assert "_env_ids_tensor" not in masked_resample_source

    switch_source = inspect.getsource(transition_compliance_operational_state)
    assert switch_source.index("command._resample(slice(None))") < switch_source.rindex(
        "write_compliance_command_wrench(command)"
    )

    state_condition_source = inspect.getsource(ComplianceCommandState.condition.fget)
    assert "return self._condition" in state_condition_source
    assert "encode_compliance_condition" not in state_condition_source

    for fast_function in (
        _clamp_vector_norm_unchecked,
        _reshape_site_data,
        _expanded_site_data_unchecked,
        _expanded_enabled_unchecked,
        _expanded_current_reference,
        _align_trailing_components,
        _rotate_vectors_wxyz_unchecked,
        _virtual_force_from_reference_delta_unchecked,
        _common_to_world_vectors_unchecked,
        _world_to_common_positions_unchecked,
        _world_to_body_vectors_unchecked,
        ResidualWrenchLimiter._limit_unchecked,
    ):
        source = inspect.getsource(fast_function)
        for synchronizing_call in (".item(", "torch.isfinite(", "torch.allclose(", ".all()"):
            assert synchronizing_call not in source


def test_hydra_composes_opt_in_command_and_event_without_changing_release_files():
    config_dir = str((ROOT / "gear_sonic" / "config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        baseline = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                "num_envs=1",
            ],
        )
        cfg = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release_motion_compliance",
                "num_envs=1",
            ],
        )

    assert cfg.num_envs == 1
    assert cfg.manager_env.commands.motion_compliance.num_future_frames == 10
    assert cfg.manager_env.commands.motion_compliance.enabled is False
    assert cfg.manager_env.commands.motion_compliance.reference_body_names == (
        cfg.manager_env.commands.motion.body_names
    )
    assert cfg.manager_env.commands._target_.endswith("ComplianceCommandsCfg")
    assert cfg.manager_env.events._target_.endswith("ComplianceEventsCfg")
    assert "motion_compliance_apply" not in cfg.manager_env.events
    assert cfg.manager_env.events.motion_compliance_reset.mode == "reset"

    def interval_contract(events):
        result = {}
        for name, term in events.items():
            if name == "_target_" or term is None or term.get("mode") != "interval":
                continue
            result[name] = OmegaConf.to_container(term.interval_range_s, resolve=True)
        return result

    assert interval_contract(cfg.manager_env.events) == interval_contract(
        baseline.manager_env.events
    )


def test_only_sonic_adapter_python_may_import_isaaclab_and_core_has_no_body_names():
    for path in COMPLIANCE_PACKAGE.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports_isaaclab = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_isaaclab |= any(alias.name.startswith("isaaclab") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports_isaaclab |= node.module.startswith("isaaclab")
        if imports_isaaclab:
            assert "adapters/sonic" in path.as_posix()

    core_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (COMPLIANCE_PACKAGE / "core").glob("*.py")
    ).lower()
    assert "wrist_yaw_link" not in core_text
    assert "torso_link" not in core_text
