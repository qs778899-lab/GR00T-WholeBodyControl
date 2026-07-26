"""Contract tests for the reusable compliance-control core."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from gear_sonic.compliance_control.core import (
    ComplianceSpec,
    ForceEventScheduleSpec,
    compute_compliance_metrics,
    encode_compliance_condition,
    event_envelope,
    hard_gate_residual,
    sample_site_mask,
    select_reference,
    stiffness_from_threshold,
    virtual_force_from_reference_delta,
)


CORE_DIR = Path(__file__).parents[1] / "compliance_control" / "core"


def test_hard_gated_residual_is_bitwise_off_per_row_and_differentiable_when_on():
    base = torch.tensor(
        [[-0.0, 1.0, -2.0], [3.0, 4.0, 5.0]],
        requires_grad=True,
    )
    residual = torch.tensor(
        [[float("nan"), float("inf"), -float("inf")], [0.5, -1.0, 2.0]],
        requires_grad=True,
    )
    enabled = torch.tensor([False, True])
    selected = hard_gate_residual(base, residual, enabled)
    assert torch.equal(selected[0], base[0])
    assert torch.signbit(selected[0, 0]) == torch.signbit(base[0, 0])
    torch.testing.assert_close(selected[1], base[1] + residual[1])

    selected[1].sum().backward()
    assert torch.equal(base.grad, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert torch.equal(
        residual.grad,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def test_hard_gated_residual_rejects_implicit_or_mismatched_contracts():
    base = torch.zeros(2, 3)
    with pytest.raises(TypeError, match="boolean"):
        hard_gate_residual(base, base, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="shapes must match"):
        hard_gate_residual(base, torch.zeros(2, 4), torch.tensor([False, True]))
    with pytest.raises(ValueError, match="leading dimensions"):
        hard_gate_residual(base, base, torch.zeros(3, dtype=torch.bool))


def test_core_has_no_simulator_robot_layout_or_concrete_frame_dependency():
    forbidden_import_roots = {"isaaclab"}
    forbidden_robot_tokens = (
        "left_wrist",
        "right_wrist",
        "mujoco_to_isaaclab",
        "isaaclab_to_mujoco",
        "torso",
    )

    for path in CORE_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", maxsplit=1)[0] for alias in node.names}
                assert roots.isdisjoint(forbidden_import_roots), path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", maxsplit=1)[0] not in forbidden_import_roots, path
            elif isinstance(node, ast.Constant) and type(node.value) is int:
                assert node.value not in {14, 29}, path
        lowered = source.lower()
        assert all(token not in lowered for token in forbidden_robot_tokens), path


def test_schema_and_api_state_frame_sign_and_upstream_formula_contract():
    spec = ComplianceSpec()
    assert spec.condition_size == 3
    assert spec.reference_frame_contract == "adapter_supplied_common_cartesian_frame"
    assert spec.force_sign_convention == "force_on_robot"
    assert spec.stiffness_range_n_per_m == pytest.approx((200.0, 400.0))
    assert spec.tracking_gain_n_per_m == pytest.approx(100.0)
    assert spec.tracking_force_cap_n == pytest.approx(5.0)
    doc = inspect.getdoc(virtual_force_from_reference_delta)
    assert doc is not None
    assert "adapter-supplied common Cartesian frame" in doc
    assert "force_on_robot" in doc
    assert "nominal + tracking" in doc


def test_schema_rejects_invalid_values():
    ForceEventScheduleSpec()
    with pytest.raises(ValueError):
        ComplianceSpec(force_threshold_range_n=(20.0, 10.0))
    for invalid in (0.0, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            ComplianceSpec(reference_displacement_m=invalid)
    with pytest.raises(ValueError):
        ComplianceSpec(tracking_gain_n_per_m=math.nan)
    with pytest.raises(ValueError):
        ComplianceSpec(tracking_force_cap_n=math.inf)
    with pytest.raises(ValueError):
        ForceEventScheduleSpec(duration_steps_range=(0, 1))


def test_condition_is_explicit_and_disabled_rows_are_exact_zero():
    enabled = torch.tensor([False, True, False])
    thresholds = torch.tensor([10.0, 15.0, 20.0])

    condition = encode_compliance_condition(enabled, thresholds)

    torch.testing.assert_close(condition[0], torch.zeros(3), rtol=0.0, atol=0.0)
    torch.testing.assert_close(condition[1], torch.tensor([1.0, 15.0, 300.0]))
    torch.testing.assert_close(condition[2], torch.zeros(3), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("invalid", [0.5, -1.0, 2.0, math.nan, math.inf, -math.inf])
def test_enabled_rejects_non_binary_or_non_finite_values(invalid):
    with pytest.raises(ValueError):
        encode_compliance_condition(invalid, 10.0)

    original = torch.zeros((1, 2, 3))
    compliant = original + 0.01
    active = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError):
        select_reference(original, compliant, active, enabled=invalid)
    with pytest.raises(ValueError):
        virtual_force_from_reference_delta(
            original,
            compliant,
            active,
            10.0,
            enabled=invalid,
        )


def test_enabled_rejects_non_numeric_types_and_schedule_uses_same_binary_contract():
    with pytest.raises(TypeError):
        encode_compliance_condition("on", 10.0)
    with pytest.raises(ValueError):
        sample_site_mask(torch.tensor([0.0, 0.5]), num_sites=2)


def test_displacement_rejects_non_finite_values_at_math_and_force_api():
    original = torch.zeros((1, 2, 3))
    compliant = original + 0.01
    active = torch.ones((1, 2), dtype=torch.bool)
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            stiffness_from_threshold(10.0, invalid)
        with pytest.raises(ValueError):
            virtual_force_from_reference_delta(
                original,
                compliant,
                active,
                10.0,
                reference_displacement_m=invalid,
            )


@pytest.mark.parametrize("num_sites", [2, 7, 17])
def test_reference_modifier_is_site_count_agnostic_and_hard_off_is_bitwise(num_sites):
    original = torch.arange(num_sites * 3, dtype=torch.float64).reshape(1, num_sites, 3)
    original[..., 0] *= -0.0
    compliant = original + 0.25
    active = torch.zeros((1, num_sites), dtype=torch.bool)
    active[:, -1] = True
    original_before = original.clone()
    compliant_before = compliant.clone()

    selected = select_reference(original, compliant, active)

    torch.testing.assert_close(selected[:, :-1], original[:, :-1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(selected[:, -1], compliant[:, -1])
    torch.testing.assert_close(original, original_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(compliant, compliant_before, rtol=0.0, atol=0.0)

    stale_active_mask = torch.ones((1, num_sites), dtype=torch.bool)
    disabled = select_reference(original, compliant, stale_active_mask, enabled=0)
    assert disabled.dtype == original.dtype
    assert disabled.device == original.device
    assert torch.equal(disabled.view(torch.int64), original.view(torch.int64))

    disabled_force = virtual_force_from_reference_delta(
        original,
        compliant,
        stale_active_mask,
        force_threshold_n=10.0,
        enabled=False,
    )
    torch.testing.assert_close(disabled_force, torch.zeros_like(original), rtol=0.0, atol=0.0)


def test_multi_future_reference_broadcasts_batch_site_mask_and_threshold():
    batch_size, num_future, num_sites = 2, 4, 17
    original = torch.zeros((batch_size, num_future, num_sites, 3), dtype=torch.float64)
    compliant = original.clone()
    compliant[..., 0] = 0.05
    active = torch.zeros((batch_size, num_sites), dtype=torch.bool)
    active[0, 0] = True
    active[0, -1] = True
    active[1, 3] = True
    thresholds = torch.linspace(10.0, 20.0, num_sites, dtype=torch.float64).repeat(
        batch_size,
        1,
    )

    selected = select_reference(
        original,
        compliant,
        active,
        enabled=torch.tensor([True, False]),
    )
    force = virtual_force_from_reference_delta(
        original,
        compliant,
        active,
        thresholds,
        enabled=torch.tensor([True, False]),
    )

    assert selected.shape == original.shape
    assert selected.dtype == original.dtype and selected.device == original.device
    assert force.dtype == original.dtype and force.device == original.device
    torch.testing.assert_close(selected[0, :, 0], compliant[0, :, 0])
    torch.testing.assert_close(selected[0, :, -1], compliant[0, :, -1])
    torch.testing.assert_close(selected[0, :, 1:-1], original[0, :, 1:-1])
    torch.testing.assert_close(selected[1], original[1], rtol=0.0, atol=0.0)
    norms = torch.linalg.vector_norm(force, dim=-1)
    torch.testing.assert_close(norms[0, :, 0], thresholds[0, 0].expand(num_future))
    torch.testing.assert_close(norms[0, :, -1], thresholds[0, -1].expand(num_future))
    torch.testing.assert_close(norms[1], torch.zeros_like(norms[1]), rtol=0.0, atol=0.0)


def test_select_reference_validates_shape_dtype_device_and_mask_type():
    original = torch.zeros((1, 2, 3), dtype=torch.float32)
    active = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError):
        select_reference(original, torch.zeros((1, 3, 3)), active)
    with pytest.raises(TypeError):
        select_reference(original, original.to(torch.float64), active)
    with pytest.raises(ValueError):
        select_reference(original, torch.zeros_like(original, device="meta"), active)
    with pytest.raises(TypeError):
        select_reference(original.to(torch.int64), original.to(torch.int64), active)
    with pytest.raises(TypeError):
        select_reference(original, original, active.float())
    with pytest.raises(ValueError):
        select_reference(original, original, torch.ones((1, 3), dtype=torch.bool))


def test_force_matches_upstream_nominal_plus_capped_tracking_formula_and_sign():
    original = torch.zeros((1, 1, 3))
    compliant = torch.tensor([[[0.05, 0.0, 0.0]]])
    current = torch.zeros_like(original)
    active = torch.ones((1, 1), dtype=torch.bool)

    nominal = virtual_force_from_reference_delta(original, compliant, active, 10.0)
    full = virtual_force_from_reference_delta(
        original,
        compliant,
        active,
        10.0,
        current_reference=current,
    )
    explicit_nominal = virtual_force_from_reference_delta(
        original,
        compliant,
        active,
        10.0,
        current_reference=current,
        include_tracking_term=False,
    )

    torch.testing.assert_close(nominal, torch.tensor([[[10.0, 0.0, 0.0]]]))
    torch.testing.assert_close(full, torch.tensor([[[15.0, 0.0, 0.0]]]))
    torch.testing.assert_close(explicit_nominal, nominal)
    assert full[..., 0].item() > 0.0


def test_tracking_term_uses_compliant_minus_current_and_is_independently_capped():
    original = torch.zeros((1, 1, 3))
    compliant = torch.tensor([[[0.05, 0.0, 0.0]]])
    current = torch.tensor([[[0.20, 0.0, 0.0]]])
    active = torch.ones((1, 1), dtype=torch.bool)

    force = virtual_force_from_reference_delta(
        original,
        compliant,
        active,
        10.0,
        current_reference=current,
    )

    torch.testing.assert_close(force, torch.tensor([[[5.0, 0.0, 0.0]]]))


def test_force_rejects_non_finite_and_mismatched_typed_inputs():
    original = torch.zeros((1, 2, 3))
    compliant = original + 0.01
    active = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError):
        virtual_force_from_reference_delta(original, compliant, active, math.nan)
    with pytest.raises(ValueError):
        virtual_force_from_reference_delta(
            original,
            compliant,
            active,
            10.0,
            current_reference=original,
            tracking_gain_n_per_m=math.nan,
        )
    with pytest.raises(ValueError):
        virtual_force_from_reference_delta(
            original,
            compliant,
            active,
            10.0,
            current_reference=original,
            tracking_force_cap_n=math.inf,
        )
    with pytest.raises(TypeError):
        virtual_force_from_reference_delta(
            original,
            compliant,
            active,
            torch.ones((1, 2), dtype=torch.float64),
        )
    invalid_current = original.clone()
    invalid_current[0, 0, 0] = math.nan
    with pytest.raises(ValueError):
        virtual_force_from_reference_delta(
            original,
            compliant,
            active,
            10.0,
            current_reference=invalid_current,
        )


def test_reference_and_full_force_paths_are_differentiable_with_future_axis():
    batch_size, num_future, num_sites = 2, 3, 17
    original = torch.zeros(
        (batch_size, num_future, num_sites, 3),
        requires_grad=True,
    )
    compliant = torch.full_like(original, 0.01, requires_grad=True)
    current = torch.zeros((batch_size, num_sites, 3), requires_grad=True)
    thresholds = torch.full((batch_size, num_sites), 10.0, requires_grad=True)
    active = torch.zeros((batch_size, num_sites), dtype=torch.bool)
    active[0, 0] = True
    active[1, -1] = True

    selected = select_reference(original, compliant, active)
    force = virtual_force_from_reference_delta(
        original,
        compliant,
        active,
        thresholds,
        current_reference=current,
    )
    loss = selected.square().sum() + force.square().sum()
    loss.backward()

    for gradient in (original.grad, compliant.grad, current.grad, thresholds.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
    assert compliant.grad[0, :, 0].abs().sum() > 0.0
    assert compliant.grad[1, :, -1].abs().sum() > 0.0
    inactive = ~active[:, None, :, None].expand_as(compliant)
    torch.testing.assert_close(
        compliant.grad[inactive],
        torch.zeros_like(compliant.grad[inactive]),
        rtol=0.0,
        atol=0.0,
    )


def test_event_schedule_and_site_sampling_are_generic():
    steps = torch.tensor([-1, 0, 1, 2, 3, 4, 5])
    envelope = event_envelope(steps, start_step=0, duration_steps=4, ramp_fraction=0.25)
    torch.testing.assert_close(envelope, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))

    generator = torch.Generator().manual_seed(4)
    enabled = torch.tensor([True, True, False])
    site_mask = sample_site_mask(
        enabled,
        num_sites=5,
        site_activation_probability=0.0,
        generator=generator,
    )
    assert site_mask.shape == (3, 5)
    assert site_mask[:2].sum(dim=-1).tolist() == [1, 1]
    assert not site_mask[2].any()


def test_site_sampling_uses_fixed_shape_without_data_dependent_cuda_indices():
    source = inspect.getsource(sample_site_mask)
    assert ".nonzero(" not in source
    assert ".item(" not in source

    enabled = torch.tensor([True, True, False])
    missing_generator = torch.Generator().manual_seed(41)
    full_generator = torch.Generator().manual_seed(41)
    missing_mask = sample_site_mask(
        enabled,
        num_sites=5,
        site_activation_probability=0.0,
        generator=missing_generator,
    )
    full_mask = sample_site_mask(
        enabled,
        num_sites=5,
        site_activation_probability=1.0,
        generator=full_generator,
    )

    assert missing_mask[:2].sum(dim=-1).tolist() == [1, 1]
    assert full_mask[:2].all()
    assert torch.equal(missing_generator.get_state(), full_generator.get_state())


def test_metrics_hard_gate_and_detect_inactive_candidate_reference_pollution():
    original = torch.zeros((1, 2, 4, 3))
    compliant = original.clone()
    compliant[:, :, 1, 0] = 0.05
    compliant[:, :, 2, 1] = 0.02
    active = torch.tensor([[False, True, False, False]])
    actual = select_reference(original, compliant, active)
    force = virtual_force_from_reference_delta(original, compliant, active, 10.0)

    metrics = compute_compliance_metrics(
        original,
        compliant,
        actual,
        force,
        active,
        enabled=True,
    )
    disabled_metrics = compute_compliance_metrics(
        original,
        compliant,
        original,
        force,
        torch.ones_like(active),
        enabled=False,
    )

    assert metrics.selected_tracking_error.shape == (1, 2)
    torch.testing.assert_close(metrics.selected_tracking_error, torch.zeros((1, 2)))
    torch.testing.assert_close(metrics.active_reference_yield, torch.full((1, 2), 0.05))
    torch.testing.assert_close(
        metrics.inactive_reference_drift,
        torch.full((1, 2), 0.02 / 3.0),
    )
    torch.testing.assert_close(metrics.peak_virtual_force, torch.full((1, 2), 10.0))
    torch.testing.assert_close(metrics.active_site_fraction, torch.full((1, 2), 0.25))
    torch.testing.assert_close(disabled_metrics.active_reference_yield, torch.zeros((1, 2)))
    torch.testing.assert_close(disabled_metrics.peak_virtual_force, torch.zeros((1, 2)))
    torch.testing.assert_close(disabled_metrics.active_site_fraction, torch.zeros((1, 2)))
    torch.testing.assert_close(
        disabled_metrics.inactive_reference_drift,
        torch.full((1, 2), (0.05 + 0.02) / 4.0),
    )
