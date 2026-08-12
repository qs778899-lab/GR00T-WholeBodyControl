"""CPU tests for the thin SONIC-to-portable Phase-6 trace collector."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.compliance_control.adapters.sonic.evaluation import (
    NaturalMotionTimeoutObserver,
    PolicyActionByteEvidence,
    SONIC_ACTION_RESIDUAL_PREFIX,
    SONIC_EVALUATION_MANAGER_PROVENANCE,
    SONIC_EVALUATION_TERMINATION_NAMES,
    SONIC_RELEASE_CHECKPOINT_SHA256,
    SONIC_RELEASE_CHECKPOINT_STEP,
    SONIC_RELEASE_TRACKING_BODY_NAMES,
    SONIC_TRAINED_CHECKPOINT_SHA256,
    SONIC_TRAINED_CHECKPOINT_STEP,
    SonicEvaluationProtocol,
    SonicEvaluationSnapshot,
    SonicEvaluationTraceCollector,
    apply_sonic_evaluation_protocol,
    assert_g1_only_encoder_selection,
    clear_and_assert_owned_composer_wrench,
    exercise_sonic_evaluation_reset_event,
    policy_action_row_sha256,
    snapshot_from_sonic_commands,
    validate_policy_action_byte_parity,
    validate_sonic_evaluation_checkpoint_role,
    validate_sonic_evaluation_config_provenance,
    validate_sonic_evaluation_manager_provenance,
    validate_sonic_evaluation_event_names,
)
from gear_sonic.compliance_control.adapters.sonic.event import reset_compliance_wrench
from gear_sonic.compliance_control.adapters.sonic.state import (
    ComplianceCommandState,
    ComplianceSamplingSpec,
)
from gear_sonic.compliance_control.adapters.sonic.frames import world_to_common_positions
from gear_sonic.compliance_control.evaluation import (
    EvaluationTrace,
    RegressionCriteria,
    TrialMode,
    TrialSpec,
    alignment_digest,
    evaluate_trial_suite,
)
from tasks.motion_compliance_finetune.artifacts.phase6_validate_sonic_collection_reports import (
    SONIC_AUDITED_MOTION_SHA256,
    SONIC_PHASE6_ENVIRONMENT_INVARIANTS,
    SONIC_PHASE6_PORTABLE_CRITERIA,
    validate_sonic_collection_suite,
)


SITE_IDS = ("site_alpha", "site_beta")
POINT_IDS = tuple(f"point_{index}" for index in range(4))


def _snapshot(
    *,
    motion_id: str = "dataset_motion:7:start_frame:12",
    enabled: bool = False,
    active_site_indices: tuple[int, ...] = (),
    force_n: float = 0.0,
) -> SonicEvaluationSnapshot:
    batch_size = 1
    sites = len(SITE_IDS)
    points = len(POINT_IDS)
    original = np.zeros((batch_size, sites, 3), dtype=np.float32)
    selected = original.copy()
    active = np.zeros((batch_size, sites), dtype=np.bool_)
    force = np.zeros_like(original)
    for site_index in active_site_indices:
        active[:, site_index] = True
        selected[:, site_index, 0] = 0.05
        force[:, site_index, 0] = force_n
    orientations = np.zeros((batch_size, sites, 4), dtype=np.float32)
    orientations[..., 3] = 1.0
    points_array = np.zeros((batch_size, points, 3), dtype=np.float32)
    return SonicEvaluationSnapshot(
        motion_ids=(motion_id,),
        site_ids=SITE_IDS,
        point_ids=POINT_IDS,
        original_site_positions_m=original,
        selected_site_positions_m=selected,
        measured_site_positions_m=original.copy(),
        original_site_orientations_xyzw=orientations,
        measured_site_orientations_xyzw=orientations.copy(),
        reference_points_global_m=points_array,
        measured_points_global_m=points_array.copy(),
        reference_points_local_m=points_array.copy(),
        measured_points_local_m=points_array.copy(),
        force_on_robot_n=force,
        owned_wrench_force_peak_n=np.asarray([force_n], dtype=np.float32),
        owned_wrench_torque_peak_nm=np.asarray([0.0], dtype=np.float32),
        owned_force_buffer_max_abs_difference_n=np.asarray([0.0], dtype=np.float32),
        owned_torque_buffer_max_abs_difference_nm=np.asarray([0.0], dtype=np.float32),
        compliance_enabled=np.asarray([enabled], dtype=np.bool_),
        active_site_mask=active,
    )


def _collector(trial_name: str = "trial") -> SonicEvaluationTraceCollector:
    return SonicEvaluationTraceCollector(
        trial_name=trial_name,
        seed_id=23,
        step_dt_s=0.02,
        site_ids=SITE_IDS,
        point_ids=POINT_IDS,
        max_rows=32,
    )


def test_lifecycle_collector_uses_natural_timeout_and_sticky_failure_on_final_row():
    collector = _collector("single_alpha")
    collector.record_post_reset(_snapshot(enabled=True, active_site_indices=(0,)))
    collector.record_post_step(
        _snapshot(enabled=True, active_site_indices=(0,), force_n=3.0),
        terminal_mask=np.asarray([False]),
        success_mask=np.asarray([False]),
        fall_mask=np.asarray([False]),
    )
    collector.record_post_step(
        _snapshot(enabled=True, active_site_indices=(0,), force_n=4.0),
        terminal_mask=np.asarray([False]),
        success_mask=np.asarray([False]),
        fall_mask=np.asarray([False]),
    )
    trace = collector.finalize(natural_timeout_env_ids=[0], failed_env_ids=[0])

    assert trace.motion_ids == (_snapshot().motion_ids[0],) * 3
    assert trace.sequence_ids == ("env_0000:episode_0000",) * 3
    np.testing.assert_array_equal(trace.frame_indices, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(trace.timestamps_s, np.asarray([0.0, 0.02, 0.04]))
    np.testing.assert_array_equal(trace.reset_mask, np.asarray([True, False, False]))
    np.testing.assert_array_equal(trace.terminal_mask, np.asarray([False, False, True]))
    np.testing.assert_array_equal(trace.fall_mask, np.asarray([False, False, True]))
    assert np.linalg.norm(trace.force_on_robot_n[0], axis=-1).max() == 0.0
    assert np.linalg.norm(trace.force_on_robot_n[-1], axis=-1).max() == 4.0


def test_natural_timeout_finalize_marks_success_and_duplicate_initial_reset_is_replaced():
    collector = _collector("off")
    collector.record_post_reset(_snapshot())
    collector.record_post_reset(_snapshot())
    collector.record_post_step(
        _snapshot(),
        terminal_mask=np.asarray([False]),
        success_mask=np.asarray([False]),
        fall_mask=np.asarray([False]),
    )
    trace = collector.finalize(natural_timeout_env_ids=[0])

    assert trace.sequence_ids == ("env_0000:episode_0000",) * 2
    np.testing.assert_array_equal(trace.frame_indices, np.asarray([0, 1]))
    np.testing.assert_array_equal(trace.terminal_mask, np.asarray([False, True]))
    np.testing.assert_array_equal(trace.success_mask, np.asarray([False, True]))
    assert not trace.fall_mask.any()


def test_collector_rejects_publish_before_natural_motion_timeout():
    collector = _collector("incomplete")
    collector.record_post_reset(_snapshot())
    collector.record_post_step(
        _snapshot(),
        terminal_mask=np.asarray([False]),
        success_mask=np.asarray([False]),
        fall_mask=np.asarray([False]),
    )
    with pytest.raises(RuntimeError, match="natural motion timeout"):
        collector.finalize(natural_timeout_env_ids=[])


def test_collector_rejects_motion_layout_event_and_bound_violations():
    collector = _collector()
    collector.record_post_reset(_snapshot())
    with pytest.raises(ValueError, match="motion identity"):
        collector.record_post_step(
            _snapshot(motion_id="different"),
            terminal_mask=np.asarray([False]),
            success_mask=np.asarray([False]),
            fall_mask=np.asarray([False]),
        )

    collector = _collector()
    collector.record_post_reset(_snapshot())
    with pytest.raises(ValueError, match="require terminal"):
        collector.record_post_step(
            _snapshot(),
            terminal_mask=np.asarray([False]),
            success_mask=np.asarray([True]),
            fall_mask=np.asarray([False]),
        )

    changed_layout = replace(_snapshot(), site_ids=("other", "site_beta"))
    with pytest.raises(ValueError, match="site layout"):
        collector.record_post_reset(changed_layout)

    bounded = SonicEvaluationTraceCollector(
        trial_name="bounded",
        seed_id=0,
        step_dt_s=0.02,
        site_ids=SITE_IDS,
        point_ids=POINT_IDS,
        max_rows=1,
    )
    bounded.record_post_reset(_snapshot())
    with pytest.raises(RuntimeError, match="max_rows"):
        bounded.record_post_step(
            _snapshot(),
            terminal_mask=np.asarray([False]),
            success_mask=np.asarray([False]),
            fall_mask=np.asarray([False]),
        )


class _FakeMotionLib:
    @staticmethod
    def get_motion_ids_in_dataset(motion_ids):
        return motion_ids + 7


def test_sonic_snapshot_maps_names_frames_quaternions_and_selected_targets():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    reference_points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    measured_points = reference_points + torch.tensor([[[0.0, 0.1, 0.0]]])
    reference_quaternions = identity.repeat(1, 3, 1)
    reference_quaternions[0, 2] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    tracking = SimpleNamespace(
        motion_lib=_FakeMotionLib(),
        motion_ids=torch.tensor([2]),
        motion_start_time_steps=torch.tensor([11]),
        body_pos_w=reference_points,
        robot_body_pos_w=measured_points,
        body_quat_w=reference_quaternions,
        anchor_pos_w=torch.zeros(1, 3),
        anchor_quat_w=identity.repeat(1, 1),
        robot_anchor_pos_w=torch.zeros(1, 3),
        robot_anchor_quat_w=identity.repeat(1, 1),
        cfg=SimpleNamespace(body_names=["point_a", "point_b", "point_c"]),
    )
    original = torch.zeros(1, 2, 2, 3)
    compliant = original.clone()
    compliant[:, :, 0, 0] = 0.05
    current = torch.zeros(1, 2, 3)
    site_quaternions = identity.repeat(1, 2, 1)
    site_quaternions[0, 1] = torch.tensor([0.5, -0.5, 0.5, -0.5])
    site_state = SimpleNamespace(
        original_reference_common=original,
        compliant_reference_common=compliant,
        current_reference_common=current,
        site_body_position_world=torch.zeros(1, 2, 3),
        site_offset_world=torch.zeros(1, 2, 3),
        site_quaternion_world=site_quaternions,
        anchor_position_world=torch.zeros(1, 3),
        anchor_quaternion_world=identity.repeat(1, 1),
    )
    state = SimpleNamespace(
        active_site_mask=torch.tensor([[True, False]]),
        enabled=torch.tensor([True]),
        site_force_world=torch.tensor([[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    )
    composer_force = torch.zeros(1, 5, 3)
    composer_force[0, 1, 0] = 3.0
    composer = SimpleNamespace(
        composed_force_as_torch=composer_force,
        composed_torque_as_torch=torch.zeros_like(composer_force),
    )
    command = SimpleNamespace(
        num_envs=1,
        cfg=SimpleNamespace(site_body_names=["site_alpha", "site_beta"]),
        body_map=SimpleNamespace(
            reference_site_indices=(2, 0),
            articulation_site_indices=(1, 3),
            reference_anchor_index=0,
            num_sites=2,
        ),
        application_body_ids=torch.tensor([1, 3, 0]),
        _application_force_body=torch.tensor(
            [[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        _application_torque_body=torch.zeros(1, 3, 3),
        robot=SimpleNamespace(permanent_wrench_composer=composer),
        state=state,
        _tracking_term=lambda: tracking,
        _site_tracking_state=lambda: site_state,
        _reference_world_state=lambda: torch.zeros(1, 2, 2, 3),
    )

    snapshot = snapshot_from_sonic_commands(command)

    assert snapshot.motion_ids == ("dataset_motion:9:start_frame:11",)
    assert snapshot.point_ids == ("point_a", "point_b", "point_c")
    np.testing.assert_array_equal(
        snapshot.selected_site_positions_m[0, 0],
        np.asarray([0.05, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(snapshot.selected_site_positions_m[0, 1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        snapshot.original_site_orientations_xyzw[0, 0],
        [0.5, 0.5, 0.5, 0.5],
    )
    np.testing.assert_array_equal(
        snapshot.measured_site_orientations_xyzw[0, 1],
        [-0.5, 0.5, -0.5, 0.5],
    )
    np.testing.assert_array_equal(snapshot.force_on_robot_n[0, 0], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(
        snapshot.measured_points_local_m - snapshot.reference_points_local_m,
        measured_points.numpy() - reference_points.numpy(),
    )

    # The trace must read actual composer rows, not trust a cleared command
    # cache.  Anchor force/torque is included in reset-staleness evidence.
    command.state.site_force_world.zero_()
    composer_force[0, 0, 1] = 2.0
    composer.composed_torque_as_torch[0, 0, 2] = 5.0
    stale = snapshot_from_sonic_commands(command)
    np.testing.assert_array_equal(stale.force_on_robot_n[0, 0], [3.0, 0.0, 0.0])
    assert stale.owned_wrench_force_peak_n.tolist() == [3.0]
    assert stale.owned_wrench_torque_peak_nm.tolist() == [5.0]
    assert stale.owned_force_buffer_max_abs_difference_n.tolist() == [2.0]
    assert stale.owned_torque_buffer_max_abs_difference_nm.tolist() == [5.0]


def _shared_reference_frame_command(
    *,
    robot_torso_position_world: torch.Tensor,
    robot_torso_quaternion_world: torch.Tensor,
    measured_site_shift_world: torch.Tensor,
    active_first_site: bool,
):
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    half = float(np.sqrt(0.5))
    reference_torso_quaternion = torch.tensor([half, 0.0, 0.0, half])
    reference_points = torch.tensor(
        [[[1.0, 2.0, 0.0], [1.0, 3.0, 0.0], [0.0, 2.0, 0.0]]]
    )
    reference_quaternions = identity.repeat(1, 3, 1)
    reference_quaternions[:, 0] = reference_torso_quaternion
    reference_sites_world = reference_points[:, 1:]
    reference_future = reference_sites_world[:, None].repeat(1, 2, 1, 1)
    original_robot_common = world_to_common_positions(
        reference_future,
        robot_torso_position_world[:, None, None, :],
        robot_torso_quaternion_world[:, None, None, :],
    )
    compliant_robot_common = original_robot_common.clone()
    compliant_robot_common[..., 0] += 0.05
    measured_site_world = reference_sites_world + measured_site_shift_world
    site_quaternions = identity.repeat(1, 2, 1)
    site_state = SimpleNamespace(
        original_reference_common=original_robot_common,
        compliant_reference_common=compliant_robot_common,
        current_reference_common=original_robot_common[:, 0],
        site_body_position_world=measured_site_world,
        site_offset_world=torch.zeros(1, 2, 3),
        site_quaternion_world=site_quaternions,
        anchor_position_world=robot_torso_position_world,
        anchor_quaternion_world=robot_torso_quaternion_world,
    )
    force_body = torch.zeros(1, 3, 3)
    if active_first_site:
        # Reference torso is +90 deg about Z, so +world-Y is +reference-X.
        force_body[0, 0, 1] = 4.0
    composer_force = torch.zeros(1, 4, 3)
    composer_force[:, torch.tensor([1, 2, 0])] = force_body
    composer = SimpleNamespace(
        composed_force_as_torch=composer_force,
        composed_torque_as_torch=torch.zeros_like(composer_force),
    )
    active = torch.tensor([[active_first_site, False]])
    tracking = SimpleNamespace(
        motion_lib=_FakeMotionLib(),
        motion_ids=torch.tensor([2]),
        motion_start_time_steps=torch.tensor([0]),
        body_pos_w=reference_points,
        robot_body_pos_w=reference_points.clone(),
        body_quat_w=reference_quaternions,
        anchor_pos_w=reference_points[:, 0],
        anchor_quat_w=reference_torso_quaternion.repeat(1, 1),
        robot_anchor_pos_w=reference_points[:, 0],
        cfg=SimpleNamespace(body_names=["torso", "site_alpha", "site_beta"]),
    )
    return SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(site_body_names=["site_alpha", "site_beta"]),
        body_map=SimpleNamespace(
            reference_site_indices=(1, 2),
            articulation_site_indices=(1, 2),
            reference_anchor_index=0,
            num_sites=2,
        ),
        application_body_ids=torch.tensor([1, 2, 0]),
        _application_force_body=force_body,
        _application_torque_body=torch.zeros_like(force_body),
        robot=SimpleNamespace(permanent_wrench_composer=composer),
        state=SimpleNamespace(
            active_site_mask=active,
            enabled=torch.tensor([active_first_site]),
            site_force_world=torch.zeros(1, 2, 3),
        ),
        _tracking_term=lambda: tracking,
        _site_tracking_state=lambda: site_state,
        _reference_world_state=lambda: reference_future,
    )


def test_shared_reference_torso_frame_is_not_polluted_by_trial_robot_anchor_pose():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    off_command = _shared_reference_frame_command(
        robot_torso_position_world=torch.zeros(1, 3),
        robot_torso_quaternion_world=identity,
        measured_site_shift_world=torch.zeros(1, 2, 3),
        active_first_site=False,
    )
    candidate_shift = torch.zeros(1, 2, 3)
    candidate_shift[0, 0, 1] = 0.01
    candidate_command = _shared_reference_frame_command(
        robot_torso_position_world=torch.tensor([[3.0, -1.0, 0.0]]),
        robot_torso_quaternion_world=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        measured_site_shift_world=candidate_shift,
        active_first_site=True,
    )

    off = snapshot_from_sonic_commands(off_command)
    candidate = snapshot_from_sonic_commands(candidate_command)

    assert off.original_site_positions_m.tobytes() == (
        candidate.original_site_positions_m.tobytes()
    )
    np.testing.assert_allclose(
        candidate.measured_site_positions_m - off.measured_site_positions_m,
        [[[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        candidate.force_on_robot_n[0, 0],
        [4.0, 0.0, 0.0],
        atol=1.0e-6,
    )
    measured_yield = (
        candidate.measured_site_positions_m[0, 0]
        - off.measured_site_positions_m[0, 0]
    )
    force_direction = candidate.force_on_robot_n[0, 0] / np.linalg.norm(
        candidate.force_on_robot_n[0, 0]
    )
    assert float(np.dot(measured_yield, force_direction)) == pytest.approx(
        0.01,
        abs=1.0e-6,
    )
    assert candidate.selected_site_positions_m[:, 1].tobytes() == (
        candidate.original_site_positions_m[:, 1].tobytes()
    )


def test_local_tracking_points_remove_root_translation_in_one_reference_basis():
    command = _shared_reference_frame_command(
        robot_torso_position_world=torch.zeros(1, 3),
        robot_torso_quaternion_world=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        measured_site_shift_world=torch.zeros(1, 2, 3),
        active_first_site=False,
    )
    tracking = command._tracking_term()
    root_translation = torch.tensor([[[-0.3, 0.4, 0.2]]])
    tracking.robot_body_pos_w = tracking.body_pos_w + root_translation
    tracking.robot_anchor_pos_w = tracking.anchor_pos_w + root_translation[:, 0]

    snapshot = snapshot_from_sonic_commands(command)

    assert np.linalg.norm(
        snapshot.measured_points_global_m - snapshot.reference_points_global_m
    ) > 0.0
    np.testing.assert_allclose(
        snapshot.measured_points_local_m,
        snapshot.reference_points_local_m,
        atol=1.0e-6,
    )


class _FakeProtocolCommand:
    def __init__(self, num_sites: int):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            site_body_names=[f"site_{index}" for index in range(num_sites)],
            reference_displacement_m=0.05,
        )
        self.state = ComplianceCommandState(
            self.num_envs,
            num_sites,
            3,
            ComplianceSamplingSpec(),
        )
        self.time_left = torch.zeros(self.num_envs)
        self.operational_enabled = False
        self.application_clear_count = 0

    def set_operational_enabled(self, enabled: bool):
        self.operational_enabled = enabled

    def _clear_application_buffers_prevalidated(self, ids):
        self.application_clear_count += int(ids.numel())


@pytest.mark.parametrize("num_sites", [1, 2, 5])
def test_protocol_application_is_layout_driven_and_reset_force_stays_zero(num_sites: int):
    command = _FakeProtocolCommand(num_sites)
    active_site = f"site_{num_sites - 1}"
    protocol = SonicEvaluationProtocol(
        enabled=True,
        active_site_ids=(active_site,),
        force_threshold_n=12.0,
        reference_offset_common_m=(0.0, 0.04, 0.0),
    )
    apply_sonic_evaluation_protocol(command, protocol, env_ids=[1])

    assert command.operational_enabled is True
    assert command.state.enabled.tolist() == [False, True]
    assert command.state.active_site_mask[1].sum().item() == 1
    assert command.state.active_site_mask[1, num_sites - 1]
    assert command.state.condition[1].tolist() == pytest.approx([1.0, 12.0, 240.0])
    assert command.state.reference_offset_common[1, num_sites - 1].tolist() == pytest.approx(
        [0.0, 0.04, 0.0]
    )
    assert torch.count_nonzero(command.state.site_force_world) == 0
    assert command.application_clear_count == 1


def test_protocol_rejects_unknown_site_and_disabled_active_combination():
    with pytest.raises(ValueError, match="disabled protocol"):
        SonicEvaluationProtocol(enabled=False, active_site_ids=("site",))
    with pytest.raises(ValueError, match="host is disabled"):
        SonicEvaluationProtocol(enabled=True, operational_enabled=False)
    command = _FakeProtocolCommand(2)
    with pytest.raises(ValueError, match="unknown SONIC sites"):
        apply_sonic_evaluation_protocol(
            command,
            SonicEvaluationProtocol(enabled=True, active_site_ids=("unknown",)),
        )

    overlay_off = SonicEvaluationProtocol(enabled=False, operational_enabled=True)
    apply_sonic_evaluation_protocol(command, overlay_off)
    assert command.operational_enabled is True
    assert not command.state.enabled.any()
    assert not command.state.active_site_mask.any()


class SceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


SceneEntityCfg.__module__ = "isaaclab.managers.scene_entity_cfg"


def _fake_exceeded_anchor_height(
    env,
    command_name,
    threshold,
    threshold_adaptive=False,
    down_threshold=0.5,
    root_height_threshold=1.0,
):
    raise AssertionError("provenance-only fake must not execute")


def _fake_exceeded_anchor_ori(env, asset_cfg, command_name, threshold):
    raise AssertionError("provenance-only fake must not execute")


def _fake_exceeded_body_height(
    env,
    command_name,
    threshold,
    threshold_adaptive=False,
    down_threshold=0.5,
    body_names=None,
    root_height_threshold=0.5,
):
    raise AssertionError("provenance-only fake must not execute")


def _fake_tracking_time_out(env, command_name):
    raise AssertionError("provenance-only fake must not execute")


for _function, _target_name in (
    (_fake_exceeded_anchor_height, "exceeded_anchor_height"),
    (_fake_exceeded_anchor_ori, "exceeded_anchor_ori"),
    (_fake_exceeded_body_height, "exceeded_body_height"),
    (_fake_tracking_time_out, "tracking_time_out"),
):
    _function.__module__ = "gear_sonic.envs.manager_env.mdp.terminations"
    _function.__qualname__ = _target_name


def _phase6_termination_cfgs():
    return [
        SimpleNamespace(
            func=_fake_exceeded_anchor_height,
            time_out=False,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "threshold_adaptive": False,
                "down_threshold": 0.25,
            },
        ),
        SimpleNamespace(
            func=_fake_exceeded_anchor_ori,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 1.0,
            },
        ),
        SimpleNamespace(
            func=_fake_exceeded_body_height,
            time_out=False,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "threshold_adaptive": False,
                "down_threshold": 0.25,
            },
        ),
        SimpleNamespace(
            func=_fake_tracking_time_out,
            time_out=True,
            params={"command_name": "motion"},
        ),
    ]


class _FakeEventManager:
    def __init__(self, env=None):
        self.active_terms = {"reset": ["motion_compliance_reset"]}
        self.cfg = SimpleNamespace(
            func=reset_compliance_wrench,
            mode="reset",
            min_step_count_between_reset=0,
            params={"command_name": "motion_compliance"},
        )
        self.env = env
        self.apply_calls = []

    def get_term_cfg(self, name):
        assert name == "motion_compliance_reset"
        return self.cfg

    def apply(self, *, mode, env_ids, global_env_step_count):
        self.apply_calls.append((mode, env_ids.clone(), global_env_step_count))
        self.cfg.func(self.env, env_ids, **self.cfg.params)


class _FakeTerminationManager:
    def __init__(self):
        self.active_terms = ["anchor_pos", "anchor_ori_full", "ee_body_pos", "time_out"]
        self._term_cfgs = _phase6_termination_cfgs()
        self._terminated_buf = torch.zeros(1, dtype=torch.bool)
        self._truncated_buf = torch.zeros(1, dtype=torch.bool)
        self._term_dones = torch.zeros(1, 4, dtype=torch.bool)
        self.calls = 0

    @property
    def terminated(self):
        return self._terminated_buf

    @property
    def time_outs(self):
        return self._truncated_buf

    def compute(self):
        self.calls += 1
        self._terminated_buf.zero_()
        self._truncated_buf.zero_()
        self._term_dones.zero_()
        self._terminated_buf[:] = self.calls == 1
        self._truncated_buf[:] = self.calls == 3
        self._term_dones[:, 0] = self._terminated_buf
        self._term_dones[:, 3] = self._truncated_buf
        return self._terminated_buf | self._truncated_buf


def test_natural_timeout_observer_preserves_real_fall_without_auto_reset():
    manager = _FakeTerminationManager()
    observer = NaturalMotionTimeoutObserver(manager)
    observer.install()

    for _ in range(3):
        assert not manager.compute().any()
        assert not manager.terminated.any()
        assert not manager.time_outs.any()

    assert observer.sticky_terminated.tolist() == [True]
    assert observer.sticky_time_out.tolist() == [True]
    assert observer.first_terminated_step.tolist() == [1]
    assert observer.first_time_out_step.tolist() == [3]
    assert observer.sticky_terms.tolist() == [[True, False, False, True]]
    observer.assert_natural_timeout_completion(3)
    report = observer.report()
    assert report["auto_reset_suppressed"] is True
    observer.restore()
    assert manager.compute().tolist() == [False]


def test_natural_timeout_observer_rejects_missing_or_wrong_eval_semantics():
    manager = _FakeTerminationManager()
    manager._term_cfgs[-1].time_out = False
    with pytest.raises(ValueError, match="runtime termination"):
        NaturalMotionTimeoutObserver(manager)

    manager = _FakeTerminationManager()
    observer = NaturalMotionTimeoutObserver(manager)
    observer.install()
    manager.compute()
    manager.compute()
    with pytest.raises(RuntimeError, match="not observed"):
        observer.assert_natural_timeout_completion(2)
    observer.restore()


def test_manager_provenance_pins_config_runtime_functions_and_effective_params():
    configured = deepcopy(SONIC_EVALUATION_MANAGER_PROVENANCE["configured"])
    validated_config = validate_sonic_evaluation_config_provenance(
        configured["terminations"],
        configured["events"],
    )
    result = validate_sonic_evaluation_manager_provenance(
        _FakeTerminationManager(),
        _FakeEventManager(),
        configured_provenance=validated_config,
    )
    assert result == SONIC_EVALUATION_MANAGER_PROVENANCE

    changed_config = deepcopy(configured)
    changed_config["terminations"]["ee_body_pos"]["params"]["body_names"][2] = (
        "left_elbow_link"
    )
    with pytest.raises(ValueError, match="composed termination/event"):
        validate_sonic_evaluation_config_provenance(
            changed_config["terminations"],
            changed_config["events"],
        )

    changed_runtime = _FakeTerminationManager()
    changed_runtime._term_cfgs[0].params["threshold"] = 0.5
    with pytest.raises(ValueError, match="runtime termination"):
        validate_sonic_evaluation_manager_provenance(
            changed_runtime,
            _FakeEventManager(),
            configured_provenance=validated_config,
        )

    changed_event = _FakeEventManager()
    changed_event.cfg.mode = "startup"
    with pytest.raises(ValueError, match="runtime reset event"):
        validate_sonic_evaluation_manager_provenance(
            _FakeTerminationManager(),
            changed_event,
            configured_provenance=validated_config,
        )


def test_action_byte_evidence_is_exact_and_detects_signed_zero_or_row_change():
    action = torch.tensor([[0.0, 1.0, -2.0]], dtype=torch.float32)
    same = action.clone()
    signed_zero = action.clone()
    signed_zero[0, 0] = -0.0
    assert policy_action_row_sha256(action) == policy_action_row_sha256(same)
    assert policy_action_row_sha256(action) != policy_action_row_sha256(signed_zero)

    evidence = PolicyActionByteEvidence()
    evidence.update(action)
    evidence.update(same)
    report = evidence.report()
    assert report["step_count"] == 2
    assert report["row_sha256"][0] == report["row_sha256"][1]
    with pytest.raises(ValueError, match="dtype/shape"):
        evidence.update(torch.zeros(1, 4))

    candidate = dict(report)
    validate_policy_action_byte_parity(report, candidate)
    candidate["row_sha256"] = list(candidate["row_sha256"])
    candidate["row_sha256"][0] = policy_action_row_sha256(signed_zero)
    with pytest.raises(ValueError, match="aggregate|parity"):
        validate_policy_action_byte_parity(report, candidate)


def test_g1_only_encoder_selection_is_asserted_per_snapshot():
    tracking = SimpleNamespace(
        encoder_sample_probs_dict={"g1": 1.0, "teleop": 0.0, "smpl": 0.0},
        encoder_index=torch.tensor([[1, 0, 0], [1, 0, 0]]),
    )
    assert_g1_only_encoder_selection(tracking)
    tracking.encoder_index[1] = torch.tensor([0, 1, 0])
    with pytest.raises(RuntimeError, match="non-G1"):
        assert_g1_only_encoder_selection(tracking)


class _WritableComposer:
    def __init__(self):
        self.composed_force_as_torch = torch.ones(1, 4, 3)
        self.composed_torque_as_torch = torch.ones(1, 4, 3)

    def set_forces_and_torques(self, *, forces, torques, body_ids, env_ids, is_global):
        assert is_global is False
        self.composed_force_as_torch[env_ids[:, None], body_ids[None, :]] = forces
        self.composed_torque_as_torch[env_ids[:, None], body_ids[None, :]] = torques


def test_post_timeout_cleanup_zeroes_and_reads_all_owned_composer_rows():
    composer = _WritableComposer()
    force = torch.full((1, 3, 3), 7.0)
    torque = torch.full((1, 3, 3), 8.0)

    def clear_wrench(ids):
        force[ids] = 0.0
        torque[ids] = 0.0

    command = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        application_body_ids=torch.tensor([1, 2, 0]),
        robot=SimpleNamespace(permanent_wrench_composer=composer),
        clear_wrench=clear_wrench,
        body_wrench_for_envs=lambda ids: (force[ids], torque[ids], ids),
    )
    report = clear_and_assert_owned_composer_wrench(command)
    assert report["owned_force_peak_n"] == 0.0
    assert report["owned_torque_peak_nm"] == 0.0
    assert torch.count_nonzero(composer.composed_force_as_torch[:, [0, 1, 2]]) == 0
    assert torch.count_nonzero(composer.composed_torque_as_torch[:, [0, 1, 2]]) == 0


def _reset_event_fixture(*, force_n: float):
    composer = _WritableComposer()
    force = torch.zeros(1, 3, 3)
    torque = torch.zeros_like(force)
    force[0, 0, 0] = force_n
    body_ids = torch.tensor([1, 2, 0])
    env_ids = torch.tensor([0])
    composer.set_forces_and_torques(
        forces=force,
        torques=torque,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=False,
    )

    def clear_wrench(ids):
        force[ids] = 0.0
        torque[ids] = 0.0

    command = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        application_body_ids=body_ids,
        robot=SimpleNamespace(permanent_wrench_composer=composer),
        clear_wrench=clear_wrench,
        body_wrench_for_envs=lambda ids: (force[ids], torque[ids], ids),
        wrench_dirty=True,
        operational_enabled=True,
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_term=lambda name: command if name == "motion_compliance" else None
        )
    )
    return command, _FakeEventManager(env)


def test_configured_reset_event_follows_nonzero_force_and_clears_both_buffers():
    command, event_manager = _reset_event_fixture(force_n=7.0)
    report = exercise_sonic_evaluation_reset_event(
        event_manager,
        command,
        global_env_step_count=17,
    )
    assert report["pre_reset"]["command_force_peak_n"] == 7.0
    assert report["pre_reset"]["composer_force_peak_n"] == 7.0
    assert all(value == 0.0 for value in report["post_reset"].values())
    assert len(event_manager.apply_calls) == 1
    assert event_manager.apply_calls[0][0] == "reset"
    assert event_manager.apply_calls[0][2] == 17

    command, event_manager = _reset_event_fixture(force_n=0.0)
    with pytest.raises(RuntimeError, match="must follow a nonzero"):
        exercise_sonic_evaluation_reset_event(
            event_manager,
            command,
            global_env_step_count=17,
        )
    assert event_manager.apply_calls == []


def test_checkpoint_role_and_deterministic_event_contract_reject_mislabeled_evidence():
    residual_keys = tuple(f"{SONIC_ACTION_RESIDUAL_PREFIX}{index}" for index in range(6))
    validate_sonic_evaluation_checkpoint_role(
        protocol_role="baseline",
        checkpoint_sha256=SONIC_RELEASE_CHECKPOINT_SHA256,
        global_step=SONIC_RELEASE_CHECKPOINT_STEP,
        missing_policy_keys=residual_keys,
        unexpected_policy_keys=(),
        expected_action_residual_keys=residual_keys,
    )
    validate_sonic_evaluation_checkpoint_role(
        protocol_role="off",
        checkpoint_sha256=SONIC_TRAINED_CHECKPOINT_SHA256,
        global_step=SONIC_TRAINED_CHECKPOINT_STEP,
        missing_policy_keys=(),
        unexpected_policy_keys=(),
        expected_action_residual_keys=residual_keys,
    )
    with pytest.raises(ValueError, match="official"):
        validate_sonic_evaluation_checkpoint_role(
            protocol_role="baseline",
            checkpoint_sha256=SONIC_TRAINED_CHECKPOINT_SHA256,
            global_step=SONIC_TRAINED_CHECKPOINT_STEP,
            missing_policy_keys=(),
            unexpected_policy_keys=(),
            expected_action_residual_keys=residual_keys,
        )
    with pytest.raises(ValueError, match="strict"):
        validate_sonic_evaluation_checkpoint_role(
            protocol_role="off",
            checkpoint_sha256=SONIC_TRAINED_CHECKPOINT_SHA256,
            global_step=SONIC_TRAINED_CHECKPOINT_STEP,
            missing_policy_keys=(residual_keys[0],),
            unexpected_policy_keys=(),
            expected_action_residual_keys=residual_keys,
        )
    with pytest.raises(ValueError, match="unsupported"):
        validate_sonic_evaluation_checkpoint_role(
            protocol_role="bogus",
            checkpoint_sha256=SONIC_TRAINED_CHECKPOINT_SHA256,
            global_step=SONIC_TRAINED_CHECKPOINT_STEP,
            missing_policy_keys=(),
            unexpected_policy_keys=(),
            expected_action_residual_keys=residual_keys,
        )

    validate_sonic_evaluation_event_names(("motion_compliance_reset",))
    with pytest.raises(ValueError, match="permits only"):
        validate_sonic_evaluation_event_names(
            ("physics_material", "motion_compliance_reset")
        )


def test_isaaclab_bridge_source_uses_pre_reset_terminal_and_post_reset_callbacks():
    path = (
        Path(__file__).parents[1]
        / "compliance_control"
        / "adapters"
        / "sonic"
        / "evaluation_recorder.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def record_post_step" in source
    assert "self.env.reset_buf" in source
    assert "def record_post_reset" in source
    assert "apply_sonic_evaluation_protocol" in source
    assert "DatasetExportMode.EXPORT_NONE" in source


def _valid_sonic_suite_evidence():
    definitions = (
        ("released_baseline", "baseline", []),
        ("overlay_off", "off", []),
        ("enabled_no_contact", "no_contact", []),
        ("single_left", "single_site", ["left_wrist_yaw_link"]),
        ("single_right", "single_site", ["right_wrist_yaw_link"]),
        (
            "simultaneous",
            "multi_site",
            ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        ),
    )
    row_count = 3
    site_ids = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    point_ids = SONIC_RELEASE_TRACKING_BODY_NAMES
    traces = {}
    portable_specs = []
    for name, mode, active_sites in definitions:
        original_sites = np.zeros((row_count, len(site_ids), 3), dtype=np.float32)
        selected_sites = original_sites.copy()
        measured_sites = original_sites.copy()
        force = np.zeros_like(original_sites)
        enabled = np.full(
            row_count,
            mode not in {"baseline", "off"},
            dtype=np.bool_,
        )
        active = np.zeros((row_count, len(site_ids)), dtype=np.bool_)
        for site_id in active_sites:
            site_index = site_ids.index(site_id)
            active[:, site_index] = True
            selected_sites[:, site_index, 0] = 0.05
            measured_sites[1:, site_index, 0] = 0.049
            force[1:, site_index, 0] = 6.0
        orientations = np.zeros((row_count, len(site_ids), 4), dtype=np.float32)
        orientations[..., 3] = 1.0
        points = np.zeros((row_count, len(point_ids), 3), dtype=np.float32)
        terminal = np.asarray([False, False, True], dtype=np.bool_)
        success = terminal.copy()
        trace = EvaluationTrace(
            trial_name=name,
            motion_ids=("dataset_motion:0:start_frame:0",) * row_count,
            sequence_ids=("env_0000:episode_0000",) * row_count,
            seed_ids=np.zeros(row_count, dtype=np.int64),
            frame_indices=np.arange(row_count, dtype=np.int64),
            timestamps_s=np.arange(row_count, dtype=np.float64) * 0.02,
            site_ids=site_ids,
            point_ids=point_ids,
            original_site_positions_m=original_sites,
            selected_site_positions_m=selected_sites,
            measured_site_positions_m=measured_sites,
            original_site_orientations_xyzw=orientations,
            measured_site_orientations_xyzw=orientations.copy(),
            reference_points_global_m=points,
            measured_points_global_m=points.copy(),
            reference_points_local_m=points.copy(),
            measured_points_local_m=points.copy(),
            force_on_robot_n=force,
            compliance_enabled=enabled,
            active_site_mask=active,
            terminal_mask=terminal,
            success_mask=success,
            fall_mask=np.zeros(row_count, dtype=np.bool_),
            reset_mask=np.asarray([True, False, False], dtype=np.bool_),
        )
        traces[name] = trace
        portable_specs.append(
            TrialSpec(
                name=name,
                mode=TrialMode(mode),
                expected_active_site_ids=tuple(active_sites),
            )
        )
    criteria_kwargs = dict(SONIC_PHASE6_PORTABLE_CRITERIA)
    criteria_kwargs["endpoint_site_ids"] = tuple(criteria_kwargs["endpoint_site_ids"])
    criteria_kwargs["endpoint_tracking_point_ids"] = tuple(
        criteria_kwargs["endpoint_tracking_point_ids"]
    )
    paired = evaluate_trial_suite(
        traces,
        portable_specs,
        baseline_name="released_baseline",
        criteria=RegressionCriteria(**criteria_kwargs),
    )
    steps = 2
    row_hash = "00" * 32
    action_digest = hashlib.sha256()
    for _ in range(steps):
        action_digest.update(bytes.fromhex(row_hash))
    action = {
        "schema_version": "policy_action_bytes_v1",
        "dtype": "torch.float32",
        "shape_per_step": [1, 29],
        "step_count": steps,
        "row_sha256": [row_hash] * steps,
        "aggregate_sha256": action_digest.hexdigest(),
    }
    residual_keys = [f"{SONIC_ACTION_RESIDUAL_PREFIX}{index}" for index in range(6)]
    motion = {
        "file": "/audited/motion.pkl",
        "file_sha256": SONIC_AUDITED_MOTION_SHA256,
        "dataset_motion_id": 0,
        "internal_motion_id": 0,
        "key": "walk_forward_amateur_001__A001",
        "start_frame": 0,
        "initial_time_step": 0,
        "total_target_50hz_steps": steps,
        "target_fps": 50,
    }
    coordinates = {
        "world": "right-handed, Z-up, X-forward",
        "input_quaternion": "WXYZ",
        "persisted_quaternion": "XYZW",
    }
    collections = {}
    trace_hashes = {}
    for name, mode, active_sites in definitions:
        trace_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()
        trace_hashes[name] = trace_hash
        environment = deepcopy(SONIC_PHASE6_ENVIRONMENT_INVARIANTS)
        environment.update(
            {
                "host_operational_enabled": mode != "baseline",
                "logical_condition_enabled": mode not in {"baseline", "off"},
            }
        )
        baseline = mode == "baseline"
        collections[name] = {
            "schema_version": "sonic_phase6_collection_v3",
            "evidence_kind": "real_sonic_simulator_trace",
            "trial_name": name,
            "protocol": mode,
            "active_site_ids": active_sites,
            "protocol_parameters": {
                "force_threshold_n": 10.0,
                "reference_offset_common_m": [0.05, 0.0, 0.0],
                "derived_stiffness_n_per_m": 200.0,
                "resolved_initial_condition": (
                    [0.0, 0.0, 0.0]
                    if mode in {"baseline", "off"}
                    else [1.0, 10.0, 200.0]
                ),
            },
            "seed": 0,
            "executed_steps": steps,
            "natural_motion_timeout_observed": True,
            "motion": deepcopy(motion),
            "tracking_body_layout": list(SONIC_RELEASE_TRACKING_BODY_NAMES),
            "policy_step_dt_s": 0.02,
            "coordinate_convention": deepcopy(coordinates),
            "deterministic_environment": environment,
            "checkpoint_sha256": (
                SONIC_RELEASE_CHECKPOINT_SHA256
                if baseline
                else SONIC_TRAINED_CHECKPOINT_SHA256
            ),
            "checkpoint_global_step": (
                SONIC_RELEASE_CHECKPOINT_STEP
                if baseline
                else SONIC_TRAINED_CHECKPOINT_STEP
            ),
            "checkpoint_role": "official_release" if baseline else "accepted_step6",
            "checkpoint_load": {
                "missing_policy_keys": residual_keys if baseline else [],
                "unexpected_policy_keys": [],
                "expected_action_residual_keys": residual_keys,
            },
            "trace_sha256": trace_hash,
            "alignment_sha256": alignment_digest(traces[name]),
            "policy_action_evidence": deepcopy(action),
            "termination_evidence": {
                "schema_version": "natural_motion_timeout_observer_v1",
                "compute_count": steps,
                "sticky_time_out": [True],
                "first_time_out_step": [steps],
                "term_names": list(SONIC_EVALUATION_TERMINATION_NAMES),
                "term_observation_counts": [[0, 0, 0, 1]],
                "first_term_step": [[-1, -1, -1, steps]],
            },
            "manager_provenance": deepcopy(SONIC_EVALUATION_MANAGER_PROVENANCE),
            "reset_event_evidence": (
                {
                    "schema_version": "sonic_phase6_reset_event_evidence_v1",
                    "event_name": "motion_compliance_reset",
                    "resolved_func_target": (
                        "gear_sonic.compliance_control.adapters.sonic.event:"
                        "reset_compliance_wrench"
                    ),
                    "mode": "reset",
                    "global_env_step_count": 1,
                    "pre_reset": {
                        "command_force_peak_n": 6.0,
                        "command_torque_peak_nm": 0.0,
                        "composer_force_peak_n": 6.0,
                        "composer_torque_peak_nm": 0.0,
                        "force_max_abs_difference_n": 0.0,
                        "torque_max_abs_difference_nm": 0.0,
                    },
                    "post_reset": {
                        "command_force_peak_n": 0.0,
                        "command_torque_peak_nm": 0.0,
                        "composer_force_peak_n": 0.0,
                        "composer_torque_peak_nm": 0.0,
                        "force_max_abs_difference_n": 0.0,
                        "torque_max_abs_difference_nm": 0.0,
                    },
                }
                if mode in {"single_site", "multi_site"}
                else None
            ),
            "actual_composer_evidence": {
                "source": "permanent_wrench_composer_body_local_owned_rows",
                "reset_owned_force_peak_n": 0.0,
                "reset_owned_torque_peak_nm": 0.0,
                "owned_force_buffer_max_abs_difference_n": 0.0,
                "owned_torque_buffer_max_abs_difference_nm": 0.0,
            },
            "post_timeout_clear_evidence": {
                "owned_force_peak_n": 0.0,
                "owned_torque_peak_nm": 0.0,
            },
            "metrics": deepcopy(paired["trials"][name]),
        }
    paired["trace_sha256_by_trial"] = deepcopy(trace_hashes)
    report_hashes = {name: "cd" * 32 for name in collections}
    return paired, collections, report_hashes, trace_hashes, traces


def _validate_fake_sonic_suite(
    paired,
    collections,
    report_hashes,
    trace_hashes,
    traces,
    **kwargs,
):
    return validate_sonic_collection_suite(
        paired,
        collections,
        paired_report_sha256="ef" * 32,
        collection_report_sha256=report_hashes,
        observed_trace_sha256=trace_hashes,
        observed_traces=traces,
        **kwargs,
    )


def test_sonic_suite_validator_accepts_protocol_specific_gates_and_pins_evidence():
    paired, collections, report_hashes, trace_hashes, traces = (
        _valid_sonic_suite_evidence()
    )
    result = _validate_fake_sonic_suite(
        paired,
        collections,
        report_hashes,
        trace_hashes,
        traces,
    )
    assert result["acceptance"]["passed"] is True

    lax = deepcopy(paired)
    lax["acceptance"]["criteria"]["endpoint_rmse_regression_m"] = 100.0
    result = _validate_fake_sonic_suite(
        lax, collections, report_hashes, trace_hashes, traces
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "portable_phase6_criteria" in failed

    lax_active_tracking = deepcopy(paired)
    lax_active_tracking["acceptance"]["criteria"][
        "active_selected_endpoint_rmse_regression_m"
    ] = 1.0
    result = _validate_fake_sonic_suite(
        lax_active_tracking,
        collections,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {
        check["name"]
        for check in result["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "portable_phase6_criteria" in failed

    rebound = deepcopy(paired)
    rebound["trace_sha256_by_trial"]["single_left"] = "11" * 32
    result = _validate_fake_sonic_suite(
        rebound, collections, report_hashes, trace_hashes, traces
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "portable_vs_observed_trace_sha256:single_left" in failed

    replaced_motion = deepcopy(collections)
    replaced_motion["single_left"]["motion"]["file_sha256"] = "22" * 32
    result = _validate_fake_sonic_suite(
        paired,
        replaced_motion,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "motion_file_sha256:single_left" in failed

    changed_stimulus = deepcopy(collections)
    changed_stimulus["single_left"]["protocol_parameters"][
        "force_threshold_n"
    ] = 9.0
    result = _validate_fake_sonic_suite(
        paired,
        changed_stimulus,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "protocol_parameters:single_left" in failed

    changed_dt = deepcopy(collections)
    changed_dt["single_left"]["policy_step_dt_s"] = 0.01
    result = _validate_fake_sonic_suite(
        paired,
        changed_dt,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "policy_step_dt_s:single_left" in failed

    changed_manager = deepcopy(collections)
    changed_manager["single_left"]["manager_provenance"]["runtime"][
        "terminations"
    ][0]["effective_params"]["threshold"] = 0.5
    result = _validate_fake_sonic_suite(
        paired,
        changed_manager,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "manager_provenance:single_left" in failed

    stale_reset = deepcopy(collections)
    stale_reset["single_left"]["reset_event_evidence"]["post_reset"][
        "composer_force_peak_n"
    ] = 1.0
    result = _validate_fake_sonic_suite(
        paired,
        stale_reset,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "post_reset_exact_zero_composer_force_peak_n:single_left" in failed

    missing_nonzero_reset = deepcopy(collections)
    missing_nonzero_reset["single_left"]["reset_event_evidence"]["pre_reset"][
        "command_force_peak_n"
    ] = 0.0
    result = _validate_fake_sonic_suite(
        paired,
        missing_nonzero_reset,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "pre_reset_command_force_nonzero:single_left" in failed

    inactive_reset = deepcopy(collections)
    inactive_reset["overlay_off"]["reset_event_evidence"] = deepcopy(
        collections["single_left"]["reset_event_evidence"]
    )
    result = _validate_fake_sonic_suite(
        paired,
        inactive_reset,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "inactive_protocol_no_reset_event:overlay_off" in failed

    boolean_ids = deepcopy(collections)
    for report in boolean_ids.values():
        report["seed"] = False
        report["motion"]["dataset_motion_id"] = False
        report["motion"]["internal_motion_id"] = False
        report["motion"]["start_frame"] = False
        report["motion"]["initial_time_step"] = False
    result = _validate_fake_sonic_suite(
        paired,
        boolean_ids,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "seed_pinned_zero:released_baseline" in failed
    assert "motion_dataset_id:released_baseline" in failed
    assert "motion_internal_id:released_baseline" in failed
    assert "motion_start_frame:released_baseline" in failed
    assert "motion_initial_time:released_baseline" in failed

    result = _validate_fake_sonic_suite(
        paired,
        collections,
        report_hashes,
        trace_hashes,
        traces,
        reset_owned_force_tolerance_n=1.0,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "reset_owned_force_tolerance_pinned" in failed


def test_sonic_suite_validator_rejects_fabricated_portable_check_success():
    paired, collections, report_hashes, trace_hashes, traces = (
        _valid_sonic_suite_evidence()
    )
    paired["acceptance"]["checks"].pop()
    result = _validate_fake_sonic_suite(
        paired,
        collections,
        report_hashes,
        trace_hashes,
        traces,
    )
    failed = {check["name"] for check in result["acceptance"]["checks"] if not check["passed"]}
    assert "portable_report_recomputed_from_bound_traces" in failed


def test_sonic_suite_validator_pins_exact_six_protocol_site_semantics():
    paired, collections, report_hashes, trace_hashes, traces = (
        _valid_sonic_suite_evidence()
    )
    duplicated_single = deepcopy(paired)
    single_right = next(
        spec
        for spec in duplicated_single["trial_specs"]
        if spec["name"] == "single_right"
    )
    single_right["expected_active_site_ids"] = ["left_wrist_yaw_link"]
    with pytest.raises(ValueError, match="each wrist exactly once"):
        _validate_fake_sonic_suite(
            duplicated_single,
            collections,
            report_hashes,
            trace_hashes,
            traces,
        )

    active_baseline = deepcopy(paired)
    baseline = next(
        spec
        for spec in active_baseline["trial_specs"]
        if spec["mode"] == "baseline"
    )
    baseline["expected_active_site_ids"] = ["left_wrist_yaw_link"]
    with pytest.raises(ValueError, match="baseline trial must have no"):
        _validate_fake_sonic_suite(
            active_baseline,
            collections,
            report_hashes,
            trace_hashes,
            traces,
        )
