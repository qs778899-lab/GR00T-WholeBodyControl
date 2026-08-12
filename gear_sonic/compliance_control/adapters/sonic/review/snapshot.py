"""Name-resolved SONIC snapshot mapping into the review trace schema."""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import require_sonic_release_tracking_body_names
from ..frames import (
    frame_positions_to_world,
    quaternion_rotate_wxyz_prevalidated,
    world_positions_to_frame_prevalidated,
)
from ..observation import build_sonic_compliance_targets_prevalidated
from .trace import SonicReviewSnapshot


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _normalized_xyzw(quaternion_wxyz: torch.Tensor, *, label: str) -> np.ndarray:
    if not torch.isfinite(quaternion_wxyz).all():
        raise ValueError(f"{label} quaternions are not finite")
    norms = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    if (norms <= torch.finfo(quaternion_wxyz.dtype).eps).any():
        raise ValueError(f"{label} quaternions contain zero norms")
    normalized = quaternion_wxyz / norms
    return _numpy(normalized[..., (1, 2, 3, 0)])


def capture_sonic_review_snapshot(motion: object, command: object) -> SonicReviewSnapshot:
    """Capture one pre-transition sample with explicit reference/index spaces."""

    body_names = require_sonic_release_tracking_body_names(motion.cfg.body_names)
    if tuple(command.reference_body_names) != body_names:
        raise AssertionError("command reference order differs from release body order")
    if command.num_envs != 1:
        raise AssertionError("review snapshot requires one environment")
    num_future = command.cfg.num_future_frames
    reference_positions_future_w = motion.body_pos_w_multi_future.view(
        1,
        num_future,
        len(body_names),
        3,
    )
    reference_quaternions_future_wxyz = motion.body_quat_w_multi_future.view(
        1,
        num_future,
        len(body_names),
        4,
    )
    anchor_position_w, anchor_quaternion_wxyz = command._anchor_pose_w()  # noqa: SLF001
    targets = build_sonic_compliance_targets_prevalidated(
        reference_positions_w=reference_positions_future_w,
        reference_quaternions_wxyz=reference_quaternions_future_wxyz,
        articulation_positions_w=command.robot.data.body_pos_w,
        articulation_quaternions_wxyz=command.robot.data.body_quat_w,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
        state=command.state,
        use_target_damper=command.use_target_damper,
    )
    if targets.damper_used:
        raise AssertionError("formal review requires the undamped CHIP target")

    reference_indices = torch.tensor(
        command.sites.reference_indices,
        dtype=torch.long,
        device=command.device,
    )
    offsets = command.application_offsets_local()
    reference_site_quaternions_wxyz = motion.body_quat_w.index_select(
        1,
        reference_indices,
    )
    original_site_world = motion.body_pos_w.index_select(1, reference_indices)
    original_site_world = original_site_world + quaternion_rotate_wxyz_prevalidated(
        reference_site_quaternions_wxyz,
        offsets,
    )
    world_force = command.state.force_on_robot_w
    compliance = command.state.compliance
    selected_site_world = original_site_world - compliance * world_force
    selected_common_from_world = world_positions_to_frame_prevalidated(
        selected_site_world,
        frame=command.sites.spec.common_frame,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
    )
    if not torch.allclose(
        selected_common_from_world,
        targets.observed_target_common[:, 0],
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise AssertionError("SONIC target builder violates signed CHIP world equivalence")
    # Exercise the inverse transform as a second independent consistency check.
    selected_world_roundtrip = frame_positions_to_world(
        targets.observed_target_common[:, 0],
        frame=command.sites.spec.common_frame,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
    )
    if not torch.allclose(
        selected_world_roundtrip,
        selected_site_world,
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise AssertionError("selected target common/world roundtrip failed")

    reference_points_world = motion.body_pos_w
    measured_points_world = motion.robot_body_pos_w
    reference_points_local = world_positions_to_frame_prevalidated(
        reference_points_world,
        frame=command.sites.spec.common_frame,
        anchor_position_w=motion.anchor_pos_w,
        anchor_quaternion_wxyz=motion.anchor_quat_w,
    )
    measured_points_local = world_positions_to_frame_prevalidated(
        measured_points_world,
        frame=command.sites.spec.common_frame,
        anchor_position_w=motion.robot_anchor_pos_w,
        anchor_quaternion_wxyz=motion.robot_anchor_quat_w,
    )
    absolute_frame = motion.motion_start_time_steps + motion.time_steps
    frame_value = int(absolute_frame[0].detach().cpu().item())
    actual_site_quaternions_wxyz = command.current_site_quaternions_wxyz()
    return SonicReviewSnapshot(
        reference_frame=frame_value,
        original_site_positions_m=_numpy(original_site_world[0]),
        selected_site_positions_m=_numpy(selected_site_world[0]),
        measured_site_positions_m=_numpy(command.current_site_positions_w()[0]),
        original_site_orientations_xyzw=_normalized_xyzw(
            reference_site_quaternions_wxyz[0],
            label="reference site",
        ),
        measured_site_orientations_xyzw=_normalized_xyzw(
            actual_site_quaternions_wxyz[0],
            label="actual site",
        ),
        reference_points_global_m=_numpy(reference_points_world[0]),
        measured_points_global_m=_numpy(measured_points_world[0]),
        reference_points_local_m=_numpy(reference_points_local[0]),
        measured_points_local_m=_numpy(measured_points_local[0]),
        force_on_robot_n=_numpy(world_force[0]),
        force_on_robot_world_n=_numpy(world_force[0]),
        force_on_robot_common_n=_numpy(targets.force_on_robot_common[0, 0]),
        compliance_m_per_n=_numpy(compliance[0]),
        active_site_mask=_numpy(command.state.site_mask[0]),
    )
