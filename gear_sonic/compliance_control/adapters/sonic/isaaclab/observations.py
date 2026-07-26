"""Read-only Isaac Lab observation entrypoints for SONIC compliance."""

from __future__ import annotations

import torch

from .command import SonicComplianceCommand
from ..observation import build_sonic_compliance_targets_prevalidated


def sonic_compliance_target(
    env,
    *,
    motion_command_name: str = "motion",
    compliance_command_name: str = "force",
    non_flatten: bool = True,
) -> torch.Tensor:
    """Return the sole actor-facing observed target without advancing state."""

    motion = env.command_manager.get_term(motion_command_name)
    compliance = env.command_manager.get_term(compliance_command_name)
    if not isinstance(compliance, SonicComplianceCommand):
        raise TypeError(
            f"command {compliance_command_name!r} is not a SonicComplianceCommand"
        )
    runtime_reference_names = tuple(motion.cfg.body_names)
    if runtime_reference_names != compliance.reference_body_names:
        raise ValueError(
            "runtime motion body order differs from compliance resolver input; "
            "recompose the Hydra command config"
        )

    num_envs = compliance.num_envs
    num_future = compliance.cfg.num_future_frames
    num_reference_bodies = len(runtime_reference_names)
    reference_positions_w = motion.body_pos_w_multi_future.view(
        num_envs,
        num_future,
        num_reference_bodies,
        3,
    )
    reference_quaternions_wxyz = motion.body_quat_w_multi_future.view(
        num_envs,
        num_future,
        num_reference_bodies,
        4,
    )
    anchor_position_w, anchor_quaternion_wxyz = compliance._anchor_pose_w()  # noqa: SLF001
    targets = build_sonic_compliance_targets_prevalidated(
        reference_positions_w=reference_positions_w,
        reference_quaternions_wxyz=reference_quaternions_wxyz,
        articulation_positions_w=compliance.robot.data.body_pos_w,
        articulation_quaternions_wxyz=compliance.robot.data.body_quat_w,
        anchor_position_w=anchor_position_w,
        anchor_quaternion_wxyz=anchor_quaternion_wxyz,
        state=compliance.state,
        use_target_damper=compliance.use_target_damper,
    )
    if non_flatten:
        return targets.observed_target_common.reshape(num_envs, num_future, -1)
    return targets.observed_target_common.reshape(num_envs, -1)
