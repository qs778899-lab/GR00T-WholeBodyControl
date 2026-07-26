"""Yield-aware endpoint rewards layered beside SONIC's dense tracking rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
import torch

from .command import _articulation_body_data
from .contracts import (
    current_endpoint_position_errors_from_command,
    gated_mean_gaussian_reward,
    quaternion_error_magnitude_wxyz,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class MotionComplianceRewardsCfg:
    """Released dense rewards plus explicit compliant endpoint terms."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    tracking_relative_body_ori = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    anti_shake_ang_vel = None
    tracking_vr_5point_local = None
    feet_acc = None
    tracking_compliant_endpoint_pos = None
    tracking_endpoint_ori = None


def _motion_compliance_command(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def endpoint_position_errors_per_site(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected and original current-frame errors for every site.

    Both tensors retain the exact order of ``site_body_names``, so left/right
    degradation is reportable independently rather than hidden by the mean.
    """

    command = _motion_compliance_command(env, command_name)
    # IsaacLab evaluates rewards before the command manager computes the next
    # command.  Re-read the current articulation state here so the endpoint
    # error is aligned with the just-finished physics step.  Keep this local:
    # reward evaluation must not mutate command-owned force/reference caches.
    return current_endpoint_position_errors_from_command(command)


def endpoint_position_error_per_site(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return selected-target position error in configured site order."""

    return endpoint_position_errors_per_site(env, command_name)[0]


def endpoint_orientation_error_per_site(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return original-reference orientation error per configured site.

    Rotational compliance is intentionally absent in Phase 2, so orientation
    never switches to a yielded target.  Reference orientation uses future
    frame zero, matching the current-frame force and position reward.
    """

    command = _motion_compliance_command(env, command_name)
    tracking = command._tracking_term()
    reference_quaternion = tracking.body_quat_w_multi_future.reshape(
        command.num_envs,
        command.state.num_future_frames,
        len(command.cfg.reference_body_names),
        4,
    )[:, 0, command.body_map.reference_site_indices]
    current_quaternion = _articulation_body_data(command.robot, "quat")[
        :, command.body_map.articulation_site_indices
    ]
    return quaternion_error_magnitude_wxyz(
        reference_quaternion,
        current_quaternion,
    )


def tracking_compliant_endpoint_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward active positions against yielded targets and inactive ones exactly original."""

    command = _motion_compliance_command(env, command_name)
    selected_error, original_error = endpoint_position_errors_per_site(env, command_name)
    command.record_endpoint_errors(
        selected_position_error_m=selected_error,
        original_position_error_m=original_error,
    )
    return gated_mean_gaussian_reward(
        selected_error,
        command.state.enabled,
        std,
    )


def tracking_endpoint_orientation(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Keep explicit endpoint orientation tracking against the original motion."""

    command = _motion_compliance_command(env, command_name)
    error = endpoint_orientation_error_per_site(env, command_name)
    command.record_endpoint_errors(orientation_error_rad=error)
    return gated_mean_gaussian_reward(error, command.state.enabled, std)
