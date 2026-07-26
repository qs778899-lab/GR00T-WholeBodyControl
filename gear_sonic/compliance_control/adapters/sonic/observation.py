"""Actor-safe and critic-only observations for motion compliance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
import torch

from .contracts import (
    condition_from_command,
    current_site_force_from_command,
    site_mask_from_command,
    threshold_from_command,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class MotionCompliancePolicyCfg(ObsGroup):
    """Released proprioception plus one public three-value condition."""

    # Keep the selected release PolicyCfg declaration order exactly; condition
    # is appended so the original 930 proprioceptive columns do not move.
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    gravity_dir = None
    motion_compliance_condition = None


@configclass
class MotionCompliancePrivilegedCfg(ObsGroup):
    """Released critic state plus configurable-site compliance state."""

    command_multi_future = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b = None
    body_pos = None
    body_ori = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    motion_compliance_condition = None
    motion_compliance_threshold = None
    motion_compliance_site_force = None
    motion_compliance_site_mask = None


def _compliance_command(env: ManagerBasedEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def motion_compliance_condition(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Return exactly ``[enable, enable*threshold, enable*Kp]`` to the actor."""

    return condition_from_command(_compliance_command(env, command_name))


def motion_compliance_threshold(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Return the sampled scalar force threshold to the critic only."""

    return threshold_from_command(_compliance_command(env, command_name))


def motion_compliance_site_force(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Return current-frame applied site forces in current-anchor coordinates.

    The residual limiter preserves each site wrench and adds compensation only
    at the anchor, so the cached common-frame site force is the applied site
    force.  Its flattened width is ``3 * configured_num_sites``.
    """

    return current_site_force_from_command(_compliance_command(env, command_name))


def motion_compliance_site_mask(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Return the configured-site active mask to the critic only."""

    return site_mask_from_command(_compliance_command(env, command_name))
