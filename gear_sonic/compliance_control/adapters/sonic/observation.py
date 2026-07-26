"""Actor-safe and critic-only observations for motion compliance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
import torch

from gear_sonic.envs.manager_env.mdp.observations import ObservationsCfg

from .contracts import (
    condition_from_command,
    current_site_force_from_command,
    site_mask_from_command,
    threshold_from_command,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class MotionComplianceConditionCfg(ObsGroup):
    """Actor-visible condition kept separate from released proprioception."""

    motion_compliance_condition = None


@configclass
class MotionCompliancePrivilegedCfg(ObsGroup):
    """Critic-only site state kept separate from released critic input."""

    motion_compliance_threshold = None
    motion_compliance_site_force = None
    motion_compliance_site_mask = None


@configclass
class MotionComplianceObservationsCfg(ObservationsCfg):
    """Add two groups without changing released policy/critic groups."""

    motion_compliance_condition: MotionComplianceConditionCfg = None
    motion_compliance_privileged: MotionCompliancePrivilegedCfg = None


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
