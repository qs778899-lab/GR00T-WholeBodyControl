"""Isaac Lab event entrypoints for compliance-force application and reset."""

from __future__ import annotations

import torch

from .command import SonicComplianceCommand


def _get_command(env, command_name: str) -> SonicComplianceCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, SonicComplianceCommand):
        raise TypeError(f"command {command_name!r} is not a SonicComplianceCommand")
    return command


def apply_compliance_force(
    env,
    env_ids: torch.Tensor | None,
    *,
    command_name: str = "force",
) -> None:
    """Sample and persist world-frame force-on-robot at arbitrary active sites."""

    _get_command(env, command_name).sample_and_apply(env_ids)


def reset_compliance_force(
    env,
    env_ids: torch.Tensor | None,
    *,
    command_name: str = "force",
) -> None:
    """Clear wrench and command/damper state for resetting environments."""

    _get_command(env, command_name).reset_envs(env_ids)
