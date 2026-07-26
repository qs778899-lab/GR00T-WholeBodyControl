"""Narrow PhysX wrench writer/reset boundary with IsaacLab API feature detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _set_body_wrench(
    asset: Any,
    forces_body: torch.Tensor,
    torques_body: torch.Tensor,
    body_ids: torch.Tensor,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> None:
    """Prefer WrenchComposer and isolate the deprecated setter fallback."""

    composer = getattr(asset, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "set_forces_and_torques"):
        composer.set_forces_and_torques(
            forces=forces_body,
            torques=torques_body,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=False,
        )
        return
    setter = getattr(asset, "set_external_force_and_torque", None)
    if setter is None:
        raise RuntimeError("articulation has no supported external-wrench writer")
    setter(
        forces=forces_body,
        torques=torques_body,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=False,
    )


def _reset_wrench_composer(
    asset: Any,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    *,
    zero_forces_body: torch.Tensor,
    zero_torques_body: torch.Tensor,
    body_ids: torch.Tensor,
) -> None:
    composer = getattr(asset, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "reset"):
        composer.reset(env_ids)
        return
    _set_body_wrench(
        asset,
        zero_forces_body,
        zero_torques_body,
        body_ids,
        env_ids,
    )


def apply_compliance_wrench(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None,
    command_name: str = "motion_compliance",
) -> None:
    """Write current link-frame wrenches, avoiding composer pose caching."""

    command = env.command_manager.get_term(command_name)
    if not command.operational_enabled:
        if not command.wrench_dirty:
            return
        command.clear_wrench(env_ids)
        forces, torques, resolved_env_ids = command.body_wrench_for_envs(env_ids)
        _set_body_wrench(
            command.robot,
            forces,
            torques,
            command.application_body_ids,
            resolved_env_ids,
        )
        command.mark_wrench_cleared()
        return
    forces, torques, resolved_env_ids = command.body_wrench_for_envs(env_ids)
    _set_body_wrench(
        command.robot,
        forces,
        torques,
        command.application_body_ids,
        resolved_env_ids,
    )
    command.mark_wrench_applied()


def reset_compliance_wrench(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    command_name: str = "motion_compliance",
) -> None:
    """Clear command outputs and the articulation composer to prevent stale force."""

    command = env.command_manager.get_term(command_name)
    was_dirty = command.wrench_dirty
    command.clear_wrench(env_ids)
    if not was_dirty:
        return
    zero_forces, zero_torques, resolved_env_ids = command.body_wrench_for_envs(env_ids)
    _reset_wrench_composer(
        command.robot,
        resolved_env_ids,
        zero_forces_body=zero_forces,
        zero_torques_body=zero_torques,
        body_ids=command.application_body_ids,
    )
