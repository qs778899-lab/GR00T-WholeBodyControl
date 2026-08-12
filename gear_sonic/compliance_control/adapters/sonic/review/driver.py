"""Thin runtime driver that applies deterministic review samples to SONIC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..frames import world_vectors_to_frame_prevalidated
from ..sampling import limit_peak_forces_by_net_wrench_prevalidated
from .protocol import DeterministicForceProtocol, ProtocolSample
from .roles import REVIEW_SITE_NAMES, ReviewRole


@dataclass(frozen=True, slots=True)
class AppliedProtocolSample:
    """The final safety-checked vectors written for one pre-transition sample."""

    protocol: ProtocolSample
    force_on_robot_common_n: torch.Tensor
    force_on_robot_world_n: torch.Tensor
    compliance_m_per_n: torch.Tensor
    active_site_mask: torch.Tensor
    command_enabled: torch.Tensor


class SonicReviewProtocolDriver:
    """Own deterministic command-state and permanent-wrench writes for one env."""

    def __init__(
        self,
        command: object,
        role: ReviewRole,
        *,
        protocol: DeterministicForceProtocol | None = None,
    ) -> None:
        if not isinstance(role, ReviewRole):
            raise TypeError("role must be a ReviewRole")
        self.command = command
        self.role = role
        self.protocol = protocol or DeterministicForceProtocol()
        state = getattr(command, "state", None)
        sites = getattr(command, "sites", None)
        site_spec = getattr(sites, "spec", None)
        site_names = tuple(getattr(site_spec, "site_names", ()))
        if site_names != REVIEW_SITE_NAMES:
            raise AssertionError("review command sites must use the ordered wrist contract")
        if getattr(state, "num_envs", None) != 1:
            raise AssertionError("review driver requires exactly one environment")

    @property
    def state(self):
        return self.command.state

    def reset(self) -> None:
        """Use the accepted command reset path and prove all command buffers zero."""

        self.command.reset_envs(None)
        state = self.state
        for field_name in (
            "enabled",
            "site_mask",
            "compliance",
            "force_on_robot_w",
            "peak_force_on_robot_w",
            "pulse_active",
        ):
            value = getattr(state, field_name)
            if torch.count_nonzero(value).item() != 0:
                raise AssertionError(f"reset left nonzero command state: {field_name}")

    def apply(self, frame_index: int, frame_count: int) -> AppliedProtocolSample:
        """Write one exact sample immediately before its corresponding action."""

        sample = self.protocol.sample(self.role, frame_index, frame_count)
        state = self.state
        device = state.device
        dtype = state.dtype
        requested_world_force = torch.tensor(
            sample.force_on_robot_world_n,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        compliance = torch.tensor(
            sample.compliance_m_per_n,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        site_mask = torch.tensor(
            sample.active_site_mask,
            device=device,
            dtype=torch.bool,
        ).unsqueeze(0)
        command_enabled = torch.tensor(
            [sample.compliance_enabled],
            device=device,
            dtype=torch.bool,
        )
        _, anchor_quaternion_wxyz = self.command._anchor_pose_w()  # noqa: SLF001
        application_positions_w = self.command.current_site_positions_w()
        wrench_origin_w = self.command.robot.data.body_pos_w[
            :, self.command.anchor_body_index
        ]
        final_world_force = limit_peak_forces_by_net_wrench_prevalidated(
            requested_world_force,
            application_positions_w,
            wrench_origin_w,
            max_net_force_n=self.command.cfg.max_net_force_n,
            max_net_torque_nm=self.command.cfg.max_net_torque_nm,
        )
        final_common_force = world_vectors_to_frame_prevalidated(
            final_world_force,
            frame=self.command.sites.spec.common_frame,
            anchor_quaternion_wxyz=anchor_quaternion_wxyz,
        )
        if not torch.equal(final_world_force, requested_world_force):
            raise AssertionError(
                "review force hit a resultant-wrench limit; matched bytes are no longer pinned"
            )
        env_ids = torch.zeros(1, dtype=torch.long, device=device)
        state.set_samples(
            env_ids,
            enabled=command_enabled,
            site_mask=site_mask,
            compliance=compliance,
            force_on_robot_w=final_world_force,
        )
        self.command.wrench.set_world_forces_prevalidated(
            final_world_force,
            body_quaternions_wxyz=self.command.current_site_quaternions_wxyz(),
            application_offsets_local=self.command._application_offsets_local,  # noqa: SLF001
        )
        self.command._wrench_write_gate.mark_written()  # noqa: SLF001
        return AppliedProtocolSample(
            protocol=sample,
            force_on_robot_common_n=final_common_force.clone(),
            force_on_robot_world_n=final_world_force.clone(),
            compliance_m_per_n=compliance.clone(),
            active_site_mask=site_mask.clone(),
            command_enabled=command_enabled.clone(),
        )


def gate_actor_observations(
    observations: Mapping[str, torch.Tensor],
    role: ReviewRole,
) -> dict[str, torch.Tensor]:
    """Return role-specific actor inputs without mutating manager observations."""

    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping")
    if not isinstance(role, ReviewRole):
        raise TypeError("role must be a ReviewRole")
    result = dict(observations)
    command = result.get("compliance_command")
    if not isinstance(command, torch.Tensor):
        raise KeyError("observations must contain tensor compliance_command")
    if role.actor_hard_off:
        result["compliance_command"] = torch.zeros_like(command)
    return result


def refresh_compliance_observations(
    raw_env: object,
    observations: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Refresh only command-owned groups after the deterministic pre-step write."""

    from ..isaaclab.observations import (
        sonic_compliance_actor_command,
        sonic_compliance_force_common,
        sonic_compliance_target,
    )

    result = dict(observations)
    result["compliance_target"] = sonic_compliance_target(
        raw_env,
        non_flatten=False,
    )
    result["compliance_command"] = sonic_compliance_actor_command(raw_env)
    result["compliance_force"] = sonic_compliance_force_common(raw_env)
    return result
