"""Replaceable whole-robot residual wrench limiting and anchor compensation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ...core.math import _clamp_vector_norm_unchecked


@dataclass(frozen=True)
class WrenchLimitResult:
    """Site wrench, anchor compensation, and residual net wrench in world frame."""

    site_force_world: torch.Tensor
    site_torque_world: torch.Tensor
    anchor_force_world: torch.Tensor
    anchor_torque_world: torch.Tensor
    raw_net_force_world: torch.Tensor
    raw_net_torque_world: torch.Tensor
    residual_force_world: torch.Tensor
    residual_torque_world: torch.Tensor


class ResidualWrenchLimiter:
    """Limit residual net wrench by compensating excess at an anchor body."""

    def __init__(self, max_force_n: float = 20.0, max_torque_nm: float = 10.0):
        if (
            not math.isfinite(max_force_n)
            or not math.isfinite(max_torque_nm)
            or max_force_n <= 0.0
            or max_torque_nm <= 0.0
        ):
            raise ValueError("wrench limits must be finite and positive")
        self.max_force_n = float(max_force_n)
        self.max_torque_nm = float(max_torque_nm)

    def __call__(
        self,
        site_position_world: torch.Tensor,
        anchor_position_world: torch.Tensor,
        site_force_world: torch.Tensor,
        site_torque_world: torch.Tensor | None = None,
    ) -> WrenchLimitResult:
        if site_position_world.shape != site_force_world.shape:
            raise ValueError("site positions and forces must have identical shapes")
        if site_position_world.ndim != 3 or site_position_world.shape[-1] != 3:
            raise ValueError("site tensors must have shape [batch, sites, 3]")
        if anchor_position_world.shape != (site_position_world.shape[0], 3):
            raise ValueError("anchor_position_world must have shape [batch, 3]")
        for name, tensor in (
            ("site_position_world", site_position_world),
            ("anchor_position_world", anchor_position_world),
            ("site_force_world", site_force_world),
        ):
            if tensor.dtype != site_position_world.dtype:
                raise TypeError(f"{name} must share the site-position dtype")
            if tensor.device != site_position_world.device:
                raise ValueError(f"{name} must share the site-position device")
            if not tensor.is_floating_point() or tensor.is_complex():
                raise TypeError(f"{name} must have a real floating dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if site_torque_world is None:
            site_torque_world = torch.zeros_like(site_force_world)
        if site_torque_world.shape != site_force_world.shape:
            raise ValueError("site torques and forces must have identical shapes")
        if site_torque_world.dtype != site_force_world.dtype:
            raise TypeError("site torques and forces must share dtype")
        if site_torque_world.device != site_force_world.device:
            raise ValueError("site torques and forces must share device")
        if not torch.isfinite(site_torque_world).all():
            raise ValueError("site torques must be finite")

        return self._limit_unchecked(
            site_position_world,
            anchor_position_world,
            site_force_world,
            site_torque_world,
        )

    def _limit_unchecked(
        self,
        site_position_world: torch.Tensor,
        anchor_position_world: torch.Tensor,
        site_force_world: torch.Tensor,
        site_torque_world: torch.Tensor | None = None,
    ) -> WrenchLimitResult:
        """Limit without tensor-value validation for the simulator hot path."""

        if site_torque_world is None:
            site_torque_world = torch.zeros_like(site_force_world)
        lever_world = site_position_world - anchor_position_world.unsqueeze(1)
        raw_net_force = site_force_world.sum(dim=1)
        raw_net_torque = (
            torch.cross(lever_world, site_force_world, dim=-1) + site_torque_world
        ).sum(dim=1)
        residual_force = _clamp_vector_norm_unchecked(raw_net_force, self.max_force_n)
        residual_torque = _clamp_vector_norm_unchecked(raw_net_torque, self.max_torque_nm)
        anchor_force = residual_force - raw_net_force
        anchor_torque = residual_torque - raw_net_torque
        return WrenchLimitResult(
            site_force_world=site_force_world,
            site_torque_world=site_torque_world,
            anchor_force_world=anchor_force,
            anchor_torque_world=anchor_torque,
            raw_net_force_world=raw_net_force,
            raw_net_torque_world=raw_net_torque,
            residual_force_world=residual_force,
            residual_torque_world=residual_torque,
        )
