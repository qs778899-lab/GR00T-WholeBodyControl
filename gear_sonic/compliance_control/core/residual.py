"""Tracker-agnostic gated residual networks for compliant policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ComplianceResidualLayout:
    """Dimension contract for a variable-site compliance residual."""

    condition_dim: int
    num_sites: int
    cartesian_dim: int
    context_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        for name in (
            "condition_dim",
            "num_sites",
            "cartesian_dim",
            "context_dim",
            "output_dim",
        ):
            _positive_integer(getattr(self, name), name=name)

    @property
    def command_dim(self) -> int:
        """Enable flag, ordered site mask, and per-axis inverse stiffness."""

        return 1 + self.num_sites + self.num_sites * self.cartesian_dim

    @property
    def input_dim(self) -> int:
        """Total MLP input width."""

        return self.condition_dim + self.command_dim + self.context_dim


class ComplianceResidualMLP(nn.Module):
    """Small hard-gated residual with an exactly zero-initialized output head.

    The actor-safe command layout is ``[enable, site_mask, compliance]``.  A
    sample is enabled only when the global flag, at least one site mask, and a
    non-zero compliance value at a selected site all agree.  Disabled rows are
    zeroed before the MLP, then the bounded result is multiplied by that hard
    gate, preserving an exact zero residual in stiff mode.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        num_sites: int,
        cartesian_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        residual_limit: float = 0.25,
    ) -> None:
        super().__init__()
        self.layout = ComplianceResidualLayout(
            condition_dim=condition_dim,
            num_sites=num_sites,
            cartesian_dim=cartesian_dim,
            context_dim=context_dim,
            output_dim=output_dim,
        )
        if isinstance(hidden_dims, str | bytes):
            raise TypeError("hidden_dims must be a sequence of positive integers")
        normalized_hidden_dims = tuple(hidden_dims)
        if not normalized_hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")
        for width in normalized_hidden_dims:
            _positive_integer(width, name="hidden_dims entry")
        if not isinstance(residual_limit, int | float):
            raise TypeError("residual_limit must be a finite positive real number")
        if not math.isfinite(float(residual_limit)) or residual_limit <= 0.0:
            raise ValueError("residual_limit must be a finite positive real number")
        self.residual_limit = float(residual_limit)

        layers: list[nn.Module] = []
        input_width = self.layout.input_dim
        for width in normalized_hidden_dims:
            layers.extend((nn.Linear(input_width, width), nn.SiLU()))
            input_width = width
        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_width, self.layout.output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _validate_input(self, tensor: torch.Tensor, *, name: str, width: int) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.ndim < 2 or tensor.shape[-1] != width:
            raise ValueError(f"{name} final dimension must be {width}")

    def hard_enable(self, actor_command: torch.Tensor) -> torch.Tensor:
        """Return the non-differentiable per-row compliance gate."""

        self._validate_input(
            actor_command,
            name="actor_command",
            width=self.layout.command_dim,
        )
        site_start = 1
        compliance_start = site_start + self.layout.num_sites
        globally_enabled = actor_command[..., 0] > 0.5
        site_mask = actor_command[..., site_start:compliance_start] > 0.5
        compliance = actor_command[..., compliance_start:].reshape(
            *actor_command.shape[:-1],
            self.layout.num_sites,
            self.layout.cartesian_dim,
        )
        compliant_site = (compliance != 0.0).any(dim=-1)
        return globally_enabled & (site_mask & compliant_site).any(dim=-1)

    def forward(
        self,
        condition: torch.Tensor,
        actor_command: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``hard_enable * clamp(raw_residual)`` without fixed sites."""

        self._validate_input(
            condition,
            name="condition",
            width=self.layout.condition_dim,
        )
        self._validate_input(
            actor_command,
            name="actor_command",
            width=self.layout.command_dim,
        )
        self._validate_input(context, name="context", width=self.layout.context_dim)
        if condition.shape[:-1] != actor_command.shape[:-1]:
            raise ValueError("condition and actor_command leading dimensions must match")
        if condition.shape[:-1] != context.shape[:-1]:
            raise ValueError("condition and context leading dimensions must match")
        if condition.dtype != actor_command.dtype or condition.dtype != context.dtype:
            raise TypeError("residual inputs must use one dtype")
        if condition.device != actor_command.device or condition.device != context.device:
            raise ValueError("residual inputs must use one device")

        hard_enable = self.hard_enable(actor_command)
        inputs = torch.cat((condition, actor_command, context), dim=-1)
        safe_inputs = torch.where(hard_enable.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        raw_residual = self.output_layer(self.trunk(safe_inputs))
        bounded_residual = torch.clamp(
            raw_residual,
            min=-self.residual_limit,
            max=self.residual_limit,
        )
        return hard_enable.to(bounded_residual.dtype).unsqueeze(-1) * bounded_residual
