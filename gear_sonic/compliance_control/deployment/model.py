"""Export graph for a bounded, hard-gated action residual."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import nn


class ExportableActionResidual(nn.Module):
    """Reconstruct the trained residual without its release policy backbone.

    ``release_action_context`` excludes the public condition.  This graph
    concatenates it exactly once, matching training, sanitizes rejected rows,
    and returns only the bounded delta to be composed by the host runtime.
    """

    def __init__(
        self,
        release_context_width: int,
        condition_width: int,
        action_width: int,
        *,
        hidden_dims: Sequence[int],
        max_abs_delta: float,
    ) -> None:
        super().__init__()
        if type(release_context_width) is not int or release_context_width <= 0:
            raise ValueError("release_context_width must be a positive integer")
        if type(condition_width) is not int or condition_width <= 0:
            raise ValueError("condition_width must be a positive integer")
        if type(action_width) is not int or action_width <= 0:
            raise ValueError("action_width must be a positive integer")
        if len(hidden_dims) != 2 or any(type(width) is not int or width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain exactly two positive integers")
        if (
            isinstance(max_abs_delta, bool)
            or not isinstance(max_abs_delta, (int, float))
            or not math.isfinite(float(max_abs_delta))
            or float(max_abs_delta) <= 0.0
        ):
            raise ValueError("max_abs_delta must be finite and positive")
        self.release_context_width = release_context_width
        self.condition_width = condition_width
        self.action_width = action_width
        self.max_abs_delta = float(max_abs_delta)
        first, second = hidden_dims
        self.residual = nn.Sequential(
            nn.Linear(release_context_width + condition_width, first),
            nn.SiLU(),
            nn.Linear(first, second),
            nn.SiLU(),
            nn.Linear(second, action_width),
        )

    def load_linear_state(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        source_names: Sequence[str],
    ) -> None:
        """Load exactly six ordered linear tensors into the isolated graph."""

        target_names = (
            "residual.0.weight",
            "residual.0.bias",
            "residual.2.weight",
            "residual.2.bias",
            "residual.4.weight",
            "residual.4.bias",
        )
        if len(source_names) != len(target_names) or set(tensors) != set(source_names):
            raise ValueError("residual export requires exactly six declared tensors")
        translated = {
            target_name: tensors[source_name].detach().to(device="cpu").contiguous()
            for target_name, source_name in zip(target_names, source_names, strict=True)
        }
        self.load_state_dict(translated, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(
        self,
        release_action_context: torch.Tensor,
        motion_compliance_condition: torch.Tensor,
    ) -> torch.Tensor:
        if release_action_context.shape[:-1] != motion_compliance_condition.shape[:-1]:
            raise ValueError("release context and condition leading shapes differ")
        if release_action_context.shape[-1] != self.release_context_width:
            raise ValueError("release context width differs from export contract")
        if motion_compliance_condition.shape[-1] != self.condition_width:
            raise ValueError("condition width differs from export contract")
        if (
            release_action_context.dtype != motion_compliance_condition.dtype
            or release_action_context.device != motion_compliance_condition.device
        ):
            raise ValueError("release context and condition dtype/device differ")
        enabled = motion_compliance_condition[..., :1] > 0.5
        residual_context = torch.cat(
            (release_action_context, motion_compliance_condition), dim=-1
        )
        safe_context = torch.where(
            enabled,
            residual_context,
            torch.zeros_like(residual_context),
        )
        delta = torch.tanh(self.residual(safe_context)) * self.max_abs_delta
        return torch.where(enabled, delta, torch.zeros_like(delta))
