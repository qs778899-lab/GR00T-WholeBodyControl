"""Hard-gated residual composition independent of robot and simulator."""

from __future__ import annotations

import torch


def hard_gate_residual(
    base: torch.Tensor,
    residual: torch.Tensor,
    enabled: torch.Tensor,
) -> torch.Tensor:
    """Select ``base + residual`` only for enabled rows.

    Selection, rather than multiplication by zero, makes every disabled row
    bitwise equal to ``base`` (including signed zero) even when the rejected
    residual contains NaN or Inf.
    """

    if not isinstance(base, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError("base and residual must be tensors")
    if base.shape != residual.shape:
        raise ValueError(
            f"base and residual shapes must match; got {base.shape} and {residual.shape}"
        )
    if base.dtype != residual.dtype or base.device != residual.device:
        raise ValueError("base and residual dtype/device must match")
    if base.is_complex() or not base.is_floating_point():
        raise TypeError("base and residual must use a real floating dtype")
    if not isinstance(enabled, torch.Tensor):
        raise TypeError("enabled must be a tensor")
    if enabled.device != base.device:
        raise ValueError("enabled device must match base")
    if enabled.dtype != torch.bool:
        raise TypeError("enabled must have boolean dtype")
    if enabled.ndim == base.ndim - 1:
        enabled = enabled.unsqueeze(-1)
    if enabled.ndim != base.ndim or enabled.shape[-1] != 1:
        raise ValueError("enabled must match base leading dimensions with an optional final 1")
    try:
        enabled = torch.broadcast_to(enabled, (*base.shape[:-1], 1))
    except RuntimeError as error:
        raise ValueError("enabled leading dimensions are not broadcastable to base") from error
    return torch.where(enabled, base + residual, base)
