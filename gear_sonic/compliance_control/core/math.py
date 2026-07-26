"""Tensor math for the public compliance condition and bounded virtual forces."""

from __future__ import annotations

import math
from numbers import Real

import torch


def _floating_tensor(value: torch.Tensor | Real, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if device is not None and value.device != device:
            raise ValueError(f"tensor device {value.device} does not match required device {device}")
        if value.is_complex():
            raise TypeError("value must be real")
        if value.dtype == torch.bool:
            raise TypeError("value must not have boolean dtype")
        tensor = value
    elif isinstance(value, Real) and not isinstance(value, bool):
        tensor = torch.as_tensor(value, device=device)
    else:
        raise TypeError("value must be a real scalar or tensor")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor


def _binary_enabled_tensor(
    enabled: torch.Tensor | bool | Real,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Validate a binary global gate and return it as a boolean tensor."""

    if isinstance(enabled, torch.Tensor):
        if device is not None and enabled.device != device:
            raise ValueError(
                f"enabled device {enabled.device} does not match required device {device}"
            )
        if enabled.is_complex():
            raise TypeError("enabled must be boolean or real-valued binary data")
        tensor = enabled
    elif isinstance(enabled, (bool, Real)):
        tensor = torch.as_tensor(enabled, device=device)
    else:
        raise TypeError("enabled must be boolean or finite 0/1 data")

    if tensor.dtype == torch.bool:
        return tensor
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError("enabled must contain only finite 0/1 values")
    if not ((tensor == 0) | (tensor == 1)).all():
        raise ValueError("enabled must contain only finite 0/1 values")
    return tensor.bool()


def _as_batch_column(value: torch.Tensor | Real, *, device: torch.device | None = None) -> torch.Tensor:
    tensor = _floating_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 2 and tensor.shape[-1] == 1:
        return tensor
    raise ValueError("condition values must be scalar, [batch], or [batch, 1]")


def stiffness_from_threshold(
    force_threshold_n: torch.Tensor | Real,
    reference_displacement_m: float = 0.05,
) -> torch.Tensor:
    """Derive conditioning stiffness from force threshold and reference displacement."""

    if isinstance(reference_displacement_m, bool) or not isinstance(
        reference_displacement_m, Real
    ):
        raise TypeError("reference_displacement_m must be a real scalar")
    if not math.isfinite(reference_displacement_m) or reference_displacement_m <= 0.0:
        raise ValueError("reference_displacement_m must be finite and positive")
    threshold = _floating_tensor(force_threshold_n)
    if not torch.isfinite(threshold).all() or (threshold < 0.0).any():
        raise ValueError("force_threshold_n must be finite and non-negative")
    return threshold / reference_displacement_m


def encode_compliance_condition(
    enabled: torch.Tensor | bool | Real,
    force_threshold_n: torch.Tensor | Real,
    reference_displacement_m: float = 0.05,
) -> torch.Tensor:
    """Encode ``[enable, enable * threshold, enable * stiffness]`` per batch item.

    Disabled rows are constructed with ``torch.where`` so they remain exactly
    zero and cannot leak a sampled threshold into the stiff-mode policy input.
    """

    threshold = _as_batch_column(force_threshold_n)
    enabled_tensor = _binary_enabled_tensor(enabled, device=threshold.device)
    if enabled_tensor.ndim == 0:
        enabled_column = enabled_tensor.reshape(1, 1)
    elif enabled_tensor.ndim == 1:
        enabled_column = enabled_tensor.unsqueeze(-1)
    elif enabled_tensor.ndim == 2 and enabled_tensor.shape[-1] == 1:
        enabled_column = enabled_tensor
    else:
        raise ValueError("enabled must be scalar, [batch], or [batch, 1]")
    enabled_column, threshold = torch.broadcast_tensors(enabled_column, threshold)
    if not torch.isfinite(threshold).all() or (threshold < 0.0).any():
        raise ValueError("force_threshold_n must be finite and non-negative")

    flag = enabled_column
    stiffness = stiffness_from_threshold(threshold, reference_displacement_m)
    zeros = torch.zeros_like(threshold)
    return torch.cat(
        (
            flag.to(threshold.dtype),
            torch.where(flag, threshold, zeros),
            torch.where(flag, stiffness, zeros),
        ),
        dim=-1,
    )


def clamp_vector_norm(
    vectors: torch.Tensor,
    max_norm: torch.Tensor | Real,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Clamp vectors along their final axis without changing in-range values."""

    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a tensor")
    if vectors.ndim == 0:
        raise ValueError("vectors must have a final vector dimension")
    if not vectors.is_floating_point() or vectors.is_complex():
        raise TypeError("vectors must have a real floating dtype")
    limit = _floating_tensor(max_norm, device=vectors.device).to(vectors.dtype)
    if not torch.isfinite(limit).all() or (limit < 0.0).any():
        raise ValueError("max_norm must be finite and non-negative")
    return _clamp_vector_norm_unchecked(vectors, limit, eps=eps)


def _clamp_vector_norm_unchecked(
    vectors: torch.Tensor,
    max_norm: torch.Tensor | Real,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Adapter-internal clamp without value reductions; caller owns validation."""

    if isinstance(max_norm, torch.Tensor):
        limit = max_norm.to(device=vectors.device, dtype=vectors.dtype)
    else:
        limit = torch.as_tensor(max_norm, device=vectors.device, dtype=vectors.dtype)
    while limit.ndim < vectors.ndim:
        limit = limit.unsqueeze(-1)

    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scaled = vectors * (limit / norm.clamp_min(eps))
    return torch.where(norm > limit, scaled, vectors)
