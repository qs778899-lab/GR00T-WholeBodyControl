"""Tracker-independent temporal scheduling for virtual interaction events."""

from __future__ import annotations

import torch

from .math import _binary_enabled_tensor


def event_envelope(
    step: torch.Tensor | int,
    start_step: torch.Tensor | int,
    duration_steps: torch.Tensor | int,
    *,
    ramp_fraction: float = 0.25,
) -> torch.Tensor:
    """Return a smooth rise/hold/fall event envelope in ``[0, 1]``."""

    if not 0.0 <= ramp_fraction <= 0.5:
        raise ValueError("ramp_fraction must be within [0, 0.5]")
    step_tensor = torch.as_tensor(step)
    if not step_tensor.is_floating_point():
        step_tensor = step_tensor.to(torch.float32)
    start_tensor = torch.as_tensor(start_step, device=step_tensor.device, dtype=step_tensor.dtype)
    duration_tensor = torch.as_tensor(
        duration_steps, device=step_tensor.device, dtype=step_tensor.dtype
    )
    if (duration_tensor <= 0.0).any():
        raise ValueError("duration_steps must be positive")

    phase = (step_tensor - start_tensor) / duration_tensor
    active = (phase >= 0.0) & (phase < 1.0)
    if ramp_fraction == 0.0:
        return active.to(step_tensor.dtype)

    rise = (phase / ramp_fraction).clamp(0.0, 1.0)
    fall = ((1.0 - phase) / ramp_fraction).clamp(0.0, 1.0)
    linear = torch.minimum(rise, fall)
    smooth = linear.square() * (3.0 - 2.0 * linear)
    return torch.where(active, smooth, torch.zeros_like(smooth))


def sample_site_mask(
    enabled: torch.Tensor,
    num_sites: int,
    *,
    site_activation_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample independent interaction sites, with at least one for enabled rows."""

    if not isinstance(enabled, torch.Tensor):
        raise TypeError("enabled must be a tensor with shape [batch]")
    enabled_mask = _binary_enabled_tensor(enabled, device=enabled.device)
    if enabled_mask.ndim != 1:
        raise ValueError("enabled must have shape [batch]")
    if num_sites <= 0:
        raise ValueError("num_sites must be positive")
    if not 0.0 <= site_activation_probability <= 1.0:
        raise ValueError("site_activation_probability must be within [0, 1]")

    random_scores = torch.rand(
        (enabled.shape[0], num_sites), device=enabled.device, generator=generator
    )
    sampled = random_scores < site_activation_probability
    sampled &= enabled_mask.unsqueeze(-1)

    missing = enabled_mask & ~sampled.any(dim=-1)
    fallback_site = random_scores.argmax(dim=-1, keepdim=True)
    fallback_mask = torch.arange(num_sites, device=enabled.device).reshape(1, -1)
    fallback_mask = fallback_mask == fallback_site
    return torch.where(missing.unsqueeze(-1), fallback_mask, sampled)
