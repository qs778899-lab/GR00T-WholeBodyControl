"""Portable schedules for future external-force profiles."""

import math

import torch


def pyramid_phase_weight(
    phase: torch.Tensor,
    *,
    rise_end: float = 0.2,
    fall_start: float = 0.8,
) -> torch.Tensor:
    """Map normalized force phase to a continuous rise/hold/fall weight.

    Values outside `[0, 1]` map to zero. The operation is shape-preserving,
    differentiable away from segment boundaries, and does not mutate `phase`.
    """

    if not isinstance(phase, torch.Tensor):
        raise TypeError("phase must be a torch.Tensor")
    if not phase.is_floating_point():
        raise TypeError("phase must use a floating-point dtype")
    if not math.isfinite(rise_end) or not math.isfinite(fall_start):
        raise ValueError("schedule boundaries must be finite")
    if not 0.0 < rise_end <= fall_start < 1.0:
        raise ValueError("schedule must satisfy 0 < rise_end <= fall_start < 1")

    rise = phase / rise_end
    fall = (1.0 - phase) / (1.0 - fall_start)
    return torch.minimum(rise, fall).clamp(min=0.0, max=1.0)
