"""Portable stateful target damping for CHIP-style compliant goals."""

import math

import torch


def _validate_cartesian_targets(targets: torch.Tensor, *, name: str) -> None:
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not targets.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if targets.ndim < 2 or targets.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [batch, ..., xyz]")
    if not torch.isfinite(targets).all():
        raise ValueError(f"{name} must contain only finite values")


class TargetDamper:
    """Maintain `g_t = alpha * x_eef + (1 - alpha) * g_prev` state.

    State is detached after every update so this utility cannot retain an
    unbounded autograd graph across control steps. The returned target remains
    differentiable with respect to the current end-effector positions.
    """

    def __init__(self, alpha: float) -> None:
        if not isinstance(alpha, int | float) or not math.isfinite(alpha):
            raise TypeError("alpha must be a finite real number")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        self.alpha = float(alpha)
        self._previous_target: torch.Tensor | None = None

    @property
    def initialized(self) -> bool:
        return self._previous_target is not None

    @property
    def previous_target(self) -> torch.Tensor:
        if self._previous_target is None:
            raise RuntimeError("target damper is not initialized; call reset first")
        return self._previous_target.clone()

    def reset(
        self,
        current_eef_positions: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reset all environments, or selected batch rows, to current EEF targets."""

        _validate_cartesian_targets(current_eef_positions, name="current_eef_positions")
        current = current_eef_positions.detach()
        if reset_mask is None:
            self._previous_target = current.clone()
            return self.previous_target
        if self._previous_target is None:
            raise RuntimeError("partial reset requires initialized target-damper state")
        self._validate_state_compatibility(current_eef_positions)
        if not isinstance(reset_mask, torch.Tensor):
            raise TypeError("reset_mask must be a torch.Tensor")
        if reset_mask.dtype is not torch.bool:
            raise TypeError("reset_mask must use torch.bool")
        if reset_mask.device != current_eef_positions.device:
            raise ValueError("reset_mask must use the target tensor device")
        if tuple(reset_mask.shape) != (current_eef_positions.shape[0],):
            raise ValueError("reset_mask must have shape [batch]")

        mask = reset_mask.view(
            current_eef_positions.shape[0],
            *([1] * (current_eef_positions.ndim - 1)),
        )
        self._previous_target = torch.where(mask, current, self._previous_target).clone()
        return self.previous_target

    def update(self, current_eef_positions: torch.Tensor) -> torch.Tensor:
        """Compute one damped target step and store a detached copy as state."""

        _validate_cartesian_targets(current_eef_positions, name="current_eef_positions")
        if self._previous_target is None:
            raise RuntimeError("target damper is not initialized; call reset first")
        self._validate_state_compatibility(current_eef_positions)
        damped_target = (
            self.alpha * current_eef_positions
            + (1.0 - self.alpha) * self._previous_target
        )
        if not torch.isfinite(damped_target).all():
            raise ValueError("damped target became non-finite")
        self._previous_target = damped_target.detach().clone()
        return damped_target

    def _validate_state_compatibility(self, current_eef_positions: torch.Tensor) -> None:
        assert self._previous_target is not None
        if self._previous_target.shape != current_eef_positions.shape:
            raise ValueError("current EEF shape must match target-damper state")
        if self._previous_target.dtype != current_eef_positions.dtype:
            raise TypeError("current EEF dtype must match target-damper state")
        if self._previous_target.device != current_eef_positions.device:
            raise ValueError("current EEF device must match target-damper state")
