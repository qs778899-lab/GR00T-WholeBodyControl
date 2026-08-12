"""Pure deterministic force schedule for matched CHIP review trials."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .roles import REVIEW_SITE_NAMES, ReviewRole


def _readonly(name: str, value: object, shape: tuple[int, ...], kind: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if kind == "float" and array.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating dtype")
    if kind == "bool" and array.dtype.kind != "b":
        raise TypeError(f"{name} must use a boolean dtype")
    if kind == "float" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ProtocolSample:
    """One pre-transition compliance command with a pinned world-frame wrench."""

    frame_index: int
    frame_count: int
    compliance_enabled: bool
    residual_enabled: bool
    active_site_mask: np.ndarray
    force_on_robot_world_n: np.ndarray
    compliance_m_per_n: np.ndarray
    profile_weight: float

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or type(self.frame_count) is not int:
            raise TypeError("frame_index and frame_count must be integers")
        if self.frame_count < 8 or not 0 <= self.frame_index < self.frame_count:
            raise ValueError("frame index/count do not describe a valid review clip")
        if type(self.compliance_enabled) is not bool or type(self.residual_enabled) is not bool:
            raise TypeError("mode gates must be bool")
        if self.residual_enabled and not self.compliance_enabled:
            raise ValueError("residual_enabled requires compliance_enabled")
        if not math.isfinite(self.profile_weight) or not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("profile_weight must be finite within [0, 1]")
        sites = len(REVIEW_SITE_NAMES)
        object.__setattr__(
            self,
            "active_site_mask",
            _readonly("active_site_mask", self.active_site_mask, (sites,), "bool"),
        )
        for field_name in ("force_on_robot_world_n", "compliance_m_per_n"):
            object.__setattr__(
                self,
                field_name,
                _readonly(field_name, getattr(self, field_name), (sites, 3), "float"),
            )
        if np.any(self.active_site_mask & ~self.compliance_enabled):
            raise ValueError("active sites require compliance_enabled")
        inactive = ~self.active_site_mask
        if np.any(self.force_on_robot_world_n[inactive] != 0.0):
            raise ValueError("inactive sites must have exact-zero force")
        if np.any(self.compliance_m_per_n[inactive] != 0.0):
            raise ValueError("inactive sites must have exact-zero compliance")


@dataclass(frozen=True, slots=True)
class DeterministicForceProtocol:
    """One smooth matched schedule with fixed wrist-relative force directions."""

    fps: int = 50
    peak_force_n: float = 5.0
    compliance_m_per_n: float = 0.02
    active_start_fraction: float = 0.20
    active_stop_fraction: float = 0.80
    ramp_fraction_of_active: float = 0.10

    def __post_init__(self) -> None:
        if type(self.fps) is not int or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        for field_name in ("peak_force_n", "compliance_m_per_n"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not (
            0.0 < self.active_start_fraction < self.active_stop_fraction < 1.0
        ):
            raise ValueError("active fractions must satisfy 0 < start < stop < 1")
        if not 0.0 < self.ramp_fraction_of_active < 0.5:
            raise ValueError("ramp_fraction_of_active must be within (0, 0.5)")

    def active_bounds(self, frame_count: int) -> tuple[int, int, int]:
        """Return inclusive-start/exclusive-stop and ramp-frame counts."""

        if type(frame_count) is not int or frame_count < 8:
            raise ValueError("frame_count must be an integer of at least eight")
        start = max(1, math.ceil(frame_count * self.active_start_fraction))
        stop = min(frame_count - 1, math.floor(frame_count * self.active_stop_fraction))
        if stop - start < 4:
            raise ValueError("clip is too short for the deterministic force schedule")
        ramp = max(1, math.floor((stop - start) * self.ramp_fraction_of_active))
        return start, stop, ramp

    def _weight(self, frame_index: int, frame_count: int) -> float:
        start, stop, ramp = self.active_bounds(frame_count)
        if frame_index < start or frame_index >= stop:
            return 0.0
        relative = frame_index - start
        remaining = stop - frame_index
        if relative < ramp:
            phase = (relative + 1) / ramp
            return float(math.sin(0.5 * math.pi * phase) ** 2)
        if remaining <= ramp:
            phase = remaining / ramp
            return float(math.sin(0.5 * math.pi * phase) ** 2)
        return 1.0

    def sample(self, role: ReviewRole, frame_index: int, frame_count: int) -> ProtocolSample:
        """Build one role-specific sample; matched A/B roles are byte-identical."""

        if not isinstance(role, ReviewRole):
            raise TypeError("role must be a ReviewRole")
        if type(frame_index) is not int or not 0 <= frame_index < frame_count:
            raise ValueError("frame_index must lie within the clip")
        weight = self._weight(frame_index, frame_count) if role.external_force_enabled else 0.0
        active = np.zeros(len(REVIEW_SITE_NAMES), dtype=np.bool_)
        force = np.zeros((len(REVIEW_SITE_NAMES), 3), dtype=np.float64)
        compliance = np.zeros_like(force)
        if weight > 0.0:
            for site_name in role.active_site_names:
                index = REVIEW_SITE_NAMES.index(site_name)
                active[index] = True
                lateral_sign = 1.0 if index == 0 else -1.0
                force[index, 1] = lateral_sign * self.peak_force_n * weight
                compliance[index, :] = self.compliance_m_per_n
        return ProtocolSample(
            frame_index=frame_index,
            frame_count=frame_count,
            compliance_enabled=role.compliance_enabled,
            residual_enabled=role.residual_enabled,
            active_site_mask=active,
            force_on_robot_world_n=force,
            compliance_m_per_n=compliance,
            profile_weight=weight,
        )


def chip_selected_target(original_target_m: np.ndarray, sample: ProtocolSample) -> np.ndarray:
    """Apply the signed undamped CHIP target relation in the common frame."""

    if not isinstance(sample, ProtocolSample):
        raise TypeError("sample must be a ProtocolSample")
    original = np.asarray(original_target_m)
    expected_shape = (len(REVIEW_SITE_NAMES), 3)
    if original.shape != expected_shape or original.dtype.kind != "f":
        raise ValueError(f"original_target_m must have shape {expected_shape} and float dtype")
    if not np.isfinite(original).all():
        raise ValueError("original_target_m must be finite")
    selected = original - sample.compliance_m_per_n * sample.force_on_robot_world_n
    return np.where(sample.active_site_mask[:, None], selected, original)
