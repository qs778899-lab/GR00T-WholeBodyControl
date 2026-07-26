"""Configuration schemas for tracker-independent compliance control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


def _validate_finite_pair(name: str, values: tuple[float, float], *, positive: bool) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = (float(value) for value in values)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} must contain finite values")
    if positive and low <= 0.0:
        raise ValueError(f"{name} lower bound must be positive")
    if high < low:
        raise ValueError(f"{name} upper bound must be greater than or equal to its lower bound")


@dataclass(frozen=True)
class ComplianceSpec:
    """Global public contract for policy-level compliant tracking.

    Physical site names, skeleton layouts, and simulator handles intentionally
    do not belong in this schema.  Adapters supply those mappings.
    """

    condition_size: ClassVar[int] = 3
    reference_frame_contract: ClassVar[str] = "adapter_supplied_common_cartesian_frame"
    force_sign_convention: ClassVar[str] = "force_on_robot"

    force_threshold_range_n: tuple[float, float] = (10.0, 20.0)
    reference_displacement_m: float = 0.05
    tracking_gain_n_per_m: float = 100.0
    tracking_force_cap_n: float = 5.0
    max_net_force_n: float = 20.0
    max_net_torque_nm: float = 10.0

    def __post_init__(self) -> None:
        _validate_finite_pair(
            "force_threshold_range_n", self.force_threshold_range_n, positive=True
        )
        if not math.isfinite(self.reference_displacement_m) or self.reference_displacement_m <= 0.0:
            raise ValueError("reference_displacement_m must be finite and positive")
        if not math.isfinite(self.tracking_gain_n_per_m) or self.tracking_gain_n_per_m < 0.0:
            raise ValueError("tracking_gain_n_per_m must be finite and non-negative")
        if not math.isfinite(self.tracking_force_cap_n) or self.tracking_force_cap_n < 0.0:
            raise ValueError("tracking_force_cap_n must be finite and non-negative")
        if not math.isfinite(self.max_net_force_n) or self.max_net_force_n <= 0.0:
            raise ValueError("max_net_force_n must be finite and positive")
        if not math.isfinite(self.max_net_torque_nm) or self.max_net_torque_nm <= 0.0:
            raise ValueError("max_net_torque_nm must be finite and positive")

    @property
    def stiffness_range_n_per_m(self) -> tuple[float, float]:
        """Return the threshold-derived policy conditioning stiffness range."""

        low, high = self.force_threshold_range_n
        return low / self.reference_displacement_m, high / self.reference_displacement_m


@dataclass(frozen=True)
class ForceEventScheduleSpec:
    """Sampling and temporal-envelope settings for virtual force events."""

    enable_probability: float = 0.75
    site_activation_probability: float = 0.5
    duration_steps_range: tuple[int, int] = (100, 800)
    ramp_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name, value in (
            ("enable_probability", self.enable_probability),
            ("site_activation_probability", self.site_activation_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if len(self.duration_steps_range) != 2:
            raise ValueError("duration_steps_range must contain exactly two values")
        low, high = self.duration_steps_range
        if low <= 0 or high < low:
            raise ValueError("duration_steps_range must be positive and ordered")
        if not math.isfinite(self.ramp_fraction) or not 0.0 <= self.ramp_fraction <= 0.5:
            raise ValueError("ramp_fraction must be finite and within [0, 0.5]")
