"""Tracker-neutral schemas for aligned compliance evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np


class TrialMode(str, Enum):
    """Protocol role of one trace in a paired evaluation suite."""

    BASELINE = "baseline"
    OFF = "off"
    NO_CONTACT = "no_contact"
    SINGLE_SITE = "single_site"
    MULTI_SITE = "multi_site"


def _as_name_tuple(name: str, values: object, *, expected_length: int | None = None) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of names, not a scalar string")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of names") from exc
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if name in {"site_ids", "point_ids"} and len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _readonly_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    kind: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if kind == "float":
        if array.dtype.kind not in "fc" or array.dtype.kind == "c":
            raise TypeError(f"{name} must have a real floating dtype")
    elif kind == "integer":
        if array.dtype.kind not in "iu" or array.dtype.kind == "b":
            raise TypeError(f"{name} must have an integer dtype")
    elif kind == "boolean":
        if array.dtype.kind != "b":
            raise TypeError(f"{name} must have a boolean dtype")
    else:  # pragma: no cover - internal programming error
        raise AssertionError(f"unsupported array kind: {kind}")
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class EvaluationTrace:
    """One standardized trace with caller-owned site and tracking-point layouts.

    Each row is identified by motion, sequence, seed, frame, and timestamp.
    Physical arrays deliberately retain non-finite values so the evaluator can
    report them instead of silently dropping a failed trial.  Alignment keys
    themselves must always be finite and strictly ordered within each sequence.

    Quaternion arrays use caller-declared ``xyzw`` order.  ``reset_mask`` marks
    post-reset snapshots, which lets the evaluator detect a wrench that was not
    cleared at reset.
    """

    trial_name: str
    motion_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    seed_ids: np.ndarray
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    site_ids: tuple[str, ...]
    point_ids: tuple[str, ...]
    original_site_positions_m: np.ndarray
    selected_site_positions_m: np.ndarray
    measured_site_positions_m: np.ndarray
    original_site_orientations_xyzw: np.ndarray
    measured_site_orientations_xyzw: np.ndarray
    reference_points_global_m: np.ndarray
    measured_points_global_m: np.ndarray
    reference_points_local_m: np.ndarray
    measured_points_local_m: np.ndarray
    force_on_robot_n: np.ndarray
    compliance_enabled: np.ndarray
    active_site_mask: np.ndarray
    terminal_mask: np.ndarray
    success_mask: np.ndarray
    fall_mask: np.ndarray
    reset_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.trial_name, str) or not self.trial_name:
            raise ValueError("trial_name must be a non-empty string")

        motion_ids = _as_name_tuple("motion_ids", self.motion_ids)
        row_count = len(motion_ids)
        sequence_ids = _as_name_tuple(
            "sequence_ids", self.sequence_ids, expected_length=row_count
        )
        site_ids = _as_name_tuple("site_ids", self.site_ids)
        point_ids = _as_name_tuple("point_ids", self.point_ids)
        object.__setattr__(self, "motion_ids", motion_ids)
        object.__setattr__(self, "sequence_ids", sequence_ids)
        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "point_ids", point_ids)

        vector_shape = (row_count, len(site_ids), 3)
        quaternion_shape = (row_count, len(site_ids), 4)
        point_shape = (row_count, len(point_ids), 3)
        field_specs = {
            "seed_ids": ((row_count,), "integer"),
            "frame_indices": ((row_count,), "integer"),
            "timestamps_s": ((row_count,), "float"),
            "original_site_positions_m": (vector_shape, "float"),
            "selected_site_positions_m": (vector_shape, "float"),
            "measured_site_positions_m": (vector_shape, "float"),
            "original_site_orientations_xyzw": (quaternion_shape, "float"),
            "measured_site_orientations_xyzw": (quaternion_shape, "float"),
            "reference_points_global_m": (point_shape, "float"),
            "measured_points_global_m": (point_shape, "float"),
            "reference_points_local_m": (point_shape, "float"),
            "measured_points_local_m": (point_shape, "float"),
            "force_on_robot_n": (vector_shape, "float"),
            "compliance_enabled": ((row_count,), "boolean"),
            "active_site_mask": ((row_count, len(site_ids)), "boolean"),
            "terminal_mask": ((row_count,), "boolean"),
            "success_mask": ((row_count,), "boolean"),
            "fall_mask": ((row_count,), "boolean"),
            "reset_mask": ((row_count,), "boolean"),
        }
        for field_name, (shape, kind) in field_specs.items():
            object.__setattr__(
                self,
                field_name,
                _readonly_array(field_name, getattr(self, field_name), shape=shape, kind=kind),
            )

        if not np.isfinite(self.timestamps_s).all():
            raise ValueError("timestamps_s must be finite")
        if np.any(self.frame_indices < 0):
            raise ValueError("frame_indices must be non-negative")
        if np.any(self.success_mask & ~self.terminal_mask):
            raise ValueError("success_mask must be a subset of terminal_mask")
        if np.any(self.success_mask & self.fall_mask):
            raise ValueError("success_mask and fall_mask must be disjoint")
        if np.any(self.fall_mask & ~self.terminal_mask):
            raise ValueError("fall_mask must be a subset of terminal_mask")
        if np.any(self.active_site_mask & ~self.compliance_enabled[:, None]):
            raise ValueError("active sites require compliance_enabled")

        seen_rows: set[tuple[str, str, int, int]] = set()
        prior_by_sequence: dict[tuple[str, str, int], tuple[int, float]] = {}
        for row in range(row_count):
            seed = int(self.seed_ids[row])
            frame = int(self.frame_indices[row])
            identity = (motion_ids[row], sequence_ids[row], seed, frame)
            if identity in seen_rows:
                raise ValueError("alignment rows must have unique motion/sequence/seed/frame keys")
            seen_rows.add(identity)
            sequence_key = identity[:3]
            previous = prior_by_sequence.get(sequence_key)
            timestamp = float(self.timestamps_s[row])
            if previous is not None:
                if frame <= previous[0]:
                    raise ValueError("frame_indices must increase within each sequence")
                if timestamp <= previous[1]:
                    raise ValueError("timestamps_s must increase within each sequence")
            prior_by_sequence[sequence_key] = (frame, timestamp)

        rows_by_sequence: dict[tuple[str, str, int], list[int]] = {}
        for row in range(row_count):
            key = (motion_ids[row], sequence_ids[row], int(self.seed_ids[row]))
            rows_by_sequence.setdefault(key, []).append(row)
        for rows in rows_by_sequence.values():
            terminal_rows = [row for row in rows if self.terminal_mask[row]]
            if terminal_rows != [rows[-1]]:
                raise ValueError("each sequence must terminate exactly once on its final row")
            reset_rows = [row for row in rows if self.reset_mask[row]]
            if reset_rows != [rows[0]]:
                raise ValueError("each sequence must have exactly one reset snapshot on its first row")


@dataclass(frozen=True)
class TrialSpec:
    """Expected activation protocol for one named trace."""

    name: str
    mode: TrialMode
    expected_active_site_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("trial spec name must be a non-empty string")
        try:
            mode = TrialMode(self.mode)
        except ValueError as exc:
            raise ValueError(f"unsupported trial mode: {self.mode}") from exc
        object.__setattr__(self, "mode", mode)
        if isinstance(self.expected_active_site_ids, (str, bytes)):
            raise TypeError("expected_active_site_ids must be a sequence, not a scalar string")
        active = tuple(self.expected_active_site_ids)
        if any(not isinstance(value, str) or not value for value in active):
            raise ValueError("expected_active_site_ids must contain non-empty strings")
        if len(set(active)) != len(active):
            raise ValueError("expected_active_site_ids must not contain duplicates")
        object.__setattr__(self, "expected_active_site_ids", active)

        if mode in {TrialMode.BASELINE, TrialMode.OFF, TrialMode.NO_CONTACT} and active:
            raise ValueError(f"{mode.value} does not accept expected active sites")
        if mode is TrialMode.SINGLE_SITE and len(active) != 1:
            raise ValueError("single_site requires exactly one expected active site")
        if mode is TrialMode.MULTI_SITE and len(active) < 2:
            raise ValueError("multi_site requires at least two expected active sites")


@dataclass(frozen=True)
class RegressionCriteria:
    """Caller-selected acceptance limits for paired stiff/off validation."""

    endpoint_site_ids: tuple[str, ...]
    max_success_rate_drop: float = 0.01
    local_mpjpe_absolute_regression_m: float = 0.003
    local_mpjpe_relative_regression: float = 0.10
    endpoint_rmse_regression_m: float = 0.005
    no_contact_endpoint_delta_m: float = 0.005
    reset_wrench_tolerance_n: float = 1.0e-6
    inactive_force_tolerance_n: float = 1.0e-6
    inactive_yield_tolerance_m: float = 1.0e-9
    minimum_active_force_peak_n: float = 1.0e-6
    minimum_active_yield_peak_m: float = 1.0e-9
    minimum_active_measured_yield_peak_m: float = 1.0e-6
    minimum_active_measured_yield_along_force_peak_m: float = 1.0e-6
    inactive_cross_coupling_rmse_m: float = 0.005
    inactive_cross_coupling_p95_m: float = 0.005

    def __post_init__(self) -> None:
        if isinstance(self.endpoint_site_ids, (str, bytes)):
            raise TypeError("endpoint_site_ids must be a sequence, not a scalar string")
        endpoint_sites = tuple(self.endpoint_site_ids)
        if not endpoint_sites:
            raise ValueError("endpoint_site_ids must not be empty")
        if any(not isinstance(value, str) or not value for value in endpoint_sites):
            raise ValueError("endpoint_site_ids must contain non-empty strings")
        if len(set(endpoint_sites)) != len(endpoint_sites):
            raise ValueError("endpoint_site_ids must not contain duplicates")
        object.__setattr__(self, "endpoint_site_ids", endpoint_sites)
        for field_name in (
            "max_success_rate_drop",
            "local_mpjpe_absolute_regression_m",
            "local_mpjpe_relative_regression",
            "endpoint_rmse_regression_m",
            "no_contact_endpoint_delta_m",
            "reset_wrench_tolerance_n",
            "inactive_force_tolerance_n",
            "inactive_yield_tolerance_m",
            "minimum_active_force_peak_n",
            "minimum_active_yield_peak_m",
            "minimum_active_measured_yield_peak_m",
            "minimum_active_measured_yield_along_force_peak_m",
            "inactive_cross_coupling_rmse_m",
            "inactive_cross_coupling_p95_m",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
