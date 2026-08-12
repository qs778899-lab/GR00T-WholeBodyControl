"""Tracker-neutral acceptance for baseline, no-contact, and matched-force trials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .alignment import TraceAlignmentError, alignment_digest, assert_strict_alignment
from .metrics import evaluate_trace
from .schema import EvaluationTrace, RegressionCriteria


def _names(label: str, values: object, *, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc
    if len(result) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} value(s)")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class MatchedInteractionSpec:
    """One caller-owned matched-force reference/candidate comparison."""

    name: str
    reference_trial: str
    candidate_trial: str
    active_site_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "reference_trial", "candidate_trial"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.reference_trial == self.candidate_trial:
            raise ValueError("matched interaction trials must be distinct")
        object.__setattr__(
            self,
            "active_site_ids",
            _names("active_site_ids", self.active_site_ids),
        )


@dataclass(frozen=True, slots=True)
class ReviewSuiteSpec:
    """Caller-owned role map for one aligned motion/seed review suite."""

    baseline_trial: str
    hard_off_trial: str
    no_contact_trial: str
    interactions: tuple[MatchedInteractionSpec, ...]

    def __post_init__(self) -> None:
        for field_name in ("baseline_trial", "hard_off_trial", "no_contact_trial"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        base_names = (self.baseline_trial, self.hard_off_trial, self.no_contact_trial)
        if len(set(base_names)) != len(base_names):
            raise ValueError("baseline, hard-off, and no-contact trials must be distinct")
        if isinstance(self.interactions, MatchedInteractionSpec):
            raise TypeError("interactions must be a sequence")
        interactions = tuple(self.interactions)
        if not interactions or any(
            not isinstance(value, MatchedInteractionSpec) for value in interactions
        ):
            raise ValueError("interactions must contain matched interaction specs")
        object.__setattr__(self, "interactions", interactions)
        names = [item.name for item in interactions]
        if len(set(names)) != len(names):
            raise ValueError("interaction names must not contain duplicates")
        trial_names = [name for item in interactions for name in (
            item.reference_trial,
            item.candidate_trial,
        )]
        if len(set(trial_names)) != len(trial_names):
            raise ValueError("each interaction trial may be used only once")
        if set(base_names).intersection(trial_names):
            raise ValueError("interaction trials must not reuse base trial names")

    @property
    def trial_names(self) -> tuple[str, ...]:
        return (
            self.baseline_trial,
            self.hard_off_trial,
            self.no_contact_trial,
            *(name for pair in self.interactions for name in (
                pair.reference_trial,
                pair.candidate_trial,
            )),
        )


def _exact_array(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return (
        reference.dtype == candidate.dtype
        and reference.shape == candidate.shape
        and reference.tobytes(order="C") == candidate.tobytes(order="C")
    )


def assert_matched_stimulus(
    reference: EvaluationTrace,
    candidate: EvaluationTrace,
) -> None:
    """Require identical aligned samples, selected targets, and force schedule."""

    assert_strict_alignment(reference, candidate)
    for field_name in (
        "selected_site_positions_m",
        "force_on_robot_n",
        "compliance_m_per_n",
        "compliance_enabled",
        "active_site_mask",
        "reset_mask",
    ):
        if not _exact_array(getattr(reference, field_name), getattr(candidate, field_name)):
            raise TraceAlignmentError(f"unmatched stimulus field {field_name}")


def _norm(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values64).all(axis=-1)
    result = np.full(values64.shape[:-1], np.nan, dtype=np.float64)
    if np.any(finite):
        rows = values64[finite]
        scale = np.max(np.abs(rows), axis=-1)
        nonzero = scale > 0.0
        norms = np.zeros_like(scale)
        normalized = rows[nonzero] / scale[nonzero, None]
        norms[nonzero] = scale[nonzero] * np.sqrt(
            np.sum(np.square(normalized), axis=-1)
        )
        result[finite] = norms
    return result


def _quaternion_angle(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    reference64 = np.asarray(reference, dtype=np.float64)
    measured64 = np.asarray(measured, dtype=np.float64)
    reference_norm = _norm(reference64)
    measured_norm = _norm(measured64)
    valid = (
        np.isfinite(reference64).all(axis=-1)
        & np.isfinite(measured64).all(axis=-1)
        & (reference_norm > 0.0)
        & (measured_norm > 0.0)
    )
    result = np.full(reference_norm.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        reference_unit = reference64[valid] / reference_norm[valid, None]
        measured_unit = measured64[valid] / measured_norm[valid, None]
        dot = np.sum(reference_unit * measured_unit, axis=-1)
        result[valid] = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    return result


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flattened[np.isfinite(flattened)]
    if finite.size == 0:
        return {
            "count": int(flattened.size),
            "finite_count": 0,
            "mean": None,
            "rmse": None,
            "p95": None,
            "peak": None,
        }
    scale = float(np.max(np.abs(finite)))
    if scale == 0.0:
        mean = rmse = p95 = peak = 0.0
    else:
        normalized = finite / scale
        mean = float(scale * np.mean(normalized))
        rmse = float(scale * np.sqrt(np.mean(np.square(normalized))))
        p95 = float(scale * np.percentile(normalized, 95.0))
        peak = float(np.max(finite))
    return {
        "count": int(flattened.size),
        "finite_count": int(finite.size),
        "mean": mean,
        "rmse": rmse,
        "p95": p95,
        "peak": peak,
    }


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    value: Any,
    limit: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "value": value, "limit": limit})


def _finite_number(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)


def _regression_limit(reference: float, absolute: float, relative: float) -> float:
    return max(absolute, relative * reference)


def _difference(candidate: object, reference: object) -> float | None:
    if not _finite_number(candidate) or not _finite_number(reference):
        return None
    return float(candidate) - float(reference)


def _trace_semantics(
    trace: EvaluationTrace,
    *,
    compliance_enabled: bool,
    residual_enabled: bool,
    active_sites: Sequence[str],
) -> None:
    expected_sites = set(active_sites)
    unknown = expected_sites.difference(trace.site_ids)
    if unknown:
        raise ValueError(f"expected active sites are missing from trace: {sorted(unknown)}")
    if not np.all(trace.compliance_enabled == compliance_enabled):
        raise ValueError(
            f"trial {trace.trial_name!r} has an unexpected compliance-enabled gate"
        )
    if not np.all(trace.residual_enabled == residual_enabled):
        raise ValueError(
            f"trial {trace.trial_name!r} has an unexpected residual-enabled gate"
        )
    expected_mask = np.asarray(
        [site_id in expected_sites for site_id in trace.site_ids],
        dtype=np.bool_,
    )
    observed_any = np.any(trace.active_site_mask, axis=0)
    if not np.array_equal(observed_any, expected_mask):
        raise ValueError(
            f"trial {trace.trial_name!r} active-site layout does not match its spec"
        )
    if expected_sites:
        for index, expected in enumerate(expected_mask):
            if expected and not np.any(trace.active_site_mask[:, index] & ~trace.reset_mask):
                raise ValueError(
                    f"trial {trace.trial_name!r} never activates site {trace.site_ids[index]!r}"
                )
    elif np.any(trace.active_site_mask):
        raise ValueError(f"trial {trace.trial_name!r} must not activate a site")


def _all_finite(trace: EvaluationTrace) -> bool:
    for field_name in (
        "timestamps_s",
        "original_site_positions_m",
        "selected_site_positions_m",
        "measured_site_positions_m",
        "original_site_orientations_xyzw",
        "measured_site_orientations_xyzw",
        "reference_points_global_m",
        "measured_points_global_m",
        "reference_points_local_m",
        "measured_points_local_m",
        "force_on_robot_n",
        "compliance_m_per_n",
        "policy_actions",
    ):
        if not np.isfinite(getattr(trace, field_name)).all():
            return False
    return True


def _lifecycle_checks(
    checks: list[dict[str, Any]],
    name: str,
    trace: EvaluationTrace,
    criteria: RegressionCriteria,
) -> None:
    terminal_count = int(np.count_nonzero(trace.terminal_mask))
    success_count = int(np.count_nonzero(trace.success_mask))
    fall_count = int(np.count_nonzero(trace.fall_mask))
    _check(
        checks,
        f"full_success:{name}",
        terminal_count > 0 and success_count == terminal_count,
        value={"success": success_count, "terminal": terminal_count},
        limit="success == terminal > 0",
    )
    _check(
        checks,
        f"zero_falls:{name}",
        fall_count == 0,
        value=fall_count,
        limit=0,
    )
    _check(
        checks,
        f"finite:{name}",
        _all_finite(trace),
        value=_all_finite(trace),
        limit=True,
    )
    reset_force_peak = _summary(_norm(trace.force_on_robot_n[trace.reset_mask]))["peak"]
    _check(
        checks,
        f"reset_force_peak_n:{name}",
        _finite_number(reset_force_peak)
        and float(reset_force_peak) <= criteria.reset_wrench_tolerance_n,
        value=reset_force_peak,
        limit=criteria.reset_wrench_tolerance_n,
    )


def _tracking_errors(trace: EvaluationTrace) -> dict[str, np.ndarray]:
    return {
        "original_endpoint": _norm(
            trace.measured_site_positions_m - trace.original_site_positions_m
        ),
        "selected_endpoint": _norm(
            trace.measured_site_positions_m - trace.selected_site_positions_m
        ),
        "orientation": _quaternion_angle(
            trace.original_site_orientations_xyzw,
            trace.measured_site_orientations_xyzw,
        ),
        "global": _norm(
            trace.measured_points_global_m - trace.reference_points_global_m
        ),
        "local": _norm(
            trace.measured_points_local_m - trace.reference_points_local_m
        ),
    }


def _base_tracking_checks(
    checks: list[dict[str, Any]],
    label: str,
    reference: EvaluationTrace,
    candidate: EvaluationTrace,
    criteria: RegressionCriteria,
    endpoint_limit_m: float,
    absolute_endpoint_delta: bool = False,
) -> dict[str, Any]:
    assert_strict_alignment(reference, candidate)
    reference_errors = _tracking_errors(reference)
    candidate_errors = _tracking_errors(candidate)
    report: dict[str, Any] = {"sites": {}}
    for site_id in criteria.endpoint_site_ids:
        if site_id not in reference.site_ids:
            raise ValueError(f"endpoint site {site_id!r} must exist in every trace")
        index = reference.site_ids.index(site_id)
        ref_summary = _summary(reference_errors["original_endpoint"][:, index])
        cand_summary = _summary(candidate_errors["original_endpoint"][:, index])
        rmse_regression = _difference(cand_summary["rmse"], ref_summary["rmse"])
        endpoint_value = (
            abs(rmse_regression)
            if absolute_endpoint_delta and rmse_regression is not None
            else rmse_regression
        )
        _check(
            checks,
            f"endpoint_rmse_regression_m:{label}:{site_id}",
            _finite_number(endpoint_value) and float(endpoint_value) <= endpoint_limit_m,
            value=rmse_regression,
            limit={"absolute": endpoint_limit_m}
            if absolute_endpoint_delta
            else endpoint_limit_m,
        )
        report["sites"][site_id] = {
            "reference": ref_summary,
            "candidate": cand_summary,
            "rmse_regression_m": rmse_regression,
        }
    for kind, absolute, relative in (
        (
            "local",
            criteria.local_mpjpe_absolute_regression_m,
            criteria.local_mpjpe_relative_regression,
        ),
        (
            "global",
            criteria.active_invariant_global_mpjpe_absolute_regression_m,
            criteria.active_invariant_global_mpjpe_relative_regression,
        ),
    ):
        reference_mean = _summary(reference_errors[kind])["mean"]
        candidate_mean = _summary(candidate_errors[kind])["mean"]
        regression = _difference(candidate_mean, reference_mean)
        limit = (
            _regression_limit(float(reference_mean), absolute, relative)
            if _finite_number(reference_mean)
            else None
        )
        _check(
            checks,
            f"{kind}_mpjpe_regression_m:{label}",
            _finite_number(regression)
            and _finite_number(limit)
            and float(regression) <= float(limit),
            value=regression,
            limit=limit,
        )
        report[f"{kind}_mpjpe"] = {
            "reference_mean_m": reference_mean,
            "candidate_mean_m": candidate_mean,
            "regression_m": regression,
            "limit_m": limit,
        }
    return report


def _interaction_report(
    checks: list[dict[str, Any]],
    spec: MatchedInteractionSpec,
    reference: EvaluationTrace,
    candidate: EvaluationTrace,
    criteria: RegressionCriteria,
) -> dict[str, Any]:
    assert_matched_stimulus(reference, candidate)
    _trace_semantics(
        reference,
        compliance_enabled=True,
        residual_enabled=False,
        active_sites=spec.active_site_ids,
    )
    _trace_semantics(
        candidate,
        compliance_enabled=True,
        residual_enabled=True,
        active_sites=spec.active_site_ids,
    )
    reference_errors = _tracking_errors(reference)
    candidate_errors = _tracking_errors(candidate)
    endpoint_point_by_site = dict(
        zip(criteria.endpoint_site_ids, criteria.endpoint_tracking_point_ids, strict=True)
    )
    missing_points = set(endpoint_point_by_site.values()).difference(reference.point_ids)
    if missing_points:
        raise ValueError(f"endpoint tracking points must exist: {sorted(missing_points)}")
    missing_sites = set(spec.active_site_ids).difference(endpoint_point_by_site)
    if missing_sites:
        raise ValueError(
            "each active site must have an explicit endpoint-to-tracking-point mapping"
        )

    interaction_rows = np.any(candidate.active_site_mask, axis=1) & ~candidate.reset_mask
    point_invariant = np.ones(
        (len(candidate.motion_ids), len(candidate.point_ids)),
        dtype=np.bool_,
    )
    sites: dict[str, Any] = {}
    measured_shift = _norm(
        candidate.measured_site_positions_m - reference.measured_site_positions_m
    )
    force_norm = _norm(candidate.force_on_robot_n)
    selected_yield = _norm(
        candidate.selected_site_positions_m - candidate.original_site_positions_m
    )
    for site_id in spec.active_site_ids:
        site_index = candidate.site_ids.index(site_id)
        active = candidate.active_site_mask[:, site_index] & ~candidate.reset_mask
        point_index = candidate.point_ids.index(endpoint_point_by_site[site_id])
        point_invariant[active, point_index] = False
        candidate_selected = _summary(candidate_errors["selected_endpoint"][active, site_index])
        reference_selected = _summary(reference_errors["selected_endpoint"][active, site_index])
        candidate_orientation = _summary(candidate_errors["orientation"][active, site_index])
        reference_orientation = _summary(reference_errors["orientation"][active, site_index])
        selected_rmse_regression = _difference(
            candidate_selected["rmse"], reference_selected["rmse"]
        )
        selected_p95_regression = _difference(
            candidate_selected["p95"], reference_selected["p95"]
        )
        orientation_rmse_regression = _difference(
            candidate_orientation["rmse"], reference_orientation["rmse"]
        )
        orientation_p95_regression = _difference(
            candidate_orientation["p95"], reference_orientation["p95"]
        )
        values_and_limits = (
            (
                "selected_endpoint_rmse_regression_m",
                selected_rmse_regression,
                criteria.active_selected_endpoint_rmse_regression_m,
            ),
            (
                "selected_endpoint_p95_regression_m",
                selected_p95_regression,
                criteria.active_selected_endpoint_p95_regression_m,
            ),
            (
                "orientation_rmse_regression_rad",
                orientation_rmse_regression,
                criteria.active_orientation_rmse_regression_rad,
            ),
            (
                "orientation_p95_regression_rad",
                orientation_p95_regression,
                criteria.active_orientation_p95_regression_rad,
            ),
        )
        for metric_name, value, limit in values_and_limits:
            _check(
                checks,
                f"active_{metric_name}:{spec.name}:{site_id}",
                _finite_number(value) and float(value) <= limit,
                value=value,
                limit=limit,
            )
        active_force_peak = _summary(force_norm[active, site_index])["peak"]
        active_yield_peak = _summary(selected_yield[active, site_index])["peak"]
        measured_yield_peak = _summary(measured_shift[active, site_index])["peak"]
        for metric_name, value, minimum in (
            ("force_peak_n", active_force_peak, criteria.minimum_active_force_peak_n),
            ("reference_yield_peak_m", active_yield_peak, criteria.minimum_active_yield_peak_m),
            (
                "measured_yield_peak_m",
                measured_yield_peak,
                criteria.minimum_active_measured_yield_peak_m,
            ),
        ):
            _check(
                checks,
                f"active_{metric_name}:{spec.name}:{site_id}",
                _finite_number(value) and float(value) >= minimum,
                value=value,
                limit={"minimum": minimum},
            )
        force_vectors = candidate.force_on_robot_n[active, site_index]
        shift_vectors = (
            candidate.measured_site_positions_m[active, site_index]
            - reference.measured_site_positions_m[active, site_index]
        )
        force_magnitudes = _norm(force_vectors)
        along_force = np.full(force_magnitudes.shape, np.nan, dtype=np.float64)
        nonzero_force = np.isfinite(force_magnitudes) & (force_magnitudes > 0.0)
        along_force[nonzero_force] = np.sum(
            shift_vectors[nonzero_force]
            * force_vectors[nonzero_force]
            / force_magnitudes[nonzero_force, None],
            axis=-1,
        )
        along_force_peak = _summary(along_force)["peak"]
        _check(
            checks,
            f"active_measured_yield_along_force_peak_m:{spec.name}:{site_id}",
            _finite_number(along_force_peak)
            and float(along_force_peak)
            >= criteria.minimum_active_measured_yield_along_force_peak_m,
            value=along_force_peak,
            limit={
                "minimum": criteria.minimum_active_measured_yield_along_force_peak_m
            },
        )
        sites[site_id] = {
            "tracking_point_id": endpoint_point_by_site[site_id],
            "active_rows": int(np.count_nonzero(active)),
            "reference_selected_endpoint_error_m": reference_selected,
            "candidate_selected_endpoint_error_m": candidate_selected,
            "reference_orientation_error_rad": reference_orientation,
            "candidate_orientation_error_rad": candidate_orientation,
            "selected_endpoint_rmse_regression_m": selected_rmse_regression,
            "selected_endpoint_p95_regression_m": selected_p95_regression,
            "orientation_rmse_regression_rad": orientation_rmse_regression,
            "orientation_p95_regression_rad": orientation_p95_regression,
            "force_norm_n": _summary(force_norm[active, site_index]),
            "reference_yield_m": _summary(selected_yield[active, site_index]),
            "measured_yield_m": _summary(measured_shift[active, site_index]),
            "measured_yield_along_force_m": _summary(along_force),
        }

    for site_id in candidate.site_ids:
        if site_id in spec.active_site_ids:
            continue
        site_index = candidate.site_ids.index(site_id)
        rows = interaction_rows & ~candidate.active_site_mask[:, site_index]
        if not np.any(rows):
            continue
        force_peak = _summary(force_norm[rows, site_index])["peak"]
        yield_peak = _summary(selected_yield[rows, site_index])["peak"]
        shift_summary = _summary(measured_shift[rows, site_index])
        _check(
            checks,
            f"inactive_force_peak_n:{spec.name}:{site_id}",
            _finite_number(force_peak)
            and float(force_peak) <= criteria.inactive_force_tolerance_n,
            value=force_peak,
            limit=criteria.inactive_force_tolerance_n,
        )
        _check(
            checks,
            f"inactive_yield_peak_m:{spec.name}:{site_id}",
            _finite_number(yield_peak)
            and float(yield_peak) <= criteria.inactive_yield_tolerance_m,
            value=yield_peak,
            limit=criteria.inactive_yield_tolerance_m,
        )
        for metric_name, key, limit in (
            ("rmse", "rmse", criteria.inactive_cross_coupling_rmse_m),
            ("p95", "p95", criteria.inactive_cross_coupling_p95_m),
        ):
            value = shift_summary[key]
            _check(
                checks,
                f"inactive_cross_coupling_{metric_name}_m:{spec.name}:{site_id}",
                _finite_number(value) and float(value) <= limit,
                value=value,
                limit=limit,
            )
        sites[site_id] = {
            "inactive_rows": int(np.count_nonzero(rows)),
            "inactive_force_norm_n": _summary(force_norm[rows, site_index]),
            "inactive_reference_yield_m": _summary(selected_yield[rows, site_index]),
            "inactive_cross_coupling_m": shift_summary,
        }

    invariant_rows = interaction_rows[:, None] & point_invariant
    local_reference = reference_errors["local"][invariant_rows]
    local_candidate = candidate_errors["local"][invariant_rows]
    global_reference = reference_errors["global"][invariant_rows]
    global_candidate = candidate_errors["global"][invariant_rows]
    invariant: dict[str, Any] = {}
    for kind, reference_values, candidate_values, absolute, relative in (
        (
            "local",
            local_reference,
            local_candidate,
            criteria.active_invariant_local_mpjpe_absolute_regression_m,
            criteria.active_invariant_local_mpjpe_relative_regression,
        ),
        (
            "global",
            global_reference,
            global_candidate,
            criteria.active_invariant_global_mpjpe_absolute_regression_m,
            criteria.active_invariant_global_mpjpe_relative_regression,
        ),
    ):
        reference_mean = _summary(reference_values)["mean"]
        candidate_mean = _summary(candidate_values)["mean"]
        if _finite_number(reference_mean) and _finite_number(candidate_mean):
            regression = float(candidate_mean) - float(reference_mean)
            limit = _regression_limit(float(reference_mean), absolute, relative)
            passed = regression <= limit
        else:
            regression = limit = None
            passed = False
        _check(
            checks,
            f"active_invariant_{kind}_mpjpe_regression_m:{spec.name}",
            passed,
            value=regression,
            limit=limit,
        )
        invariant[kind] = {
            "reference_mean_m": reference_mean,
            "candidate_mean_m": candidate_mean,
            "regression_m": regression,
            "limit_m": limit,
        }

    action_changed = not _exact_array(
        reference.policy_actions[interaction_rows],
        candidate.policy_actions[interaction_rows],
    )
    _check(
        checks,
        f"residual_action_activation:{spec.name}",
        action_changed,
        value=action_changed,
        limit=True,
    )
    return {
        "name": spec.name,
        "reference_trial": spec.reference_trial,
        "candidate_trial": spec.candidate_trial,
        "alignment_sha256": alignment_digest(reference),
        "active_site_ids": list(spec.active_site_ids),
        "interaction_rows": int(np.count_nonzero(interaction_rows)),
        "invariant_point_samples": int(np.count_nonzero(invariant_rows)),
        "sites": sites,
        "invariant_tracking": invariant,
        "policy_actions_differ": action_changed,
    }


def evaluate_matched_review_suite(
    traces: Mapping[str, EvaluationTrace],
    spec: ReviewSuiteSpec,
    *,
    criteria: RegressionCriteria,
) -> dict[str, Any]:
    """Evaluate one complete caller-owned baseline and matched-force suite."""

    if not isinstance(traces, Mapping):
        raise TypeError("traces must be a mapping")
    if not isinstance(spec, ReviewSuiteSpec):
        raise TypeError("spec must be a ReviewSuiteSpec")
    if not isinstance(criteria, RegressionCriteria):
        raise TypeError("criteria must be RegressionCriteria")
    expected = set(spec.trial_names)
    if set(traces) != expected:
        raise ValueError(
            f"trace names do not match the suite: missing={sorted(expected - set(traces))}, "
            f"extra={sorted(set(traces) - expected)}"
        )
    for name, trace in traces.items():
        if not isinstance(trace, EvaluationTrace):
            raise TypeError(f"trace {name!r} must be an EvaluationTrace")
        if trace.trial_name != name:
            raise ValueError(f"trace key/name mismatch for {name!r}")

    baseline = traces[spec.baseline_trial]
    hard_off = traces[spec.hard_off_trial]
    no_contact = traces[spec.no_contact_trial]
    _trace_semantics(
        baseline,
        compliance_enabled=False,
        residual_enabled=False,
        active_sites=(),
    )
    _trace_semantics(
        hard_off,
        compliance_enabled=False,
        residual_enabled=False,
        active_sites=(),
    )
    _trace_semantics(
        no_contact,
        compliance_enabled=True,
        residual_enabled=True,
        active_sites=(),
    )

    checks: list[dict[str, Any]] = []
    reports = {name: evaluate_trace(trace) for name, trace in traces.items()}
    for name, trace in traces.items():
        _lifecycle_checks(checks, name, trace, criteria)

    hard_off_tracking = _base_tracking_checks(
        checks,
        "hard_off",
        baseline,
        hard_off,
        criteria,
        criteria.endpoint_rmse_regression_m,
    )
    no_contact_tracking = _base_tracking_checks(
        checks,
        "no_contact",
        hard_off,
        no_contact,
        criteria,
        criteria.no_contact_endpoint_delta_m,
        absolute_endpoint_delta=True,
    )
    for label, reference, candidate in (
        ("release_to_hard_off", baseline, hard_off),
        ("hard_off_to_no_contact", hard_off, no_contact),
    ):
        exact_actions = _exact_array(reference.policy_actions, candidate.policy_actions)
        _check(
            checks,
            f"exact_policy_actions:{label}",
            exact_actions,
            value=exact_actions,
            limit=True,
        )
    for name in (spec.baseline_trial, spec.hard_off_trial, spec.no_contact_trial):
        trace = traces[name]
        force_peak = _summary(_norm(trace.force_on_robot_n))["peak"]
        target_yield_peak = _summary(
            _norm(trace.selected_site_positions_m - trace.original_site_positions_m)
        )["peak"]
        for metric_name, value, limit in (
            ("inactive_force_peak_n", force_peak, criteria.inactive_force_tolerance_n),
            (
                "inactive_reference_yield_peak_m",
                target_yield_peak,
                criteria.inactive_yield_tolerance_m,
            ),
        ):
            _check(
                checks,
                f"{metric_name}:{name}",
                _finite_number(value) and float(value) <= limit,
                value=value,
                limit=limit,
            )
        zero_force = _exact_array(
            trace.force_on_robot_n,
            np.zeros_like(trace.force_on_robot_n),
        )
        zero_yield = _exact_array(
            trace.selected_site_positions_m,
            trace.original_site_positions_m,
        )
        _check(
            checks,
            f"exact_zero_force:{name}",
            zero_force,
            value=zero_force,
            limit=True,
        )
        _check(
            checks,
            f"exact_zero_reference_yield:{name}",
            zero_yield,
            value=zero_yield,
            limit=True,
        )
        if name in (spec.baseline_trial, spec.hard_off_trial):
            zero_compliance = _exact_array(
                trace.compliance_m_per_n,
                np.zeros_like(trace.compliance_m_per_n),
            )
            _check(
                checks,
                f"exact_zero_compliance:{name}",
                zero_compliance,
                value=zero_compliance,
                limit=True,
            )

    interactions = {}
    for interaction in spec.interactions:
        interactions[interaction.name] = _interaction_report(
            checks,
            interaction,
            traces[interaction.reference_trial],
            traces[interaction.candidate_trial],
            criteria,
        )

    return {
        "schema_version": "compliance_review_v1",
        "suite": {
            "baseline_trial": spec.baseline_trial,
            "hard_off_trial": spec.hard_off_trial,
            "no_contact_trial": spec.no_contact_trial,
            "interactions": [
                {
                    "name": item.name,
                    "reference_trial": item.reference_trial,
                    "candidate_trial": item.candidate_trial,
                    "active_site_ids": list(item.active_site_ids),
                }
                for item in spec.interactions
            ],
        },
        "criteria": {
            field_name: getattr(criteria, field_name)
            for field_name in criteria.__dataclass_fields__
        },
        "trials": reports,
        "hard_off_tracking": hard_off_tracking,
        "no_contact_tracking": no_contact_tracking,
        "interactions": interactions,
        "acceptance": {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
        },
    }
