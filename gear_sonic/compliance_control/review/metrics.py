"""Site-count- and tracker-layout-agnostic compliance evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .alignment import alignment_digest, assert_strict_alignment
from .schema import EvaluationTrace, RegressionCriteria, TrialMode, TrialSpec


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
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
        mean = 0.0
        rmse = 0.0
    else:
        scaled = finite / scale
        mean = float(scale * np.mean(scaled))
        rmse = float(scale * np.sqrt(np.mean(np.square(scaled))))
    return {
        "count": int(flattened.size),
        "finite_count": int(finite.size),
        "mean": mean,
        "rmse": rmse,
        "p95": float(scale * np.percentile(finite / scale, 95.0)) if scale else 0.0,
        "peak": float(np.max(finite)),
    }


def _vector_norm(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    output = np.full(values64.shape[:-1], np.nan, dtype=np.float64)
    valid = np.isfinite(values64).all(axis=-1)
    if not np.any(valid):
        return output
    finite_values = values64[valid]
    scale = np.max(np.abs(finite_values), axis=-1)
    norms = np.zeros_like(scale)
    nonzero = scale > 0.0
    normalized = finite_values[nonzero] / scale[nonzero, None]
    with np.errstate(over="ignore", invalid="ignore"):
        norms[nonzero] = scale[nonzero] * np.sqrt(
            np.sum(np.square(normalized), axis=-1)
        )
    norms[~np.isfinite(norms)] = np.nan
    output[valid] = norms
    return output


def _quaternion_angle_rad(reference: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, int]:
    reference64 = np.asarray(reference, dtype=np.float64)
    measured64 = np.asarray(measured, dtype=np.float64)
    reference_norm = _vector_norm(reference64)
    measured_norm = _vector_norm(measured64)
    valid = (
        np.isfinite(reference64).all(axis=-1)
        & np.isfinite(measured64).all(axis=-1)
        & (reference_norm > 0.0)
        & (measured_norm > 0.0)
    )
    angles = np.full(reference_norm.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        reference_unit = reference64[valid] / reference_norm[valid, None]
        measured_unit = measured64[valid] / measured_norm[valid, None]
        dot = np.sum(reference_unit * measured_unit, axis=-1)
        angles[valid] = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    return angles, int(np.size(valid) - np.count_nonzero(valid))


def _finiteness_report(trace: EvaluationTrace) -> dict[str, Any]:
    names = (
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
    )
    counts = {
        name: int(np.size(getattr(trace, name)) - np.count_nonzero(np.isfinite(getattr(trace, name))))
        for name in names
    }
    nonzero = {name: count for name, count in counts.items() if count}
    return {
        "all_finite": not nonzero,
        "nonfinite_count": int(sum(nonzero.values())),
        "nonfinite_fields": nonzero,
    }


def _masked_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    return _finite_summary(values[mask])


def evaluate_trace(trace: EvaluationTrace) -> dict[str, Any]:
    """Compute endpoint, pose, force, yield, lifecycle, and validity metrics."""

    original_error = _vector_norm(trace.measured_site_positions_m - trace.original_site_positions_m)
    selected_error = _vector_norm(trace.measured_site_positions_m - trace.selected_site_positions_m)
    yield_vectors = trace.selected_site_positions_m - trace.original_site_positions_m
    yield_norm = _vector_norm(yield_vectors)
    force_norm = _vector_norm(trace.force_on_robot_n)
    orientation_error, invalid_orientation_count = _quaternion_angle_rad(
        trace.original_site_orientations_xyzw,
        trace.measured_site_orientations_xyzw,
    )
    global_error = _vector_norm(trace.measured_points_global_m - trace.reference_points_global_m)
    local_error = _vector_norm(trace.measured_points_local_m - trace.reference_points_local_m)

    force_direction = np.zeros_like(trace.force_on_robot_n, dtype=np.float64)
    valid_force = np.isfinite(force_norm) & (force_norm > 0.0)
    force_direction[valid_force] = (
        trace.force_on_robot_n[valid_force] / force_norm[valid_force, None]
    )
    yield_along_force = np.sum(yield_vectors * force_direction, axis=-1)
    yield_along_force[~valid_force] = np.nan

    sites: dict[str, Any] = {}
    for site_index, site_id in enumerate(trace.site_ids):
        active = trace.active_site_mask[:, site_index]
        inactive = ~active
        sites[site_id] = {
            "original_endpoint_error_m": _finite_summary(original_error[:, site_index]),
            "selected_endpoint_error_m": _finite_summary(selected_error[:, site_index]),
            "orientation_error_rad": _finite_summary(orientation_error[:, site_index]),
            "force_norm_n": _finite_summary(force_norm[:, site_index]),
            "reference_yield_m": _finite_summary(yield_norm[:, site_index]),
            "yield_along_force_m": _finite_summary(yield_along_force[:, site_index]),
            "active_original_endpoint_error_m": _masked_summary(
                original_error[:, site_index], active
            ),
            "active_selected_endpoint_error_m": _masked_summary(
                selected_error[:, site_index], active
            ),
            "active_orientation_error_rad": _masked_summary(
                orientation_error[:, site_index], active
            ),
            "active_force_norm_n": _masked_summary(force_norm[:, site_index], active),
            "active_reference_yield_m": _masked_summary(yield_norm[:, site_index], active),
            "active_fraction": float(np.mean(active)),
            "inactive_force_norm_n": _masked_summary(force_norm[:, site_index], inactive),
            "inactive_reference_yield_m": _masked_summary(yield_norm[:, site_index], inactive),
        }

    terminal_count = int(np.count_nonzero(trace.terminal_mask))
    success_count = int(np.count_nonzero(trace.success_mask))
    reset_force = force_norm[trace.reset_mask]
    input_finiteness = _finiteness_report(trace)
    derived_arrays = (
        original_error,
        selected_error,
        yield_norm,
        force_norm,
        orientation_error,
        global_error,
        local_error,
    )
    derived_nonfinite_count = sum(
        int(array.size - np.count_nonzero(np.isfinite(array))) for array in derived_arrays
    )
    return {
        "trial_name": trace.trial_name,
        "alignment_sha256": alignment_digest(trace),
        "row_count": len(trace.motion_ids),
        "sequence_count": len(set(zip(trace.motion_ids, trace.sequence_ids, trace.seed_ids.tolist()))),
        "site_count": len(trace.site_ids),
        "point_count": len(trace.point_ids),
        "sites": sites,
        "global_mpjpe_m": _finite_summary(global_error),
        "local_mpjpe_m": _finite_summary(local_error),
        "lifecycle": {
            "terminal_count": terminal_count,
            "success_count": success_count,
            "success_rate": None if terminal_count == 0 else success_count / terminal_count,
            "fall_count": int(np.count_nonzero(trace.fall_mask)),
            "reset_count": int(np.count_nonzero(trace.reset_mask)),
            "post_reset_force_peak_n": _finite_summary(reset_force)["peak"],
        },
        "activation": {
            "enabled_fraction": float(np.mean(trace.compliance_enabled)),
            "active_sample_count": int(np.count_nonzero(trace.active_site_mask)),
            "simultaneous_active_peak": int(np.max(np.sum(trace.active_site_mask, axis=1))),
        },
        "validity": {
            **input_finiteness,
            "derived_nonfinite_count": derived_nonfinite_count,
            "invalid_orientation_count": invalid_orientation_count,
            "all_valid": input_finiteness["all_finite"]
            and invalid_orientation_count == 0
            and derived_nonfinite_count == 0,
        },
    }


def compare_aligned_traces(reference: EvaluationTrace, candidate: EvaluationTrace) -> dict[str, Any]:
    """Compare a candidate against an exactly aligned paired reference trace."""

    assert_strict_alignment(reference, candidate)
    reference_report = evaluate_trace(reference)
    candidate_report = evaluate_trace(candidate)
    measured_yield_vector = (
        candidate.measured_site_positions_m - reference.measured_site_positions_m
    )
    paired_shift = _vector_norm(measured_yield_vector)
    candidate_force_norm = _vector_norm(candidate.force_on_robot_n)
    valid_force = np.isfinite(candidate_force_norm) & (candidate_force_norm > 0.0)
    force_direction = np.zeros_like(candidate.force_on_robot_n, dtype=np.float64)
    force_direction[valid_force] = (
        candidate.force_on_robot_n[valid_force]
        / candidate_force_norm[valid_force, None]
    )
    measured_yield_along_force = np.sum(
        measured_yield_vector * force_direction,
        axis=-1,
    )
    measured_yield_along_force[~valid_force] = np.nan
    global_pose_shift = _vector_norm(
        candidate.measured_points_global_m - reference.measured_points_global_m
    )
    local_pose_shift = _vector_norm(
        candidate.measured_points_local_m - reference.measured_points_local_m
    )

    sites: dict[str, Any] = {}
    for site_index, site_id in enumerate(reference.site_ids):
        inactive = ~candidate.active_site_mask[:, site_index]
        reference_site = reference_report["sites"][site_id]
        candidate_site = candidate_report["sites"][site_id]
        sites[site_id] = {
            "paired_endpoint_shift_m": _finite_summary(paired_shift[:, site_index]),
            "active_paired_endpoint_shift_m": _masked_summary(
                paired_shift[:, site_index], candidate.active_site_mask[:, site_index]
            ),
            "active_measured_yield_m": _masked_summary(
                paired_shift[:, site_index], candidate.active_site_mask[:, site_index]
            ),
            "active_measured_yield_along_force_m": _masked_summary(
                measured_yield_along_force[:, site_index],
                candidate.active_site_mask[:, site_index],
            ),
            "inactive_cross_coupling_shift_m": _masked_summary(
                paired_shift[:, site_index], inactive
            ),
            "original_endpoint_rmse_delta_m": _optional_difference(
                candidate_site["original_endpoint_error_m"]["rmse"],
                reference_site["original_endpoint_error_m"]["rmse"],
            ),
            "selected_endpoint_rmse_delta_m": _optional_difference(
                candidate_site["selected_endpoint_error_m"]["rmse"],
                reference_site["selected_endpoint_error_m"]["rmse"],
            ),
            "orientation_rmse_delta_rad": _optional_difference(
                candidate_site["orientation_error_rad"]["rmse"],
                reference_site["orientation_error_rad"]["rmse"],
            ),
            "force_peak_delta_n": _optional_difference(
                candidate_site["force_norm_n"]["peak"],
                reference_site["force_norm_n"]["peak"],
            ),
            "yield_rmse_delta_m": _optional_difference(
                candidate_site["reference_yield_m"]["rmse"],
                reference_site["reference_yield_m"]["rmse"],
            ),
        }
    return {
        "reference_trial": reference.trial_name,
        "candidate_trial": candidate.trial_name,
        "alignment_sha256": alignment_digest(reference),
        "sites": sites,
        "paired_global_pose_shift_m": _finite_summary(global_pose_shift),
        "paired_local_pose_shift_m": _finite_summary(local_pose_shift),
        "global_mpjpe_delta_m": _optional_difference(
            candidate_report["global_mpjpe_m"]["mean"],
            reference_report["global_mpjpe_m"]["mean"],
        ),
        "local_mpjpe_delta_m": _optional_difference(
            candidate_report["local_mpjpe_m"]["mean"],
            reference_report["local_mpjpe_m"]["mean"],
        ),
        "success_rate_delta": _optional_difference(
            candidate_report["lifecycle"]["success_rate"],
            reference_report["lifecycle"]["success_rate"],
        ),
        "fall_count_delta": (
            candidate_report["lifecycle"]["fall_count"]
            - reference_report["lifecycle"]["fall_count"]
        ),
    }


def _evaluate_active_tracking_preservation(
    reference: EvaluationTrace,
    candidate: EvaluationTrace,
    *,
    expected_active_site_ids: Sequence[str],
    endpoint_point_by_site: Mapping[str, str],
) -> dict[str, Any]:
    """Compare active compliant tracking against the aligned overlay-off trace.

    Endpoint position is measured against the yielded target. Orientation stays
    referenced to the original target because the controller implements no
    rotational compliance. Whole-body preservation excludes only the tracking
    point whose corresponding site is active on that row.
    """

    assert_strict_alignment(reference, candidate)
    reference_original_error = _vector_norm(
        reference.measured_site_positions_m - reference.original_site_positions_m
    )
    candidate_selected_error = _vector_norm(
        candidate.measured_site_positions_m - candidate.selected_site_positions_m
    )
    reference_orientation_error, _ = _quaternion_angle_rad(
        reference.original_site_orientations_xyzw,
        reference.measured_site_orientations_xyzw,
    )
    candidate_orientation_error, _ = _quaternion_angle_rad(
        candidate.original_site_orientations_xyzw,
        candidate.measured_site_orientations_xyzw,
    )

    sites: dict[str, Any] = {}
    interaction_rows = np.zeros(len(candidate.motion_ids), dtype=np.bool_)
    invariant_point_mask = np.zeros(
        (len(candidate.motion_ids), len(candidate.point_ids)),
        dtype=np.bool_,
    )
    for site_id in expected_active_site_ids:
        site_index = candidate.site_ids.index(site_id)
        active = candidate.active_site_mask[:, site_index] & ~candidate.reset_mask
        interaction_rows |= active
        candidate_endpoint = _masked_summary(
            candidate_selected_error[:, site_index],
            active,
        )
        reference_endpoint = _masked_summary(
            reference_original_error[:, site_index],
            active,
        )
        candidate_orientation = _masked_summary(
            candidate_orientation_error[:, site_index],
            active,
        )
        reference_orientation = _masked_summary(
            reference_orientation_error[:, site_index],
            active,
        )
        sites[site_id] = {
            "tracking_point_id": endpoint_point_by_site[site_id],
            "active_selected_endpoint_error_m": candidate_endpoint,
            "off_original_endpoint_error_m": reference_endpoint,
            "selected_endpoint_rmse_regression_m": _optional_difference(
                candidate_endpoint["rmse"],
                reference_endpoint["rmse"],
            ),
            "selected_endpoint_p95_regression_m": _optional_difference(
                candidate_endpoint["p95"],
                reference_endpoint["p95"],
            ),
            "active_orientation_error_rad": candidate_orientation,
            "off_orientation_error_rad": reference_orientation,
            "orientation_rmse_regression_rad": _optional_difference(
                candidate_orientation["rmse"],
                reference_orientation["rmse"],
            ),
            "orientation_p95_regression_rad": _optional_difference(
                candidate_orientation["p95"],
                reference_orientation["p95"],
            ),
        }
    invariant_point_mask[interaction_rows] = True
    for site_id in expected_active_site_ids:
        site_index = candidate.site_ids.index(site_id)
        point_index = candidate.point_ids.index(endpoint_point_by_site[site_id])
        active = candidate.active_site_mask[:, site_index] & ~candidate.reset_mask
        invariant_point_mask[active, point_index] = False

    reference_global_error = _vector_norm(
        reference.measured_points_global_m - reference.reference_points_global_m
    )
    candidate_global_error = _vector_norm(
        candidate.measured_points_global_m - candidate.reference_points_global_m
    )
    reference_local_error = _vector_norm(
        reference.measured_points_local_m - reference.reference_points_local_m
    )
    candidate_local_error = _vector_norm(
        candidate.measured_points_local_m - candidate.reference_points_local_m
    )
    off_global = _masked_summary(reference_global_error, invariant_point_mask)
    active_global = _masked_summary(candidate_global_error, invariant_point_mask)
    off_local = _masked_summary(reference_local_error, invariant_point_mask)
    active_local = _masked_summary(candidate_local_error, invariant_point_mask)
    return {
        "interaction_row_count": int(np.count_nonzero(interaction_rows)),
        "invariant_point_sample_count": int(np.count_nonzero(invariant_point_mask)),
        "sites": sites,
        "invariant_global_mpjpe_m": active_global,
        "off_invariant_global_mpjpe_m": off_global,
        "invariant_global_mpjpe_regression_m": _optional_difference(
            active_global["mean"],
            off_global["mean"],
        ),
        "invariant_local_mpjpe_m": active_local,
        "off_invariant_local_mpjpe_m": off_local,
        "invariant_local_mpjpe_regression_m": _optional_difference(
            active_local["mean"],
            off_local["mean"],
        ),
    }


def _optional_difference(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(candidate - reference)


def _validate_protocol(trace: EvaluationTrace, spec: TrialSpec) -> None:
    if trace.trial_name != spec.name:
        raise ValueError(f"trace/spec name mismatch: {trace.trial_name} != {spec.name}")
    site_set = set(trace.site_ids)
    expected = set(spec.expected_active_site_ids)
    if not expected.issubset(site_set):
        raise ValueError(f"trial {spec.name} references unknown active sites")
    observed = {
        trace.site_ids[index]
        for index in np.flatnonzero(np.any(trace.active_site_mask, axis=0)).tolist()
    }

    if spec.mode in {TrialMode.BASELINE, TrialMode.OFF}:
        if np.any(trace.compliance_enabled) or np.any(trace.active_site_mask):
            raise ValueError(f"trial {spec.name} must be fully disabled")
    elif spec.mode is TrialMode.NO_CONTACT:
        if not np.all(trace.compliance_enabled):
            raise ValueError(f"trial {spec.name} must remain enabled on every row")
        if np.any(trace.active_site_mask):
            raise ValueError(f"trial {spec.name} must not activate a force site")
    elif spec.mode is TrialMode.SINGLE_SITE:
        if not np.all(trace.compliance_enabled):
            raise ValueError(f"trial {spec.name} must remain enabled on every row")
        if observed != expected:
            raise ValueError(f"trial {spec.name} active-site union does not match its spec")
        if np.max(np.sum(trace.active_site_mask, axis=1)) > 1:
            raise ValueError(f"trial {spec.name} activates more than one site at a time")
    elif spec.mode is TrialMode.MULTI_SITE:
        if not np.all(trace.compliance_enabled):
            raise ValueError(f"trial {spec.name} must remain enabled on every row")
        if observed != expected:
            raise ValueError(f"trial {spec.name} active-site union does not match its spec")
        expected_indices = [trace.site_ids.index(site_id) for site_id in spec.expected_active_site_ids]
        simultaneous = np.all(trace.active_site_mask[:, expected_indices], axis=1)
        if not np.any(simultaneous):
            raise ValueError(f"trial {spec.name} never activates its expected sites simultaneously")


def evaluate_trial_suite(
    traces: Mapping[str, EvaluationTrace],
    specs: Sequence[TrialSpec],
    *,
    baseline_name: str,
    criteria: RegressionCriteria,
) -> dict[str, Any]:
    """Evaluate a paired baseline/off/no-contact/interaction trace suite."""

    spec_by_name = {spec.name: spec for spec in specs}
    if len(spec_by_name) != len(specs):
        raise ValueError("trial spec names must be unique")
    if set(traces) != set(spec_by_name):
        raise ValueError("trace and trial-spec names must match exactly")
    if baseline_name not in traces:
        raise ValueError("baseline_name must identify a supplied trace")
    if spec_by_name[baseline_name].mode is not TrialMode.BASELINE:
        raise ValueError("baseline_name must identify a baseline-mode trial")
    baseline = traces[baseline_name]
    unknown_endpoints = set(criteria.endpoint_site_ids) - set(baseline.site_ids)
    if unknown_endpoints:
        raise ValueError("criteria endpoint_site_ids must exist in the trace layout")
    unknown_endpoint_points = set(criteria.endpoint_tracking_point_ids) - set(
        baseline.point_ids
    )
    if unknown_endpoint_points:
        raise ValueError(
            "criteria endpoint_tracking_point_ids must exist in the trace layout"
        )
    endpoint_point_by_site = dict(
        zip(criteria.endpoint_site_ids, criteria.endpoint_tracking_point_ids, strict=True)
    )

    modes = {mode: [spec for spec in specs if spec.mode is mode] for mode in TrialMode}
    if len(modes[TrialMode.BASELINE]) != 1:
        raise ValueError("exactly one baseline-mode trial is required")
    if len(modes[TrialMode.OFF]) != 1:
        raise ValueError("exactly one off-mode trial is required")
    if not modes[TrialMode.NO_CONTACT]:
        raise ValueError("at least one no-contact trial is required")
    for endpoint_site_id in criteria.endpoint_site_ids:
        matching_single = [
            spec
            for spec in modes[TrialMode.SINGLE_SITE]
            if spec.expected_active_site_ids == (endpoint_site_id,)
        ]
        if len(matching_single) != 1:
            raise ValueError(
                f"exactly one single-site trial is required for endpoint {endpoint_site_id}"
            )
    endpoint_site_set = set(criteria.endpoint_site_ids)
    matching_multi = [
        spec
        for spec in modes[TrialMode.MULTI_SITE]
        if set(spec.expected_active_site_ids) == endpoint_site_set
        and len(spec.expected_active_site_ids) == len(criteria.endpoint_site_ids)
    ]
    if not matching_multi:
        raise ValueError("a simultaneous multi-site trial is required for all endpoint sites")

    off_name = modes[TrialMode.OFF][0].name
    trial_reports: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    off_comparisons: dict[str, Any] = {}
    for spec in specs:
        trace = traces[spec.name]
        _validate_protocol(trace, spec)
        assert_strict_alignment(baseline, trace)
        trial_reports[spec.name] = evaluate_trace(trace)
        if spec.name != baseline_name:
            comparisons[spec.name] = compare_aligned_traces(baseline, trace)
        if spec.name not in {baseline_name, off_name}:
            off_comparisons[spec.name] = compare_aligned_traces(traces[off_name], trace)

    active_tracking_to_off = {
        spec.name: _evaluate_active_tracking_preservation(
            traces[off_name],
            traces[spec.name],
            expected_active_site_ids=spec.expected_active_site_ids,
            endpoint_point_by_site=endpoint_point_by_site,
        )
        for spec in specs
        if spec.mode in {TrialMode.SINGLE_SITE, TrialMode.MULTI_SITE}
    }

    no_contact_names = [spec.name for spec in specs if spec.mode is TrialMode.NO_CONTACT]
    acceptance = _assess_suite(
        baseline_name=baseline_name,
        off_name=off_name,
        no_contact_names=no_contact_names,
        specs=specs,
        trial_reports=trial_reports,
        comparisons=comparisons,
        criteria=criteria,
        traces=traces,
        active_tracking_to_off=active_tracking_to_off,
    )
    return {
        "schema_version": "compliance_evaluation_v2",
        "baseline_trial": baseline_name,
        "off_trial": off_name,
        "alignment_sha256": alignment_digest(baseline),
        "trial_order": [spec.name for spec in specs],
        "trial_specs": [
            {
                "name": spec.name,
                "mode": spec.mode.value,
                "expected_active_site_ids": list(spec.expected_active_site_ids),
            }
            for spec in specs
        ],
        "trials": trial_reports,
        "paired_to_baseline": comparisons,
        "paired_to_off": off_comparisons,
        "active_tracking_to_off": active_tracking_to_off,
        "acceptance": acceptance,
    }


def _assess_suite(
    *,
    baseline_name: str,
    off_name: str,
    no_contact_names: Sequence[str],
    specs: Sequence[TrialSpec],
    trial_reports: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    criteria: RegressionCriteria,
    traces: Mapping[str, EvaluationTrace],
    active_tracking_to_off: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    baseline_report = trial_reports[baseline_name]
    off_report = trial_reports[off_name]
    success_drop = _optional_difference(
        baseline_report["lifecycle"]["success_rate"],
        off_report["lifecycle"]["success_rate"],
    )
    checks.append(
        _upper_bound_check(
            "off_success_rate_drop",
            success_drop,
            criteria.max_success_rate_drop,
        )
    )
    local_delta = comparisons[off_name]["local_mpjpe_delta_m"]
    baseline_local = baseline_report["local_mpjpe_m"]["mean"]
    local_limit = None
    if baseline_local is not None:
        local_limit = max(
            criteria.local_mpjpe_absolute_regression_m,
            baseline_local * criteria.local_mpjpe_relative_regression,
        )
    checks.append(_upper_bound_check("off_local_mpjpe_regression_m", local_delta, local_limit))

    for site_id in criteria.endpoint_site_ids:
        endpoint_delta = comparisons[off_name]["sites"][site_id][
            "original_endpoint_rmse_delta_m"
        ]
        checks.append(
            _upper_bound_check(
                f"off_endpoint_rmse_regression_m:{site_id}",
                endpoint_delta,
                criteria.endpoint_rmse_regression_m,
            )
        )

    for trial_name in no_contact_names:
        no_contact_pair = compare_aligned_traces(traces[off_name], traces[trial_name])
        for site_id in criteria.endpoint_site_ids:
            endpoint_delta = no_contact_pair["sites"][site_id][
                "original_endpoint_rmse_delta_m"
            ]
            checks.append(
                _absolute_bound_check(
                    f"no_contact_endpoint_rmse_delta_m:{trial_name}:{site_id}",
                    endpoint_delta,
                    criteria.no_contact_endpoint_delta_m,
                )
            )

    for spec in specs:
        report = trial_reports[spec.name]
        checks.append(
            {
                "name": f"zero_falls:{spec.name}",
                "value": report["lifecycle"]["fall_count"],
                "limit": 0,
                "passed": report["lifecycle"]["fall_count"] == 0,
            }
        )
        checks.append(
            {
                "name": f"success_rate_one:{spec.name}",
                "value": report["lifecycle"]["success_rate"],
                "limit": 1.0,
                "passed": report["lifecycle"]["success_rate"] == 1.0,
            }
        )
        checks.append(
            {
                "name": f"finite_and_valid:{spec.name}",
                "value": report["validity"]["all_valid"],
                "limit": True,
                "passed": bool(report["validity"]["all_valid"]),
            }
        )
        post_reset_peak = report["lifecycle"]["post_reset_force_peak_n"]
        checks.append(
            {
                "name": f"reset_evidence:{spec.name}",
                "value": report["lifecycle"]["reset_count"],
                "limit": ">=1",
                "passed": report["lifecycle"]["reset_count"] >= 1,
            }
        )
        checks.append(
            _upper_bound_check(
                f"post_reset_force_peak_n:{spec.name}",
                post_reset_peak,
                criteria.reset_wrench_tolerance_n,
            )
        )
        for site_id in traces[spec.name].site_ids:
            site_report = report["sites"][site_id]
            if site_report["inactive_force_norm_n"]["count"] > 0:
                checks.append(
                    _upper_bound_check(
                        f"inactive_force_peak_n:{spec.name}:{site_id}",
                        site_report["inactive_force_norm_n"]["peak"],
                        criteria.inactive_force_tolerance_n,
                    )
                )
            if site_report["inactive_reference_yield_m"]["count"] > 0:
                checks.append(
                    _upper_bound_check(
                        f"inactive_yield_peak_m:{spec.name}:{site_id}",
                        site_report["inactive_reference_yield_m"]["peak"],
                        criteria.inactive_yield_tolerance_m,
                    )
                )
        if spec.mode in {TrialMode.SINGLE_SITE, TrialMode.MULTI_SITE}:
            paired_to_off = compare_aligned_traces(traces[off_name], traces[spec.name])
            active_tracking = active_tracking_to_off[spec.name]
            for site_id in spec.expected_active_site_ids:
                site_report = report["sites"][site_id]
                paired_site = paired_to_off["sites"][site_id]
                tracking_site = active_tracking["sites"][site_id]
                checks.append(
                    _lower_bound_check(
                        f"active_force_peak_n:{spec.name}:{site_id}",
                        site_report["active_force_norm_n"]["peak"],
                        criteria.minimum_active_force_peak_n,
                    )
                )
                checks.append(
                    _lower_bound_check(
                        f"active_yield_peak_m:{spec.name}:{site_id}",
                        site_report["active_reference_yield_m"]["peak"],
                        criteria.minimum_active_yield_peak_m,
                    )
                )
                checks.append(
                    _lower_bound_check(
                        f"active_measured_yield_peak_m:{spec.name}:{site_id}",
                        paired_site["active_measured_yield_m"]["peak"],
                        criteria.minimum_active_measured_yield_peak_m,
                    )
                )
                checks.append(
                    _lower_bound_check(
                        f"active_measured_yield_along_force_peak_m:{spec.name}:{site_id}",
                        paired_site["active_measured_yield_along_force_m"]["peak"],
                        criteria.minimum_active_measured_yield_along_force_peak_m,
                    )
                )
                checks.append(
                    _upper_bound_check(
                        f"active_selected_endpoint_rmse_regression_m:{spec.name}:{site_id}",
                        tracking_site["selected_endpoint_rmse_regression_m"],
                        criteria.active_selected_endpoint_rmse_regression_m,
                    )
                )
                checks.append(
                    _upper_bound_check(
                        f"active_selected_endpoint_p95_regression_m:{spec.name}:{site_id}",
                        tracking_site["selected_endpoint_p95_regression_m"],
                        criteria.active_selected_endpoint_p95_regression_m,
                    )
                )
                checks.append(
                    _upper_bound_check(
                        f"active_orientation_rmse_regression_rad:{spec.name}:{site_id}",
                        tracking_site["orientation_rmse_regression_rad"],
                        criteria.active_orientation_rmse_regression_rad,
                    )
                )
                checks.append(
                    _upper_bound_check(
                        f"active_orientation_p95_regression_rad:{spec.name}:{site_id}",
                        tracking_site["orientation_p95_regression_rad"],
                        criteria.active_orientation_p95_regression_rad,
                    )
                )
            off_invariant_local = active_tracking["off_invariant_local_mpjpe_m"]["mean"]
            active_local_limit = None
            if off_invariant_local is not None:
                active_local_limit = max(
                    criteria.active_invariant_local_mpjpe_absolute_regression_m,
                    off_invariant_local
                    * criteria.active_invariant_local_mpjpe_relative_regression,
                )
            checks.append(
                _upper_bound_check(
                    f"active_invariant_local_mpjpe_regression_m:{spec.name}",
                    active_tracking["invariant_local_mpjpe_regression_m"],
                    active_local_limit,
                )
            )
            off_invariant_global = active_tracking["off_invariant_global_mpjpe_m"]["mean"]
            active_global_limit = None
            if off_invariant_global is not None:
                active_global_limit = max(
                    criteria.active_invariant_global_mpjpe_absolute_regression_m,
                    off_invariant_global
                    * criteria.active_invariant_global_mpjpe_relative_regression,
                )
            checks.append(
                _upper_bound_check(
                    f"active_invariant_global_mpjpe_regression_m:{spec.name}",
                    active_tracking["invariant_global_mpjpe_regression_m"],
                    active_global_limit,
                )
            )
            inactive_site_ids = set(traces[spec.name].site_ids) - set(
                spec.expected_active_site_ids
            )
            for site_id in sorted(inactive_site_ids):
                cross_coupling = paired_to_off["sites"][site_id][
                    "inactive_cross_coupling_shift_m"
                ]
                checks.append(
                    _upper_bound_check(
                        f"inactive_cross_coupling_rmse_m:{spec.name}:{site_id}",
                        cross_coupling["rmse"],
                        criteria.inactive_cross_coupling_rmse_m,
                    )
                )
                checks.append(
                    _upper_bound_check(
                        f"inactive_cross_coupling_p95_m:{spec.name}:{site_id}",
                        cross_coupling["p95"],
                        criteria.inactive_cross_coupling_p95_m,
                    )
                )
    return {
        "passed": all(check["passed"] for check in checks),
        "criteria": {
            "endpoint_site_ids": list(criteria.endpoint_site_ids),
            "endpoint_tracking_point_ids": list(criteria.endpoint_tracking_point_ids),
            "max_success_rate_drop": criteria.max_success_rate_drop,
            "local_mpjpe_absolute_regression_m": criteria.local_mpjpe_absolute_regression_m,
            "local_mpjpe_relative_regression": criteria.local_mpjpe_relative_regression,
            "endpoint_rmse_regression_m": criteria.endpoint_rmse_regression_m,
            "no_contact_endpoint_delta_m": criteria.no_contact_endpoint_delta_m,
            "reset_wrench_tolerance_n": criteria.reset_wrench_tolerance_n,
            "inactive_force_tolerance_n": criteria.inactive_force_tolerance_n,
            "inactive_yield_tolerance_m": criteria.inactive_yield_tolerance_m,
            "minimum_active_force_peak_n": criteria.minimum_active_force_peak_n,
            "minimum_active_yield_peak_m": criteria.minimum_active_yield_peak_m,
            "minimum_active_measured_yield_peak_m": (
                criteria.minimum_active_measured_yield_peak_m
            ),
            "minimum_active_measured_yield_along_force_peak_m": (
                criteria.minimum_active_measured_yield_along_force_peak_m
            ),
            "inactive_cross_coupling_rmse_m": criteria.inactive_cross_coupling_rmse_m,
            "inactive_cross_coupling_p95_m": criteria.inactive_cross_coupling_p95_m,
            "active_selected_endpoint_rmse_regression_m": (
                criteria.active_selected_endpoint_rmse_regression_m
            ),
            "active_selected_endpoint_p95_regression_m": (
                criteria.active_selected_endpoint_p95_regression_m
            ),
            "active_orientation_rmse_regression_rad": (
                criteria.active_orientation_rmse_regression_rad
            ),
            "active_orientation_p95_regression_rad": (
                criteria.active_orientation_p95_regression_rad
            ),
            "active_invariant_local_mpjpe_absolute_regression_m": (
                criteria.active_invariant_local_mpjpe_absolute_regression_m
            ),
            "active_invariant_local_mpjpe_relative_regression": (
                criteria.active_invariant_local_mpjpe_relative_regression
            ),
            "active_invariant_global_mpjpe_absolute_regression_m": (
                criteria.active_invariant_global_mpjpe_absolute_regression_m
            ),
            "active_invariant_global_mpjpe_relative_regression": (
                criteria.active_invariant_global_mpjpe_relative_regression
            ),
        },
        "checks": checks,
    }


def _upper_bound_check(
    name: str,
    value: float | None,
    limit: float | None,
) -> dict[str, Any]:
    passed = value is not None and limit is not None and value <= limit
    return {"name": name, "value": value, "limit": limit, "passed": bool(passed)}


def _absolute_bound_check(name: str, value: float | None, limit: float) -> dict[str, Any]:
    passed = value is not None and abs(value) <= limit
    return {"name": name, "value": value, "limit": limit, "passed": bool(passed)}


def _lower_bound_check(name: str, value: float | None, limit: float) -> dict[str, Any]:
    passed = value is not None and value >= limit
    return {"name": name, "value": value, "limit": limit, "passed": bool(passed)}
