"""CPU contract tests for tracker-neutral paired compliance evaluation."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile

import numpy as np

try:
    import pytest
except ModuleNotFoundError as error:  # pragma: no cover - unittest portability gate
    raise unittest.SkipTest("pytest is required for the portable review suite") from error

from gear_sonic.compliance_control.review import (
    EvaluationTrace,
    MatchedInteractionSpec,
    RegressionCriteria,
    ReviewPanelSpec,
    ReviewSuiteSpec,
    ReviewVideoSpec,
    TraceAlignmentError,
    TrialMode,
    TrialSpec,
    alignment_digest,
    assert_strict_alignment,
    build_review_video_manifest,
    compare_aligned_traces,
    evaluate_matched_review_suite,
    evaluate_trace,
    evaluate_trial_suite,
    load_report_json_with_sha256,
    load_trace_npz,
    load_trace_npz_with_sha256,
    validate_review_video_manifest,
    write_report_json_atomic,
    write_review_video_manifest_atomic,
    write_trace_npz_atomic,
)

REVIEW_DIR = Path(__file__).parents[1] / "compliance_control" / "review"
SITE_IDS = ("endpoint_a", "endpoint_b", "balance_probe")
POINT_IDS = ("endpoint_a", "endpoint_b") + tuple(
    f"tracking_point_{index}" for index in range(5)
)


def _make_trace(
    trial_name: str,
    *,
    endpoint_error_m: float,
    enabled: bool = False,
    active_sites: tuple[str, ...] = (),
    nonfinite: bool = False,
    fall: bool = False,
) -> EvaluationTrace:
    row_count = 8
    site_count = len(SITE_IDS)
    point_count = len(POINT_IDS)
    original_site = np.zeros((row_count, site_count, 3), dtype=np.float64)
    original_site[..., 0] = np.arange(row_count, dtype=np.float64)[:, None] * 0.01
    selected_site = original_site.copy()
    measured_site = original_site.copy()
    measured_site[..., 2] += endpoint_error_m
    reference_global = np.zeros((row_count, point_count, 3), dtype=np.float64)
    measured_global = reference_global.copy()
    measured_global[..., 2] = endpoint_error_m
    reference_local = np.zeros((row_count, point_count, 3), dtype=np.float64)
    measured_local = reference_local.copy()
    measured_local[..., 1] = endpoint_error_m

    original_orientation = np.zeros((row_count, site_count, 4), dtype=np.float64)
    original_orientation[..., 3] = 1.0
    measured_orientation = original_orientation.copy()
    angle = 0.1
    measured_orientation[..., 0] = np.sin(angle / 2.0)
    measured_orientation[..., 3] = np.cos(angle / 2.0)

    compliance_enabled = np.full(row_count, enabled, dtype=np.bool_)
    residual_enabled = np.full(row_count, enabled, dtype=np.bool_)
    active_mask = np.zeros((row_count, site_count), dtype=np.bool_)
    force = np.zeros((row_count, site_count, 3), dtype=np.float64)
    compliance = np.zeros((row_count, site_count, 3), dtype=np.float64)
    for site_id in active_sites:
        site_index = SITE_IDS.index(site_id)
        active_mask[1:3, site_index] = True
        selected_site[1:3, site_index, 0] += 0.02
        measured_site[1:3, site_index, 0] += 0.01
        force[1:3, site_index, 0] = 4.0
        compliance[1:3, site_index, 0] = 0.005

    terminal = np.zeros(row_count, dtype=np.bool_)
    terminal[[3, 7]] = True
    success = terminal.copy()
    fall_mask = np.zeros(row_count, dtype=np.bool_)
    if fall:
        success[7] = False
        fall_mask[7] = True
    reset = np.zeros(row_count, dtype=np.bool_)
    reset[[0, 4]] = True
    if nonfinite:
        measured_site[2, 0, 0] = np.nan

    return EvaluationTrace(
        trial_name=trial_name,
        motion_ids=("motion_alpha",) * 4 + ("motion_beta",) * 4,
        sequence_ids=("sequence_0",) * 4 + ("sequence_1",) * 4,
        seed_ids=np.full(row_count, 23, dtype=np.int64),
        frame_indices=np.tile(np.arange(4, dtype=np.int64), 2),
        timestamps_s=np.tile(np.arange(4, dtype=np.float64) * 0.02, 2),
        site_ids=SITE_IDS,
        point_ids=POINT_IDS,
        original_site_positions_m=original_site,
        selected_site_positions_m=selected_site,
        measured_site_positions_m=measured_site,
        original_site_orientations_xyzw=original_orientation,
        measured_site_orientations_xyzw=measured_orientation,
        reference_points_global_m=reference_global,
        measured_points_global_m=measured_global,
        reference_points_local_m=reference_local,
        measured_points_local_m=measured_local,
        force_on_robot_n=force,
        force_on_robot_world_n=force,
        force_on_robot_common_n=force,
        compliance_m_per_n=compliance,
        compliance_enabled=compliance_enabled,
        residual_enabled=residual_enabled,
        active_site_mask=active_mask,
        policy_actions=np.zeros((row_count, 5), dtype=np.float64),
        terminal_mask=terminal,
        success_mask=success,
        fall_mask=fall_mask,
        reset_mask=reset,
    )


def _suite_inputs(*, nonfinite_trial: str | None = None):
    traces = {
        "released_baseline": _make_trace("released_baseline", endpoint_error_m=0.010),
        "overlay_off": _make_trace("overlay_off", endpoint_error_m=0.011),
        "enabled_no_contact": _make_trace(
            "enabled_no_contact", endpoint_error_m=0.0115, enabled=True
        ),
        "single_a": _make_trace(
            "single_a",
            endpoint_error_m=0.012,
            enabled=True,
            active_sites=("endpoint_a",),
            nonfinite=nonfinite_trial == "single_a",
        ),
        "single_b": _make_trace(
            "single_b",
            endpoint_error_m=0.012,
            enabled=True,
            active_sites=("endpoint_b",),
        ),
        "simultaneous": _make_trace(
            "simultaneous",
            endpoint_error_m=0.012,
            enabled=True,
            active_sites=("endpoint_a", "endpoint_b"),
        ),
    }
    specs = (
        TrialSpec("released_baseline", TrialMode.BASELINE),
        TrialSpec("overlay_off", TrialMode.OFF),
        TrialSpec("enabled_no_contact", TrialMode.NO_CONTACT),
        TrialSpec("single_a", TrialMode.SINGLE_SITE, ("endpoint_a",)),
        TrialSpec("single_b", TrialMode.SINGLE_SITE, ("endpoint_b",)),
        TrialSpec(
            "simultaneous",
            TrialMode.MULTI_SITE,
            ("endpoint_a", "endpoint_b"),
        ),
    )
    criteria = RegressionCriteria(
        endpoint_site_ids=("endpoint_a", "endpoint_b"),
        endpoint_tracking_point_ids=("endpoint_a", "endpoint_b"),
    )
    return traces, specs, criteria


MATCHED_ROLES = {
    "release_baseline",
    "chip_hard_off",
    "enabled_no_contact",
    "single_left_stiff",
    "single_left_compliant",
    "single_right_stiff",
    "single_right_compliant",
    "simultaneous_stiff",
    "simultaneous_compliant",
}


def _make_matched_trace(
    trial_name: str,
    *,
    active_sites: tuple[str, ...] = (),
    enabled: bool = False,
    residual: bool = False,
) -> EvaluationTrace:
    row_count = 6
    site_count = len(SITE_IDS)
    point_count = len(POINT_IDS)
    original_site = np.zeros((row_count, site_count, 3), dtype=np.float64)
    original_site[..., 0] = np.arange(row_count, dtype=np.float64)[:, None] * 0.001
    selected_site = original_site.copy()
    measured_site = original_site.copy()
    measured_site[..., 2] += 0.010
    original_orientation = np.zeros((row_count, site_count, 4), dtype=np.float64)
    original_orientation[..., 3] = 1.0
    measured_orientation = original_orientation.copy()
    measured_orientation[..., 1] = np.sin(0.01)
    measured_orientation[..., 3] = np.cos(0.01)

    reference_global = np.zeros((row_count, point_count, 3), dtype=np.float64)
    reference_global[..., 0] = np.arange(row_count, dtype=np.float64)[:, None] * 0.002
    measured_global = reference_global.copy()
    measured_global[..., 2] += 0.010
    reference_local = np.zeros((row_count, point_count, 3), dtype=np.float64)
    measured_local = reference_local.copy()
    measured_local[..., 2] += 0.010

    active_mask = np.zeros((row_count, site_count), dtype=np.bool_)
    force = np.zeros((row_count, site_count, 3), dtype=np.float64)
    compliance = np.zeros((row_count, site_count, 3), dtype=np.float64)
    actions = np.zeros((row_count, 5), dtype=np.float64)
    for site_id in active_sites:
        site_index = SITE_IDS.index(site_id)
        point_index = POINT_IDS.index(site_id)
        active_mask[1:5, site_index] = True
        force[1:5, site_index, 0] = 5.0
        compliance[1:5, site_index, 0] = 0.02
        selected_site[1:5, site_index] -= (
            compliance[1:5, site_index] * force[1:5, site_index]
        )
        if residual:
            measured_site[1:5, site_index, 0] += 0.002
            measured_global[1:5, point_index, 0] += 0.002
            measured_local[1:5, point_index, 0] += 0.002
            actions[1:5, site_index] = 0.001

    terminal = np.zeros(row_count, dtype=np.bool_)
    terminal[-1] = True
    success = terminal.copy()
    reset = np.zeros(row_count, dtype=np.bool_)
    reset[0] = True
    return EvaluationTrace(
        trial_name=trial_name,
        motion_ids=("review_motion",) * row_count,
        sequence_ids=("sequence_0",) * row_count,
        seed_ids=np.zeros(row_count, dtype=np.int64),
        frame_indices=np.arange(row_count, dtype=np.int64),
        timestamps_s=np.arange(row_count, dtype=np.float64) / 50.0,
        site_ids=SITE_IDS,
        point_ids=POINT_IDS,
        original_site_positions_m=original_site,
        selected_site_positions_m=selected_site,
        measured_site_positions_m=measured_site,
        original_site_orientations_xyzw=original_orientation,
        measured_site_orientations_xyzw=measured_orientation,
        reference_points_global_m=reference_global,
        measured_points_global_m=measured_global,
        reference_points_local_m=reference_local,
        measured_points_local_m=measured_local,
        force_on_robot_n=force,
        force_on_robot_world_n=force,
        force_on_robot_common_n=force,
        compliance_m_per_n=compliance,
        compliance_enabled=np.full(row_count, enabled, dtype=np.bool_),
        residual_enabled=np.full(row_count, residual, dtype=np.bool_),
        active_site_mask=active_mask,
        policy_actions=actions,
        terminal_mask=terminal,
        success_mask=success,
        fall_mask=np.zeros(row_count, dtype=np.bool_),
        reset_mask=reset,
    )


def _matched_suite_inputs():
    traces = {
        "release_baseline": _make_matched_trace("release_baseline"),
        "chip_hard_off": _make_matched_trace("chip_hard_off"),
        "enabled_no_contact": _make_matched_trace(
            "enabled_no_contact", enabled=True, residual=True
        ),
    }
    interactions = []
    for name, active_sites in (
        ("single_left", ("endpoint_a",)),
        ("single_right", ("endpoint_b",)),
        ("simultaneous", ("endpoint_a", "endpoint_b")),
    ):
        reference_name = f"{name}_stiff"
        candidate_name = f"{name}_compliant"
        traces[reference_name] = _make_matched_trace(
            reference_name,
            active_sites=active_sites,
            enabled=True,
        )
        traces[candidate_name] = _make_matched_trace(
            candidate_name,
            active_sites=active_sites,
            enabled=True,
            residual=True,
        )
        interactions.append(
            MatchedInteractionSpec(
                name=name,
                reference_trial=reference_name,
                candidate_trial=candidate_name,
                active_site_ids=active_sites,
            )
        )
    spec = ReviewSuiteSpec(
        baseline_trial="release_baseline",
        hard_off_trial="chip_hard_off",
        no_contact_trial="enabled_no_contact",
        interactions=tuple(interactions),
    )
    criteria = RegressionCriteria(
        endpoint_site_ids=("endpoint_a", "endpoint_b"),
        endpoint_tracking_point_ids=("endpoint_a", "endpoint_b"),
    )
    assert set(traces) == MATCHED_ROLES
    return traces, spec, criteria


def _failed_checks(report: dict[str, object]) -> set[str]:
    acceptance = report["acceptance"]
    assert isinstance(acceptance, dict)
    checks = acceptance["checks"]
    assert isinstance(checks, list)
    return {str(check["name"]) for check in checks if not check["passed"]}


def test_trace_reports_per_site_pose_force_yield_lifecycle_and_finiteness():
    trace = _make_trace(
        "active_trial",
        endpoint_error_m=0.012,
        enabled=True,
        active_sites=("endpoint_a",),
        fall=True,
    )
    report = evaluate_trace(trace)

    assert report["row_count"] == 8
    assert report["site_count"] == len(SITE_IDS)
    assert report["point_count"] == len(POINT_IDS)
    assert report["sites"]["endpoint_a"]["original_endpoint_error_m"]["rmse"] == pytest.approx(
        0.013
    )
    assert report["sites"]["endpoint_a"]["orientation_error_rad"]["p95"] == pytest.approx(
        0.1
    )
    assert report["sites"]["endpoint_a"]["force_norm_n"]["peak"] == pytest.approx(4.0)
    assert report["sites"]["endpoint_a"]["active_force_norm_n"]["count"] == 2
    assert report["sites"]["endpoint_a"]["active_original_endpoint_error_m"][
        "p95"
    ] == pytest.approx(np.hypot(0.012, 0.01))
    assert report["sites"]["endpoint_a"]["reference_yield_m"]["peak"] == pytest.approx(
        0.02
    )
    assert report["sites"]["endpoint_b"]["inactive_force_norm_n"]["peak"] == 0.0
    assert report["global_mpjpe_m"]["mean"] == pytest.approx(0.012)
    assert report["local_mpjpe_m"]["p95"] == pytest.approx(0.012)
    assert report["lifecycle"] == {
        "terminal_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
        "fall_count": 1,
        "reset_count": 2,
        "post_reset_force_peak_n": 0.0,
    }
    assert report["activation"]["simultaneous_active_peak"] == 1
    assert report["validity"] == {
        "all_finite": True,
        "nonfinite_count": 0,
        "nonfinite_fields": {},
        "derived_nonfinite_count": 0,
        "invalid_orientation_count": 0,
        "all_valid": True,
    }


def test_strict_alignment_rejects_motion_seed_frame_timestamp_and_layout_changes():
    reference = _make_trace("reference", endpoint_error_m=0.01)
    aligned = _make_trace("candidate", endpoint_error_m=0.02)
    assert_strict_alignment(reference, aligned)
    assert alignment_digest(reference) == alignment_digest(aligned)

    changed_reference_global = aligned.reference_points_global_m.copy()
    changed_reference_global.flat[0] = np.nextafter(changed_reference_global.flat[0], 1.0)
    changed_reference_local = aligned.reference_points_local_m.copy()
    changed_reference_local.flat[0] = np.nextafter(changed_reference_local.flat[0], 1.0)
    changed_original_site = aligned.original_site_positions_m.copy()
    changed_original_site.flat[0] = np.nextafter(changed_original_site.flat[0], 1.0)
    changed_original_orientation = aligned.original_site_orientations_xyzw.copy()
    changed_original_orientation.flat[0] = np.nextafter(
        changed_original_orientation.flat[0],
        1.0,
    )
    mutations = {
        "motion_ids": ("different",) * 4 + aligned.motion_ids[4:],
        "sequence_ids": ("different",) * 4 + aligned.sequence_ids[4:],
        "seed_ids": np.array([99, 99, 99, 99, *aligned.seed_ids[4:]], dtype=np.int64),
        "frame_indices": np.array([*aligned.frame_indices[:-1], 4], dtype=np.int64),
        "timestamps_s": np.array(
            [np.nextafter(aligned.timestamps_s[0], 1.0), *aligned.timestamps_s[1:]],
            dtype=np.float64,
        ),
        "site_ids": ("different",) + aligned.site_ids[1:],
        "point_ids": ("different",) + aligned.point_ids[1:],
        "reference_points_global_m": changed_reference_global,
        "reference_points_local_m": changed_reference_local,
        "original_site_positions_m": changed_original_site,
        "original_site_orientations_xyzw": changed_original_orientation,
    }
    for field_name, value in mutations.items():
        candidate = replace(aligned, **{field_name: value})
        with pytest.raises(TraceAlignmentError, match=field_name):
            assert_strict_alignment(reference, candidate)

    signed_zero_timestamps = aligned.timestamps_s.copy()
    signed_zero_timestamps[0] = -0.0
    signed_zero = replace(aligned, timestamps_s=signed_zero_timestamps)
    with pytest.raises(TraceAlignmentError, match="timestamps_s"):
        assert_strict_alignment(reference, signed_zero)
    assert alignment_digest(reference) != alignment_digest(signed_zero)


def test_trace_schema_rejects_ambiguous_rows_and_invalid_events():
    trace = _make_trace("valid", endpoint_error_m=0.01)
    with pytest.raises(ValueError, match="unique|increase"):
        replace(trace, frame_indices=np.zeros_like(trace.frame_indices))
    with pytest.raises(ValueError, match="increase"):
        replace(trace, timestamps_s=np.zeros_like(trace.timestamps_s))
    bad_success = trace.success_mask.copy()
    bad_success[1] = True
    with pytest.raises(ValueError, match="subset"):
        replace(trace, success_mask=bad_success)
    overlapping_fall = trace.fall_mask.copy()
    overlapping_fall[3] = True
    with pytest.raises(ValueError, match="disjoint"):
        replace(trace, fall_mask=overlapping_fall)
    nonterminal_fall = trace.fall_mask.copy()
    nonterminal_fall[1] = True
    with pytest.raises(ValueError, match="subset"):
        replace(trace, fall_mask=nonterminal_fall)
    bad_active = trace.active_site_mask.copy()
    bad_active[1, 0] = True
    with pytest.raises(ValueError, match="require compliance_enabled"):
        replace(trace, active_site_mask=bad_active)
    with pytest.raises(ValueError, match="duplicates"):
        replace(trace, site_ids=("duplicate", "duplicate", "other"))


def test_suite_reports_all_protocols_cross_coupling_and_acceptance():
    traces, specs, criteria = _suite_inputs()
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )

    assert report["schema_version"] == "compliance_evaluation_v2"
    assert report["acceptance"]["passed"] is True
    assert set(report["paired_to_baseline"]) == set(traces) - {"released_baseline"}
    simultaneous = report["trials"]["simultaneous"]
    assert simultaneous["activation"]["simultaneous_active_peak"] == 2
    assert simultaneous["sites"]["endpoint_a"]["force_norm_n"]["peak"] == 4.0
    pair = report["paired_to_baseline"]["single_a"]
    assert pair["sites"]["endpoint_b"]["inactive_cross_coupling_shift_m"]["rmse"] == pytest.approx(
        0.002
    )
    assert pair["paired_global_pose_shift_m"]["mean"] == pytest.approx(0.002)
    off_pair = report["paired_to_off"]["single_a"]
    assert off_pair["sites"]["endpoint_b"]["inactive_cross_coupling_shift_m"][
        "rmse"
    ] == pytest.approx(0.001)
    active_tracking = report["active_tracking_to_off"]["single_a"]
    assert active_tracking["interaction_row_count"] == 2
    assert active_tracking["invariant_point_sample_count"] == 12
    assert active_tracking["sites"]["endpoint_a"]["tracking_point_id"] == "endpoint_a"
    assert active_tracking["sites"]["endpoint_a"][
        "selected_endpoint_rmse_regression_m"
    ] == pytest.approx(np.hypot(0.012, 0.01) - 0.011)
    check_names = {check["name"] for check in report["acceptance"]["checks"]}
    assert "off_success_rate_drop" in check_names
    assert "off_local_mpjpe_regression_m" in check_names
    assert "off_endpoint_rmse_regression_m:endpoint_a" in check_names
    assert "no_contact_endpoint_rmse_delta_m:enabled_no_contact:endpoint_b" in check_names
    assert "post_reset_force_peak_n:simultaneous" in check_names
    assert "inactive_force_peak_n:overlay_off:endpoint_a" in check_names
    assert (
        "active_selected_endpoint_rmse_regression_m:single_a:endpoint_a"
        in check_names
    )
    assert "active_orientation_rmse_regression_rad:single_a:endpoint_a" in check_names
    assert "active_invariant_local_mpjpe_regression_m:simultaneous" in check_names
    assert "active_invariant_global_mpjpe_regression_m:simultaneous" in check_names


def test_inactive_force_or_yield_fails_off_mode_acceptance():
    traces, specs, criteria = _suite_inputs()
    off = traces["overlay_off"]
    stale_force = off.force_on_robot_n.copy()
    stale_force[2, 0, 0] = 0.1
    traces["overlay_off"] = replace(off, force_on_robot_n=stale_force)
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "inactive_force_peak_n:overlay_off:endpoint_a" in failed
    assert report["acceptance"]["passed"] is False

    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    cross_force = single.force_on_robot_n.copy()
    cross_force[1, SITE_IDS.index("endpoint_b"), 0] = 9.0
    traces["single_a"] = replace(single, force_on_robot_n=cross_force)
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "inactive_force_peak_n:single_a:endpoint_b" in failed

    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    cross_yield = single.selected_site_positions_m.copy()
    cross_yield[:, SITE_IDS.index("endpoint_b"), 0] += 0.01
    traces["single_a"] = replace(single, selected_site_positions_m=cross_yield)
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "inactive_yield_peak_m:single_a:endpoint_b" in failed

    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    traces["single_a"] = replace(
        single,
        force_on_robot_n=np.zeros_like(single.force_on_robot_n),
        selected_site_positions_m=single.original_site_positions_m.copy(),
    )
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "active_force_peak_n:single_a:endpoint_a" in failed
    assert "active_yield_peak_m:single_a:endpoint_a" in failed


def test_interaction_requires_measured_yield_along_actual_force():
    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    traces["single_a"] = replace(
        single,
        measured_site_positions_m=traces["overlay_off"].measured_site_positions_m.copy(),
    )

    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "active_measured_yield_peak_m:single_a:endpoint_a" in failed
    assert "active_measured_yield_along_force_peak_m:single_a:endpoint_a" in failed


def test_matched_suite_requires_signed_chip_hindsight_target_separately_from_yield():
    traces, spec, criteria = _matched_suite_inputs()
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    assert report["acceptance"]["passed"] is True

    reference = traces["single_left_stiff"]
    candidate = traces["single_left_compliant"]
    selected = candidate.selected_site_positions_m.copy()
    active = candidate.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    selected[active, SITE_IDS.index("endpoint_a"), 0] *= -1.0
    traces["single_left_stiff"] = replace(
        reference,
        selected_site_positions_m=selected,
    )
    traces["single_left_compliant"] = replace(
        candidate,
        selected_site_positions_m=selected,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    assert (
        "signed_hindsight_target_peak_m:single_left:candidate:endpoint_a"
        in _failed_checks(report)
    )


def test_inactive_hand_measured_cross_coupling_fails_rmse_and_p95():
    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    measured = single.measured_site_positions_m.copy()
    measured[:, SITE_IDS.index("endpoint_b"), 0] += 0.02
    traces["single_a"] = replace(single, measured_site_positions_m=measured)

    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "inactive_cross_coupling_rmse_m:single_a:endpoint_b" in failed
    assert "inactive_cross_coupling_p95_m:single_a:endpoint_b" in failed


def test_active_selected_endpoint_and_orientation_regressions_fail_closed():
    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    measured = single.measured_site_positions_m.copy()
    active = single.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    measured[active, SITE_IDS.index("endpoint_a"), 2] += 0.03
    traces["single_a"] = replace(single, measured_site_positions_m=measured)
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "active_selected_endpoint_rmse_regression_m:single_a:endpoint_a" in failed
    assert "active_selected_endpoint_p95_regression_m:single_a:endpoint_a" in failed

    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    measured_orientation = single.measured_site_orientations_xyzw.copy()
    active = single.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    angle = 0.35
    measured_orientation[active, SITE_IDS.index("endpoint_a"), :] = (
        np.sin(angle / 2.0),
        0.0,
        0.0,
        np.cos(angle / 2.0),
    )
    traces["single_a"] = replace(
        single,
        measured_site_orientations_xyzw=measured_orientation,
    )
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "active_orientation_rmse_regression_rad:single_a:endpoint_a" in failed
    assert "active_orientation_p95_regression_rad:single_a:endpoint_a" in failed


def test_active_invariant_whole_body_tracking_excludes_only_yielded_points():
    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    active_rows = np.any(single.active_site_mask, axis=1)
    measured_global = single.measured_points_global_m.copy()
    measured_local = single.measured_points_local_m.copy()
    measured_global[active_rows, 2:, 0] += 0.05
    measured_local[active_rows, 2:, 0] += 0.05
    traces["single_a"] = replace(
        single,
        measured_points_global_m=measured_global,
        measured_points_local_m=measured_local,
    )
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "active_invariant_local_mpjpe_regression_m:single_a" in failed
    assert "active_invariant_global_mpjpe_regression_m:single_a" in failed

    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    active_rows = single.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    measured_global = single.measured_points_global_m.copy()
    measured_local = single.measured_points_local_m.copy()
    point_index = POINT_IDS.index("endpoint_a")
    measured_global[active_rows, point_index, 0] += 1.0
    measured_local[active_rows, point_index, 0] += 1.0
    traces["single_a"] = replace(
        single,
        measured_points_global_m=measured_global,
        measured_points_local_m=measured_local,
    )
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "active_invariant_local_mpjpe_regression_m:single_a" not in failed
    assert "active_invariant_global_mpjpe_regression_m:single_a" not in failed


def test_active_tracking_point_mapping_is_explicit_and_validated():
    with pytest.raises(ValueError, match="one-to-one"):
        RegressionCriteria(
            endpoint_site_ids=("endpoint_a", "endpoint_b"),
            endpoint_tracking_point_ids=("endpoint_a",),
        )
    traces, specs, _ = _suite_inputs()
    criteria = RegressionCriteria(
        endpoint_site_ids=("endpoint_a", "endpoint_b"),
        endpoint_tracking_point_ids=("endpoint_a", "missing_point"),
    )
    with pytest.raises(ValueError, match="must exist"):
        evaluate_trial_suite(
            traces,
            specs,
            baseline_name="released_baseline",
            criteria=criteria,
        )


def test_each_formal_trial_requires_no_fall_and_full_success():
    traces, specs, criteria = _suite_inputs()
    single = traces["single_a"]
    success = single.success_mask.copy()
    fall = single.fall_mask.copy()
    success[-1] = False
    fall[-1] = True
    traces["single_a"] = replace(single, success_mask=success, fall_mask=fall)

    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    failed = {
        check["name"]
        for check in report["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "zero_falls:single_a" in failed
    assert "success_rate_one:single_a" in failed


def test_suite_requires_complete_protocol_enabled_rows_and_reset_evidence():
    traces, specs, criteria = _suite_inputs()
    incomplete_specs = tuple(spec for spec in specs if spec.mode is not TrialMode.NO_CONTACT)
    incomplete_traces = {spec.name: traces[spec.name] for spec in incomplete_specs}
    with pytest.raises(ValueError, match="no-contact"):
        evaluate_trial_suite(
            incomplete_traces,
            incomplete_specs,
            baseline_name="released_baseline",
            criteria=criteria,
        )

    traces, specs, criteria = _suite_inputs()
    no_contact = traces["enabled_no_contact"]
    partly_enabled = no_contact.compliance_enabled.copy()
    partly_enabled[-1] = False
    traces["enabled_no_contact"] = replace(
        no_contact,
        compliance_enabled=partly_enabled,
        residual_enabled=partly_enabled,
    )
    with pytest.raises(ValueError, match="every row"):
        evaluate_trial_suite(
            traces,
            specs,
            baseline_name="released_baseline",
            criteria=criteria,
        )

    traces, specs, criteria = _suite_inputs()
    trace = traces["released_baseline"]
    with pytest.raises(ValueError, match="reset snapshot"):
        replace(trace, reset_mask=np.zeros_like(trace.reset_mask))
    with pytest.raises(ValueError, match="terminate exactly once"):
        replace(
            trace,
            terminal_mask=np.zeros_like(trace.terminal_mask),
            success_mask=np.zeros_like(trace.success_mask),
        )


def test_protocol_validation_rejects_wrong_single_and_no_contact_activation():
    traces, specs, criteria = _suite_inputs()
    traces["enabled_no_contact"] = _make_trace(
        "enabled_no_contact",
        endpoint_error_m=0.01,
        enabled=True,
        active_sites=("endpoint_a",),
    )
    with pytest.raises(ValueError, match="must not activate"):
        evaluate_trial_suite(
            traces,
            specs,
            baseline_name="released_baseline",
            criteria=criteria,
        )

    traces, specs, criteria = _suite_inputs()
    traces["single_a"] = _make_trace(
        "single_a",
        endpoint_error_m=0.01,
        enabled=True,
        active_sites=("endpoint_b",),
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_trial_suite(
            traces,
            specs,
            baseline_name="released_baseline",
            criteria=criteria,
        )


def test_nonfinite_physics_is_reported_and_fails_acceptance_without_crashing():
    trace = _make_trace(
        "nonfinite",
        endpoint_error_m=0.01,
        enabled=True,
        active_sites=("endpoint_a",),
        nonfinite=True,
    )
    report = evaluate_trace(trace)
    assert report["validity"]["all_finite"] is False
    assert report["validity"]["nonfinite_fields"] == {"measured_site_positions_m": 1}
    assert report["sites"]["endpoint_a"]["original_endpoint_error_m"]["finite_count"] == 7

    traces, specs, criteria = _suite_inputs(nonfinite_trial="single_a")
    suite = evaluate_trial_suite(
        traces,
        specs,
        baseline_name="released_baseline",
        criteria=criteria,
    )
    assert suite["acceptance"]["passed"] is False
    failed = {
        check["name"]
        for check in suite["acceptance"]["checks"]
        if not check["passed"]
    }
    assert "finite_and_valid:single_a" in failed


def test_paired_comparison_rejects_even_one_timestamp_bit_change():
    reference = _make_trace("reference", endpoint_error_m=0.01)
    candidate = _make_trace("candidate", endpoint_error_m=0.02)
    timestamps = candidate.timestamps_s.copy()
    timestamps[2] = np.nextafter(timestamps[2], np.inf)
    candidate = replace(candidate, timestamps_s=timestamps)
    with pytest.raises(TraceAlignmentError, match="timestamps_s"):
        compare_aligned_traces(reference, candidate)


def test_atomic_bounded_json_and_npz_round_trip(tmp_path: Path):
    trace = _make_trace("round_trip", endpoint_error_m=0.01)
    trace_path = tmp_path / "trace.npz"
    report_path = tmp_path / "report.json"

    write_trace_npz_atomic(trace, trace_path)
    loaded = load_trace_npz(trace_path)
    loaded_with_hash, trace_sha256 = load_trace_npz_with_sha256(trace_path)
    assert alignment_digest(loaded) == alignment_digest(trace)
    assert alignment_digest(loaded_with_hash) == alignment_digest(trace)
    assert trace_sha256 == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(loaded.measured_site_positions_m, trace.measured_site_positions_m)
    assert loaded.trial_name == trace.trial_name
    with pytest.raises(FileExistsError):
        write_trace_npz_atomic(trace, trace_path)
    symlink_path = tmp_path / "trace_link.npz"
    symlink_path.symlink_to(trace_path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_trace_npz(symlink_path)

    report = evaluate_trace(trace)
    write_report_json_atomic(report, report_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    loaded_report, report_sha256 = load_report_json_with_sha256(report_path)
    assert loaded_report == report
    assert report_sha256 == hashlib.sha256(report_path.read_bytes()).hexdigest()
    report_symlink = tmp_path / "report_link.json"
    report_symlink.symlink_to(report_path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_report_json_with_sha256(report_symlink)
    with pytest.raises(ValueError, match="max_bytes"):
        load_report_json_with_sha256(report_path, max_bytes=8)
    with pytest.raises(FileExistsError):
        write_report_json_atomic(report, report_path)
    with pytest.raises(ValueError, match="max_bytes"):
        write_report_json_atomic(report, tmp_path / "too_small.json", max_bytes=8)
    with pytest.raises(ValueError, match="max_bytes"):
        write_trace_npz_atomic(trace, tmp_path / "too_small.npz", max_bytes=8)
    assert not list(tmp_path.glob(".*.tmp"))


def test_portable_evaluation_has_no_simulator_robot_or_fixed_layout_dependency():
    forbidden_import_roots = {"isaaclab"}
    forbidden_tokens = (
        "sonic",
        "g1",
        "isaaclab",
        "mujoco",
        "left_wrist",
        "right_wrist",
    )
    for path in REVIEW_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", maxsplit=1)[0] for alias in node.names}
                assert roots.isdisjoint(forbidden_import_roots), path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", maxsplit=1)[0] not in forbidden_import_roots, path
            elif isinstance(node, ast.Constant) and type(node.value) is int:
                assert node.value not in {14, 29}, path
        lowered = source.lower()
        assert all(token not in lowered for token in forbidden_tokens), path


@pytest.mark.parametrize("point_count", [2, 7, 17])
def test_evaluation_accepts_caller_owned_tracking_point_layouts(point_count: int):
    trace = _make_trace("layout", endpoint_error_m=0.01)
    point_ids = tuple(f"point_{index}" for index in range(point_count))
    shape = (len(trace.motion_ids), point_count, 3)
    zeros = np.zeros(shape, dtype=np.float32)
    trace = replace(
        trace,
        point_ids=point_ids,
        reference_points_global_m=zeros,
        measured_points_global_m=zeros,
        reference_points_local_m=zeros,
        measured_points_local_m=zeros,
    )
    report = evaluate_trace(trace)
    assert report["point_count"] == point_count
    assert report["global_mpjpe_m"]["mean"] == 0.0


def test_matched_review_suite_passes_all_nine_tracking_first_roles():
    traces, spec, criteria = _matched_suite_inputs()
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)

    assert spec.trial_names == (
        "release_baseline",
        "chip_hard_off",
        "enabled_no_contact",
        "single_left_stiff",
        "single_left_compliant",
        "single_right_stiff",
        "single_right_compliant",
        "simultaneous_stiff",
        "simultaneous_compliant",
    )
    assert report["schema_version"] == "compliance_review_v1"
    assert report["acceptance"]["passed"] is True
    assert _failed_checks(report) == set()
    assert report["interactions"]["simultaneous"]["active_site_ids"] == [
        "endpoint_a",
        "endpoint_b",
    ]
    check_names = {
        check["name"] for check in report["acceptance"]["checks"]
    }
    assert {
        "exact_policy_actions:release_to_hard_off",
        "exact_policy_actions:hard_off_to_no_contact",
        "endpoint_rmse_regression_m:hard_off:endpoint_a",
        "endpoint_rmse_regression_m:no_contact:endpoint_b",
        "active_selected_endpoint_rmse_regression_m:single_left:endpoint_a",
        "active_orientation_p95_regression_rad:single_right:endpoint_b",
        "active_invariant_local_mpjpe_regression_m:simultaneous",
        "active_invariant_global_mpjpe_regression_m:simultaneous",
        "active_measured_yield_along_force_peak_m:single_left:endpoint_a",
        "inactive_cross_coupling_p95_m:single_left:endpoint_b",
        "residual_action_activation:simultaneous",
        "exact_zero_force:release_baseline",
        "exact_zero_compliance:chip_hard_off",
        "exact_zero_reference_yield:enabled_no_contact",
    }.issubset(check_names)


def test_matched_review_suite_requires_exact_roles_and_matched_stimulus():
    traces, spec, criteria = _matched_suite_inputs()
    missing = dict(traces)
    missing.pop("single_right_compliant")
    with pytest.raises(ValueError, match="missing"):
        evaluate_matched_review_suite(missing, spec, criteria=criteria)

    extra = dict(traces)
    extra["unexpected"] = replace(
        traces["chip_hard_off"], trial_name="unexpected"
    )
    with pytest.raises(ValueError, match="extra"):
        evaluate_matched_review_suite(extra, spec, criteria=criteria)

    candidate = traces["single_left_compliant"]
    mutations = {}
    for field_name in (
        "selected_site_positions_m",
        "force_on_robot_n",
        "force_on_robot_world_n",
        "compliance_m_per_n",
        "active_site_mask",
    ):
        changed = np.array(getattr(candidate, field_name), copy=True)
        changed.flat[-1] = not changed.flat[-1] if changed.dtype.kind == "b" else 1.0
        mutations[field_name] = changed
    for field_name, changed in mutations.items():
        changed_traces = dict(traces)
        changed_traces["single_left_compliant"] = replace(
            candidate,
            **{field_name: changed},
        )
        with pytest.raises(TraceAlignmentError, match=field_name):
            evaluate_matched_review_suite(changed_traces, spec, criteria=criteria)


def test_matched_review_suite_fails_closed_for_tracking_and_contact_regressions():
    traces, spec, criteria = _matched_suite_inputs()
    hard_off = traces["chip_hard_off"]
    measured_site = hard_off.measured_site_positions_m.copy()
    measured_site[:, SITE_IDS.index("endpoint_a"), 2] += 0.006
    measured_local = hard_off.measured_points_local_m.copy()
    measured_local[..., 1] += 0.020
    measured_global = hard_off.measured_points_global_m.copy()
    measured_global[..., 1] += 0.020
    traces["chip_hard_off"] = replace(
        hard_off,
        measured_site_positions_m=measured_site,
        measured_points_local_m=measured_local,
        measured_points_global_m=measured_global,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "endpoint_rmse_regression_m:hard_off:endpoint_a" in failed
    assert "local_mpjpe_regression_m:hard_off" in failed
    assert "global_mpjpe_regression_m:hard_off" in failed

    traces, spec, criteria = _matched_suite_inputs()
    no_contact = traces["enabled_no_contact"]
    measured = no_contact.measured_site_positions_m.copy()
    measured[:, SITE_IDS.index("endpoint_b"), 2] += 0.006
    actions = no_contact.policy_actions.copy()
    actions[2, 0] = 1.0e-6
    traces["enabled_no_contact"] = replace(
        no_contact,
        measured_site_positions_m=measured,
        policy_actions=actions,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "endpoint_rmse_regression_m:no_contact:endpoint_b" in failed
    assert "exact_policy_actions:hard_off_to_no_contact" in failed

    traces, spec, criteria = _matched_suite_inputs()
    reference = traces["single_left_stiff"]
    candidate = traces["single_left_compliant"]
    active = candidate.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    measured = candidate.measured_site_positions_m.copy()
    measured[active, SITE_IDS.index("endpoint_a"), 0] += 0.013
    orientation = candidate.measured_site_orientations_xyzw.copy()
    orientation[active, SITE_IDS.index("endpoint_a"), :] = (
        0.0,
        np.sin(0.07),
        0.0,
        np.cos(0.07),
    )
    measured_local = candidate.measured_points_local_m.copy()
    measured_global = candidate.measured_points_global_m.copy()
    measured_local[active, 2:, 1] += 0.020
    measured_global[active, 2:, 1] += 0.020
    measured[:, SITE_IDS.index("endpoint_b"), 1] += 0.011
    traces["single_left_compliant"] = replace(
        candidate,
        measured_site_positions_m=measured,
        measured_site_orientations_xyzw=orientation,
        measured_points_local_m=measured_local,
        measured_points_global_m=measured_global,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "active_selected_endpoint_rmse_regression_m:single_left:endpoint_a" in failed
    assert "active_selected_endpoint_p95_regression_m:single_left:endpoint_a" in failed
    assert "active_orientation_rmse_regression_rad:single_left:endpoint_a" in failed
    assert "active_orientation_p95_regression_rad:single_left:endpoint_a" in failed
    assert "active_invariant_local_mpjpe_regression_m:single_left" in failed
    assert "active_invariant_global_mpjpe_regression_m:single_left" in failed
    assert "inactive_cross_coupling_rmse_m:single_left:endpoint_b" in failed
    assert "inactive_cross_coupling_p95_m:single_left:endpoint_b" in failed
    assert reference.trial_name == "single_left_stiff"


def test_matched_review_suite_fails_contact_direction_action_lifecycle_and_finiteness():
    traces, spec, criteria = _matched_suite_inputs()
    hard_off = traces["chip_hard_off"]
    compliance = hard_off.compliance_m_per_n.copy()
    compliance[1, 0, 0] = np.nextafter(0.0, 1.0)
    traces["chip_hard_off"] = replace(
        hard_off,
        compliance_m_per_n=compliance,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    assert "exact_zero_compliance:chip_hard_off" in _failed_checks(report)

    traces, spec, criteria = _matched_suite_inputs()
    candidate = traces["single_left_compliant"]
    active = candidate.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    measured = candidate.measured_site_positions_m.copy()
    reference_measured = traces["single_left_stiff"].measured_site_positions_m
    measured[active, SITE_IDS.index("endpoint_a")] = reference_measured[
        active, SITE_IDS.index("endpoint_a")
    ]
    traces["single_left_compliant"] = replace(
        candidate,
        measured_site_positions_m=measured,
        policy_actions=traces["single_left_stiff"].policy_actions,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "active_measured_yield_peak_m:single_left:endpoint_a" in failed
    assert "active_measured_yield_along_force_peak_m:single_left:endpoint_a" in failed
    assert "residual_action_activation:single_left" in failed

    traces, spec, criteria = _matched_suite_inputs()
    candidate = traces["single_left_compliant"]
    active = candidate.active_site_mask[:, SITE_IDS.index("endpoint_a")]
    measured = candidate.measured_site_positions_m.copy()
    reference_measured = traces["single_left_stiff"].measured_site_positions_m
    measured[active, SITE_IDS.index("endpoint_a"), 0] = (
        reference_measured[active, SITE_IDS.index("endpoint_a"), 0] - 0.002
    )
    traces["single_left_compliant"] = replace(
        candidate,
        measured_site_positions_m=measured,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    assert (
        "active_measured_yield_along_force_peak_m:single_left:endpoint_a"
        in _failed_checks(report)
    )

    traces, spec, criteria = _matched_suite_inputs()
    candidate = traces["simultaneous_compliant"]
    success = candidate.success_mask.copy()
    fall = candidate.fall_mask.copy()
    success[-1] = False
    fall[-1] = True
    measured = candidate.measured_site_positions_m.copy()
    measured[..., 0] = np.nan
    traces["simultaneous_compliant"] = replace(
        candidate,
        success_mask=success,
        fall_mask=fall,
        measured_site_positions_m=measured,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "full_success:simultaneous_compliant" in failed
    assert "zero_falls:simultaneous_compliant" in failed
    assert "finite:simultaneous_compliant" in failed
    assert report["acceptance"]["passed"] is False


def test_matched_review_suite_enforces_reset_inactive_and_active_minima():
    traces, spec, criteria = _matched_suite_inputs()
    reference = traces["single_left_stiff"]
    candidate = traces["single_left_compliant"]
    force = reference.force_on_robot_n.copy()
    force[0, SITE_IDS.index("endpoint_a"), 0] = 2.0e-6
    traces["single_left_stiff"] = replace(reference, force_on_robot_n=force)
    traces["single_left_compliant"] = replace(candidate, force_on_robot_n=force)
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    assert "reset_force_peak_n:single_left_stiff" in _failed_checks(report)
    assert "reset_force_peak_n:single_left_compliant" in _failed_checks(report)

    traces, spec, criteria = _matched_suite_inputs()
    reference = traces["single_left_stiff"]
    candidate = traces["single_left_compliant"]
    force = reference.force_on_robot_n.copy()
    selected = reference.selected_site_positions_m.copy()
    inactive_index = SITE_IDS.index("endpoint_b")
    force[2, inactive_index, 0] = 2.0e-6
    selected[2, inactive_index, 0] += 2.0e-9
    for role, trace in (
        ("single_left_stiff", reference),
        ("single_left_compliant", candidate),
    ):
        traces[role] = replace(
            trace,
            force_on_robot_n=force,
            selected_site_positions_m=selected,
        )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "inactive_force_peak_n:single_left:endpoint_b" in failed
    assert "inactive_yield_peak_m:single_left:endpoint_b" in failed

    traces, spec, criteria = _matched_suite_inputs()
    reference = traces["single_left_stiff"]
    candidate = traces["single_left_compliant"]
    active_index = SITE_IDS.index("endpoint_a")
    active = reference.active_site_mask[:, active_index]
    weak_force = reference.force_on_robot_n.copy()
    weak_force[active, active_index, 0] = 5.0e-7
    tiny_yield = reference.selected_site_positions_m.copy()
    tiny_yield[active, active_index, 0] = (
        reference.original_site_positions_m[active, active_index, 0] + 5.0e-10
    )
    for role, trace in (
        ("single_left_stiff", reference),
        ("single_left_compliant", candidate),
    ):
        traces[role] = replace(
            trace,
            force_on_robot_n=weak_force,
            selected_site_positions_m=tiny_yield,
        )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "active_force_peak_n:single_left:endpoint_a" in failed
    assert "active_reference_yield_peak_m:single_left:endpoint_a" in failed


def test_relative_tracking_limits_are_used_when_larger_than_absolute_limits():
    traces, spec, criteria = _matched_suite_inputs()
    baseline = traces["release_baseline"]
    hard_off = traces["chip_hard_off"]
    no_contact = traces["enabled_no_contact"]
    baseline_local = baseline.reference_points_local_m.copy()
    baseline_global = baseline.reference_points_global_m.copy()
    baseline_local[..., 2] += 0.100
    baseline_global[..., 2] += 0.100
    passing_local = hard_off.reference_points_local_m.copy()
    passing_global = hard_off.reference_points_global_m.copy()
    passing_local[..., 2] += 0.109
    passing_global[..., 2] += 0.109
    no_contact_local = no_contact.reference_points_local_m.copy()
    no_contact_global = no_contact.reference_points_global_m.copy()
    no_contact_local[..., 2] += 0.109
    no_contact_global[..., 2] += 0.109
    traces["release_baseline"] = replace(
        baseline,
        measured_points_local_m=baseline_local,
        measured_points_global_m=baseline_global,
    )
    traces["chip_hard_off"] = replace(
        hard_off,
        measured_points_local_m=passing_local,
        measured_points_global_m=passing_global,
    )
    traces["enabled_no_contact"] = replace(
        no_contact,
        measured_points_local_m=no_contact_local,
        measured_points_global_m=no_contact_global,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "local_mpjpe_regression_m:hard_off" not in failed
    assert "global_mpjpe_regression_m:hard_off" not in failed

    failing_local = hard_off.reference_points_local_m.copy()
    failing_global = hard_off.reference_points_global_m.copy()
    failing_local[..., 2] += 0.111
    failing_global[..., 2] += 0.111
    traces["chip_hard_off"] = replace(
        traces["chip_hard_off"],
        measured_points_local_m=failing_local,
        measured_points_global_m=failing_global,
    )
    report = evaluate_matched_review_suite(traces, spec, criteria=criteria)
    failed = _failed_checks(report)
    assert "local_mpjpe_regression_m:hard_off" in failed
    assert "global_mpjpe_regression_m:hard_off" in failed


def test_matched_review_suite_accepts_caller_owned_extra_sites_and_actions():
    traces, spec, criteria = _matched_suite_inputs()
    expanded = {}
    extra_count = 4
    for name, trace in traces.items():
        rows = len(trace.motion_ids)
        extra_vectors = np.zeros((rows, extra_count, 3), dtype=np.float64)
        extra_quaternions = np.zeros((rows, extra_count, 4), dtype=np.float64)
        extra_quaternions[..., 3] = 1.0
        expanded[name] = replace(
            trace,
            site_ids=trace.site_ids
            + tuple(f"auxiliary_{index}" for index in range(extra_count)),
            original_site_positions_m=np.concatenate(
                (trace.original_site_positions_m, extra_vectors), axis=1
            ),
            selected_site_positions_m=np.concatenate(
                (trace.selected_site_positions_m, extra_vectors), axis=1
            ),
            measured_site_positions_m=np.concatenate(
                (trace.measured_site_positions_m, extra_vectors), axis=1
            ),
            original_site_orientations_xyzw=np.concatenate(
                (trace.original_site_orientations_xyzw, extra_quaternions), axis=1
            ),
            measured_site_orientations_xyzw=np.concatenate(
                (trace.measured_site_orientations_xyzw, extra_quaternions), axis=1
            ),
            force_on_robot_n=np.concatenate(
                (trace.force_on_robot_n, extra_vectors), axis=1
            ),
            force_on_robot_world_n=np.concatenate(
                (trace.force_on_robot_world_n, extra_vectors), axis=1
            ),
            force_on_robot_common_n=np.concatenate(
                (trace.force_on_robot_common_n, extra_vectors), axis=1
            ),
            compliance_m_per_n=np.concatenate(
                (trace.compliance_m_per_n, extra_vectors), axis=1
            ),
            active_site_mask=np.concatenate(
                (
                    trace.active_site_mask,
                    np.zeros((rows, extra_count), dtype=np.bool_),
                ),
                axis=1,
            ),
            policy_actions=np.pad(trace.policy_actions, ((0, 0), (0, 8))),
        )
    report = evaluate_matched_review_suite(expanded, spec, criteria=criteria)
    assert report["acceptance"]["passed"] is True
    assert report["trials"]["chip_hard_off"]["site_count"] == len(SITE_IDS) + extra_count


def test_trace_loader_rejects_duplicate_members_and_pickle_payload(tmp_path: Path):
    trace_path = tmp_path / "source.npz"
    write_trace_npz_atomic(_make_matched_trace("source"), trace_path)
    with zipfile.ZipFile(trace_path) as source:
        entries = [(item.filename, source.read(item.filename)) for item in source.infolist()]

    duplicate_path = tmp_path / "duplicate.npz"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_path, "w") as target:
            for filename, payload in entries:
                target.writestr(filename, payload)
            target.writestr(entries[0][0], entries[0][1])
    with pytest.raises(ValueError, match="duplicate"):
        load_trace_npz(duplicate_path)

    object_payload = io.BytesIO()
    np.save(object_payload, np.asarray([object()], dtype=object), allow_pickle=True)
    pickle_path = tmp_path / "pickle.npz"
    with zipfile.ZipFile(pickle_path, "w") as target:
        for filename, payload in entries:
            target.writestr(
                filename,
                object_payload.getvalue()
                if filename == "policy_actions.npy"
                else payload,
            )
    with pytest.raises(ValueError, match="allow_pickle=False|Object arrays"):
        load_trace_npz(pickle_path)


def _video_fixture(tmp_path: Path):
    left = _make_matched_trace("left_panel")
    right = _make_matched_trace("right_panel")
    left_trace = tmp_path / "left.npz"
    right_trace = tmp_path / "right.npz"
    left_summary = tmp_path / "left.json"
    right_summary = tmp_path / "right.json"
    metrics = tmp_path / "metrics.json"
    video = tmp_path / "comparison.mp4"
    write_trace_npz_atomic(left, left_trace)
    write_trace_npz_atomic(right, right_trace)
    write_report_json_atomic(evaluate_trace(left), left_summary)
    write_report_json_atomic(evaluate_trace(right), right_summary)
    write_report_json_atomic({"schema_version": "test_metrics_v1"}, metrics)
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=50",
            "-frames:v",
            str(len(left.motion_ids)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ),
        check=True,
    )
    spec = ReviewVideoSpec(
        comparison_name="left_vs_right",
        motion_id="review_motion",
        seed=0,
        branch_commit="0123456789abcdef",
        left=ReviewPanelSpec("left_panel", left_trace, left_summary, "a" * 64),
        right=ReviewPanelSpec("right_panel", right_trace, right_summary, "b" * 64),
        metrics_path=metrics,
        video_path=video,
        width=320,
        height=240,
    )
    return spec


def test_video_manifest_binds_exact_probe_frames_hashes_and_panel_order(tmp_path: Path):
    spec = _video_fixture(tmp_path)
    manifest = build_review_video_manifest(spec)
    assert manifest["panel_order"] == ["left_panel", "right_panel"]
    assert manifest["video_probe"] == {
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "width": 320,
        "height": 240,
        "frame_rate": "50",
        "frame_count": 6,
        "duration_s": pytest.approx(0.12),
    }
    assert manifest["trace_frame_count"] == 6
    assert manifest["video_sha256"] == hashlib.sha256(
        spec.video_path.read_bytes()
    ).hexdigest()
    assert manifest["panels"][0]["trace_sha256"] == hashlib.sha256(
        spec.left.trace_path.read_bytes()
    ).hexdigest()

    manifest_path = tmp_path / "manifest.json"
    write_review_video_manifest_atomic(spec, manifest_path)
    assert validate_review_video_manifest(manifest_path, spec) == json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    with pytest.raises(FileExistsError):
        write_review_video_manifest_atomic(spec, manifest_path)


def test_video_manifest_rejects_missing_extra_rebound_and_symlink_artifacts(tmp_path: Path):
    spec = _video_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    write_review_video_manifest_atomic(spec, manifest_path)

    missing_spec = replace(
        spec,
        metrics_path=tmp_path / "missing.json",
    )
    with pytest.raises(ValueError, match="regular non-symlink"):
        build_review_video_manifest(missing_spec)

    extra_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra_payload["unexpected"] = True
    extra_manifest = tmp_path / "extra.json"
    write_report_json_atomic(extra_payload, extra_manifest)
    with pytest.raises(ValueError, match="extra, missing, or rebound"):
        validate_review_video_manifest(extra_manifest, spec)

    reversed_spec = replace(spec, left=spec.right, right=spec.left)
    with pytest.raises(ValueError, match="extra, missing, or rebound"):
        validate_review_video_manifest(manifest_path, reversed_spec)

    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:r=50",
            "-frames:v",
            "6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(spec.video_path),
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="extra, missing, or rebound"):
        validate_review_video_manifest(manifest_path, spec)

    write_report_json_atomic(
        {"changed": True},
        spec.left.summary_path,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="extra, missing, or rebound"):
        validate_review_video_manifest(manifest_path, spec)

    symlink_summary = tmp_path / "summary_link.json"
    symlink_summary.symlink_to(spec.right.summary_path)
    symlink_spec = replace(
        spec,
        left=replace(spec.left, summary_path=symlink_summary),
    )
    with pytest.raises(ValueError, match="regular non-symlink"):
        build_review_video_manifest(symlink_spec)


def test_video_manifest_rejects_wrong_encoding_and_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec = _video_fixture(tmp_path)
    wrong_rate = tmp_path / "wrong_rate.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:r=25",
            "-frames:v",
            "6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(wrong_rate),
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="frame rate"):
        build_review_video_manifest(replace(spec, video_path=wrong_rate))

    wrong_format = tmp_path / "wrong_format.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:r=50",
            "-frames:v",
            "6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv444p",
            str(wrong_format),
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="pixel format"):
        build_review_video_manifest(replace(spec, video_path=wrong_format))

    wrong_codec = tmp_path / "wrong_codec.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=yellow:s=320x240:r=50",
            "-frames:v",
            "6",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            str(wrong_codec),
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="codec"):
        build_review_video_manifest(replace(spec, video_path=wrong_codec))

    wrong_dimensions = tmp_path / "wrong_dimensions.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=322x240:r=50",
            "-frames:v",
            "6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(wrong_dimensions),
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="dimensions"):
        build_review_video_manifest(replace(spec, video_path=wrong_dimensions))

    wrong_frames = tmp_path / "wrong_frames.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:r=50",
            "-frames:v",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(wrong_frames),
        ),
        check=True,
    )
    with pytest.raises(ValueError, match="frame count"):
        build_review_video_manifest(replace(spec, video_path=wrong_frames))

    import gear_sonic.compliance_control.review.video as video_module

    real_probe = video_module.probe_video_with_sha256

    def wrong_duration(path, *, ffprobe="ffprobe", max_bytes=512 * 1024 * 1024):
        probe, digest = real_probe(path, ffprobe=ffprobe, max_bytes=max_bytes)
        return {**probe, "duration_s": probe["duration_s"] + 1.0}, digest

    monkeypatch.setattr(video_module, "probe_video_with_sha256", wrong_duration)
    with pytest.raises(ValueError, match="duration"):
        build_review_video_manifest(spec)


def test_portable_review_cli_help_has_no_side_effects(tmp_path: Path):
    completed = subprocess.run(
        (
            sys.executable,
            "-B",
            "-m",
            "gear_sonic.compliance_control.review",
            "--help",
        ),
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[3],
        env={**dict(__import__("os").environ), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert "probe" in completed.stdout
    assert list(tmp_path.iterdir()) == []
