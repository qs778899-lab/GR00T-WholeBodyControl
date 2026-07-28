"""CPU contract tests for tracker-neutral paired compliance evaluation."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.compliance_control.evaluation import (
    EvaluationTrace,
    RegressionCriteria,
    TraceAlignmentError,
    TrialMode,
    TrialSpec,
    alignment_digest,
    assert_strict_alignment,
    compare_aligned_traces,
    evaluate_trace,
    evaluate_trial_suite,
    load_trace_npz,
    load_trace_npz_with_sha256,
    write_report_json_atomic,
    write_trace_npz_atomic,
)


EVALUATION_DIR = Path(__file__).parents[1] / "compliance_control" / "evaluation"
SITE_IDS = ("endpoint_a", "endpoint_b", "balance_probe")
POINT_IDS = tuple(f"tracking_point_{index}" for index in range(7))


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
    active_mask = np.zeros((row_count, site_count), dtype=np.bool_)
    force = np.zeros((row_count, site_count, 3), dtype=np.float64)
    for site_id in active_sites:
        site_index = SITE_IDS.index(site_id)
        active_mask[1:3, site_index] = True
        selected_site[1:3, site_index, 0] += 0.02
        measured_site[1:3, site_index, 0] += 0.01
        force[1:3, site_index, 0] = 4.0

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
        compliance_enabled=compliance_enabled,
        active_site_mask=active_mask,
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
            endpoint_error_m=0.013,
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
    criteria = RegressionCriteria(endpoint_site_ids=("endpoint_a", "endpoint_b"))
    return traces, specs, criteria


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

    assert report["schema_version"] == "compliance_evaluation_v1"
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
    check_names = {check["name"] for check in report["acceptance"]["checks"]}
    assert "off_success_rate_drop" in check_names
    assert "off_local_mpjpe_regression_m" in check_names
    assert "off_endpoint_rmse_regression_m:endpoint_a" in check_names
    assert "no_contact_endpoint_rmse_delta_m:enabled_no_contact:endpoint_b" in check_names
    assert "post_reset_force_peak_n:simultaneous" in check_names
    assert "inactive_force_peak_n:overlay_off:endpoint_a" in check_names


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
        replace(trace, terminal_mask=np.zeros_like(trace.terminal_mask), success_mask=np.zeros_like(trace.success_mask))


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
    for path in EVALUATION_DIR.glob("*.py"):
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
