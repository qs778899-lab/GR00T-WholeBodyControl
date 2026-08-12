#!/usr/bin/env python3
"""Evaluate standardized paired compliance traces without importing a simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gear_sonic.compliance_control.evaluation import (  # noqa: E402
    RegressionCriteria,
    TrialMode,
    TrialSpec,
    evaluate_trial_suite,
    load_trace_npz_with_sha256,
    write_report_json_atomic,
)


def _parse_site_ids(value: str) -> tuple[str, ...]:
    if value == "-":
        return ()
    parts = tuple(value.split(","))
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError("active site IDs must be comma-separated non-empty names")
    if len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError("active site IDs must not contain duplicates")
    return parts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact motion/sequence/seed/frame/timestamp alignment and "
            "evaluate portable compliance trace NPZ files."
        )
    )
    parser.add_argument(
        "--trial",
        action="append",
        nargs=4,
        required=True,
        metavar=("NAME", "MODE", "TRACE_NPZ", "ACTIVE_SITE_IDS_OR_DASH"),
        help=(
            "Add a paired trace. MODE is baseline/off/no_contact/single_site/"
            "multi_site; the final field is '-' or comma-separated caller-owned site IDs."
        ),
    )
    parser.add_argument("--baseline", required=True, help="Name of the baseline-mode trial")
    parser.add_argument(
        "--endpoint-site",
        action="append",
        required=True,
        help="Caller-selected endpoint site ID; repeat in mapping order",
    )
    parser.add_argument(
        "--endpoint-point",
        action="append",
        required=True,
        help=(
            "Caller-selected tracking point corresponding one-to-one with each "
            "--endpoint-site"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-trace-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-report-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-success-rate-drop", type=float, default=0.01)
    parser.add_argument("--local-mpjpe-absolute-regression-m", type=float, default=0.003)
    parser.add_argument("--local-mpjpe-relative-regression", type=float, default=0.10)
    parser.add_argument("--endpoint-rmse-regression-m", type=float, default=0.005)
    parser.add_argument("--no-contact-endpoint-delta-m", type=float, default=0.005)
    parser.add_argument("--reset-wrench-tolerance-n", type=float, default=1.0e-6)
    parser.add_argument("--inactive-force-tolerance-n", type=float, default=1.0e-6)
    parser.add_argument("--inactive-yield-tolerance-m", type=float, default=1.0e-7)
    parser.add_argument("--minimum-active-force-peak-n", type=float, default=5.0)
    parser.add_argument("--minimum-active-yield-peak-m", type=float, default=0.049)
    parser.add_argument("--minimum-active-measured-yield-peak-m", type=float, default=0.0005)
    parser.add_argument(
        "--minimum-active-measured-yield-along-force-peak-m",
        type=float,
        default=0.0001,
    )
    parser.add_argument("--inactive-cross-coupling-rmse-m", type=float, default=0.005)
    parser.add_argument("--inactive-cross-coupling-p95-m", type=float, default=0.010)
    parser.add_argument(
        "--active-selected-endpoint-rmse-regression-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--active-selected-endpoint-p95-regression-m",
        type=float,
        default=0.010,
    )
    parser.add_argument(
        "--active-orientation-rmse-regression-rad",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--active-orientation-p95-regression-rad",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--active-invariant-local-mpjpe-absolute-regression-m",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--active-invariant-local-mpjpe-relative-regression",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--active-invariant-global-mpjpe-absolute-regression-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--active-invariant-global-mpjpe-relative-regression",
        type=float,
        default=0.10,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    traces = {}
    trace_sha256_by_trial = {}
    specs = []
    for name, mode_value, trace_path, active_value in args.trial:
        if name in traces:
            raise ValueError(f"duplicate trial name: {name}")
        trace, trace_sha256 = load_trace_npz_with_sha256(
            trace_path,
            max_bytes=args.max_trace_bytes,
        )
        if trace.trial_name != name:
            raise ValueError(f"trace name mismatch: expected {name}, got {trace.trial_name}")
        traces[name] = trace
        trace_sha256_by_trial[name] = trace_sha256
        specs.append(
            TrialSpec(
                name=name,
                mode=TrialMode(mode_value),
                expected_active_site_ids=_parse_site_ids(active_value),
            )
        )

    criteria = RegressionCriteria(
        endpoint_site_ids=tuple(args.endpoint_site),
        endpoint_tracking_point_ids=tuple(args.endpoint_point),
        max_success_rate_drop=args.max_success_rate_drop,
        local_mpjpe_absolute_regression_m=args.local_mpjpe_absolute_regression_m,
        local_mpjpe_relative_regression=args.local_mpjpe_relative_regression,
        endpoint_rmse_regression_m=args.endpoint_rmse_regression_m,
        no_contact_endpoint_delta_m=args.no_contact_endpoint_delta_m,
        reset_wrench_tolerance_n=args.reset_wrench_tolerance_n,
        inactive_force_tolerance_n=args.inactive_force_tolerance_n,
        inactive_yield_tolerance_m=args.inactive_yield_tolerance_m,
        minimum_active_force_peak_n=args.minimum_active_force_peak_n,
        minimum_active_yield_peak_m=args.minimum_active_yield_peak_m,
        minimum_active_measured_yield_peak_m=(
            args.minimum_active_measured_yield_peak_m
        ),
        minimum_active_measured_yield_along_force_peak_m=(
            args.minimum_active_measured_yield_along_force_peak_m
        ),
        inactive_cross_coupling_rmse_m=args.inactive_cross_coupling_rmse_m,
        inactive_cross_coupling_p95_m=args.inactive_cross_coupling_p95_m,
        active_selected_endpoint_rmse_regression_m=(
            args.active_selected_endpoint_rmse_regression_m
        ),
        active_selected_endpoint_p95_regression_m=(
            args.active_selected_endpoint_p95_regression_m
        ),
        active_orientation_rmse_regression_rad=(
            args.active_orientation_rmse_regression_rad
        ),
        active_orientation_p95_regression_rad=(
            args.active_orientation_p95_regression_rad
        ),
        active_invariant_local_mpjpe_absolute_regression_m=(
            args.active_invariant_local_mpjpe_absolute_regression_m
        ),
        active_invariant_local_mpjpe_relative_regression=(
            args.active_invariant_local_mpjpe_relative_regression
        ),
        active_invariant_global_mpjpe_absolute_regression_m=(
            args.active_invariant_global_mpjpe_absolute_regression_m
        ),
        active_invariant_global_mpjpe_relative_regression=(
            args.active_invariant_global_mpjpe_relative_regression
        ),
    )
    report = evaluate_trial_suite(
        traces,
        specs,
        baseline_name=args.baseline,
        criteria=criteria,
    )
    report["trace_sha256_by_trial"] = trace_sha256_by_trial
    write_report_json_atomic(
        report,
        args.output,
        max_bytes=args.max_report_bytes,
        overwrite=args.overwrite,
    )
    print(
        "MOTION_COMPLIANCE_PHASE6_EVALUATION_COMPLETE "
        f"passed={str(report['acceptance']['passed']).lower()} "
        f"trials={len(traces)} output={args.output}"
    )
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
