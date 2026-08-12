#!/usr/bin/env python3
"""Evaluate one nine-role CHIP review suite and optionally compose five videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--compose-videos", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _suite_spec():
    from gear_sonic.compliance_control.review import (
        MatchedInteractionSpec,
        ReviewSuiteSpec,
    )

    return ReviewSuiteSpec(
        baseline_trial="release_baseline",
        hard_off_trial="chip_hard_off",
        no_contact_trial="enabled_no_contact",
        interactions=(
            MatchedInteractionSpec(
                "single_left",
                "single_left_stiff",
                "single_left_compliant",
                ("left_wrist_yaw_link",),
            ),
            MatchedInteractionSpec(
                "single_right",
                "single_right_stiff",
                "single_right_compliant",
                ("right_wrist_yaw_link",),
            ),
            MatchedInteractionSpec(
                "simultaneous",
                "simultaneous_stiff",
                "simultaneous_compliant",
                ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from gear_sonic.compliance_control.adapters.sonic.review.roles import (
        REVIEW_COMPARISONS,
        REVIEW_ROLE_NAMES,
    )

    metrics = args.metrics or args.motion_root / "metrics.json"
    video_root = args.motion_root / "review_videos"
    plan = {
        "schema_version": "sonic_chip_review_evaluate_plan_v1",
        "motion_root": str(args.motion_root),
        "roles": list(REVIEW_ROLE_NAMES),
        "trace_inputs": [
            str(args.motion_root / role / "trace.npz") for role in REVIEW_ROLE_NAMES
        ],
        "metrics_output": str(metrics),
        "compose_videos": bool(args.compose_videos),
        "video_outputs": [
            str(video_root / f"{name}.mp4") for name, _, _ in REVIEW_COMPARISONS
        ],
        "would_write": bool(not args.dry_run),
        "simulator_imported": False,
    }
    if args.dry_run:
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0

    from gear_sonic.compliance_control.review import (
        RegressionCriteria,
        compose_review_panels,
        evaluate_matched_review_suite,
        load_trace_npz,
        write_report_json_atomic,
    )

    traces = {
        role: load_trace_npz(args.motion_root / role / "trace.npz")
        for role in REVIEW_ROLE_NAMES
    }
    report = evaluate_matched_review_suite(
        traces,
        _suite_spec(),
        criteria=RegressionCriteria(
            endpoint_site_ids=("left_wrist_yaw_link", "right_wrist_yaw_link"),
            endpoint_tracking_point_ids=(
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        ),
    )
    write_report_json_atomic(report, metrics)
    if not report["acceptance"]["passed"]:
        print("CHIP_REVIEW_EVALUATION_FAILED", str(metrics), flush=True)
        return 2
    if args.compose_videos:
        video_root.mkdir(parents=False, exist_ok=False)
        for name, left_role, right_role in REVIEW_COMPARISONS:
            compose_review_panels(
                args.motion_root / left_role / "panel.mp4",
                args.motion_root / right_role / "panel.mp4",
                video_root / f"{name}.mp4",
                ffmpeg=args.ffmpeg,
            )
    print("CHIP_REVIEW_EVALUATION_PASS", str(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
