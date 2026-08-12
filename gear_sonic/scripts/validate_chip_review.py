#!/usr/bin/env python3
"""Publish or independently revalidate one motion's five CHIP video manifests."""

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
    parser.add_argument("--branch-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--publish-manifests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    from gear_sonic.compliance_control.adapters.sonic.review.roles import (
        REVIEW_COMPARISONS,
    )

    video_root = args.motion_root / "review_videos"
    manifest_root = video_root / "manifests"
    plan = {
        "schema_version": "sonic_chip_review_validate_plan_v1",
        "motion_root": str(args.motion_root),
        "branch_commit": args.branch_commit,
        "seed": args.seed,
        "publish_manifests": bool(args.publish_manifests),
        "manifest_paths": [
            str(manifest_root / f"{name}.json") for name, _, _ in REVIEW_COMPARISONS
        ],
        "would_write": bool(args.publish_manifests and not args.dry_run),
        "simulator_imported": False,
    }
    if args.dry_run:
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0

    from gear_sonic.compliance_control.review import (
        ReviewPanelSpec,
        ReviewVideoSpec,
        load_report_json_with_sha256,
        validate_review_video_manifest,
        write_report_json_atomic,
        write_review_video_manifest_atomic,
    )

    metrics = args.motion_root / "metrics.json"
    metrics_payload, metrics_sha = load_report_json_with_sha256(metrics)
    if not isinstance(metrics_payload, dict) or not metrics_payload.get("acceptance", {}).get(
        "passed"
    ):
        raise RuntimeError("review metrics are absent or did not pass")
    summaries = {}
    for _, left_role, right_role in REVIEW_COMPARISONS:
        for role in (left_role, right_role):
            if role in summaries:
                continue
            payload, _ = load_report_json_with_sha256(
                args.motion_root / role / "summary.json"
            )
            if not isinstance(payload, dict) or payload.get("role") != role:
                raise ValueError(f"invalid role summary: {role}")
            summaries[role] = payload
    if args.publish_manifests:
        manifest_root.mkdir(parents=False, exist_ok=False)
    validated = []
    for comparison, left_role, right_role in REVIEW_COMPARISONS:
        spec = ReviewVideoSpec(
            comparison_name=comparison,
            motion_id=args.motion_root.name,
            seed=args.seed,
            branch_commit=args.branch_commit,
            left=ReviewPanelSpec(
                left_role,
                args.motion_root / left_role / "trace.npz",
                args.motion_root / left_role / "summary.json",
                str(summaries[left_role]["checkpoint_sha256"]),
            ),
            right=ReviewPanelSpec(
                right_role,
                args.motion_root / right_role / "trace.npz",
                args.motion_root / right_role / "summary.json",
                str(summaries[right_role]["checkpoint_sha256"]),
            ),
            metrics_path=metrics,
            video_path=video_root / f"{comparison}.mp4",
            width=1920,
            height=720,
        )
        manifest_path = manifest_root / f"{comparison}.json"
        if args.publish_manifests:
            write_review_video_manifest_atomic(spec, manifest_path, ffprobe=args.ffprobe)
        manifest = validate_review_video_manifest(
            manifest_path,
            spec,
            ffprobe=args.ffprobe,
        )
        validated.append(
            {
                "comparison": comparison,
                "manifest": str(manifest_path.resolve()),
                "video_sha256": manifest["video_sha256"],
            }
        )
    if args.publish_manifests:
        write_report_json_atomic(
            {
                "schema_version": "sonic_chip_review_validation_v1",
                "motion_id": args.motion_root.name,
                "branch_commit": args.branch_commit,
                "seed": args.seed,
                "metrics_sha256": metrics_sha,
                "comparisons": validated,
            },
            video_root / "validation.json",
        )
    print("CHIP_REVIEW_VALIDATION_PASS", json.dumps(validated, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
