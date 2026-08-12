#!/usr/bin/env python3
"""Collect one full, trace-aligned SONIC CHIP review role and panel video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    from gear_sonic.compliance_control.adapters.sonic.review.roles import (
        REVIEW_ROLE_NAMES,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=REVIEW_ROLE_NAMES)
    parser.add_argument("--motion-id", required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--smpl-motion-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--diagnostic-frames",
        type=int,
        help="non-formal fixed cutoff; omit to require the complete natural timeout",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _branch_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.diagnostic_frames is not None and args.diagnostic_frames < 8:
        raise ValueError("--diagnostic-frames must be at least eight")
    if not args.motion_id or "/" in args.motion_id or "\\" in args.motion_id:
        raise ValueError("--motion-id must be a safe non-empty name")
    for path in (args.motion_file, args.official_checkpoint, args.trained_checkpoint):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    if not args.smpl_motion_dir.is_dir() or args.smpl_motion_dir.is_symlink():
        raise NotADirectoryError(args.smpl_motion_dir)
    if args.output_root.is_symlink():
        raise ValueError("--output-root must not be a symlink")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_inputs(args)
    from gear_sonic.compliance_control.adapters.sonic.review.config import (
        ReviewArtifactPaths,
        build_review_dry_run_plan,
        compose_review_config,
    )
    from gear_sonic.compliance_control.adapters.sonic.review.roles import (
        get_review_role,
    )

    role = get_review_role(args.role)
    checkpoint = (
        args.official_checkpoint
        if role.checkpoint_kind == "official"
        else args.trained_checkpoint
    )
    checkpoint_sha256 = _sha256(checkpoint)
    if args.dry_run:
        plan = build_review_dry_run_plan(
            role,
            motion_id=args.motion_id,
            motion_file=args.motion_file,
            smpl_motion_dir=args.smpl_motion_dir,
            checkpoint=checkpoint,
            output_root=args.output_root,
            seed=args.seed,
        )
        plan["checkpoint_sha256"] = checkpoint_sha256
        plan["branch_commit"] = _branch_commit()
        plan["diagnostic_frames"] = args.diagnostic_frames
        plan["publication_kind"] = (
            "formal_natural_timeout"
            if args.diagnostic_frames is None
            else "diagnostic_fixed_cutoff_nonformal"
        )
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0

    config = compose_review_config(
        role,
        motion_file=args.motion_file,
        smpl_motion_dir=args.smpl_motion_dir,
        seed=args.seed,
        experiment_dir=(args.output_root / args.motion_id / role.name / "runtime"),
    )
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        {
            "headless": True,
            "enable_cameras": True,
            "device": args.device,
        }
    )
    simulation_app = app_launcher.app
    try:
        from gear_sonic.compliance_control.adapters.sonic.review.runtime import (
            collect_sonic_review_role,
        )

        summary = collect_sonic_review_role(
            config=config,
            role=role,
            motion_id=args.motion_id,
            seed=args.seed,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            branch_commit=_branch_commit(),
            paths=ReviewArtifactPaths(args.output_root, args.motion_id, role.name),
            device=args.device,
            diagnostic_frame_limit=args.diagnostic_frames,
        )
        print(
            "CHIP_REVIEW_COLLECT_PASS",
            json.dumps(summary, allow_nan=False, sort_keys=True),
            flush=True,
        )
        result = 0
    except BaseException:
        traceback.print_exc()
        result = 1
    try:
        simulation_app.close()
    except BaseException:
        traceback.print_exc()
        result = 1
    return result


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 1
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
