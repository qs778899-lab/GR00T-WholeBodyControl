#!/usr/bin/env python3
"""Independently audit the non-formal 32-frame CHIP rendered smoke."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRAINED_CHECKPOINT_SHA256 = (
    "71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02"
)
_FORMAL_OUTPUT_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/"
    "runtime_video_validation_v1"
)
_EXPECTED_ROLE = "simultaneous_compliant"
_EXPECTED_MOTION = "original"
_EXPECTED_FRAMES = 32
_MAX_TRACE_BYTES = 64 * 1024 * 1024
_MAX_TREE_BYTES = 256 * 1024 * 1024
_MAX_LOG_BYTES = 64 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--branch-commit", required=True)
    parser.add_argument(
        "--formal-output-root",
        type=Path,
        default=_FORMAL_OUTPUT_ROOT,
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _load_trace(path: Path, expected_fields: tuple[str, ...]) -> tuple[dict, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("diagnostic trace must be a regular non-symlink file") from error
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode) or not 0 < status.st_size <= _MAX_TRACE_BYTES:
            raise ValueError("diagnostic trace violates its file-size contract")
        digest = _sha256_stream(stream)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise ValueError("diagnostic trace contains duplicate ZIP members")
            if sum(member.file_size for member in members) > _MAX_TRACE_BYTES:
                raise ValueError("diagnostic trace exceeds its uncompressed cap")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_fields):
                raise ValueError("diagnostic trace fields do not match the schema")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return arrays, digest


def _audit_trace(arrays: dict, schema: str) -> None:
    scalar_strings = {
        "schema_version": schema,
        "role": _EXPECTED_ROLE,
        "motion_id": _EXPECTED_MOTION,
    }
    for name, expected in scalar_strings.items():
        value = arrays[name]
        if value.shape != () or value.dtype.kind != "U" or value.item() != expected:
            raise AssertionError(f"diagnostic trace scalar mismatch: {name}")
    if arrays["seed"].shape != () or int(arrays["seed"].item()) != 0:
        raise AssertionError("diagnostic trace seed mismatch")
    expected_indices = np.arange(_EXPECTED_FRAMES, dtype=np.int64)
    for name in ("frame_indices", "reference_frames"):
        if not np.array_equal(arrays[name], expected_indices):
            raise AssertionError(f"diagnostic trace index mismatch: {name}")
    if not np.array_equal(
        arrays["timestamps_s"],
        np.arange(_EXPECTED_FRAMES, dtype=np.float64) / 50.0,
    ):
        raise AssertionError("diagnostic timestamps are not exact 50 Hz samples")
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise AssertionError(f"diagnostic trace contains non-finite data: {name}")
    if arrays["policy_actions"].shape != (_EXPECTED_FRAMES, 29):
        raise AssertionError("diagnostic action shape mismatch")
    site_vectors = (
        "original_site_positions_m",
        "selected_site_positions_m",
        "measured_site_positions_m",
        "force_on_robot_n",
        "force_on_robot_world_n",
        "force_on_robot_common_n",
        "compliance_m_per_n",
    )
    for name in site_vectors:
        if arrays[name].shape != (_EXPECTED_FRAMES, 2, 3):
            raise AssertionError(f"diagnostic site-vector shape mismatch: {name}")
    for name in (
        "reference_points_global_m",
        "measured_points_global_m",
        "reference_points_local_m",
        "measured_points_local_m",
    ):
        if arrays[name].shape != (_EXPECTED_FRAMES, 14, 3):
            raise AssertionError(f"diagnostic tracking-point shape mismatch: {name}")
    for name in ("terminal_mask", "timeout_mask", "fall_mask"):
        if arrays[name].shape != (_EXPECTED_FRAMES,) or np.any(arrays[name]):
            raise AssertionError(f"diagnostic lifecycle mask is nonzero: {name}")
    expected_reset = np.arange(_EXPECTED_FRAMES) == 0
    if not np.array_equal(arrays["reset_mask"], expected_reset):
        raise AssertionError("diagnostic reset mask mismatch")
    if arrays["active_site_mask"].shape != (_EXPECTED_FRAMES, 2):
        raise AssertionError("diagnostic active-site layout mismatch")
    if not np.any(np.all(arrays["active_site_mask"], axis=-1)):
        raise AssertionError("diagnostic never activated both wrists")
    if not np.array_equal(
        arrays["force_on_robot_n"], arrays["force_on_robot_world_n"]
    ):
        raise AssertionError("diagnostic evaluation/world force bytes differ")
    force_norm = np.linalg.norm(arrays["force_on_robot_world_n"], axis=-1)
    if abs(float(np.max(force_norm)) - 5.0) > 1.0e-6:
        raise AssertionError("diagnostic did not reach the pinned 5 N force")
    expected_selected = (
        arrays["original_site_positions_m"]
        - arrays["compliance_m_per_n"] * arrays["force_on_robot_world_n"]
    )
    if not np.allclose(
        arrays["selected_site_positions_m"],
        expected_selected,
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise AssertionError("diagnostic signed selected-target relation failed")


def _audit_tree(root: Path) -> tuple[int, int]:
    total_bytes = 0
    largest_log = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AssertionError(f"diagnostic tree contains a symlink: {path}")
        if path.name in {"__pycache__", ".pytest_cache"}:
            raise AssertionError(f"diagnostic tree contains a cache: {path}")
        if path.suffix in {".pyc", ".pyo", ".tmp", ".part"}:
            raise AssertionError(f"diagnostic tree contains a temporary file: {path}")
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            if path.suffix == ".log":
                largest_log = max(largest_log, size)
    if total_bytes > _MAX_TREE_BYTES or largest_log > _MAX_LOG_BYTES:
        raise AssertionError("diagnostic tree exceeds its capacity contract")
    return total_bytes, largest_log


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.branch_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.branch_commit
    ):
        raise ValueError("--branch-commit must be a lowercase 40-character Git hash")
    if args.formal_output_root.exists() or args.formal_output_root.is_symlink():
        raise AssertionError("formal Phase-5 output root must remain absent")
    root = args.run_root.resolve(strict=True)
    if args.run_root.is_symlink() or not root.is_dir():
        raise NotADirectoryError(args.run_root)
    role_root = root / _EXPECTED_MOTION / _EXPECTED_ROLE
    trace_path = role_root / "trace.npz"
    summary_path = role_root / "summary.json"
    video_path = role_root / "panel.mp4"

    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from gear_sonic.compliance_control.adapters.sonic.contracts import (
        SONIC_RELEASE_TRACKING_BODY_NAMES,
    )
    from gear_sonic.compliance_control.adapters.sonic.review.diagnostic import (
        DIAGNOSTIC_TRACE_FIELDS,
        DIAGNOSTIC_TRACE_SCHEMA,
    )
    from gear_sonic.compliance_control.adapters.sonic.review.roles import (
        REVIEW_SITE_NAMES,
    )
    from gear_sonic.compliance_control.review import (
        load_report_json_with_sha256,
        probe_video_with_sha256,
    )

    arrays, trace_sha256 = _load_trace(trace_path, DIAGNOSTIC_TRACE_FIELDS)
    _audit_trace(arrays, DIAGNOSTIC_TRACE_SCHEMA)
    summary, _ = load_report_json_with_sha256(summary_path)
    if not isinstance(summary, dict):
        raise TypeError("diagnostic summary must be a JSON object")
    expected_summary = {
        "schema_version": "sonic_chip_review_diagnostic_v1",
        "role": _EXPECTED_ROLE,
        "checkpoint_kind": "trained",
        "checkpoint_sha256": _TRAINED_CHECKPOINT_SHA256,
        "checkpoint_load_semantics": "native_strict_resume",
        "branch_commit": args.branch_commit,
        "motion_id": _EXPECTED_MOTION,
        "seed": 0,
        "frame_count": _EXPECTED_FRAMES,
        "trace_kind": "diagnostic_fixed_cutoff_nonformal",
        "natural_timeout_count": 0,
        "fall_count": 0,
        "trace_reset_count": 1,
        "command_reset_count": 2,
        "composer_owned_reset_force_peak_n": 0.0,
        "composer_owned_reset_torque_peak_nm": 0.0,
        "finite_observations": True,
        "finite_actions": True,
        "body_names": list(SONIC_RELEASE_TRACKING_BODY_NAMES),
        "site_names": list(REVIEW_SITE_NAMES),
        "force_evaluation_frame": "world",
        "force_common_frame": "heading_local",
        "observation_dims": {
            "actor_obs": 930,
            "critic_obs": 1645,
            "tokenizer": 1761,
            "compliance_target": 60,
            "compliance_command": 9,
            "compliance_force": 6,
        },
    }
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            raise AssertionError(f"diagnostic summary mismatch: {name}")
    if int(summary.get("source_motion_frame_count", 0)) <= _EXPECTED_FRAMES:
        raise AssertionError("diagnostic source motion is not longer than its cutoff")
    if summary.get("trace_sha256") != trace_sha256:
        raise AssertionError("diagnostic trace hash does not match its summary")
    if Path(summary.get("trace", "")).resolve() != trace_path.resolve():
        raise AssertionError("diagnostic trace path provenance mismatch")
    if Path(summary.get("panel_video", "")).resolve() != video_path.resolve():
        raise AssertionError("diagnostic video path provenance mismatch")
    if float(summary.get("peak_world_force_n", 0.0)) < 4.999:
        raise AssertionError("diagnostic summary did not record the 5 N stimulus")
    if float(summary.get("peak_latent_residual", 0.0)) <= 0.0:
        raise AssertionError("diagnostic summary did not record residual activation")
    probe, video_sha256 = probe_video_with_sha256(video_path, ffprobe=args.ffprobe)
    expected_probe = {
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "width": 960,
        "height": 720,
        "frame_rate": "50",
        "frame_count": _EXPECTED_FRAMES,
    }
    for name, expected in expected_probe.items():
        if probe.get(name) != expected:
            raise AssertionError(f"diagnostic video probe mismatch: {name}")
    if abs(float(probe["duration_s"]) - _EXPECTED_FRAMES / 50.0) > 0.01:
        raise AssertionError("diagnostic video duration mismatch")
    if summary.get("panel_video_probe") != probe:
        raise AssertionError("live video probe differs from the summary")
    if summary.get("panel_video_sha256") != video_sha256:
        raise AssertionError("live video hash differs from the summary")
    total_bytes, largest_log = _audit_tree(root)
    print(
        "CHIP_PHASE4_RENDERED_SMOKE_AUDIT_PASS",
        f"frames={_EXPECTED_FRAMES}",
        f"trace_sha256={trace_sha256}",
        f"video_sha256={video_sha256}",
        f"tree_bytes={total_bytes}",
        f"largest_log_bytes={largest_log}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
