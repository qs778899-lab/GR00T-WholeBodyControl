#!/usr/bin/env python3
"""Run bounded paired CHIP evaluation and separate residual ONNX export."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from gear_sonic.compliance_control.adapters.sonic.export import (
    PHASE5_ACCEPTED_ONNXRUNTIME_VERSION,
    SonicResidualExportSpec,
    export_sonic_actor_residual_onnx,
)
from gear_sonic.compliance_control.core import (
    PairedEvaluationThresholds,
    compare_aligned_tracking_traces,
)
from gear_sonic.compliance_control.postprocess import (
    load_tracking_trace,
    paired_result_to_dict,
    write_json_new_atomic,
)


_DEFAULT_RUNS_ROOT = (_REPOSITORY_ROOT / "compliance_control/runs/chip").resolve()
_MAX_WORKFLOW_BYTES = 500_000_000
_MAX_LOG_BYTES = 64_000_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=_DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--smpl-motion-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--onnxruntime-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--onnxruntime-version",
        default=PHASE5_ACCEPTED_ONNXRUNTIME_VERSION,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_new_run_root(path: Path, *, runs_root: Path) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"run root already exists: {path}")
    allowed_root = runs_root.resolve()
    resolved = path.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError(f"run root must be a strict child of {allowed_root}")
    if os.path.lexists(resolved):
        raise FileExistsError(f"run root already exists: {resolved}")
    return resolved


def _assert_artifact_tree_safe(
    root: Path,
    *,
    max_workflow_bytes: int = _MAX_WORKFLOW_BYTES,
    max_log_bytes: int = _MAX_LOG_BYTES,
) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AssertionError(f"Phase-5 workflow contains a symlink: {path}")
        if path.is_file():
            if path.suffix == ".log" and path.stat().st_size > max_log_bytes:
                raise AssertionError(f"Phase-5 log exceeds cap: {path}")
            total += path.stat().st_size
    if total > max_workflow_bytes:
        raise AssertionError("Phase-5 workflow exceeds the configured byte cap")
    return total


def _rollout_command(
    *,
    mode: str,
    run_root: Path,
    checkpoint: Path,
    motion_file: Path,
    smpl_motion_dir: Path,
    steps: int,
    seed: int,
    device: str,
) -> list[str]:
    output = run_root / mode
    return [
        sys.executable,
        "-B",
        str(_REPOSITORY_ROOT / "gear_sonic/scripts/run_chip_phase5_rollout.py"),
        "--mode",
        mode,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--motion-file",
        str(motion_file),
        "--smpl-motion-dir",
        str(smpl_motion_dir),
        "--checkpoint",
        str(checkpoint),
        "--trace",
        str(output / "trace.npz"),
        "--summary",
        str(output / "rollout.json"),
        "--headless",
        "--device",
        device,
    ]


def _run_logged(command: list[str], log_path: Path) -> float:
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=_REPOSITORY_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    duration = time.monotonic() - start
    if log_path.stat().st_size > _MAX_LOG_BYTES:
        raise AssertionError(f"rollout log exceeds cap: {log_path}")
    text = log_path.read_text(encoding="utf-8")
    if completed.returncode != 0 or "CHIP_PHASE5_ROLLOUT_PASS" not in text:
        raise RuntimeError(f"Phase-5 rollout failed; inspect {log_path}")
    return duration


def _onnx_parity_command(
    *,
    python: Path,
    checkpoint: Path,
    onnx_path: Path,
    report_path: Path,
    expected_version: str,
) -> list[str]:
    return [
        str(python),
        "-B",
        str(_REPOSITORY_ROOT / "gear_sonic/scripts/verify_chip_phase5_onnx.py"),
        "--checkpoint",
        str(checkpoint),
        "--onnx",
        str(onnx_path),
        "--report",
        str(report_path),
        "--expected-version",
        expected_version,
    ]


def _run_onnx_parity(command: list[str], log_path: Path) -> float:
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=_REPOSITORY_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    duration = time.monotonic() - start
    if log_path.stat().st_size > _MAX_LOG_BYTES:
        raise AssertionError(f"ONNX parity log exceeds cap: {log_path}")
    text = log_path.read_text(encoding="utf-8")
    if (
        completed.returncode != 0
        or "CHIP_PHASE5_ONNXRUNTIME_PARITY_PASS" not in text
    ):
        raise RuntimeError(f"Phase-5 ONNX Runtime parity failed; inspect {log_path}")
    return duration


def main() -> int:
    args = _parse_args()
    if args.steps < 200:
        raise ValueError("--steps must be at least 200 for the exposure/alignment gate")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.onnxruntime_version != PHASE5_ACCEPTED_ONNXRUNTIME_VERSION:
        raise ValueError(
            "--onnxruntime-version must match the pinned Phase-5 acceptance version"
        )
    for path in (args.checkpoint, args.motion_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.smpl_motion_dir.is_dir():
        raise NotADirectoryError(args.smpl_motion_dir)
    onnxruntime_python = args.onnxruntime_python.resolve()
    if not onnxruntime_python.is_file():
        raise FileNotFoundError(onnxruntime_python)
    runs_root = args.runs_root.resolve()
    run_root = _safe_new_run_root(args.run_root, runs_root=runs_root)
    checkpoint = args.checkpoint.resolve()
    motion_file = args.motion_file.resolve()
    smpl_motion_dir = args.smpl_motion_dir.resolve()
    commands = {
        mode: _rollout_command(
            mode=mode,
            run_root=run_root,
            checkpoint=checkpoint,
            motion_file=motion_file,
            smpl_motion_dir=smpl_motion_dir,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
        )
        for mode in ("stiff", "compliant")
    }
    spec = SonicResidualExportSpec(
        site_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
        num_future_frames=10,
        cartesian_dim=3,
        context_dim=930,
        output_dim=64,
        hidden_dims=(256, 128),
        residual_limit=0.25,
        common_frame="heading_local:pelvis",
    )
    thresholds = PairedEvaluationThresholds(
        min_aligned_frames=200,
        min_exposed_frames_per_site=20,
        max_upper_endpoint_regression_m=0.05,
        max_upper_endpoint_orientation_regression_rad=0.25,
        max_global_mpjpe_regression_m=0.03,
        max_local_mpjpe_regression_m=0.03,
        min_paired_displacement_m=1.0e-6,
        min_compliant_success_rate=1.0,
        max_compliant_fall_rate=0.0,
        min_peak_force_n=1.0,
        max_peak_force_n=30.0,
    )
    onnx_path = run_root / "export/compliance_residual.onnx"
    parity_path = run_root / "export/parity.json"
    parity_command = _onnx_parity_command(
        python=onnxruntime_python,
        checkpoint=checkpoint,
        onnx_path=onnx_path,
        report_path=parity_path,
        expected_version=args.onnxruntime_version,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "runs_root": str(runs_root),
                    "comparison_semantics": (
                        "matched reference/force/compliance; stiff is the release-equivalent "
                        "zero-residual policy, not an unforced baseline"
                    ),
                    "commands": commands,
                    "onnx_runtime": {
                        "python": str(onnxruntime_python),
                        "expected_version": args.onnxruntime_version,
                        "command": parity_command,
                    },
                    "export": {
                        "checkpoint": str(checkpoint),
                        "onnx": str(run_root / "export/compliance_residual.onnx"),
                        "spec": {
                            "condition_dim": spec.condition_dim,
                            "command_dim": spec.command_dim,
                            "context_dim": spec.context_dim,
                            "output_dim": spec.output_dim,
                        },
                    },
                    "thresholds": thresholds.__dict__
                    if hasattr(thresholds, "__dict__")
                    else {
                        name: getattr(thresholds, name)
                        for name in thresholds.__dataclass_fields__
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    run_root.mkdir(parents=True)
    for mode in commands:
        (run_root / mode).mkdir()
    export_dir = run_root / "export"
    export_dir.mkdir()
    started = time.monotonic()
    manifest = export_sonic_actor_residual_onnx(
        checkpoint_path=checkpoint,
        output_path=onnx_path,
        spec=spec,
    )
    parity_duration = _run_onnx_parity(
        parity_command,
        export_dir / "parity.log",
    )
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("runtime") != "onnxruntime.InferenceSession":
        raise AssertionError("accepted export parity did not use ONNX Runtime")
    if parity.get("runtime_version") != PHASE5_ACCEPTED_ONNXRUNTIME_VERSION:
        raise AssertionError("accepted export ONNX Runtime version mismatch")
    if parity.get("providers") != ["CPUExecutionProvider"]:
        raise AssertionError("accepted export ONNX Runtime provider mismatch")

    durations = {}
    for mode, command in commands.items():
        durations[mode] = _run_logged(command, run_root / f"{mode}.log")

    stiff = load_tracking_trace(run_root / "stiff/trace.npz")
    compliant = load_tracking_trace(run_root / "compliant/trace.npz")
    result = compare_aligned_tracking_traces(
        stiff,
        compliant,
        thresholds=thresholds,
        alignment_atol=1.0e-5,
    )
    result_payload = paired_result_to_dict(result)
    result_payload["thresholds"] = {
        name: getattr(thresholds, name) for name in thresholds.__dataclass_fields__
    }
    write_json_new_atomic(run_root / "paired_metrics.json", result_payload)

    rollout_summaries = {
        mode: json.loads((run_root / mode / "rollout.json").read_text(encoding="utf-8"))
        for mode in commands
    }
    if rollout_summaries["stiff"]["peak_latent_residual"] != 0.0:
        raise AssertionError("stiff rollout did not retain exact residual-off parity")
    if rollout_summaries["compliant"]["peak_latent_residual"] <= 0.0:
        raise AssertionError("compliant rollout did not activate the residual")
    if not result.passed:
        failed = [name for name, passed in result.checks if not passed]
        raise AssertionError(f"paired evaluation acceptance failed: {failed}")

    workflow_bytes_before_final_manifest = _assert_artifact_tree_safe(run_root)
    workflow = {
        "schema_version": 1,
        "complete": True,
        "marker": "CHIP_PHASE5_EVAL_EXPORT_PASS",
        "run_root": str(run_root),
        "runs_root": str(runs_root),
        "checkpoint": str(checkpoint),
        "evaluation_claim": "chain_validation_not_performance_proof",
        "comparison_semantics": (
            "matched reference/force/compliance; stiff is the release-equivalent "
            "zero-residual policy, not the unforced Phase-4 stiff training log"
        ),
        "steps": args.steps,
        "seed": args.seed,
        "durations_s": durations,
        "onnx_parity_duration_s": parity_duration,
        "total_duration_s": time.monotonic() - started,
        "workflow_bytes_before_final_manifest": workflow_bytes_before_final_manifest,
        "paired_metrics": str(run_root / "paired_metrics.json"),
        "export_manifest": str(onnx_path.with_suffix(".json")),
        "export_parity": str(export_dir / "parity.json"),
        "onnxruntime_version": parity["runtime_version"],
        "onnxruntime_providers": parity["providers"],
        "onnx_sha256": manifest["onnx_sha256"],
    }
    workflow_json_bytes = len(
        (
            json.dumps(workflow, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    )
    if workflow_bytes_before_final_manifest + workflow_json_bytes > _MAX_WORKFLOW_BYTES:
        raise AssertionError("Phase-5 final workflow would exceed the 500 MB cap")
    write_json_new_atomic(run_root / "workflow.json", workflow)
    final_bytes = _assert_artifact_tree_safe(run_root)
    print(
        "CHIP_PHASE5_EVAL_EXPORT_PASS",
        f"aligned_frames={result.aligned_frames}",
        f"upper_stiff_m={result.stiff.upper_endpoint_mpjpe_m:.9g}",
        f"upper_compliant_m={result.compliant.upper_endpoint_mpjpe_m:.9g}",
        (
            "upper_orientation_compliant_rad="
            f"{result.compliant.upper_endpoint_orientation_rmse_rad:.9g}"
        ),
        f"paired_yield_mean_m={result.compliance_response.displacement_mean_m:.9g}",
        (
            "paired_yield_along_force_m="
            f"{result.compliance_response.displacement_along_force_mean_m:.9g}"
        ),
        f"peak_force_n={result.compliant.peak_force_n:.9g}",
        f"workflow_bytes={final_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
