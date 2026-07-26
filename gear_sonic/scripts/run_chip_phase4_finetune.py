#!/usr/bin/env python3
"""Run the bounded SONIC Phase-4 stiff, residual, and resume smokes serially."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
CENTRAL_RUNS_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip"
)
OFFICIAL_CHECKPOINT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/"
    "official_assets/sonic_release/last.pt"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
SAMPLE_ROBOT_MOTION = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/"
    "sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl"
)
SAMPLE_SMPL_MOTION = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/"
    "sample_data/smpl_filtered"
)
COMPATIBILITY_LIBRARY_DIR = Path(
    "/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu"
)
COMPATIBILITY_VULKAN_ICD = Path(
    "/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json"
)
NUM_ENVIRONMENTS = 16
INITIAL_FINAL_STEP = 5
RESUME_FINAL_STEP = 6
MAX_RUN_BYTES = 1_200_000_000
MAX_LOG_BYTES = 64_000_000
MAX_WORKFLOW_BYTES = 2_500_000_000
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)")


@dataclass(frozen=True, slots=True)
class Phase4Assets:
    """Immutable external inputs; generated data never shares these paths."""

    checkpoint: Path = OFFICIAL_CHECKPOINT
    robot_motion: Path = SAMPLE_ROBOT_MOTION
    smpl_motion: Path = SAMPLE_SMPL_MOTION


@dataclass(frozen=True, slots=True)
class Phase4Layout:
    """One collision-free workflow below the centralized CHIP run root."""

    root: Path
    stiff: Path
    initial: Path
    resume: Path


@dataclass(frozen=True, slots=True)
class Phase4Commands:
    """Argument-vector commands for testability and safe subprocess execution."""

    stiff: tuple[str, ...]
    initial: tuple[str, ...]
    resume: tuple[str, ...]


def _resolved_descendant(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"{label} must be a child of {resolved_root}: {resolved}")
    return resolved


def make_layout(run_root: Path | None = None) -> Phase4Layout:
    """Resolve one workflow layout without creating or deleting anything."""

    if run_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = CENTRAL_RUNS_ROOT / f"phase4_{timestamp}"
    root = _resolved_descendant(run_root, CENTRAL_RUNS_ROOT, label="run_root")
    return Phase4Layout(
        root=root,
        stiff=root / "stiff_release_step5",
        initial=root / "compliance_residual_step5",
        resume=root / "compliance_residual_step6_resume",
    )


def _motion_overrides(assets: Phase4Assets) -> list[str]:
    return [
        (
            "++manager_env.commands.motion.motion_lib_cfg.motion_file="
            f"{assets.robot_motion.resolve()}"
        ),
        (
            "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
            f"{assets.smpl_motion.resolve()}"
        ),
    ]


def _output_overrides(experiment_dir: Path) -> list[str]:
    return [
        f"experiment_dir={experiment_dir}",
        f"save_dir={experiment_dir / '.hydra'}",
        f"output_dir={experiment_dir / 'output'}",
    ]


def _compliance_smoke_overrides() -> list[str]:
    return [
        "manager_env.commands.force.enabled=true",
        "manager_env.commands.force.enabled_probability=1.0",
        "manager_env.commands.force.site_probability=1.0",
        "manager_env.commands.force.pulse_interval_range_s=[0.02,0.04]",
        "manager_env.commands.force.compliance_values_m_per_n=[0.02,0.05]",
    ]


def build_commands(
    layout: Phase4Layout,
    assets: Phase4Assets,
    *,
    python_executable: Path,
) -> Phase4Commands:
    """Build the exact release warm-start and residual/resume smoke commands."""

    entrypoint = REPOSITORY_ROOT / "gear_sonic/train_agent_trl.py"
    prefix = (str(python_executable), "-B", str(entrypoint))
    common = (
        f"num_envs={NUM_ENVIRONMENTS}",
        "headless=true",
        "use_wandb=false",
    )
    motion = tuple(_motion_overrides(assets))

    stiff = (
        *prefix,
        "+exp=manager/universal_token/all_modes/sonic_release",
        f"+checkpoint={assets.checkpoint.resolve()}",
        "+resume=false",
        *common,
        f"++algo.config.num_learning_iterations={INITIAL_FINAL_STEP}",
        *motion,
        *_output_overrides(layout.stiff),
    )
    initial = (
        *prefix,
        "+exp=manager/universal_token/all_modes/sonic_release_compliance_finetune_smoke",
        f"+checkpoint={assets.checkpoint.resolve()}",
        "+resume=false",
        *common,
        f"++algo.config.num_learning_iterations={INITIAL_FINAL_STEP}",
        *motion,
        *_output_overrides(layout.initial),
        *_compliance_smoke_overrides(),
        "callbacks.model_save.save_last_frequency=5",
        "chip_phase4.audit_mode=official_init",
        "chip_phase4.expected_source_checkpoint_step=41550",
        "chip_phase4.expected_start_step=0",
        "chip_phase4.expected_final_step=5",
    )
    resume_source = layout.resume / "resume_input_step5.pt"
    resume = (
        *prefix,
        "+exp=manager/universal_token/all_modes/sonic_release_compliance_finetune_smoke",
        f"+checkpoint={resume_source}",
        "+resume=true",
        *common,
        "++algo.config.num_learning_iterations=1",
        *motion,
        *_output_overrides(layout.resume),
        *_compliance_smoke_overrides(),
        "callbacks.model_save.save_last_frequency=1",
        "chip_phase4.audit_mode=branch_resume",
        "chip_phase4.expected_source_checkpoint_step=5",
        "chip_phase4.expected_start_step=5",
        "chip_phase4.expected_final_step=6",
    )
    return Phase4Commands(stiff=stiff, initial=initial, resume=resume)


def compatibility_environment() -> dict[str, str]:
    """Return the validated 580.159.03 userspace environment for child jobs."""

    nvml = COMPATIBILITY_LIBRARY_DIR / "libnvidia-ml.so.580.159.03"
    cuda = COMPATIBILITY_LIBRARY_DIR / "libcuda.so.580.159.03"
    for path in (nvml, cuda, COMPATIBILITY_VULKAN_ICD):
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = os.environ.copy()
    environment.update(
        {
            "LD_LIBRARY_PATH": str(COMPATIBILITY_LIBRARY_DIR),
            "LD_PRELOAD": f"{nvml}:{cuda}",
            "VK_ICD_FILENAMES": str(COMPATIBILITY_VULKAN_ICD),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_bounded(
    command: tuple[str, ...],
    *,
    log_path: Path,
    environment: dict[str, str],
    expected_final_step: int,
) -> float:
    print(f"PHASE4_START step={expected_final_step} log={log_path}", flush=True)
    started = time.monotonic()
    written = 0
    observed_iterations: set[int] = set()
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            for line in process.stdout:
                written += len(line.encode("utf-8"))
                if written > MAX_LOG_BYTES:
                    process.terminate()
                    raise RuntimeError(f"training log exceeded {MAX_LOG_BYTES} bytes")
                stream.write(line)
                normalized = ANSI_ESCAPE.sub("", line)
                for match in ITERATION_PATTERN.finditer(normalized):
                    iteration = int(match.group(1))
                    if iteration not in observed_iterations:
                        observed_iterations.add(iteration)
                        print(f"PHASE4_ITERATION step={iteration}", flush=True)
        return_code = process.wait()
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if return_code != 0:
        tail = "".join(log_path.read_text(encoding="utf-8").splitlines(True)[-40:])
        raise RuntimeError(
            f"training exited with status {return_code}; tail of {log_path}:\n{tail}"
        )
    if expected_final_step not in observed_iterations:
        raise AssertionError(
            f"training log did not reach learning iteration {expected_final_step}"
        )
    elapsed = time.monotonic() - started
    print(f"PHASE4_PROCESS_PASS step={expected_final_step} elapsed_s={elapsed:.3f}", flush=True)
    return elapsed


def _directory_usage_bytes(root: Path) -> tuple[int, int]:
    total = 0
    largest_log = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(directory) / filename
            try:
                size = path.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
            total += size
            if path.suffix == ".log":
                largest_log = max(largest_log, size)
    return total, largest_log


def _validate_assets(assets: Phase4Assets) -> None:
    for path in (assets.checkpoint, assets.robot_motion):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not assets.smpl_motion.is_dir():
        raise NotADirectoryError(assets.smpl_motion)
    exact_paths = (
        (assets.checkpoint, OFFICIAL_CHECKPOINT, "checkpoint"),
        (assets.robot_motion, SAMPLE_ROBOT_MOTION, "robot motion"),
        (assets.smpl_motion, SAMPLE_SMPL_MOTION, "SMPL motion"),
    )
    for actual, expected, label in exact_paths:
        if actual.resolve() != expected.resolve():
            raise ValueError(f"Phase-4 acceptance requires pinned {label}: {expected}")
    digest = hashlib.sha256()
    with assets.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != OFFICIAL_CHECKPOINT_SHA256:
        raise AssertionError("official SONIC checkpoint SHA-256 mismatch")


def _audit_checkpoint(
    *,
    checkpoint: Path,
    report: Path,
    official: Path,
    expected_step: int,
    source_branch: Path | None = None,
) -> dict[str, Any]:
    from gear_sonic.compliance_control.adapters.sonic.phase4_training import (
        audit_sonic_phase4_checkpoint,
    )

    return audit_sonic_phase4_checkpoint(
        checkpoint_path=checkpoint,
        official_checkpoint_path=official,
        audit_report_path=report,
        expected_step=expected_step,
        source_branch_checkpoint_path=source_branch,
        max_run_bytes=MAX_RUN_BYTES,
        max_log_bytes=MAX_LOG_BYTES,
    )


def run_workflow(
    layout: Phase4Layout,
    assets: Phase4Assets,
    commands: Phase4Commands,
) -> dict[str, Any]:
    """Run all Phase-4 jobs serially and retain independent step-5 evidence."""

    _validate_assets(assets)
    if layout.root.exists():
        raise FileExistsError(f"refusing to reuse Phase-4 run root: {layout.root}")
    layout.root.mkdir(parents=True)
    for directory in (layout.stiff, layout.initial):
        directory.mkdir()
    environment = compatibility_environment()
    manifest_path = layout.root / "workflow.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "root": str(layout.root),
        "resume_boundary": (
            "strict model/optimizer/scheduler/global-step restoration; environment "
            "payload is loaded, but CPU/CUDA/Python/NumPy RNG and private compliance "
            "generator/countdown state are not checkpointed, so trajectory-bitwise "
            "resume is not claimed"
        ),
        "commands": {
            "stiff": list(commands.stiff),
            "initial": list(commands.initial),
            "resume": list(commands.resume),
        },
    }
    _atomic_json(manifest_path, manifest)
    try:
        stiff_seconds = _run_bounded(
            commands.stiff,
            log_path=layout.root / "stiff_release.log",
            environment=environment,
            expected_final_step=INITIAL_FINAL_STEP,
        )
        initial_seconds = _run_bounded(
            commands.initial,
            log_path=layout.root / "compliance_step5.log",
            environment=environment,
            expected_final_step=INITIAL_FINAL_STEP,
        )
        initial_checkpoint = layout.initial / "last.pt"
        step5_audit = _audit_checkpoint(
            checkpoint=initial_checkpoint,
            report=layout.initial / "phase4_audit.json",
            official=assets.checkpoint,
            expected_step=INITIAL_FINAL_STEP,
        )
        _atomic_json(layout.initial / "step5_checkpoint_audit.json", step5_audit)

        layout.resume.mkdir()
        resume_source = layout.resume / "resume_input_step5.pt"
        resume_source.symlink_to(initial_checkpoint)
        resume_seconds = _run_bounded(
            commands.resume,
            log_path=layout.root / "compliance_step6_resume.log",
            environment=environment,
            expected_final_step=RESUME_FINAL_STEP,
        )
        step6_audit = _audit_checkpoint(
            checkpoint=layout.resume / "last.pt",
            report=layout.resume / "phase4_audit.json",
            official=assets.checkpoint,
            expected_step=RESUME_FINAL_STEP,
            source_branch=initial_checkpoint,
        )
        _atomic_json(layout.resume / "step6_checkpoint_audit.json", step6_audit)

        total_bytes, largest_log = _directory_usage_bytes(layout.root)
        if total_bytes > MAX_WORKFLOW_BYTES:
            raise AssertionError(f"Phase-4 workflow exceeds byte budget: {total_bytes}")
        if largest_log > MAX_LOG_BYTES:
            raise AssertionError(f"Phase-4 workflow log exceeds byte budget: {largest_log}")
        manifest.update(
            {
                "status": "PASSED",
                "durations_s": {
                    "stiff": stiff_seconds,
                    "initial": initial_seconds,
                    "resume": resume_seconds,
                },
                "step5_audit": step5_audit,
                "step6_audit": step6_audit,
                # This value is intentionally measured before the final,
                # larger manifest replaces the RUNNING document.  Naming the
                # boundary avoids a self-referential size claim whose decimal
                # width can itself change the final file size.
                "workflow_bytes_before_final_manifest": total_bytes,
                "largest_log_bytes": largest_log,
            }
        )
        _atomic_json(manifest_path, manifest)
        final_total_bytes, final_largest_log = _directory_usage_bytes(layout.root)
        if final_total_bytes > MAX_WORKFLOW_BYTES:
            raise AssertionError(
                "Phase-4 final workflow exceeds byte budget: "
                f"{final_total_bytes}"
            )
        if final_largest_log > MAX_LOG_BYTES:
            raise AssertionError(
                "Phase-4 final workflow log exceeds byte budget: "
                f"{final_largest_log}"
            )
        print(f"CHIP_PHASE4_FINETUNE_PASS root={layout.root}", flush=True)
        return manifest
    except BaseException as error:
        manifest.update(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        help=f"new child directory below {CENTRAL_RUNS_ROOT}",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--checkpoint", type=Path, default=OFFICIAL_CHECKPOINT)
    parser.add_argument("--motion-file", type=Path, default=SAMPLE_ROBOT_MOTION)
    parser.add_argument("--smpl-motion-dir", type=Path, default=SAMPLE_SMPL_MOTION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the three commands without creating files or starting Isaac Lab",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    layout = make_layout(args.run_root)
    assets = Phase4Assets(
        checkpoint=args.checkpoint,
        robot_motion=args.motion_file,
        smpl_motion=args.smpl_motion_dir,
    )
    commands = build_commands(layout, assets, python_executable=args.python)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "root": str(layout.root),
                    "stiff": shlex.join(commands.stiff),
                    "initial": shlex.join(commands.initial),
                    "resume": shlex.join(commands.resume),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run_workflow(layout, assets, commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
