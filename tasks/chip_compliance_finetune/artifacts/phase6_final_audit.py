#!/usr/bin/env python3
"""Read-only final audit for the accepted CHIP finetune/evaluation handoff.

The audit deliberately accepts every repository, asset, and run location on the
command line.  Only the golden hashes and baseline commit are experiment
specific, so the filesystem/provenance checks can be reused after moving the
worktree or adapting the compliance package to another universal tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


BASELINE_COMMIT = "4141c34280abb67c82e115342a8720f4a83d750d"
PHASE5_COMMIT = "c925a0da115d1d6e0cc296c4a94b00a57c6461b8"
PHASE6_ALLOWED_CHANGES = {
    "gear_sonic/scripts/run_chip_compliance_smoke.py": "M",
    "gear_sonic/scripts/run_chip_phase3_shape_smoke.py": "M",
    "gear_sonic/scripts/run_chip_phase5_rollout.py": "M",
    "gear_sonic/tests/compliance/test_phase6_entrypoint_help.py": "A",
    "tasks/chip_compliance_finetune/artifacts/phase6_final_audit.py": "A",
    "tasks/chip_compliance_finetune/phase6_handoff.md": "A",
    "tasks/chip_compliance_finetune/log.md": "M",
    "tasks/chip_compliance_finetune/plan.md": "M",
    "tasks/chip_compliance_finetune/status.md": "M",
    "tasks/chip_compliance_finetune/test_matrix.md": "M",
}
OFFICIAL_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
OFFICIAL_CONFIG_SHA256 = (
    "f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7"
)
ROBOT_MOTION_SHA256 = (
    "005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b"
)
SMPL_MOTION_HASHES = {
    "walk_forward_amateur_001__A001.pkl": (
        "f31a00cd23cedb9b6cc50805d912276234a35a40678529d726df3b1dec3682d8"
    ),
    "walk_forward_amateur_001__A001_M.pkl": (
        "49cbf3c604f78952474d3bcecb6bbc0b4a136eab78dc3ab8580869594383bb4f"
    ),
}
PHASE4_STEP5_SHA256 = (
    "b306e1f233be6cd05682afcd8fee6d690611bebe8020a331d50875fe8a48d82a"
)
PHASE4_STEP6_SHA256 = (
    "71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02"
)
PHASE5_ONNX_SHA256 = (
    "a4ccbc9e216dd97fe5181a12f5ded7a9e544c1a477fd114c909b8564bc83e2f3"
)

# ``tree_layout_digest_v1`` hashes sorted directory names, relative file
# names/sizes/content hashes, and symlink destinations normalized relative to
# the audited root.  It is stable if the complete evidence directory is moved.
PHASE4_TREE_DIGEST = (
    "34cba4405dee146c7dd5f29d4731001737e8ae85f6f4d79e3928317b5bb02503"
)
PHASE5_TREE_DIGEST = (
    "9efef42178353072faa457f49934c6fa67ffbf852628470e1f9bbc384046c81e"
)
PHASE4_COUNTS = (31, 9, 1)
PHASE5_COUNTS = (14, 3, 0)
PHASE4_BYTES = 318_016_496
PHASE5_BYTES = 1_655_744
PHASE4_LARGEST_LOG_BYTES = 55_249
PHASE5_MAX_LOG_BYTES = 64_000_000
EXPECTED_RESUME_LINK = (
    "compliance_residual_step6_resume/resume_input_step5.pt"
)
EXPECTED_RESUME_TARGET = "compliance_residual_step5/last.pt"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument("--phase5-commit", default=PHASE5_COMMIT)
    parser.add_argument("--expected-branch", default="experiment/chip-compliance")
    parser.add_argument("--refs-snapshot", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--smpl-motion-dir", type=Path, required=True)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument(
        "--nvidia-smi",
        type=Path,
        default=Path("/usr/bin/nvidia-smi"),
        help="compatibility-environment nvidia-smi used for the idle-GPU gate",
    )
    parser.add_argument(
        "--skip-gpu-process-check",
        action="store_true",
        help="skip only NVML; cannot emit the final acceptance marker",
    )
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AssertionError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise AssertionError(f"{label} must be a real directory: {path}")
    return path.resolve()


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(_regular_file(path, label=label))
    if actual != expected:
        raise AssertionError(f"{label} SHA-256 mismatch: {actual}")


def _relative_internal_target(link: Path, root: Path) -> str:
    try:
        return link.resolve(strict=True).relative_to(root).as_posix()
    except (FileNotFoundError, ValueError) as error:
        raise AssertionError(f"symlink escapes or is broken: {link}") from error


def tree_layout_digest_v1(root: Path) -> dict[str, Any]:
    """Hash one bounded tree without following links or depending on its root."""

    root = _directory(root, label="artifact root")
    digest = hashlib.sha256()
    files = directories = symlinks = total_bytes = largest_log = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = _relative_internal_target(path, root)
            digest.update(
                b"L\0"
                + relative.encode("utf-8")
                + b"\0"
                + target.encode("utf-8")
                + b"\n"
            )
            total_bytes += path.lstat().st_size
            symlinks += 1
        elif path.is_file():
            size = path.stat().st_size
            content_hash = _sha256(path)
            digest.update(
                b"F\0"
                + relative.encode("utf-8")
                + b"\0"
                + str(size).encode("ascii")
                + b"\0"
                + content_hash.encode("ascii")
                + b"\n"
            )
            total_bytes += size
            files += 1
            if path.suffix == ".log":
                largest_log = max(largest_log, size)
        elif path.is_dir():
            digest.update(b"D\0" + relative.encode("utf-8") + b"\n")
            directories += 1
        else:
            raise AssertionError(f"unsupported artifact entry: {path}")
    return {
        "digest": digest.hexdigest(),
        "files": files,
        "directories": directories,
        "symlinks": symlinks,
        "bytes": total_bytes,
        "largest_log_bytes": largest_log,
    }


def _assert_tree(
    root: Path,
    *,
    expected_digest: str,
    expected_counts: tuple[int, int, int],
    expected_bytes: int,
    label: str,
) -> dict[str, Any]:
    result = tree_layout_digest_v1(root)
    counts = (result["files"], result["directories"], result["symlinks"])
    if result["digest"] != expected_digest:
        raise AssertionError(f"{label} tree digest mismatch: {result['digest']}")
    if counts != expected_counts:
        raise AssertionError(f"{label} inventory mismatch: {counts}")
    if result["bytes"] != expected_bytes:
        raise AssertionError(f"{label} byte count mismatch: {result['bytes']}")
    return result


def _audit_assets(args: argparse.Namespace) -> None:
    _require_hash(
        args.official_checkpoint,
        OFFICIAL_CHECKPOINT_SHA256,
        label="official checkpoint",
    )
    _require_hash(args.official_config, OFFICIAL_CONFIG_SHA256, label="official config")
    _require_hash(args.robot_motion, ROBOT_MOTION_SHA256, label="robot motion")
    smpl_root = _directory(args.smpl_motion_dir, label="SMPL motion directory")
    actual = {
        path.relative_to(smpl_root).as_posix(): _sha256(path)
        for path in sorted(smpl_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if actual != SMPL_MOTION_HASHES:
        raise AssertionError(f"SMPL motion inventory/hash mismatch: {actual}")
    if any(path.is_symlink() for path in smpl_root.rglob("*")):
        raise AssertionError("SMPL motion directory contains a symlink")


def _audit_phase4(args: argparse.Namespace) -> dict[str, Any]:
    root = _directory(args.phase4_root, label="Phase-4 accepted root")
    layout = _assert_tree(
        root,
        expected_digest=PHASE4_TREE_DIGEST,
        expected_counts=PHASE4_COUNTS,
        expected_bytes=PHASE4_BYTES,
        label="Phase-4 accepted artifact",
    )
    if layout["largest_log_bytes"] != PHASE4_LARGEST_LOG_BYTES:
        raise AssertionError("Phase-4 largest-log golden mismatch")
    links = [path for path in root.rglob("*") if path.is_symlink()]
    expected_link = root / EXPECTED_RESUME_LINK
    if links != [expected_link]:
        raise AssertionError(f"unexpected Phase-4 symlink inventory: {links}")
    if _relative_internal_target(expected_link, root) != EXPECTED_RESUME_TARGET:
        raise AssertionError("Phase-4 resume link target mismatch")

    workflow = json.loads((root / "workflow.json").read_text(encoding="utf-8"))
    if workflow.get("status") != "PASSED" or workflow.get("schema_version") != 1:
        raise AssertionError("Phase-4 workflow is not a passed schema-v1 artifact")
    if Path(workflow.get("root", "")).resolve() != root:
        raise AssertionError("Phase-4 workflow root provenance mismatch")
    if workflow.get("resume_boundary", "").find("trajectory-bitwise") < 0:
        raise AssertionError("Phase-4 resume limitation is missing")

    step5 = root / "compliance_residual_step5/last.pt"
    step6 = root / "compliance_residual_step6_resume/last.pt"
    _require_hash(step5, PHASE4_STEP5_SHA256, label="Phase-4 step-5 checkpoint")
    _require_hash(step6, PHASE4_STEP6_SHA256, label="Phase-4 step-6 checkpoint")

    from gear_sonic.compliance_control.adapters.sonic.phase4_training import (
        audit_sonic_phase4_checkpoint,
    )

    step5_result = audit_sonic_phase4_checkpoint(
        checkpoint_path=step5,
        official_checkpoint_path=args.official_checkpoint,
        audit_report_path=root / "compliance_residual_step5/phase4_audit.json",
        expected_step=5,
    )
    step6_result = audit_sonic_phase4_checkpoint(
        checkpoint_path=step6,
        official_checkpoint_path=args.official_checkpoint,
        audit_report_path=root / "compliance_residual_step6_resume/phase4_audit.json",
        expected_step=6,
        source_branch_checkpoint_path=step5,
    )
    for expected_step, result in ((5, step5_result), (6, step6_result)):
        if (
            result.get("step") != expected_step
            or result.get("legacy_policy_tensors") != 55
            or result.get("legacy_value_tensors") != 17
            or result.get("actor_residual_tensors") != 6
            or result.get("critic_residual_tensors") != 6
            or result.get("optimizer_parameter_count") != 12
            or result.get("trainable_scalar_count") != 770_753
        ):
            raise AssertionError(f"Phase-4 step-{expected_step} semantic audit mismatch")
    return layout


def _audit_phase5(args: argparse.Namespace) -> dict[str, Any]:
    root = _directory(args.phase5_root, label="Phase-5 accepted root")
    layout = _assert_tree(
        root,
        expected_digest=PHASE5_TREE_DIGEST,
        expected_counts=PHASE5_COUNTS,
        expected_bytes=PHASE5_BYTES,
        label="Phase-5 accepted artifact",
    )
    if layout["largest_log_bytes"] > PHASE5_MAX_LOG_BYTES:
        raise AssertionError("Phase-5 log exceeds the accepted capacity limit")
    onnx_path = root / "export/compliance_residual.onnx"
    _require_hash(onnx_path, PHASE5_ONNX_SHA256, label="Phase-5 residual ONNX")
    workflow = json.loads((root / "workflow.json").read_text(encoding="utf-8"))
    if (
        workflow.get("complete") is not True
        or workflow.get("marker") != "CHIP_PHASE5_EVAL_EXPORT_PASS"
        or workflow.get("evaluation_claim")
        != "chain_validation_not_performance_proof"
        or workflow.get("steps") != 300
        or workflow.get("onnx_sha256") != PHASE5_ONNX_SHA256
    ):
        raise AssertionError("Phase-5 workflow provenance/claim mismatch")
    if Path(workflow.get("run_root", "")).resolve() != root:
        raise AssertionError("Phase-5 workflow root provenance mismatch")
    checkpoint = Path(workflow.get("checkpoint", "")).resolve()
    expected_checkpoint = (
        args.phase4_root / "compliance_residual_step6_resume/last.pt"
    ).resolve()
    if checkpoint != expected_checkpoint or _sha256(checkpoint) != PHASE4_STEP6_SHA256:
        raise AssertionError("Phase-5 checkpoint provenance mismatch")
    metrics = json.loads((root / "paired_metrics.json").read_text(encoding="utf-8"))
    if (
        metrics.get("passed") is not True
        or metrics.get("aligned_frames") != 300
        or not all(metrics.get("checks", {}).values())
    ):
        raise AssertionError("Phase-5 paired metric golden did not pass")
    return layout


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    result = _run(["git", *arguments], cwd=repository, check=check)
    return result.stdout.strip()


def _audit_repository(args: argparse.Namespace) -> None:
    repository = _directory(args.repository_root, label="repository root")
    if _git(repository, "branch", "--show-current") != args.expected_branch:
        raise AssertionError("unexpected CHIP experiment branch")
    if _git(repository, "cat-file", "-t", args.baseline_commit) != "commit":
        raise AssertionError("baseline object is not a commit")
    _git(repository, "merge-base", "--is-ancestor", args.baseline_commit, "HEAD")
    if _git(repository, "cat-file", "-t", args.phase5_commit) != "commit":
        raise AssertionError("Phase-5 audit boundary is not a commit")
    _git(repository, "merge-base", "--is-ancestor", args.phase5_commit, "HEAD")

    changes = _git(repository, "diff", "--name-status", args.baseline_commit, "--")
    for line in changes.splitlines():
        if not line:
            continue
        status, _, path = line.partition("\t")
        if status != "A":
            raise AssertionError(f"baseline path was modified instead of extended: {line}")
        if not path.startswith(
            (
                "gear_sonic/compliance_control/",
                "gear_sonic/config/",
                "gear_sonic/scripts/",
                "gear_sonic/tests/compliance/",
                "tasks/chip_compliance_finetune/",
            )
        ):
            raise AssertionError(f"addition is outside CHIP scope: {path}")

    release_paths = (
        "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml",
        "gear_sonic/config/actor_critic",
        "gear_sonic/config/aux_losses",
        "gear_sonic/config/manager_env/observations/policy",
        "gear_sonic/config/manager_env/observations/critic",
        "gear_sonic/config/manager_env/rewards",
        "gear_sonic/envs/manager_env/mdp/observations.py",
        "gear_sonic/envs/manager_env/mdp/rewards.py",
        "gear_sonic/trl/modules/universal_token_modules.py",
        "gear_sonic/trl/trainer/ppo_trainer.py",
        "gear_sonic/trl/trainer/ppo_trainer_aux_loss.py",
        "gear_sonic_deploy/policy/release",
    )
    release_diff = _run(
        ["git", "diff", "--exit-code", args.baseline_commit, "--", *release_paths],
        cwd=repository,
        check=False,
    )
    if release_diff.returncode != 0 or release_diff.stdout or release_diff.stderr:
        raise AssertionError("released source/config/deployment paths differ from baseline")

    snapshot = _regular_file(args.refs_snapshot, label="pre-experiment refs snapshot")
    for line in snapshot.read_text(encoding="utf-8").splitlines():
        ref_name, expected = line.split()
        actual = _git(repository, "show-ref", "--verify", "--hash", ref_name)
        if actual != expected:
            raise AssertionError(f"existing ref moved: {ref_name} {actual} != {expected}")

    # The baseline audit proves that CHIP is additive.  This stricter boundary
    # prevents Phase 6 from changing any Phase-1..5 implementation accepted at
    # PHASE5_COMMIT while still allowing its explicit handoff/help paths.
    phase6_changes: dict[str, str] = {}
    phase6_diff = _git(repository, "diff", "--name-status", args.phase5_commit, "--")
    for line in phase6_diff.splitlines():
        if not line:
            continue
        status, _, path = line.partition("\t")
        if status not in {"A", "M"} or not path:
            raise AssertionError(f"invalid Phase-6 change type: {line}")
        if path in phase6_changes:
            raise AssertionError(f"duplicate Phase-6 changed path: {path}")
        phase6_changes[path] = status
    untracked = _git(repository, "ls-files", "--others", "--exclude-standard")
    for path in untracked.splitlines():
        if not path:
            continue
        if path in phase6_changes:
            raise AssertionError(f"Phase-6 path is both diffed and untracked: {path}")
        phase6_changes[path] = "A"
    if phase6_changes != PHASE6_ALLOWED_CHANGES:
        unexpected = sorted(set(phase6_changes) - set(PHASE6_ALLOWED_CHANGES))
        missing = sorted(set(PHASE6_ALLOWED_CHANGES) - set(phase6_changes))
        wrong_status = sorted(
            path
            for path in set(phase6_changes) & set(PHASE6_ALLOWED_CHANGES)
            if phase6_changes[path] != PHASE6_ALLOWED_CHANGES[path]
        )
        raise AssertionError(
            "Phase-6 diff is not the exact Phase-5-head allowlist: "
            f"unexpected={unexpected}, missing={missing}, "
            f"wrong_status={wrong_status}"
        )

    hygiene_roots = (
        repository / "gear_sonic/compliance_control",
        repository / "gear_sonic/tests/compliance",
        repository / "tasks/chip_compliance_finetune",
    )
    text_suffixes = {".md", ".py", ".yaml", ".yml", ".json", ".txt"}
    for root in hygiene_roots:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise AssertionError(f"source/task tree contains a symlink: {path}")
            if path.name in {"__pycache__", ".pytest_cache"}:
                raise AssertionError(f"cache directory remains: {path}")
            if path.suffix in {".pyc", ".pyo", ".tmp", ".part"}:
                raise AssertionError(f"cache/temporary file remains: {path}")
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            if not text.endswith("\n") or text.endswith("\n\n"):
                raise AssertionError(f"invalid final newline: {path}")
            if any(line != line.rstrip() for line in text.splitlines()):
                raise AssertionError(f"trailing whitespace: {path}")

    script_root = repository / "gear_sonic/scripts"
    for path in script_root.glob("*chip*.py"):
        if path.is_symlink() or not path.is_file():
            raise AssertionError(f"CHIP script is not a regular file: {path}")
        text = path.read_text(encoding="utf-8")
        if not text.endswith("\n") or text.endswith("\n\n"):
            raise AssertionError(f"invalid final newline: {path}")
        if any(line != line.rstrip() for line in text.splitlines()):
            raise AssertionError(f"trailing whitespace: {path}")
    script_cache = script_root / "__pycache__"
    if script_cache.is_dir() and any(script_cache.glob("*chip*.py[co]")):
        raise AssertionError("CHIP entrypoint bytecode remains in gear_sonic/scripts")
    for suffix in (".tmp", ".part"):
        if any(script_root.glob(f"*chip*{suffix}")):
            raise AssertionError(f"CHIP entrypoint temporary {suffix} file remains")

    _run(["git", "diff", "--check"], cwd=repository)
    _run(["git", "diff", "--cached", "--check"], cwd=repository)


def _audit_workflow_processes() -> None:
    markers = (
        "train_agent_trl.py",
        "run_chip_compliance_smoke.py",
        "run_chip_phase3_shape_smoke.py",
        "run_chip_phase4_finetune.py",
        "run_chip_phase5_eval_export.py",
        "run_chip_phase5_rollout.py",
        "isaac-sim",
        "isaacsim",
        "omni.kit.app",
        "/kit/kit",
    )
    live: list[str] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in markers):
            live.append(f"{process.name}:{command.strip()}")
    if live:
        raise AssertionError(f"training/Isaac workflow remains active: {live}")


def _audit_gpu_processes(nvidia_smi: Path) -> None:
    if nvidia_smi.is_symlink() or not nvidia_smi.is_file():
        raise AssertionError(f"nvidia-smi is unavailable: {nvidia_smi}")
    device_result = _run(
        [str(nvidia_smi), "--query-gpu=uuid", "--format=csv,noheader"],
        check=False,
    )
    if device_result.returncode != 0 or not device_result.stdout.strip():
        raise AssertionError(
            "nvidia-smi did not resolve a real GPU through the compatibility "
            f"environment: {device_result.stderr.strip()}"
        )
    result = _run(
        [
            str(nvidia_smi),
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"nvidia-smi process query failed: {result.stderr.strip()}")
    if result.stdout.strip():
        raise AssertionError(f"GPU compute application remains: {result.stdout.strip()}")


def main() -> int:
    args = _parse_args()
    if args.baseline_commit != BASELINE_COMMIT:
        raise AssertionError("Phase-6 requires the pinned official SONIC baseline")
    if args.phase5_commit != PHASE5_COMMIT:
        raise AssertionError("Phase-6 requires the pinned accepted Phase-5 commit")
    repository = _directory(args.repository_root, label="repository root")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    _audit_repository(args)
    _audit_assets(args)
    phase4 = _audit_phase4(args)
    phase5 = _audit_phase5(args)
    _audit_workflow_processes()
    if args.skip_gpu_process_check:
        print(
            "CHIP_PHASE6_STRUCTURAL_AUDIT_PASS",
            f"phase4_digest={phase4['digest']}",
            f"phase4_bytes={phase4['bytes']}",
            f"phase5_digest={phase5['digest']}",
            f"phase5_bytes={phase5['bytes']}",
            "workflow_process_gate=PASSED",
            "gpu_process_gate=SKIPPED_NOT_ACCEPTED",
            flush=True,
        )
        return 0
    _audit_gpu_processes(args.nvidia_smi)
    print(
        "CHIP_PHASE6_FINAL_AUDIT_PASS",
        f"phase4_digest={phase4['digest']}",
        f"phase4_bytes={phase4['bytes']}",
        f"phase5_digest={phase5['digest']}",
        f"phase5_bytes={phase5['bytes']}",
        f"official_sha256={OFFICIAL_CHECKPOINT_SHA256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
