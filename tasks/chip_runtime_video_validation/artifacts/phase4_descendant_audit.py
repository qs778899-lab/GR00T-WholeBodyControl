#!/usr/bin/env python3
"""Read-only accepted-CHIP audit with the pinned review-task descendant boundary."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_COMMIT = "3dbfb6f211511bb04fedcd326f3265cdafcfa68c"
_EXPECTED_BRANCH = "experiment/chip-runtime-video-validation"
_FORMAL_OUTPUT_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/"
    "runtime_video_validation_v1"
)

_ALLOWED_MODIFIED_PATHS = {
    "gear_sonic/compliance_control/__init__.py",
}
_ALLOWED_ADDED_PREFIXES = (
    "gear_sonic/compliance_control/adapters/sonic/review/",
    "gear_sonic/compliance_control/review/",
    "gear_sonic/config/compliance_review_role/",
    "tasks/chip_runtime_video_validation/",
)
_ALLOWED_ADDED_PATHS = {
    "gear_sonic/config/exp/manager/universal_token/all_modes/"
    "sonic_release_compliance_review.yaml",
    "gear_sonic/config/manager_env/events/tracking/"
    "chip_compliance_review.yaml",
    "gear_sonic/scripts/evaluate_chip_review.py",
    "gear_sonic/scripts/run_chip_review_collect.py",
    "gear_sonic/scripts/validate_chip_review.py",
    "gear_sonic/tests/compliance/test_chip_review_core.py",
    "gear_sonic/tests/compliance/test_chip_review_phase4_diagnostic.py",
    "gear_sonic/tests/compliance/test_chip_review_sonic_phase3.py",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--expected-branch", default=_EXPECTED_BRANCH)
    parser.add_argument("--source-commit", default=_SOURCE_COMMIT)
    parser.add_argument("--refs-snapshot", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--robot-motion", type=Path, required=True)
    parser.add_argument("--smpl-motion-dir", type=Path, required=True)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument(
        "--formal-output-root",
        type=Path,
        default=_FORMAL_OUTPUT_ROOT,
    )
    parser.add_argument("--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument("--skip-gpu-process-check", action="store_true")
    return parser


def _load_legacy_audit(repository: Path):
    path = (
        repository
        / "tasks/chip_compliance_finetune/artifacts/phase6_final_audit.py"
    )
    spec = importlib.util.spec_from_file_location("accepted_chip_phase6_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the accepted CHIP Phase-6 audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_changes(text: str) -> dict[str, str]:
    changes: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        status, separator, path = line.partition("\t")
        if not separator or status not in {"A", "M"} or not path:
            raise AssertionError(f"invalid descendant change: {line}")
        if path in changes:
            raise AssertionError(f"duplicate descendant change: {path}")
        changes[path] = status
    return changes


def _allowed_change(path: str, status: str) -> bool:
    if status == "M":
        return path in _ALLOWED_MODIFIED_PATHS
    return path in _ALLOWED_ADDED_PATHS or path.startswith(_ALLOWED_ADDED_PREFIXES)


def _audit_source_hygiene(repository: Path) -> None:
    roots = (
        repository / "gear_sonic/compliance_control",
        repository / "gear_sonic/tests/compliance",
        repository / "tasks/chip_compliance_finetune",
        repository / "tasks/chip_runtime_video_validation",
    )
    text_suffixes = {".md", ".py", ".yaml", ".yml", ".json", ".txt"}
    for root in roots:
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


def _audit_descendant_repository(args: argparse.Namespace, legacy) -> str:
    repository = legacy._directory(args.repository_root, label="repository root")
    if legacy._git(repository, "branch", "--show-current") != args.expected_branch:
        raise AssertionError("unexpected runtime-review experiment branch")
    if args.source_commit != _SOURCE_COMMIT:
        raise AssertionError("review audit requires the pinned accepted CHIP source")
    legacy._git(repository, "merge-base", "--is-ancestor", args.source_commit, "HEAD")
    for ref_name in (
        "experiment/chip-compliance",
        "origin/experiment/chip-compliance",
    ):
        if legacy._git(repository, "rev-parse", ref_name) != args.source_commit:
            raise AssertionError(f"accepted CHIP ref moved: {ref_name}")

    changes = _parse_changes(
        legacy._git(repository, "diff", "--name-status", args.source_commit, "--")
    )
    untracked = legacy._git(repository, "ls-files", "--others", "--exclude-standard")
    for path in untracked.splitlines():
        if not path:
            continue
        if path in changes:
            raise AssertionError(f"path is both diffed and untracked: {path}")
        changes[path] = "A"
    rejected = {
        path: status
        for path, status in changes.items()
        if not _allowed_change(path, status)
    }
    if rejected:
        raise AssertionError(f"change escapes the runtime-review boundary: {rejected}")

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
    release_diff = legacy._run(
        ["git", "diff", "--exit-code", legacy.BASELINE_COMMIT, "--", *release_paths],
        cwd=repository,
        check=False,
    )
    if release_diff.returncode != 0 or release_diff.stdout or release_diff.stderr:
        raise AssertionError("released source/config/deployment paths changed")
    protected_ref_state = legacy._audit_ref_snapshot(
        repository,
        legacy._regular_file(args.refs_snapshot, label="pre-experiment refs snapshot"),
    )
    legacy._run(["git", "diff", "--check"], cwd=repository)
    legacy._run(["git", "diff", "--cached", "--check"], cwd=repository)
    _audit_source_hygiene(repository)
    if args.formal_output_root.exists() or args.formal_output_root.is_symlink():
        raise AssertionError("formal Phase-5 runtime-review output already exists")
    return protected_ref_state


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = args.repository_root.resolve(strict=True)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    legacy = _load_legacy_audit(repository)
    protected_ref_state = _audit_descendant_repository(args, legacy)
    legacy._audit_assets(args)
    phase4 = legacy._audit_phase4(args)
    phase5 = legacy._audit_phase5(args)
    legacy._audit_workflow_processes()
    if args.skip_gpu_process_check:
        marker = "CHIP_PHASE4_DESCENDANT_STRUCTURAL_AUDIT_PASS"
        gpu_gate = "SKIPPED_NOT_ACCEPTED"
    else:
        legacy._audit_gpu_processes(args.nvidia_smi)
        marker = "CHIP_PHASE4_DESCENDANT_FINAL_AUDIT_PASS"
        gpu_gate = "PASSED"
    print(
        marker,
        f"source_commit={args.source_commit}",
        f"phase4_digest={phase4['digest']}",
        f"phase5_digest={phase5['digest']}",
        f"protected_ref_gate={protected_ref_state}",
        f"gpu_process_gate={gpu_gate}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
