#!/usr/bin/env python3
"""Read-only source, asset, tool, and fresh-output audit for CHIP review work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


SOURCE_COMMIT = "3dbfb6f211511bb04fedcd326f3265cdafcfa68c"
EXPECTED_BRANCH = "experiment/chip-runtime-video-validation"
EXPECTED_HASHES = {
    "official_assets/sonic_release/last.pt":
        "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909",
    "official_assets/sample_data/robot_filtered/210531/"
    "walk_forward_amateur_001__A001.pkl":
        "005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b",
    "official_assets/sample_data/robot_filtered/210531/"
    "walk_forward_amateur_001__A001_M.pkl":
        "7d9ec8a24acbb952cfce2048e2d3b5c156e8ae0c43e32443eb5ea42cbb22038e",
    "official_assets/sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl":
        "f31a00cd23cedb9b6cc50805d912276234a35a40678529d726df3b1dec3682d8",
    "official_assets/sample_data/smpl_filtered/walk_forward_amateur_001__A001_M.pkl":
        "49cbf3c604f78952474d3bcecb6bbc0b4a136eab78dc3ab8580869594383bb4f",
    "runs/chip/phase4_acceptance_resume_fix/compliance_residual_step6_resume/last.pt":
        "71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02",
    "runs/chip/phase5_acceptance/export/compliance_residual.onnx":
        "a4ccbc9e216dd97fe5181a12f5ded7a9e544c1a477fd114c909b8564bc83e2f3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo_root.resolve(strict=True)
    runs_root = args.runs_root.resolve(strict=True)
    output_requested = Path(os.path.abspath(args.output_root))
    if os.path.lexists(output_requested):
        raise FileExistsError(f"formal output must be fresh: {output_requested}")
    if output_requested.parent.resolve(strict=True) != runs_root:
        raise ValueError("formal output must be a direct child of --runs-root")

    branch = _git(repo, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"unexpected branch: {branch}")
    if _git(repo, "merge-base", "HEAD", SOURCE_COMMIT) != SOURCE_COMMIT:
        raise RuntimeError("worktree does not descend from the pinned CHIP source")
    if _git(repo, "rev-parse", SOURCE_COMMIT) != SOURCE_COMMIT:
        raise RuntimeError("pinned CHIP source commit is unavailable")

    artifact_root = runs_root.parents[1]
    observed = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = artifact_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")
        observed[relative] = actual

    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable is None:
            raise RuntimeError(f"{name} is unavailable")
        version = subprocess.run(
            (executable, "-version"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        tools[name] = {"path": executable, "version": version}

    print(
        "CHIP_RUNTIME_VIDEO_PHASE1_CONTRACT_PASS",
        json.dumps(
            {
                "branch": branch,
                "source_commit": SOURCE_COMMIT,
                "asset_hashes": observed,
                "tools": tools,
                "fresh_output_root": str(output_requested),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
