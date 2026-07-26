"""Filesystem ownership for compliance finetuning artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MOTION_COMPLIANCE_RUNS_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion"
)
OFFICIAL_SONIC_RELEASE_CHECKPOINT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/"
    "sonic_release/last.pt"
)
OFFICIAL_SAMPLE_ROBOT_MOTION = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/"
    "sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl"
)
OFFICIAL_SAMPLE_SMPL_MOTION = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/"
    "sample_data/smpl_filtered"
)


def validate_motion_compliance_run_path(
    path: str | os.PathLike[str],
    *,
    runs_root: str | os.PathLike[str] = MOTION_COMPLIANCE_RUNS_ROOT,
) -> Path:
    """Resolve a path and require it to be below the owned central runs root."""

    resolved_root = Path(runs_root).expanduser().resolve(strict=False)
    resolved_path = Path(path).expanduser().resolve(strict=False)
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(
            "motion-compliance artifacts must be written below "
            f"{resolved_root}; got {resolved_path}"
        )
    return resolved_path


def validate_distinct_artifact_paths(**named_paths: Any) -> dict[str, Path]:
    """Resolve artifact paths and reject any read/write target collision."""

    resolved = {
        name: Path(path).expanduser().resolve(strict=False)
        for name, path in named_paths.items()
    }
    owners: dict[Path, str] = {}
    for name, path in resolved.items():
        previous = owners.get(path)
        if previous is not None:
            raise ValueError(
                f"artifact paths must be distinct: {previous} and {name} both resolve to {path}"
            )
        owners[path] = name
    return resolved
