"""Bounded, non-pickle I/O for frame-aligned evaluation traces."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
import torch

from ..core import (
    AlignedTrackingTrace,
    CartesianFrameKind,
    CartesianFrameSpec,
    CartesianRotation,
    PairedEvaluationResult,
)


_TRACE_SCHEMA_VERSION = 2
_MAX_TRACE_METADATA_BYTES = 1_000_000
_MAX_TRACE_UNCOMPRESSED_BYTES = 64_000_000
_TENSOR_FIELDS = (
    "sample_index",
    "episode_id",
    "motion_id",
    "reference_frame",
    "time_s",
    "valid",
    "reference_positions_w",
    "actual_positions_w",
    "reference_positions_local",
    "actual_positions_local",
    "reference_site_positions_w",
    "actual_site_positions_w",
    "reference_site_quaternions_wxyz",
    "actual_site_quaternions_wxyz",
    "force_on_robot_w",
    "enabled",
    "site_mask",
    "compliance_m_per_n",
)


def _metadata_path(trace_path: Path) -> Path:
    if trace_path.suffix != ".npz":
        raise ValueError("evaluation trace path must end in .npz")
    return trace_path.with_suffix(".json")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _write_json_temporary(path: Path, payload: Any) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish_new(temporary_path: Path, destination: Path) -> None:
    """Atomically add a file without following or replacing an existing path."""

    os.link(temporary_path, destination, follow_symlinks=False)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Write one final-newline JSON document via same-directory replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _write_json_temporary(path, payload)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_new_atomic(path: str | Path, payload: Any) -> None:
    """Atomically create JSON while refusing files, symlinks, and broken links."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(path):
        raise FileExistsError(path)
    temporary_path = _write_json_temporary(path, payload)
    try:
        _publish_new(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_tracking_trace(trace: AlignedTrackingTrace, path: str | Path) -> None:
    """Persist tensors in NPZ and schema/name metadata in adjacent JSON."""

    path = Path(path)
    metadata_path = _metadata_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(path) or _lexists(metadata_path):
        raise FileExistsError("trace NPZ or metadata path already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    metadata_temporary_path: Path | None = None
    published_trace = False
    arrays = {
        name: getattr(trace, name).detach().cpu().numpy()
        for name in _TENSOR_FIELDS
    }
    try:
        np.savez_compressed(temporary_path, **arrays)
        metadata_temporary_path = _write_json_temporary(
            metadata_path,
            {
                "schema_version": _TRACE_SCHEMA_VERSION,
                "mode": trace.mode,
                "body_names": list(trace.body_names),
                "site_names": list(trace.site_names),
                "local_frame": {
                    "kind": trace.local_frame.kind.value,
                    "anchor": trace.local_frame.anchor,
                    "rotation": trace.local_frame.rotation.value,
                },
                "fell": trace.fell,
                "horizon_reached": trace.horizon_reached,
                "termination_sample": trace.termination_sample,
                "tensor_fields": list(_TENSOR_FIELDS),
            },
        )
        _publish_new(temporary_path, path)
        published_trace = True
        _publish_new(metadata_temporary_path, metadata_path)
    except BaseException:
        if published_trace and path.exists():
            if path.stat().st_ino == temporary_path.stat().st_ino:
                path.unlink()
        raise
    finally:
        temporary_path.unlink(missing_ok=True)
        if metadata_temporary_path is not None:
            metadata_temporary_path.unlink(missing_ok=True)


def load_tracking_trace(
    path: str | Path,
    *,
    max_uncompressed_bytes: int = _MAX_TRACE_UNCOMPRESSED_BYTES,
) -> AlignedTrackingTrace:
    """Load a trace with pickle disabled and re-run the complete core contract."""

    path = Path(path)
    metadata_path = _metadata_path(path)
    if path.is_symlink() or metadata_path.is_symlink():
        raise ValueError("evaluation traces must not use symlinks")
    if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes <= 0:
        raise ValueError("max_uncompressed_bytes must be a positive integer")
    if metadata_path.stat().st_size > _MAX_TRACE_METADATA_BYTES:
        raise ValueError("evaluation trace metadata exceeds the byte cap")
    try:
        with zipfile.ZipFile(path) as archive:
            uncompressed_bytes = sum(item.file_size for item in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError("evaluation trace NPZ is not a valid ZIP archive") from exc
    if uncompressed_bytes > max_uncompressed_bytes:
        raise ValueError("evaluation trace exceeds the uncompressed byte cap")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != _TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation trace schema version")
    if metadata.get("tensor_fields") != list(_TENSOR_FIELDS):
        raise ValueError("evaluation trace tensor schema mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(_TENSOR_FIELDS):
            raise ValueError("evaluation trace NPZ fields mismatch")
        tensors = {
            name: torch.from_numpy(np.array(archive[name], copy=True))
            for name in _TENSOR_FIELDS
        }
    return AlignedTrackingTrace(
        mode=metadata["mode"],
        body_names=tuple(metadata["body_names"]),
        site_names=tuple(metadata["site_names"]),
        local_frame=CartesianFrameSpec(
            kind=CartesianFrameKind(metadata["local_frame"]["kind"]),
            anchor=metadata["local_frame"]["anchor"],
            rotation=CartesianRotation(metadata["local_frame"]["rotation"]),
        ),
        fell=metadata["fell"],
        horizon_reached=metadata["horizon_reached"],
        termination_sample=metadata["termination_sample"],
        **tensors,
    )


def paired_result_to_dict(result: PairedEvaluationResult) -> dict[str, Any]:
    """Convert an immutable result into a stable JSON payload."""

    return {
        "schema_version": 1,
        "passed": result.passed,
        "aligned_frames": result.aligned_frames,
        "checks": {name: passed for name, passed in result.checks},
        "stiff": asdict(result.stiff),
        "compliant": asdict(result.compliant),
        "compliance_response": asdict(result.compliance_response),
    }
