"""Bounded atomic persistence for portable evaluation traces and reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile

import numpy as np

from .schema import EvaluationTrace


TRACE_SCHEMA_VERSION = "compliance_trace_v1"


_TRACE_ARRAY_FIELDS = (
    "seed_ids",
    "frame_indices",
    "timestamps_s",
    "original_site_positions_m",
    "selected_site_positions_m",
    "measured_site_positions_m",
    "original_site_orientations_xyzw",
    "measured_site_orientations_xyzw",
    "reference_points_global_m",
    "measured_points_global_m",
    "reference_points_local_m",
    "measured_points_local_m",
    "force_on_robot_n",
    "compliance_enabled",
    "active_site_mask",
    "terminal_mask",
    "success_mask",
    "fall_mask",
    "reset_mask",
)


def _validate_byte_limit(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(temporary_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary_path, output_path)
    else:
        os.link(temporary_path, output_path)
        temporary_path.unlink()
    _fsync_directory(output_path.parent)


def write_report_json_atomic(
    report: object,
    output_path: str | Path,
    *,
    max_bytes: int = 4 * 1024 * 1024,
    overwrite: bool = False,
) -> Path:
    """Write one finite JSON report atomically and enforce a hard size bound."""

    _validate_byte_limit(max_bytes)
    encoded = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("serialized report exceeds max_bytes")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary_path, output, overwrite=overwrite)
        temporary_path = None
        return output
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _trace_arrays(trace: EvaluationTrace) -> dict[str, np.ndarray]:
    arrays = {field_name: np.asarray(getattr(trace, field_name)) for field_name in _TRACE_ARRAY_FIELDS}
    arrays.update(
        {
            "schema_version": np.asarray(TRACE_SCHEMA_VERSION),
            "trial_name": np.asarray(trace.trial_name),
            "motion_ids": np.asarray(trace.motion_ids, dtype=np.str_),
            "sequence_ids": np.asarray(trace.sequence_ids, dtype=np.str_),
            "site_ids": np.asarray(trace.site_ids, dtype=np.str_),
            "point_ids": np.asarray(trace.point_ids, dtype=np.str_),
        }
    )
    return arrays


def write_trace_npz_atomic(
    trace: EvaluationTrace,
    output_path: str | Path,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    overwrite: bool = False,
) -> Path:
    """Write one compressed trace without partial publication or unbounded logs."""

    _validate_byte_limit(max_bytes)
    arrays = _trace_arrays(trace)
    uncompressed_bytes = sum(array.nbytes for array in arrays.values())
    if uncompressed_bytes > max_bytes:
        raise ValueError("uncompressed trace exceeds max_bytes")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary_path.stat().st_size > max_bytes:
            raise ValueError("serialized trace exceeds max_bytes")
        _publish(temporary_path, output, overwrite=overwrite)
        temporary_path = None
        return output
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_trace_npz(path: str | Path, *, max_bytes: int = 64 * 1024 * 1024) -> EvaluationTrace:
    """Load one bounded schema-exact trace with NumPy pickle support disabled."""

    _validate_byte_limit(max_bytes)
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError("trace path must be a regular non-symlink file") from exc
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("trace path must be a regular non-symlink file")
        if status.st_size > max_bytes:
            raise ValueError("serialized trace exceeds max_bytes")
        try:
            with zipfile.ZipFile(stream) as zip_archive:
                members = zip_archive.infolist()
                member_names = [member.filename for member in members]
                if len(member_names) != len(set(member_names)):
                    raise ValueError("trace archive must not contain duplicate members")
                if sum(member.file_size for member in members) > max_bytes:
                    raise ValueError("uncompressed trace exceeds max_bytes")
        except zipfile.BadZipFile as exc:
            raise ValueError("trace is not a valid NPZ archive") from exc

        expected = set(_TRACE_ARRAY_FIELDS) | {
            "schema_version",
            "trial_name",
            "motion_ids",
            "sequence_ids",
            "site_ids",
            "point_ids",
        }
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValueError("trace archive fields do not match the schema")
            string_arrays = {
                name: np.asarray(archive[name])
                for name in (
                    "schema_version",
                    "trial_name",
                    "motion_ids",
                    "sequence_ids",
                    "site_ids",
                    "point_ids",
                )
            }
            for name in ("schema_version", "trial_name"):
                if string_arrays[name].shape != () or string_arrays[name].dtype.kind != "U":
                    raise ValueError(f"{name} must be a scalar Unicode string")
            for name in ("motion_ids", "sequence_ids", "site_ids", "point_ids"):
                if string_arrays[name].ndim != 1 or string_arrays[name].dtype.kind != "U":
                    raise ValueError(f"{name} must be a one-dimensional Unicode array")
            if str(string_arrays["schema_version"].item()) != TRACE_SCHEMA_VERSION:
                raise ValueError("unsupported trace schema version")
            values = {name: np.array(archive[name], copy=True) for name in _TRACE_ARRAY_FIELDS}
            return EvaluationTrace(
                trial_name=str(string_arrays["trial_name"].item()),
                motion_ids=tuple(string_arrays["motion_ids"].tolist()),
                sequence_ids=tuple(string_arrays["sequence_ids"].tolist()),
                site_ids=tuple(string_arrays["site_ids"].tolist()),
                point_ids=tuple(string_arrays["point_ids"].tolist()),
                **values,
            )
