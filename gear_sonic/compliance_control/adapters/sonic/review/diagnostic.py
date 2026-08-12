"""Bounded non-formal trace for a short rendered SONIC review smoke."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from .trace import SonicReviewSnapshot

DIAGNOSTIC_TRACE_SCHEMA = "sonic_chip_review_diagnostic_trace_v1"
DIAGNOSTIC_TRACE_FIELDS = (
    "schema_version",
    "role",
    "motion_id",
    "seed",
    "frame_indices",
    "reference_frames",
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
    "force_on_robot_world_n",
    "force_on_robot_common_n",
    "compliance_m_per_n",
    "active_site_mask",
    "policy_actions",
    "terminal_mask",
    "timeout_mask",
    "fall_mask",
    "reset_mask",
)


class ReviewDiagnosticAccumulator:
    """Collect a fixed cutoff without mislabelling it as a natural timeout."""

    def __init__(self, *, role: str, motion_id: str, seed: int, fps: int = 50) -> None:
        for field_name, value in (("role", role), ("motion_id", motion_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(fps) is not int or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.role = role
        self.motion_id = motion_id
        self.seed = seed
        self.fps = fps
        self._rows: list[dict[str, object]] = []

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def append(
        self,
        snapshot: SonicReviewSnapshot,
        *,
        policy_action: np.ndarray,
        terminal: bool,
        timed_out: bool,
        fall: bool,
    ) -> None:
        if not isinstance(snapshot, SonicReviewSnapshot):
            raise TypeError("snapshot must be a SonicReviewSnapshot")
        index = len(self._rows)
        if snapshot.reference_frame != index:
            raise AssertionError("diagnostic reference frame must equal sample index")
        if any(type(value) is not bool for value in (terminal, timed_out, fall)):
            raise TypeError("diagnostic lifecycle flags must be bool")
        if timed_out and not terminal:
            raise ValueError("timed_out requires terminal")
        if fall != (terminal and not timed_out):
            raise ValueError("fall must describe a non-timeout terminal")
        action = np.asarray(policy_action)
        if action.ndim != 1 or action.size == 0 or action.dtype.kind != "f":
            raise ValueError("policy_action must be a non-empty floating vector")
        float_fields = (
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
            "force_on_robot_world_n",
            "force_on_robot_common_n",
            "compliance_m_per_n",
        )
        if not np.isfinite(action).all() or any(
            not np.isfinite(np.asarray(getattr(snapshot, name))).all()
            for name in float_fields
        ):
            raise ValueError("diagnostic sample contains a non-finite value")
        self._rows.append(
            {
                "snapshot": snapshot,
                "policy_action": np.array(action, copy=True),
                "terminal": terminal,
                "timed_out": timed_out,
                "fall": fall,
            }
        )

    def finish(self, *, expected_frame_count: int) -> dict[str, np.ndarray]:
        if type(expected_frame_count) is not int or expected_frame_count < 8:
            raise ValueError("expected_frame_count must be an integer of at least eight")
        if len(self._rows) != expected_frame_count:
            raise AssertionError("diagnostic trace did not reach its fixed cutoff")
        if any(
            bool(row[name])
            for row in self._rows
            for name in ("terminal", "timed_out", "fall")
        ):
            raise AssertionError("diagnostic trace terminated before its fixed cutoff")
        snapshots = [row["snapshot"] for row in self._rows]
        assert all(isinstance(value, SonicReviewSnapshot) for value in snapshots)

        def stack(field_name: str) -> np.ndarray:
            return np.stack([getattr(value, field_name) for value in snapshots])

        return {
            "schema_version": np.asarray(DIAGNOSTIC_TRACE_SCHEMA),
            "role": np.asarray(self.role),
            "motion_id": np.asarray(self.motion_id),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "frame_indices": np.arange(expected_frame_count, dtype=np.int64),
            "reference_frames": np.asarray(
                [value.reference_frame for value in snapshots],
                dtype=np.int64,
            ),
            "timestamps_s": np.arange(expected_frame_count, dtype=np.float64)
            / self.fps,
            "original_site_positions_m": stack("original_site_positions_m"),
            "selected_site_positions_m": stack("selected_site_positions_m"),
            "measured_site_positions_m": stack("measured_site_positions_m"),
            "original_site_orientations_xyzw": stack(
                "original_site_orientations_xyzw"
            ),
            "measured_site_orientations_xyzw": stack(
                "measured_site_orientations_xyzw"
            ),
            "reference_points_global_m": stack("reference_points_global_m"),
            "measured_points_global_m": stack("measured_points_global_m"),
            "reference_points_local_m": stack("reference_points_local_m"),
            "measured_points_local_m": stack("measured_points_local_m"),
            "force_on_robot_n": stack("force_on_robot_n"),
            "force_on_robot_world_n": stack("force_on_robot_world_n"),
            "force_on_robot_common_n": stack("force_on_robot_common_n"),
            "compliance_m_per_n": stack("compliance_m_per_n"),
            "active_site_mask": stack("active_site_mask"),
            "policy_actions": np.stack(
                [row["policy_action"] for row in self._rows]
            ),
            "terminal_mask": np.asarray(
                [row["terminal"] for row in self._rows], dtype=np.bool_
            ),
            "timeout_mask": np.asarray(
                [row["timed_out"] for row in self._rows], dtype=np.bool_
            ),
            "fall_mask": np.asarray(
                [row["fall"] for row in self._rows], dtype=np.bool_
            ),
            "reset_mask": np.arange(expected_frame_count) == 0,
        }


def write_diagnostic_trace_atomic(
    arrays: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> Path:
    """Publish one schema-exact diagnostic NPZ without overwrite."""

    if not isinstance(arrays, dict):
        raise TypeError("arrays must be a dictionary")
    if set(arrays) != set(DIAGNOSTIC_TRACE_FIELDS):
        raise ValueError("diagnostic trace fields do not match the schema")
    schema = np.asarray(arrays.get("schema_version"))
    if (
        schema.shape != ()
        or schema.dtype.kind != "U"
        or schema.item() != DIAGNOSTIC_TRACE_SCHEMA
    ):
        raise ValueError("arrays do not use the diagnostic trace schema")
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise TypeError("diagnostic trace arrays must not use object dtype")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    uncompressed_bytes = sum(np.asarray(value).nbytes for value in arrays.values())
    if uncompressed_bytes > max_bytes:
        raise ValueError("uncompressed diagnostic trace exceeds max_bytes")
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("diagnostic trace output must end in .npz")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise NotADirectoryError("diagnostic trace parent must be a real directory")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size > max_bytes:
            raise ValueError("serialized diagnostic trace exceeds max_bytes")
        os.link(temporary, output)
        temporary.unlink()
        temporary = None
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
