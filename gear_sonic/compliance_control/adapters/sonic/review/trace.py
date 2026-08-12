"""Frame-exact accumulation into the tracker-neutral review trace schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....review import EvaluationTrace
from .roles import REVIEW_SITE_NAMES, ReviewRole


@dataclass(frozen=True, slots=True)
class SonicReviewSnapshot:
    """One SONIC state snapshot captured before its corresponding transition."""

    reference_frame: int
    original_site_positions_m: np.ndarray
    selected_site_positions_m: np.ndarray
    measured_site_positions_m: np.ndarray
    original_site_orientations_xyzw: np.ndarray
    measured_site_orientations_xyzw: np.ndarray
    reference_points_global_m: np.ndarray
    measured_points_global_m: np.ndarray
    reference_points_local_m: np.ndarray
    measured_points_local_m: np.ndarray
    force_on_robot_n: np.ndarray
    force_on_robot_world_n: np.ndarray
    force_on_robot_common_n: np.ndarray
    compliance_m_per_n: np.ndarray
    active_site_mask: np.ndarray


class ReviewTraceAccumulator:
    """Collect one natural-timeout rollout without interpolation or reset suffixes."""

    def __init__(
        self,
        *,
        role: ReviewRole,
        motion_id: str,
        seed: int,
        point_ids: tuple[str, ...],
        fps: int = 50,
    ) -> None:
        if not isinstance(role, ReviewRole):
            raise TypeError("role must be a ReviewRole")
        if not isinstance(motion_id, str) or not motion_id:
            raise ValueError("motion_id must be a non-empty string")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(point_ids, (str, bytes)) or not point_ids:
            raise ValueError("point_ids must be a non-empty ordered tuple")
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point_ids must be unique")
        if type(fps) is not int or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.role = role
        self.motion_id = motion_id
        self.seed = seed
        self.point_ids = tuple(point_ids)
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
        reset: bool,
        terminal: bool,
        success: bool,
        fall: bool,
    ) -> None:
        """Append one sample only after its transition outcome is known."""

        if not isinstance(snapshot, SonicReviewSnapshot):
            raise TypeError("snapshot must be a SonicReviewSnapshot")
        sample_index = len(self._rows)
        if snapshot.reference_frame != sample_index:
            raise AssertionError(
                "reference frame must equal the zero-based policy sample index"
            )
        if type(reset) is not bool or type(terminal) is not bool:
            raise TypeError("lifecycle flags must be bool")
        if type(success) is not bool or type(fall) is not bool:
            raise TypeError("lifecycle flags must be bool")
        if success and (not terminal or fall):
            raise ValueError("success requires a non-fall terminal sample")
        if fall and not terminal:
            raise ValueError("fall requires a terminal sample")
        if reset != (sample_index == 0):
            raise ValueError("only the first trace sample may be the reset snapshot")
        if self._rows and bool(self._rows[-1]["terminal"]):
            raise RuntimeError("a terminal trace cannot accept an auto-reset suffix")
        action = np.asarray(policy_action)
        if action.ndim != 1 or action.size == 0 or action.dtype.kind != "f":
            raise ValueError("policy_action must be a non-empty floating vector")
        if not np.isfinite(action).all():
            raise ValueError("policy_action must be finite")
        self._rows.append(
            {
                "snapshot": snapshot,
                "policy_action": np.array(action, copy=True),
                "reset": reset,
                "terminal": terminal,
                "success": success,
                "fall": fall,
            }
        )

    def finish(self, *, expected_frame_count: int) -> EvaluationTrace:
        """Require one successful natural timeout and build an immutable trace."""

        if type(expected_frame_count) is not int or expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be a positive integer")
        if len(self._rows) != expected_frame_count:
            raise AssertionError(
                f"trace has {len(self._rows)} rows, expected {expected_frame_count}"
            )
        if not self._rows or not bool(self._rows[-1]["terminal"]):
            raise AssertionError("trace must end at the natural timeout")
        if not bool(self._rows[-1]["success"]) or bool(self._rows[-1]["fall"]):
            raise AssertionError("formal trace must finish successfully without a fall")
        snapshots = [row["snapshot"] for row in self._rows]
        assert all(isinstance(value, SonicReviewSnapshot) for value in snapshots)

        def stack(field_name: str) -> np.ndarray:
            return np.stack([getattr(value, field_name) for value in snapshots], axis=0)

        row_count = len(self._rows)
        return EvaluationTrace(
            trial_name=self.role.name,
            motion_ids=(self.motion_id,) * row_count,
            sequence_ids=(f"{self.motion_id}:seed-{self.seed}",) * row_count,
            seed_ids=np.full(row_count, self.seed, dtype=np.int64),
            frame_indices=np.arange(row_count, dtype=np.int64),
            timestamps_s=np.arange(row_count, dtype=np.float64) / self.fps,
            site_ids=REVIEW_SITE_NAMES,
            point_ids=self.point_ids,
            original_site_positions_m=stack("original_site_positions_m"),
            selected_site_positions_m=stack("selected_site_positions_m"),
            measured_site_positions_m=stack("measured_site_positions_m"),
            original_site_orientations_xyzw=stack(
                "original_site_orientations_xyzw"
            ),
            measured_site_orientations_xyzw=stack(
                "measured_site_orientations_xyzw"
            ),
            reference_points_global_m=stack("reference_points_global_m"),
            measured_points_global_m=stack("measured_points_global_m"),
            reference_points_local_m=stack("reference_points_local_m"),
            measured_points_local_m=stack("measured_points_local_m"),
            force_on_robot_n=stack("force_on_robot_n"),
            force_on_robot_world_n=stack("force_on_robot_world_n"),
            force_on_robot_common_n=stack("force_on_robot_common_n"),
            compliance_m_per_n=stack("compliance_m_per_n"),
            compliance_enabled=np.full(
                row_count,
                self.role.compliance_enabled,
                dtype=np.bool_,
            ),
            residual_enabled=np.full(
                row_count,
                self.role.residual_enabled,
                dtype=np.bool_,
            ),
            active_site_mask=stack("active_site_mask"),
            policy_actions=np.stack(
                [row["policy_action"] for row in self._rows],
                axis=0,
            ),
            terminal_mask=np.asarray(
                [row["terminal"] for row in self._rows],
                dtype=np.bool_,
            ),
            success_mask=np.asarray(
                [row["success"] for row in self._rows],
                dtype=np.bool_,
            ),
            fall_mask=np.asarray(
                [row["fall"] for row in self._rows],
                dtype=np.bool_,
            ),
            reset_mask=np.asarray(
                [row["reset"] for row in self._rows],
                dtype=np.bool_,
            ),
        )
