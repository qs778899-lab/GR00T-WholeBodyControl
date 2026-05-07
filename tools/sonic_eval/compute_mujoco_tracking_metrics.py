#!/usr/bin/env python3
"""
Compute Isaac-eval-style tracking metrics from MuJoCo sim2sim logs.

This script strictly reuses the same metric function as Isaac eval:
`smpl_sim.smpllib.smpl_eval.compute_metrics_lite`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.sonic_eval.motionlib_provider import load_motionlib_sequence
from tools.sonic_eval.motionlib_provider import G1_DEFAULT_ANGLES_ISAACLAB


MUJOCO_29_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

STATE43_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

MUJOCO_TO_ISAACLAB_29 = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
ISAACLAB_TO_MUJOCO_29 = [0] * 29
for mj_i, isa_i in enumerate(MUJOCO_TO_ISAACLAB_29):
    ISAACLAB_TO_MUJOCO_29[isa_i] = mj_i

BODY_FRAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
SUBSET_NAMES = {
    "legs": [
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    ],
    "vr_3points": ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
    "other_upper_bodies": [
        "pelvis",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
    ],
    "foot": ["left_ankle_roll_link", "right_ankle_roll_link"],
}


def _mean_or_zero(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


@dataclass
class GTData:
    q43_source: np.ndarray
    body_pos_w_source: np.ndarray | None
    motion_keys: list[str]
    source: str | None = None

    @property
    def num_frames(self) -> int:
        return int(self.q43_source.shape[0])


@dataclass
class LogData:
    q29: np.ndarray
    row_index: np.ndarray
    motion_name: np.ndarray | None


def _target_motion_name(gt_motion_dir: Path | None, target_motion_name: str | None) -> str | None:
    if target_motion_name:
        return target_motion_name
    if gt_motion_dir is not None:
        return gt_motion_dir.name
    return None


def _load_log_mask(
    logs_dir: Path,
    target_motion_name: str | None,
    use_motion_playing_mask: bool = True,
) -> np.ndarray | None:
    motion_playing_csv = logs_dir / "motion_playing.csv"
    motion_name_csv = logs_dir / "motion_name.csv"

    play_mask = None
    name_mask = None

    if use_motion_playing_mask and motion_playing_csv.exists():
        play_df = pd.read_csv(motion_playing_csv)
        play_val = play_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if play_val.shape[1] >= 1:
            play_mask = play_val[:, -1] > 0.5

    if target_motion_name is not None and motion_name_csv.exists():
        motion_name_df = pd.read_csv(motion_name_csv)
        name_mask = motion_name_df.iloc[:, -1].astype(str).to_numpy() == target_motion_name

    if play_mask is None and name_mask is None:
        return None

    masks = [m for m in [play_mask, name_mask] if m is not None]
    n = min(len(m) for m in masks)
    out = np.ones((n,), dtype=bool)
    for m in masks:
        out &= m[:n]
    return out


def _load_motion_name_series(logs_dir: Path) -> np.ndarray | None:
    motion_name_csv = logs_dir / "motion_name.csv"
    if not motion_name_csv.exists():
        return None
    motion_name_df = pd.read_csv(motion_name_csv)
    return motion_name_df.iloc[:, -1].astype(str).to_numpy()


def read_logs_q29(
    logs_dir: Path,
    target_motion_name: str | None,
    use_motion_playing_mask: bool = True,
) -> np.ndarray:
    q_csv = logs_dir / "q.csv"
    if not q_csv.exists():
        raise FileNotFoundError(f"missing {q_csv}")
    q_df = pd.read_csv(q_csv)
    q_val = q_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if q_val.shape[1] < 29:
        raise ValueError("q.csv numeric columns less than 29")
    q29 = q_val[:, -29:]

    log_mask = _load_log_mask(
        logs_dir=logs_dir,
        target_motion_name=target_motion_name,
        use_motion_playing_mask=use_motion_playing_mask,
    )
    if log_mask is None:
        return q29

    n = min(len(q29), len(log_mask))
    q29 = q29[:n]
    log_mask = log_mask[:n]

    if not np.any(log_mask):
        motion_desc = target_motion_name if target_motion_name is not None else "<unknown>"
        raise ValueError(
            "No valid log rows matched the requested motion/play mask. "
            f"target_motion_name={motion_desc}"
        )
    return q29[log_mask]


def read_logs_data(
    logs_dir: Path,
    target_motion_name: str | None,
    use_motion_playing_mask: bool = True,
) -> LogData:
    q_csv = logs_dir / "q.csv"
    if not q_csv.exists():
        raise FileNotFoundError(f"missing {q_csv}")
    q_df = pd.read_csv(q_csv)
    q_val = q_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if q_val.shape[1] < 29:
        raise ValueError("q.csv numeric columns less than 29")
    q29 = q_val[:, -29:]
    row_idx = np.arange(len(q29), dtype=np.int64)

    log_mask = _load_log_mask(
        logs_dir=logs_dir,
        target_motion_name=target_motion_name,
        use_motion_playing_mask=use_motion_playing_mask,
    )
    if log_mask is not None:
        n = min(len(q29), len(log_mask))
        q29 = q29[:n]
        row_idx = row_idx[:n]
        log_mask = log_mask[:n]
        if not np.any(log_mask):
            motion_desc = target_motion_name if target_motion_name is not None else "<unknown>"
            raise ValueError(
                "No valid log rows matched the requested motion/play mask. "
                f"target_motion_name={motion_desc}"
            )
        q29 = q29[log_mask]
        row_idx = row_idx[log_mask]

    motion_name = _load_motion_name_series(logs_dir)
    if motion_name is not None:
        # Some deploy CSVs can end a few rows earlier than q.csv during shutdown.
        # Align by the shared prefix so downstream filtering does not index past
        # the motion-name series.
        shared_n = min(len(q29), len(row_idx), len(motion_name))
        q29 = q29[:shared_n]
        row_idx = row_idx[:shared_n]
        motion_name = motion_name[row_idx]
    return LogData(q29=q29, row_index=row_idx, motion_name=motion_name)


def _finite_difference(values: np.ndarray, fps: int) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    vel = np.zeros_like(values, dtype=np.float64)
    vel[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    vel[0] = (values[1] - values[0]) * fps
    vel[-1] = (values[-1] - values[-2]) * fps
    return vel


def _prepend_stand_transition_q29(
    q29_mujoco: np.ndarray,
    target_fps: int,
    stand_frames: int,
    blend_frames: int,
) -> np.ndarray:
    if stand_frames <= 0 and blend_frames <= 0:
        return q29_mujoco
    if len(q29_mujoco) == 0:
        return q29_mujoco

    default_pose_isaac = G1_DEFAULT_ANGLES_ISAACLAB.detach().cpu().numpy().astype(np.float64)
    # Convert default pose to MuJoCo order.
    default_pose_mj = default_pose_isaac[ISAACLAB_TO_MUJOCO_29]
    prefix_parts = []
    if stand_frames > 0:
        prefix_parts.append(np.repeat(default_pose_mj[None, :], stand_frames, axis=0))
    if blend_frames > 0:
        alpha = np.linspace(0.0, 1.0, blend_frames + 1, dtype=np.float64)[:-1, None]
        prefix_parts.append((1.0 - alpha) * default_pose_mj[None, :] + alpha * q29_mujoco[:1])
    if not prefix_parts:
        return q29_mujoco
    prefix = np.concatenate(prefix_parts, axis=0).astype(np.float64)
    out = np.concatenate([prefix, q29_mujoco], axis=0).astype(np.float64)
    _ = _finite_difference(out, target_fps)  # Keep parity with stream preprocessing pipeline.
    return out


def _prepend_stand_transition_body_pos_w(
    body_pos_w: np.ndarray,
    stand_frames: int,
    blend_frames: int,
) -> np.ndarray:
    if stand_frames <= 0 and blend_frames <= 0:
        return body_pos_w
    if len(body_pos_w) == 0:
        return body_pos_w

    default_pose = body_pos_w[:1]
    prefix_parts = []
    if stand_frames > 0:
        prefix_parts.append(np.repeat(default_pose, stand_frames, axis=0))
    if blend_frames > 0:
        alpha = np.linspace(0.0, 1.0, blend_frames + 1, dtype=np.float64)[:-1]
        alpha = alpha[:, None, None]
        prefix_parts.append((1.0 - alpha) * default_pose + alpha * body_pos_w[:1])
    if not prefix_parts:
        return body_pos_w
    prefix = np.concatenate(prefix_parts, axis=0).astype(np.float64)
    return np.concatenate([prefix, body_pos_w], axis=0).astype(np.float64)


def _auto_align_by_q29_mae(
    q29_log: np.ndarray,
    q29_gt: np.ndarray,
    lag_min: int,
    lag_max: int,
    min_overlap: int,
) -> tuple[int, int, float]:
    best_lag = 0
    best_overlap = 0
    best_mae = float("inf")
    for lag in range(lag_min, lag_max + 1):
        if lag >= 0:
            a = q29_log[lag:]
            b = q29_gt
        else:
            a = q29_log
            b = q29_gt[-lag:]
        overlap = min(len(a), len(b))
        if overlap < min_overlap:
            continue
        mae = float(np.mean(np.abs(a[:overlap] - b[:overlap])))
        if mae < best_mae:
            best_mae = mae
            best_lag = lag
            best_overlap = overlap
    if not np.isfinite(best_mae):
        raise ValueError(
            f"auto alignment failed: no overlap >= {min_overlap} in lag range [{lag_min}, {lag_max}]"
        )
    return best_lag, best_overlap, best_mae


def load_gt_body_q29_from_motion_dir(motion_dir: Path) -> np.ndarray:
    joint_pos_csv = motion_dir / "joint_pos.csv"
    if not joint_pos_csv.exists():
        raise FileNotFoundError(f"missing {joint_pos_csv}")
    jp = pd.read_csv(joint_pos_csv).select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if jp.shape[1] < 29:
        raise ValueError("joint_pos.csv numeric columns less than 29")
    jp_isaac = jp[:, :29]
    return jp_isaac[:, ISAACLAB_TO_MUJOCO_29]


def load_gt(parquet: Path, gt_source: str, gt_motion_dir: Path | None = None) -> GTData:
    df = pd.read_parquet(parquet)
    state = np.stack(df["observation.state"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    action = np.stack(df["action.wbc"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    if state.shape[1] != 43:
        raise ValueError(f"observation.state must be 43D, got {state.shape[1]}")
    if action.shape[1] != 43:
        raise ValueError(f"action.wbc must be 43D, got {action.shape[1]}")

    if gt_source == "observation.state":
        q43 = state
    elif gt_source == "action.wbc":
        q43 = action
    else:
        raise ValueError(f"unsupported gt_source: {gt_source}")

    idx43 = {n: i for i, n in enumerate(STATE43_NAMES)}
    body29_idx = [idx43[n] for n in MUJOCO_29_NAMES]

    if gt_motion_dir is not None:
        body29_mj = load_gt_body_q29_from_motion_dir(gt_motion_dir)
        n_sync = min(len(q43), len(body29_mj))
        q43 = q43[:n_sync].copy()
        q43[:, body29_idx] = body29_mj[:n_sync]

    motion_keys = [f"frame_{i:06d}" for i in range(len(q43))]
    return GTData(q43_source=q43, body_pos_w_source=None, motion_keys=motion_keys, source="parquet")


def load_gt_motionlib(
    motion_file: Path,
    motion_name: str | None,
    target_fps: int,
    num_future_frames: int,
    dt_future_ref_frames: float,
    device: str,
    prefer_motionlib_robot: bool,
    use_isaacsim_app: bool,
    stream_start_frame: int,
    stream_prepend_stand_frames: int,
    stream_blend_from_stand_frames: int,
) -> GTData:
    seq = load_motionlib_sequence(
        motion_file=motion_file,
        motion_name=motion_name,
        target_fps=target_fps,
        num_future_frames=num_future_frames,
        dt_future_ref_frames=dt_future_ref_frames,
        device=device,
        prefer_motionlib_robot=prefer_motionlib_robot,
        use_isaacsim_app=use_isaacsim_app,
    )
    q43 = np.zeros((seq.num_frames, len(STATE43_NAMES)), dtype=np.float64)
    idx43 = {n: i for i, n in enumerate(STATE43_NAMES)}
    body29_idx = [idx43[n] for n in MUJOCO_29_NAMES]
    q29 = seq.q29_mujoco().astype(np.float64)
    if stream_start_frame > 0:
        if stream_start_frame >= len(q29):
            raise ValueError(
                f"--stream-start-frame={stream_start_frame} is out of range for GT length {len(q29)}"
            )
        q29 = q29[stream_start_frame:]
    q29 = _prepend_stand_transition_q29(
        q29_mujoco=q29,
        target_fps=target_fps,
        stand_frames=max(0, stream_prepend_stand_frames),
        blend_frames=max(0, stream_blend_from_stand_frames),
    )
    q43 = np.zeros((len(q29), len(STATE43_NAMES)), dtype=np.float64)
    q43[:, body29_idx] = q29
    body_pos_w = seq.body_pos_w.detach().cpu().numpy().astype(np.float64)
    if stream_start_frame > 0:
        body_pos_w = body_pos_w[stream_start_frame:]
    body_pos_w = _prepend_stand_transition_body_pos_w(
        body_pos_w=body_pos_w,
        stand_frames=max(0, stream_prepend_stand_frames),
        blend_frames=max(0, stream_blend_from_stand_frames),
    )
    if len(body_pos_w) != len(q29):
        raise ValueError(
            f"motionlib body_pos_w length mismatch after stream slicing: {len(body_pos_w)} vs q29 {len(q29)}"
        )
    body_pos_w = body_pos_w[:, : len(BODY_FRAMES), :]
    motion_keys = [f"{seq.motion_name}:frame_{i:06d}" for i in range(len(q43))]
    return GTData(
        q43_source=q43,
        body_pos_w_source=body_pos_w,
        motion_keys=motion_keys,
        source=seq.source,
    )


class BodyFK:
    def __init__(self, urdf: Path, body_frames: list[str]):
        try:
            import pinocchio as pin
        except Exception as e:
            raise RuntimeError(
                "pinocchio is required for --actual-source q_fk, but import failed"
            ) from e
        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame_ids = [self.model.getFrameId(n) for n in body_frames]
        if any(fid >= len(self.model.frames) for fid in self.frame_ids):
            raise ValueError("Some body frames not found in URDF")

    def body_pos(self, q43: np.ndarray) -> np.ndarray:
        self._pin.framesForwardKinematics(self.model, self.data, q43)
        return np.stack(
            [np.asarray(self.data.oMf[fid].translation, dtype=np.float64) for fid in self.frame_ids],
            axis=0,
        )


def read_logged_body_pos_w_14(logs_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = logs_dir / "body_pos_w_14.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if numeric.shape[1] < 2 + 3 * len(BODY_FRAMES):
        raise ValueError(
            f"body_pos_w_14.csv has insufficient numeric columns: {numeric.shape[1]}"
        )
    row_idx = np.rint(numeric[:, 0]).astype(np.int64)
    body_pos_flat = numeric[:, 2 : 2 + 3 * len(BODY_FRAMES)]
    body_pos = body_pos_flat.reshape(-1, len(BODY_FRAMES), 3)
    return row_idx, body_pos


def read_step_sync_body_pos_w_14(logs_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = logs_dir / "sim2sim_step_sync_body_pos_w_14.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    expected_cols = 3 + 2 * 3 * len(BODY_FRAMES)
    if numeric.shape[1] < expected_cols:
        raise ValueError(
            f"sim2sim_step_sync_body_pos_w_14.csv has insufficient numeric columns: {numeric.shape[1]}"
        )
    source_frame_idx = np.rint(numeric[:, 2]).astype(np.int64)
    actual_flat = numeric[:, 3 : 3 + 3 * len(BODY_FRAMES)]
    ref_flat = numeric[:, 3 + 3 * len(BODY_FRAMES) : 3 + 2 * 3 * len(BODY_FRAMES)]
    actual = actual_flat.reshape(-1, len(BODY_FRAMES), 3)
    ref = ref_flat.reshape(-1, len(BODY_FRAMES), 3)
    return source_frame_idx, actual, ref


def read_sim_source_frame_index(logs_dir: Path) -> np.ndarray:
    csv_path = logs_dir / "sim_source_frame_index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if numeric.shape[1] < 1:
        raise ValueError("sim_source_frame_index.csv numeric columns are empty")
    return np.rint(numeric[:, -1]).astype(np.int64)


def align_actual_body_pos_by_source_frame(
    deploy_source_frame_idx: np.ndarray,
    sim_source_frame_idx: np.ndarray,
    sim_body_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sim_source_frame_idx) != len(sim_body_pos):
        raise ValueError(
            f"sim_source_frame_index/body_pos_w_14 length mismatch: {len(sim_source_frame_idx)} vs {len(sim_body_pos)}"
        )

    source_to_rows: dict[int, list[int]] = {}
    for i, src in enumerate(sim_source_frame_idx.tolist()):
        if src < 0:
            continue
        source_to_rows.setdefault(int(src), []).append(i)

    missing_sources = sorted({int(src) for src in deploy_source_frame_idx.tolist() if int(src) not in source_to_rows})
    if missing_sources:
        raise ValueError(
            "sim body_pos_w_14 logs are missing source_frame_index values required for alignment; "
            f"first few missing={missing_sources[:10]}"
        )

    taken_counts: dict[int, int] = {}
    aligned_rows: list[np.ndarray] = []
    for src in deploy_source_frame_idx.tolist():
        src_i = int(src)
        rows = source_to_rows[src_i]
        occ = taken_counts.get(src_i, 0)
        chosen = rows[min(occ, len(rows) - 1)]
        aligned_rows.append(sim_body_pos[chosen])
        taken_counts[src_i] = occ + 1
    return np.stack(aligned_rows, axis=0), np.asarray(
        [int(src) for src in deploy_source_frame_idx.tolist()], dtype=np.int64
    )


def filter_step_sync_body_pos_by_source_frame(
    source_frame_idx: np.ndarray,
    actual_body_pos: np.ndarray,
    ref_body_pos: np.ndarray,
    gt_num_frames: int,
    valid_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (len(source_frame_idx) == len(actual_body_pos) == len(ref_body_pos)):
        raise ValueError(
            "sim2sim step-sync logs length mismatch: "
            f"src={len(source_frame_idx)} actual={len(actual_body_pos)} ref={len(ref_body_pos)}"
        )
    valid = source_frame_idx >= 0
    valid &= source_frame_idx < gt_num_frames
    if valid_only:
        if not np.any(valid):
            raise ValueError("sim2sim step-sync logs contain no valid source_frame_index rows")
        source_frame_idx = source_frame_idx[valid]
        actual_body_pos = actual_body_pos[valid]
        ref_body_pos = ref_body_pos[valid]

    # Defensive deduplication for legacy MuJoCo logs that wrote the same strict
    # source frame across many physics steps. Isaac-style evaluation expects one
    # aligned sample per GT frame, not tens of repeated rows for the same source
    # frame index.
    if len(source_frame_idx) > 1:
        keep = np.ones(len(source_frame_idx), dtype=bool)
        keep[1:] = source_frame_idx[1:] != source_frame_idx[:-1]
        source_frame_idx = source_frame_idx[keep]
        actual_body_pos = actual_body_pos[keep]
        ref_body_pos = ref_body_pos[keep]
    return source_frame_idx, actual_body_pos, ref_body_pos


def _compute_metrics_with_smpl(
    pred_pos_all: list[np.ndarray],
    gt_pos_all: list[np.ndarray],
) -> tuple[dict[str, list[np.ndarray]], str]:
    try:
        from smpl_sim.smpllib.smpl_eval import compute_metrics_lite  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "smpl_sim.smpllib.smpl_eval.compute_metrics_lite is required for IsaacSim-aligned metrics, "
            "but import failed. Install/activate env with smpl_sim."
        ) from e
    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all, concatenate=False)  # type: ignore
    return metrics, "smpl_sim.compute_metrics_lite"


def _aggregate_metrics_per_motion(metrics: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Match Isaac callback reduction:
    - if key contains 'mpjpe': mean over time
    - else: sum over time
    """
    reduced: dict[str, np.ndarray] = {}
    for k, seq_list in metrics.items():
        vals = []
        for arr in seq_list:
            a = np.asarray(arr, dtype=np.float64)
            vals.append(float(np.mean(a)) if "mpjpe" in k else float(np.sum(a)))
        reduced[k] = np.asarray(vals, dtype=np.float64)
    return reduced


def _as_frame_metric(arr_like: Any, expected_frames: int, name: str) -> np.ndarray:
    """
    Normalize metric output to per-frame 1D vector of length expected_frames.
    If backend returns extra non-time dimensions, reduce them by mean.
    """
    arr = np.asarray(arr_like, dtype=np.float64)
    if arr.ndim == 0:
        return np.repeat(arr.reshape(1), expected_frames).astype(np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != expected_frames:
            raise ValueError(
                f"{name} frame length mismatch: got {arr.shape[0]}, expected {expected_frames}"
            )
        return arr

    # Prefer axis 0 as time axis when it matches expected frame count.
    if arr.shape[0] == expected_frames:
        return arr.reshape(expected_frames, -1).mean(axis=1)
    # Fallback: if last axis matches expected frame count, reduce others.
    if arr.shape[-1] == expected_frames:
        return arr.reshape(-1, expected_frames).mean(axis=0)

    raise ValueError(
        f"{name} shape {arr.shape} cannot be normalized to {expected_frames} frames"
    )


def _mean_episode_metric_sum(metric_name: str, frame_values: np.ndarray) -> float:
    arr = np.asarray(frame_values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if "mpjpe" in metric_name:
        return float(np.sum(arr))
    return float(np.sum(arr))


def _episode_metric_from_sum(metric_name: str, metric_sum: float, frame_count: int) -> float:
    if frame_count <= 0:
        return 0.0
    return float(metric_sum / float(frame_count))


def _infer_episode_key(
    motion_name_series: np.ndarray | None,
    eval_motion_keys: list[str],
    target_motion_name: str | None,
) -> str:
    if motion_name_series is not None and len(motion_name_series) > 0:
        unique_names = pd.unique(motion_name_series.astype(str))
        if len(unique_names) == 1:
            return str(unique_names[0])
    if target_motion_name:
        return str(target_motion_name)
    if eval_motion_keys:
        first = eval_motion_keys[0]
        if ":frame_" in first:
            return first.split(":frame_", 1)[0]
        return first
    return "episode_000000"


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim Isaac-style eval metrics")
    parser.add_argument("--gt-format", choices=["parquet", "motionlib"], default="parquet")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--motion-file", type=Path, default=None)
    parser.add_argument("--motion-name", type=str, default=None)
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--dt-future-ref-frames", type=float, default=0.1)
    parser.add_argument("--motionlib-device", type=str, default="cpu")
    parser.add_argument("--no-motionlib-robot", action="store_true")
    parser.add_argument(
        "--use-isaacsim-app",
        action="store_true",
        help="start IsaacSim SimulationApp before official TrackingCommand preprocessing",
    )
    parser.add_argument(
        "--stream-start-frame",
        type=int,
        default=0,
        help="match stream_motionlib_to_deploy --start-frame when using --gt-format motionlib",
    )
    parser.add_argument(
        "--stream-prepend-stand-frames",
        type=int,
        default=0,
        help="match stream_motionlib_to_deploy --prepend-stand-frames for GT reconstruction",
    )
    parser.add_argument(
        "--stream-blend-from-stand-frames",
        type=int,
        default=0,
        help="match stream_motionlib_to_deploy --blend-from-stand-frames for GT reconstruction",
    )
    parser.add_argument("--logs-dir", type=Path, required=True, help="deploy logs dir containing q.csv")
    parser.add_argument(
        "--ignore-motion-playing-mask",
        action="store_true",
        help="ignore motion_playing.csv mask (useful for streamed/ZMQ runs where playing_0 may stay 0)",
    )
    parser.add_argument(
        "--streamed-only",
        action="store_true",
        help="evaluate only rows where motion_name == 'streamed'",
    )
    parser.add_argument(
        "--align-mode",
        choices=["source_frame_index", "index", "auto_q29"],
        default="source_frame_index",
        help="frame alignment mode: source_frame_index (strict), index (legacy), auto_q29 (lag search)",
    )
    parser.add_argument(
        "--actual-source",
        choices=["q_fk", "body_pos_w_14", "step_sync_body_pos_w_14"],
        default="step_sync_body_pos_w_14",
        help="source for actual robot trajectory: step_sync_body_pos_w_14 (strict step-synchronous sim/ref world-frame log), body_pos_w_14 (world-frame sim log aligned by source frame), or q_fk (legacy fixed-base FK)",
    )
    parser.add_argument(
        "--sim-valid-only",
        action="store_true",
        help="for body_pos_w_14 actual logs, evaluate only the span where sim_source_frame_index is valid (>=0)",
    )
    parser.add_argument("--align-lag-min", type=int, default=-150, help="min lag for auto_q29")
    parser.add_argument("--align-lag-max", type=int, default=150, help="max lag for auto_q29")
    parser.add_argument("--align-min-overlap", type=int, default=200, help="min overlap for auto_q29")
    parser.add_argument(
        "--gt-source",
        type=str,
        choices=["observation.state", "action.wbc"],
        default="action.wbc",
    )
    parser.add_argument("--gt-motion-dir", type=Path, default=None)
    parser.add_argument("--target-motion-name", type=str, default=None)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"),
    )
    parser.add_argument("--success-thresh-mpjpe-l-mm", type=float, default=30.0)
    parser.add_argument("--success-thresh-mpjpe-g-mm", type=float, default=200.0)
    parser.add_argument("--success-thresh-mpjpe-pa-mm", type=float, default=30.0)
    parser.add_argument("--out-json", type=Path, default=Path("/tmp/sonic_mujoco_eval_metrics.json"))
    args = parser.parse_args()

    if args.gt_format == "parquet":
        if args.parquet is None:
            raise ValueError("--parquet is required when --gt-format parquet")
        gt = load_gt(args.parquet, args.gt_source, args.gt_motion_dir)
        target_motion_name = _target_motion_name(args.gt_motion_dir, args.target_motion_name)
    else:
        if args.motion_file is None:
            raise ValueError("--motion-file is required when --gt-format motionlib")
        gt = load_gt_motionlib(
            motion_file=args.motion_file,
            motion_name=args.motion_name,
            target_fps=args.target_fps,
            num_future_frames=args.num_future_frames,
            dt_future_ref_frames=args.dt_future_ref_frames,
            device=args.motionlib_device,
            prefer_motionlib_robot=not args.no_motionlib_robot,
            use_isaacsim_app=args.use_isaacsim_app,
            stream_start_frame=max(0, args.stream_start_frame),
            stream_prepend_stand_frames=max(0, args.stream_prepend_stand_frames),
            stream_blend_from_stand_frames=max(0, args.stream_blend_from_stand_frames),
        )
        # ZMQ streamed motions are named by deploy (usually "streamed"), not by
        # the source motion_lib key. Only filter by name when explicitly asked.
        target_motion_name = args.target_motion_name
    logs = read_logs_data(
        logs_dir=args.logs_dir,
        target_motion_name=target_motion_name,
        use_motion_playing_mask=not args.ignore_motion_playing_mask,
    )
    q29 = logs.q29
    motion_name_series = logs.motion_name
    row_idx = logs.row_index
    streamed_row_start = None
    if args.streamed_only and motion_name_series is not None:
        streamed_mask = motion_name_series == "streamed"
        if np.any(streamed_mask):
            streamed_idx = np.nonzero(streamed_mask)[0]
            streamed_row_start = int(row_idx[streamed_idx[0]])
            q29 = q29[streamed_mask]
            row_idx = row_idx[streamed_mask]
            motion_name_series = motion_name_series[streamed_mask]
        else:
            raise ValueError("--streamed-only was set but motion_name.csv has no 'streamed' rows")

    fk = BodyFK(args.urdf, BODY_FRAMES) if args.actual_source == "q_fk" else None

    idx43 = {n: i for i, n in enumerate(STATE43_NAMES)}
    body29_idx_in_43 = [idx43[n] for n in MUJOCO_29_NAMES]
    body_name_to_idx = {n: i for i, n in enumerate(BODY_FRAMES)}
    subset_indices = {
        "": np.arange(len(BODY_FRAMES), dtype=np.int64),
        "_legs": np.asarray([body_name_to_idx[n] for n in SUBSET_NAMES["legs"]], dtype=np.int64),
        "_vr_3points": np.asarray([body_name_to_idx[n] for n in SUBSET_NAMES["vr_3points"]], dtype=np.int64),
        "_other_upper_bodies": np.asarray(
            [body_name_to_idx[n] for n in SUBSET_NAMES["other_upper_bodies"]],
            dtype=np.int64,
        ),
        "_foot": np.asarray([body_name_to_idx[n] for n in SUBSET_NAMES["foot"]], dtype=np.int64),
    }

    align_lag_frames = 0
    align_overlap = None
    align_q29_mae = None
    source_frame_valid = None
    source_frame_kept = None
    eval_motion_keys: list[str] = []
    if args.align_mode == "source_frame_index":
        sf_csv = args.logs_dir / "source_frame_index.csv"
        if not sf_csv.exists():
            raise FileNotFoundError(
                f"missing {sf_csv}; rebuild/re-run deploy with updated logger to enable strict frame-index alignment"
            )
        sf_df = pd.read_csv(sf_csv)
        sf_val = sf_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if sf_val.shape[1] < 1:
            raise ValueError("source_frame_index.csv numeric columns are empty")
        sf_all = sf_val[:, -1]
        if np.max(row_idx, initial=-1) >= len(sf_all):
            raise ValueError(
                "source_frame_index.csv shorter than q.csv after filtering; logs are inconsistent"
            )
        source_frame_idx = np.rint(sf_all[row_idx]).astype(np.int64)
        valid = source_frame_idx >= 0
        valid &= source_frame_idx < gt.num_frames
        if not np.any(valid):
            raise ValueError("No valid source_frame_index rows for strict alignment")
        source_frame_valid = int(np.sum(valid))
        source_frame_kept = int(np.sum(valid))
        src = source_frame_idx[valid]
        q29 = q29[valid]
        q43_gt = gt.q43_source[src]
        gt_body_pos_w = (
            gt.body_pos_w_source[src]
            if gt.body_pos_w_source is not None
            else None
        )
        n = len(src)
        eval_motion_keys = [gt.motion_keys[int(i)] for i in src.tolist()]
    elif args.align_mode == "auto_q29":
        align_lag_frames, align_overlap, align_q29_mae = _auto_align_by_q29_mae(
            q29_log=q29,
            q29_gt=gt.q43_source[:, [idx43[nm] for nm in MUJOCO_29_NAMES]],
            lag_min=args.align_lag_min,
            lag_max=args.align_lag_max,
            min_overlap=max(1, args.align_min_overlap),
        )
        if align_lag_frames >= 0:
            q29_aligned = q29[align_lag_frames:]
            q43_gt_aligned = gt.q43_source
        else:
            q29_aligned = q29
            q43_gt_aligned = gt.q43_source[-align_lag_frames:]
        n = min(len(q29_aligned), len(q43_gt_aligned))
        q29 = q29_aligned[:n]
        q43_gt = q43_gt_aligned[:n]
        gt_body_pos_w = gt.body_pos_w_source[:n] if gt.body_pos_w_source is not None else None
        eval_motion_keys = gt.motion_keys[:n]
    else:
        n = min(len(q29), gt.num_frames)
        q29 = q29[:n]
        q43_gt = gt.q43_source[:n]
        gt_body_pos_w = gt.body_pos_w_source[:n] if gt.body_pos_w_source is not None else None
        eval_motion_keys = gt.motion_keys[:n]

    actual_alignment_info: dict[str, Any] = {}
    gt_body_source = "motionlib_body_pos_w" if gt.body_pos_w_source is not None else "fk_from_q43"
    if args.actual_source == "step_sync_body_pos_w_14":
        if args.align_mode != "source_frame_index":
            raise ValueError("--actual-source step_sync_body_pos_w_14 currently requires --align-mode source_frame_index")
        step_src, step_actual_pos, step_ref_pos = read_step_sync_body_pos_w_14(args.logs_dir)
        step_src, step_actual_pos, step_ref_pos = filter_step_sync_body_pos_by_source_frame(
            source_frame_idx=step_src,
            actual_body_pos=step_actual_pos,
            ref_body_pos=step_ref_pos,
            gt_num_frames=gt.num_frames,
            valid_only=bool(args.sim_valid_only),
        )
        pred_pos_arr = step_actual_pos
        gt_pos_arr = step_ref_pos
        q43_gt = gt.q43_source[step_src]
        if gt.body_pos_w_source is not None:
            gt_body_pos_w = gt.body_pos_w_source[step_src]
        eval_motion_keys = [gt.motion_keys[int(i)] for i in step_src.tolist()]
        n = len(step_src)
        source_frame_valid = int(np.sum((step_src >= 0) & (step_src < gt.num_frames)))
        source_frame_kept = int(n)
        actual_alignment_info = {
            "step_sync_rows": int(n),
            "step_sync_gt_comparison_source": "mujoco_ref_body_pos_w_14",
        }
        gt_body_source = "mujoco_ref_body_pos_w_14"
    elif args.actual_source == "body_pos_w_14":
        sim_row_idx, sim_body_pos = read_logged_body_pos_w_14(args.logs_dir)
        _ = sim_row_idx  # row index kept for CSV sanity only; explicit alignment uses source_frame_index.
        sim_source_frame_idx = read_sim_source_frame_index(args.logs_dir)
        if args.align_mode != "source_frame_index":
            raise ValueError("--actual-source body_pos_w_14 currently requires --align-mode source_frame_index")
        aligned_src = src
        if args.sim_valid_only:
            valid_sim_sources = sim_source_frame_idx[sim_source_frame_idx >= 0]
            if valid_sim_sources.size == 0:
                raise ValueError("sim_source_frame_index.csv contains no valid source frame indices")
            sim_min = int(valid_sim_sources.min())
            sim_max = int(valid_sim_sources.max())
            sim_valid_mask = (src >= sim_min) & (src <= sim_max)
            if not np.any(sim_valid_mask):
                raise ValueError(
                    f"No deploy-aligned source frames fall inside sim valid span [{sim_min}, {sim_max}]"
                )
            q29 = q29[sim_valid_mask]
            q43_gt = q43_gt[sim_valid_mask]
            if gt_body_pos_w is not None:
                gt_body_pos_w = gt_body_pos_w[sim_valid_mask]
            eval_motion_keys = [eval_motion_keys[i] for i in np.nonzero(sim_valid_mask)[0].tolist()]
            aligned_src = src[sim_valid_mask]
            n = len(aligned_src)

        pred_pos_arr, aligned_src = align_actual_body_pos_by_source_frame(
            deploy_source_frame_idx=aligned_src,
            sim_source_frame_idx=sim_source_frame_idx,
            sim_body_pos=sim_body_pos,
        )
        if gt_body_pos_w is None:
            raise ValueError("motionlib/parquet GT body_pos_w is unavailable for world-frame comparison")
        gt_pos_arr = gt_body_pos_w
        actual_alignment_info = {
            "step_sync_rows": None,
            "step_sync_gt_comparison_source": None,
        }
    else:
        pred_pos_all: list[np.ndarray] = []
        gt_pos_all: list[np.ndarray] = []
        for i in range(n):
            q_pred = q43_gt[i].copy()
            q_pred[body29_idx_in_43] = q29[i]
            pred_pos_all.append(fk.body_pos(q_pred))
            gt_pos_all.append(fk.body_pos(q43_gt[i]))
        pred_pos_arr = np.stack(pred_pos_all, axis=0)  # [T, B, 3]
        gt_pos_arr = np.stack(gt_pos_all, axis=0)
        actual_alignment_info = {
            "step_sync_rows": None,
            "step_sync_gt_comparison_source": None,
        }
    keypoint_err_mm = np.linalg.norm(pred_pos_arr - gt_pos_arr, axis=-1) * 1000.0  # [T, B]

    # Build per-sequence format expected by compute_metrics_lite: list of [T, B, 3]
    metrics_all_raw, metrics_impl = _compute_metrics_with_smpl(
        [pred_pos_arr],
        [gt_pos_arr],
    )
    reduced_all = _aggregate_metrics_per_motion(metrics_all_raw)

    # Subsets
    subset_raw: dict[str, dict[str, list[np.ndarray]]] = {}
    for suffix, sidx in subset_indices.items():
        if suffix == "":
            continue
        subset_raw[suffix], _ = _compute_metrics_with_smpl(
            [pred_pos_arr[:, sidx, :]],
            [gt_pos_arr[:, sidx, :]],
        )
    reduced_subsets = {sfx: _aggregate_metrics_per_motion(m) for sfx, m in subset_raw.items()}

    # Merge metrics (Isaac-like key naming)
    per_motion_metrics: dict[str, np.ndarray] = {}
    for k, v in reduced_all.items():
        per_motion_metrics[k] = v
    for sfx, d in reduced_subsets.items():
        for k, v in d.items():
            per_motion_metrics[f"{k}{sfx}"] = v

    # Derive frame-level metrics, then aggregate them with Isaac-style episode semantics:
    # one streamed rollout/log replay corresponds to one episode.
    mpjpe_g_frame = _as_frame_metric(metrics_all_raw["mpjpe_g"][0], n, "mpjpe_g")
    mpjpe_l_frame = _as_frame_metric(metrics_all_raw["mpjpe_l"][0], n, "mpjpe_l")
    mpjpe_pa_frame = _as_frame_metric(metrics_all_raw["mpjpe_pa"][0], n, "mpjpe_pa")

    terminated = (
        (mpjpe_l_frame > args.success_thresh_mpjpe_l_mm)
        | (mpjpe_g_frame > args.success_thresh_mpjpe_g_mm)
        | (mpjpe_pa_frame > args.success_thresh_mpjpe_pa_mm)
    )
    episode_terminated = bool(np.any(terminated))
    success_mask = np.asarray([not episode_terminated], dtype=bool)
    progress = np.asarray([1.0 if not episode_terminated else float(np.mean(~terminated))], dtype=np.float64)
    episode_key = _infer_episode_key(motion_name_series, eval_motion_keys, target_motion_name)

    # Keep explicit per-frame detail for debugging/sanity-checking.
    frame_metrics_dict: dict[str, Any] = {}
    frame_metrics_dict["mpjpe_g"] = mpjpe_g_frame.tolist()
    frame_metrics_dict["mpjpe_l"] = mpjpe_l_frame.tolist()
    frame_metrics_dict["mpjpe_pa"] = mpjpe_pa_frame.tolist()

    # Fill subset frame-level from same smpl_sim backend for consistency.
    for suffix, sidx in subset_indices.items():
        if suffix == "":
            continue
        sub = subset_raw[suffix]
        frame_metrics_dict[f"mpjpe_g{suffix}"] = _as_frame_metric(
            sub["mpjpe_g"][0], n, f"mpjpe_g{suffix}"
        ).tolist()
        frame_metrics_dict[f"mpjpe_l{suffix}"] = _as_frame_metric(
            sub["mpjpe_l"][0], n, f"mpjpe_l{suffix}"
        ).tolist()
        frame_metrics_dict[f"mpjpe_pa{suffix}"] = _as_frame_metric(
            sub["mpjpe_pa"][0], n, f"mpjpe_pa{suffix}"
        ).tolist()

    frame_metrics_dict["terminated"] = terminated.tolist()
    frame_metrics_dict["motion_keys"] = eval_motion_keys
    frame_metrics_dict["keypoint_names"] = BODY_FRAMES
    frame_metrics_dict["keypoint_tracking_error_mm_by_frame"] = {
        BODY_FRAMES[j]: keypoint_err_mm[:, j].tolist() for j in range(len(BODY_FRAMES))
    }

    metric_names = list(per_motion_metrics.keys())
    episode_metric_sums: dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name in frame_metrics_dict:
            episode_metric_sums[metric_name] = _mean_episode_metric_sum(
                metric_name, np.asarray(frame_metrics_dict[metric_name], dtype=np.float64)
            )
        else:
            metric_value = float(per_motion_metrics[metric_name][0]) if per_motion_metrics[metric_name].size else 0.0
            episode_metric_sums[metric_name] = metric_value * float(n)

    episode_metric_means = {
        metric_name: _episode_metric_from_sum(metric_name, metric_sum, n)
        for metric_name, metric_sum in episode_metric_sums.items()
    }
    for j, name in enumerate(BODY_FRAMES):
        kp_arr = keypoint_err_mm[:, j]
        metric_name = f"kp_err_mm::{name}"
        episode_metric_sums[metric_name] = float(np.sum(kp_arr))
        episode_metric_means[metric_name] = _episode_metric_from_sum(
            metric_name, episode_metric_sums[metric_name], n
        )

    all_metrics_dict: dict[str, Any] = {
        metric_name: [metric_value] for metric_name, metric_value in episode_metric_means.items()
    }
    all_metrics_dict["terminated"] = [episode_terminated]
    all_metrics_dict["progress"] = progress.tolist()
    all_metrics_dict["motion_keys"] = [episode_key]
    all_metrics_dict["sampling_prob"] = [1.0]

    failed_metrics_dict: dict[str, Any] = {}
    if episode_terminated:
        for metric_name, metric_value in episode_metric_means.items():
            failed_metrics_dict[metric_name] = [metric_value]
        failed_metrics_dict["motion_keys"] = [episode_key]
        failed_metrics_dict["sampling_prob"] = [1.0]

    metrics_all = dict(episode_metric_means)
    metrics_success = (
        dict(episode_metric_means)
        if not episode_terminated
        else {metric_name: 0.0 for metric_name in episode_metric_means.keys()}
    )
    metrics_success["success_rate"] = float(success_mask.astype(np.float64).mean()) if n > 0 else 0.0
    metrics_success["progress_rate"] = float(progress.mean()) if n > 0 else 0.0

    failed_idx = [0] if episode_terminated else []

    result = {
        "gt_format": args.gt_format,
        "parquet": str(args.parquet) if args.parquet is not None else None,
        "motion_file": str(args.motion_file) if args.motion_file is not None else None,
        "motion_name": args.motion_name,
        "logs_dir": str(args.logs_dir),
        "actual_source": args.actual_source,
        "gt_body_source": gt_body_source,
        "motionlib_source": gt.source,
        "num_frames": int(n),
        "alignment": {
            "mode": args.align_mode,
            "lag_frames_log_vs_gt": int(align_lag_frames),
            "auto_overlap_frames": int(align_overlap) if align_overlap is not None else None,
            "auto_q29_mae_rad": float(align_q29_mae) if align_q29_mae is not None else None,
            "streamed_only": bool(args.streamed_only),
            "sim_valid_only": bool(args.sim_valid_only),
            "streamed_row_start_in_logs": streamed_row_start,
            "source_frame_valid_rows": source_frame_valid,
            "source_frame_kept_rows": source_frame_kept,
            **actual_alignment_info,
        },
        "gt_source": args.gt_source,
        "gt_motion_dir": str(args.gt_motion_dir) if args.gt_motion_dir is not None else None,
        "target_motion_name": target_motion_name,
        "metric_backend": metrics_impl,
        "thresholds": {
            "mpjpe_l_mm": float(args.success_thresh_mpjpe_l_mm),
            "mpjpe_g_mm": float(args.success_thresh_mpjpe_g_mm),
            "mpjpe_pa_mm": float(args.success_thresh_mpjpe_pa_mm),
        },
        "metrics_all": metrics_all,
        "metrics_success": metrics_success,
        "all_metrics_dict": all_metrics_dict,
        "frame_metrics_dict": frame_metrics_dict,
        "failed_metrics_dict": failed_metrics_dict,
        "failed_idxes": failed_idx,
        "failed_keys": failed_metrics_dict.get("motion_keys", []),
        "success_keys": [episode_key] if not episode_terminated else [],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
