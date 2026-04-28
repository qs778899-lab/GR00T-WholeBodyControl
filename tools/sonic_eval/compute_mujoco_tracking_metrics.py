#!/usr/bin/env python3
"""
Compute Isaac-eval-style tracking metrics from MuJoCo sim2sim logs.

This script tries to reuse the same metric function as Isaac eval:
`smpl_sim.smpllib.smpl_eval.compute_metrics_lite`.

If `smpl_sim` is unavailable, it can fall back to a local implementation for:
- mpjpe_g / mpjpe_l / mpjpe_pa
and their subset variants:
- *_legs, *_vr_3points, *_other_upper_bodies, *_foot

To strictly match IsaacSim eval definitions, run with default behavior
(fallback disabled) and ensure `smpl_sim` is importable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pinocchio as pin

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
    motion_keys: list[str]

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
    return GTData(q43_source=q43, motion_keys=motion_keys)


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
    motion_keys = [f"{seq.motion_name}:frame_{i:06d}" for i in range(len(q43))]
    return GTData(q43_source=q43, motion_keys=motion_keys)


class BodyFK:
    def __init__(self, urdf: Path, body_frames: list[str]):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame_ids = [self.model.getFrameId(n) for n in body_frames]
        if any(fid >= len(self.model.frames) for fid in self.frame_ids):
            raise ValueError("Some body frames not found in URDF")

    def body_pos(self, q43: np.ndarray) -> np.ndarray:
        pin.framesForwardKinematics(self.model, self.data, q43)
        return np.stack(
            [np.asarray(self.data.oMf[fid].translation, dtype=np.float64) for fid in self.frame_ids],
            axis=0,
        )


def _similarity_transform(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    mu_x = pred.mean(axis=0, keepdims=True)
    mu_y = gt.mean(axis=0, keepdims=True)
    x0 = pred - mu_x
    y0 = gt - mu_y
    var_x = np.sum(x0 * x0)
    if var_x < 1e-12:
        return pred.copy()
    k = x0.T @ y0
    u, s, vt = np.linalg.svd(k)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt
    scale = float(np.sum(s) / var_x)
    t = mu_y - scale * (mu_x @ r)
    return scale * (pred @ r) + t


def _mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1) * 1000.0


def _compute_metrics_lite_fallback(pred_pos_all: list[np.ndarray], gt_pos_all: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    out_g: list[np.ndarray] = []
    out_l: list[np.ndarray] = []
    out_pa: list[np.ndarray] = []
    for p_seq, g_seq in zip(pred_pos_all, gt_pos_all):
        t = p_seq.shape[0]
        g_arr = np.zeros((t,), dtype=np.float64)
        l_arr = np.zeros((t,), dtype=np.float64)
        pa_arr = np.zeros((t,), dtype=np.float64)
        for i in range(t):
            p = p_seq[i]
            g = g_seq[i]
            g_arr[i] = float(_mpjpe_mm(p, g).mean())
            p_l = p - p[0:1]
            g_l = g - g[0:1]
            l_arr[i] = float(_mpjpe_mm(p_l, g_l).mean())
            p_pa = _similarity_transform(p, g)
            pa_arr[i] = float(_mpjpe_mm(p_pa, g).mean())
        out_g.append(g_arr)
        out_l.append(l_arr)
        out_pa.append(pa_arr)
    return {"mpjpe_g": out_g, "mpjpe_l": out_l, "mpjpe_pa": out_pa}


def _compute_metrics_with_optional_smpl(
    pred_pos_all: list[np.ndarray],
    gt_pos_all: list[np.ndarray],
    allow_fallback_metrics: bool,
) -> tuple[dict[str, list[np.ndarray]], str]:
    try:
        from smpl_sim.smpllib.smpl_eval import compute_metrics_lite  # type: ignore
    except Exception as e:
        if not allow_fallback_metrics:
            raise RuntimeError(
                "smpl_sim.smpllib.smpl_eval.compute_metrics_lite is required for IsaacSim-aligned metrics, "
                "but import failed. Install/activate env with smpl_sim, or pass --allow-fallback-metrics "
                "if you explicitly accept non-official fallback metrics."
            ) from e
        return _compute_metrics_lite_fallback(pred_pos_all, gt_pos_all), "fallback_local_mpjpe"
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
    parser.add_argument("--align-lag-min", type=int, default=-150, help="min lag for auto_q29")
    parser.add_argument("--align-lag-max", type=int, default=150, help="max lag for auto_q29")
    parser.add_argument("--align-min-overlap", type=int, default=200, help="min overlap for auto_q29")
    parser.add_argument(
        "--allow-fallback-metrics",
        action="store_true",
        help="allow local fallback metric implementation when smpl_sim is unavailable (not IsaacSim-official)",
    )
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

    fk = BodyFK(args.urdf, BODY_FRAMES)

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
        q29 = q29[valid]
        src = source_frame_idx[valid]
        q43_gt = gt.q43_source[src]
        n = len(q29)
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
        eval_motion_keys = gt.motion_keys[:n]
    else:
        n = min(len(q29), gt.num_frames)
        q29 = q29[:n]
        q43_gt = gt.q43_source[:n]
        eval_motion_keys = gt.motion_keys[:n]

    pred_pos_all: list[np.ndarray] = []
    gt_pos_all: list[np.ndarray] = []
    for i in range(n):
        q_pred = q43_gt[i].copy()
        q_pred[body29_idx_in_43] = q29[i]
        pred_pos_all.append(fk.body_pos(q_pred))
        gt_pos_all.append(fk.body_pos(q43_gt[i]))
    pred_pos_arr = np.stack(pred_pos_all, axis=0)  # [T, B, 3]
    gt_pos_arr = np.stack(gt_pos_all, axis=0)

    # Build per-sequence format expected by compute_metrics_lite: list of [T, B, 3]
    metrics_all_raw, metrics_impl = _compute_metrics_with_optional_smpl(
        [pred_pos_arr],
        [gt_pos_arr],
        allow_fallback_metrics=args.allow_fallback_metrics,
    )
    reduced_all = _aggregate_metrics_per_motion(metrics_all_raw)

    # Subsets
    subset_raw: dict[str, dict[str, list[np.ndarray]]] = {}
    for suffix, sidx in subset_indices.items():
        if suffix == "":
            continue
        subset_raw[suffix], _ = _compute_metrics_with_optional_smpl(
            [pred_pos_arr[:, sidx, :]],
            [gt_pos_arr[:, sidx, :]],
            allow_fallback_metrics=args.allow_fallback_metrics,
        )
    reduced_subsets = {sfx: _aggregate_metrics_per_motion(m) for sfx, m in subset_raw.items()}

    # Merge metrics (Isaac-like key naming)
    per_motion_metrics: dict[str, np.ndarray] = {}
    for k, v in reduced_all.items():
        per_motion_metrics[k] = v
    for sfx, d in reduced_subsets.items():
        for k, v in d.items():
            per_motion_metrics[f"{k}{sfx}"] = v

    # For this script each frame is treated as one "motion key"
    # (single-episode online eval proxy).
    # Derive frame-level proxy metrics for termination/success from unaggregated all-body mpjpe.
    # If smpl_sim path returned extra keys, we still use mpjpe_* for termination.
    mpjpe_g_frame = _compute_metrics_lite_fallback([pred_pos_arr], [gt_pos_arr])["mpjpe_g"][0]
    mpjpe_l_frame = _compute_metrics_lite_fallback([pred_pos_arr], [gt_pos_arr])["mpjpe_l"][0]
    mpjpe_pa_frame = _compute_metrics_lite_fallback([pred_pos_arr], [gt_pos_arr])["mpjpe_pa"][0]

    terminated = (
        (mpjpe_l_frame > args.success_thresh_mpjpe_l_mm)
        | (mpjpe_g_frame > args.success_thresh_mpjpe_g_mm)
        | (mpjpe_pa_frame > args.success_thresh_mpjpe_pa_mm)
    )
    success_mask = ~terminated
    progress = np.ones_like(mpjpe_l_frame, dtype=np.float64)

    # Expand aggregated metrics to frame-wise arrays for all_metrics_dict compatibility.
    # For keys lacking frame-level definition, repeat scalar mean per frame.
    all_metrics_dict: dict[str, Any] = {}
    for k, v in per_motion_metrics.items():
        scalar = float(v[0]) if v.size else 0.0
        all_metrics_dict[k] = [scalar] * n

    # Overwrite known frame-level keys with true per-frame values.
    all_metrics_dict["mpjpe_g"] = mpjpe_g_frame.tolist()
    all_metrics_dict["mpjpe_l"] = mpjpe_l_frame.tolist()
    all_metrics_dict["mpjpe_pa"] = mpjpe_pa_frame.tolist()

    # Fill subset frame-level via fallback exact computation.
    for suffix, sidx in subset_indices.items():
        if suffix == "":
            continue
        sub = _compute_metrics_lite_fallback(
            [pred_pos_arr[:, sidx, :]],
            [gt_pos_arr[:, sidx, :]],
        )
        all_metrics_dict[f"mpjpe_g{suffix}"] = sub["mpjpe_g"][0].tolist()
        all_metrics_dict[f"mpjpe_l{suffix}"] = sub["mpjpe_l"][0].tolist()
        all_metrics_dict[f"mpjpe_pa{suffix}"] = sub["mpjpe_pa"][0].tolist()

    all_metrics_dict["terminated"] = terminated.tolist()
    all_metrics_dict["progress"] = progress.tolist()
    all_metrics_dict["motion_keys"] = eval_motion_keys
    all_metrics_dict["sampling_prob"] = [1.0 / n] * n if n > 0 else []

    failed_idx = np.nonzero(terminated)[0]
    failed_metrics_dict: dict[str, Any] = {}
    for k, v in all_metrics_dict.items():
        if isinstance(v, list) and len(v) == n:
            failed_metrics_dict[k] = [v[i] for i in failed_idx.tolist()]
    failed_metrics_dict["motion_keys"] = [eval_motion_keys[i] for i in failed_idx.tolist()]
    failed_metrics_dict["sampling_prob"] = [1.0 / n] * len(failed_idx) if n > 0 else []

    metrics_all = {}
    metrics_success = {}
    for k, v in all_metrics_dict.items():
        if not isinstance(v, list) or len(v) != n or k in {"motion_keys", "sampling_prob", "terminated", "progress"}:
            continue
        arr = np.asarray(v, dtype=np.float64)
        metrics_all[k] = _mean_or_zero(arr)
        metrics_success[k] = _mean_or_zero(arr[success_mask]) if np.any(success_mask) else 0.0
    metrics_success["success_rate"] = float(np.mean(success_mask.astype(np.float64))) if n > 0 else 0.0
    metrics_success["progress_rate"] = float(np.mean(progress)) if n > 0 else 0.0

    result = {
        "gt_format": args.gt_format,
        "parquet": str(args.parquet) if args.parquet is not None else None,
        "motion_file": str(args.motion_file) if args.motion_file is not None else None,
        "motion_name": args.motion_name,
        "logs_dir": str(args.logs_dir),
        "num_frames": int(n),
        "alignment": {
            "mode": args.align_mode,
            "lag_frames_log_vs_gt": int(align_lag_frames),
            "auto_overlap_frames": int(align_overlap) if align_overlap is not None else None,
            "auto_q29_mae_rad": float(align_q29_mae) if align_q29_mae is not None else None,
            "streamed_only": bool(args.streamed_only),
            "streamed_row_start_in_logs": streamed_row_start,
            "source_frame_valid_rows": source_frame_valid,
            "source_frame_kept_rows": source_frame_kept,
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
        "failed_metrics_dict": failed_metrics_dict,
        "failed_idxes": failed_idx.tolist(),
        "failed_keys": failed_metrics_dict.get("motion_keys", []),
        "success_keys": [eval_motion_keys[i] for i in np.nonzero(success_mask)[0].tolist()],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
