#!/usr/bin/env python3
"""
Real-time error visualization for Sonic deploy in MuJoCo.

Preferred source:
- ZMQ g1_debug topic (requires pyzmq + msgpack-numpy in runtime env)
Fallback source:
- StateLogger CSV logs (q.csv, action.csv, timestamp inferred by row index and dt)

GT source:
- official parquet episode with action.wbc(43) / observation.state(43) / observation.eef_state(14)

Error definitions (all in MuJoCo 29 order unless specified):
- measured tracking error: body_q_measured - gt_body_q
- control tracking error:  last_action - gt_body_q
- overshoot metric: max(0, sign(gt_delta) * (pred - gt)) per joint, where gt_delta = gt_t - gt_{t-1}

End-effector accuracy:
- FK on 43D predicted/measured state vs:
  - FK(gt 43D source), or
  - gt observation.eef_state
  selected by --eef-gt-source
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import time

# Optional runtime deps for live ZMQ mode.
try:
    import zmq  # type: ignore
    import msgpack  # type: ignore
    import msgpack_numpy as mnp  # type: ignore

    mnp.patch()
    HAS_ZMQ = True
except Exception:
    HAS_ZMQ = False


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

# 43D observation.state order from official dataset meta/info.json.
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

# deploy/parquet conversion mapping used in parquet_to_mujoco_motion.py
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
for mj_i, isaac_i in enumerate(MUJOCO_TO_ISAACLAB_29):
    ISAACLAB_TO_MUJOCO_29[isaac_i] = mj_i


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rot_err_rad(q_pred_wxyz: np.ndarray, q_gt_wxyz: np.ndarray) -> float:
    r_pred = R.from_quat(quat_wxyz_to_xyzw(q_pred_wxyz))
    r_gt = R.from_quat(quat_wxyz_to_xyzw(q_gt_wxyz))
    return float((r_pred * r_gt.inv()).magnitude())


def summary_stats(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {"mean": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(np.square(x)))),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


@dataclass
class GTData:
    dt: float
    source_name: str
    body_q_29: np.ndarray
    q43_source: np.ndarray
    eef_14: np.ndarray
    root_quat_4: np.ndarray
    gt_from_motion_dir: bool = False

    @property
    def num_frames(self) -> int:
        return int(self.body_q_29.shape[0])


def _target_motion_name(gt_motion_dir: Path | None, target_motion_name: str | None) -> str | None:
    if target_motion_name:
        return target_motion_name
    if gt_motion_dir is not None:
        return gt_motion_dir.name
    return None


def _load_log_mask(logs_dir: Path, target_motion_name: str | None) -> np.ndarray | None:
    motion_playing_csv = logs_dir / "motion_playing.csv"
    motion_name_csv = logs_dir / "motion_name.csv"

    play_mask = None
    name_mask = None

    if motion_playing_csv.exists():
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


def _load_motion_dir_body_q29_mujoco(motion_dir: Path) -> np.ndarray:
    joint_pos_csv = motion_dir / "joint_pos.csv"
    if not joint_pos_csv.exists():
        raise FileNotFoundError(f"missing {joint_pos_csv}")
    jp = pd.read_csv(joint_pos_csv).select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if jp.shape[1] < 29:
        raise ValueError(f"joint_pos.csv numeric columns less than 29: {jp.shape[1]}")
    jp_isaac = jp[:, :29]
    return jp_isaac[:, ISAACLAB_TO_MUJOCO_29]


def load_gt_from_parquet(parquet_path: Path, gt_source: str, gt_motion_dir: Path | None = None) -> GTData:
    df = pd.read_parquet(parquet_path)

    state = np.stack(df["observation.state"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    if state.shape[1] != 43:
        raise ValueError(f"observation.state must be 43D, got {state.shape}")

    action = np.stack(df["action.wbc"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    if action.shape[1] != 43:
        raise ValueError(f"action.wbc must be 43D, got {action.shape}")

    eef = np.stack(df["observation.eef_state"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    root = np.stack(df["observation.root_orientation"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)

    if gt_source == "observation.state":
        q43_source = state
    elif gt_source == "action.wbc":
        q43_source = action
    else:
        raise ValueError(f"unsupported gt_source: {gt_source}")

    # Build 29-index map by names.
    idx43 = {n: i for i, n in enumerate(STATE43_NAMES)}
    body29_idx = [idx43[n] for n in MUJOCO_29_NAMES]
    body_q_29 = q43_source[:, body29_idx].copy()

    gt_from_motion_dir = False
    if gt_motion_dir is not None:
        body_q_29_from_motion = _load_motion_dir_body_q29_mujoco(gt_motion_dir)
        n_sync = min(len(body_q_29_from_motion), len(body_q_29))
        body_q_29 = body_q_29_from_motion[:n_sync]
        q43_source = q43_source[:n_sync].copy()
        q43_source[:, body29_idx] = body_q_29
        eef = eef[:n_sync]
        root = root[:n_sync]
        gt_from_motion_dir = True

    if "timestamp" in df.columns and len(df) >= 2:
        t = np.asarray(df["timestamp"], dtype=np.float64)
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        dt = float(np.median(d)) if len(d) else 0.02
    else:
        dt = 0.02

    return GTData(
        dt=dt,
        source_name=gt_source,
        body_q_29=body_q_29,
        q43_source=q43_source,
        eef_14=eef,
        root_quat_4=root,
        gt_from_motion_dir=gt_from_motion_dir,
    )


class WristFK:
    def __init__(self, urdf_path: Path, left_frame: str = "left_wrist_yaw_link", right_frame: str = "right_wrist_yaw_link"):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.left_frame_id = self.model.getFrameId(left_frame)
        self.right_frame_id = self.model.getFrameId(right_frame)

    def eef_from_q43(self, q43: np.ndarray) -> np.ndarray:
        pin.framesForwardKinematics(self.model, self.data, q43)

        l_pose = self.data.oMf[self.left_frame_id]
        r_pose = self.data.oMf[self.right_frame_id]

        l_pos = np.asarray(l_pose.translation)
        r_pos = np.asarray(r_pose.translation)

        l_q_xyzw = R.from_matrix(np.asarray(l_pose.rotation)).as_quat()
        r_q_xyzw = R.from_matrix(np.asarray(r_pose.rotation)).as_quat()

        l_q_wxyz = quat_xyzw_to_wxyz(l_q_xyzw)
        r_q_wxyz = quat_xyzw_to_wxyz(r_q_xyzw)

        return np.concatenate([l_pos, l_q_wxyz, r_pos, r_q_wxyz], axis=0)


def get_eef_gt(
    gt: GTData,
    fk: WristFK,
    idx: int,
    eef_gt_source: str,
) -> np.ndarray:
    if eef_gt_source == "from_gt_q43":
        return fk.eef_from_q43(gt.q43_source[idx])
    if eef_gt_source == "observation.eef_state":
        return gt.eef_14[idx]
    raise ValueError(f"unsupported eef_gt_source: {eef_gt_source}")


def print_live_line(step: int, t_sec: float, pos_rmse: float, ctrl_rmse: float, eef_pos_cm: float, eef_rot_deg: float) -> None:
    print(
        f"step={step:05d} t={t_sec:7.3f}s | "
        f"q_meas_rmse={pos_rmse:7.4f} rad | "
        f"q_ctrl_rmse={ctrl_rmse:7.4f} rad | "
        f"eef_pos={eef_pos_cm:6.2f} cm | eef_rot={eef_rot_deg:6.2f} deg"
    )


def run_offline_logs_mode(args: argparse.Namespace, gt: GTData, fk: WristFK) -> dict[str, Any]:
    logs_dir = Path(args.logs_dir)
    q_csv = logs_dir / "q.csv"
    action_csv = logs_dir / "action.csv"

    if not q_csv.exists():
        raise FileNotFoundError(f"missing {q_csv}")
    if not action_csv.exists():
        raise FileNotFoundError(f"missing {action_csv}")

    q_df = pd.read_csv(q_csv)
    a_df = pd.read_csv(action_csv)

    # Robustly pick numeric columns after first metadata columns.
    q_val = q_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    a_val = a_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)

    # Last 29 numeric columns are joint values in q.csv/action.csv format.
    if q_val.shape[1] < 29 or a_val.shape[1] < 29:
        raise ValueError("q.csv/action.csv numeric columns less than 29")
    q29 = q_val[:, -29:]
    act29 = a_val[:, -29:]

    target_motion_name = _target_motion_name(args.gt_motion_dir, args.target_motion_name)
    log_mask = _load_log_mask(logs_dir, target_motion_name)
    if log_mask is not None:
        n_mask = min(len(q29), len(act29), len(log_mask))
        q29 = q29[:n_mask]
        act29 = act29[:n_mask]
        log_mask = log_mask[:n_mask]
        if not np.any(log_mask):
            raise ValueError(
                "No valid log rows matched the requested motion/play mask. "
                f"target_motion_name={target_motion_name}"
            )
        q29 = q29[log_mask]
        act29 = act29[log_mask]

    n = min(len(q29), len(act29), gt.num_frames)
    q29 = q29[:n]
    act29 = act29[:n]

    idx43 = {n_: i for i, n_ in enumerate(STATE43_NAMES)}
    body43_idx = [idx43[n_] for n_ in MUJOCO_29_NAMES]

    pos_err_all = []
    ctrl_err_all = []
    overshoot_all = []
    eef_pos_err_all = []
    eef_rot_err_all = []

    for i in range(n):
        gt29 = gt.body_q_29[i]

        pos_err = q29[i] - gt29
        ctrl_err = act29[i] - gt29
        pos_err_all.append(pos_err)
        ctrl_err_all.append(ctrl_err)

        if i == 0:
            gt_delta = np.zeros_like(gt29)
        else:
            gt_delta = gt.body_q_29[i] - gt.body_q_29[i - 1]
        overshoot = np.maximum(0.0, np.sign(gt_delta) * (q29[i] - gt29))
        overshoot_all.append(overshoot)

        q43_pred = gt.q43_source[i].copy()
        q43_pred[body43_idx] = q29[i]
        eef_pred = fk.eef_from_q43(q43_pred)
        eef_gt = get_eef_gt(gt, fk, i, args.eef_gt_source)

        l_pos = np.linalg.norm(eef_pred[0:3] - eef_gt[0:3])
        r_pos = np.linalg.norm(eef_pred[7:10] - eef_gt[7:10])
        l_rot = rot_err_rad(eef_pred[3:7], eef_gt[3:7])
        r_rot = rot_err_rad(eef_pred[10:14], eef_gt[10:14])

        eef_pos = max(l_pos, r_pos)
        eef_rot = max(l_rot, r_rot)
        eef_pos_err_all.append(eef_pos)
        eef_rot_err_all.append(eef_rot)

        if args.print_every > 0 and (i % args.print_every == 0):
            print_live_line(
                step=i,
                t_sec=i * gt.dt,
                pos_rmse=float(np.sqrt(np.mean(pos_err**2))),
                ctrl_rmse=float(np.sqrt(np.mean(ctrl_err**2))),
                eef_pos_cm=float(eef_pos * 100.0),
                eef_rot_deg=float(np.rad2deg(eef_rot)),
            )

    pos_err_all = np.asarray(pos_err_all)
    ctrl_err_all = np.asarray(ctrl_err_all)
    overshoot_all = np.asarray(overshoot_all)
    eef_pos_err_all = np.asarray(eef_pos_err_all)
    eef_rot_err_all = np.asarray(eef_rot_err_all)

    result = {
        "mode": "offline_logs",
        "gt_source": gt.source_name,
        "eef_gt_source": args.eef_gt_source,
        "gt_from_motion_dir": gt.gt_from_motion_dir,
        "target_motion_name": target_motion_name,
        "num_frames": int(n),
        "dt": gt.dt,
        "joint_measured_error_rad": summary_stats(np.linalg.norm(pos_err_all, axis=1)),
        "joint_control_error_rad": summary_stats(np.linalg.norm(ctrl_err_all, axis=1)),
        "joint_measured_overshoot_rad": summary_stats(np.linalg.norm(overshoot_all, axis=1)),
        "eef_pos_error_m": summary_stats(eef_pos_err_all),
        "eef_rot_error_rad": summary_stats(eef_rot_err_all),
    }
    return result


def _iter_numeric_tail_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        _ = f.readline()  # header
        while True:
            line = f.readline()
            if not line:
                yield None
                continue
            parts = line.strip().split(",")
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if vals:
                yield np.asarray(vals, dtype=np.float64)
            else:
                yield None


def run_tail_logs_mode(args: argparse.Namespace, gt: GTData, fk: WristFK) -> dict[str, Any]:
    logs_dir = Path(args.logs_dir)
    q_csv = logs_dir / "q.csv"
    action_csv = logs_dir / "action.csv"
    if not q_csv.exists():
        raise FileNotFoundError(f"missing {q_csv}")
    if not action_csv.exists():
        raise FileNotFoundError(f"missing {action_csv}")

    idx43 = {n_: i for i, n_ in enumerate(STATE43_NAMES)}
    body43_idx = [idx43[n_] for n_ in MUJOCO_29_NAMES]

    q_iter = _iter_numeric_tail_rows(q_csv)
    a_iter = _iter_numeric_tail_rows(action_csv)

    pos_norm = []
    ctrl_norm = []
    over_norm = []
    eef_pos = []
    eef_rot = []

    step = 0
    idle_rounds = 0
    while step < args.max_steps:
        q_row = next(q_iter)
        a_row = next(a_iter)

        if q_row is None or a_row is None:
            idle_rounds += 1
            if idle_rounds > args.tail_timeout_rounds:
                break
            time.sleep(args.tail_poll_sec)
            continue
        idle_rounds = 0

        if q_row.shape[0] < 29 or a_row.shape[0] < 29:
            continue
        q29 = q_row[-29:]
        act29 = a_row[-29:]

        gt_idx = min(step, gt.num_frames - 1)
        gt29 = gt.body_q_29[gt_idx]

        pe = q29 - gt29
        ce = act29 - gt29
        if gt_idx == 0:
            gdel = np.zeros_like(gt29)
        else:
            gdel = gt.body_q_29[gt_idx] - gt.body_q_29[gt_idx - 1]
        ov = np.maximum(0.0, np.sign(gdel) * (q29 - gt29))

        q43_pred = gt.q43_source[gt_idx].copy()
        q43_pred[body43_idx] = q29
        eef_pred = fk.eef_from_q43(q43_pred)
        eef_gt = get_eef_gt(gt, fk, gt_idx, args.eef_gt_source)

        l_pos = np.linalg.norm(eef_pred[0:3] - eef_gt[0:3])
        r_pos = np.linalg.norm(eef_pred[7:10] - eef_gt[7:10])
        l_rot = rot_err_rad(eef_pred[3:7], eef_gt[3:7])
        r_rot = rot_err_rad(eef_pred[10:14], eef_gt[10:14])
        ep = max(l_pos, r_pos)
        er = max(l_rot, r_rot)

        pos_norm.append(float(np.linalg.norm(pe)))
        ctrl_norm.append(float(np.linalg.norm(ce)))
        over_norm.append(float(np.linalg.norm(ov)))
        eef_pos.append(float(ep))
        eef_rot.append(float(er))

        if args.print_every > 0 and (step % args.print_every == 0):
            print_live_line(
                step=step,
                t_sec=step * gt.dt,
                pos_rmse=float(np.sqrt(np.mean(pe**2))),
                ctrl_rmse=float(np.sqrt(np.mean(ce**2))),
                eef_pos_cm=ep * 100.0,
                eef_rot_deg=np.rad2deg(er),
            )
        step += 1

    result = {
        "mode": "tail_logs",
        "gt_source": gt.source_name,
        "eef_gt_source": args.eef_gt_source,
        "gt_from_motion_dir": gt.gt_from_motion_dir,
        "num_frames": int(step),
        "dt": gt.dt,
        "joint_measured_error_rad": summary_stats(np.asarray(pos_norm)),
        "joint_control_error_rad": summary_stats(np.asarray(ctrl_norm)),
        "joint_measured_overshoot_rad": summary_stats(np.asarray(over_norm)),
        "eef_pos_error_m": summary_stats(np.asarray(eef_pos)),
        "eef_rot_error_rad": summary_stats(np.asarray(eef_rot)),
    }
    return result


def _unpack_msgpack_zmq(raw: bytes, topic: str) -> dict:
    payload = raw[len(topic) :]
    msg = msgpack.unpackb(payload, raw=False)
    if isinstance(msg, dict):
        out: dict[str, Any] = {}
        for k, v in msg.items():
            key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            out[key] = v
        return out
    return {}


def run_live_zmq_mode(args: argparse.Namespace, gt: GTData, fk: WristFK) -> dict[str, Any]:
    if not HAS_ZMQ:
        raise RuntimeError(
            "live_zmq mode requires pyzmq + msgpack_numpy in runtime env. "
            "Install them into your running environment or use --mode offline_logs."
        )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.zmq_topic)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://{args.zmq_host}:{args.zmq_port}")

    idx43 = {n_: i for i, n_ in enumerate(STATE43_NAMES)}
    body43_idx = [idx43[n_] for n_ in MUJOCO_29_NAMES]

    pos_norm = []
    ctrl_norm = []
    over_norm = []
    eef_pos = []
    eef_rot = []

    step = 0
    try:
        while step < args.max_steps:
            raw = sock.recv()
            msg = _unpack_msgpack_zmq(raw, args.zmq_topic)

            q_meas = np.asarray(msg.get("body_q_measured", msg.get("body_q", np.zeros(29))), dtype=np.float64)
            q_ctrl = np.asarray(msg.get("last_action", np.zeros(29)), dtype=np.float64)
            if q_meas.shape[0] != 29 or q_ctrl.shape[0] != 29:
                continue

            gt_idx = min(step, gt.num_frames - 1)
            gt29 = gt.body_q_29[gt_idx]

            pe = q_meas - gt29
            ce = q_ctrl - gt29

            if gt_idx == 0:
                gdel = np.zeros_like(gt29)
            else:
                gdel = gt.body_q_29[gt_idx] - gt.body_q_29[gt_idx - 1]
            ov = np.maximum(0.0, np.sign(gdel) * (q_meas - gt29))

            q43_pred = gt.q43_source[gt_idx].copy()
            q43_pred[body43_idx] = q_meas
            eef_pred = fk.eef_from_q43(q43_pred)
            eef_gt = get_eef_gt(gt, fk, gt_idx, args.eef_gt_source)

            l_pos = np.linalg.norm(eef_pred[0:3] - eef_gt[0:3])
            r_pos = np.linalg.norm(eef_pred[7:10] - eef_gt[7:10])
            l_rot = rot_err_rad(eef_pred[3:7], eef_gt[3:7])
            r_rot = rot_err_rad(eef_pred[10:14], eef_gt[10:14])
            ep = max(l_pos, r_pos)
            er = max(l_rot, r_rot)

            pos_norm.append(float(np.linalg.norm(pe)))
            ctrl_norm.append(float(np.linalg.norm(ce)))
            over_norm.append(float(np.linalg.norm(ov)))
            eef_pos.append(float(ep))
            eef_rot.append(float(er))

            if args.print_every > 0 and (step % args.print_every == 0):
                print_live_line(
                    step=step,
                    t_sec=step * gt.dt,
                    pos_rmse=float(np.sqrt(np.mean(pe**2))),
                    ctrl_rmse=float(np.sqrt(np.mean(ce**2))),
                    eef_pos_cm=ep * 100.0,
                    eef_rot_deg=np.rad2deg(er),
                )

            step += 1

    finally:
        sock.close()
        ctx.term()

    result = {
        "mode": "live_zmq",
        "gt_source": gt.source_name,
        "eef_gt_source": args.eef_gt_source,
        "gt_from_motion_dir": gt.gt_from_motion_dir,
        "num_frames": int(step),
        "dt": gt.dt,
        "joint_measured_error_rad": summary_stats(np.asarray(pos_norm)),
        "joint_control_error_rad": summary_stats(np.asarray(ctrl_norm)),
        "joint_measured_overshoot_rad": summary_stats(np.asarray(over_norm)),
        "eef_pos_error_m": summary_stats(np.asarray(eef_pos)),
        "eef_rot_error_rad": summary_stats(np.asarray(eef_rot)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime/offline error visualization for Sonic deploy")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--gt-source",
        type=str,
        choices=["observation.state", "action.wbc"],
        default="action.wbc",
        help="Ground-truth source for joint control tracking error",
    )
    parser.add_argument(
        "--eef-gt-source",
        type=str,
        choices=["from_gt_q43", "observation.eef_state"],
        default="from_gt_q43",
        help=(
            "GT source for end-effector error. "
            "from_gt_q43 uses FK on the same --gt-source q43 trajectory."
        ),
    )
    parser.add_argument(
        "--gt-motion-dir",
        type=Path,
        default=None,
        help=(
            "Optional converted motion folder (contains joint_pos.csv). "
            "If set, body-joint GT is read from this folder to exactly reuse "
            "parquet_to_mujoco_motion conversion output."
        ),
    )
    parser.add_argument(
        "--target-motion-name",
        type=str,
        default=None,
        help="Optional explicit target motion name used to filter logs by motion_name.csv",
    )
    parser.add_argument("--urdf", type=Path, default=Path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"))
    parser.add_argument("--mode", choices=["live_zmq", "offline_logs", "tail_logs"], default="offline_logs")
    parser.add_argument("--logs-dir", type=Path, default=None, help="required for offline_logs")
    parser.add_argument("--zmq-host", type=str, default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, default=5557)
    parser.add_argument("--zmq-topic", type=str, default="g1_debug")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--tail-poll-sec", type=float, default=0.02)
    parser.add_argument("--tail-timeout-rounds", type=int, default=500)
    parser.add_argument("--out-json", type=Path, default=Path("/tmp/sonic_error_metrics.json"))
    args = parser.parse_args()

    gt = load_gt_from_parquet(args.parquet, args.gt_source, args.gt_motion_dir)
    fk = WristFK(args.urdf)

    if args.mode == "offline_logs":
        if args.logs_dir is None:
            raise ValueError("--logs-dir is required when --mode offline_logs")
        result = run_offline_logs_mode(args, gt, fk)
    elif args.mode == "tail_logs":
        if args.logs_dir is None:
            raise ValueError("--logs-dir is required when --mode tail_logs")
        result = run_tail_logs_mode(args, gt, fk)
    else:
        result = run_live_zmq_mode(args, gt, fk)

    print("\n=== FINAL METRICS ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
