#!/usr/bin/env python3
"""
Offline end-effector accuracy on parquet trajectory.

Computes FK wrist poses from observation.state (43D) and compares against
observation.eef_state (14D = left[pos+quat] + right[pos+quat]).

This gives a deterministic sanity metric for dataset consistency and provides
an end-effector error baseline independent of runtime ZMQ availability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def rot_err_rad(q_pred_wxyz: np.ndarray, q_gt_wxyz: np.ndarray) -> float:
    r_pred = R.from_quat(quat_wxyz_to_xyzw(q_pred_wxyz))
    r_gt = R.from_quat(quat_wxyz_to_xyzw(q_gt_wxyz))
    rel = r_pred * r_gt.inv()
    return float(rel.magnitude())


def summary_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(np.square(x)))),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline wrist EEF accuracy from parquet")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"),
    )
    parser.add_argument("--left-frame", type=str, default="left_wrist_yaw_link")
    parser.add_argument("--right-frame", type=str, default="right_wrist_yaw_link")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)
    if not args.urdf.exists():
        raise FileNotFoundError(args.urdf)

    df = pd.read_parquet(args.parquet)
    for c in ["observation.state", "observation.eef_state"]:
        if c not in df.columns:
            raise KeyError(f"missing column: {c}")

    q_all = np.stack(df["observation.state"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    eef = np.stack(df["observation.eef_state"].apply(np.asarray).to_numpy(), axis=0).astype(np.float64)
    if q_all.shape[1] != 43:
        raise ValueError(f"observation.state must be 43D, got {q_all.shape[1]}")
    if eef.shape[1] != 14:
        raise ValueError(f"observation.eef_state must be 14D, got {eef.shape[1]}")

    model = pin.buildModelFromUrdf(str(args.urdf))
    data = model.createData()
    lf = model.getFrameId(args.left_frame)
    rf = model.getFrameId(args.right_frame)

    left_pos_err = []
    right_pos_err = []
    left_rot_err = []
    right_rot_err = []

    for i in range(len(df)):
        q = q_all[i]
        pin.framesForwardKinematics(model, data, q)

        l_pose = data.oMf[lf]
        r_pose = data.oMf[rf]

        l_pos = np.asarray(l_pose.translation)
        r_pos = np.asarray(r_pose.translation)
        l_quat_xyzw = R.from_matrix(np.asarray(l_pose.rotation)).as_quat()
        r_quat_xyzw = R.from_matrix(np.asarray(r_pose.rotation)).as_quat()
        l_quat_wxyz = np.array([l_quat_xyzw[3], l_quat_xyzw[0], l_quat_xyzw[1], l_quat_xyzw[2]])
        r_quat_wxyz = np.array([r_quat_xyzw[3], r_quat_xyzw[0], r_quat_xyzw[1], r_quat_xyzw[2]])

        l_gt_pos = eef[i, 0:3]
        l_gt_quat = eef[i, 3:7]
        r_gt_pos = eef[i, 7:10]
        r_gt_quat = eef[i, 10:14]

        left_pos_err.append(float(np.linalg.norm(l_pos - l_gt_pos)))
        right_pos_err.append(float(np.linalg.norm(r_pos - r_gt_pos)))
        left_rot_err.append(rot_err_rad(l_quat_wxyz, l_gt_quat))
        right_rot_err.append(rot_err_rad(r_quat_wxyz, r_gt_quat))

    left_pos_err = np.asarray(left_pos_err)
    right_pos_err = np.asarray(right_pos_err)
    left_rot_err = np.asarray(left_rot_err)
    right_rot_err = np.asarray(right_rot_err)

    result = {
        "parquet": str(args.parquet),
        "num_frames": int(len(df)),
        "left_pos_err_m": summary_stats(left_pos_err),
        "right_pos_err_m": summary_stats(right_pos_err),
        "left_rot_err_rad": summary_stats(left_rot_err),
        "right_rot_err_rad": summary_stats(right_rot_err),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
