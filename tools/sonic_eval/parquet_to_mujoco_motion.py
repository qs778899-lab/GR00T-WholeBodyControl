#!/usr/bin/env python3
"""
Convert one official exported Sonic parquet episode to gear_sonic_deploy motion CSV format.

Input:
- parquet episode (e.g. /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet)
- optional dataset meta/info.json for authoritative joint names and modality slices

Output directory layout (single motion folder):
- <output_root>/<motion_name>/joint_pos.csv
- <output_root>/<motion_name>/joint_vel.csv
- <output_root>/<motion_name>/body_pos.csv
- <output_root>/<motion_name>/body_quat.csv
- <output_root>/<motion_name>/body_lin_vel.csv
- <output_root>/<motion_name>/body_ang_vel.csv
- <output_root>/<motion_name>/metadata.txt
- <output_root>/<motion_name>/info.txt

Notes:
- No modification to project core files.
- Uses official meta/info.json joint names when available.
- Supports taking encoder motion joints from either:
  - `observation.state` (default-compatible behavior), or
  - `action.wbc` (robot control from collected data; requested sim2sim analysis path).
- Body part indexes defaults to [0] (root only), which is sufficient for encoder_mode_4 +
  motion_joint_positions_10frame_step5 + motion_joint_velocities_10frame_step5 +
  motion_anchor_orientation_10frame_step5 in g1 mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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


def _ensure_2d_array(series: pd.Series, name: str) -> np.ndarray:
    arr = np.stack(series.apply(np.asarray).to_numpy(), axis=0)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after stacking, got shape={arr.shape}")
    return arr


def _find_meta_info(parquet_path: Path, explicit_meta_info: Path | None) -> Path | None:
    if explicit_meta_info is not None:
        if not explicit_meta_info.exists():
            raise FileNotFoundError(f"meta info.json not found: {explicit_meta_info}")
        return explicit_meta_info

    # Try dataset-root style: <dataset_root>/data/chunk-xxx/episode_xxxxxx.parquet -> <dataset_root>/meta/info.json
    try:
        dataset_root = parquet_path.parents[2]
        candidate = dataset_root / "meta" / "info.json"
        if candidate.exists():
            return candidate
    except IndexError:
        return None
    return None


def _load_joint_names_from_info(info_json_path: Path) -> list[str] | None:
    with info_json_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    names = (
        info.get("features", {})
        .get("observation.state", {})
        .get("names", None)
    )
    if isinstance(names, list) and names:
        return names
    return None


def _save_csv(path: Path, data: np.ndarray, headers: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data, columns=list(headers))
    df.to_csv(path, index=False, float_format="%.9f")


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    out[0] = (x[1] - x[0]) / dt
    out[-1] = (x[-1] - x[-2]) / dt
    return out


def convert(
    parquet_path: Path,
    output_root: Path,
    motion_name: str,
    meta_info_json: Path | None = None,
    dt_override: float | None = None,
    joint_source: str = "observation.state",
    joint_vel_source: str | None = None,
) -> Path:
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    required_cols = ["action.wbc", "observation.state", "observation.root_orientation"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"missing required parquet column: {c}")

    info_path = _find_meta_info(parquet_path, meta_info_json)
    state_joint_names = None
    if info_path is not None:
        state_joint_names = _load_joint_names_from_info(info_path)

    action_wbc = _ensure_2d_array(df["action.wbc"], "action.wbc")
    observation_state = _ensure_2d_array(df["observation.state"], "observation.state")

    if action_wbc.shape[1] != observation_state.shape[1]:
        raise ValueError(
            "action.wbc and observation.state width mismatch: "
            f"{action_wbc.shape[1]} vs {observation_state.shape[1]}"
        )

    if state_joint_names is None:
        # Fallback assumption based on official export format used in this repo:
        # observation.state/action.wbc order = [29 body + 14 hands]
        if observation_state.shape[1] < 29:
            raise ValueError(
                "Cannot infer 29 body joints from observation.state without meta/info.json"
            )
        body_indices = list(range(29))
    else:
        name_to_idx = {n: i for i, n in enumerate(state_joint_names)}
        missing = [n for n in MUJOCO_29_NAMES if n not in name_to_idx]
        if missing:
            raise ValueError(f"Missing body joint names in meta info: {missing}")
        body_indices = [name_to_idx[n] for n in MUJOCO_29_NAMES]

    # IMPORTANT: deploy motion reader expects joint_pos in ISAACLAB order internally.
    # The official parquet body slice is in MuJoCo order.
    # This remap matches policy_parameters.hpp mujoco_to_isaaclab.
    mujoco_to_isaaclab = [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
    ]

    if joint_source == "observation.state":
        source_43 = observation_state
    elif joint_source == "action.wbc":
        source_43 = action_wbc
    else:
        raise ValueError(f"unsupported joint_source: {joint_source}")

    body_pos_mujoco = source_43[:, body_indices]
    joint_pos = body_pos_mujoco[:, mujoco_to_isaaclab]

    # Time step for finite-difference velocity (also used by body velocity synthesis below).
    if dt_override is not None:
        dt = float(dt_override)
    elif "timestamp" in df.columns and len(df) >= 2:
        t = np.asarray(df["timestamp"], dtype=np.float64)
        dts = np.diff(t)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        dt = float(np.median(dts)) if len(dts) else 0.02
    else:
        dt = 0.02

    # Velocity source policy:
    # - If user explicitly set --joint-vel-source, follow it.
    # - Otherwise:
    #   - action.wbc source -> finite_diff (self-consistent control-domain motion)
    #   - observation.state source -> observation.body_dq (dataset measured velocity)
    if joint_vel_source is None:
        joint_vel_source = "finite_diff" if joint_source == "action.wbc" else "observation.body_dq"

    if joint_vel_source == "observation.body_dq":
        if "observation.body_dq" not in df.columns:
            raise KeyError("missing required parquet column for velocity source: observation.body_dq")
        body_dq = _ensure_2d_array(df["observation.body_dq"], "observation.body_dq")
        if body_dq.shape[1] != 29:
            raise ValueError(f"observation.body_dq must have 29 dims, got {body_dq.shape[1]}")
        joint_vel = body_dq[:, mujoco_to_isaaclab]
    elif joint_vel_source == "finite_diff":
        joint_vel = _finite_diff(joint_pos, dt)
    else:
        raise ValueError(f"unsupported joint_vel_source: {joint_vel_source}")

    root_quat = _ensure_2d_array(df["observation.root_orientation"], "observation.root_orientation")
    if root_quat.shape[1] != 4:
        raise ValueError(f"observation.root_orientation must have 4 dims, got {root_quat.shape[1]}")

    # Root position is not directly in this export; set to zeros for minimal g1 encoder-required path.
    n = len(df)
    body_pos = np.zeros((n, 3), dtype=np.float64)
    body_quat = root_quat.astype(np.float64)

    body_lin_vel = _finite_diff(body_pos, dt)

    # Angular velocity from quaternion finite difference (small-angle approx around identity on dq)
    # Since only root quat is provided and this is a minimal smoke path, we keep zero if not needed.
    body_ang_vel = np.zeros((n, 3), dtype=np.float64)

    output_root = output_root.expanduser().resolve()
    motion_dir = output_root / motion_name
    motion_dir.mkdir(parents=True, exist_ok=True)

    _save_csv(motion_dir / "joint_pos.csv", joint_pos, [f"joint_{i}" for i in range(29)])
    _save_csv(motion_dir / "joint_vel.csv", joint_vel, [f"joint_vel_{i}" for i in range(29)])
    _save_csv(motion_dir / "body_pos.csv", body_pos, ["body_0_x", "body_0_y", "body_0_z"])
    _save_csv(motion_dir / "body_quat.csv", body_quat, ["body_0_w", "body_0_x", "body_0_y", "body_0_z"])
    _save_csv(
        motion_dir / "body_lin_vel.csv",
        body_lin_vel,
        ["body_0_vel_x", "body_0_vel_y", "body_0_vel_z"],
    )
    _save_csv(
        motion_dir / "body_ang_vel.csv",
        body_ang_vel,
        ["body_0_angvel_x", "body_0_angvel_y", "body_0_angvel_z"],
    )

    metadata_txt = motion_dir / "metadata.txt"
    metadata_txt.write_text(
        "\n".join(
            [
                f"Metadata for: {motion_name}",
                "=" * 30,
                "",
                "Body part indexes:",
                "[0]",
                "",
                f"Total timesteps: {n}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    info = {
        "source_parquet": str(parquet_path),
        "resolved_meta_info_json": str(info_path) if info_path else None,
        "num_frames": int(n),
        "dt": dt,
        "input_dims": {
            "action.wbc": int(action_wbc.shape[1]),
            "observation.state": int(observation_state.shape[1]),
            "observation.body_dq": int(
                _ensure_2d_array(df["observation.body_dq"], "observation.body_dq").shape[1]
            )
            if "observation.body_dq" in df.columns
            else None,
            "observation.root_orientation": int(root_quat.shape[1]),
        },
        "joint_source": joint_source,
        "joint_vel_source": joint_vel_source,
        "body_joint_names_mujoco_29": MUJOCO_29_NAMES,
        "selected_body_indices_from_state": body_indices,
        "joint_order_written_to_motion_csv": "isaaclab_order_29",
        "mapping_used": {
            "mujoco_to_isaaclab": mujoco_to_isaaclab,
        },
        "notes": [
            "joint_pos/joint_vel are written in IsaacLab order to match deploy internal expectations.",
            "body_pos is zeroed (root-only minimal mode).",
            "body_ang_vel is zeroed for smoke test.",
        ],
    }
    (motion_dir / "info.txt").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return motion_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Sonic parquet to deploy motion csv folder")
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to one parquet episode file",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/sonic_motions_from_parquet"),
        help="Root output directory containing generated motion folder",
    )
    parser.add_argument(
        "--motion-name",
        type=str,
        default="episode_from_parquet",
        help="Generated motion folder name",
    )
    parser.add_argument(
        "--meta-info-json",
        type=Path,
        default=None,
        help="Optional explicit path to dataset meta/info.json",
    )
    parser.add_argument(
        "--joint-source",
        type=str,
        choices=["observation.state", "action.wbc"],
        default="observation.state",
        help="Source column used to build motion joint positions for encoder input",
    )
    parser.add_argument(
        "--joint-vel-source",
        type=str,
        choices=["observation.body_dq", "finite_diff"],
        default=None,
        help="Joint velocity source; default auto-selects by --joint-source",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional fixed dt override (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_dir = convert(
        parquet_path=args.parquet,
        output_root=args.output_root,
        motion_name=args.motion_name,
        meta_info_json=args.meta_info_json,
        dt_override=args.dt,
        joint_source=args.joint_source,
        joint_vel_source=args.joint_vel_source,
    )
    print(f"[OK] Motion folder generated: {motion_dir}")


if __name__ == "__main__":
    main()
