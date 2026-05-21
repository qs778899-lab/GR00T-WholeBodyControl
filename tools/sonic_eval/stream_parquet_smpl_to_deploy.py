#!/usr/bin/env python3
"""Stream a SONIC data-collection parquet episode to gear_sonic_deploy via ZMQ Protocol v3.

This is the deploy-only path for replaying parquet SMPL data through the
human-motion-encoder (mode_id=2). It mirrors what pico_manager_thread_server.py
sends to deploy at real teleop time, so NO SMPL->G1 retargeting and NO pkl
intermediate is needed: the 6 G1 wrist DOFs are already retargeted inside the
parquet's ``teleop.{left,right}_wrist_joints`` fields by the data collector.

See sim2sim_human_encoder_plan.md sections 0 and 5.1.deploy-only.

Zero-intrusion: imports helpers from stream_motionlib_smpl_to_deploy.py and
stream_motionlib_to_deploy.py without modifying them. No motion_lib / fk_batch
is loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.sonic_eval.stream_motionlib_to_deploy import (
    _finite_difference,
    _prepend_stand_transition,
)
from tools.sonic_eval.stream_motionlib_smpl_to_deploy import (
    PackedPublisherSMPL,
    _canonicalize_smpl_joints,
    _remove_smpl_base_rot_wxyz,
    _smpl_root_ytoz_up,
)


# G1 IsaacLab actuated DOF indices the SMPL encoder reads from joint_pos[29].
G1_L_WRIST_ROLL_IDX = 23
G1_L_WRIST_PITCH_IDX = 25
G1_L_WRIST_YAW_IDX = 27
G1_R_WRIST_ROLL_IDX = 24
G1_R_WRIST_PITCH_IDX = 26
G1_R_WRIST_YAW_IDX = 28
G1_NUM_ACTUATED = 29

SMPL_NUM_JOINTS = 24
SMPL_BODY_JOINTS_IN_PARQUET = 21  # teleop.smpl_pose stores joints 1..21 (no root, no hands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True,
                        help="path to a single-episode parquet file produced by run_data_exporter.py")
    parser.add_argument("--target-fps", type=int, default=50,
                        help="parquet is recorded at 50Hz by default; only used for joint_vel finite-diff")
    parser.add_argument("--num-future-frames", type=int, default=10,
                        help="must cover deploy's future observation window for prestart prebuffering")

    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--port", type=int, default=5596)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--initial-burst-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--smpl-stream-mode-filter",
        type=str,
        default="auto",
        choices=["auto", "off"],
        help=(
            "Filter out rows where teleop.stream_mode != 1 (SMPL mode). "
            "'auto' (default): trim to the longest contiguous SMPL-mode segment, abort if none exists. "
            "Recommended because parquets collected in PLANNER mode (stream_mode=0) have all SMPL "
            "fields stuck at identity / zero, which makes the policy see constant input across "
            "episodes (symptom: robot does the same forward-step-then-stagger-back behavior). "
            "'off': skip filtering; send ALL parquet rows verbatim (diagnostic only)."
        ),
    )
    parser.add_argument("--prepend-stand-frames", type=int, default=0)
    parser.add_argument("--blend-from-stand-frames", type=int, default=0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--catch-up", action="store_true")
    parser.add_argument("--send-command", action="store_true")
    parser.add_argument("--command-repeat", type=int, default=3)
    parser.add_argument("--command-interval", type=float, default=0.05)
    parser.add_argument("--command-heartbeat-interval", type=float, default=0.5)
    parser.add_argument("--startup-delay", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--frame-index-source",
        type=str,
        choices=["row", "smpl"],
        default="row",
        help=(
            "'row' (default): use streamer's contiguous row index as frame_index. "
            "'smpl': use parquet's teleop.smpl_frame_index (may have gaps / stale frames)."
        ),
    )

    parser.add_argument(
        "--smpl-y-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "treat parquet teleop.body_quat_w / teleop.smpl_joints as Y-up "
            "(matches sonic_release.yaml `smpl_y_up: true`). Only affects smpl_processed anchor."
        ),
    )
    parser.add_argument(
        "--smpl-anchor-mode",
        type=str,
        choices=["parquet_body_quat", "smpl_processed"],
        default="parquet_body_quat",
        help=(
            "which root quaternion to use as the 'reference orientation' for SMPL encoder mode. "
            "'parquet_body_quat' (default, recommended): use teleop.body_quat_w directly. "
            "This mirrors what pico_manager_thread_server.py sends to deploy at real teleop time, "
            "so the encoder sees the exact same input distribution as production teleop. "
            "'smpl_processed' (diagnostic): apply Y->Z + remove_smpl_base_rot to teleop.body_quat_w. "
            "Matches IsaacSim training's TrackingCommand.smpl_root_quat_w semantics."
        ),
    )
    parser.add_argument(
        "--smpl-joints-mode",
        type=str,
        choices=["passthrough", "re_canonicalize"],
        default="passthrough",
        help=(
            "'passthrough' (default, recommended): send teleop.smpl_joints as-is. "
            "Pico sender (pico_manager_thread_server.py:476-477) already applies "
            "quat_apply(quat_inv(processed_root), FK_output) before writing to ZMQ/parquet, "
            "so the parquet smpl_joints already match the training encoder input distribution "
            "(R^-1 * FK_output where R is the processed root). Empirically verified: "
            "parquet pelvis varies, pkl pelvis stays fixed at J[0] -- this confirms parquet "
            "joints have R^-1 already applied while pkl joints are raw FK output. "
            "'re_canonicalize' (diagnostic only): apply quat_apply(quat_inv(R), parquet_joints) "
            "AGAIN. This is the PRE-FIX bug behaviour that caused feet/hands distortion -- "
            "encoder sees R^-2 * FK_output, way off training distribution. Only kept for A/B."
        ),
    )
    return parser.parse_args()


def _stack_column(df: pd.DataFrame, col: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Stack a (T,) Series of numpy arrays into a single ndarray; assert per-row shape."""
    if col not in df.columns:
        raise KeyError(f"parquet missing required column: {col!r}")
    arr = np.stack([np.asarray(v) for v in df[col].to_numpy()], axis=0)
    if arr.shape[1:] != expected_shape:
        raise ValueError(f"column {col!r} per-row shape {arr.shape[1:]} != expected {expected_shape}")
    return arr.astype(np.float32)


def _filter_and_fill_smpl_data(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Filter parquet to the longest segment with valid SMPL data, hold-last filling sparse glitches.

    Two invalid signals:
      (a) stream_mode != 1: episode collected in PLANNER mode; SMPL fields are placeholder zeros.
          Cannot be recovered; trimmed out entirely.
      (b) stream_mode == 1 but body_quat_w == identity (1,0,0,0): pico dropped a VR frame and
          wrote a placeholder. Sparse and recoverable -- forward-fill from previous valid frame
          (mirrors how live pico->deploy behaves when VR data hiccups for a few frames).

    Strategy:
      1. Find longest contiguous run of stream_mode == 1
      2. Within that run, forward-fill identity-body_quat rows from prior valid neighbors
      3. If the run starts with identity rows, back-fill from the first valid frame
      4. Abort if no row has stream_mode == 1
    """
    if "teleop.stream_mode" not in df.columns:
        print(f"[warn] parquet has no 'teleop.stream_mode' column; skipping filter for {path.name}")
        return df

    sm = np.asarray(df["teleop.stream_mode"].to_numpy()).astype(np.int64).reshape(-1)
    bq = np.stack([np.asarray(v) for v in df["teleop.body_quat_w"].to_numpy()]).astype(np.float32)
    is_identity = (np.abs(bq[:, 0] - 1.0) < 1e-3) & (np.linalg.norm(bq[:, 1:], axis=1) < 1e-3)
    total = len(sm)

    in_smpl_mode = sm == 1
    if int(in_smpl_mode.sum()) == 0:
        sm_unique = np.unique(sm).tolist()
        raise ValueError(
            f"parquet {path.name} has 0/{total} frames with stream_mode == 1. "
            f"Found stream_mode values {sm_unique}. This parquet was collected in PLANNER mode "
            f"and its SMPL fields are placeholder zeros. Cannot drive SMPL encoder from it. "
            f"Either pick an SMPL-mode episode or pass --smpl-stream-mode-filter off "
            f"(diagnostic only; policy will see constant input)."
        )

    # Step 1: longest contiguous run of stream_mode == 1
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i in range(total):
        if in_smpl_mode[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    seg_end = best_start + best_len
    seg_id_count = int(is_identity[best_start:seg_end].sum())
    seg_valid_count = best_len - seg_id_count

    if seg_valid_count == 0:
        raise ValueError(
            f"parquet {path.name}: longest stream_mode==1 segment [{best_start},{seg_end}) "
            f"contains {best_len} rows but ALL are identity body_quat. Effectively no SMPL data."
        )

    sm_dropped = total - best_len  # rows dropped due to stream_mode != 1
    if sm_dropped == 0 and seg_id_count == 0:
        print(f"[smpl-mode-filter] {path.name}: all {total} rows valid")
        return df

    # Step 2: forward-fill identity rows in the segment from prior valid neighbors
    sub = df.iloc[best_start:seg_end].reset_index(drop=True).copy()
    sub_id = is_identity[best_start:seg_end]

    # Find the FIRST valid row in the segment for back-fill if needed
    first_valid = int(np.where(~sub_id)[0][0])

    fill_columns = [c for c in sub.columns if c.startswith("teleop.")]
    for i in range(len(sub)):
        if sub_id[i]:
            donor = i - 1
            while donor >= 0 and sub_id[donor]:
                donor -= 1
            if donor < 0:
                donor = first_valid  # back-fill from first valid
            for c in fill_columns:
                sub.at[i, c] = sub.at[donor, c]

    print(
        f"[smpl-mode-filter] {path.name}: trimmed {total}->{best_len} rows "
        f"(kept [{best_start},{seg_end})); "
        f"dropped {sm_dropped} non-SMPL-mode rows; "
        f"hold-last filled {seg_id_count} identity-body_quat glitches in segment"
    )
    return sub


def _load_parquet_episode(path: Path, smpl_stream_mode_filter: str = "auto") -> dict:
    """Load a single parquet episode and produce the numpy arrays the streamer needs."""
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise ValueError(f"empty parquet: {path}")

    if smpl_stream_mode_filter == "auto":
        df = _filter_and_fill_smpl_data(df, path)

    smpl_joints = _stack_column(df, "teleop.smpl_joints", (72,)).reshape(-1, SMPL_NUM_JOINTS, 3)
    smpl_pose_21 = _stack_column(df, "teleop.smpl_pose", (63,)).reshape(
        -1, SMPL_BODY_JOINTS_IN_PARQUET, 3
    )
    body_quat_w = _stack_column(df, "teleop.body_quat_w", (4,))
    left_wrist = _stack_column(df, "teleop.left_wrist_joints", (3,))
    right_wrist = _stack_column(df, "teleop.right_wrist_joints", (3,))

    smpl_frame_idx = None
    if "teleop.smpl_frame_index" in df.columns:
        smpl_frame_idx = np.asarray(
            [int(np.asarray(v).flatten()[0]) for v in df["teleop.smpl_frame_index"].to_numpy()],
            dtype=np.int64,
        )

    # Normalize body_quat_w (defensive — pico applies lerp which can drift slightly).
    norms = np.linalg.norm(body_quat_w, axis=-1, keepdims=True).clip(min=1e-8)
    body_quat_w = (body_quat_w / norms).astype(np.float32)

    return {
        "smpl_joints": smpl_joints,
        "smpl_pose_21": smpl_pose_21,
        "body_quat_w": body_quat_w,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "smpl_frame_idx": smpl_frame_idx,
    }


def _build_joint_pos(left_wrist: np.ndarray, right_wrist: np.ndarray) -> np.ndarray:
    """Build joint_pos[T, 29]: zeros everywhere except the 6 G1 wrist DOFs filled from parquet.

    The pico data collector (run_data_exporter.py:410-447) extracted these wrist DOFs from
    joint_pos[23,25,27] / joint_pos[24,26,28] that pico_manager_thread_server.py:1346-1403
    had already retargeted analytically from SMPL. We invert that extraction here.
    """
    t = left_wrist.shape[0]
    joint_pos = np.zeros((t, G1_NUM_ACTUATED), dtype=np.float32)
    joint_pos[:, G1_L_WRIST_ROLL_IDX] = left_wrist[:, 0]
    joint_pos[:, G1_L_WRIST_PITCH_IDX] = left_wrist[:, 1]
    joint_pos[:, G1_L_WRIST_YAW_IDX] = left_wrist[:, 2]
    joint_pos[:, G1_R_WRIST_ROLL_IDX] = right_wrist[:, 0]
    joint_pos[:, G1_R_WRIST_PITCH_IDX] = right_wrist[:, 1]
    joint_pos[:, G1_R_WRIST_YAW_IDX] = right_wrist[:, 2]
    return joint_pos


def _build_smpl_pose_24(
    smpl_pose_21: np.ndarray, body_quat_w_wxyz: np.ndarray
) -> np.ndarray:
    """Construct smpl_pose[T, 24, 3] for the ZMQ wire format.

    Slot layout:
      - index 0 (root/pelvis): axis-angle derived from body_quat_w
      - indices 1..21 (body):  parquet teleop.smpl_pose (which stores joints 1..21)
      - indices 22..23 (hands): zeros (parquet has no hand axis-angle; deploy doesn't read these)
    """
    t = smpl_pose_21.shape[0]
    out = np.zeros((t, SMPL_NUM_JOINTS, 3), dtype=np.float32)
    # quat -> axis-angle (wxyz convention; w in [-1,1])
    w = np.clip(body_quat_w_wxyz[:, 0], -1.0, 1.0)
    xyz = body_quat_w_wxyz[:, 1:4]
    angle = 2.0 * np.arccos(w)  # [T]
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))  # = sin(angle/2)
    safe = sin_half > 1e-8
    axis = np.zeros_like(xyz)
    axis[safe] = xyz[safe] / sin_half[safe, None]
    out[:, 0, :] = (axis * angle[:, None]).astype(np.float32)
    out[:, 1 : 1 + SMPL_BODY_JOINTS_IN_PARQUET, :] = smpl_pose_21
    return out


def _compute_reference_root_quat(
    body_quat_w_wxyz: np.ndarray, mode: str, smpl_y_up: bool
) -> np.ndarray:
    """Select the reference root quaternion based on --smpl-anchor-mode."""
    if mode == "parquet_body_quat":
        return body_quat_w_wxyz.astype(np.float32)
    if mode == "smpl_processed":
        q = body_quat_w_wxyz
        if smpl_y_up:
            q = _smpl_root_ytoz_up(q)
        q = _remove_smpl_base_rot_wxyz(q)
        norms = np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
        return (q / norms).astype(np.float32)
    raise ValueError(f"Unknown --smpl-anchor-mode: {mode}")


def main() -> None:
    args = parse_args()

    data = _load_parquet_episode(args.parquet, smpl_stream_mode_filter=args.smpl_stream_mode_filter)
    n = data["smpl_joints"].shape[0]

    # Build joint_pos from parquet's pre-retargeted G1 wrist DOFs.
    joint_pos = _build_joint_pos(data["left_wrist"], data["right_wrist"])
    joint_vel = _finite_difference(joint_pos, args.target_fps)

    # Build smpl_pose[T,24,3] for the wire format.
    smpl_pose_24 = _build_smpl_pose_24(data["smpl_pose_21"], data["body_quat_w"])

    # Reference root quaternion (drives both anchor obs and joints canonicalization).
    reference_root_quat = _compute_reference_root_quat(
        data["body_quat_w"], args.smpl_anchor_mode, args.smpl_y_up
    )

    # smpl_joints handling: pico sender (pico_manager_thread_server.py:476-477) already
    # applies quat_apply(quat_inv(processed_root), FK_output) before writing to parquet,
    # so the values already match training encoder distribution (R^-1 * FK_output).
    # Default 'passthrough' sends as-is. 're_canonicalize' applies the inverse rotation
    # AGAIN (yielding R^-2 * FK_output) -- this was the pre-fix bug that caused distortion.
    if args.smpl_joints_mode == "re_canonicalize":
        smpl_joints = _canonicalize_smpl_joints(data["smpl_joints"], reference_root_quat)
    else:
        smpl_joints = data["smpl_joints"]

    # Use SMPL pelvis position as body_pos_w (deploy SMPL anchor obs doesn't read this,
    # but ZMQ Protocol v3 requires the field to be present and shape-consistent).
    body_pos_w = data["smpl_joints"][:, 0, :].astype(np.float32)

    # Slice [start_frame, end_frame).
    end = n if args.end_frame is None else min(args.end_frame, n)
    start = max(0, args.start_frame)
    if start >= end:
        raise ValueError(f"empty frame range: start={start}, end={end}")
    joint_pos = joint_pos[start:end]
    joint_vel = joint_vel[start:end]
    smpl_pose_24 = smpl_pose_24[start:end]
    smpl_joints = smpl_joints[start:end]
    body_pos_w = body_pos_w[start:end]
    body_quat_for_stream = reference_root_quat[start:end]
    smpl_frame_idx_sliced = (
        data["smpl_frame_idx"][start:end] if data["smpl_frame_idx"] is not None else None
    )

    # Prepend stand transition on joint side; pad SMPL side with first-frame repeats so
    # frame counts stay aligned (Protocol v3 requires equal frame counts across all fields).
    joint_pos_full, joint_vel_full, body_quat_full = _prepend_stand_transition(
        dof_pos=joint_pos,
        dof_vel=joint_vel,
        root_quat=body_quat_for_stream,
        target_fps=args.target_fps,
        stand_frames=max(0, args.prepend_stand_frames),
        blend_frames=max(0, args.blend_from_stand_frames),
    )
    num_prefix = len(joint_pos_full) - len(joint_pos)
    if num_prefix > 0:
        body_pos_w_full = np.concatenate(
            [np.repeat(body_pos_w[:1], num_prefix, axis=0), body_pos_w], axis=0
        ).astype(np.float32)
        smpl_pose_full = np.concatenate(
            [np.repeat(smpl_pose_24[:1], num_prefix, axis=0), smpl_pose_24], axis=0
        ).astype(np.float32)
        smpl_joints_full = np.concatenate(
            [np.repeat(smpl_joints[:1], num_prefix, axis=0), smpl_joints], axis=0
        ).astype(np.float32)
    else:
        body_pos_w_full = body_pos_w
        smpl_pose_full = smpl_pose_24
        smpl_joints_full = smpl_joints

    sent_frames = len(joint_pos_full)
    motion_start_frame = max(0, args.prepend_stand_frames) + max(0, args.blend_from_stand_frames)

    pub = PackedPublisherSMPL(
        args.host, args.port, verbose=args.verbose, motion_start_frame=motion_start_frame
    )
    print(
        f"[parquet-smpl-stream] parquet={args.parquet.name} frames={sent_frames} "
        f"(motion={len(joint_pos)} + prefix={num_prefix}) "
        f"anchor_mode={args.smpl_anchor_mode} joints_mode={args.smpl_joints_mode} "
        f"smpl_y_up={args.smpl_y_up} frame_index_source={args.frame_index_source} "
        f"endpoint={pub.endpoint}"
    )

    def _frame_indices(i: int, j: int) -> np.ndarray:
        if args.frame_index_source == "smpl" and smpl_frame_idx_sliced is not None:
            # Map streamer row index back to parquet's smpl_frame_index; prefix frames
            # reuse the first motion frame's smpl_frame_index.
            sliced = np.empty(j - i, dtype=np.int64)
            for k, row in enumerate(range(i, j)):
                motion_row = max(0, row - num_prefix)
                sliced[k] = int(smpl_frame_idx_sliced[min(motion_row, len(smpl_frame_idx_sliced) - 1)])
            return sliced
        return np.arange(i, j, dtype=np.int64)

    def _send_range(istart: int, iend: int) -> None:
        for i in range(istart, iend, args.chunk_size):
            j = min(i + args.chunk_size, iend)
            pub.send_pose(
                joint_pos=joint_pos_full[i:j],
                joint_vel=joint_vel_full[i:j],
                smpl_joints=smpl_joints_full[i:j],
                smpl_pose=smpl_pose_full[i:j],
                body_pos_w=body_pos_w_full[i:j],
                body_quat_w=body_quat_full[i:j],
                frame_indices=_frame_indices(i, j),
                catch_up=args.catch_up,
            )

    try:
        time.sleep(args.startup_delay)
        frame_period = 1.0 / float(args.target_fps)
        prestart_frames = min(
            sent_frames,
            max(0, args.initial_burst_frames, args.chunk_size, args.num_future_frames),
        )
        if prestart_frames > 0:
            _send_range(0, prestart_frames)
            if args.verbose:
                print(f"[startup] prebuffered 0..{prestart_frames - 1} before start command")

        last_command_heartbeat = time.monotonic()
        if args.send_command:
            for _ in range(max(1, args.command_repeat)):
                pub.send_command(start=True, stop=False, planner=False)
                time.sleep(max(0.0, args.command_interval))
            time.sleep(0.1)
            last_command_heartbeat = time.monotonic()

        burst_end = min(sent_frames, max(prestart_frames, args.initial_burst_frames))
        for i in range(burst_end, sent_frames, args.chunk_size):
            j = min(i + args.chunk_size, sent_frames)
            if args.send_command and args.command_heartbeat_interval > 0.0:
                now = time.monotonic()
                if now - last_command_heartbeat >= args.command_heartbeat_interval:
                    pub.send_command(start=True, stop=False, planner=False)
                    last_command_heartbeat = now
            _send_range(i, j)
            if args.realtime:
                time.sleep((j - i) * frame_period)
        print("[OK] parquet smpl stream complete")
    finally:
        pub.close()


if __name__ == "__main__":
    main()
