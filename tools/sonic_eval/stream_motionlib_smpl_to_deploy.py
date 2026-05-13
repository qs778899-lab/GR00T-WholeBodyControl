#!/usr/bin/env python3
"""Stream SONIC motion_lib robot pkl + SMPL pkl pair to gear_sonic_deploy via ZMQ Protocol v3.

This is the SMPL-encoder (mode_id=2) variant of stream_motionlib_to_deploy.py.
It loads the official paired data:
  - robot pkl: sample_data/robot_filtered/.../*.pkl  (G1 fitted pose_aa + dof)
  - smpl  pkl: sample_data/smpl_filtered/*.pkl       (raw SMPL pose_aa + smpl_joints + transl)
and sends a Protocol v3 ZMQ packet so deploy switches to encoder_mode=2 (smpl).

Wire format (v3) per zmq_endpoint_interface.hpp:919-996:
  required: smpl_joints, smpl_pose, joint_pos, joint_vel, body_pos_w, body_quat_w, frame_index
  optional: catch_up

Robot-side fields are still required because deploy's SMPL encoder consumes
motion_joint_positions_wrists_10frame_step1 (G1 wrist DOFs at indices {23..28})
even when running in SMPL mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import time

import joblib
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.sonic_eval.motionlib_provider import load_motionlib_sequence
from tools.sonic_eval.stream_motionlib_to_deploy import (
    HEADER_SIZE,
    _finite_difference,
    _prepend_stand_transition,
)


SMPL_NUM_JOINTS = 24
SMPL_POSE_DIMS = 24  # send the full SMPL pose_aa [T, 24, 3]; deploy supports any num_smpl_poses


class PackedPublisherSMPL:
    """ZMQ Protocol v3 publisher.

    Mirrors PackedPublisher from stream_motionlib_to_deploy.py but uses
    header version=3 and adds smpl_joints / smpl_pose payload fields.
    """

    def __init__(self, host: str, port: int, verbose: bool = False, motion_start_frame: int = 0):
        import zmq

        self.verbose = verbose
        self.motion_start_frame = int(motion_start_frame)
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.endpoint = f"tcp://{host}:{port}"
        self.publisher.bind(self.endpoint)

    def close(self) -> None:
        self.publisher.close()
        self.context.term()

    def _send(self, topic: bytes, header: dict, buffers: list[bytes]) -> None:
        header_json = json.dumps(header).encode("utf-8")
        if len(header_json) > HEADER_SIZE:
            raise ValueError(f"packed header too large: {len(header_json)} > {HEADER_SIZE}")
        message = topic + header_json + b"\x00" * (HEADER_SIZE - len(header_json)) + b"".join(buffers)
        self.publisher.send(message)

    def send_command(self, start: bool, stop: bool, planner: bool) -> None:
        header = {
            "v": 1,
            "endian": "le",
            "count": 1,
            "fields": [
                {"name": "start", "dtype": "u8", "shape": [1]},
                {"name": "stop", "dtype": "u8", "shape": [1]},
                {"name": "planner", "dtype": "u8", "shape": [1]},
            ],
        }
        data = [
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        ]
        self._send(b"command", header, data)
        if self.verbose:
            print(f"[command] start={start} stop={stop} planner={planner}")

    def send_pose(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        smpl_joints: np.ndarray,
        smpl_pose: np.ndarray,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        frame_indices: np.ndarray,
        catch_up: bool,
    ) -> None:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        smpl_joints = np.asarray(smpl_joints, dtype=np.float32)
        smpl_pose = np.asarray(smpl_pose, dtype=np.float32)
        body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
        body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)

        n, num_joints = joint_pos.shape
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                f"joint_vel/joint_pos shape mismatch: {joint_vel.shape} vs {joint_pos.shape}"
            )
        if smpl_joints.shape != (n, SMPL_NUM_JOINTS, 3):
            raise ValueError(
                f"smpl_joints must be [N,{SMPL_NUM_JOINTS},3], got {smpl_joints.shape}"
            )
        if smpl_pose.shape != (n, SMPL_POSE_DIMS, 3):
            raise ValueError(
                f"smpl_pose must be [N,{SMPL_POSE_DIMS},3], got {smpl_pose.shape}"
            )
        if body_pos_w.ndim == 3 and body_pos_w.shape[1] == 1:
            body_pos_w = body_pos_w[:, 0, :]
        if body_pos_w.shape != (n, 3):
            raise ValueError(f"body_pos_w must be [N,3], got {body_pos_w.shape}")
        if body_quat_w.ndim == 3 and body_quat_w.shape[1] == 1:
            body_quat_w = body_quat_w[:, 0, :]
        if body_quat_w.shape != (n, 4):
            raise ValueError(f"body_quat_w must be [N,4], got {body_quat_w.shape}")

        header = {
            "v": 3,
            "endian": "le",
            "count": n,
            "motion_start_frame": self.motion_start_frame,
            "fields": [
                {"name": "joint_pos", "dtype": "f32", "shape": [n, num_joints]},
                {"name": "joint_vel", "dtype": "f32", "shape": [n, num_joints]},
                {"name": "smpl_joints", "dtype": "f32", "shape": [n, SMPL_NUM_JOINTS, 3]},
                {"name": "smpl_pose", "dtype": "f32", "shape": [n, SMPL_POSE_DIMS, 3]},
                {"name": "body_pos_w", "dtype": "f32", "shape": [n, 3]},
                {"name": "body_quat_w", "dtype": "f32", "shape": [n, 4]},
                {"name": "frame_index", "dtype": "i64", "shape": [n]},
                {"name": "catch_up", "dtype": "u8", "shape": [1]},
            ],
        }
        self._send(
            b"pose",
            header,
            [
                joint_pos.tobytes(),
                joint_vel.tobytes(),
                smpl_joints.tobytes(),
                smpl_pose.tobytes(),
                body_pos_w.tobytes(),
                body_quat_w.tobytes(),
                frame_indices.tobytes(),
                struct.pack("B", 1 if catch_up else 0),
            ],
        )
        if self.verbose:
            print(f"[pose v3] frames={frame_indices[0]}..{frame_indices[-1]} n={n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True,
                        help="G1 fitted robot motion pkl (robot_filtered/*.pkl format)")
    parser.add_argument("--smpl-motion-file", type=Path, required=True,
                        help="Raw SMPL motion pkl (smpl_filtered/*.pkl format)")
    parser.add_argument("--motion-name", type=str, default=None,
                        help="motion key inside robot pkl (default: only key if single)")
    parser.add_argument("--target-fps", type=int, default=50)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--dt-future-ref-frames", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-motionlib-robot", action="store_true")
    parser.add_argument(
        "--use-isaacsim-app",
        action="store_true",
        help="start IsaacSim SimulationApp before official TrackingCommand preprocessing",
    )
    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--port", type=int, default=5596)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument(
        "--initial-burst-frames",
        type=int,
        default=0,
        help=(
            "send this many frames immediately before realtime pacing; "
            "must cover deploy's future observation window"
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--prepend-stand-frames",
        type=int,
        default=0,
        help="prepend this many default standing-pose frames before the motion",
    )
    parser.add_argument(
        "--blend-from-stand-frames",
        type=int,
        default=0,
        help="prepend a linear joint-space blend from default standing pose to the first streamed frame",
    )
    parser.add_argument("--realtime", action="store_true", help="sleep according to --target-fps")
    parser.add_argument("--catch-up", action="store_true")
    parser.add_argument("--send-command", action="store_true",
                        help="send start=true planner=false before pose stream")
    parser.add_argument("--command-repeat", type=int, default=3)
    parser.add_argument("--command-interval", type=float, default=0.05)
    parser.add_argument("--command-heartbeat-interval", type=float, default=0.5)
    parser.add_argument("--startup-delay", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--smpl-y-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "treat the SMPL pkl as Y-up (matches sonic_release.yaml `smpl_y_up: true`). "
            "When set, applies the Y→Z up rotation to SMPL root before remove_smpl_base_rot. "
            "Pass --no-smpl-y-up only if the pkl was already in Z-up."
        ),
    )
    parser.add_argument(
        "--smpl-anchor-mode",
        type=str,
        choices=["robot_root", "smpl_processed", "smpl_raw"],
        default="robot_root",
        help=(
            "which root quaternion to use as the 'reference orientation' for SMPL encoder mode. "
            "'robot_root' (default, recommended): G1 motion's root quaternion from "
            "motionlib (== robot encoder pipeline's body_quat_w source). Produces ref viz that "
            "is visually aligned with actual G1, and re-uses base_sim.py's existing yaw+XY anchor "
            "mechanism with zero sim modifications. Encoder anchor_orientation obs is ~2-3° "
            "offset from training distribution due to SMPL T-pose vs G1 rest-pose convention "
            "difference (trained policy is robust to this small shift). "
            "'smpl_processed' (diagnostic / strict training match): per-frame SMPL root quat "
            "with Y→Z + remove_smpl_base_rot applied (== TrackingCommand.smpl_root_quat_w). "
            "Encoder obs exactly matches training distribution but ref viz has ~2-3° pitch/roll "
            "offset from actual G1 (uncorrectable by yaw-only sim anchor). "
            "'smpl_raw': raw SMPL pose_aa[:, 0, :] axis-angle → quat without Y→Z or base-rot "
            "removal. Wrong; only kept for ablation studies."
        ),
    )
    parser.add_argument(
        "--smpl-joints-mode",
        type=str,
        choices=["canonicalized", "raw"],
        default="canonicalized",
        help=(
            "how to fill ZMQ.smpl_joints (drives smpl_joints_10frame_step1 observation): "
            "'canonicalized' (recommended): apply quat_apply(quat_inv(R), pkl_joints) per frame, "
            "where R is the SAME root quat as --smpl-anchor-mode (enforced internal consistency "
            "to match training observation `smpl_joints_multi_future_local`). "
            "'raw': send pkl values directly (no canonicalization). PRE-FIX behaviour that caused "
            "feet-walking-sideways; only kept for ablation."
        ),
    )
    return parser.parse_args()


def _axis_angle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle [..., 3] to quaternion wxyz [..., 4]."""
    aa = np.asarray(aa, dtype=np.float32)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < 1e-8
    safe_angle = np.where(small, np.ones_like(angle), angle)
    half = 0.5 * angle
    cos_h = np.cos(half)
    sin_h = np.sin(half)
    axis = np.where(small, np.zeros_like(aa), aa / safe_angle)
    xyz = axis * sin_h
    return np.concatenate([cos_h, xyz], axis=-1).astype(np.float32)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions in wxyz format, broadcasting on leading dims."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate a wxyz quaternion (assumes unit norm)."""
    q = np.asarray(q, dtype=np.float32)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1).astype(np.float32)


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vectors v by wxyz quaternion q. q: [..., 4], v: [..., 3]."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:4]
    t = 2.0 * np.cross(q_xyz, v)
    return (v + q_w * t + np.cross(q_xyz, t)).astype(np.float32)


def _smpl_root_ytoz_up(root_quat_yup_wxyz: np.ndarray) -> np.ndarray:
    """Match isaac_utils/rotations.smpl_root_ytoz_up: rotate root quat 90° about X axis.
    Input/output are wxyz quaternions.
    """
    base_rot = _axis_angle_to_quat_wxyz(np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32))
    # base_rot is shape (4,). Broadcast to match leading dims of root_quat.
    n_dims = root_quat_yup_wxyz.ndim
    while base_rot.ndim < n_dims:
        base_rot = base_rot[None, :]
    return _quat_mul_wxyz(np.broadcast_to(base_rot, root_quat_yup_wxyz.shape), root_quat_yup_wxyz)


def _remove_smpl_base_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    """Match isaac_utils/rotations.remove_smpl_base_rot: right-multiply by conjugate of [0.5,0.5,0.5,0.5].

    SMPL's T-pose default orientation is [0.5,0.5,0.5,0.5] (120° about [1,1,1] axis).
    Conjugating it out aligns with a neutral standing pose.
    """
    base_rot = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    base_rot_conj = _quat_conjugate_wxyz(base_rot)
    while base_rot_conj.ndim < quat.ndim:
        base_rot_conj = base_rot_conj[None, :]
    return _quat_mul_wxyz(quat, np.broadcast_to(base_rot_conj, quat.shape))


def _compute_smpl_root_quat_w(pose_aa_T72: np.ndarray, smpl_y_up: bool = True) -> np.ndarray:
    """Reproduce TrackingCommand.smpl_root_quat_w from a SMPL pose array.

    pose_aa_T72: [T, 72] SMPL pose in axis-angle (24 joints × 3); first 3 are root.
    Returns: [T, 4] wxyz quaternions in Z-up world frame with SMPL T-pose base rotation removed.
    """
    root_aa = pose_aa_T72[..., :3]            # [T, 3]
    root_quat = _axis_angle_to_quat_wxyz(root_aa)  # [T, 4]
    if smpl_y_up:
        root_quat = _smpl_root_ytoz_up(root_quat)
    root_quat = _remove_smpl_base_rot_wxyz(root_quat)
    # Normalize for numerical safety
    norms = np.linalg.norm(root_quat, axis=-1, keepdims=True).clip(min=1e-8)
    return (root_quat / norms).astype(np.float32)


def _canonicalize_smpl_joints(
    smpl_joints_T243: np.ndarray, smpl_root_quat_wxyz_T4: np.ndarray
) -> np.ndarray:
    """Apply quat_inv(smpl_root_quat) to each frame's 24 joints.

    Matches the training-time observation `smpl_joints_multi_future_local` which does
    ``ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)`` per frame.
    """
    inv_root = _quat_conjugate_wxyz(smpl_root_quat_wxyz_T4)   # [T, 4]; unit quats: conj == inv
    # Broadcast inv_root to [T, 24, 4]
    inv_root_b = np.broadcast_to(inv_root[:, None, :], (inv_root.shape[0], 24, 4)).copy()
    return _quat_apply_wxyz(inv_root_b, smpl_joints_T243).astype(np.float32)


def _load_smpl_pkl(path: Path) -> dict:
    """Load smpl_filtered/*.pkl (flat dict) and return the relevant arrays."""
    d = joblib.load(path)
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict in {path}, got {type(d).__name__}")
    needed = ["pose_aa", "smpl_joints"]
    for k in needed:
        if k not in d:
            raise KeyError(f"smpl pkl {path} missing key {k!r}; has keys={list(d.keys())}")
    return d


def main() -> None:
    args = parse_args()

    # 1) Robot side via the existing motionlib provider (re-uses TrackingCommand offline)
    seq = load_motionlib_sequence(
        motion_file=args.motion_file,
        motion_name=args.motion_name,
        target_fps=args.target_fps,
        num_future_frames=args.num_future_frames,
        dt_future_ref_frames=args.dt_future_ref_frames,
        device=args.device,
        prefer_motionlib_robot=not args.no_motionlib_robot,
        use_isaacsim_app=args.use_isaacsim_app,
    )
    dof_pos = seq.dof_pos.detach().cpu().numpy().astype(np.float32)
    dof_vel = seq.dof_vel.detach().cpu().numpy().astype(np.float32)
    body_pos_w_full = seq.body_pos_w.detach().cpu().numpy().astype(np.float32)
    if body_pos_w_full.ndim != 3 or body_pos_w_full.shape[1] < 1 or body_pos_w_full.shape[2] != 3:
        raise ValueError(f"Unexpected body_pos_w shape: {body_pos_w_full.shape}")
    root_pos_w = body_pos_w_full[:, 0, :]
    root_quat = seq.root_quat_w.detach().cpu().numpy().astype(np.float32)

    # 2) SMPL side: raw pose_aa[T,72] + smpl_joints[T,24,3] from smpl_filtered pkl.
    smpl = _load_smpl_pkl(args.smpl_motion_file)
    smpl_pose_72 = np.asarray(smpl["pose_aa"], dtype=np.float32)
    smpl_joints_raw = np.asarray(smpl["smpl_joints"], dtype=np.float32)
    if smpl_pose_72.ndim != 2 or smpl_pose_72.shape[1] != 72:
        raise ValueError(f"smpl pose_aa must be [T,72], got {smpl_pose_72.shape}")
    if smpl_joints_raw.ndim != 3 or smpl_joints_raw.shape[1:] != (24, 3):
        raise ValueError(f"smpl_joints must be [T,24,3], got {smpl_joints_raw.shape}")
    smpl_pose = smpl_pose_72.reshape(-1, 24, 3)

    # 2a) Pick the "reference root quaternion" based on --smpl-anchor-mode.
    #     CRITICAL design: the SAME root quat is used for BOTH (i) the streamed body_quat_w
    #     (drives smpl_anchor_orientation observation in deploy) AND (ii) the canonicalization
    #     of smpl_joints (drives smpl_joints_10frame_step1 observation). This ensures the two
    #     SMPL observations are internally consistent — both expressed relative to the SAME
    #     "reference body frame" — so the trained encoder sees a coherent input distribution.
    if args.smpl_anchor_mode == "robot_root":
        # G1 motion's root quaternion from motionlib (== robot encoder pipeline's body_quat_w).
        # Re-uses base_sim.py's existing yaw+XY anchor for visual alignment. Encoder obs is
        # ~2-3° offset from training distribution (rely on policy robustness).
        reference_root_quat = root_quat
    elif args.smpl_anchor_mode == "smpl_processed":
        # Y→Z up (if smpl_y_up) then remove_smpl_base_rot. Strict training distribution match.
        # Ref viz has ~2-3° pitch/roll offset from actual G1 (yaw-only sim anchor can't fix).
        reference_root_quat = _compute_smpl_root_quat_w(smpl_pose_72, smpl_y_up=args.smpl_y_up)
    elif args.smpl_anchor_mode == "smpl_raw":
        # Diagnostic only: raw axis-angle → quat with no Y→Z or base-rot removal.
        reference_root_quat = _axis_angle_to_quat_wxyz(smpl_pose[:, 0, :])
    else:
        raise ValueError(f"Unknown --smpl-anchor-mode: {args.smpl_anchor_mode}")

    # 2b) Canonicalize smpl_joints with quat_inv(reference_root_quat) per-frame, matching
    #     training observation `smpl_joints_multi_future_local`.
    if args.smpl_joints_mode == "canonicalized":
        smpl_joints = _canonicalize_smpl_joints(smpl_joints_raw, reference_root_quat)
    else:
        smpl_joints = smpl_joints_raw

    # 3) Frame-align robot and SMPL sequences (both should be at target_fps).
    n_robot = dof_pos.shape[0]
    n_smpl = smpl_pose.shape[0]
    n = min(n_robot, n_smpl, reference_root_quat.shape[0])
    if n_robot != n_smpl:
        print(
            f"[warn] robot frames ({n_robot}) != smpl frames ({n_smpl}); "
            f"truncating both to {n}"
        )
    dof_pos = dof_pos[:n]
    dof_vel = dof_vel[:n]
    root_pos_w = root_pos_w[:n]
    smpl_pose = smpl_pose[:n]
    smpl_joints = smpl_joints[:n]
    reference_root_quat = reference_root_quat[:n]

    # 4) Slice user-specified [start_frame, end_frame).
    end = n if args.end_frame is None else min(args.end_frame, n)
    start = max(0, args.start_frame)
    if start >= end:
        raise ValueError(f"empty frame range: start={start}, end={end}")
    dof_pos = dof_pos[start:end]
    dof_vel = dof_vel[start:end]
    root_pos_w = root_pos_w[start:end]
    smpl_pose = smpl_pose[start:end]
    smpl_joints = smpl_joints[start:end]
    reference_root_quat = reference_root_quat[start:end]

    # 5) The streamed body_quat_w IS the reference_root_quat by construction.
    body_quat_for_stream = reference_root_quat
    print(
        f"[smpl-stream] anchor_mode={args.smpl_anchor_mode} "
        f"joints_mode={args.smpl_joints_mode} smpl_y_up={args.smpl_y_up} "
        f"(canonicalize uses same root as anchor for internal consistency)"
    )

    # 6) Prepend stand transition on DOF side (existing util). Apply same prefix length to
    #    smpl side via repeat-of-first-frame, so frame counts stay aligned.
    dof_pos_full, dof_vel_full, body_quat_full = _prepend_stand_transition(
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        root_quat=body_quat_for_stream,
        target_fps=args.target_fps,
        stand_frames=max(0, args.prepend_stand_frames),
        blend_frames=max(0, args.blend_from_stand_frames),
    )
    num_prefix = len(dof_pos_full) - len(dof_pos)
    if num_prefix > 0:
        prefix_root_pos = np.repeat(root_pos_w[:1], num_prefix, axis=0).astype(np.float32)
        root_pos_w_full = np.concatenate([prefix_root_pos, root_pos_w], axis=0).astype(np.float32)
        prefix_smpl_pose = np.repeat(smpl_pose[:1], num_prefix, axis=0).astype(np.float32)
        smpl_pose_full = np.concatenate([prefix_smpl_pose, smpl_pose], axis=0).astype(np.float32)
        prefix_smpl_joints = np.repeat(smpl_joints[:1], num_prefix, axis=0).astype(np.float32)
        smpl_joints_full = np.concatenate(
            [prefix_smpl_joints, smpl_joints], axis=0
        ).astype(np.float32)
    else:
        root_pos_w_full = root_pos_w
        smpl_pose_full = smpl_pose
        smpl_joints_full = smpl_joints

    sent_frames = len(dof_pos_full)
    end = sent_frames
    start = 0

    motion_start_frame = max(0, args.prepend_stand_frames) + max(0, args.blend_from_stand_frames)
    pub = PackedPublisherSMPL(
        args.host, args.port, verbose=args.verbose, motion_start_frame=motion_start_frame
    )
    print(
        f"[smpl-stream] {seq.motion_name} robot_source={seq.source} "
        f"smpl_pkl={args.smpl_motion_file.name} "
        f"frames={sent_frames} fps={seq.fps} endpoint={pub.endpoint}"
    )

    def _send_range(istart: int, iend: int) -> None:
        for i in range(istart, iend, args.chunk_size):
            j = min(i + args.chunk_size, iend)
            idx = np.arange(i, j, dtype=np.int64)
            pub.send_pose(
                joint_pos=dof_pos_full[i:j],
                joint_vel=dof_vel_full[i:j],
                smpl_joints=smpl_joints_full[i:j],
                smpl_pose=smpl_pose_full[i:j],
                body_pos_w=root_pos_w_full[i:j],
                body_quat_w=body_quat_full[i:j],
                frame_indices=idx,
                catch_up=args.catch_up,
            )

    try:
        time.sleep(args.startup_delay)
        frame_period = 1.0 / float(args.target_fps)
        prestart_frames = min(
            end,
            max(
                0,
                args.initial_burst_frames,
                args.chunk_size,
                args.num_future_frames,
            ),
        )
        if prestart_frames > start:
            _send_range(start, prestart_frames)
            if args.verbose:
                print(
                    f"[startup] prebuffered {start}..{prestart_frames - 1} before start command"
                )

        last_command_heartbeat = time.monotonic()
        if args.send_command:
            for _ in range(max(1, args.command_repeat)):
                pub.send_command(start=True, stop=False, planner=False)
                time.sleep(max(0.0, args.command_interval))
            time.sleep(0.1)
            last_command_heartbeat = time.monotonic()

        burst_end = min(end, max(prestart_frames, start + max(0, args.initial_burst_frames)))
        for i in range(burst_end, end, args.chunk_size):
            j = min(i + args.chunk_size, end)
            if args.send_command and args.command_heartbeat_interval > 0.0:
                now = time.monotonic()
                if now - last_command_heartbeat >= args.command_heartbeat_interval:
                    pub.send_command(start=True, stop=False, planner=False)
                    last_command_heartbeat = now
            _send_range(i, j)
            if args.realtime:
                time.sleep((j - i) * frame_period)
        print("[OK] smpl stream complete")
    finally:
        pub.close()


if __name__ == "__main__":
    main()
