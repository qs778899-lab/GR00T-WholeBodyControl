#!/usr/bin/env python3
"""Stream official SONIC motion_lib PKLs to gear_sonic_deploy over ZMQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.sonic_eval.motionlib_provider import load_motionlib_sequence
from tools.sonic_eval.motionlib_provider import G1_DEFAULT_ANGLES_ISAACLAB


HEADER_SIZE = 1280


class PackedPublisher:
    def __init__(self, host: str, port: int, verbose: bool = False):
        import zmq

        self.verbose = verbose
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
        body_quat_w: np.ndarray,
        frame_indices: np.ndarray,
        catch_up: bool,
    ) -> None:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        n, num_joints = joint_pos.shape
        if body_quat_w.ndim == 3 and body_quat_w.shape[1] == 1:
            body_quat_w = body_quat_w[:, 0, :]
        if body_quat_w.shape != (n, 4):
            raise ValueError(f"body_quat_w must be [N,4], got {body_quat_w.shape}")
        header = {
            "v": 1,
            "endian": "le",
            "count": n,
            "fields": [
                {"name": "joint_pos", "dtype": "f32", "shape": [n, num_joints]},
                {"name": "joint_vel", "dtype": "f32", "shape": [n, num_joints]},
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
                body_quat_w.tobytes(),
                frame_indices.tobytes(),
                struct.pack("B", 1 if catch_up else 0),
            ],
        )
        if self.verbose:
            print(f"[pose] frames={frame_indices[0]}..{frame_indices[-1]} n={n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--motion-name", type=str, default=None)
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
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument(
        "--initial-burst-frames",
        type=int,
        default=60,
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
    parser.add_argument("--send-command", action="store_true", help="send start=true planner=false before pose stream")
    parser.add_argument(
        "--command-repeat",
        type=int,
        default=3,
        help="number of start/planner=false commands to publish when --send-command is set",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=0.05,
        help="seconds between repeated command messages",
    )
    parser.add_argument(
        "--command-heartbeat-interval",
        type=float,
        default=0.5,
        help=(
            "while streaming, resend start/planner command at this interval (seconds); "
            "set <=0 to disable"
        ),
    )
    parser.add_argument("--startup-delay", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _finite_difference(values: np.ndarray, fps: int) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    vel = np.zeros_like(values, dtype=np.float32)
    vel[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    vel[0] = (values[1] - values[0]) * fps
    vel[-1] = (values[-1] - values[-2]) * fps
    return vel


def _prepend_stand_transition(
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    root_quat: np.ndarray,
    target_fps: int,
    stand_frames: int,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stand_frames <= 0 and blend_frames <= 0:
        return dof_pos, dof_vel, root_quat
    if len(dof_pos) == 0:
        return dof_pos, dof_vel, root_quat

    default_pose = G1_DEFAULT_ANGLES_ISAACLAB.detach().cpu().numpy().astype(np.float32)
    prefix_parts = []
    if stand_frames > 0:
        prefix_parts.append(np.repeat(default_pose[None, :], stand_frames, axis=0))
    if blend_frames > 0:
        # Exclude alpha=1.0; the original first frame follows immediately after.
        alpha = np.linspace(0.0, 1.0, blend_frames + 1, dtype=np.float32)[:-1, None]
        prefix_parts.append((1.0 - alpha) * default_pose[None, :] + alpha * dof_pos[:1])
    if not prefix_parts:
        return dof_pos, dof_vel, root_quat

    prefix_pos = np.concatenate(prefix_parts, axis=0).astype(np.float32)
    new_pos = np.concatenate([prefix_pos, dof_pos], axis=0).astype(np.float32)
    new_vel = _finite_difference(new_pos, target_fps)
    # Keep reference orientation constant during the warm-start. Deploy applies
    # heading alignment, so joint discontinuity is the more important hazard.
    prefix_quat = np.repeat(root_quat[:1], len(prefix_pos), axis=0).astype(np.float32)
    new_quat = np.concatenate([prefix_quat, root_quat], axis=0).astype(np.float32)
    return new_pos, new_vel, new_quat


def main() -> None:
    args = parse_args()
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
    root_quat = seq.root_quat_w.detach().cpu().numpy().astype(np.float32)
    end = seq.num_frames if args.end_frame is None else min(args.end_frame, seq.num_frames)
    start = max(0, args.start_frame)
    if start >= end:
        raise ValueError(f"empty frame range: start={start}, end={end}")
    dof_pos = dof_pos[start:end]
    dof_vel = dof_vel[start:end]
    root_quat = root_quat[start:end]
    dof_pos, dof_vel, root_quat = _prepend_stand_transition(
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        root_quat=root_quat,
        target_fps=args.target_fps,
        stand_frames=max(0, args.prepend_stand_frames),
        blend_frames=max(0, args.blend_from_stand_frames),
    )
    sent_frames = len(dof_pos)
    end = sent_frames
    start = 0

    pub = PackedPublisher(args.host, args.port, verbose=args.verbose)
    print(
        f"[motionlib] {seq.motion_name} source={seq.source} source_frames={seq.num_frames} "
        f"stream={start}:{end} sent_frames={sent_frames} fps={seq.fps} endpoint={pub.endpoint}"
    )
    try:
        time.sleep(args.startup_delay)
        last_command_heartbeat = time.monotonic()
        if args.send_command:
            for _ in range(max(1, args.command_repeat)):
                pub.send_command(start=True, stop=False, planner=False)
                time.sleep(max(0.0, args.command_interval))
            time.sleep(0.1)
            last_command_heartbeat = time.monotonic()
        frame_period = 1.0 / float(args.target_fps)

        burst_end = min(end, start + max(0, args.initial_burst_frames))
        for i in range(start, burst_end, args.chunk_size):
            j = min(i + args.chunk_size, burst_end)
            idx = np.arange(i, j, dtype=np.int64)
            if args.send_command and args.command_heartbeat_interval > 0.0:
                now = time.monotonic()
                if now - last_command_heartbeat >= args.command_heartbeat_interval:
                    pub.send_command(start=True, stop=False, planner=False)
                    last_command_heartbeat = now
            pub.send_pose(
                joint_pos=dof_pos[i:j],
                joint_vel=dof_vel[i:j],
                body_quat_w=root_quat[i:j],
                frame_indices=idx,
                catch_up=args.catch_up,
            )

        for i in range(burst_end, end, args.chunk_size):
            j = min(i + args.chunk_size, end)
            idx = np.arange(i, j, dtype=np.int64)
            if args.send_command and args.command_heartbeat_interval > 0.0:
                now = time.monotonic()
                if now - last_command_heartbeat >= args.command_heartbeat_interval:
                    pub.send_command(start=True, stop=False, planner=False)
                    last_command_heartbeat = now
            pub.send_pose(
                joint_pos=dof_pos[i:j],
                joint_vel=dof_vel[i:j],
                body_quat_w=root_quat[i:j],
                frame_indices=idx,
                catch_up=args.catch_up,
            )
            if args.realtime:
                time.sleep((j - i) * frame_period)
        print("[OK] stream complete")
    finally:
        pub.close()


if __name__ == "__main__":
    main()
