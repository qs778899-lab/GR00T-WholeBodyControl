#!/usr/bin/env python3
"""Background recorder that subscribes to deploy's g1_debug ZMQ topic and writes a single
parquet file when stopped. Output schema mirrors run_data_exporter.py's official format
(see gear_sonic/data/features_sonic_vla.py), minus the video / image columns.

Designed for sim2sim replay scenarios (no real camera). The recorder is driven by:
  (a) deploy's ZMQOutputHandler publishing on tcp://<host>:<port> topic 'g1_debug'
      (msgpack frames with body_q, last_action, base_quat, token_state, ...)
  (b) The streamer pushing the latest teleop snapshot (smpl_joints, smpl_pose, ...) via
      update_teleop_snapshot() before each chunk is sent over ZMQ.

Zero-intrusion: reuses gear_sonic.utils.data_collection.zmq_state_subscriber.ZMQStateSubscriber
for ZMQ + msgpack handling. No other repo files are modified by this module.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber


def _compute_projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Project world gravity (0,0,-1) into base frame using base_quat (wxyz)."""
    w, x, y, z = float(base_quat_wxyz[0]), float(base_quat_wxyz[1]), float(base_quat_wxyz[2]), float(base_quat_wxyz[3])
    # R = quat_to_matrix(wxyz); g_world = (0,0,-1); g_body = R^T @ g_world = -R^T[:, 2]
    # R[:,2] = ( 2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y) )
    gx = -(2.0 * (x * z + w * y))
    gy = -(2.0 * (y * z - w * x))
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.array([gx, gy, gz], dtype=np.float64)


def _expand_actuated_to_whole(
    body_q: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    robot_model: Any | None,
) -> np.ndarray:
    """Build the 43-dim whole-body configuration that observation.state expects.

    If a RobotModel is provided, use its pinocchio-based get_configuration_from_actuated_joints
    (the official path). Otherwise fall back to naive concat (body + left + right) which is
    correct in value but may differ from the official joint ordering inside q0.
    """
    body_q = np.asarray(body_q, dtype=np.float64).reshape(-1)
    left_hand_q = np.asarray(left_hand_q, dtype=np.float64).reshape(-1)
    right_hand_q = np.asarray(right_hand_q, dtype=np.float64).reshape(-1)
    if robot_model is not None:
        return np.asarray(
            robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=left_hand_q,
                right_hand_actuated_joint_values=right_hand_q,
            ),
            dtype=np.float64,
        )
    return np.concatenate([body_q, left_hand_q, right_hand_q]).astype(np.float64)


class DeployStateRecorder:
    """Subscribe to deploy g1_debug + capture streamer teleop snapshots, then dump parquet.

    Threading model: a single daemon thread polls the ZMQ socket at ~50 Hz; each new
    proprio message is paired with the latest teleop snapshot (set by the streamer thread
    via update_teleop_snapshot) and appended to an in-memory list. save() converts the
    list to a DataFrame and writes parquet.
    """

    def __init__(
        self,
        zmq_host: str = "localhost",
        zmq_port: int = 5608,
        poll_hz: float = 100.0,
        try_load_robot_model: bool = True,
        verbose: bool = False,
    ):
        self._sub = ZMQStateSubscriber(host=zmq_host, port=zmq_port)
        self._poll_period = 1.0 / float(poll_hz)
        self._verbose = verbose

        self._teleop_lock = threading.Lock()
        self._latest_teleop: dict[str, np.ndarray] = {}

        self._frames: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_index: int | None = None

        self._robot_model: Any | None = None
        if try_load_robot_model:
            try:
                from gear_sonic.data.features_sonic_vla import get_g1_robot_model
                self._robot_model = get_g1_robot_model()
                if verbose:
                    print("[recorder] loaded G1 RobotModel for joint expansion")
            except Exception as exc:
                print(
                    f"[recorder] G1 RobotModel unavailable ({exc.__class__.__name__}: {exc}); "
                    "falling back to naive concat for observation.state / action.wbc"
                )

    def update_teleop_snapshot(self, snap: dict[str, np.ndarray]) -> None:
        """Called by streamer with the latest teleop.* fields about to be sent."""
        with self._teleop_lock:
            # Shallow copy of array references is fine — caller is expected to hand over
            # freshly built per-frame arrays that they won't mutate after this call.
            self._latest_teleop = dict(snap)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="DeployStateRecorder", daemon=True)
        self._thread.start()
        if self._verbose:
            print(f"[recorder] background poll thread started ({1.0/self._poll_period:.0f} Hz)")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        try:
            self._sub.close()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            msg = self._sub.get_msg(clear=True)
            if msg is None:
                time.sleep(self._poll_period)
                continue
            # Deduplicate consecutive identical proprio frames using the monotonic 'index'.
            idx = msg.get("index", None)
            if idx is not None and self._last_index is not None and idx == self._last_index:
                time.sleep(self._poll_period)
                continue
            self._last_index = idx
            with self._teleop_lock:
                teleop_snap = self._latest_teleop  # shallow ref; built fresh each chunk
            row = self._build_row(msg, teleop_snap)
            if row is not None:
                self._frames.append(row)
            time.sleep(self._poll_period)

    def _build_row(self, proprio: dict, teleop: dict[str, np.ndarray]) -> dict[str, Any] | None:
        try:
            body_q = np.asarray(proprio["body_q"], dtype=np.float64)
            left_hand_q = np.asarray(proprio.get("left_hand_q", np.zeros(7)), dtype=np.float64)
            right_hand_q = np.asarray(proprio.get("right_hand_q", np.zeros(7)), dtype=np.float64)
            last_action = np.asarray(proprio["last_action"], dtype=np.float64)
            last_left_action = np.asarray(
                proprio.get("last_left_hand_action", np.zeros(7)), dtype=np.float64
            )
            last_right_action = np.asarray(
                proprio.get("last_right_hand_action", np.zeros(7)), dtype=np.float64
            )
        except KeyError as exc:
            if self._verbose:
                print(f"[recorder] dropping proprio msg with missing key: {exc}")
            return None

        whole_q = _expand_actuated_to_whole(body_q, left_hand_q, right_hand_q, self._robot_model)
        whole_action_wbc = _expand_actuated_to_whole(
            last_action, last_left_action, last_right_action, self._robot_model
        )

        base_quat = np.asarray(proprio.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        projected_gravity = _compute_projected_gravity(base_quat)
        init_base_quat = np.asarray(
            proprio.get("init_base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
        )
        cpp_rot_offset = np.asarray(
            proprio.get("init_ref_data_root_rot_array", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64
        )

        delta_heading = proprio.get("delta_heading", 0.0)
        if isinstance(delta_heading, np.ndarray):
            delta_heading = float(delta_heading.flat[0])
        else:
            delta_heading = float(delta_heading)

        token_state = proprio.get("token_state", None)
        if token_state is None or (isinstance(token_state, np.ndarray) and token_state.size == 0):
            motion_token = np.zeros(64, dtype=np.float64)
        else:
            motion_token = np.asarray(token_state, dtype=np.float64).reshape(-1)
            if motion_token.size < 64:
                motion_token = np.pad(motion_token, (0, 64 - motion_token.size))
            elif motion_token.size > 64:
                motion_token = motion_token[:64]

        ros_ts = proprio.get("ros_timestamp", 0.0)
        if ros_ts == 0.0:
            ros_ts = time.time()

        row: dict[str, Any] = {
            "observation.state": whole_q,
            "action.wbc": whole_action_wbc,
            "observation.root_orientation": base_quat,
            "observation.projected_gravity": projected_gravity,
            "observation.cpp_rotation_offset": cpp_rot_offset,
            "observation.init_base_quat": init_base_quat,
            "teleop.delta_heading": np.array([delta_heading], dtype=np.float64),
            "action.motion_token": motion_token,
            "timestamp": np.float32(ros_ts),
        }
        # Overlay teleop snapshot from streamer (smpl_joints, smpl_pose, body_quat_w, wrist joints, ...)
        for k, v in teleop.items():
            row[k] = v
        return row

    def save(self, parquet_path: Path) -> int:
        """Write accumulated frames to a flat parquet file. Returns row count."""
        if not self._frames:
            print(f"[recorder] no frames captured; skipping write to {parquet_path}")
            return 0
        df = pd.DataFrame(self._frames)
        df.insert(0, "frame_index", np.arange(len(df), dtype=np.int64))
        df.insert(1, "episode_index", np.zeros(len(df), dtype=np.int64))
        df.insert(2, "index", np.arange(len(df), dtype=np.int64))
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        print(
            f"[recorder] wrote {len(df)} rows × {len(df.columns)} columns to {parquet_path} "
            f"({parquet_path.stat().st_size / 1024:.1f} KB)"
        )
        return len(df)
