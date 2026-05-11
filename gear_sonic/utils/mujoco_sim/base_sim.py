"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import os
import pathlib
from pathlib import Path
import pickle
from copy import deepcopy
import csv
import json
import tempfile
from threading import Lock, Thread
import time
from collections import OrderedDict, deque
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.link_error_plot import Sim2SimLinkErrorPlot
from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names, get_subtree_geom_ids
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REFERENCE_NAME_PREFIX = "ref_"
PACKED_ZMQ_HEADER_SIZE = 1280
G1_ISAACLAB_TO_MUJOCO_DOF = np.array(
    [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
        11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int32,
)

SIM2SIM_BODY_FRAMES = [
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


class PackedZMQSubscriber:
    """Non-blocking subscriber for packed ZMQ motion streams."""

    _DTYPE_MAP = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "bool": np.bool_,
    }

    def __init__(self, host: str, port: int, topic: str):
        import zmq

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic
        self._msg = None
        print(f"[PackedZMQSubscriber] Connected to tcp://{host}:{port} (topic: {topic})")

    def _unpack(self, packed_data: bytes) -> dict:
        topic_bytes = self._topic.encode("utf-8")
        if not packed_data.startswith(topic_bytes):
            raise ValueError(f"Message does not start with expected topic '{self._topic}'")

        offset = len(topic_bytes)
        if len(packed_data) < offset + PACKED_ZMQ_HEADER_SIZE:
            raise ValueError(
                f"Packed data too small: {len(packed_data)} < {offset + PACKED_ZMQ_HEADER_SIZE}"
            )

        header_bytes = packed_data[offset : offset + PACKED_ZMQ_HEADER_SIZE]
        null_idx = header_bytes.find(b"\x00")
        if null_idx > 0:
            header_bytes = header_bytes[:null_idx]

        header = json.loads(header_bytes.decode("utf-8"))
        result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
        current_offset = offset + PACKED_ZMQ_HEADER_SIZE

        for field in header.get("fields", []):
            dtype = self._DTYPE_MAP.get(field["dtype"])
            if dtype is None:
                raise ValueError(f"Unsupported dtype in packed message: {field['dtype']}")
            shape = tuple(field["shape"])
            n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            result[field["name"]] = (
                np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
                .reshape(shape)
                .copy()
            )
            current_offset += n_bytes
        return result

    def _poll(self):
        import zmq

        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        self._msg = self._unpack(raw)

    def get_msg(self, clear: bool = True):
        self._poll()
        msg = self._msg
        if clear:
            self._msg = None
        return msg

    def close(self):
        self._socket.close()
        self._ctx.term()


class ReferenceMotionVisualizer:
    """Drive a translucent reference robot from the raw pose stream.

    The deploy debug topic is used only for `source_frame_index` timing sync.
    Reference pose content for both visualization and strict metrics GT must
    come from the original `pose` stream, not deploy's heading-corrected target.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        body_joint_names: list[str],
        left_hand_joint_names: list[str],
        right_hand_joint_names: list[str],
        alpha: float,
        host: str,
        port: int,
        topic: str,
        translation_mode: str,
        pose_port: int | None = None,
        pose_topic: str = "pose",
        allow_midrun_realign: bool = False,
        align_delay_frames: int = 50,
    ):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.visible = False
        self.enabled = True
        self._latest_pose = None
        self.translation_mode = translation_mode
        self._align_delay_frames = max(1, int(align_delay_frames))
        self.allow_midrun_realign = bool(allow_midrun_realign)
        self._target_anchor_base_pos = None
        self._actual_anchor_base_pos = None
        self._start_aligned_gt_base_pos = None
        self._start_aligned_actual_base_pos = None
        self._start_aligned_gt_yaw = None
        self._start_aligned_actual_yaw = None
        self._start_aligned_yaw_delta = None
        self._prev_target_base_pos = None

        root_joint_name = f"{REFERENCE_NAME_PREFIX}floating_base_joint"
        root_body_name = f"{REFERENCE_NAME_PREFIX}pelvis"
        root_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        root_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_joint_id == -1 or root_body_id == -1:
            raise ValueError("reference robot was not found in the loaded MuJoCo model")

        self.root_qpos_adr = mj_model.jnt_qposadr[root_joint_id]
        self.root_qvel_adr = mj_model.jnt_dofadr[root_joint_id]
        self.ref_root_body_id = root_body_id
        self.actual_root_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.actual_root_body_id == -1:
            raise ValueError("actual robot pelvis was not found in the loaded MuJoCo model")
        self.body_qpos_adrs = []
        self.body_qvel_adrs = []
        for joint_name in body_joint_names:
            ref_joint_name = f"{REFERENCE_NAME_PREFIX}{joint_name}"
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, ref_joint_name)
            if joint_id == -1:
                raise ValueError(f"reference joint missing from MuJoCo model: {ref_joint_name}")
            self.body_qpos_adrs.append(mj_model.jnt_qposadr[joint_id])
            self.body_qvel_adrs.append(mj_model.jnt_dofadr[joint_id])

        self.body_qpos_adrs = np.array(self.body_qpos_adrs, dtype=np.int32)
        self.body_qvel_adrs = np.array(self.body_qvel_adrs, dtype=np.int32)
        self.ref_hand_qpos_adrs = []
        self.actual_hand_qpos_adrs = []
        for joint_name in list(left_hand_joint_names) + list(right_hand_joint_names):
            actual_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ref_joint_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{REFERENCE_NAME_PREFIX}{joint_name}"
            )
            if actual_joint_id == -1 or ref_joint_id == -1:
                continue
            self.actual_hand_qpos_adrs.append(mj_model.jnt_qposadr[actual_joint_id])
            self.ref_hand_qpos_adrs.append(mj_model.jnt_qposadr[ref_joint_id])
        self.actual_hand_qpos_adrs = np.array(self.actual_hand_qpos_adrs, dtype=np.int32)
        self.ref_hand_qpos_adrs = np.array(self.ref_hand_qpos_adrs, dtype=np.int32)
        self.ref_geom_ids = np.array(get_subtree_geom_ids(mj_model, root_body_id), dtype=np.int32)
        self._shown_once = False
        self._last_pose_log_time = 0.0
        self._last_debug_visibility_log_time = 0.0
        self._last_pose_frame_index = None
        self._last_debug_source_frame_index = None
        self._latest_debug_source_frame_index = None
        self._current_applied_source_frame_index = None
        self._current_exact_pose = None
        self._current_display_pose = None
        self._latest_control_tick_reference_pose = None
        self._display_frame_index = None
        self._display_frame_phase = 0.0
        self._last_display_update_time = time.monotonic()
        self._display_fps = 50.0
        self._display_debug_lag_frames = 1
        self._pose_frame_buffer = OrderedDict()
        self._max_pose_frame_buffer = 4096
        self._pending_debug_source_frames = deque()
        self._translation_anchor_ready = False
        self._raw_pose_stream_seen = False
        self._actual_root_pos_history = deque(maxlen=10)
        self._actual_root_history_dt = 1.0 / 50.0
        self._actual_root_history_last_sim_time = None
        self._anchor_actual_speed_threshold = 0.05
        self._anchor_actual_min_history = 10
        self._anchor_wait_logged = False
        self._delayed_align_frame_count = 0
        self.set_visible(False)

        self.debug_subscriber = ZMQStateSubscriber(host=host, port=port, topic=topic, conflate=False)
        self.pose_subscriber = None
        if pose_port is not None:
            try:
                self.pose_subscriber = PackedZMQSubscriber(host=host, port=pose_port, topic=pose_topic)
            except Exception as exc:
                print(f"[ReferenceMotionVisualizer] pose subscriber disabled: {exc}")

    def close(self):
        self.debug_subscriber.close()
        if self.pose_subscriber is not None:
            self.pose_subscriber.close()

    @property
    def latest_debug_source_frame_index(self) -> int | None:
        return self._latest_debug_source_frame_index

    @property
    def current_applied_source_frame_index(self) -> int | None:
        return self._current_applied_source_frame_index

    @property
    def current_exact_pose(self):
        return self._current_exact_pose

    def reset_anchor(self):
        self._target_anchor_base_pos = None
        self._actual_anchor_base_pos = None
        self._start_aligned_gt_base_pos = None
        self._start_aligned_actual_base_pos = None
        self._start_aligned_gt_yaw = None
        self._start_aligned_actual_yaw = None
        self._start_aligned_yaw_delta = None
        self._prev_target_base_pos = None
        self._last_pose_frame_index = None
        self._last_debug_source_frame_index = None
        self._latest_debug_source_frame_index = None
        self._current_applied_source_frame_index = None
        self._current_exact_pose = None
        self._current_display_pose = None
        self._latest_control_tick_reference_pose = None
        self._display_frame_index = None
        self._display_frame_phase = 0.0
        self._last_display_update_time = time.monotonic()
        self._pose_frame_buffer.clear()
        self._pending_debug_source_frames.clear()
        self._translation_anchor_ready = False
        self._raw_pose_stream_seen = False
        self._actual_root_pos_history.clear()
        self._actual_root_history_last_sim_time = None
        self._anchor_wait_logged = False
        self._delayed_align_frame_count = 0
        self._shown_once = False
        self.set_visible(False)

    def set_visible(self, visible: bool):
        self.visible = visible
        alpha = self.alpha if visible and self.enabled else 0.0
        self.mj_model.geom_rgba[self.ref_geom_ids, 3] = alpha
        print(
            "[ReferenceMotionVisualizer] set_visible "
            f"visible={visible} enabled={self.enabled} alpha={alpha:.3f} "
            f"ref_geoms={len(self.ref_geom_ids)}"
        )

    def toggle(self):
        self.set_visible(not self.visible)
        print(
            f"[ReferenceMotionVisualizer] reference motion visualization "
            f"{'enabled' if self.visible else 'hidden'}"
        )

    def poll(self):
        # Always ingest the raw pose stream first so we keep a frame-indexed buffer
        # of the original streamed reference. Debug is only a timing/sync channel
        # telling us which source frame deploy is currently consuming.
        pose_msg = self.pose_subscriber.get_msg() if self.pose_subscriber is not None else None
        if pose_msg is not None:
            self._consume_pose_stream_msg(pose_msg)

        debug_msgs = self.debug_subscriber.get_msgs()
        if debug_msgs:
            for debug_msg in debug_msgs:
                self._consume_debug_msg(debug_msg)

        self._advance_pending_debug_frame()

        # If deploy has not published a source frame yet, fall back to the newest
        # buffered pose so the reference still appears before control starts.
        if self.pose_subscriber is not None and self._latest_debug_source_frame_index is None:
            pose = self._get_latest_buffered_pose()
            if pose is not None:
                self._set_latest_pose(*pose)
        else:
            self._advance_display_pose()

    def _consume_debug_msg(self, msg: dict) -> bool:
        source_frame_index = msg.get("applied_source_frame_index")
        if source_frame_index is None:
            source_frame_index = msg.get("source_frame_index")
        if source_frame_index is not None:
            try:
                source_frame_index = int(source_frame_index)
            except (TypeError, ValueError):
                source_frame_index = None
        if source_frame_index is not None and source_frame_index >= 0:
            raw_required = ("ref_base_trans_raw", "ref_base_quat_raw", "ref_body_q_raw")
            if all(key in msg for key in raw_required):
                raw_base_pos = np.asarray(msg["ref_base_trans_raw"], dtype=np.float64)
                raw_base_quat = np.asarray(msg["ref_base_quat_raw"], dtype=np.float64)
                raw_body_q = np.asarray(msg["ref_body_q_raw"], dtype=np.float64)
                if raw_base_pos.shape == (3,) and raw_base_quat.shape == (4,) and raw_body_q.shape == (29,):
                    exact_pose = (
                        raw_base_pos.copy(),
                        raw_base_quat.copy(),
                        raw_body_q.copy(),
                    )
                    self._latest_control_tick_reference_pose = (
                        int(source_frame_index),
                        exact_pose,
                    )
                    self._current_applied_source_frame_index = int(source_frame_index)
                    self._current_exact_pose = exact_pose
                    self._set_latest_pose(*exact_pose, synchronized=True)
            if (
                self.allow_midrun_realign
                and (
                self._last_debug_source_frame_index is not None
                and source_frame_index < self._last_debug_source_frame_index
                )
            ):
                self.reset_anchor()
                print("[ReferenceMotionVisualizer] reset translation anchor at new debug stream start")
            self._last_debug_source_frame_index = source_frame_index
            self._latest_debug_source_frame_index = source_frame_index
            if self._latest_control_tick_reference_pose is None or self._latest_control_tick_reference_pose[0] != int(source_frame_index):
                self._enqueue_debug_source_frame(source_frame_index)
            return True

        # Ignore deploy's heading/root-corrected target pose content when a raw
        # pose stream exists. It is not GT and should not drive reference state.
        if self.pose_subscriber is not None or self._raw_pose_stream_seen:
            return bool("source_frame_index" in msg)

        # Fallback only when no raw pose stream is available at all.
        required = ("base_trans_target", "base_quat_target", "body_q_target")
        if not all(key in msg for key in required):
            return False
        base_pos = np.asarray(msg["base_trans_target"], dtype=np.float64)
        base_quat = np.asarray(msg["base_quat_target"], dtype=np.float64)
        body_q = np.asarray(msg["body_q_target"], dtype=np.float64)
        if base_pos.shape != (3,) or base_quat.shape != (4,) or body_q.shape != (29,):
            return False
        self._set_latest_pose(base_pos, base_quat, body_q, synchronized=False)
        return True

    def _consume_pose_stream_msg(self, msg: dict) -> bool:
        required = ("body_pos_w", "body_quat_w", "joint_pos")
        if not all(key in msg for key in required):
            return False

        body_pos = np.asarray(msg["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(msg["body_quat_w"], dtype=np.float64)
        body_q = np.asarray(msg["joint_pos"], dtype=np.float64)
        if body_pos.ndim == 1 and body_pos.shape == (3,):
            body_pos = body_pos.reshape(1, 3)
        if body_quat.ndim == 1 and body_quat.shape == (4,):
            body_quat = body_quat.reshape(1, 4)
        if body_q.ndim == 1 and body_q.shape == (29,):
            body_q = body_q.reshape(1, 29)
        if body_pos.ndim != 2 or body_quat.ndim != 2 or body_q.ndim != 2:
            return False
        if body_pos.shape[1] != 3 or body_quat.shape[1] != 4 or body_q.shape[1] != 29:
            return False
        if not (body_pos.shape[0] == body_quat.shape[0] == body_q.shape[0]):
            return False
        self._raw_pose_stream_seen = True
        body_q = body_q[:, G1_ISAACLAB_TO_MUJOCO_DOF]

        frame_index_arr = None
        if "frame_index" in msg:
            frame_index_arr = np.asarray(msg["frame_index"], dtype=np.int64).reshape(-1)
            if frame_index_arr.size > 0:
                frame_index = int(frame_index_arr[-1])
            else:
                frame_index = None
        else:
            frame_index = None
        if (
            self.allow_midrun_realign
            and (
            frame_index is not None
            and self._last_pose_frame_index is not None
            and frame_index <= self._last_pose_frame_index
            )
        ):
            self.reset_anchor()
            print("[ReferenceMotionVisualizer] re-anchored reference root at new stream start")
        self._last_pose_frame_index = frame_index

        if frame_index_arr is not None and frame_index_arr.size == body_pos.shape[0]:
            if self._display_frame_index is None:
                self._display_frame_index = int(frame_index_arr[0])
            for i, source_frame_index in enumerate(frame_index_arr.tolist()):
                self._store_pose_frame(
                    int(source_frame_index),
                    body_pos[i].copy(),
                    body_quat[i].copy(),
                    body_q[i].copy(),
                )
        else:
            start_idx = frame_index - body_pos.shape[0] + 1 if frame_index is not None else None
            for i in range(body_pos.shape[0]):
                inferred_index = start_idx + i if start_idx is not None else None
                if inferred_index is not None:
                    if self._display_frame_index is None:
                        self._display_frame_index = int(inferred_index)
                    self._store_pose_frame(
                        int(inferred_index),
                        body_pos[i].copy(),
                        body_quat[i].copy(),
                        body_q[i].copy(),
                    )

        if self._latest_debug_source_frame_index is not None:
            matched_pose = self._get_exact_pose_for_frame(self._latest_debug_source_frame_index)
            if matched_pose is not None:
                self._current_applied_source_frame_index = int(self._latest_debug_source_frame_index)
                self._current_exact_pose = tuple(np.asarray(x, dtype=np.float64).copy() for x in matched_pose)

        # Update display immediately toward the latest debug-synchronized pose so
        # reference timing stays close to the actual robot instead of free-running
        # on a local playback clock.
        self._advance_display_pose(force_pose=(body_pos[-1], body_quat[-1], body_q[-1]))
        return True

    def _enqueue_debug_source_frame(self, source_frame_index: int):
        if self._pending_debug_source_frames:
            if self._pending_debug_source_frames[-1] == source_frame_index:
                return
        elif self._current_applied_source_frame_index == source_frame_index:
            return
        self._pending_debug_source_frames.append(source_frame_index)

    def _advance_pending_debug_frame(self) -> bool:
        if self._latest_control_tick_reference_pose is not None:
            return False
        while self._pending_debug_source_frames:
            source_frame_index = int(self._pending_debug_source_frames[0])
            exact_pose = self._get_exact_pose_for_frame(source_frame_index)
            if exact_pose is None:
                break
            self._pending_debug_source_frames.popleft()
            self._current_applied_source_frame_index = source_frame_index
            self._current_exact_pose = tuple(np.asarray(x, dtype=np.float64).copy() for x in exact_pose)
            return True
        return False

    def _store_pose_frame(
        self,
        frame_index: int,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        body_q: np.ndarray,
    ):
        self._pose_frame_buffer[frame_index] = (base_pos, base_quat, body_q)
        self._pose_frame_buffer.move_to_end(frame_index)
        while len(self._pose_frame_buffer) > self._max_pose_frame_buffer:
            self._pose_frame_buffer.popitem(last=False)

    def _get_latest_buffered_pose(self):
        if not self._pose_frame_buffer:
            return None
        return next(reversed(self._pose_frame_buffer.values()))

    def _get_buffered_pose_for_frame(self, frame_index: int):
        if not self._pose_frame_buffer:
            return None
        pose = self._pose_frame_buffer.get(frame_index)
        if pose is not None:
            return pose
        older_keys = [k for k in self._pose_frame_buffer.keys() if k <= frame_index]
        if older_keys:
            return self._pose_frame_buffer[older_keys[-1]]
        newer_keys = [k for k in self._pose_frame_buffer.keys() if k > frame_index]
        if newer_keys:
            return self._pose_frame_buffer[newer_keys[0]]
        return None

    def _get_exact_pose_for_frame(self, frame_index: int):
        if not self._pose_frame_buffer:
            return None
        return self._pose_frame_buffer.get(frame_index)

    def _set_latest_pose(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        body_q: np.ndarray,
        synchronized: bool = False,
    ):
        self._latest_pose = (base_pos, base_quat, body_q)
        self._current_display_pose = tuple(
            np.asarray(x, dtype=np.float64).copy() for x in (base_pos, base_quat, body_q)
        )
        if not synchronized:
            self._current_applied_source_frame_index = None
            self._current_exact_pose = None
        if synchronized and self.translation_mode == "delta_aligned" and not self._translation_anchor_ready:
            self._target_anchor_base_pos = base_pos.copy()
            self._actual_anchor_base_pos = self._get_actual_robot_root_pos()
            self._translation_anchor_ready = True
            print(
                "[ReferenceMotionVisualizer] locked translation anchor "
                f"at actual x={self._actual_anchor_base_pos[0]:.3f} "
                f"y={self._actual_anchor_base_pos[1]:.3f} "
                f"z={self._actual_anchor_base_pos[2]:.3f}"
            )
        if synchronized and self.translation_mode == "start_aligned_xy" and self._start_aligned_gt_base_pos is None:
            actual_root_pos_now = self._get_actual_robot_root_pos()
            sim_time_now = float(self.mj_data.time)
            min_gap = 0.5 * self._actual_root_history_dt
            if (
                self._actual_root_history_last_sim_time is None
                or sim_time_now - self._actual_root_history_last_sim_time >= min_gap
            ):
                self._actual_root_pos_history.append(actual_root_pos_now.copy())
                self._actual_root_history_last_sim_time = sim_time_now
            if not self._is_actual_root_static():
                if not self._anchor_wait_logged:
                    self._anchor_wait_logged = True
                    print(
                        "[ReferenceMotionVisualizer] waiting for actual root to stabilize "
                        f"before locking start-aligned anchor "
                        f"(history={len(self._actual_root_pos_history)}/"
                        f"{self._anchor_actual_min_history}, "
                        f"thresh={self._anchor_actual_speed_threshold:.3f} m/s)"
                    )
            else:
                self._start_aligned_gt_base_pos = base_pos.copy()
                self._start_aligned_actual_base_pos = actual_root_pos_now.copy()
                self._start_aligned_gt_yaw = self._quat_to_yaw(base_quat)
                self._start_aligned_actual_yaw = self._get_actual_robot_root_yaw()
                self._start_aligned_yaw_delta = self._wrap_to_pi(
                    self._start_aligned_actual_yaw - self._start_aligned_gt_yaw
                )
                print(
                    "[ReferenceMotionVisualizer] locked start-aligned XY origin "
                    f"gt=({self._start_aligned_gt_base_pos[0]:.3f}, {self._start_aligned_gt_base_pos[1]:.3f}) "
                    f"actual=({self._start_aligned_actual_base_pos[0]:.3f}, {self._start_aligned_actual_base_pos[1]:.3f}) "
                    f"yaw_delta_deg={np.degrees(self._start_aligned_yaw_delta):.2f}"
                )
        if synchronized and self.translation_mode == "delayed_align" and self._start_aligned_gt_base_pos is None:
            self._delayed_align_frame_count += 1
            delay = int(self._align_delay_frames)
            if self._delayed_align_frame_count <= delay:
                if self._delayed_align_frame_count == 1:
                    print(
                        f"[ReferenceMotionVisualizer] delayed_align: waiting {delay} frames "
                        f"before locking anchor"
                    )
            if self._delayed_align_frame_count >= delay:
                actual_root_pos_now = self._get_actual_robot_root_pos()
                self._start_aligned_gt_base_pos = base_pos.copy()
                self._start_aligned_actual_base_pos = actual_root_pos_now.copy()
                self._start_aligned_gt_yaw = self._quat_to_yaw(base_quat)
                self._start_aligned_actual_yaw = self._get_actual_robot_root_yaw()
                self._start_aligned_yaw_delta = self._wrap_to_pi(
                    self._start_aligned_actual_yaw - self._start_aligned_gt_yaw
                )
                print(
                    f"[ReferenceMotionVisualizer] locked delayed_align anchor "
                    f"at frame={self._delayed_align_frame_count} "
                    f"gt=({self._start_aligned_gt_base_pos[0]:.3f}, {self._start_aligned_gt_base_pos[1]:.3f}) "
                    f"actual=({self._start_aligned_actual_base_pos[0]:.3f}, {self._start_aligned_actual_base_pos[1]:.3f}) "
                    f"yaw_delta_deg={np.degrees(self._start_aligned_yaw_delta):.2f}"
                )
        if self.translation_mode != "delta_aligned" or self._translation_anchor_ready:
            self._maybe_refresh_translation_anchor(base_pos)
        now = time.monotonic()
        if now - self._last_pose_log_time >= 1.0:
            self._last_pose_log_time = now
            print(
                "[ReferenceMotionVisualizer] root pose "
                f"x={base_pos[0]:.3f} y={base_pos[1]:.3f} z={base_pos[2]:.3f}"
            )
        if not self._shown_once and self._is_anchor_ready():
            self._shown_once = True
            self.set_visible(True)
            print("[ReferenceMotionVisualizer] reference pose stream detected")

    def _advance_display_pose(self, force_pose: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None):
        target_frame_index = None
        if self._latest_debug_source_frame_index is not None:
            target_frame_index = max(
                0,
                int(self._latest_debug_source_frame_index) - int(self._display_debug_lag_frames),
            )
        if target_frame_index is not None:
            pose = self._get_buffered_pose_for_frame(target_frame_index)
            if pose is not None:
                self._display_frame_index = target_frame_index
                self._display_frame_phase = 0.0
                self._set_latest_pose(*pose, synchronized=True)
                return

        if force_pose is not None and self._display_frame_index is None:
            self._set_latest_pose(*force_pose, synchronized=True)
            return
        if not self._pose_frame_buffer:
            if force_pose is not None:
                self._set_latest_pose(*force_pose, synchronized=True)
            return

        now = time.monotonic()
        elapsed = max(0.0, now - self._last_display_update_time)
        self._last_display_update_time = now
        self._display_frame_phase += elapsed * self._display_fps

        sorted_keys = list(self._pose_frame_buffer.keys())
        if self._display_frame_index is None:
            self._display_frame_index = int(sorted_keys[0])

        steps = int(self._display_frame_phase)
        if steps > 0:
            self._display_frame_phase -= steps
            latest_available = int(sorted_keys[-1])
            self._display_frame_index = min(self._display_frame_index + steps, latest_available)

        pose = self._get_buffered_pose_for_frame(int(self._display_frame_index))
        if pose is None and force_pose is not None:
            pose = force_pose
        if pose is not None:
            self._set_latest_pose(*pose, synchronized=True)

    def apply(self):
        if self._current_display_pose is None or not self.enabled:
            return False

        base_pos, base_quat, body_q = self._current_display_pose
        base_pos, base_quat = self._transform_reference_root_pose(base_pos, base_quat)
        if base_pos is None:
            return False
        self.mj_data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = base_pos
        self.mj_data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = base_quat
        self.mj_data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6] = 0.0
        self.mj_data.qpos[self.body_qpos_adrs] = body_q
        if self.ref_hand_qpos_adrs.size > 0 and self.actual_hand_qpos_adrs.size > 0:
            self.mj_data.qpos[self.ref_hand_qpos_adrs] = self.mj_data.qpos[self.actual_hand_qpos_adrs]
        self.mj_data.qvel[self.body_qvel_adrs] = 0.0
        self._log_reference_debug_state(base_pos)
        return True

    def _compute_reference_body_pos(
        self,
        body_ids: np.ndarray,
        apply_visual_alignment: bool,
    ) -> tuple[int | None, np.ndarray | None]:
        if self._current_applied_source_frame_index is None or self._current_exact_pose is None:
            return None, None

        base_pos, base_quat, body_q = self._current_exact_pose
        if apply_visual_alignment:
            root_pos, root_quat = self._transform_reference_root_pose(base_pos, base_quat)
            if root_pos is None:
                return None, None
        else:
            root_pos = np.asarray(base_pos, dtype=np.float64).copy()
            root_quat = np.asarray(base_quat, dtype=np.float64).copy()
        saved_root_qpos = self.mj_data.qpos[self.root_qpos_adr : self.root_qpos_adr + 7].copy()
        saved_root_qvel = self.mj_data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6].copy()
        saved_body_qpos = self.mj_data.qpos[self.body_qpos_adrs].copy()
        saved_body_qvel = self.mj_data.qvel[self.body_qvel_adrs].copy()
        saved_hand_qpos = None
        if self.ref_hand_qpos_adrs.size > 0:
            saved_hand_qpos = self.mj_data.qpos[self.ref_hand_qpos_adrs].copy()

        try:
            self.mj_data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = root_pos
            self.mj_data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = root_quat
            self.mj_data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6] = 0.0
            self.mj_data.qpos[self.body_qpos_adrs] = body_q
            self.mj_data.qvel[self.body_qvel_adrs] = 0.0
            if self.ref_hand_qpos_adrs.size > 0 and self.actual_hand_qpos_adrs.size > 0:
                self.mj_data.qpos[self.ref_hand_qpos_adrs] = self.mj_data.qpos[self.actual_hand_qpos_adrs]
            mujoco.mj_forward(self.mj_model, self.mj_data)
            body_pos = np.asarray(self.mj_data.xpos[body_ids], dtype=np.float64).copy()
        finally:
            self.mj_data.qpos[self.root_qpos_adr : self.root_qpos_adr + 7] = saved_root_qpos
            self.mj_data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6] = saved_root_qvel
            self.mj_data.qpos[self.body_qpos_adrs] = saved_body_qpos
            self.mj_data.qvel[self.body_qvel_adrs] = saved_body_qvel
            if saved_hand_qpos is not None:
                self.mj_data.qpos[self.ref_hand_qpos_adrs] = saved_hand_qpos
            mujoco.mj_forward(self.mj_model, self.mj_data)

        return self._current_applied_source_frame_index, body_pos

    def compute_exact_reference_body_pos(
        self,
        body_ids: np.ndarray,
        ref_body_ids: np.ndarray | None = None,
    ) -> tuple[int | None, np.ndarray | None]:
        target_body_ids = ref_body_ids if ref_body_ids is not None else body_ids
        return self._compute_reference_body_pos(target_body_ids, apply_visual_alignment=True)

    def get_control_tick_reference_pose(
        self,
    ) -> tuple[int | None, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
        if self._latest_control_tick_reference_pose is None:
            return None, None
        source_frame_index, pose = self._latest_control_tick_reference_pose
        return int(source_frame_index), tuple(np.asarray(x, dtype=np.float64).copy() for x in pose)

    def _get_actual_robot_root_pos(self) -> np.ndarray:
        return self.mj_data.qpos[:3].copy()

    def _is_actual_root_static(self) -> bool:
        if len(self._actual_root_pos_history) < self._anchor_actual_min_history:
            return False
        history = np.asarray(self._actual_root_pos_history, dtype=np.float64)
        diffs = np.linalg.norm(np.diff(history, axis=0), axis=1)
        max_speed = float(np.max(diffs)) / max(self._actual_root_history_dt, 1e-6)
        return max_speed < self._anchor_actual_speed_threshold

    def _is_anchor_ready(self) -> bool:
        if self.translation_mode == "delta_aligned":
            return bool(self._translation_anchor_ready)
        if self.translation_mode in ("start_aligned_xy", "delayed_align"):
            return self._start_aligned_gt_base_pos is not None
        return True

    def _get_actual_robot_root_yaw(self) -> float:
        return self._quat_to_yaw(np.asarray(self.mj_data.qpos[3:7], dtype=np.float64))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _quat_to_yaw(quat_wxyz: np.ndarray) -> float:
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        rot = Rotation.from_quat(quat, scalar_first=True)
        return float(rot.as_euler("zyx")[0])

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        return Rotation.from_euler("z", yaw).as_quat(scalar_first=True).astype(np.float64)

    def _apply_start_aligned_yaw_to_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        if self._start_aligned_yaw_delta is None:
            return np.asarray(quat_wxyz, dtype=np.float64).copy()
        rot_delta = Rotation.from_euler("z", self._start_aligned_yaw_delta)
        rot_in = Rotation.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True)
        return (rot_delta * rot_in).as_quat(scalar_first=True).astype(np.float64)

    def _maybe_refresh_translation_anchor(self, target_base_pos: np.ndarray):
        if self.translation_mode != "delta_aligned":
            self._prev_target_base_pos = target_base_pos.copy()
            return
        if not self.allow_midrun_realign:
            self._prev_target_base_pos = target_base_pos.copy()
            return
        if self._target_anchor_base_pos is None or self._actual_anchor_base_pos is None:
            self._target_anchor_base_pos = target_base_pos.copy()
            self._actual_anchor_base_pos = self._get_actual_robot_root_pos()
            self._prev_target_base_pos = target_base_pos.copy()
            return

        # If the streamed target snaps back near its initial root after having moved
        # away, treat that as a new clip start and re-anchor to the current actual pose.
        prev_target = self._prev_target_base_pos
        if prev_target is not None:
            moved_away = np.linalg.norm(prev_target - self._target_anchor_base_pos) > 0.15
            returned_to_start = np.linalg.norm(target_base_pos - self._target_anchor_base_pos) < 0.03
            if moved_away and returned_to_start:
                self._target_anchor_base_pos = target_base_pos.copy()
                self._actual_anchor_base_pos = self._get_actual_robot_root_pos()
                print("[ReferenceMotionVisualizer] re-anchored reference root translation")
        self._prev_target_base_pos = target_base_pos.copy()

    def _transform_reference_root_pose(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        out_pos = base_pos.copy()
        out_quat = np.asarray(base_quat, dtype=np.float64).copy()
        if self.translation_mode == "delta_aligned":
            if not self._translation_anchor_ready:
                return None, out_quat
            if self._target_anchor_base_pos is None or self._actual_anchor_base_pos is None:
                self._target_anchor_base_pos = base_pos.copy()
                self._actual_anchor_base_pos = self._get_actual_robot_root_pos()
            return self._actual_anchor_base_pos + (base_pos - self._target_anchor_base_pos), out_quat
        if self.translation_mode in ("start_aligned_xy", "delayed_align"):
            if (
                self._start_aligned_gt_base_pos is None
                or self._start_aligned_actual_base_pos is None
                or self._start_aligned_yaw_delta is None
            ):
                return None, out_quat
            rel_xy = out_pos[:2] - self._start_aligned_gt_base_pos[:2]
            c = float(np.cos(self._start_aligned_yaw_delta))
            s = float(np.sin(self._start_aligned_yaw_delta))
            rot_rel_xy = np.array(
                [
                    c * rel_xy[0] - s * rel_xy[1],
                    s * rel_xy[0] + c * rel_xy[1],
                ],
                dtype=np.float64,
            )
            out_pos[:2] = self._start_aligned_actual_base_pos[:2] + rot_rel_xy
            out_quat = self._apply_start_aligned_yaw_to_quat(out_quat)
            return out_pos, out_quat
        return out_pos, out_quat

    def _transform_reference_base_pos(self, base_pos: np.ndarray) -> np.ndarray | None:
        out = base_pos.copy()
        if self.translation_mode == "delta_aligned":
            if not self._translation_anchor_ready:
                return None
            if self._target_anchor_base_pos is None or self._actual_anchor_base_pos is None:
                self._target_anchor_base_pos = base_pos.copy()
                self._actual_anchor_base_pos = self._get_actual_robot_root_pos()
            return self._actual_anchor_base_pos + (base_pos - self._target_anchor_base_pos)
        if self.translation_mode in ("start_aligned_xy", "delayed_align"):
            if self._start_aligned_gt_base_pos is None or self._start_aligned_actual_base_pos is None:
                return None
            offset_xy = self._start_aligned_actual_base_pos[:2] - self._start_aligned_gt_base_pos[:2]
            out[:2] = out[:2] + offset_xy
            return out
        return out

    def _log_reference_debug_state(self, applied_base_pos: np.ndarray):
        now = time.monotonic()
        if now - self._last_debug_visibility_log_time < 1.0:
            return
        self._last_debug_visibility_log_time = now

        actual_root_pos = np.asarray(self.mj_data.xpos[self.actual_root_body_id], dtype=np.float64).copy()
        ref_root_pos_scene = np.asarray(self.mj_data.xpos[self.ref_root_body_id], dtype=np.float64).copy()
        dist = float(np.linalg.norm(ref_root_pos_scene - actual_root_pos))
        alpha_mean = float(np.mean(self.mj_model.geom_rgba[self.ref_geom_ids, 3])) if self.ref_geom_ids.size > 0 else 0.0
        print(
            "[ReferenceMotionVisualizer] debug "
            f"visible={self.visible} alpha_mean={alpha_mean:.3f} "
            f"source_frame={self._current_applied_source_frame_index} "
            f"actual_root=({actual_root_pos[0]:.3f}, {actual_root_pos[1]:.3f}, {actual_root_pos[2]:.3f}) "
            f"ref_root_scene=({ref_root_pos_scene[0]:.3f}, {ref_root_pos_scene[1]:.3f}, {ref_root_pos_scene[2]:.3f}) "
            f"ref_root_applied=({applied_base_pos[0]:.3f}, {applied_base_pos[1]:.3f}, {applied_base_pos[2]:.3f}) "
            f"root_dist={dist:.4f}"
        )


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen
        self.reference_visualizer = None
        self.generated_scene_path = None
        self._sim2sim_eval_logger = None
        self._last_sim_step_perf_log_time = 0.0

        self.init_scene()
        self._init_sim2sim_eval_logger()
        self._link_error_plot: Sim2SimLinkErrorPlot | None = None
        self._plot_link_indices: list[int] | None = None
        self._init_link_error_plot()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _init_sim2sim_eval_logger(self):
        if not self.config.get("ENABLE_SIM2SIM_EVAL_LOGGING", False):
            return
        logs_dir = Path(self.config.get("SIM2SIM_EVAL_LOGS_DIR", "/tmp/sonic_logs/official_walk_zmq01"))
        self._sim2sim_eval_logger = Sim2SimEvalLogger(logs_dir=logs_dir, body_names=SIM2SIM_BODY_FRAMES)

    def _init_link_error_plot(self):
        if not self.config.get("ENABLE_SIM2SIM_ERROR_PLOT", False):
            return
        plot_links: list[str] = self.config.get("SIM2SIM_ERROR_PLOT_LINKS") or []
        if not plot_links:
            return
        valid_names = set(SIM2SIM_BODY_FRAMES)
        unknown = [ln for ln in plot_links if ln not in valid_names]
        if unknown:
            print(f"[LinkErrorPlot] Warning: unknown link names ignored: {unknown}")
        indices = [SIM2SIM_BODY_FRAMES.index(ln) for ln in plot_links if ln in valid_names]
        valid_links = [SIM2SIM_BODY_FRAMES[i] for i in indices]
        if not valid_links:
            return
        self._plot_link_indices = indices
        self._link_error_plot = Sim2SimLinkErrorPlot(
            link_names=valid_links,
            ymax_mm=float(self.config.get("SIM2SIM_ERROR_PLOT_YMAX_MM", 300.0)),
            refresh_hz=float(self.config.get("SIM2SIM_ERROR_PLOT_REFRESH_HZ", 20.0)),
        )
        self._link_error_plot.start()
        print(f"[LinkErrorPlot] Started for links: {valid_links}")

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"]
        xml_path = self._maybe_create_reference_visualization_scene(xml_path)
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        self.elastic_band = None
        self.band_attached_link = None
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id
            try:
                self.viewer.user_scn.ngeom = 0
            except Exception:
                pass

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        self.body_joint_names = []
        self.left_hand_joint_names = []
        self.right_hand_joint_names = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if name.startswith(REFERENCE_NAME_PREFIX):
                continue
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
                self.body_joint_names.append(name)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
                self.left_hand_joint_names.append(name)
            elif "right_hand" in name:
                self.right_hand_index.append(i)
                self.right_hand_joint_names.append(name)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

        self._init_reference_visualizer()
        self._tracking_overlay = (
            Sim2SimTrackingOverlay(self.viewer, SIM2SIM_BODY_FRAMES) if self.viewer is not None else None
        )

    def _maybe_create_reference_visualization_scene(self, xml_path: pathlib.Path) -> str:
        if not self.config.get("ENABLE_REFERENCE_MOTION_VISUALIZATION", False):
            return str(xml_path)

        scene_tree = ET.parse(xml_path)
        scene_root = scene_tree.getroot()
        scene_dir = xml_path.parent

        include_elements = scene_root.findall("include")
        if not include_elements:
            print(
                "[ReferenceMotionVisualizer] no <include> found in scene XML; "
                "reference visualization disabled"
            )
            return str(xml_path)

        include_path = None
        for include_element in include_elements:
            include_file = include_element.get("file")
            if include_file is None:
                continue
            resolved = (scene_dir / include_file).resolve()
            include_element.set("file", str(resolved))
            if include_path is None:
                include_path = resolved

        if include_path is None:
            print(
                "[ReferenceMotionVisualizer] failed to resolve robot include; "
                "reference visualization disabled"
            )
            return str(xml_path)

        robot_tree = ET.parse(include_path)
        robot_body = robot_tree.getroot().find("./worldbody/body")
        scene_worldbody = scene_root.find("./worldbody")
        if robot_body is None or scene_worldbody is None:
            print(
                "[ReferenceMotionVisualizer] robot/worldbody missing from XML; "
                "reference visualization disabled"
            )
            return str(xml_path)

        ref_body = deepcopy(robot_body)
        self._prefix_reference_subtree(ref_body)
        scene_worldbody.append(ref_body)

        fd, temp_path = tempfile.mkstemp(
            prefix="mujoco_reference_scene_",
            suffix=".xml",
            dir=scene_dir,
        )
        os.close(fd)
        scene_tree.write(temp_path, encoding="utf-8", xml_declaration=False)
        self.generated_scene_path = temp_path
        return temp_path

    def _prefix_reference_subtree(self, root: ET.Element):
        alpha = float(np.clip(self.config.get("REFERENCE_MOTION_ALPHA", 0.35), 0.0, 1.0))
        for element in root.iter():
            if "name" in element.attrib:
                element.set("name", f"{REFERENCE_NAME_PREFIX}{element.get('name')}")
            if element.tag != "geom":
                continue
            element.set("contype", "0")
            element.set("conaffinity", "0")
            rgba_str = element.get("rgba")
            if rgba_str is None:
                rgb = np.array([0.88, 0.88, 0.88], dtype=np.float64)
            else:
                rgba = np.fromstring(rgba_str, sep=" ", dtype=np.float64)
                rgb = rgba[:3] if rgba.size >= 3 else np.array([0.7, 0.7, 0.7], dtype=np.float64)
            ref_rgb = np.array([1.0, 0.15, 0.15], dtype=np.float64)
            element.set(
                "rgba",
                f"{ref_rgb[0]:.6f} {ref_rgb[1]:.6f} {ref_rgb[2]:.6f} {alpha:.6f}",
            )

    def _init_reference_visualizer(self):
        if not self.config.get("ENABLE_REFERENCE_MOTION_VISUALIZATION", False):
            return
        try:
            self.reference_visualizer = ReferenceMotionVisualizer(
                mj_model=self.mj_model,
                mj_data=self.mj_data,
                body_joint_names=self.body_joint_names,
                left_hand_joint_names=self.left_hand_joint_names,
                right_hand_joint_names=self.right_hand_joint_names,
                alpha=self.config.get("REFERENCE_MOTION_ALPHA", 0.35),
                host=self.config.get("REFERENCE_MOTION_ZMQ_HOST", "127.0.0.1"),
                port=int(self.config.get("REFERENCE_MOTION_ZMQ_PORT", 5608)),
                topic=self.config.get("REFERENCE_MOTION_ZMQ_TOPIC", "g1_debug"),
                pose_port=int(self.config.get("REFERENCE_MOTION_POSE_ZMQ_PORT", 5596)),
                pose_topic=self.config.get("REFERENCE_MOTION_POSE_ZMQ_TOPIC", "pose"),
                allow_midrun_realign=bool(
                    self.config.get("REFERENCE_MOTION_ALLOW_MIDRUN_REALIGN", False)
                ),
                translation_mode=self.config.get(
                    "REFERENCE_MOTION_TRANSLATION_MODE", "delayed_align"
                ),
                align_delay_frames=int(self.config.get(
                    "REFERENCE_MOTION_ALIGN_DELAY_FRAMES", 50
                )),
            )
        except Exception as exc:
            self.reference_visualizer = None
            print(f"[ReferenceMotionVisualizer] disabled: {exc}")

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        step_wall_start = time.monotonic()
        is_new_control_frame = False
        if self.unitree_bridge is not None:
            try:
                _, _, is_new_control_frame = self.unitree_bridge.GetAction()
            except Exception:
                is_new_control_frame = False
        t_after_get_action = time.monotonic()
        self.obs = self.prepare_obs()
        t_after_prepare_obs = time.monotonic()
        self.unitree_bridge.PublishLowState(self.obs)
        t_after_publish_lowstate = time.monotonic()
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        t_after_mj_step = time.monotonic()
        actual_body_pos_pre_ref = None
        if self._sim2sim_eval_logger is not None:
            self._sim2sim_eval_logger._resolve_body_ids(self.mj_model)
            actual_body_pos_pre_ref = np.asarray(
                self.mj_data.xpos[self._sim2sim_eval_logger.body_ids],
                dtype=np.float64,
            ).copy()
        reference_pose_applied = self.update_reference_motion_visualization()
        if reference_pose_applied:
            mujoco.mj_forward(self.mj_model, self.mj_data)

        self._log_sim2sim_eval_frame(
            actual_body_pos=actual_body_pos_pre_ref,
            write_step_sync=bool(is_new_control_frame),
        )
        t_after_logging = time.monotonic()

        self.check_fall()
        t_after_check_fall = time.monotonic()

        now = t_after_check_fall
        if now - self._last_sim_step_perf_log_time >= 1.0:
            self._last_sim_step_perf_log_time = now
            print(
                "[BaseSim] sim_step heartbeat "
                f"sim_time={float(self.mj_data.time):.3f} "
                f"is_new_control_frame={int(bool(is_new_control_frame))} "
                f"get_action_ms={(t_after_get_action - step_wall_start) * 1e3:.2f} "
                f"prepare_obs_ms={(t_after_prepare_obs - t_after_get_action) * 1e3:.2f} "
                f"publish_lowstate_ms={(t_after_publish_lowstate - t_after_prepare_obs) * 1e3:.2f} "
                f"mj_step_ms={(t_after_mj_step - t_after_publish_lowstate) * 1e3:.2f} "
                f"post_step_ms={(t_after_check_fall - t_after_mj_step) * 1e3:.2f} "
                f"total_ms={(t_after_check_fall - step_wall_start) * 1e3:.2f}"
            )

    def _log_sim2sim_eval_frame(
        self,
        actual_body_pos: np.ndarray | None = None,
        write_step_sync: bool = True,
    ):
        if self._sim2sim_eval_logger is None and self._tracking_overlay is None:
            return

        body_ids = None
        ref_body_ids = None
        if self._sim2sim_eval_logger is not None:
            self._sim2sim_eval_logger._resolve_body_ids(self.mj_model)
            body_ids = self._sim2sim_eval_logger.body_ids
            ref_body_ids = self._sim2sim_eval_logger.ref_body_ids
        elif self._tracking_overlay is not None:
            self._tracking_overlay._resolve_body_ids(self.mj_model)
            body_ids = self._tracking_overlay._body_ids

        if actual_body_pos is None and body_ids is not None:
            actual_body_pos = np.asarray(
                self.mj_data.xpos[body_ids],
                dtype=np.float64,
            ).copy()
        source_frame_index = None
        ref_body_pos = None
        if self.reference_visualizer is not None and body_ids is not None:
            source_frame_index, ref_body_pos = self.reference_visualizer.compute_exact_reference_body_pos(
                body_ids, ref_body_ids=ref_body_ids
            )

        if self._tracking_overlay is not None:
            self._tracking_overlay.update(
                mj_model=self.mj_model,
                actual_body_pos=actual_body_pos,
                ref_body_pos=ref_body_pos,
                source_frame_index=source_frame_index,
            )

        if (
            self._link_error_plot is not None
            and actual_body_pos is not None
            and ref_body_pos is not None
            and source_frame_index is not None
            and int(source_frame_index) >= 0
        ):
            full_err_mm = np.linalg.norm(actual_body_pos - ref_body_pos, axis=1) * 1000.0
            selected_err = full_err_mm[self._plot_link_indices]
            self._link_error_plot.push(int(source_frame_index), selected_err)

        if self._sim2sim_eval_logger is None:
            return
        self._sim2sim_eval_logger.log(
            mj_model=self.mj_model,
            mj_data=self.mj_data,
            source_frame_index=source_frame_index,
            actual_body_pos=actual_body_pos,
            ref_body_pos=ref_body_pos,
            write_step_sync=write_step_sync,
        )

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            if self._tracking_overlay is not None:
                self._tracking_overlay.render()
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "r" and self.reference_visualizer is not None:
            self.reference_visualizer.toggle()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def update_reference_motion_visualization(self):
        if self.reference_visualizer is None:
            return False
        self.reference_visualizer.poll()
        return self.reference_visualizer.apply()

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.reference_visualizer is not None:
            self.reference_visualizer.reset_anchor()

    def close(self):
        if self._sim2sim_eval_logger is not None:
            self._sim2sim_eval_logger.close()
            self._sim2sim_eval_logger = None
        if self._link_error_plot is not None:
            self._link_error_plot.close()
            self._link_error_plot = None
        if self.reference_visualizer is not None:
            self.reference_visualizer.close()
            self.reference_visualizer = None
        if self.generated_scene_path and os.path.exists(self.generated_scene_path):
            os.remove(self.generated_scene_path)
            self.generated_scene_path = None


class Sim2SimEvalLogger:
    def __init__(self, logs_dir: Path, body_names: list[str]):
        self.logs_dir = logs_dir
        self.body_names = body_names
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._row_index = 0
        self._body_ids = None
        self._ref_body_ids = None
        self._last_step_sync_source_frame_index = None

        self._body_pos_file = open(self.logs_dir / "body_pos_w_14.csv", "w", newline="", encoding="utf-8")
        self._source_frame_file = open(
            self.logs_dir / "sim_source_frame_index.csv", "w", newline="", encoding="utf-8"
        )
        self._step_sync_file = open(
            self.logs_dir / "sim2sim_step_sync_body_pos_w_14.csv", "w", newline="", encoding="utf-8"
        )
        self._body_pos_writer = csv.writer(self._body_pos_file)
        self._source_frame_writer = csv.writer(self._source_frame_file)
        self._step_sync_writer = csv.writer(self._step_sync_file)

        body_header = ["index", "sim_time"]
        for name in self.body_names:
            body_header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        self._body_pos_writer.writerow(body_header)
        self._source_frame_writer.writerow(["index", "sim_time", "source_frame_index"])
        step_sync_header = ["index", "sim_time", "source_frame_index"]
        for prefix in ("actual", "ref"):
            for name in self.body_names:
                step_sync_header.extend([f"{prefix}_{name}_x", f"{prefix}_{name}_y", f"{prefix}_{name}_z"])
        self._step_sync_writer.writerow(step_sync_header)
        print(f"[Sim2SimEvalLogger] writing body_pos_w_14.csv: {self.logs_dir / 'body_pos_w_14.csv'}")
        print(
            "[Sim2SimEvalLogger] writing sim2sim_step_sync_body_pos_w_14.csv: "
            f"{self.logs_dir / 'sim2sim_step_sync_body_pos_w_14.csv'}"
        )
        print(
            "[Sim2SimEvalLogger] writing sim_source_frame_index.csv: "
            f"{self.logs_dir / 'sim_source_frame_index.csv'}"
        )

    def _resolve_body_ids(self, mj_model: mujoco.MjModel):
        if self._body_ids is not None:
            return
        body_ids = []
        ref_body_ids = []
        for name in self.body_names:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                raise ValueError(f"MuJoCo body not found for sim2sim eval logging: {name}")
            body_ids.append(body_id)
            ref_name = f"{REFERENCE_NAME_PREFIX}{name}"
            ref_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, ref_name)
            ref_body_ids.append(ref_body_id)
        self._body_ids = np.asarray(body_ids, dtype=np.int32)
        if all(bid != -1 for bid in ref_body_ids):
            self._ref_body_ids = np.asarray(ref_body_ids, dtype=np.int32)
        else:
            self._ref_body_ids = None

    @property
    def body_ids(self) -> np.ndarray | None:
        return self._body_ids

    @property
    def ref_body_ids(self) -> np.ndarray | None:
        return self._ref_body_ids

    def log(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        source_frame_index: int | None,
        actual_body_pos: np.ndarray,
        ref_body_pos: np.ndarray | None,
        write_step_sync: bool = True,
    ):
        self._resolve_body_ids(mj_model)
        row = [self._row_index, float(mj_data.time)]
        row.extend(np.asarray(actual_body_pos, dtype=np.float64).reshape(-1).tolist())
        self._body_pos_writer.writerow(row)
        source_frame_value = int(source_frame_index) if source_frame_index is not None else -1
        self._source_frame_writer.writerow(
            [
                self._row_index,
                float(mj_data.time),
                source_frame_value,
            ]
        )
        should_write_step_sync = (
            bool(write_step_sync)
            and
            source_frame_value >= 0
            and ref_body_pos is not None
            and source_frame_value != self._last_step_sync_source_frame_index
        )
        if should_write_step_sync:
            step_sync_row = [self._row_index, float(mj_data.time), source_frame_value]
            step_sync_row.extend(np.asarray(actual_body_pos, dtype=np.float64).reshape(-1).tolist())
            step_sync_row.extend(np.asarray(ref_body_pos, dtype=np.float64).reshape(-1).tolist())
            self._step_sync_writer.writerow(step_sync_row)
            self._last_step_sync_source_frame_index = source_frame_value
        self._row_index += 1

    def close(self):
        self._body_pos_file.close()
        self._source_frame_file.close()
        self._step_sync_file.close()


class Sim2SimTrackingOverlay:
    def __init__(self, viewer, body_names: list[str]):
        self.viewer = viewer
        self.body_names = body_names
        self._body_ids = None
        self.last_actual_body_pos = None
        self.last_ref_body_pos = None
        self.last_source_frame_index = None
        self.last_mean_error_mm = None
        self.last_max_error_mm = None

    def _resolve_body_ids(self, mj_model: mujoco.MjModel):
        if self._body_ids is not None:
            return
        body_ids = []
        for name in self.body_names:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                raise ValueError(f"MuJoCo body not found for tracking overlay: {name}")
            body_ids.append(body_id)
        self._body_ids = np.asarray(body_ids, dtype=np.int32)

    def update(
        self,
        mj_model: mujoco.MjModel,
        actual_body_pos: np.ndarray | None,
        ref_body_pos: np.ndarray | None,
        source_frame_index: int | None,
    ):
        self._resolve_body_ids(mj_model)
        self.last_source_frame_index = source_frame_index
        self.last_actual_body_pos = (
            None if actual_body_pos is None else np.asarray(actual_body_pos, dtype=np.float64).copy()
        )
        self.last_ref_body_pos = (
            None if ref_body_pos is None else np.asarray(ref_body_pos, dtype=np.float64).copy()
        )
        if self.last_actual_body_pos is None or self.last_ref_body_pos is None:
            self.last_mean_error_mm = None
            self.last_max_error_mm = None
            return
        err = np.linalg.norm(self.last_actual_body_pos - self.last_ref_body_pos, axis=1) * 1000.0
        self.last_mean_error_mm = float(np.mean(err)) if err.size > 0 else None
        self.last_max_error_mm = float(np.max(err)) if err.size > 0 else None

    def render(self):
        if self.viewer is None or self.last_actual_body_pos is None or self.last_ref_body_pos is None:
            return
        try:
            scn = self.viewer.user_scn
        except AttributeError:
            return
        scn.ngeom = 0

        err_vec = self.last_ref_body_pos - self.last_actual_body_pos
        err_mm = np.linalg.norm(err_vec, axis=1) * 1000.0
        max_geoms = max(0, int(getattr(scn, "maxgeom", 0)) - 1)
        num_lines = min(len(err_mm), max_geoms)
        for i in range(num_lines):
            if err_mm[i] <= 1e-9:
                continue
            geom = scn.geoms[scn.ngeom]
            red = float(np.clip(err_mm[i] / 30.0, 0.0, 1.0))
            green = float(np.clip(1.0 - err_mm[i] / 30.0, 0.0, 1.0))
            rgba = np.array([red, green, 0.1, 0.95], dtype=np.float32)
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                0.006,
                self.last_actual_body_pos[i],
                self.last_ref_body_pos[i],
            )
            scn.ngeom += 1

        if scn.ngeom < getattr(scn, "maxgeom", 0):
            geom = scn.geoms[scn.ngeom]
            if self.last_mean_error_mm is None:
                sphere_pos = self.last_actual_body_pos[0]
                rgba = np.array([0.8, 0.8, 0.8, 0.0], dtype=np.float32)
                size = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            else:
                sphere_pos = self.last_actual_body_pos[0]
                max_err = float(self.last_max_error_mm or 0.0)
                rgba = np.array(
                    [
                        float(np.clip(max_err / 30.0, 0.0, 1.0)),
                        float(np.clip(1.0 - max_err / 30.0, 0.0, 1.0)),
                        0.2,
                        0.9,
                    ],
                    dtype=np.float32,
                )
                size = np.array([0.015, 0.015, 0.015], dtype=np.float64)
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                sphere_pos,
                np.eye(3, dtype=np.float64).reshape(-1),
                rgba,
            )
            scn.ngeom += 1

        try:
            self.viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "sim2sim tracking",
                (
                    f"src={self.last_source_frame_index} | "
                    f"mean={0.0 if self.last_mean_error_mm is None else self.last_mean_error_mm:.3f} mm | "
                    f"max={0.0 if self.last_max_error_mm is None else self.last_max_error_mm:.3f} mm"
                ),
            )
        except Exception:
            pass

        try:
            if self.last_actual_body_pos is None or self.last_ref_body_pos is None:
                return
            err_mm = np.linalg.norm(self.last_actual_body_pos - self.last_ref_body_pos, axis=1) * 1000.0
            left_lines = []
            right_lines = []
            for name, err in zip(self.body_names, err_mm.tolist()):
                left_lines.append(name)
                right_lines.append(f"{err:7.3f} mm")
            self.viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "\n".join(left_lines),
                "\n".join(right_lines),
            )
        except Exception:
            pass


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        # ChannelFactory is initialized in run_sim_loop.py via
        # gear_sonic.utils.mujoco_sim.simulator_factory.init_channel().
        # Re-initializing here can trigger CycloneDDS domain conflicts.

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            self.sim_env.close()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
