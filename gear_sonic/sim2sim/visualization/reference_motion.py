"""Reference robot visualization driven from sim2sim pose/debug streams."""

import time
from collections import OrderedDict, deque

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.sim2sim.constants import G1_ISAACLAB_TO_MUJOCO_DOF, REFERENCE_NAME_PREFIX
from gear_sonic.sim2sim.protocol.packed_zmq import PackedZMQSubscriber
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_geom_ids


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
        self._align_delay_frames = int(align_delay_frames)  # 0 = auto-detect from stream
        self._align_delay_frames_auto = (self._align_delay_frames <= 0)
        self._motion_start_frame_received = False
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
        self._last_missing_debug_sync_log_time = 0.0
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
        self._last_missing_debug_sync_log_time = 0.0
        self._motion_start_frame_received = False
        if self._align_delay_frames_auto:
            self._align_delay_frames = 0
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
            latest = self._get_latest_buffered_pose_with_index()
            if latest is not None:
                frame_index, pose = latest
                self._set_latest_pose(*pose, synchronized=True)
                self._current_applied_source_frame_index = int(frame_index)
                self._current_exact_pose = tuple(np.asarray(x, dtype=np.float64).copy() for x in pose)
                self._log_missing_debug_sync_warning()
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

        if (
            self._align_delay_frames_auto
            and not self._motion_start_frame_received
            and "motion_start_frame" in msg
        ):
            self._motion_start_frame_received = True
            msf = int(msg["motion_start_frame"])
            self._align_delay_frames = max(1, msf) if msf > 0 else 50
            print(
                f"[ReferenceMotionVisualizer] auto align_delay_frames={self._align_delay_frames}"
                f" (motion_start_frame={msf} from stream)"
            )

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

    def _get_latest_buffered_pose_with_index(self):
        if not self._pose_frame_buffer:
            return None
        frame_index = next(reversed(self._pose_frame_buffer.keys()))
        return int(frame_index), self._pose_frame_buffer[frame_index]

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
            delay = int(self._align_delay_frames) if self._align_delay_frames > 0 else 50
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

    def _log_missing_debug_sync_warning(self):
        now = time.monotonic()
        if now - self._last_missing_debug_sync_log_time < 5.0:
            return
        self._last_missing_debug_sync_log_time = now
        print(
            "[ReferenceMotionVisualizer] raw pose stream is visible, but deploy "
            "source-frame debug has not been received. Reference visualization is "
            "using pose-stream timing fallback; pass --enable-sim2sim-debug to deploy "
            "for strict source-frame sync and error plots."
        )

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
