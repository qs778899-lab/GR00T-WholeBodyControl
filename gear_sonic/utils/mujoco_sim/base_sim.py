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
import json
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names, get_subtree_geom_ids
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REFERENCE_NAME_PREFIX = "ref_"
PACKED_ZMQ_HEADER_SIZE = 1280


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
    """Drive a translucent reference robot from the pose stream or deploy debug pose."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        body_joint_names: list[str],
        alpha: float,
        host: str,
        port: int,
        topic: str,
        translation_mode: str,
        pose_port: int | None = None,
        pose_topic: str = "pose",
    ):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.visible = False
        self.enabled = True
        self._latest_pose = None
        self.translation_mode = translation_mode
        self._target_anchor_base_pos = None
        self._actual_anchor_base_pos = None
        self._prev_target_base_pos = None

        root_joint_name = f"{REFERENCE_NAME_PREFIX}floating_base_joint"
        root_body_name = f"{REFERENCE_NAME_PREFIX}pelvis"
        root_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        root_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_joint_id == -1 or root_body_id == -1:
            raise ValueError("reference robot was not found in the loaded MuJoCo model")

        self.root_qpos_adr = mj_model.jnt_qposadr[root_joint_id]
        self.root_qvel_adr = mj_model.jnt_dofadr[root_joint_id]
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
        self.ref_geom_ids = np.array(get_subtree_geom_ids(mj_model, root_body_id), dtype=np.int32)
        self._shown_once = False
        self.set_visible(False)

        self.debug_subscriber = ZMQStateSubscriber(host=host, port=port, topic=topic)
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

    def reset_anchor(self):
        self._target_anchor_base_pos = None
        self._actual_anchor_base_pos = None
        self._prev_target_base_pos = None

    def set_visible(self, visible: bool):
        self.visible = visible
        alpha = self.alpha if visible and self.enabled else 0.0
        self.mj_model.geom_rgba[self.ref_geom_ids, 3] = alpha

    def toggle(self):
        self.set_visible(not self.visible)
        print(
            f"[ReferenceMotionVisualizer] reference motion visualization "
            f"{'enabled' if self.visible else 'hidden'}"
        )

    def poll(self):
        pose_msg = self.pose_subscriber.get_msg() if self.pose_subscriber is not None else None
        if pose_msg is not None and self._consume_pose_stream_msg(pose_msg):
            return

        debug_msg = self.debug_subscriber.get_msg()
        if debug_msg is None:
            return
        if not self._consume_debug_msg(debug_msg):
            return

    def _consume_debug_msg(self, msg: dict) -> bool:
        required = ("base_trans_target", "base_quat_target", "body_q_target")
        if not all(key in msg for key in required):
            return False

        base_pos = np.asarray(msg["base_trans_target"], dtype=np.float64)
        base_quat = np.asarray(msg["base_quat_target"], dtype=np.float64)
        body_q = np.asarray(msg["body_q_target"], dtype=np.float64)
        if base_pos.shape != (3,) or base_quat.shape != (4,) or body_q.shape != (29,):
            return False
        self._set_latest_pose(base_pos, base_quat, body_q)
        return True

    def _consume_pose_stream_msg(self, msg: dict) -> bool:
        required = ("body_pos_w", "body_quat_w", "joint_pos")
        if not all(key in msg for key in required):
            return False

        body_pos = np.asarray(msg["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(msg["body_quat_w"], dtype=np.float64)
        body_q = np.asarray(msg["joint_pos"], dtype=np.float64)
        if body_pos.ndim == 2:
            base_pos = body_pos[0]
        elif body_pos.ndim == 1 and body_pos.shape == (3,):
            base_pos = body_pos
        else:
            return False

        if body_quat.ndim == 2:
            base_quat = body_quat[0]
        elif body_quat.ndim == 1 and body_quat.shape == (4,):
            base_quat = body_quat
        else:
            return False

        if body_q.ndim == 2:
            body_q = body_q[0]
        if base_pos.shape != (3,) or base_quat.shape != (4,) or body_q.shape != (29,):
            return False
        self._set_latest_pose(base_pos, base_quat, body_q)
        return True

    def _set_latest_pose(self, base_pos: np.ndarray, base_quat: np.ndarray, body_q: np.ndarray):
        self._latest_pose = (base_pos, base_quat, body_q)
        self._maybe_refresh_translation_anchor(base_pos)
        if not self._shown_once:
            self._shown_once = True
            self.set_visible(True)
            print("[ReferenceMotionVisualizer] reference pose stream detected")

    def apply(self):
        if self._latest_pose is None or not self.enabled:
            return False

        base_pos, base_quat, body_q = self._latest_pose
        if self.translation_mode == "delta_aligned":
            if self._target_anchor_base_pos is None or self._actual_anchor_base_pos is None:
                self._target_anchor_base_pos = base_pos.copy()
                self._actual_anchor_base_pos = self._get_actual_robot_root_pos()
            base_pos = self._actual_anchor_base_pos + (base_pos - self._target_anchor_base_pos)
        self.mj_data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = base_pos
        self.mj_data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = base_quat
        self.mj_data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6] = 0.0
        self.mj_data.qpos[self.body_qpos_adrs] = body_q
        self.mj_data.qvel[self.body_qvel_adrs] = 0.0
        return True

    def _get_actual_robot_root_pos(self) -> np.ndarray:
        return self.mj_data.qpos[:3].copy()

    def _maybe_refresh_translation_anchor(self, target_base_pos: np.ndarray):
        if self.translation_mode != "delta_aligned":
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

        self.init_scene()
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
            lightened_rgb = np.clip(0.45 + 0.55 * rgb, 0.0, 1.0)
            element.set(
                "rgba",
                f"{lightened_rgb[0]:.6f} {lightened_rgb[1]:.6f} {lightened_rgb[2]:.6f} {alpha:.6f}",
            )

    def _init_reference_visualizer(self):
        if not self.config.get("ENABLE_REFERENCE_MOTION_VISUALIZATION", False):
            return
        try:
            self.reference_visualizer = ReferenceMotionVisualizer(
                mj_model=self.mj_model,
                mj_data=self.mj_data,
                body_joint_names=self.body_joint_names,
                alpha=self.config.get("REFERENCE_MOTION_ALPHA", 0.35),
                host=self.config.get("REFERENCE_MOTION_ZMQ_HOST", "127.0.0.1"),
                port=int(self.config.get("REFERENCE_MOTION_ZMQ_PORT", 5608)),
                topic=self.config.get("REFERENCE_MOTION_ZMQ_TOPIC", "g1_debug"),
                pose_port=int(self.config.get("REFERENCE_MOTION_POSE_ZMQ_PORT", 5556)),
                pose_topic=self.config.get("REFERENCE_MOTION_POSE_ZMQ_TOPIC", "pose"),
                translation_mode=self.config.get(
                    "REFERENCE_MOTION_TRANSLATION_MODE", "delta_aligned"
                ),
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
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
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
        reference_pose_applied = self.update_reference_motion_visualization()
        if reference_pose_applied:
            mujoco.mj_forward(self.mj_model, self.mj_data)

        self.check_fall()

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
        if self.reference_visualizer is not None:
            self.reference_visualizer.close()
            self.reference_visualizer = None
        if self.generated_scene_path and os.path.exists(self.generated_scene_path):
            os.remove(self.generated_scene_path)
            self.generated_scene_path = None


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
