"""MuJoCo lifecycle hook for sim2sim visualization and evaluation logging."""

import os
import pathlib
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from gear_sonic.sim2sim.constants import REFERENCE_NAME_PREFIX, SIM2SIM_BODY_FRAMES
from gear_sonic.sim2sim.logging.eval_logger import Sim2SimEvalLogger
from gear_sonic.sim2sim.visualization.overlay import Sim2SimTrackingOverlay
from gear_sonic.sim2sim.visualization.reference_motion import ReferenceMotionVisualizer
from gear_sonic.utils.mujoco_sim.link_error_plot import Sim2SimLinkErrorPlot


def maybe_create_reference_visualization_scene(
    config: dict[str, Any],
    xml_path: pathlib.Path,
) -> tuple[str, str | None]:
    if not config.get("ENABLE_REFERENCE_MOTION_VISUALIZATION", False):
        return str(xml_path), None

    scene_tree = ET.parse(xml_path)
    scene_root = scene_tree.getroot()
    scene_dir = xml_path.parent

    include_elements = scene_root.findall("include")
    if not include_elements:
        print(
            "[ReferenceMotionVisualizer] no <include> found in scene XML; "
            "reference visualization disabled"
        )
        return str(xml_path), None

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
        return str(xml_path), None

    robot_tree = ET.parse(include_path)
    robot_body = robot_tree.getroot().find("./worldbody/body")
    scene_worldbody = scene_root.find("./worldbody")
    if robot_body is None or scene_worldbody is None:
        print(
            "[ReferenceMotionVisualizer] robot/worldbody missing from XML; "
            "reference visualization disabled"
        )
        return str(xml_path), None

    ref_body = deepcopy(robot_body)
    _prefix_reference_subtree(ref_body, alpha=float(config.get("REFERENCE_MOTION_ALPHA", 0.35)))
    scene_worldbody.append(ref_body)

    fd, temp_path = tempfile.mkstemp(
        prefix="mujoco_reference_scene_",
        suffix=".xml",
        dir=scene_dir,
    )
    os.close(fd)
    scene_tree.write(temp_path, encoding="utf-8", xml_declaration=False)
    return temp_path, temp_path


def _prefix_reference_subtree(root: ET.Element, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for element in root.iter():
        if "name" in element.attrib:
            element.set("name", f"{REFERENCE_NAME_PREFIX}{element.get('name')}")
        if element.tag != "geom":
            continue
        element.set("contype", "0")
        element.set("conaffinity", "0")
        ref_rgb = np.array([1.0, 0.15, 0.15], dtype=np.float64)
        element.set(
            "rgba",
            f"{ref_rgb[0]:.6f} {ref_rgb[1]:.6f} {ref_rgb[2]:.6f} {alpha:.6f}",
        )


class Sim2SimMujocoHook:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        body_joint_names: list[str],
        left_hand_joint_names: list[str],
        right_hand_joint_names: list[str],
        viewer,
        generated_scene_path: str | None = None,
    ):
        self.config = config
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.viewer = viewer
        self.generated_scene_path = generated_scene_path

        self.eval_logger: Sim2SimEvalLogger | None = None
        self.link_error_plot: Sim2SimLinkErrorPlot | None = None
        self.plot_link_indices: list[int] | None = None
        self.reference_visualizer: ReferenceMotionVisualizer | None = None
        self.tracking_overlay: Sim2SimTrackingOverlay | None = None

        self._init_eval_logger()
        self._init_link_error_plot()
        self._init_reference_visualizer(
            body_joint_names=body_joint_names,
            left_hand_joint_names=left_hand_joint_names,
            right_hand_joint_names=right_hand_joint_names,
        )
        self.tracking_overlay = (
            Sim2SimTrackingOverlay(self.viewer, SIM2SIM_BODY_FRAMES)
            if self.viewer is not None
            else None
        )

    def _init_eval_logger(self):
        if not self.config.get("ENABLE_SIM2SIM_EVAL_LOGGING", False):
            return
        logs_dir = Path(self.config.get("SIM2SIM_EVAL_LOGS_DIR", "/tmp/sonic_logs/official_walk_zmq01"))
        self.eval_logger = Sim2SimEvalLogger(logs_dir=logs_dir, body_names=SIM2SIM_BODY_FRAMES)

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
        self.plot_link_indices = indices
        self.link_error_plot = Sim2SimLinkErrorPlot(
            link_names=valid_links,
            ymax_mm=float(self.config.get("SIM2SIM_ERROR_PLOT_YMAX_MM", 300.0)),
            refresh_hz=float(self.config.get("SIM2SIM_ERROR_PLOT_REFRESH_HZ", 20.0)),
        )
        self.link_error_plot.start()
        print(f"[LinkErrorPlot] Started for links: {valid_links}")

    def _init_reference_visualizer(
        self,
        *,
        body_joint_names: list[str],
        left_hand_joint_names: list[str],
        right_hand_joint_names: list[str],
    ):
        if not self.config.get("ENABLE_REFERENCE_MOTION_VISUALIZATION", False):
            return
        try:
            self.reference_visualizer = ReferenceMotionVisualizer(
                mj_model=self.mj_model,
                mj_data=self.mj_data,
                body_joint_names=body_joint_names,
                left_hand_joint_names=left_hand_joint_names,
                right_hand_joint_names=right_hand_joint_names,
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
                align_delay_frames=int(self.config.get("REFERENCE_MOTION_ALIGN_DELAY_FRAMES", 50)),
            )
        except Exception as exc:
            self.reference_visualizer = None
            print(f"[ReferenceMotionVisualizer] disabled: {exc}")

    def capture_actual_body_pos_pre_ref(self) -> np.ndarray | None:
        if self.eval_logger is None:
            return None
        self.eval_logger._resolve_body_ids(self.mj_model)
        return np.asarray(
            self.mj_data.xpos[self.eval_logger.body_ids],
            dtype=np.float64,
        ).copy()

    def update_reference_visualization(self) -> bool:
        if self.reference_visualizer is None:
            return False
        self.reference_visualizer.poll()
        return self.reference_visualizer.apply()

    def log_frame(
        self,
        *,
        actual_body_pos: np.ndarray | None = None,
        write_step_sync: bool = True,
    ):
        if self.eval_logger is None and self.tracking_overlay is None:
            return

        body_ids = None
        ref_body_ids = None
        if self.eval_logger is not None:
            self.eval_logger._resolve_body_ids(self.mj_model)
            body_ids = self.eval_logger.body_ids
            ref_body_ids = self.eval_logger.ref_body_ids
        elif self.tracking_overlay is not None:
            self.tracking_overlay._resolve_body_ids(self.mj_model)
            body_ids = self.tracking_overlay._body_ids

        if actual_body_pos is None and body_ids is not None:
            actual_body_pos = np.asarray(
                self.mj_data.xpos[body_ids],
                dtype=np.float64,
            ).copy()

        source_frame_index = None
        ref_body_pos = None
        if self.reference_visualizer is not None and body_ids is not None:
            source_frame_index, ref_body_pos = self.reference_visualizer.compute_exact_reference_body_pos(
                body_ids,
                ref_body_ids=ref_body_ids,
            )

        if self.tracking_overlay is not None:
            self.tracking_overlay.update(
                mj_model=self.mj_model,
                actual_body_pos=actual_body_pos,
                ref_body_pos=ref_body_pos,
                source_frame_index=source_frame_index,
            )

        if (
            self.link_error_plot is not None
            and actual_body_pos is not None
            and ref_body_pos is not None
            and source_frame_index is not None
            and int(source_frame_index) >= 0
        ):
            full_err_mm = np.linalg.norm(actual_body_pos - ref_body_pos, axis=1) * 1000.0
            selected_err = full_err_mm[self.plot_link_indices]
            motion_start = (
                self.reference_visualizer._align_delay_frames
                if self.reference_visualizer is not None
                else 0
            )
            plot_frame = int(source_frame_index) - max(0, motion_start)
            self.link_error_plot.push(plot_frame, selected_err)

        if self.eval_logger is None:
            return
        self.eval_logger.log(
            mj_model=self.mj_model,
            mj_data=self.mj_data,
            source_frame_index=source_frame_index,
            actual_body_pos=actual_body_pos,
            ref_body_pos=ref_body_pos,
            write_step_sync=write_step_sync,
        )

    def render_overlay(self):
        if self.tracking_overlay is not None:
            self.tracking_overlay.render()

    def toggle_reference_visualization(self) -> bool:
        if self.reference_visualizer is None:
            return False
        self.reference_visualizer.toggle()
        return True

    def reset_anchor(self):
        if self.reference_visualizer is not None:
            self.reference_visualizer.reset_anchor()

    def close(self):
        if self.eval_logger is not None:
            self.eval_logger.close()
            self.eval_logger = None
        if self.link_error_plot is not None:
            self.link_error_plot.close()
            self.link_error_plot = None
        if self.reference_visualizer is not None:
            self.reference_visualizer.close()
            self.reference_visualizer = None
        if self.generated_scene_path and os.path.exists(self.generated_scene_path):
            os.remove(self.generated_scene_path)
            self.generated_scene_path = None
