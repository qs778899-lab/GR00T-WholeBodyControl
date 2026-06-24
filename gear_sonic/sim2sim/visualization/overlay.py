"""Viewer overlay for sim2sim tracking errors."""

import mujoco
import numpy as np


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

