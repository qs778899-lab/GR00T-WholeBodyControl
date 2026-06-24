"""CSV logging for sim2sim MuJoCo evaluation."""

import csv
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.sim2sim.constants import REFERENCE_NAME_PREFIX


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

