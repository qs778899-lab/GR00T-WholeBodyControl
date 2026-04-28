#!/usr/bin/env python3
"""Load SONIC motion_lib PKLs and expose Isaac-style encoder inputs.

This module is intentionally a thin adapter around the repository's original
motion preprocessing path.  It first tries to use TrackingCommand.create_offline,
optionally after starting IsaacSim's SimulationApp so pxr/USD bindings are
available in conda environments.  If Isaac dependencies are unavailable, it
falls back to the same Humanoid_Batch.fk_batch core used by MotionLibBase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Any

import joblib
import numpy as np
import torch


G1_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]

G1_ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

G1_MUJOCO_TO_ISAACLAB_BODY = [
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
]

G1_DEFAULT_ANGLES_ISAACLAB = torch.tensor(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=torch.float32,
)


def _install_open3d_stub() -> None:
    """Allow importing Humanoid_Batch when open3d is not installed.

    The FK path used here does not read mesh files.  open3d is only needed by
    optional mesh utilities in torch_humanoid_batch.py.
    """
    if "open3d" in sys.modules:
        return
    open3d = types.ModuleType("open3d")
    open3d.io = types.SimpleNamespace(
        read_triangle_mesh=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("open3d is unavailable in this environment")
        )
    )
    sys.modules["open3d"] = open3d


def _make_motion_lib_cfg(
    motion_file: Path,
    motion_name: str | None,
    target_fps: int,
    multi_thread: bool = False,
    num_future_frames: int | None = None,
    dt_future_ref_frames: float | None = None,
) -> Any:
    from easydict import EasyDict

    cfg = EasyDict(
        {
            "motion_file": str(motion_file),
            "asset": {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                "assetFileName": "g1_29dof_rev_1_0.xml",
                "urdfFileName": "",
            },
            "extend_config": [],
            "target_fps": target_fps,
            "multi_thread": multi_thread,
            "filter_motion_keys": motion_name,
            "mujoco_to_isaaclab_dof": G1_MUJOCO_TO_ISAACLAB_DOF,
            "mujoco_to_isaaclab_body": G1_MUJOCO_TO_ISAACLAB_BODY,
            "isaaclab_to_mujoco_dof": G1_ISAACLAB_TO_MUJOCO_DOF,
            "body_indexes_data": list(range(len(G1_MUJOCO_TO_ISAACLAB_BODY))),
            "body_indexes": list(range(len(G1_MUJOCO_TO_ISAACLAB_BODY))),
            "lower_joint_indices_mujoco": list(range(12)),
            "fix_height": "no_fix",
            "debug": False,
            "use_parallel_fk": False,
        }
    )
    if num_future_frames is not None:
        cfg.num_future_frames = num_future_frames
    if dt_future_ref_frames is not None:
        cfg.dt_future_ref_frames = dt_future_ref_frames
    return cfg


def _load_motion_entry(motion_file: Path, motion_name: str | None) -> tuple[str, dict[str, Any]]:
    data = joblib.load(motion_file)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty dict in {motion_file}")
    if motion_name is None:
        if len(data) != 1:
            raise ValueError("--motion-name is required when pkl contains multiple motions")
        motion_name = next(iter(data.keys()))
    if motion_name not in data:
        raise KeyError(f"motion_name={motion_name!r} not found in {motion_file}; keys={list(data)}")
    entry = data[motion_name]
    if "path" in entry:
        nested = joblib.load(entry["path"])
        entry = next(iter(nested.values()))
    return motion_name, entry


def _normalize_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_to_matrix_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat_wxyz(q)
    r, i, j, k = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    out = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return out.reshape(q.shape[:-1] + (3, 3))


def _compute_windows(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    root_quat_w: torch.Tensor,
    num_future_frames: int,
    frame_skip: int,
    robot_anchor_quat_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dof_pos.shape[0]
    base = torch.arange(n, device=dof_pos.device).view(n, 1)
    offsets = torch.arange(num_future_frames, device=dof_pos.device).view(1, -1) * frame_skip
    idx = torch.clamp(base + offsets, max=n - 1)

    # Match IsaacSim's command_multi_future_nonflat exactly:
    # TrackingCommand.command_multi_future first concatenates all future positions
    # with all future velocities as flat blocks, then the observation term reshapes
    # it to [num_future_frames, -1].  This is intentionally not per-frame
    # [q, dq] interleaving; the trained encoder/exported ONNX expects this layout.
    dof_pos_mf = dof_pos[idx]
    dof_vel_mf = dof_vel[idx]
    command_flat = torch.cat(
        [dof_pos_mf.reshape(n, -1), dof_vel_mf.reshape(n, -1)],
        dim=-1,
    )
    command_multi_future = command_flat.reshape(n, num_future_frames, -1)

    if robot_anchor_quat_w is None:
        robot_anchor_quat_w = root_quat_w
    robot_q = robot_anchor_quat_w.view(n, 1, 4).expand(n, num_future_frames, 4)
    ref_q = root_quat_w[idx]
    rel_q = _quat_mul_wxyz(_quat_conjugate_wxyz(robot_q), ref_q)
    rot6 = _quat_to_matrix_wxyz(rel_q)[..., :2].reshape(n, num_future_frames, 6)
    return command_multi_future, rot6


def _load_with_tracking_command_offline(
    motion_file: Path,
    motion_name: str | None,
    target_fps: int,
    num_future_frames: int,
    dt_future_ref_frames: float,
    device: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Use SONIC's IsaacSim offline TrackingCommand path when available.

    This is the highest-fidelity path because it imports the same command class
    used by IsaacSim/IsaacLab training and evaluation.  It requires a functional
    IsaacLab Python environment, including USD/pxr bindings.
    """
    from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand

    cfg = _make_motion_lib_cfg(
        motion_file,
        motion_name,
        target_fps,
        multi_thread=False,
        num_future_frames=num_future_frames,
        dt_future_ref_frames=dt_future_ref_frames,
    )
    command = TrackingCommand.create_offline(cfg, torch.device(device))
    command.set_motion_state(
        torch.zeros(1, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    )

    lib = command.motion_lib
    dof_pos = lib.dof_pos.detach().to(device)
    dof_vel = lib.dof_vel.detach().to(device)
    frame_ids = torch.arange(dof_pos.shape[0], device=device)
    motion_ids = torch.zeros(dof_pos.shape[0], dtype=torch.long, device=device)
    root_pos_w = lib.get_root_pos_w(motion_ids, frame_ids)
    root_quat_w = lib.get_root_quat_w(motion_ids, frame_ids)
    body_pos_w = lib.body_pos_w_full.detach().to(device)
    body_quat_w = lib.body_quat_w_full.detach().to(device)

    command_windows = []
    root_ori_windows = []
    with torch.no_grad():
        for frame in range(dof_pos.shape[0]):
            command.set_motion_state(
                torch.zeros(1, dtype=torch.long, device=device),
                torch.tensor([frame], dtype=torch.long, device=device),
            )
            command_windows.append(command.command_multi_future.reshape(1, num_future_frames, -1))
            root_ori_windows.append(
                command.root_rot_dif_l_multi_future.reshape(1, num_future_frames, 6)
            )
    command_mf = torch.cat(command_windows, dim=0).detach()
    root_ori_mf = torch.cat(root_ori_windows, dim=0).detach()
    return dof_pos, dof_vel, root_pos_w, root_quat_w, body_pos_w, body_quat_w, command_mf, root_ori_mf


def _start_isaacsim_app() -> Any:
    """Start IsaacSim headless so pxr/USD imports work in the official path."""
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True, "width": 64, "height": 64})


@dataclass
class MotionLibSequence:
    motion_name: str
    source: str
    fps: int
    original_fps: float
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    command_multi_future_nonflat: torch.Tensor
    motion_anchor_ori_b_mf_nonflat: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.dof_pos.shape[0])

    def q29_mujoco(self) -> np.ndarray:
        return self.dof_pos[:, G1_ISAACLAB_TO_MUJOCO_DOF].detach().cpu().numpy()


def load_motionlib_sequence(
    motion_file: Path,
    motion_name: str | None = None,
    target_fps: int = 50,
    num_future_frames: int = 10,
    dt_future_ref_frames: float = 0.1,
    device: str = "cpu",
    prefer_motionlib_robot: bool = True,
    use_isaacsim_app: bool = False,
    close_isaacsim_app: bool = False,
) -> MotionLibSequence:
    """Load a SONIC motion_lib pkl and return Isaac-style motion encoder tensors."""
    motion_file = motion_file.expanduser().resolve()
    motion_name, entry = _load_motion_entry(motion_file, motion_name)
    original_fps = float(entry.get("fps", 30.0))
    frame_skip = int(round(dt_future_ref_frames * target_fps))
    if frame_skip <= 0:
        raise ValueError("dt_future_ref_frames * target_fps must be >= 1 frame")

    command_mf = None
    root_ori_mf = None
    sim_app = None
    if use_isaacsim_app:
        try:
            sim_app = _start_isaacsim_app()
        except Exception as exc:  # pragma: no cover - environment-dependent fallback
            print(f"[motionlib_provider] IsaacSim app startup failed; falling back if needed: {type(exc).__name__}: {exc}")

    try:
        if prefer_motionlib_robot:
            try:
                (
                    dof_pos,
                    dof_vel,
                    root_pos_w,
                    root_quat_w,
                    body_pos_w,
                    body_quat_w,
                    command_mf,
                    root_ori_mf,
                ) = (
                    _load_with_tracking_command_offline(
                        motion_file=motion_file,
                        motion_name=motion_name,
                        target_fps=target_fps,
                        num_future_frames=num_future_frames,
                        dt_future_ref_frames=dt_future_ref_frames,
                        device=device,
                    )
                )
                source = "TrackingCommand offline"
                if sim_app is not None:
                    source += " (IsaacSim app)"
            except Exception as tracking_exc:  # pragma: no cover - environment-dependent fallback
                if sim_app is not None:
                    sim_app.close()
                    sim_app = None
                try:
                    _install_open3d_stub()
                    from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

                    cfg = _make_motion_lib_cfg(motion_file, motion_name, target_fps, multi_thread=False)
                    lib = MotionLibRobot(cfg, 1, torch.device(device))
                    lib.load_motions_for_training(max_num_seqs=1)
                    dof_pos = lib.dof_pos.detach().to(device)
                    dof_vel = lib.dof_vel.detach().to(device)
                    root_pos_w = lib.get_root_pos_w(
                        torch.zeros(dof_pos.shape[0], dtype=torch.long, device=device),
                        torch.arange(dof_pos.shape[0], device=device),
                    )
                    root_quat_w = lib.get_root_quat_w(
                        torch.zeros(dof_pos.shape[0], dtype=torch.long, device=device),
                        torch.arange(dof_pos.shape[0], device=device),
                    )
                    body_pos_w = lib.body_pos_w_full.detach().to(device)
                    body_quat_w = lib.body_quat_w_full.detach().to(device)
                    source = f"MotionLibRobot (TrackingCommand unavailable: {type(tracking_exc).__name__}: {tracking_exc})"
                except Exception as motionlib_exc:
                    dof_pos, dof_vel, root_pos_w, root_quat_w, body_pos_w, body_quat_w = (
                        _load_with_humanoid_batch(entry, target_fps, device)
                    )
                    source = (
                        "Humanoid_Batch fallback "
                        f"(TrackingCommand unavailable: {type(tracking_exc).__name__}: {tracking_exc}; "
                        f"MotionLibRobot unavailable: {type(motionlib_exc).__name__}: {motionlib_exc})"
                    )
        else:
            dof_pos, dof_vel, root_pos_w, root_quat_w, body_pos_w, body_quat_w = (
                _load_with_humanoid_batch(entry, target_fps, device)
            )
            source = "Humanoid_Batch fallback"
    finally:
        # In IsaacSim 5.x, SimulationApp.close() may tear down Kit strongly
        # enough to prevent short CLI tools from continuing to their own output
        # or ZMQ send phase.  Default to leaving it alive until process exit.
        if sim_app is not None and close_isaacsim_app:
            sim_app.close()

    if command_mf is None or root_ori_mf is None:
        command_mf, root_ori_mf = _compute_windows(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            root_quat_w=root_quat_w,
            num_future_frames=num_future_frames,
            frame_skip=frame_skip,
            robot_anchor_quat_w=root_quat_w,
        )
    return MotionLibSequence(
        motion_name=motion_name,
        source=source,
        fps=target_fps,
        original_fps=original_fps,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        command_multi_future_nonflat=command_mf,
        motion_anchor_ori_b_mf_nonflat=root_ori_mf,
    )


def _load_with_humanoid_batch(
    entry: dict[str, Any], target_fps: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _install_open3d_stub()
    from gear_sonic.isaac_utils import rotations
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    cfg = _make_motion_lib_cfg(Path("."), None, target_fps, multi_thread=False)
    Humanoid_Batch.load_mesh = lambda self: None  # type: ignore[method-assign]
    humanoid = Humanoid_Batch(cfg, device=torch.device(device))
    pose_aa = torch.as_tensor(entry["pose_aa"], dtype=torch.float32, device=device)
    trans = torch.as_tensor(entry["root_trans_offset"], dtype=torch.float32, device=device)
    fps = float(entry.get("fps", 30.0))
    motion = humanoid.fk_batch(
        pose_aa[None],
        trans[None],
        return_full=True,
        fps=fps,
        target_fps=target_fps,
        interpolate_data=True,
        use_parallel_fk=False,
    )

    dof_pos_mj = motion.dof_pos[0]
    dof_vel_mj = motion.dof_vels[0]
    dof_pos = dof_pos_mj[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    dof_vel = dof_vel_mj[:, G1_MUJOCO_TO_ISAACLAB_DOF]

    body_pos_mj = motion.global_translation[0]
    body_quat_xyzw_mj = motion.global_rotation[0]
    body_pos_w = body_pos_mj[:, G1_MUJOCO_TO_ISAACLAB_BODY]
    body_quat_w = rotations.xyzw_to_wxyz(body_quat_xyzw_mj[:, G1_MUJOCO_TO_ISAACLAB_BODY])
    root_pos_w = body_pos_w[:, 0]
    root_quat_w = body_quat_w[:, 0]
    return dof_pos, dof_vel, root_pos_w, root_quat_w, body_pos_w, body_quat_w
