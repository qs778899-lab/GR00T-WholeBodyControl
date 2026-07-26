"""Thin IsaacLab command binding for SONIC motion-compliance state and math."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
import torch

from ...core import ComplianceSpec
from ...core.reference_modifier import _virtual_force_from_reference_delta_unchecked
from .frames import (
    _common_to_world_vectors_unchecked,
    _rotate_vectors_wxyz_unchecked,
    _world_to_body_vectors_unchecked,
    _world_to_common_positions_unchecked,
)
from .mapping import resolve_body_index_map
from .state import ComplianceCommandState, ComplianceSamplingSpec
from .validation import site_body_offsets_tensor
from .wrench import ResidualWrenchLimiter

if TYPE_CHECKING:
    from .manager_cfg import MotionComplianceCommandCfg


def _articulation_body_data(robot: Articulation, field: str) -> torch.Tensor:
    for candidate in (f"body_link_{field}_w", f"body_{field}_w"):
        if hasattr(robot.data, candidate):
            return getattr(robot.data, candidate)
    raise AttributeError(f"articulation data has no world body {field} field")


class MotionComplianceCommand(CommandTerm):
    """Bind portable compliance math to SONIC references and IsaacLab state."""

    cfg: MotionComplianceCommandCfg

    def __init__(self, cfg: MotionComplianceCommandCfg, env) -> None:
        super().__init__(cfg, env)
        if type(cfg.enabled) is not bool:
            raise TypeError("enabled must be bool")
        self.operational_enabled = cfg.enabled
        self._wrench_dirty = False
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_map = resolve_body_index_map(
            cfg.reference_body_names,
            self.robot.body_names,
            cfg.site_body_names,
            cfg.anchor_body_name,
        )
        if cfg.common_frame != "current_anchor":
            raise ValueError("only the explicit current_anchor common frame is supported")
        self.site_body_offsets = site_body_offsets_tensor(
            cfg.site_body_offsets,
            num_sites=self.body_map.num_sites,
            device=self.device,
        )
        force_spec = ComplianceSpec(
            force_threshold_range_n=tuple(cfg.force_threshold_range_n),
            reference_displacement_m=cfg.reference_displacement_m,
            tracking_gain_n_per_m=cfg.tracking_gain_n_per_m,
            tracking_force_cap_n=cfg.tracking_force_cap_n,
            max_net_force_n=cfg.max_net_force_n,
            max_net_torque_nm=cfg.max_net_torque_nm,
        )

        sampling = ComplianceSamplingSpec(
            enable_probability=cfg.enable_probability,
            site_activation_probability=cfg.site_activation_probability,
            force_threshold_range_n=force_spec.force_threshold_range_n,
            reference_displacement_m=force_spec.reference_displacement_m,
            reference_offset_range_m=tuple(cfg.reference_offset_range_m),
        )
        self.state = ComplianceCommandState(
            self.num_envs,
            self.body_map.num_sites,
            cfg.num_future_frames,
            sampling,
            device=self.device,
            dtype=torch.float32,
            seed=cfg.seed,
        )
        self.wrench_limiter = ResidualWrenchLimiter(
            max_force_n=cfg.max_net_force_n,
            max_torque_nm=cfg.max_net_torque_nm,
        )
        self.application_body_ids = torch.tensor(
            (*self.body_map.articulation_site_indices, self.body_map.articulation_anchor_index),
            dtype=torch.long,
            device=self.device,
        )
        self._application_force_world = torch.zeros(
            (self.num_envs, self.body_map.num_sites + 1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._application_torque_world = torch.zeros_like(self._application_force_world)
        self._application_force_body = torch.zeros_like(self._application_force_world)
        self._application_torque_body = torch.zeros_like(self._application_force_world)
        self.metrics["active_site_fraction"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["peak_site_force_n"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self.state.condition

    @property
    def application_force_world(self) -> torch.Tensor:
        return self._application_force_world

    @property
    def application_torque_world(self) -> torch.Tensor:
        return self._application_torque_world

    @property
    def application_force_body(self) -> torch.Tensor:
        return self._application_force_body

    @property
    def application_torque_body(self) -> torch.Tensor:
        return self._application_torque_body

    @property
    def wrench_dirty(self) -> bool:
        return self._wrench_dirty

    def mark_wrench_applied(self) -> None:
        self._wrench_dirty = True

    def mark_wrench_cleared(self) -> None:
        self._wrench_dirty = False

    def set_operational_enabled(self, enabled: bool) -> None:
        """Change the host-side switch without inspecting CUDA tensor values."""

        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if enabled == self.operational_enabled:
            return
        self.operational_enabled = enabled
        if enabled:
            self.state.reset()
        else:
            self.state.disable()
            self._clear_application_buffers()

    def _resolved_env_ids(self, env_ids):
        if env_ids is None:
            return slice(None), None
        if isinstance(env_ids, slice):
            return env_ids, env_ids
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return ids, ids

    def body_wrench_for_envs(self, env_ids):
        index, writer_ids = self._resolved_env_ids(env_ids)
        return (
            self._application_force_body[index],
            self._application_torque_body[index],
            writer_ids,
        )

    def _clear_application_buffers(self, env_ids=None) -> None:
        ids = self.state._env_ids_tensor(env_ids)
        self._application_force_world[ids] = 0.0
        self._application_torque_world[ids] = 0.0
        self._application_force_body[ids] = 0.0
        self._application_torque_body[ids] = 0.0

    def clear_wrench(self, env_ids=None) -> None:
        ids = self.state._env_ids_tensor(env_ids)
        self.state.clear_dynamic(ids)
        self._clear_application_buffers(ids)

    def _tracking_term(self):
        return self._env.command_manager.get_term(self.cfg.tracking_command_name)

    def _reference_world_state(self):
        tracking = self._tracking_term()
        if tracking.num_future_frames != self.state.num_future_frames:
            raise ValueError("tracking and compliance future-frame counts differ")
        num_reference_bodies = len(self.cfg.reference_body_names)
        positions = tracking.body_pos_w_multi_future.reshape(
            self.num_envs,
            self.state.num_future_frames,
            num_reference_bodies,
            3,
        )
        quaternions = tracking.body_quat_w_multi_future.reshape(
            self.num_envs,
            self.state.num_future_frames,
            num_reference_bodies,
            4,
        )
        site_ids = self.body_map.reference_site_indices
        site_position = positions[:, :, site_ids]
        site_quaternion = quaternions[:, :, site_ids]
        offsets = self.site_body_offsets.reshape(1, 1, self.body_map.num_sites, 3)
        site_position = site_position + _rotate_vectors_wxyz_unchecked(
            site_quaternion,
            offsets,
        )
        return site_position

    def _articulation_world_state(self):
        position = _articulation_body_data(self.robot, "pos")
        quaternion = _articulation_body_data(self.robot, "quat")
        site_ids = self.body_map.articulation_site_indices
        anchor_id = self.body_map.articulation_anchor_index
        site_body_position = position[:, site_ids]
        site_quaternion = quaternion[:, site_ids]
        offset_world = _rotate_vectors_wxyz_unchecked(
            site_quaternion,
            self.site_body_offsets.reshape(1, self.body_map.num_sites, 3),
        )
        site_point_position = site_body_position + offset_world
        return (
            site_body_position,
            site_point_position,
            offset_world,
            site_quaternion,
            position[:, anchor_id],
            quaternion[:, anchor_id],
        )

    def _update_metrics(self) -> None:
        if not self.operational_enabled:
            self.metrics["active_site_fraction"].zero_()
            self.metrics["peak_site_force_n"].zero_()
            return
        self.metrics["active_site_fraction"][:] = self.state.active_site_mask.float().mean(dim=-1)
        self.metrics["peak_site_force_n"][:] = torch.linalg.vector_norm(
            self.state.site_force_world,
            dim=-1,
        ).max(dim=-1).values

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self.state._env_ids_tensor(env_ids)
        if self.operational_enabled:
            self.state.reset(ids)
        else:
            self.state.disable(ids)
        self._clear_application_buffers(ids)

    def _update_command(self) -> None:
        if not self.operational_enabled:
            if self._wrench_dirty:
                self.clear_wrench()
            return
        original_world = self._reference_world_state()
        (
            site_body_world,
            site_point_world,
            site_offset_world,
            site_quaternion_world,
            anchor_position_world,
            anchor_quaternion_world,
        ) = self._articulation_world_state()

        frame_origin_future = anchor_position_world[:, None, None, :]
        frame_quaternion_future = anchor_quaternion_world[:, None, None, :]
        original_common = _world_to_common_positions_unchecked(
            original_world,
            frame_origin_future,
            frame_quaternion_future,
        )
        current_common = _world_to_common_positions_unchecked(
            site_point_world,
            anchor_position_world[:, None, :],
            anchor_quaternion_world[:, None, :],
        )
        compliant_common = original_common + self.state.reference_offset_common[:, None]
        force_common_future = _virtual_force_from_reference_delta_unchecked(
            original_common,
            compliant_common,
            self.state.active_site_mask,
            self.state.force_threshold_n,
            current_reference=current_common,
            reference_displacement_m=self.cfg.reference_displacement_m,
            tracking_gain_n_per_m=self.cfg.tracking_gain_n_per_m,
            tracking_force_cap_n=self.cfg.tracking_force_cap_n,
            enabled=self.state.enabled,
        )
        site_force_world = _common_to_world_vectors_unchecked(
            force_common_future[:, 0],
            anchor_quaternion_world[:, None, :],
        )
        site_torque_world = torch.cross(site_offset_world, site_force_world, dim=-1)
        limited = self.wrench_limiter._limit_unchecked(
            site_body_world,
            anchor_position_world,
            site_force_world,
            site_torque_world,
        )

        self.state.original_reference_common.copy_(original_common)
        self.state.compliant_reference_common.copy_(compliant_common)
        self.state.current_reference_common.copy_(current_common)
        self.state.force_common_future.copy_(force_common_future)
        self.state.site_force_world.copy_(limited.site_force_world)
        self.state.site_torque_world.copy_(limited.site_torque_world)
        self.state.anchor_force_world.copy_(limited.anchor_force_world)
        self.state.anchor_torque_world.copy_(limited.anchor_torque_world)
        self._application_force_world[:, :-1] = limited.site_force_world
        self._application_force_world[:, -1] = limited.anchor_force_world
        self._application_torque_world[:, :-1] = limited.site_torque_world
        self._application_torque_world[:, -1] = limited.anchor_torque_world
        application_quaternion_world = torch.cat(
            (site_quaternion_world, anchor_quaternion_world[:, None]),
            dim=1,
        )
        self._application_force_body.copy_(
            _world_to_body_vectors_unchecked(
                self._application_force_world,
                application_quaternion_world,
            )
        )
        self._application_torque_body.copy_(
            _world_to_body_vectors_unchecked(
                self._application_torque_world,
                application_quaternion_world,
            )
        )
