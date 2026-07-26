"""Isaac Lab command term owning SONIC compliance state and wrench sampling."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import torch

from ....core import CartesianFrameSpec
from ..body_names import resolve_compliance_sites
from ..frames import (
    quaternion_rotate_wxyz_prevalidated,
    world_positions_to_frame_prevalidated,
)
from ..operational import ComplianceOperationalControl
from ..state import SonicComplianceCommandState
from ..wrench import ArticulationWrenchAdapter, WrenchWriteGate


def _frame_from_kind(kind: str, *, anchor_body: str) -> CartesianFrameSpec:
    if kind == "world":
        return CartesianFrameSpec.world()
    if kind == "anchor_local":
        return CartesianFrameSpec.anchor_local(anchor_body)
    if kind == "heading_local":
        return CartesianFrameSpec.heading_local(anchor_body)
    raise ValueError("frame_kind must be 'world', 'anchor_local', or 'heading_local'")


class SonicComplianceCommand(ComplianceOperationalControl, CommandTerm):
    """Independent command/state for optional CHIP-style compliant tracking."""

    cfg: SonicComplianceCommandCfg

    def __init__(self, cfg: "SonicComplianceCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.reference_body_names = tuple(cfg.reference_body_names)
        frame = _frame_from_kind(cfg.frame_kind, anchor_body=cfg.anchor_body)
        self.sites = resolve_compliance_sites(
            cfg.reference_body_names,
            self.robot.body_names,
            cfg.site_names,
            target_frame=frame,
            force_frame=frame,
            max_displacement_m=cfg.max_displacement_m,
        )
        if cfg.anchor_body not in self.robot.body_names:
            raise ValueError(f"anchor body {cfg.anchor_body!r} is missing from articulation")
        self.anchor_body_index = self.robot.body_names.index(cfg.anchor_body)
        dtype = self.robot.data.body_pos_w.dtype
        if cfg.site_offsets_local_xyz is None:
            site_offsets_local = torch.zeros(
                self.sites.spec.num_sites,
                3,
                dtype=dtype,
                device=self.device,
            )
        else:
            site_offsets_local = torch.tensor(
                cfg.site_offsets_local_xyz,
                dtype=dtype,
                device=self.device,
            )
        self.state = SonicComplianceCommandState(
            sites=self.sites,
            num_envs=self.num_envs,
            num_future_frames=cfg.num_future_frames,
            device=self.device,
            dtype=dtype,
            target_damper_alpha=cfg.target_damper_alpha,
            site_offsets_local=site_offsets_local,
        )
        self._articulation_site_indices = torch.tensor(
            self.sites.articulation_indices,
            dtype=torch.long,
            device=self.device,
        )
        self._application_offsets_local = self.state.site_offsets_local.view(
            1,
            self.sites.spec.num_sites,
            3,
        ).expand(self.num_envs, self.sites.spec.num_sites, 3).clone()
        self._compliance_values_m_per_n = torch.tensor(
            cfg.compliance_values_m_per_n,
            dtype=dtype,
            device=self.device,
        )
        self.wrench = ArticulationWrenchAdapter(
            self.robot,
            body_selection=self.sites.articulation,
            num_envs=self.num_envs,
            device=self.device,
            dtype=dtype,
        )
        self._sampling_generator = torch.Generator(device=self.state.device)
        self._sampling_generator.manual_seed(cfg.sampling_seed)
        self._all_env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.state.device,
        )
        self._time_to_next_pulse = torch.full(
            (self.num_envs,),
            math.inf,
            dtype=self.state.dtype,
            device=self.state.device,
        )
        self._operational_enabled = bool(cfg.enabled)
        self._operational_enabled_last_update = False
        self._wrench_write_gate = WrenchWriteGate()
        self.metrics["enabled_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["active_site_fraction"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Actor-safe state: enable, site mask, and compliance; never true force."""

        return self.state.actor_command

    @property
    def use_target_damper(self) -> bool:
        return self.cfg.target_damper_enabled

    @property
    def force_on_robot_w(self) -> torch.Tensor:
        """Privileged/event force buffer in the simulation world frame."""

        return self.state.force_on_robot_w

    def _anchor_pose_w(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.sites.spec.common_frame.kind.value == "world":
            return None, None
        return (
            self.robot.data.body_pos_w[:, self.anchor_body_index],
            self.robot.data.body_quat_w[:, self.anchor_body_index],
        )

    def current_site_positions_w(self) -> torch.Tensor:
        """Read offset site points using only articulation-space indices."""

        body_position_w = self.robot.data.body_pos_w.index_select(
            1,
            self._articulation_site_indices,
        )
        body_quaternion_wxyz = self.robot.data.body_quat_w.index_select(
            1,
            self._articulation_site_indices,
        )
        return body_position_w + quaternion_rotate_wxyz_prevalidated(
            body_quaternion_wxyz,
            self._application_offsets_local,
        )

    def current_site_quaternions_wxyz(self) -> torch.Tensor:
        """Read current link-frame quaternions using articulation-space indices."""

        return self.robot.data.body_quat_w.index_select(
            1,
            self._articulation_site_indices,
        )

    def application_offsets_local(self) -> torch.Tensor:
        """Expand configured link-local application offsets over environments."""

        return self._application_offsets_local.clone()

    def current_eef_common_future(
        self,
        current_site_positions_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read current offset sites and expand over the future dimension."""

        current_eef_w = (
            self.current_site_positions_w()
            if current_site_positions_w is None
            else current_site_positions_w
        )
        anchor_position_w, anchor_quaternion_wxyz = self._anchor_pose_w()
        current_common = world_positions_to_frame_prevalidated(
            current_eef_w,
            frame=self.sites.spec.common_frame,
            anchor_position_w=anchor_position_w,
            anchor_quaternion_wxyz=anchor_quaternion_wxyz,
        )
        return current_common.unsqueeze(1).expand(
            self.num_envs,
            self.cfg.num_future_frames,
            self.sites.spec.num_sites,
            3,
        ).clone()

    def reset_envs(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
    ) -> None:
        """Clear wrench/command state and reset selected damper goals."""

        ids = self.state._env_ids_tensor(env_ids)  # noqa: SLF001
        self.state.reset(self.current_eef_common_future(), ids)
        self._schedule_next_pulse(ids)
        if self._wrench_write_gate.consume_clear_on_reset(
            globally_enabled=self._operational_enabled,
        ):
            self.wrench.clear(ids if self._operational_enabled else None)

    def sample_and_apply(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
    ) -> None:
        """Sample arbitrary simultaneous site pulses; the next command step writes them."""

        if not self._operational_enabled:
            return
        ids = self.state._env_ids_tensor(env_ids)  # noqa: SLF001
        if ids.numel() == 0:
            return
        requested_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.state.device,
        )
        requested_mask[ids] = True
        start_mask = self.state.startable_pulse_mask_prevalidated(requested_mask)
        application_positions_w = self.current_site_positions_w()
        current_eef_common = (
            self.current_eef_common_future(application_positions_w)
            if self.cfg.target_damper_enabled
            else None
        )
        self._sample_and_start_masked_prevalidated(
            start_mask,
            application_positions_w=application_positions_w,
            wrench_origin_w=self.robot.data.body_pos_w[:, self.anchor_body_index],
            current_eef_common=current_eef_common,
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self.reset_envs(env_ids)

    def _resample(self, env_ids: Sequence[int]) -> None:
        """Override Isaac Lab's global-RNG ``time_left.uniform_`` lifecycle."""

        ids = self.state._env_ids_tensor(env_ids)  # noqa: SLF001
        if ids.numel() == 0:
            return
        self.time_left[ids] = math.inf
        self._resample_command(ids)
        self.command_counter[ids] += 1

    def _update_command(self) -> None:
        self._update_command_prevalidated(self._env.step_dt)

    def _update_metrics(self) -> None:
        enabled = self.state.enabled
        active_sites = enabled.unsqueeze(-1) & self.state.site_mask
        self.metrics["enabled_fraction"][:] = enabled.to(torch.float32)
        self.metrics["active_site_fraction"][:] = active_sites.to(torch.float32).mean(dim=-1)


@configclass
class SonicComplianceCommandCfg(CommandTermCfg):
    """Hydra configuration for :class:`SonicComplianceCommand`."""

    class_type: type = SonicComplianceCommand
    asset_name: str = "robot"
    reference_body_names: list[str] = dataclasses.MISSING
    site_names: list[str] = dataclasses.MISSING
    site_offsets_local_xyz: list[list[float]] | None = None
    anchor_body: str = dataclasses.MISSING
    num_future_frames: int = dataclasses.MISSING
    frame_kind: str = "heading_local"
    enabled: bool = False
    enabled_probability: float = 1.0
    site_probability: float = 0.5
    force_magnitude_range_n: tuple[float, float] = (0.0, 40.0)
    compliance_values_m_per_n: tuple[float, ...] = (0.0, 0.02, 0.05)
    force_duration_range_s: tuple[float, float] = (1.0, 3.0)
    pulse_interval_range_s: tuple[float, float] = (3.5, 6.0)
    force_rise_end: float = 0.2
    force_fall_start: float = 0.8
    max_active_sites: int = 2
    max_net_force_n: float = 30.0
    max_net_torque_nm: float = 20.0
    sampling_seed: int = 0
    max_displacement_m: float | None = 0.25
    target_damper_enabled: bool = False
    target_damper_alpha: float = 0.1
    resampling_time_range: tuple[float, float] = (math.inf, math.inf)
    debug_vis: bool = False

    def __post_init__(self) -> None:
        for name, probability in (
            ("enabled_probability", self.enabled_probability),
            ("site_probability", self.site_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        for name, value_range in (
            ("force_magnitude_range_n", self.force_magnitude_range_n),
            ("force_duration_range_s", self.force_duration_range_s),
            ("pulse_interval_range_s", self.pulse_interval_range_s),
        ):
            if len(value_range) != 2:
                raise ValueError(f"{name} must contain [min, max]")
            lower, upper = value_range
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"{name} must contain finite values")
            if lower < 0.0 or upper < lower:
                raise ValueError(f"{name} must satisfy 0 <= min <= max")
        if not self.compliance_values_m_per_n:
            raise ValueError("compliance_values_m_per_n must not be empty")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0
            for value in self.compliance_values_m_per_n
        ):
            raise ValueError(
                "compliance_values_m_per_n must contain finite non-negative values"
            )
        _frame_from_kind(self.frame_kind, anchor_body=self.anchor_body)
        if self.force_duration_range_s[0] <= 0.0:
            raise ValueError("force_duration_range_s minimum must be positive")
        if self.pulse_interval_range_s[0] <= 0.0:
            raise ValueError("pulse_interval_range_s minimum must be positive")
        if type(self.max_active_sites) is not int or self.max_active_sites <= 0:
            raise ValueError("max_active_sites must be a positive integer")
        if self.max_net_force_n <= 0.0 or self.max_net_torque_nm <= 0.0:
            raise ValueError("net wrench limits must be positive")
        if not 0.0 < self.force_rise_end <= self.force_fall_start < 1.0:
            raise ValueError("force schedule must satisfy 0 < rise_end <= fall_start < 1")
        if type(self.sampling_seed) is not int or self.sampling_seed < 0:
            raise ValueError("sampling_seed must be a non-negative integer")
