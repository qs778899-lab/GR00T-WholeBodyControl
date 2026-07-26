"""Portable per-environment state for the SONIC compliance command."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from ...core import TargetDamper, pyramid_phase_weight
from .body_names import SonicComplianceSites


class SonicComplianceCommandState:
    """Own force, compliance, mask, and target-damper buffers.

    The class intentionally has no Isaac Lab dependency.  Simulator command and
    event terms own scheduling and wrench application, while observation terms
    consume immutable snapshots of these buffers.
    """

    def __init__(
        self,
        *,
        sites: SonicComplianceSites,
        num_envs: int,
        num_future_frames: int,
        device: torch.device | str,
        dtype: torch.dtype,
        target_damper_alpha: float,
        site_offsets_local: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(sites, SonicComplianceSites):
            raise TypeError("sites must be a SonicComplianceSites")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if type(num_future_frames) is not int or num_future_frames <= 0:
            raise ValueError("num_future_frames must be a positive integer")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")

        self.sites = sites
        self.num_envs = num_envs
        self.num_future_frames = num_future_frames
        self.device = torch.device(device)
        self.dtype = dtype
        if site_offsets_local is None:
            site_offsets_local = torch.zeros(
                sites.spec.num_sites,
                3,
                dtype=dtype,
                device=self.device,
            )
        if not isinstance(site_offsets_local, torch.Tensor):
            raise TypeError("site_offsets_local must be a torch.Tensor")
        if tuple(site_offsets_local.shape) != (sites.spec.num_sites, 3):
            raise ValueError("site_offsets_local must have shape [site, xyz]")
        if site_offsets_local.dtype != dtype or site_offsets_local.device != self.device:
            raise ValueError("site_offsets_local must use state dtype and device")
        if not torch.isfinite(site_offsets_local).all():
            raise ValueError("site_offsets_local must contain only finite values")
        self._site_offsets_local = site_offsets_local.detach().clone()
        self._enabled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._site_mask = torch.zeros(
            num_envs,
            sites.spec.num_sites,
            dtype=torch.bool,
            device=self.device,
        )
        self._compliance = torch.zeros(
            num_envs,
            sites.spec.num_sites,
            3,
            dtype=dtype,
            device=self.device,
        )
        self._force_on_robot_w = torch.zeros_like(self._compliance)
        self._peak_force_on_robot_w = torch.zeros_like(self._compliance)
        self._pulse_active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._pulse_elapsed_s = torch.zeros(num_envs, dtype=dtype, device=self.device)
        self._pulse_duration_s = torch.zeros(num_envs, dtype=dtype, device=self.device)
        self._target_damper = TargetDamper(target_damper_alpha)

    @property
    def enabled(self) -> torch.Tensor:
        return self._enabled.clone()

    @property
    def site_mask(self) -> torch.Tensor:
        return self._site_mask.clone()

    @property
    def compliance(self) -> torch.Tensor:
        return self._compliance.clone()

    @property
    def force_on_robot_w(self) -> torch.Tensor:
        return self._force_on_robot_w.clone()

    @property
    def peak_force_on_robot_w(self) -> torch.Tensor:
        return self._peak_force_on_robot_w.clone()

    @property
    def pulse_active(self) -> torch.Tensor:
        return self._pulse_active.clone()

    @property
    def pulse_elapsed_s(self) -> torch.Tensor:
        return self._pulse_elapsed_s.clone()

    @property
    def pulse_duration_s(self) -> torch.Tensor:
        return self._pulse_duration_s.clone()

    @property
    def normalized_pulse_phase(self) -> torch.Tensor:
        safe_duration = self._pulse_duration_s.clamp_min(torch.finfo(self.dtype).eps)
        return torch.where(
            self._pulse_active,
            self._pulse_elapsed_s / safe_duration,
            torch.zeros_like(self._pulse_elapsed_s),
        )

    @property
    def site_offsets_local(self) -> torch.Tensor:
        return self._site_offsets_local.clone()

    @property
    def damper_initialized(self) -> bool:
        return self._target_damper.initialized

    @property
    def damped_target_common(self) -> torch.Tensor:
        return self._target_damper.previous_target

    @property
    def actor_command(self) -> torch.Tensor:
        """Return only actor-safe enable/mask/compliance state, never true force."""

        return torch.cat(
            (
                self._enabled.to(self.dtype).unsqueeze(-1),
                self._site_mask.to(self.dtype),
                self._compliance.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )

    def _env_ids_tensor(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
    ) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            if env_ids != slice(None):
                raise TypeError("only slice(None) is supported for env_ids")
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1:
                raise ValueError("env_ids tensor must be one-dimensional")
            if env_ids.dtype is torch.bool:
                if tuple(env_ids.shape) != (self.num_envs,):
                    raise ValueError("boolean env_ids must have shape [num_envs]")
                ids = env_ids.nonzero(as_tuple=False).flatten().to(self.device)
            elif env_ids.dtype in (torch.int32, torch.int64):
                ids = env_ids.to(device=self.device, dtype=torch.long)
            else:
                raise TypeError("env_ids tensor must use bool, int32, or int64")
        else:
            if isinstance(env_ids, str | bytes):
                raise TypeError("env_ids must not be str or bytes")
            ids = torch.tensor(tuple(env_ids), device=self.device, dtype=torch.long)
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
            raise IndexError("env_ids contain an out-of-range environment index")
        if ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must be unique")
        return ids

    def set_samples(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        *,
        enabled: torch.Tensor,
        site_mask: torch.Tensor,
        compliance: torch.Tensor,
        force_on_robot_w: torch.Tensor,
    ) -> None:
        """Replace sampled buffers for selected environments."""

        ids = self._env_ids_tensor(env_ids)
        count = ids.numel()
        sites = self.sites.spec.num_sites
        if enabled.dtype is not torch.bool or tuple(enabled.shape) != (count,):
            raise ValueError(f"enabled must be bool with shape [{count}]")
        if site_mask.dtype is not torch.bool or tuple(site_mask.shape) != (count, sites):
            raise ValueError(f"site_mask must be bool with shape [{count}, {sites}]")
        for name, tensor in (
            ("enabled", enabled),
            ("site_mask", site_mask),
            ("compliance", compliance),
            ("force_on_robot_w", force_on_robot_w),
        ):
            if tensor.device != self.device:
                raise ValueError(f"{name} must use state device {self.device}")

        if tuple(compliance.shape) == (count, sites):
            compliance = compliance.unsqueeze(-1).expand(count, sites, 3)
        if tuple(compliance.shape) != (count, sites, 3):
            raise ValueError(f"compliance must have shape [{count}, {sites}] or [{count}, {sites}, 3]")
        if tuple(force_on_robot_w.shape) != (count, sites, 3):
            raise ValueError(f"force_on_robot_w must have shape [{count}, {sites}, 3]")
        for name, tensor in (
            ("compliance", compliance),
            ("force_on_robot_w", force_on_robot_w),
        ):
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")
            if tensor.dtype != self.dtype:
                raise TypeError(f"{name} must use state dtype {self.dtype}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        if (compliance < 0.0).any():
            raise ValueError("compliance must be non-negative")

        requested = enabled.unsqueeze(-1) & site_mask
        self._enabled[ids] = enabled
        self._site_mask[ids] = site_mask
        self._compliance[ids] = torch.where(
            requested.unsqueeze(-1),
            compliance,
            0.0,
        )
        self._force_on_robot_w[ids] = torch.where(
            requested.unsqueeze(-1),
            force_on_robot_w,
            0.0,
        )
        self._peak_force_on_robot_w[ids] = self._force_on_robot_w[ids]
        self._pulse_active[ids] = False
        self._pulse_elapsed_s[ids] = 0.0
        self._pulse_duration_s[ids] = 0.0

    def start_pulses(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        *,
        enabled: torch.Tensor,
        site_mask: torch.Tensor,
        compliance: torch.Tensor,
        peak_force_on_robot_w: torch.Tensor,
        duration_s: torch.Tensor,
    ) -> None:
        """Start zero-weight pulse(s); `advance_force_schedule` applies the profile."""

        ids = self._env_ids_tensor(env_ids)
        if duration_s.shape != (ids.numel(),):
            raise ValueError(f"duration_s must have shape [{ids.numel()}]")
        if duration_s.dtype != self.dtype or duration_s.device != self.device:
            raise ValueError("duration_s must use state dtype and device")
        if not torch.isfinite(duration_s).all() or (duration_s <= 0.0).any():
            raise ValueError("duration_s must be finite and positive")
        self.set_samples(
            ids,
            enabled=enabled,
            site_mask=site_mask,
            compliance=compliance,
            force_on_robot_w=peak_force_on_robot_w,
        )
        active = enabled & site_mask.any(dim=-1)
        self._pulse_active[ids] = active
        self._pulse_elapsed_s[ids] = 0.0
        self._pulse_duration_s[ids] = torch.where(active, duration_s, 0.0)
        self._force_on_robot_w[ids] = 0.0

    def advance_force_schedule(
        self,
        dt_s: float,
        *,
        rise_end: float = 0.2,
        fall_start: float = 0.8,
    ) -> torch.Tensor:
        """Advance asynchronous per-environment rise/hold/fall force pulses."""

        if not isinstance(dt_s, int | float) or not math.isfinite(float(dt_s)):
            raise TypeError("dt_s must be a finite real number")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        active = self._pulse_active.clone()
        self._pulse_elapsed_s = torch.where(
            active,
            self._pulse_elapsed_s + float(dt_s),
            self._pulse_elapsed_s,
        )
        safe_duration = self._pulse_duration_s.clamp_min(torch.finfo(self.dtype).eps)
        phase = self._pulse_elapsed_s / safe_duration
        weights = pyramid_phase_weight(
            phase,
            rise_end=rise_end,
            fall_start=fall_start,
        )
        self._force_on_robot_w = torch.where(
            active[:, None, None],
            self._peak_force_on_robot_w * weights[:, None, None],
            torch.zeros_like(self._force_on_robot_w),
        )
        finished = active & (self._pulse_elapsed_s >= self._pulse_duration_s)
        self._pulse_active[finished] = False
        self._enabled[finished] = False
        self._site_mask[finished] = False
        self._compliance[finished] = 0.0
        self._force_on_robot_w[finished] = 0.0
        self._peak_force_on_robot_w[finished] = 0.0
        self._pulse_elapsed_s[finished] = 0.0
        self._pulse_duration_s[finished] = 0.0
        return self.force_on_robot_w

    def set_applied_force_prevalidated(self, force_on_robot_w: torch.Tensor) -> None:
        """Store the final limited/applied force without CUDA scalar extraction."""

        requested = self._enabled.unsqueeze(-1) & self._site_mask
        self._force_on_robot_w = torch.where(
            requested.unsqueeze(-1),
            force_on_robot_w,
            torch.zeros_like(force_on_robot_w),
        )

    def cancel_all_prevalidated(self) -> None:
        """Cancel all force pulses when a host-side global config gate turns off."""

        self._enabled.zero_()
        self._site_mask.zero_()
        self._compliance.zero_()
        self._force_on_robot_w.zero_()
        self._peak_force_on_robot_w.zero_()
        self._pulse_active.zero_()
        self._pulse_elapsed_s.zero_()
        self._pulse_duration_s.zero_()

    def reset(
        self,
        current_eef_common: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | slice | None = None,
    ) -> None:
        """Clear command buffers and reset all/selected damper rows to current EEF."""

        expected_shape = (
            self.num_envs,
            self.num_future_frames,
            self.sites.spec.num_sites,
            3,
        )
        if not isinstance(current_eef_common, torch.Tensor):
            raise TypeError("current_eef_common must be a torch.Tensor")
        if tuple(current_eef_common.shape) != expected_shape:
            raise ValueError(f"current_eef_common must have shape {expected_shape}")
        if current_eef_common.dtype != self.dtype:
            raise TypeError("current_eef_common must use state dtype")
        if current_eef_common.device != self.device:
            raise ValueError("current_eef_common must use state device")
        if not torch.isfinite(current_eef_common).all():
            raise ValueError("current_eef_common must contain only finite values")

        ids = self._env_ids_tensor(env_ids)
        self._enabled[ids] = False
        self._site_mask[ids] = False
        self._compliance[ids] = 0.0
        self._force_on_robot_w[ids] = 0.0
        self._peak_force_on_robot_w[ids] = 0.0
        self._pulse_active[ids] = False
        self._pulse_elapsed_s[ids] = 0.0
        self._pulse_duration_s[ids] = 0.0

        if not self._target_damper.initialized:
            self._target_damper.reset(current_eef_common)
            return
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask[ids] = True
        self._target_damper.reset(current_eef_common, reset_mask=reset_mask)

    def update_damper(self, current_eef_common: torch.Tensor) -> torch.Tensor:
        """Advance the damper once, only at positively compliant requested sites."""

        if not self._target_damper.initialized:
            raise RuntimeError("reset state before updating the target damper")
        previous = self._target_damper.previous_target
        if current_eef_common.shape != previous.shape:
            raise ValueError("current_eef_common shape must match target-damper state")
        if current_eef_common.dtype != self.dtype:
            raise TypeError("current_eef_common must use state dtype")
        if current_eef_common.device != self.device:
            raise ValueError("current_eef_common must use state device")
        if not torch.isfinite(current_eef_common).all():
            raise ValueError("current_eef_common must contain only finite values")

        compliant_sites = (
            self._enabled[:, None, None]
            & self._site_mask[:, None, :]
            & (self._compliance > 0.0).any(dim=-1)[:, None, :]
        )
        compliant_sites = compliant_sites.expand(
            self.num_envs,
            self.num_future_frames,
            self.sites.spec.num_sites,
        )
        update_input = torch.where(
            compliant_sites.unsqueeze(-1),
            current_eef_common,
            previous,
        )
        return self._target_damper.update(update_input)

    def update_damper_prevalidated(self, current_eef_common: torch.Tensor) -> torch.Tensor:
        """Advance lifecycle-validated command-side damper state without host sync."""

        previous = self._target_damper.previous_target
        compliant_sites = (
            self._enabled[:, None, None]
            & self._site_mask[:, None, :]
            & (self._compliance > 0.0).any(dim=-1)[:, None, :]
        ).expand(
            self.num_envs,
            self.num_future_frames,
            self.sites.spec.num_sites,
        )
        update_input = torch.where(
            compliant_sites.unsqueeze(-1),
            current_eef_common,
            previous,
        )
        return self._target_damper.update_prevalidated(update_input)

    def seed_damper_sites(
        self,
        current_eef_common: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        active_site_mask: torch.Tensor,
    ) -> None:
        """Seed newly active sites to current EEF before their first observation."""

        if not self._target_damper.initialized:
            raise RuntimeError("reset state before seeding target-damper sites")
        expected_shape = (
            self.num_envs,
            self.num_future_frames,
            self.sites.spec.num_sites,
            3,
        )
        if tuple(current_eef_common.shape) != expected_shape:
            raise ValueError(f"current_eef_common must have shape {expected_shape}")
        if current_eef_common.dtype != self.dtype or current_eef_common.device != self.device:
            raise ValueError("current_eef_common must use state dtype and device")
        if not torch.isfinite(current_eef_common).all():
            raise ValueError("current_eef_common must contain only finite values")
        ids = self._env_ids_tensor(env_ids)
        expected_mask_shape = (ids.numel(), self.sites.spec.num_sites)
        if active_site_mask.dtype is not torch.bool:
            raise TypeError("active_site_mask must use torch.bool")
        if active_site_mask.device != self.device:
            raise ValueError("active_site_mask must use state device")
        if tuple(active_site_mask.shape) != expected_mask_shape:
            raise ValueError(f"active_site_mask must have shape {expected_mask_shape}")

        full_mask = torch.zeros(
            self.num_envs,
            self.sites.spec.num_sites,
            dtype=torch.bool,
            device=self.device,
        )
        full_mask[ids] = active_site_mask
        previous = self._target_damper.previous_target
        seeded = torch.where(
            full_mask[:, None, :, None],
            current_eef_common,
            previous,
        )
        self._target_damper.reset(seeded)
