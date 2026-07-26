"""Persistent, seeded per-environment state for the SONIC compliance command."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from ...core import sample_site_mask


@dataclass(frozen=True)
class ComplianceSamplingSpec:
    """Sampling settings independent of IsaacLab manager lifecycle details."""

    enable_probability: float = 0.75
    site_activation_probability: float = 0.5
    force_threshold_range_n: tuple[float, float] = (10.0, 20.0)
    reference_displacement_m: float = 0.05
    reference_offset_range_m: tuple[float, float] = (0.02, 0.05)

    def __post_init__(self) -> None:
        for name, probability in (
            ("enable_probability", self.enable_probability),
            ("site_activation_probability", self.site_activation_probability),
        ):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name, values, positive in (
            ("force_threshold_range_n", self.force_threshold_range_n, True),
            ("reference_offset_range_m", self.reference_offset_range_m, False),
        ):
            low, high = values
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"{name} must be finite")
            if (positive and low <= 0.0) or (not positive and low < 0.0) or high < low:
                raise ValueError(f"{name} must be ordered with a valid lower bound")
        if not math.isfinite(self.reference_displacement_m):
            raise ValueError("reference_displacement_m must be finite")
        if self.reference_displacement_m <= 0.0:
            raise ValueError("reference_displacement_m must be positive")


class ComplianceCommandState:
    """Own persistent sampling, reference, force, and reset-safe buffers."""

    def __init__(
        self,
        num_envs: int,
        num_sites: int,
        num_future_frames: int,
        sampling: ComplianceSamplingSpec,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        if num_envs <= 0 or num_sites <= 0 or num_future_frames <= 0:
            raise ValueError("environment, site, and future-frame counts must be positive")
        if not dtype.is_floating_point:
            raise TypeError("state dtype must be floating")
        self.num_envs = int(num_envs)
        self.num_sites = int(num_sites)
        self.num_future_frames = int(num_future_frames)
        self.sampling = sampling
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

        self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.active_site_mask = torch.zeros(
            (self.num_envs, self.num_sites),
            dtype=torch.bool,
            device=self.device,
        )
        self.force_threshold_n = torch.zeros(
            self.num_envs,
            dtype=self.dtype,
            device=self.device,
        )
        self.stiffness_n_per_m = torch.zeros_like(self.force_threshold_n)
        self._condition = torch.zeros(
            (self.num_envs, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.reference_offset_common = torch.zeros(
            (self.num_envs, self.num_sites, 3),
            dtype=self.dtype,
            device=self.device,
        )
        reference_shape = (self.num_envs, self.num_future_frames, self.num_sites, 3)
        self.original_reference_common = torch.zeros(
            reference_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.compliant_reference_common = torch.zeros_like(self.original_reference_common)
        self.current_reference_common = torch.zeros(
            (self.num_envs, self.num_sites, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.force_common_future = torch.zeros_like(self.original_reference_common)
        self.site_force_world = torch.zeros_like(self.current_reference_common)
        self.site_torque_world = torch.zeros_like(self.current_reference_common)
        self.anchor_force_world = torch.zeros(
            (self.num_envs, 3),
            dtype=self.dtype,
            device=self.device,
        )
        self.anchor_torque_world = torch.zeros_like(self.anchor_force_world)

    @property
    def condition(self) -> torch.Tensor:
        return self._condition

    def _env_ids_tensor(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
    ) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            if env_ids != slice(None):
                raise ValueError("only slice(None) is supported for env_ids")
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            result = env_ids.to(device=self.device, dtype=torch.long)
        else:
            result = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if result.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        if result.numel() and ((result < 0).any() or (result >= self.num_envs).any()):
            raise IndexError("env_ids are out of range")
        return result

    def clear_dynamic(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Clear every reference and wrench buffer that could leak across reset."""

        ids = self._env_ids_tensor(env_ids)
        for tensor in (
            self.original_reference_common,
            self.compliant_reference_common,
            self.current_reference_common,
            self.force_common_future,
            self.site_force_world,
            self.site_torque_world,
            self.anchor_force_world,
            self.anchor_torque_world,
        ):
            tensor[ids] = 0.0

    def disable(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set the operationally-off state without random sampling."""

        ids = self._env_ids_tensor(env_ids)
        self.clear_dynamic(ids)
        self.enabled[ids] = False
        self.active_site_mask[ids] = False
        self.force_threshold_n[ids] = 0.0
        self.stiffness_n_per_m[ids] = 0.0
        self._condition[ids] = 0.0
        self.reference_offset_common[ids] = 0.0

    def resample(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Clear stale dynamics, then deterministically sample persistent command state."""

        ids = self._env_ids_tensor(env_ids)
        self.clear_dynamic(ids)
        if ids.numel() == 0:
            return

        self.enabled[ids] = (
            torch.rand(ids.numel(), device=self.device, generator=self.generator)
            < self.sampling.enable_probability
        )
        threshold_low, threshold_high = self.sampling.force_threshold_range_n
        threshold = torch.rand(
            ids.numel(),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        threshold = threshold * (threshold_high - threshold_low) + threshold_low
        self.force_threshold_n[ids] = threshold
        self.stiffness_n_per_m[ids] = threshold / self.sampling.reference_displacement_m
        enabled_value = self.enabled[ids].to(self.dtype)
        self._condition[ids] = torch.stack(
            (
                enabled_value,
                enabled_value * self.force_threshold_n[ids],
                enabled_value * self.stiffness_n_per_m[ids],
            ),
            dim=-1,
        )
        self.active_site_mask[ids] = sample_site_mask(
            self.enabled[ids],
            self.num_sites,
            site_activation_probability=self.sampling.site_activation_probability,
            generator=self.generator,
        )

        direction = torch.randn(
            (ids.numel(), self.num_sites, 3),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        direction = direction / torch.linalg.vector_norm(
            direction,
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        offset_low, offset_high = self.sampling.reference_offset_range_m
        magnitude = torch.rand(
            (ids.numel(), self.num_sites, 1),
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        magnitude = magnitude * (offset_high - offset_low) + offset_low
        sampled_offset = direction * magnitude
        self.reference_offset_common[ids] = torch.where(
            self.active_site_mask[ids].unsqueeze(-1),
            sampled_offset,
            torch.zeros_like(sampled_offset),
        )

    def sample_resampling_time(
        self,
        num_samples: int,
        resampling_time_range: tuple[float, float],
    ) -> torch.Tensor:
        """Sample command durations without consuming PyTorch's global RNG."""

        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        low, high = resampling_time_range
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
            raise ValueError("resampling_time_range must be finite, positive, and ordered")
        samples = torch.rand(
            num_samples,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )
        return samples * (high - low) + low

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Reset state with fresh persistent parameters and zero physical outputs."""

        self.resample(env_ids)
