"""Deterministic, generator-owned sampling for asynchronous force pulses."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def reschedule_pulse_countdown_prevalidated(
    time_to_next_pulse: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    globally_enabled: bool,
    interval_range_s: tuple[float, float],
    generator: torch.Generator,
) -> None:
    """Reset selected countdown rows using only the caller-owned generator."""

    if env_ids.numel() == 0:
        return
    if not globally_enabled:
        time_to_next_pulse[env_ids] = torch.inf
        return
    lower, upper = interval_range_s
    unit_samples = torch.rand(
        env_ids.numel(),
        dtype=time_to_next_pulse.dtype,
        device=time_to_next_pulse.device,
        generator=generator,
    )
    time_to_next_pulse[env_ids] = unit_samples * (upper - lower) + lower


def reschedule_pulse_countdown_mask_prevalidated(
    time_to_next_pulse: torch.Tensor,
    due_mask: torch.Tensor,
    *,
    interval_range_s: tuple[float, float],
    generator: torch.Generator,
) -> None:
    """Reschedule fixed-size candidate rows selected by a trusted bool mask."""

    lower, upper = interval_range_s
    unit_samples = torch.rand(
        time_to_next_pulse.shape,
        dtype=time_to_next_pulse.dtype,
        device=time_to_next_pulse.device,
        generator=generator,
    )
    candidates = unit_samples * (upper - lower) + lower
    time_to_next_pulse.copy_(
        torch.where(due_mask, candidates, time_to_next_pulse)
    )


def advance_pulse_countdown_prevalidated(
    time_to_next_pulse: torch.Tensor,
    dt_s: float,
    *,
    globally_enabled: bool,
) -> torch.Tensor:
    """Advance countdowns and return a fixed-size due mask without global RNG."""

    if not globally_enabled:
        time_to_next_pulse.fill_(torch.inf)
        return torch.zeros_like(time_to_next_pulse, dtype=torch.bool)
    time_to_next_pulse -= dt_s
    return time_to_next_pulse <= 0.0


@dataclass(frozen=True, slots=True)
class CompliancePulseSamples:
    enabled: torch.Tensor
    site_mask: torch.Tensor
    compliance: torch.Tensor
    peak_force_on_robot_w: torch.Tensor
    duration_s: torch.Tensor


def sample_compliance_pulses(
    *,
    num_envs: int,
    num_sites: int,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
    globally_enabled: bool,
    enabled_probability: float,
    site_probability: float,
    force_magnitude_range_n: tuple[float, float],
    compliance_values_m_per_n: tuple[float, ...],
    duration_range_s: tuple[float, float],
    max_active_sites: int,
) -> CompliancePulseSamples:
    """Sample reproducible masks, force, discrete compliance, and duration."""

    if type(num_envs) is not int or num_envs < 0:
        raise ValueError("num_envs must be a non-negative integer")
    if type(num_sites) is not int or num_sites <= 0:
        raise ValueError("num_sites must be a positive integer")
    if type(max_active_sites) is not int or max_active_sites <= 0:
        raise ValueError("max_active_sites must be a positive integer")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    if not compliance_values_m_per_n:
        raise ValueError("compliance_values_m_per_n must not be empty")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value < 0.0
        for value in compliance_values_m_per_n
    ):
        raise ValueError(
            "compliance_values_m_per_n must contain finite non-negative values"
        )
    device = torch.device(device)
    compliance_values = torch.tensor(
        compliance_values_m_per_n,
        dtype=dtype,
        device=device,
    )
    return sample_compliance_pulses_prevalidated(
        num_envs=num_envs,
        num_sites=num_sites,
        device=device,
        dtype=dtype,
        generator=generator,
        globally_enabled=globally_enabled,
        enabled_probability=enabled_probability,
        site_probability=site_probability,
        force_magnitude_range_n=force_magnitude_range_n,
        compliance_values_m_per_n=compliance_values,
        duration_range_s=duration_range_s,
        max_active_sites=max_active_sites,
    )


def sample_compliance_pulses_prevalidated(
    *,
    num_envs: int,
    num_sites: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    globally_enabled: bool,
    enabled_probability: float,
    site_probability: float,
    force_magnitude_range_n: tuple[float, float],
    compliance_values_m_per_n: torch.Tensor,
    duration_range_s: tuple[float, float],
    max_active_sites: int,
) -> CompliancePulseSamples:
    """Sample lifecycle-validated pulse tensors without CUDA scalar extraction."""

    enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if globally_enabled:
        enabled = torch.rand(num_envs, device=device, generator=generator) < enabled_probability
    site_mask = (
        torch.rand(num_envs, num_sites, device=device, generator=generator) < site_probability
    )
    site_mask &= enabled.unsqueeze(-1)
    active_limit = min(max_active_sites, num_sites)
    selection_scores = torch.rand(
        num_envs,
        num_sites,
        device=device,
        generator=generator,
    )
    selection_scores = torch.where(
        site_mask,
        selection_scores,
        torch.full_like(selection_scores, -1.0),
    )
    selected_indices = selection_scores.topk(active_limit, dim=-1).indices
    limited_mask = torch.zeros_like(site_mask)
    limited_mask.scatter_(1, selected_indices, True)
    site_mask &= limited_mask
    missing_site = enabled & ~site_mask.any(dim=-1)
    fallback = torch.randint(
        num_sites,
        (num_envs,),
        device=device,
        generator=generator,
    )
    fallback_mask = torch.zeros_like(site_mask)
    fallback_mask.scatter_(1, fallback.unsqueeze(-1), True)
    site_mask |= missing_site.unsqueeze(-1) & fallback_mask

    directions = torch.randn(
        num_envs,
        num_sites,
        3,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    direction_norm = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    directions = directions / direction_norm.clamp_min(torch.finfo(dtype).eps)
    force_min, force_max = force_magnitude_range_n
    force_uniform = torch.rand(
        num_envs,
        num_sites,
        1,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    magnitudes = force_min + (force_max - force_min) * force_uniform

    compliance_indices = torch.randint(
        compliance_values_m_per_n.shape[0],
        (num_envs, num_sites),
        device=device,
        generator=generator,
    )
    compliance = compliance_values_m_per_n[compliance_indices]

    duration_min, duration_max = duration_range_s
    duration_uniform = torch.rand(
        num_envs,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    duration_s = duration_min + (duration_max - duration_min) * duration_uniform
    return CompliancePulseSamples(
        enabled=enabled,
        site_mask=site_mask,
        compliance=compliance,
        peak_force_on_robot_w=directions * magnitudes,
        duration_s=duration_s,
    )


def mask_requested_peak_forces(
    peak_force_on_robot_w: torch.Tensor,
    enabled: torch.Tensor,
    site_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero inactive peak forces before resultant-wrench limiting."""

    if peak_force_on_robot_w.ndim != 3 or peak_force_on_robot_w.shape[-1] != 3:
        raise ValueError("peak_force_on_robot_w must have shape [env, site, xyz]")
    expected_gate_shape = tuple(peak_force_on_robot_w.shape[:2])
    if enabled.dtype is not torch.bool or tuple(enabled.shape) != expected_gate_shape[:1]:
        raise ValueError("enabled must be bool with shape [env]")
    if site_mask.dtype is not torch.bool or tuple(site_mask.shape) != expected_gate_shape:
        raise ValueError("site_mask must be bool with shape [env, site]")
    if enabled.device != peak_force_on_robot_w.device or site_mask.device != enabled.device:
        raise ValueError("force, enabled, and site_mask must use the same device")
    return mask_requested_peak_forces_prevalidated(
        peak_force_on_robot_w,
        enabled,
        site_mask,
    )


def mask_requested_peak_forces_prevalidated(
    peak_force_on_robot_w: torch.Tensor,
    enabled: torch.Tensor,
    site_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask lifecycle-validated requested forces without host synchronization."""

    requested = enabled.unsqueeze(-1) & site_mask
    return torch.where(
        requested.unsqueeze(-1),
        peak_force_on_robot_w,
        torch.zeros_like(peak_force_on_robot_w),
    )


def limit_peak_forces_by_net_wrench(
    peak_force_on_robot_w: torch.Tensor,
    application_positions_w: torch.Tensor,
    wrench_origin_w: torch.Tensor,
    *,
    max_net_force_n: float,
    max_net_torque_nm: float,
) -> torch.Tensor:
    """Uniformly scale each environment to bounded resultant force and torque."""

    if peak_force_on_robot_w.ndim != 3 or peak_force_on_robot_w.shape[-1] != 3:
        raise ValueError("peak_force_on_robot_w must have shape [env, site, xyz]")
    if application_positions_w.shape != peak_force_on_robot_w.shape:
        raise ValueError("application_positions_w must match peak-force shape")
    if wrench_origin_w.shape != (peak_force_on_robot_w.shape[0], 3):
        raise ValueError("wrench_origin_w must have shape [env, xyz]")
    if (
        peak_force_on_robot_w.dtype != application_positions_w.dtype
        or peak_force_on_robot_w.dtype != wrench_origin_w.dtype
    ):
        raise TypeError("force, position, and origin tensors must share dtype")
    if (
        peak_force_on_robot_w.device != application_positions_w.device
        or peak_force_on_robot_w.device != wrench_origin_w.device
    ):
        raise ValueError("force, position, and origin tensors must share device")
    if max_net_force_n <= 0.0 or max_net_torque_nm <= 0.0:
        raise ValueError("net wrench limits must be positive")

    return limit_peak_forces_by_net_wrench_prevalidated(
        peak_force_on_robot_w,
        application_positions_w,
        wrench_origin_w,
        max_net_force_n=max_net_force_n,
        max_net_torque_nm=max_net_torque_nm,
    )


def limit_peak_forces_by_net_wrench_prevalidated(
    peak_force_on_robot_w: torch.Tensor,
    application_positions_w: torch.Tensor,
    wrench_origin_w: torch.Tensor,
    *,
    max_net_force_n: float,
    max_net_torque_nm: float,
) -> torch.Tensor:
    """Limit lifecycle-validated force rows without CUDA scalar extraction."""

    net_force = peak_force_on_robot_w.sum(dim=1)
    moment_arms = application_positions_w - wrench_origin_w.unsqueeze(1)
    net_torque = torch.linalg.cross(moment_arms, peak_force_on_robot_w, dim=-1).sum(dim=1)
    force_norm = torch.linalg.vector_norm(net_force, dim=-1)
    torque_norm = torch.linalg.vector_norm(net_torque, dim=-1)
    tiny = torch.finfo(peak_force_on_robot_w.dtype).tiny
    force_scale = max_net_force_n / force_norm.clamp_min(tiny)
    torque_scale = max_net_torque_nm / torque_norm.clamp_min(tiny)
    scale = torch.minimum(force_scale, torque_scale).clamp(max=1.0)
    return peak_force_on_robot_w * scale[:, None, None]
