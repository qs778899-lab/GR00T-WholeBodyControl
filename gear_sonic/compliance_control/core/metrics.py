"""Portable tensor metrics for compliance-response regression tests."""

from dataclasses import dataclass

import torch

from .math import expand_compliance
from .schema import (
    ComplianceTargetSpec,
    expand_hard_gate,
    expand_site_mask,
    validate_position_tensor,
)


@dataclass(frozen=True, slots=True)
class ComplianceResponseMetrics:
    """Displacement summaries over truly compliance-exposed target slots."""

    mean_displacement_m: torch.Tensor
    max_displacement_m: torch.Tensor
    per_site_mean_displacement_m: torch.Tensor
    active_fraction: torch.Tensor


def summarize_compliance_response(
    reference_positions: torch.Tensor,
    compliant_positions: torch.Tensor,
    *,
    spec: ComplianceTargetSpec,
    compliance: torch.Tensor,
    enabled: bool | torch.Tensor = True,
    site_mask: torch.Tensor | None = None,
) -> ComplianceResponseMetrics:
    """Summarize displacement where gate, site mask, and compliance are active."""

    batch, future, sites = validate_position_tensor(
        reference_positions,
        name="reference_positions",
        expected_sites=spec.num_sites,
    )
    validate_position_tensor(
        compliant_positions,
        name="compliant_positions",
        expected_sites=spec.num_sites,
    )
    if compliant_positions.shape != reference_positions.shape:
        raise ValueError("compliant_positions shape must match reference_positions")
    if compliant_positions.dtype != reference_positions.dtype:
        raise TypeError("compliant_positions dtype must match reference_positions")
    if compliant_positions.device != reference_positions.device:
        raise ValueError("compliant_positions device must match reference_positions")
    compliance_expanded = expand_compliance(
        compliance,
        reference_positions,
        batch=batch,
        future=future,
        sites=sites,
    )

    if site_mask is None:
        requested_mask = torch.ones(
            (batch, future, sites),
            dtype=torch.bool,
            device=reference_positions.device,
        )
    else:
        requested_mask = expand_site_mask(
            site_mask,
            batch=batch,
            future=future,
            sites=sites,
            device=reference_positions.device,
        )
    if isinstance(enabled, bool):
        enabled_mask = torch.full(
            (batch, future, sites),
            enabled,
            dtype=torch.bool,
            device=reference_positions.device,
        )
    elif isinstance(enabled, torch.Tensor):
        enabled_mask = expand_hard_gate(
            enabled,
            batch=batch,
            future=future,
            sites=sites,
            device=reference_positions.device,
        )
    else:
        raise TypeError("enabled must be a bool or torch.Tensor")
    compliance_active = (compliance_expanded > 0.0).any(dim=-1)
    exposure_mask = requested_mask & enabled_mask & compliance_active

    displacement = torch.linalg.vector_norm(
        compliant_positions - reference_positions,
        dim=-1,
    )
    mask_float = exposure_mask.to(displacement.dtype)
    counts_per_site = mask_float.sum(dim=(0, 1))
    sums_per_site = (displacement * mask_float).sum(dim=(0, 1))
    per_site_mean = torch.where(
        counts_per_site > 0,
        sums_per_site / counts_per_site.clamp_min(1.0),
        torch.zeros_like(sums_per_site),
    )

    active_count = mask_float.sum()
    total_displacement = (displacement * mask_float).sum()
    mean_displacement = torch.where(
        active_count > 0,
        total_displacement / active_count.clamp_min(1.0),
        torch.zeros_like(total_displacement),
    )
    max_displacement = torch.where(exposure_mask, displacement, 0.0).amax()

    return ComplianceResponseMetrics(
        mean_displacement_m=mean_displacement,
        max_displacement_m=max_displacement,
        per_site_mean_displacement_m=per_site_mean,
        active_fraction=mask_float.mean(),
    )
