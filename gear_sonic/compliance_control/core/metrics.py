"""Portable compliance and reference-preservation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch

from .reference_modifier import (
    _expanded_enabled_mask,
    _expanded_site_mask,
    _validate_reference_pair,
    select_reference,
)


@dataclass(frozen=True)
class ComplianceMetrics:
    """Metrics over the site axis, preserving optional leading axes.

    ``inactive_reference_drift`` measures candidate compliant-reference
    pollution *before selection*: it compares ``compliant_reference`` with
    ``original_reference`` wherever the effective active mask is false.  It is
    therefore capable of detecting an adapter that modified inactive targets,
    rather than proving a tautological zero from the already-selected target.
    """

    original_tracking_error: torch.Tensor
    selected_tracking_error: torch.Tensor
    active_reference_yield: torch.Tensor
    inactive_reference_drift: torch.Tensor
    peak_virtual_force: torch.Tensor
    active_site_fraction: torch.Tensor


def _masked_site_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def _validate_metric_tensor(
    name: str,
    tensor: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must have the same shape as the references")
    if tensor.dtype != reference.dtype:
        raise TypeError(f"{name} must have the same dtype as the references")
    if tensor.device != reference.device:
        raise ValueError(f"{name} must be on the same device as the references")
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must have a real floating dtype")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")


def compute_compliance_metrics(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
    actual: torch.Tensor,
    virtual_force: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    enabled: torch.Tensor | bool | Real = True,
) -> ComplianceMetrics:
    """Compute hard-gated metrics in the adapter-supplied Cartesian frame.

    Inputs have shape ``[batch, ..., sites, 3]``.  Outputs retain all leading
    dimensions before ``sites`` (for example ``[batch, future]``).  Global
    ``enabled=false`` makes the effective active mask empty even if a stale
    site mask is supplied.
    """

    _validate_reference_pair(original_reference, compliant_reference)
    _validate_metric_tensor("original_reference", original_reference, original_reference)
    _validate_metric_tensor("compliant_reference", compliant_reference, original_reference)
    _validate_metric_tensor("actual", actual, original_reference)
    _validate_metric_tensor("virtual_force", virtual_force, original_reference)

    site_mask = _expanded_site_mask(active_mask, original_reference)
    global_gate = _expanded_enabled_mask(enabled, original_reference)
    effective_active_mask = site_mask & global_gate
    selected_reference = select_reference(
        original_reference,
        compliant_reference,
        active_mask,
        enabled=enabled,
    )

    original_error = torch.linalg.vector_norm(actual - original_reference, dim=-1)
    selected_error = torch.linalg.vector_norm(actual - selected_reference, dim=-1)
    candidate_delta = torch.linalg.vector_norm(
        compliant_reference - original_reference,
        dim=-1,
    )
    force_norm = torch.linalg.vector_norm(virtual_force, dim=-1)
    gated_force_norm = torch.where(
        effective_active_mask,
        force_norm,
        torch.zeros_like(force_norm),
    )
    inactive_mask = ~effective_active_mask

    return ComplianceMetrics(
        original_tracking_error=original_error.mean(dim=-1),
        selected_tracking_error=selected_error.mean(dim=-1),
        active_reference_yield=_masked_site_mean(candidate_delta, effective_active_mask),
        inactive_reference_drift=_masked_site_mean(candidate_delta, inactive_mask),
        peak_virtual_force=gated_force_norm.max(dim=-1).values,
        active_site_fraction=effective_active_mask.to(original_reference.dtype).mean(dim=-1),
    )
