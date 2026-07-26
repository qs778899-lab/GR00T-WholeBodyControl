"""Portable tensor math for CHIP-style hindsight compliant targets."""

import torch

from .schema import (
    ComplianceTargetSpec,
    expand_hard_gate,
    expand_site_mask,
    validate_position_tensor,
    validate_tensor_compatibility,
)


def _expand_forces(
    external_forces: torch.Tensor,
    reference_positions: torch.Tensor,
    *,
    batch: int,
    future: int,
    sites: int,
) -> torch.Tensor:
    validate_tensor_compatibility(external_forces, reference_positions, name="external_forces")
    if external_forces.ndim == 3 and tuple(external_forces.shape) == (batch, sites, 3):
        return external_forces.view(batch, 1, sites, 3).expand(batch, future, sites, 3)
    if external_forces.ndim == 4 and tuple(external_forces.shape) == (
        batch,
        future,
        sites,
        3,
    ):
        return external_forces
    raise ValueError(
        "external_forces must have shape [batch, site, xyz] or "
        f"[batch, future, site, xyz]; got {tuple(external_forces.shape)}"
    )


def expand_compliance(
    compliance: torch.Tensor,
    reference_positions: torch.Tensor,
    *,
    batch: int,
    future: int,
    sites: int,
) -> torch.Tensor:
    validate_tensor_compatibility(compliance, reference_positions, name="compliance")
    if (compliance < 0.0).any():
        raise ValueError("compliance must be non-negative inverse stiffness")
    shape = tuple(compliance.shape)
    if compliance.ndim == 2 and shape == (batch, sites):
        return compliance.view(batch, 1, sites, 1).expand(batch, future, sites, 1)
    if compliance.ndim == 3:
        isotropic_future = shape == (batch, future, sites)
        anisotropic_static = shape == (batch, sites, 3)
        if isotropic_future and anisotropic_static:
            raise ValueError(
                "compliance shape is ambiguous between isotropic future and anisotropic static; "
                "use [batch, future, site, 1] or [batch, future, site, xyz]"
            )
        if isotropic_future:
            return compliance.unsqueeze(-1)
        if anisotropic_static:
            return compliance.view(batch, 1, sites, 3).expand(batch, future, sites, 3)
    if compliance.ndim == 4 and shape in {
        (batch, future, sites, 1),
        (batch, future, sites, 3),
    }:
        return compliance
    raise ValueError(
        "compliance must be isotropic [batch, site] / [batch, future, site] or "
        "anisotropic [batch, site, xyz] / [batch, future, site, xyz]; "
        f"got {shape}"
    )


def _limit_vector_norm(vectors: torch.Tensor, max_norm: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    safe_norms = norms.clamp_min(torch.finfo(vectors.dtype).tiny)
    scale = (max_norm / safe_norms).clamp(max=1.0)
    return vectors * scale


def apply_hindsight_target(
    reference_positions: torch.Tensor,
    external_forces: torch.Tensor | None,
    compliance: torch.Tensor | None,
    *,
    spec: ComplianceTargetSpec,
    enabled: bool | torch.Tensor = True,
    site_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply `g_hind = g_ref - C * f_robot` without mutating inputs.

    Args:
        reference_positions: Cartesian targets `[batch, future, site, xyz]`.
        external_forces: Force on the robot in `spec.force_frame`, shaped
            `[batch, site, xyz]` or `[batch, future, site, xyz]`.
        compliance: Isotropic inverse stiffness in metres per newton, shaped
            `[batch, site]` or `[batch, future, site]`, or Cartesian-axis
            anisotropic compliance shaped `[batch, site, xyz]` or
            `[batch, future, site, xyz]`. A trailing singleton xyz dimension is
            also accepted for explicitly disambiguated isotropic future data.
        spec: Ordered-site, coordinate-frame, sign, and unit contract.
        enabled: Global opt-in bool or mixed-batch boolean tensor shaped
            `[batch]` or `[batch, future, site]`. Global disabled mode needs no
            force tensors. Disabled tensor elements select the exact reference.
        site_mask: Optional boolean mask `[site]`, `[batch, site]`, or
            `[batch, future, site]` for selective compliant contacts.
    """

    if not isinstance(spec, ComplianceTargetSpec):
        raise TypeError("spec must be a ComplianceTargetSpec")
    if not isinstance(enabled, bool | torch.Tensor):
        raise TypeError("enabled must be a bool or torch.Tensor")

    batch, future, sites = validate_position_tensor(
        reference_positions,
        name="reference_positions",
        expected_sites=spec.num_sites,
    )
    if isinstance(enabled, bool) and not enabled:
        return reference_positions.clone()
    if external_forces is None or compliance is None:
        raise ValueError("enabled compliance requires external_forces and compliance tensors")

    forces = _expand_forces(
        external_forces,
        reference_positions,
        batch=batch,
        future=future,
        sites=sites,
    )
    compliance_expanded = expand_compliance(
        compliance,
        reference_positions,
        batch=batch,
        future=future,
        sites=sites,
    )
    displacement = compliance_expanded * forces

    if site_mask is not None:
        mask = expand_site_mask(
            site_mask,
            batch=batch,
            future=future,
            sites=sites,
            device=reference_positions.device,
        )
        displacement = torch.where(mask.unsqueeze(-1), displacement, 0.0)
    if spec.max_displacement_m is not None:
        displacement = _limit_vector_norm(displacement, spec.max_displacement_m)

    candidate = reference_positions - displacement
    if isinstance(enabled, torch.Tensor):
        hard_gate = expand_hard_gate(
            enabled,
            batch=batch,
            future=future,
            sites=sites,
            device=reference_positions.device,
        )
        return torch.where(hard_gate.unsqueeze(-1), candidate, reference_positions)
    return candidate
