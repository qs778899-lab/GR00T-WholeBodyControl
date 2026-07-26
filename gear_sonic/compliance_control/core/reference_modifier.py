"""Portable reference selection and force-on-robot construction.

All position tensors are expressed in one common Cartesian frame selected by
the adapter.  The core neither chooses nor names that frame.  Returned vectors
use the ``force_on_robot`` sign convention and are expressed in the same frame.
"""

from __future__ import annotations

from numbers import Real

import torch

from .math import _binary_enabled_tensor, clamp_vector_norm, stiffness_from_threshold


def _validate_reference_pair(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
) -> None:
    for name, reference in (
        ("original_reference", original_reference),
        ("compliant_reference", compliant_reference),
    ):
        if not isinstance(reference, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if reference.ndim < 3 or reference.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [batch, ..., sites, 3]")
        if not reference.is_floating_point() or reference.is_complex():
            raise TypeError(f"{name} must have a real floating dtype")

    if original_reference.shape != compliant_reference.shape:
        raise ValueError("original_reference and compliant_reference must have identical shapes")
    if original_reference.dtype != compliant_reference.dtype:
        raise TypeError("original_reference and compliant_reference must have identical dtypes")
    if original_reference.device != compliant_reference.device:
        raise ValueError("original_reference and compliant_reference must be on the same device")


def _reshape_site_data(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Expand scalar, batch, per-site, or full-prefix data to ``reference[..., 0]``."""

    target_shape = reference.shape[:-1]
    batch_size = reference.shape[0]
    num_sites = reference.shape[-2]

    if tensor.ndim == 0:
        reshaped = tensor.reshape(*([1] * len(target_shape)))
    elif tensor.shape == target_shape:
        reshaped = tensor
    elif tensor.shape == (*target_shape, 1):
        reshaped = tensor.squeeze(-1)
    elif tensor.ndim == 1 and tensor.shape[0] == batch_size:
        reshaped = tensor.reshape(batch_size, *([1] * (len(target_shape) - 1)))
    elif tensor.shape == (batch_size, 1):
        reshaped = tensor.reshape(batch_size, *([1] * (len(target_shape) - 1)))
    elif tensor.shape == (batch_size, num_sites):
        reshaped = tensor.reshape(
            batch_size,
            *([1] * (len(target_shape) - 2)),
            num_sites,
        )
    elif tensor.shape == (batch_size, num_sites, 1):
        reshaped = tensor.reshape(
            batch_size,
            *([1] * (len(target_shape) - 2)),
            num_sites,
        )
    else:
        raise ValueError(
            f"{name} must be scalar, [batch], [batch, 1], [batch, sites], "
            "[batch, sites, 1], or match the full reference prefix"
        )
    return torch.broadcast_to(reshaped, target_shape)


def _expanded_site_mask(
    active_mask: torch.Tensor | bool,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(active_mask, bool):
        mask = torch.as_tensor(active_mask, device=reference.device)
    elif isinstance(active_mask, torch.Tensor):
        if active_mask.device != reference.device:
            raise ValueError("active_mask and references must be on the same device")
        if active_mask.dtype != torch.bool:
            raise TypeError("active_mask must have boolean dtype")
        mask = active_mask
    else:
        raise TypeError("active_mask must be a bool or boolean tensor")
    return _reshape_site_data(mask, reference, name="active_mask")


def _expanded_enabled_mask(
    enabled: torch.Tensor | bool | Real,
    reference: torch.Tensor,
) -> torch.Tensor:
    gate = _binary_enabled_tensor(enabled, device=reference.device)
    target_shape = reference.shape[:-1]
    batch_size = reference.shape[0]
    if gate.ndim == 0:
        gate = gate.reshape(*([1] * len(target_shape)))
    elif gate.ndim == 1 and gate.shape[0] == batch_size:
        gate = gate.reshape(batch_size, *([1] * (len(target_shape) - 1)))
    elif gate.shape == (batch_size, 1):
        gate = gate.reshape(batch_size, *([1] * (len(target_shape) - 1)))
    else:
        raise ValueError("enabled must be scalar, [batch], or [batch, 1]")
    return torch.broadcast_to(gate, target_shape)


def _expanded_site_value(
    value: torch.Tensor | Real,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != reference.device:
            raise ValueError(f"{name} and references must be on the same device")
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} and references must have identical dtypes")
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must have a real floating dtype")
        tensor = value
    elif isinstance(value, Real) and not isinstance(value, bool):
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    else:
        raise TypeError(f"{name} must be a real scalar or floating tensor")
    if not torch.isfinite(tensor).all() or (tensor < 0.0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    return _reshape_site_data(tensor, reference, name=name).unsqueeze(-1)


def _expanded_current_reference(
    current_reference: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(current_reference, torch.Tensor):
        raise TypeError("current_reference must be a tensor")
    if current_reference.dtype != reference.dtype:
        raise TypeError("current_reference and references must have identical dtypes")
    if current_reference.device != reference.device:
        raise ValueError("current_reference and references must be on the same device")
    if not current_reference.is_floating_point() or current_reference.is_complex():
        raise TypeError("current_reference must have a real floating dtype")
    if current_reference.shape == reference.shape:
        return current_reference
    expected_compact_shape = (reference.shape[0], reference.shape[-2], 3)
    if current_reference.shape != expected_compact_shape:
        raise ValueError(
            "current_reference must match the references or have shape [batch, sites, 3]"
        )
    reshape = (
        reference.shape[0],
        *([1] * (reference.ndim - 3)),
        reference.shape[-2],
        3,
    )
    return torch.broadcast_to(current_reference.reshape(reshape), reference.shape)


def select_reference(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
    active_mask: torch.Tensor | bool,
    *,
    enabled: torch.Tensor | bool | Real = True,
) -> torch.Tensor:
    """Select yielded targets only when the global gate and site are active.

    References have shape ``[batch, ..., sites, 3]`` and use an
    adapter-supplied common Cartesian frame.  A ``[batch, sites]`` mask is
    broadcast across optional dimensions such as future frames.  Global
    ``enabled=false`` overrides stale active masks and preserves the original
    reference bit-for-bit, with the same dtype and device.
    """

    _validate_reference_pair(original_reference, compliant_reference)
    mask = _expanded_site_mask(active_mask, original_reference)
    mask = mask & _expanded_enabled_mask(enabled, original_reference)
    return torch.where(mask.unsqueeze(-1), compliant_reference, original_reference)


def virtual_force_from_reference_delta(
    original_reference: torch.Tensor,
    compliant_reference: torch.Tensor,
    active_mask: torch.Tensor | bool,
    force_threshold_n: torch.Tensor | Real,
    *,
    current_reference: torch.Tensor | None = None,
    reference_displacement_m: float = 0.05,
    tracking_gain_n_per_m: torch.Tensor | Real = 100.0,
    tracking_force_cap_n: torch.Tensor | Real = 5.0,
    include_tracking_term: bool = True,
    enabled: torch.Tensor | bool | Real = True,
) -> torch.Tensor:
    """Construct bounded virtual force applied *to the robot*.

    Every position is in one adapter-supplied common Cartesian frame and the
    returned ``force_on_robot`` vector is in that same frame.  The full formula
    reproduced from ``motion_tracking`` is::

        nominal = clamp_norm((compliant - original) * Kp, threshold)
        tracking = clamp_norm((compliant - current) * tracking_gain, tracking_cap)
        force_on_robot = nominal + tracking

    where ``Kp = threshold / reference_displacement``.  Both terms are clamped
    separately, matching the source implementation.  Omitting
    ``current_reference`` or setting ``include_tracking_term=False`` preserves
    the nominal-only API.  A ``[batch, sites]`` mask and scalar, ``[batch]``, or
    ``[batch, sites]`` force parameters broadcast over optional future axes.
    """

    _validate_reference_pair(original_reference, compliant_reference)
    if not torch.isfinite(original_reference).all() or not torch.isfinite(
        compliant_reference
    ).all():
        raise ValueError("references used for force construction must be finite")
    if not isinstance(include_tracking_term, bool):
        raise TypeError("include_tracking_term must be bool")

    threshold = _expanded_site_value(
        force_threshold_n,
        original_reference,
        name="force_threshold_n",
    )
    stiffness = stiffness_from_threshold(threshold, reference_displacement_m)
    nominal_force = clamp_vector_norm(
        (compliant_reference - original_reference) * stiffness,
        threshold,
    )

    force_on_robot = nominal_force
    if include_tracking_term and current_reference is not None:
        current = _expanded_current_reference(current_reference, original_reference)
        if not torch.isfinite(current).all():
            raise ValueError("current_reference must be finite")
        tracking_gain = _expanded_site_value(
            tracking_gain_n_per_m,
            original_reference,
            name="tracking_gain_n_per_m",
        )
        tracking_cap = _expanded_site_value(
            tracking_force_cap_n,
            original_reference,
            name="tracking_force_cap_n",
        )
        tracking_force = clamp_vector_norm(
            (compliant_reference - current) * tracking_gain,
            tracking_cap,
        )
        force_on_robot = force_on_robot + tracking_force

    mask = _expanded_site_mask(active_mask, original_reference)
    mask = mask & _expanded_enabled_mask(enabled, original_reference)
    return torch.where(
        mask.unsqueeze(-1),
        force_on_robot,
        torch.zeros_like(force_on_robot),
    )
