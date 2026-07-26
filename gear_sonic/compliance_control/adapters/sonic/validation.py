"""Construction-time validation for configured SONIC adapter geometry."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def site_body_offsets_tensor(
    offsets: Sequence[Sequence[float]] | torch.Tensor,
    *,
    num_sites: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return finite body-local offsets with exact shape ``[sites, 3]``."""

    if isinstance(offsets, (str, bytes)):
        raise TypeError("site_body_offsets must be a numeric sequence, not a string")
    if num_sites <= 0:
        raise ValueError("num_sites must be positive")
    if not dtype.is_floating_point or dtype.is_complex:
        raise TypeError("site_body_offsets dtype must be real floating")
    try:
        raw = torch.as_tensor(offsets, device=device)
    except (TypeError, ValueError) as exc:
        raise ValueError("site_body_offsets must be a rectangular numeric array") from exc
    if raw.shape != (num_sites, 3):
        raise ValueError(f"site_body_offsets must have shape [{num_sites}, 3]")
    if raw.is_complex() or raw.dtype == torch.bool:
        raise TypeError("site_body_offsets must be real")
    if not torch.isfinite(raw).all():
        raise ValueError("site_body_offsets must be finite")
    return raw.to(dtype=dtype)
