"""Portable tensor and metadata contracts for universal tracking policies."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math

import torch

COMPLIANCE_UNIT = "m/N"


class CartesianFrameKind(str, Enum):
    """Supported Cartesian coordinate spaces at the portable-core boundary."""

    WORLD = "world"
    ANCHOR_LOCAL = "anchor_local"
    HEADING_LOCAL = "heading_local"


class CartesianRotation(str, Enum):
    """Rotation applied when converting world vectors into the named frame."""

    IDENTITY = "identity"
    FULL_3D = "full_3d"
    YAW_ONLY = "yaw_only"


class ForceSignConvention(str, Enum):
    """Physical meaning of the force vector used by hindsight math."""

    FORCE_ON_ROBOT = "force_on_robot"


FORCE_ON_ROBOT = ForceSignConvention.FORCE_ON_ROBOT


def _normalize_name_sequence(names: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(names, str | bytes):
        raise TypeError(f"{field_name} must be a sequence of names, not str or bytes")
    normalized = tuple(names)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one name")
    if any(not isinstance(name, str) or not name.strip() for name in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique and ordered")
    return normalized


@dataclass(frozen=True, slots=True)
class CartesianFrameSpec:
    """Structured Cartesian frame contract shared by targets and forces.

    `anchor` is a caller-defined semantic name, not a robot body index. Adapters
    own the transform into this frame and must use the declared rotation mode.
    """

    kind: CartesianFrameKind
    anchor: str | None
    rotation: CartesianRotation

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CartesianFrameKind):
            raise TypeError("kind must be a CartesianFrameKind")
        if not isinstance(self.rotation, CartesianRotation):
            raise TypeError("rotation must be a CartesianRotation")
        if self.anchor is not None and (not isinstance(self.anchor, str) or not self.anchor.strip()):
            raise ValueError("anchor must be None or a non-empty semantic name")

        if self.kind is CartesianFrameKind.WORLD:
            if self.anchor is not None or self.rotation is not CartesianRotation.IDENTITY:
                raise ValueError("world frame requires anchor=None and identity rotation")
        elif self.kind is CartesianFrameKind.ANCHOR_LOCAL:
            if self.anchor is None or self.rotation is not CartesianRotation.FULL_3D:
                raise ValueError("anchor_local frame requires an anchor and full_3d rotation")
        elif self.kind is CartesianFrameKind.HEADING_LOCAL:
            if self.anchor is None or self.rotation is not CartesianRotation.YAW_ONLY:
                raise ValueError("heading_local frame requires an anchor and yaw_only rotation")

    @classmethod
    def world(cls) -> "CartesianFrameSpec":
        return cls(
            kind=CartesianFrameKind.WORLD,
            anchor=None,
            rotation=CartesianRotation.IDENTITY,
        )

    @classmethod
    def anchor_local(cls, anchor: str) -> "CartesianFrameSpec":
        return cls(
            kind=CartesianFrameKind.ANCHOR_LOCAL,
            anchor=anchor,
            rotation=CartesianRotation.FULL_3D,
        )

    @classmethod
    def heading_local(cls, anchor: str) -> "CartesianFrameSpec":
        return cls(
            kind=CartesianFrameKind.HEADING_LOCAL,
            anchor=anchor,
            rotation=CartesianRotation.YAW_ONLY,
        )


@dataclass(frozen=True, slots=True)
class ComplianceTargetSpec:
    """Metadata for Cartesian compliant tracking targets.

    `site_names` defines the only authoritative site ordering. Adapters are
    responsible for converting tracker-specific body identifiers to this order.
    Both positions and forces must already be expressed in the named common frame.
    Compliance is inverse stiffness in metres per newton.
    """

    site_names: Sequence[str]
    target_frame: CartesianFrameSpec
    force_frame: CartesianFrameSpec
    compliance_unit: str = COMPLIANCE_UNIT
    force_sign_convention: ForceSignConvention = FORCE_ON_ROBOT
    max_displacement_m: float | None = None

    def __post_init__(self) -> None:
        site_names = _normalize_name_sequence(self.site_names, field_name="site_names")
        object.__setattr__(self, "site_names", site_names)

        if not isinstance(self.target_frame, CartesianFrameSpec):
            raise TypeError("target_frame must be a CartesianFrameSpec")
        if not isinstance(self.force_frame, CartesianFrameSpec):
            raise TypeError("force_frame must be a CartesianFrameSpec")
        if self.target_frame != self.force_frame:
            raise ValueError(
                "target and force frames must match before applying compliance: "
                f"{self.target_frame!r} != {self.force_frame!r}"
            )
        if self.compliance_unit != COMPLIANCE_UNIT:
            raise ValueError(
                f"compliance must be converted to {COMPLIANCE_UNIT!r}, got "
                f"{self.compliance_unit!r}"
            )
        if not isinstance(self.force_sign_convention, ForceSignConvention):
            raise TypeError("force_sign_convention must be a ForceSignConvention")
        if self.force_sign_convention is not FORCE_ON_ROBOT:
            raise ValueError(
                f"force sign must be {FORCE_ON_ROBOT!r}, got "
                f"{self.force_sign_convention!r}"
            )
        if self.max_displacement_m is not None:
            if not math.isfinite(self.max_displacement_m) or self.max_displacement_m <= 0.0:
                raise ValueError("max_displacement_m must be finite and positive")

    @property
    def num_sites(self) -> int:
        """Number of ordered tracking sites in the contract."""

        return len(self.site_names)

    @property
    def common_frame(self) -> CartesianFrameSpec:
        """Validated common Cartesian frame for reference positions and forces."""

        return self.target_frame


def validate_position_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_sites: int | None = None,
) -> tuple[int, int, int]:
    """Validate a `[batch, future, site, xyz]` floating-point tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [batch, future, site, xyz], got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != 3:
        raise ValueError(f"{name} xyz dimension must be 3, got {tensor.shape[-1]}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    batch, future, sites, _ = tensor.shape
    if expected_sites is not None and sites != expected_sites:
        raise ValueError(f"{name} has {sites} sites, expected {expected_sites}")
    return batch, future, sites


def validate_tensor_compatibility(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
) -> None:
    """Require a floating tensor on the reference dtype and device."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if tensor.dtype != reference.dtype:
        raise TypeError(f"{name} dtype {tensor.dtype} does not match {reference.dtype}")
    if tensor.device != reference.device:
        raise ValueError(f"{name} device {tensor.device} does not match {reference.device}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def expand_site_mask(
    site_mask: torch.Tensor,
    *,
    batch: int,
    future: int,
    sites: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand a boolean site mask to `[batch, future, site]`."""

    if not isinstance(site_mask, torch.Tensor):
        raise TypeError("site_mask must be a torch.Tensor")
    if site_mask.dtype is not torch.bool:
        raise TypeError("site_mask must use torch.bool")
    if site_mask.device != device:
        raise ValueError(f"site_mask device {site_mask.device} does not match {device}")

    if site_mask.ndim == 1 and tuple(site_mask.shape) == (sites,):
        return site_mask.view(1, 1, sites).expand(batch, future, sites)
    if site_mask.ndim == 2 and tuple(site_mask.shape) == (batch, sites):
        return site_mask.view(batch, 1, sites).expand(batch, future, sites)
    if site_mask.ndim == 3 and tuple(site_mask.shape) == (batch, future, sites):
        return site_mask
    raise ValueError(
        "site_mask must have shape [site], [batch, site], or "
        f"[batch, future, site]; got {tuple(site_mask.shape)}"
    )


def expand_hard_gate(
    enabled: torch.Tensor,
    *,
    batch: int,
    future: int,
    sites: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand a mixed stiff/compliant boolean gate to `[batch, future, site]`."""

    if enabled.dtype is not torch.bool:
        raise TypeError("enabled tensor must use torch.bool")
    if enabled.device != device:
        raise ValueError(f"enabled tensor device {enabled.device} does not match {device}")
    if enabled.ndim == 1 and tuple(enabled.shape) == (batch,):
        return enabled.view(batch, 1, 1).expand(batch, future, sites)
    if enabled.ndim == 3 and tuple(enabled.shape) == (batch, future, sites):
        return enabled
    raise ValueError(
        "enabled tensor must have shape [batch] or [batch, future, site]; "
        f"got {tuple(enabled.shape)}"
    )
