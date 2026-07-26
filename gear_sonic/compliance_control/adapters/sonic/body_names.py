"""Resolve named compliance sites in distinct SONIC index spaces."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ...core import CartesianFrameSpec, ComplianceTargetSpec


class SiteIndexSpace(str, Enum):
    """Runtime owner of a resolved body index."""

    REFERENCE = "reference"
    ARTICULATION = "articulation"


def _normalize_names(names: Sequence[str], *, field_name: str) -> tuple[str, ...]:
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
class NamedSiteIndices:
    """One ordered site selection resolved in one explicit index space."""

    index_space: SiteIndexSpace
    site_names: tuple[str, ...]
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.index_space, SiteIndexSpace):
            raise TypeError("index_space must be a SiteIndexSpace")
        names = _normalize_names(self.site_names, field_name="site_names")
        indices = tuple(self.indices)
        if len(indices) != len(names):
            raise ValueError("indices and site_names must have identical lengths")
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("indices must contain non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("indices must be unique within an index space")
        object.__setattr__(self, "site_names", names)
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True, slots=True)
class SonicComplianceSites:
    """Same ordered sites resolved independently for reference and articulation."""

    spec: ComplianceTargetSpec
    reference: NamedSiteIndices
    articulation: NamedSiteIndices

    def __post_init__(self) -> None:
        if self.reference.index_space is not SiteIndexSpace.REFERENCE:
            raise ValueError("reference selection must use the reference index space")
        if self.articulation.index_space is not SiteIndexSpace.ARTICULATION:
            raise ValueError("articulation selection must use the articulation index space")
        if self.reference.site_names != self.spec.site_names:
            raise ValueError("reference site_names must exactly match spec.site_names")
        if self.articulation.site_names != self.spec.site_names:
            raise ValueError("articulation site_names must exactly match spec.site_names")

    @property
    def reference_indices(self) -> tuple[int, ...]:
        return self.reference.indices

    @property
    def articulation_indices(self) -> tuple[int, ...]:
        return self.articulation.indices


def resolve_site_indices(
    available_body_names: Sequence[str],
    requested_site_names: Sequence[str],
    *,
    index_space: SiteIndexSpace,
) -> NamedSiteIndices:
    """Resolve ordered names inside exactly one caller-declared index space."""

    if not isinstance(index_space, SiteIndexSpace):
        raise TypeError("index_space must be a SiteIndexSpace")
    available = _normalize_names(available_body_names, field_name="available_body_names")
    requested = _normalize_names(requested_site_names, field_name="requested_site_names")
    index_by_name = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
        raise ValueError(
            f"requested compliance sites are missing from {index_space.value}: {missing}"
        )
    return NamedSiteIndices(
        index_space=index_space,
        site_names=requested,
        indices=tuple(index_by_name[name] for name in requested),
    )


def resolve_compliance_sites(
    reference_body_names: Sequence[str],
    articulation_body_names: Sequence[str],
    requested_site_names: Sequence[str],
    *,
    target_frame: CartesianFrameSpec,
    force_frame: CartesianFrameSpec,
    max_displacement_m: float | None = None,
) -> SonicComplianceSites:
    """Resolve a common site order independently in both SONIC index spaces."""

    spec = ComplianceTargetSpec(
        site_names=requested_site_names,
        target_frame=target_frame,
        force_frame=force_frame,
        max_displacement_m=max_displacement_m,
    )
    reference = resolve_site_indices(
        reference_body_names,
        spec.site_names,
        index_space=SiteIndexSpace.REFERENCE,
    )
    articulation = resolve_site_indices(
        articulation_body_names,
        spec.site_names,
        index_space=SiteIndexSpace.ARTICULATION,
    )
    return SonicComplianceSites(
        spec=spec,
        reference=reference,
        articulation=articulation,
    )
