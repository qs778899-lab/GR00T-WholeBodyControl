"""Resolve SONIC reference and articulation body spaces independently."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _validated_names(names: Sequence[str], *, space: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise TypeError(f"{space} body names must be a sequence, not a string")
    try:
        result = tuple(names)
    except TypeError as exc:
        raise TypeError(f"{space} body names must be a sequence") from exc
    if not result:
        raise ValueError(f"{space} body names must not be empty")
    for index, name in enumerate(result):
        if not isinstance(name, str):
            raise TypeError(f"{space} body name at index {index} must be a string")
        if not name.strip():
            raise ValueError(f"{space} body name at index {index} must not be empty")
    return result


def _unique_name_index(names: Sequence[str], *, space: str) -> dict[str, int]:
    validated = _validated_names(names, space=space)
    result: dict[str, int] = {}
    for index, name in enumerate(validated):
        if name in result:
            raise ValueError(f"duplicate body name {name!r} in {space} space")
        result[name] = index
    return result


def _resolve_requested(
    requested: Sequence[str],
    available: dict[str, int],
    *,
    space: str,
) -> tuple[int, ...]:
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"bodies missing from {space} space: {missing}")
    return tuple(available[name] for name in requested)


@dataclass(frozen=True)
class BodyIndexMap:
    """Parallel indices for reference-motion and articulation body arrays."""

    site_names: tuple[str, ...]
    reference_site_indices: tuple[int, ...]
    articulation_site_indices: tuple[int, ...]
    anchor_name: str
    reference_anchor_index: int
    articulation_anchor_index: int

    @property
    def num_sites(self) -> int:
        return len(self.site_names)


def resolve_body_index_map(
    reference_body_names: Sequence[str],
    articulation_body_names: Sequence[str],
    site_names: Sequence[str],
    anchor_name: str,
) -> BodyIndexMap:
    """Resolve exact names without assuming the two spaces share an ordering."""

    requested_sites = _validated_names(site_names, space="site")
    if not isinstance(anchor_name, str):
        raise TypeError("anchor_name must be a string")
    if not anchor_name.strip():
        raise ValueError("anchor_name must not be empty")
    if len(set(requested_sites)) != len(requested_sites):
        raise ValueError("site_names must be unique")
    if anchor_name in requested_sites:
        raise ValueError("anchor_name must be separate from compliance sites")

    reference_index = _unique_name_index(reference_body_names, space="reference")
    articulation_index = _unique_name_index(
        articulation_body_names,
        space="articulation",
    )
    reference_sites = _resolve_requested(
        requested_sites,
        reference_index,
        space="reference",
    )
    articulation_sites = _resolve_requested(
        requested_sites,
        articulation_index,
        space="articulation",
    )
    reference_anchor = _resolve_requested(
        (anchor_name,),
        reference_index,
        space="reference",
    )[0]
    articulation_anchor = _resolve_requested(
        (anchor_name,),
        articulation_index,
        space="articulation",
    )[0]
    return BodyIndexMap(
        site_names=requested_sites,
        reference_site_indices=reference_sites,
        articulation_site_indices=articulation_sites,
        anchor_name=anchor_name,
        reference_anchor_index=reference_anchor,
        articulation_anchor_index=articulation_anchor,
    )
