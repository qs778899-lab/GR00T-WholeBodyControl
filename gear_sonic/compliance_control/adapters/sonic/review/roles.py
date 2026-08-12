"""Pinned nine-role CHIP review protocol at the SONIC boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

REVIEW_SITE_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True, slots=True)
class ReviewRole:
    """One deterministic checkpoint/control role in the review matrix."""

    name: str
    checkpoint_kind: str
    compliance_enabled: bool
    residual_enabled: bool
    external_force_enabled: bool
    active_site_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("role name must be a non-empty string")
        if self.checkpoint_kind not in {"official", "trained"}:
            raise ValueError("checkpoint_kind must be 'official' or 'trained'")
        for field_name in (
            "compliance_enabled",
            "residual_enabled",
            "external_force_enabled",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")
        if isinstance(self.active_site_names, (str, bytes)):
            raise TypeError("active_site_names must be a sequence")
        sites = tuple(self.active_site_names)
        if len(sites) != len(set(sites)) or not set(sites).issubset(REVIEW_SITE_NAMES):
            raise ValueError("active_site_names must be unique review wrist names")
        object.__setattr__(self, "active_site_names", sites)
        if self.residual_enabled and not self.compliance_enabled:
            raise ValueError("residual_enabled requires compliance_enabled")
        if self.external_force_enabled != bool(sites):
            raise ValueError("external_force_enabled must match active_site_names")
        if sites and not self.compliance_enabled:
            raise ValueError("active sites require compliance_enabled")

    @property
    def actor_hard_off(self) -> bool:
        """Whether the actor must receive an exact-zero compliance command."""

        return not self.residual_enabled


_ROLES = (
    ReviewRole("release_baseline", "official", False, False, False),
    ReviewRole("chip_hard_off", "trained", False, False, False),
    ReviewRole("enabled_no_contact", "trained", True, True, False),
    ReviewRole(
        "single_left_stiff",
        "trained",
        True,
        False,
        True,
        (REVIEW_SITE_NAMES[0],),
    ),
    ReviewRole(
        "single_left_compliant",
        "trained",
        True,
        True,
        True,
        (REVIEW_SITE_NAMES[0],),
    ),
    ReviewRole(
        "single_right_stiff",
        "trained",
        True,
        False,
        True,
        (REVIEW_SITE_NAMES[1],),
    ),
    ReviewRole(
        "single_right_compliant",
        "trained",
        True,
        True,
        True,
        (REVIEW_SITE_NAMES[1],),
    ),
    ReviewRole(
        "simultaneous_stiff",
        "trained",
        True,
        False,
        True,
        REVIEW_SITE_NAMES,
    ),
    ReviewRole(
        "simultaneous_compliant",
        "trained",
        True,
        True,
        True,
        REVIEW_SITE_NAMES,
    ),
)

_ROLE_BY_NAME = {role.name: role for role in _ROLES}
REVIEW_ROLE_NAMES = tuple(role.name for role in _ROLES)
REVIEW_COMPARISONS = (
    ("release_to_hard_off", "release_baseline", "chip_hard_off"),
    ("hard_off_to_no_contact", "chip_hard_off", "enabled_no_contact"),
    ("single_left", "single_left_stiff", "single_left_compliant"),
    ("single_right", "single_right_stiff", "single_right_compliant"),
    ("simultaneous", "simultaneous_stiff", "simultaneous_compliant"),
)


def get_review_role(name: str) -> ReviewRole:
    """Resolve one exact role name without aliases or inferred semantics."""

    if not isinstance(name, str):
        raise TypeError("role name must be a string")
    try:
        return _ROLE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unsupported review role: {name!r}") from exc


def assert_role_config(role: ReviewRole, config: Mapping[str, object]) -> None:
    """Require Hydra role metadata to match the code-owned protocol exactly."""

    if not isinstance(role, ReviewRole):
        raise TypeError("role must be a ReviewRole")
    expected = {
        "name": role.name,
        "checkpoint_kind": role.checkpoint_kind,
        "compliance_enabled": role.compliance_enabled,
        "residual_enabled": role.residual_enabled,
        "external_force_enabled": role.external_force_enabled,
        "active_site_names": list(role.active_site_names),
    }
    observed = {key: config[key] for key in expected}
    observed["active_site_names"] = list(observed["active_site_names"])  # type: ignore[arg-type]
    if observed != expected:
        raise AssertionError(f"Hydra role config differs from protocol: {observed!r}")
