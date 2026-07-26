"""Thin SONIC adapters around the portable compliance-control core."""

from .body_names import (
    NamedSiteIndices,
    SiteIndexSpace,
    SonicComplianceSites,
    resolve_compliance_sites,
    resolve_site_indices,
)

__all__ = [
    "NamedSiteIndices",
    "SiteIndexSpace",
    "SonicComplianceSites",
    "resolve_compliance_sites",
    "resolve_site_indices",
]
