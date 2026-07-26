"""Tracker-agnostic CHIP-style compliance-control core."""

from .damper import TargetDamper
from .math import apply_hindsight_target
from .metrics import ComplianceResponseMetrics, summarize_compliance_response
from .schedule import pyramid_phase_weight
from .schema import (
    COMPLIANCE_UNIT,
    FORCE_ON_ROBOT,
    CartesianFrameKind,
    CartesianFrameSpec,
    CartesianRotation,
    ComplianceTargetSpec,
    ForceSignConvention,
)

__all__ = [
    "COMPLIANCE_UNIT",
    "FORCE_ON_ROBOT",
    "CartesianFrameKind",
    "CartesianFrameSpec",
    "CartesianRotation",
    "ComplianceResponseMetrics",
    "ComplianceTargetSpec",
    "ForceSignConvention",
    "TargetDamper",
    "apply_hindsight_target",
    "pyramid_phase_weight",
    "summarize_compliance_response",
]
