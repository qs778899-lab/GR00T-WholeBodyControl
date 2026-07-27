"""Tracker-agnostic CHIP-style compliance-control core."""

from .damper import TargetDamper
from .evaluation import (
    AlignedTrackingTrace,
    PairedComplianceResponseMetrics,
    PairedEvaluationResult,
    PairedEvaluationThresholds,
    TrackingComplianceMetrics,
    compare_aligned_tracking_traces,
    summarize_tracking_trace,
)
from .math import apply_hindsight_target, apply_hindsight_target_prevalidated
from .metrics import ComplianceResponseMetrics, summarize_compliance_response
from .residual import ComplianceResidualLayout, ComplianceResidualMLP
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
    "AlignedTrackingTrace",
    "ComplianceResponseMetrics",
    "ComplianceResidualLayout",
    "ComplianceResidualMLP",
    "ComplianceTargetSpec",
    "ForceSignConvention",
    "PairedComplianceResponseMetrics",
    "PairedEvaluationResult",
    "PairedEvaluationThresholds",
    "TargetDamper",
    "TrackingComplianceMetrics",
    "apply_hindsight_target",
    "apply_hindsight_target_prevalidated",
    "compare_aligned_tracking_traces",
    "pyramid_phase_weight",
    "summarize_compliance_response",
    "summarize_tracking_trace",
]
