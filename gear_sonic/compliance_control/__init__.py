"""Portable compliance-control primitives and tracker-specific adapters.

The public core API remains available from this package, but is loaded lazily
so tracker-neutral reporting modules do not inherit the core tensor runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "COMPLIANCE_UNIT",
    "AlignedTrackingTrace",
    "FORCE_ON_ROBOT",
    "CartesianFrameKind",
    "CartesianFrameSpec",
    "CartesianRotation",
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


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".core", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
