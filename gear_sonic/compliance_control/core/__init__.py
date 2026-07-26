"""Tracker-agnostic compliance core using the ``force_on_robot`` convention."""

from .math import clamp_vector_norm, encode_compliance_condition, stiffness_from_threshold
from .metrics import ComplianceMetrics, compute_compliance_metrics
from .reference_modifier import select_reference, virtual_force_from_reference_delta
from .residual import hard_gate_residual
from .schedule import event_envelope, sample_site_mask
from .schema import ComplianceSpec, ForceEventScheduleSpec

__all__ = [
    "ComplianceMetrics",
    "ComplianceSpec",
    "ForceEventScheduleSpec",
    "clamp_vector_norm",
    "compute_compliance_metrics",
    "encode_compliance_condition",
    "event_envelope",
    "hard_gate_residual",
    "sample_site_mask",
    "select_reference",
    "stiffness_from_threshold",
    "virtual_force_from_reference_delta",
]
