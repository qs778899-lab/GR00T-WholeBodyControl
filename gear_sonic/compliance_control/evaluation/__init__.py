"""Portable aligned-trace evaluation for compliance-control experiments."""

from .alignment import TraceAlignmentError, alignment_digest, assert_strict_alignment
from .io import load_trace_npz, write_report_json_atomic, write_trace_npz_atomic
from .metrics import compare_aligned_traces, evaluate_trace, evaluate_trial_suite
from .schema import EvaluationTrace, RegressionCriteria, TrialMode, TrialSpec

__all__ = [
    "EvaluationTrace",
    "RegressionCriteria",
    "TraceAlignmentError",
    "TrialMode",
    "TrialSpec",
    "alignment_digest",
    "assert_strict_alignment",
    "compare_aligned_traces",
    "evaluate_trace",
    "evaluate_trial_suite",
    "load_trace_npz",
    "write_report_json_atomic",
    "write_trace_npz_atomic",
]
