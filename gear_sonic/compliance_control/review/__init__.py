"""Portable aligned-trace evaluation for compliance-control experiments."""

from .alignment import TraceAlignmentError, alignment_digest, assert_strict_alignment
from .io import (
    load_trace_npz,
    load_trace_npz_with_sha256,
    write_report_json_atomic,
    write_trace_npz_atomic,
)
from .metrics import compare_aligned_traces, evaluate_trace, evaluate_trial_suite
from .schema import EvaluationTrace, RegressionCriteria, TrialMode, TrialSpec
from .suite import (
    MatchedInteractionSpec,
    ReviewSuiteSpec,
    assert_matched_stimulus,
    evaluate_matched_review_suite,
)
from .video import (
    ReviewPanelSpec,
    ReviewVideoSpec,
    build_review_video_manifest,
    probe_video_with_sha256,
    validate_review_video_manifest,
    write_review_video_manifest_atomic,
)

__all__ = [
    "EvaluationTrace",
    "RegressionCriteria",
    "MatchedInteractionSpec",
    "ReviewPanelSpec",
    "ReviewSuiteSpec",
    "ReviewVideoSpec",
    "TraceAlignmentError",
    "TrialMode",
    "TrialSpec",
    "alignment_digest",
    "assert_strict_alignment",
    "assert_matched_stimulus",
    "build_review_video_manifest",
    "compare_aligned_traces",
    "evaluate_trace",
    "evaluate_trial_suite",
    "evaluate_matched_review_suite",
    "load_trace_npz",
    "load_trace_npz_with_sha256",
    "probe_video_with_sha256",
    "validate_review_video_manifest",
    "write_report_json_atomic",
    "write_review_video_manifest_atomic",
    "write_trace_npz_atomic",
]
