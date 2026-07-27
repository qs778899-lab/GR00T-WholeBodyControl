"""Strict identity checks for paired evaluation traces."""

from __future__ import annotations

import hashlib

import numpy as np

from .schema import EvaluationTrace


class TraceAlignmentError(ValueError):
    """Raised when two traces do not describe exactly the same samples."""


def _update_names(digest: "hashlib._Hash", field_name: str, values: tuple[str, ...]) -> None:
    digest.update(field_name.encode("ascii"))
    digest.update(len(values).to_bytes(8, byteorder="little", signed=False))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)


def alignment_digest(trace: EvaluationTrace) -> str:
    """Return a stable digest of ordered sample and layout identities."""

    digest = hashlib.sha256()
    for field_name in ("motion_ids", "sequence_ids", "site_ids", "point_ids"):
        _update_names(digest, field_name, getattr(trace, field_name))
    for field_name in ("seed_ids", "frame_indices", "timestamps_s"):
        array = getattr(trace, field_name)
        digest.update(field_name.encode("ascii"))
        canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def assert_strict_alignment(reference: EvaluationTrace, candidate: EvaluationTrace) -> None:
    """Require exact motion/sequence/seed/frame/time and layout equality."""

    named_fields = (
        "motion_ids",
        "sequence_ids",
        "site_ids",
        "point_ids",
    )
    for field_name in named_fields:
        if getattr(reference, field_name) != getattr(candidate, field_name):
            raise TraceAlignmentError(f"unaligned {field_name}")
    for field_name in ("seed_ids", "frame_indices", "timestamps_s"):
        reference_array = getattr(reference, field_name)
        candidate_array = getattr(candidate, field_name)
        if (
            reference_array.dtype != candidate_array.dtype
            or reference_array.shape != candidate_array.shape
            or reference_array.tobytes(order="C") != candidate_array.tobytes(order="C")
        ):
            raise TraceAlignmentError(f"unaligned {field_name}")
