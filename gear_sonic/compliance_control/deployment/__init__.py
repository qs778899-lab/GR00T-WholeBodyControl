"""Portable export and runtime support for optional action residuals.

This package intentionally knows only tensor/layout contracts.  Tracker,
robot, observation-name, and simulator details belong in adapters.
"""

from .export import ResidualExportSpec, export_action_residual_bundle
from .model import ExportableActionResidual
from .runtime import (
    ActionResidualRuntime,
    ArtifactExpectation,
    ReleaseArtifactPin,
    load_artifact_metadata,
    load_onnxruntime_session,
)
from .schema import (
    ARTIFACT_SCHEMA,
    INPUT_CONDITION,
    INPUT_RELEASE_CONTEXT,
    OUTPUT_ACTION_DELTA,
    metadata_digest,
    validate_metadata,
)

__all__ = [
    "ARTIFACT_SCHEMA",
    "INPUT_CONDITION",
    "INPUT_RELEASE_CONTEXT",
    "OUTPUT_ACTION_DELTA",
    "ActionResidualRuntime",
    "ArtifactExpectation",
    "ReleaseArtifactPin",
    "ExportableActionResidual",
    "ResidualExportSpec",
    "export_action_residual_bundle",
    "load_artifact_metadata",
    "load_onnxruntime_session",
    "metadata_digest",
    "validate_metadata",
]
