"""Optional action-residual runtime with a structurally hard off path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schema import (
    ARTIFACT_SCHEMA,
    INPUT_CONDITION,
    INPUT_RELEASE_CONTEXT,
    OUTPUT_ACTION_DELTA,
    metadata_digest,
    validate_identifier_layout,
    validate_metadata,
    validate_sha256,
)


class ResidualSession(Protocol):
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray],
    ) -> Sequence[np.ndarray]: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReleaseArtifactPin:
    """Caller-owned identity for one unchanged release-side artifact."""

    name: str
    path: str | os.PathLike[str]
    sha256: str


@dataclass(frozen=True)
class ArtifactExpectation:
    """Host-pinned contract; callers must not trust artifact-authored values."""

    metadata_sha256: str
    checkpoint_sha256: str
    checkpoint_global_step: int
    release_context_width: int
    condition_width: int
    action_layout: tuple[str, ...]
    site_layout: tuple[str, ...]
    max_abs_delta: float
    release_artifacts: tuple[ReleaseArtifactPin, ...]
    schema: str = ARTIFACT_SCHEMA


def _validate_release_artifacts(
    pins: Sequence[ReleaseArtifactPin],
) -> tuple[ReleaseArtifactPin, ...]:
    if isinstance(pins, (str, bytes)) or not isinstance(pins, Sequence) or not pins:
        raise ValueError("release_artifacts must be a non-empty sequence")
    validated: list[ReleaseArtifactPin] = []
    names: set[str] = set()
    for index, pin in enumerate(pins):
        if not isinstance(pin, ReleaseArtifactPin):
            raise TypeError(f"release_artifacts[{index}] must be a ReleaseArtifactPin")
        if not pin.name or pin.name in names:
            raise ValueError("release artifact names must be unique and non-empty")
        names.add(pin.name)
        expected_sha = validate_sha256(
            pin.sha256, f"release_artifacts.{pin.name}.sha256"
        )
        path = Path(pin.path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"release artifact {pin.name} must be a regular non-symlink file"
            )
        path = path.resolve(strict=True)
        if _file_sha256(path) != expected_sha:
            raise ValueError(
                f"release artifact {pin.name} differs from the host-pinned base"
            )
        validated.append(
            ReleaseArtifactPin(name=pin.name, path=path, sha256=expected_sha)
        )
    return tuple(validated)


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    with path.open("r", encoding="utf-8") as source:
        return json.load(source, parse_constant=reject_constant)


def load_artifact_metadata(
    artifact_directory: str | os.PathLike[str],
    expectation: ArtifactExpectation,
) -> dict[str, Any]:
    """Verify external pins, metadata self-digest, model digest, and layouts."""

    _validate_release_artifacts(expectation.release_artifacts)
    artifact_directory = Path(artifact_directory)
    if not artifact_directory.is_dir() or artifact_directory.is_symlink():
        raise ValueError("artifact_directory must be a real directory")
    metadata_path = artifact_directory / "action_residual.metadata.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("artifact metadata must be a regular file")
    metadata = validate_metadata(_strict_json_load(metadata_path))
    expected_metadata_sha = validate_sha256(
        expectation.metadata_sha256, "expected metadata_sha256"
    )
    if metadata_digest(metadata) != expected_metadata_sha:
        raise ValueError("metadata digest differs from the host-pinned digest")
    if expectation.schema != ARTIFACT_SCHEMA or metadata["schema"] != expectation.schema:
        raise ValueError("artifact schema differs from the host contract")
    source = metadata["source_checkpoint"]
    if source["sha256"] != validate_sha256(
        expectation.checkpoint_sha256, "expected checkpoint_sha256"
    ):
        raise ValueError("artifact checkpoint digest differs from the host contract")
    if source["global_step"] != expectation.checkpoint_global_step:
        raise ValueError("artifact checkpoint step differs from the host contract")
    inputs = metadata["inputs"]
    if inputs[INPUT_RELEASE_CONTEXT]["width"] != expectation.release_context_width:
        raise ValueError("release-context width differs from the host contract")
    if inputs[INPUT_CONDITION]["width"] != expectation.condition_width:
        raise ValueError("condition width differs from the host contract")
    expected_actions = validate_identifier_layout(
        expectation.action_layout,
        "expected action_layout",
        expected_length=metadata["output"]["width"],
    )
    expected_sites = validate_identifier_layout(
        expectation.site_layout, "expected site_layout"
    )
    if tuple(metadata["action_layout"]) != expected_actions:
        raise ValueError("artifact action layout differs from the host contract")
    if tuple(metadata["site_layout"]) != expected_sites:
        raise ValueError("artifact site layout differs from the host contract")
    limit = metadata["network"]["max_abs_delta"]
    if (
        not math.isfinite(expectation.max_abs_delta)
        or expectation.max_abs_delta <= 0.0
        or limit != expectation.max_abs_delta
    ):
        raise ValueError("artifact action-delta limit differs from the host contract")
    model_path = artifact_directory / metadata["model"]["file"]
    if not model_path.is_file() or model_path.is_symlink():
        raise ValueError("artifact model must be a regular file")
    if _file_sha256(model_path) != metadata["model"]["sha256"]:
        raise ValueError("artifact model digest mismatch")
    return metadata


def load_onnxruntime_session(
    artifact_directory: str | os.PathLike[str],
    metadata: Mapping[str, Any],
):
    """Lazily import ONNX Runtime so portable hard-off imports need no ORT."""

    import onnxruntime as ort

    return ort.InferenceSession(
        str(Path(artifact_directory) / metadata["model"]["file"]),
        providers=["CPUExecutionProvider"],
    )


def _validated_gate(value: Any, leading_shape: tuple[int, ...]) -> np.ndarray:
    gate = np.asarray(value)
    if gate.shape == (*leading_shape, 1):
        gate = gate[..., 0]
    if gate.shape != leading_shape:
        raise ValueError(f"enabled gate must have shape {leading_shape}")
    if gate.dtype == np.bool_:
        return gate
    if not np.issubdtype(gate.dtype, np.number):
        raise TypeError("enabled gate must be boolean or numeric binary")
    if not np.isfinite(gate).all() or not np.logical_or(gate == 0, gate == 1).all():
        raise ValueError("enabled gate must contain only finite binary values")
    return gate.astype(np.bool_, copy=False)


class ActionResidualRuntime:
    """Compose an optional artifact delta onto an unchanged release action."""

    def __init__(
        self,
        metadata: Mapping[str, Any],
        *,
        enabled: bool,
        session: ResidualSession | None = None,
    ) -> None:
        self.metadata = validate_metadata(metadata)
        if type(enabled) is not bool:
            raise TypeError("runtime enabled switch must be boolean")
        if enabled and session is None:
            raise ValueError("enabled runtime requires a residual session")
        self.enabled = enabled
        self._session = session

    def compose(
        self,
        release_action: np.ndarray,
        release_action_context: np.ndarray,
        condition: np.ndarray,
        enabled_gate: np.ndarray,
    ) -> np.ndarray:
        release_action = np.asarray(release_action)
        if release_action.ndim < 3:
            raise ValueError("release_action must have at least [B,S,A] axes")
        if release_action.dtype != np.float32:
            raise TypeError("release_action must use float32")
        action_width = self.metadata["output"]["width"]
        if release_action.shape[-1] != action_width:
            raise ValueError("release action width differs from artifact metadata")
        leading_shape = release_action.shape[:-1]
        gate = _validated_gate(enabled_gate, leading_shape)
        # The disabled host switch and an all-off batch are structural bypasses:
        # no optional runtime object is touched and release bytes are retained.
        if not self.enabled or not np.any(gate):
            return release_action.copy()

        release_action_context = np.asarray(release_action_context)
        condition = np.asarray(condition)
        expected_context = self.metadata["inputs"][INPUT_RELEASE_CONTEXT]["width"]
        expected_condition = self.metadata["inputs"][INPUT_CONDITION]["width"]
        if release_action_context.shape != (*leading_shape, expected_context):
            raise ValueError("release action context shape differs from artifact metadata")
        if condition.shape != (*leading_shape, expected_condition):
            raise ValueError("condition shape differs from artifact metadata")
        if release_action_context.dtype != np.float32 or condition.dtype != np.float32:
            raise TypeError("residual inputs must use float32")
        if not np.isfinite(release_action[gate]).all():
            raise ValueError("enabled release actions must be finite")
        if not np.isfinite(release_action_context[gate]).all():
            raise ValueError("enabled release contexts must be finite")
        if not np.isfinite(condition[gate]).all():
            raise ValueError("enabled compliance conditions must be finite")
        enabled_condition = condition[gate]
        if not np.all(enabled_condition[:, 0] == 1.0):
            raise ValueError("enabled condition rows must encode enable=1")
        if not np.all(enabled_condition[:, 1:] > 0.0):
            raise ValueError("enabled condition threshold and stiffness must be positive")

        safe_context = np.where(gate[..., None], release_action_context, 0.0).astype(
            np.float32, copy=False
        )
        safe_condition = np.where(gate[..., None], condition, 0.0).astype(
            np.float32, copy=False
        )
        outputs = self._session.run(
            [OUTPUT_ACTION_DELTA],
            {
                INPUT_RELEASE_CONTEXT: safe_context,
                INPUT_CONDITION: safe_condition,
            },
        )
        if not isinstance(outputs, Sequence) or len(outputs) != 1:
            raise RuntimeError("residual session must return exactly one output")
        delta = np.asarray(outputs[0])
        if delta.shape != release_action.shape or delta.dtype != np.float32:
            raise RuntimeError("residual session output contract differs from metadata")
        if not np.isfinite(delta[gate]).all():
            raise RuntimeError("enabled residual output contains NaN or Inf")
        limit = float(self.metadata["network"]["max_abs_delta"])
        tolerance = max(1.0e-7, limit * 1.0e-6)
        if np.any(np.abs(delta[gate]) > limit + tolerance):
            raise RuntimeError("residual session exceeded the declared action-delta bound")
        bounded = np.clip(delta, -limit, limit)
        result = release_action.copy()
        result[gate] = release_action[gate] + bounded[gate]
        return result
