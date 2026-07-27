"""Thin SONIC adapter for the portable action-residual deployment runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np

from gear_sonic.compliance_control.deployment import (
    ActionResidualRuntime,
    ArtifactExpectation,
    ReleaseArtifactPin,
    load_artifact_metadata,
    load_onnxruntime_session,
)


@dataclass(frozen=True)
class SonicResidualContextContract:
    """Observation and semantic layout owned by the SONIC integration only."""

    token_width: int
    actor_observation_width: int
    condition_width: int
    site_layout: tuple[str, ...]
    action_layout: tuple[str, ...]
    force_threshold_range_n: tuple[float, float]
    reference_displacement_m: float

    @property
    def release_context_width(self) -> int:
        return self.token_width + self.actor_observation_width

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SonicResidualContextContract":
        if not isinstance(value, Mapping):
            raise TypeError("SONIC residual context config must be a mapping")
        context_layout = value.get("context_layout")
        if (
            isinstance(context_layout, (str, bytes))
            or not isinstance(context_layout, Sequence)
            or len(context_layout) != 2
        ):
            raise ValueError("SONIC context_layout must contain token and actor fields")
        expected_names = ("robot_motion_token", "actor_observation")
        widths: list[int] = []
        for index, (entry, expected_name) in enumerate(
            zip(context_layout, expected_names, strict=True)
        ):
            if not isinstance(entry, Mapping) or set(entry) != {"name", "width"}:
                raise ValueError(f"SONIC context_layout entry {index} schema differs")
            if entry["name"] != expected_name:
                raise ValueError("SONIC context_layout order or names differ")
            width = entry["width"]
            if type(width) is not int or width <= 0:
                raise ValueError("SONIC context widths must be positive integers")
            widths.append(width)
        condition_width = value.get("condition_width")
        if condition_width != 3 or type(condition_width) is not int:
            raise ValueError(
                "SONIC public condition width must be exactly 3: "
                "[enable, enable*threshold, enable*Kp]"
            )
        default_condition = value.get("default_enabled_condition")
        if (
            isinstance(default_condition, (str, bytes))
            or not isinstance(default_condition, Sequence)
            or len(default_condition) != condition_width
        ):
            raise ValueError("default_enabled_condition must contain three values")
        try:
            default_enable, threshold, stiffness = map(float, default_condition)
        except (TypeError, ValueError) as error:
            raise ValueError("default_enabled_condition must be numeric") from error
        if (
            not all(math.isfinite(item) for item in (default_enable, threshold, stiffness))
            or default_enable != 1.0
            or threshold <= 0.0
            or stiffness <= 0.0
        ):
            raise ValueError(
                "default_enabled_condition must be [1, positive threshold, positive Kp]"
            )
        displacement = threshold / stiffness
        site_layout = tuple(value.get("site_layout", ()))
        action_layout = tuple(value.get("action_layout", ()))
        if not site_layout or any(not isinstance(name, str) or not name for name in site_layout):
            raise ValueError("site_layout must contain non-empty identifiers")
        if len(set(site_layout)) != len(site_layout):
            raise ValueError("site_layout must not contain duplicates")
        if not action_layout or any(
            not isinstance(name, str) or not name for name in action_layout
        ):
            raise ValueError("action_layout must contain non-empty identifiers")
        if len(set(action_layout)) != len(action_layout):
            raise ValueError("action_layout must not contain duplicates")
        return cls(
            token_width=widths[0],
            actor_observation_width=widths[1],
            condition_width=condition_width,
            site_layout=site_layout,
            action_layout=action_layout,
            force_threshold_range_n=(threshold, threshold),
            reference_displacement_m=displacement,
        )


def assemble_release_action_context(
    token: np.ndarray,
    actor_observation: np.ndarray,
    contract: SonicResidualContextContract,
) -> np.ndarray:
    """Assemble the trained order: token then unchanged actor observation."""

    token = np.asarray(token)
    actor_observation = np.asarray(actor_observation)
    if token.shape[:-1] != actor_observation.shape[:-1]:
        raise ValueError("token and actor observation leading shapes differ")
    if token.shape[-1] != contract.token_width:
        raise ValueError("token width differs from SONIC deployment contract")
    if actor_observation.shape[-1] != contract.actor_observation_width:
        raise ValueError("actor observation width differs from SONIC deployment contract")
    if token.dtype != np.float32 or actor_observation.dtype != np.float32:
        raise TypeError("SONIC residual context inputs must use float32")
    return np.concatenate((token, actor_observation), axis=-1)


def encode_condition(
    enabled_gate: np.ndarray,
    force_threshold_n: np.ndarray,
    contract: SonicResidualContextContract,
) -> np.ndarray:
    """Encode public `[enable, enable*threshold, enable*Kp]` without force input."""

    enabled = np.asarray(enabled_gate)
    threshold = np.asarray(force_threshold_n)
    if enabled.shape != threshold.shape:
        raise ValueError("enabled and threshold shapes differ")
    if enabled.dtype != np.bool_:
        raise TypeError("enabled gate must use boolean dtype")
    if threshold.dtype != np.float32:
        raise TypeError("force threshold must use float32")
    low, high = contract.force_threshold_range_n
    if not np.isfinite(threshold[enabled]).all() or np.any(
        (threshold[enabled] < low) | (threshold[enabled] > high)
    ):
        raise ValueError("enabled force thresholds differ from the SONIC contract")
    enabled_float = enabled.astype(np.float32)
    return np.stack(
        (
            enabled_float,
            enabled_float * np.where(enabled, threshold, 0.0),
            enabled_float
            * np.where(enabled, threshold, 0.0)
            / contract.reference_displacement_m,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


class SonicActionResidualDeployment:
    """Explicit SONIC host switch around the portable composition runtime."""

    def __init__(
        self,
        runtime: ActionResidualRuntime,
        contract: SonicResidualContextContract,
    ) -> None:
        metadata = runtime.metadata
        if metadata["inputs"]["release_action_context"]["width"] != (
            contract.release_context_width
        ):
            raise ValueError("runtime release-context width differs from SONIC contract")
        if metadata["inputs"]["motion_compliance_condition"]["width"] != (
            contract.condition_width
        ):
            raise ValueError("runtime condition width differs from SONIC contract")
        if tuple(metadata["site_layout"]) != contract.site_layout:
            raise ValueError("runtime site layout differs from SONIC contract")
        if tuple(metadata["action_layout"]) != contract.action_layout:
            raise ValueError("runtime action layout differs from SONIC contract")
        self.runtime = runtime
        self.contract = contract

    def compose(
        self,
        release_action: np.ndarray,
        token: np.ndarray,
        actor_observation: np.ndarray,
        condition: np.ndarray,
        enabled_gate: np.ndarray,
    ) -> np.ndarray:
        context = assemble_release_action_context(token, actor_observation, self.contract)
        return self.runtime.compose(release_action, context, condition, enabled_gate)


def load_sonic_action_residual_deployment(
    value: Mapping[str, Any],
    *,
    release_artifacts: Sequence[ReleaseArtifactPin] = (),
    session_factory=load_onnxruntime_session,
) -> SonicActionResidualDeployment | None:
    """Consume the opt-in overlay without loading any artifact when disabled."""

    if not isinstance(value, Mapping):
        raise TypeError("SONIC action-residual config must be a mapping")
    if set(value) == {"motion_compliance_action_residual"}:
        value = value["motion_compliance_action_residual"]
    if not isinstance(value, Mapping):
        raise TypeError("SONIC action-residual overlay must contain a mapping")
    contract = SonicResidualContextContract.from_mapping(value)
    enabled = value.get("enabled")
    if type(enabled) is not bool:
        raise TypeError("SONIC action-residual enabled switch must be boolean")
    if not enabled:
        return None
    artifact_directory = value.get("artifact_directory")
    if not isinstance(artifact_directory, (str, os.PathLike)) or not str(
        artifact_directory
    ):
        raise ValueError("enabled SONIC residual requires artifact_directory")
    declared_release = value.get("release_artifacts")
    if (
        isinstance(declared_release, (str, bytes))
        or not isinstance(declared_release, Sequence)
        or not declared_release
    ):
        raise ValueError("enabled SONIC residual requires release_artifacts")
    supplied_release = tuple(release_artifacts)
    if len(declared_release) != len(supplied_release):
        raise ValueError("SONIC release artifact count differs from host-owned pins")
    for declared, supplied in zip(
        declared_release, supplied_release, strict=True
    ):
        if not isinstance(declared, Mapping) or set(declared) != {"name", "sha256"}:
            raise ValueError("SONIC release artifact declaration schema differs")
        if not isinstance(supplied, ReleaseArtifactPin):
            raise TypeError("SONIC release artifacts must use ReleaseArtifactPin")
        if (
            declared["name"] != supplied.name
            or declared["sha256"] != supplied.sha256
        ):
            raise ValueError(
                "SONIC release artifact identity differs from host-owned pins"
            )
    expectation = ArtifactExpectation(
        metadata_sha256=value.get("metadata_sha256"),
        checkpoint_sha256=value.get("checkpoint_sha256"),
        checkpoint_global_step=value.get("checkpoint_global_step"),
        release_context_width=contract.release_context_width,
        condition_width=contract.condition_width,
        action_layout=contract.action_layout,
        site_layout=contract.site_layout,
        max_abs_delta=value.get("max_abs_delta"),
        release_artifacts=supplied_release,
        schema=value.get("schema"),
    )
    metadata = load_artifact_metadata(artifact_directory, expectation)
    session = session_factory(artifact_directory, metadata)
    runtime = ActionResidualRuntime(metadata, enabled=True, session=session)
    return SonicActionResidualDeployment(runtime, contract)


def compose_optional_sonic_action_residual(
    deployment: SonicActionResidualDeployment | None,
    release_action: np.ndarray,
    token: np.ndarray,
    actor_observation: np.ndarray,
    condition: np.ndarray,
    enabled_gate: np.ndarray,
) -> np.ndarray:
    """Host hard switch: a missing opt-in plugin preserves release bytes."""

    if deployment is None:
        release_action = np.asarray(release_action)
        if release_action.dtype != np.float32:
            raise TypeError("release_action must use float32")
        if release_action.ndim != 3:
            raise ValueError("release_action must have [B,S,A] axes")
        gate = np.asarray(enabled_gate)
        if gate.shape == (*release_action.shape[:-1], 1):
            gate = gate[..., 0]
        if gate.shape != release_action.shape[:-1]:
            raise ValueError("enabled gate shape differs from release action")
        if gate.dtype != np.bool_:
            if not np.issubdtype(gate.dtype, np.number):
                raise TypeError("enabled gate must be boolean or numeric binary")
            if not np.isfinite(gate).all() or not np.logical_or(
                gate == 0, gate == 1
            ).all():
                raise ValueError("enabled gate must contain finite binary values")
        return release_action.copy()
    return deployment.compose(
        release_action,
        token,
        actor_observation,
        condition,
        enabled_gate,
    )
