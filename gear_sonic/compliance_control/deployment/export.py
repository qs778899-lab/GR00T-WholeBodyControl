"""Atomic export of a standalone action-residual ONNX artifact bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from .model import ExportableActionResidual
from .schema import (
    ARTIFACT_SCHEMA,
    INPUT_CONDITION,
    INPUT_RELEASE_CONTEXT,
    OUTPUT_ACTION_DELTA,
    canonical_json_bytes,
    metadata_digest,
    validate_identifier_layout,
    validate_metadata,
    validate_sha256,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@dataclass(frozen=True)
class ResidualExportSpec:
    """Caller-supplied tensor and semantic layouts for a portable export."""

    checkpoint_sha256: str
    checkpoint_global_step: int
    policy_state_key: str
    residual_tensor_names: tuple[str, ...]
    release_context_width: int
    condition_width: int
    action_width: int
    hidden_dims: tuple[int, int]
    max_abs_delta: float
    context_layout: tuple[tuple[str, int], ...]
    site_layout: tuple[str, ...]
    action_layout: tuple[str, ...]
    opset: int = 17


def _validate_spec(spec: ResidualExportSpec) -> None:
    validate_sha256(spec.checkpoint_sha256, "checkpoint_sha256")
    if type(spec.checkpoint_global_step) is not int or spec.checkpoint_global_step <= 0:
        raise ValueError("checkpoint_global_step must be a positive integer")
    if spec.policy_state_key not in ("policy_state_dict", "actor_model_state_dict"):
        raise ValueError("unsupported policy state key")
    validate_identifier_layout(
        spec.residual_tensor_names, "residual_tensor_names", expected_length=6
    )
    validate_identifier_layout(spec.site_layout, "site_layout")
    validate_identifier_layout(
        spec.action_layout, "action_layout", expected_length=spec.action_width
    )
    if type(spec.opset) is not int or spec.opset != 17:
        raise ValueError("artifact schema v1 requires ONNX opset exactly 17")
    if not spec.context_layout:
        raise ValueError("context_layout must not be empty")
    if any(
        not isinstance(name, str)
        or not name
        or type(width) is not int
        or width <= 0
        for name, width in spec.context_layout
    ):
        raise ValueError("context_layout entries must have names and positive widths")
    if len({name for name, _ in spec.context_layout}) != len(spec.context_layout):
        raise ValueError("context_layout names must be unique")
    if sum(width for _, width in spec.context_layout) != spec.release_context_width:
        raise ValueError("context_layout width differs from release_context_width")


def export_action_residual_bundle(
    tensors: Mapping[str, torch.Tensor],
    output_directory: str | os.PathLike[str],
    spec: ResidualExportSpec,
) -> dict[str, Any]:
    """Create an all-or-nothing directory containing ONNX and metadata JSON.

    The final directory must not already exist.  Both files are built and
    validated in a same-filesystem staging directory; a single directory
    rename then publishes the complete pair.
    """

    _validate_spec(spec)
    output_directory = Path(output_directory)
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"artifact output already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    model_file = "action_residual.onnx"
    metadata_file = "action_residual.metadata.json"
    try:
        module = ExportableActionResidual(
            spec.release_context_width,
            spec.condition_width,
            spec.action_width,
            hidden_dims=spec.hidden_dims,
            max_abs_delta=spec.max_abs_delta,
        )
        module.load_linear_state(tensors, source_names=spec.residual_tensor_names)
        model_path = staging / model_file
        sample_release_context = torch.zeros(
            2, 3, spec.release_context_width, dtype=torch.float32
        )
        sample_condition = torch.zeros(2, 3, spec.condition_width, dtype=torch.float32)
        torch.onnx.export(
            module,
            (sample_release_context, sample_condition),
            model_path,
            input_names=[INPUT_RELEASE_CONTEXT, INPUT_CONDITION],
            output_names=[OUTPUT_ACTION_DELTA],
            dynamic_axes={
                INPUT_RELEASE_CONTEXT: {0: "B", 1: "S"},
                INPUT_CONDITION: {0: "B", 1: "S"},
                OUTPUT_ACTION_DELTA: {0: "B", 1: "S"},
            },
            opset_version=spec.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        import onnx

        onnx.checker.check_model(onnx.load(model_path))
        context_layout: list[dict[str, Any]] = []
        offset = 0
        for name, width in spec.context_layout:
            context_layout.append({"name": name, "offset": offset, "width": width})
            offset += width
        tensor_shapes = {
            name: list(tensors[name].shape) for name in spec.residual_tensor_names
        }
        metadata: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "model_kind": "bounded_action_residual",
            "source_checkpoint": {
                "sha256": spec.checkpoint_sha256,
                "global_step": spec.checkpoint_global_step,
                "policy_state_key": spec.policy_state_key,
                "residual_tensor_names": list(spec.residual_tensor_names),
                "residual_tensor_shapes": tensor_shapes,
            },
            "model": {
                "file": model_file,
                "sha256": _file_sha256(model_path),
                "opset": spec.opset,
            },
            "inputs": {
                INPUT_RELEASE_CONTEXT: {
                    "dtype": "float32",
                    "shape": ["B", "S", spec.release_context_width],
                    "width": spec.release_context_width,
                },
                INPUT_CONDITION: {
                    "dtype": "float32",
                    "shape": ["B", "S", spec.condition_width],
                    "width": spec.condition_width,
                },
            },
            "output": {
                "name": OUTPUT_ACTION_DELTA,
                "dtype": "float32",
                "shape": ["B", "S", spec.action_width],
                "width": spec.action_width,
            },
            "network": {
                "residual_context_width": spec.release_context_width
                + spec.condition_width,
                "hidden_dims": list(spec.hidden_dims),
                "activation": "silu",
                "output_activation": "tanh",
                "max_abs_delta": spec.max_abs_delta,
            },
            "context_layout": context_layout,
            "site_layout": list(spec.site_layout),
            "action_layout": list(spec.action_layout),
            "framework_versions": {
                "torch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": _package_version("onnxruntime"),
            },
        }
        metadata["metadata_sha256"] = metadata_digest(metadata)
        validate_metadata(metadata)
        (staging / metadata_file).write_bytes(canonical_json_bytes(metadata))
        os.rename(staging, output_directory)
        return metadata
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
