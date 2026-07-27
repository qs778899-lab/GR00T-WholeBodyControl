"""Versioned, robot-independent action-residual artifact schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any


ARTIFACT_SCHEMA = "universal-tracker.action-residual.onnx.v1"
INPUT_RELEASE_CONTEXT = "release_action_context"
INPUT_CONDITION = "motion_compliance_condition"
OUTPUT_ACTION_DELTA = "action_delta"

_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "model_kind",
        "source_checkpoint",
        "model",
        "inputs",
        "output",
        "network",
        "context_layout",
        "site_layout",
        "action_layout",
        "framework_versions",
        "metadata_sha256",
    }
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the sole canonical representation used for metadata digests."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def metadata_digest(metadata: Mapping[str, Any]) -> str:
    """Digest metadata while excluding its self-authenticating digest field."""

    payload = dict(metadata)
    payload.pop("metadata_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{name} must be lowercase")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_identifier_layout(
    value: Any,
    name: str,
    *,
    expected_length: int | None = None,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of identifiers")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain non-empty string identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate identifiers")
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(
            f"{name} length differs: expected {expected_length}, got {len(result)}"
        )
    return result


def expected_residual_shapes(
    residual_context_width: int,
    hidden_dims: Sequence[int],
    action_width: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the six linear tensor shapes without assuming a robot layout."""

    residual_context_width = _positive_int(
        residual_context_width, "residual_context_width"
    )
    action_width = _positive_int(action_width, "action_width")
    if (
        isinstance(hidden_dims, (str, bytes))
        or not isinstance(hidden_dims, Sequence)
        or len(hidden_dims) != 2
    ):
        raise ValueError("network.hidden_dims must contain exactly two widths")
    first = _positive_int(hidden_dims[0], "network.hidden_dims[0]")
    second = _positive_int(hidden_dims[1], "network.hidden_dims[1]")
    return (
        (first, residual_context_width),
        (first,),
        (second, first),
        (second,),
        (action_width, second),
        (action_width,),
    )


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete artifact manifest and return a plain copy."""

    if not isinstance(metadata, Mapping):
        raise TypeError("artifact metadata must be a mapping")
    if set(metadata) != _REQUIRED_KEYS:
        raise ValueError(
            "artifact metadata keys differ: "
            f"missing={sorted(_REQUIRED_KEYS - set(metadata))}, "
            f"unexpected={sorted(set(metadata) - _REQUIRED_KEYS)}"
        )
    result = dict(metadata)
    if result["schema"] != ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported artifact schema: {result['schema']!r}")
    if result["model_kind"] != "bounded_action_residual":
        raise ValueError("unsupported residual model kind")

    source = result["source_checkpoint"]
    if not isinstance(source, Mapping) or set(source) != {
        "sha256",
        "global_step",
        "policy_state_key",
        "residual_tensor_names",
        "residual_tensor_shapes",
    }:
        raise ValueError("source_checkpoint metadata schema differs")
    validate_sha256(source["sha256"], "source_checkpoint.sha256")
    _positive_int(source["global_step"], "source_checkpoint.global_step")
    if source["policy_state_key"] not in ("policy_state_dict", "actor_model_state_dict"):
        raise ValueError("source checkpoint policy-state key is unsupported")

    model = result["model"]
    if not isinstance(model, Mapping) or set(model) != {"file", "sha256", "opset"}:
        raise ValueError("model metadata schema differs")
    if not isinstance(model["file"], str) or not model["file"] or "/" in model["file"]:
        raise ValueError("model.file must be a local filename")
    validate_sha256(model["sha256"], "model.sha256")
    if _positive_int(model["opset"], "model.opset") != 17:
        raise ValueError("artifact schema v1 requires ONNX opset exactly 17")

    inputs = result["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        INPUT_RELEASE_CONTEXT,
        INPUT_CONDITION,
    }:
        raise ValueError("artifact inputs differ from the versioned contract")
    release_context_width = _positive_int(
        inputs[INPUT_RELEASE_CONTEXT]["width"],
        f"inputs.{INPUT_RELEASE_CONTEXT}.width",
    )
    condition_width = _positive_int(
        inputs[INPUT_CONDITION]["width"],
        f"inputs.{INPUT_CONDITION}.width",
    )
    for input_name, width in (
        (INPUT_RELEASE_CONTEXT, release_context_width),
        (INPUT_CONDITION, condition_width),
    ):
        entry = inputs[input_name]
        if not isinstance(entry, Mapping) or set(entry) != {"dtype", "shape", "width"}:
            raise ValueError(f"inputs.{input_name} metadata schema differs")
        if entry["dtype"] != "float32" or entry["shape"] != ["B", "S", width]:
            raise ValueError(f"inputs.{input_name} tensor contract differs")

    output = result["output"]
    if not isinstance(output, Mapping) or set(output) != {
        "name",
        "dtype",
        "shape",
        "width",
    }:
        raise ValueError("output metadata schema differs")
    action_width = _positive_int(output["width"], "output.width")
    if (
        output["name"] != OUTPUT_ACTION_DELTA
        or output["dtype"] != "float32"
        or output["shape"] != ["B", "S", action_width]
    ):
        raise ValueError("output tensor contract differs")

    network = result["network"]
    if not isinstance(network, Mapping) or set(network) != {
        "residual_context_width",
        "hidden_dims",
        "activation",
        "output_activation",
        "max_abs_delta",
    }:
        raise ValueError("network metadata schema differs")
    residual_context_width = _positive_int(
        network["residual_context_width"], "network.residual_context_width"
    )
    if residual_context_width != release_context_width + condition_width:
        raise ValueError("residual context must be release context plus public condition")
    if network["activation"] != "silu" or network["output_activation"] != "tanh":
        raise ValueError("residual activation contract differs")
    hidden_dims = tuple(network["hidden_dims"])
    expected_shapes = expected_residual_shapes(
        residual_context_width,
        hidden_dims,
        action_width,
    )
    _finite_positive(network["max_abs_delta"], "network.max_abs_delta")

    tensor_names = validate_identifier_layout(
        source["residual_tensor_names"],
        "source_checkpoint.residual_tensor_names",
        expected_length=6,
    )
    tensor_shapes = source["residual_tensor_shapes"]
    if not isinstance(tensor_shapes, Mapping) or set(tensor_shapes) != set(tensor_names):
        raise ValueError("source residual tensor shape keys differ from names")
    actual_shapes = tuple(tuple(tensor_shapes[name]) for name in tensor_names)
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"source residual tensor shapes differ: expected {expected_shapes}, "
            f"got {actual_shapes}"
        )

    context_layout = result["context_layout"]
    if not isinstance(context_layout, Sequence) or isinstance(context_layout, (str, bytes)):
        raise TypeError("context_layout must be a sequence")
    next_offset = 0
    seen_context_names: set[str] = set()
    for entry in context_layout:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "offset", "width"}:
            raise ValueError("context_layout entry schema differs")
        name = entry["name"]
        if not isinstance(name, str) or not name or name in seen_context_names:
            raise ValueError("context_layout names must be unique non-empty strings")
        seen_context_names.add(name)
        if entry["offset"] != next_offset:
            raise ValueError("context_layout must be contiguous and ordered")
        next_offset += _positive_int(entry["width"], f"context_layout.{name}.width")
    if next_offset != release_context_width:
        raise ValueError("context_layout width differs from release context width")

    site_layout = validate_identifier_layout(result["site_layout"], "site_layout")
    action_layout = validate_identifier_layout(
        result["action_layout"], "action_layout", expected_length=action_width
    )
    if len(site_layout) <= 0 or len(action_layout) != action_width:
        raise AssertionError("unreachable validated layout")

    versions = result["framework_versions"]
    if not isinstance(versions, Mapping) or set(versions) != {
        "torch",
        "onnx",
        "onnxruntime",
    }:
        raise ValueError("framework_versions metadata schema differs")
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise ValueError("framework versions must be non-empty strings")

    declared_digest = validate_sha256(result["metadata_sha256"], "metadata_sha256")
    actual_digest = metadata_digest(result)
    if declared_digest != actual_digest:
        raise ValueError(
            f"metadata digest mismatch: declared {declared_digest}, got {actual_digest}"
        )
    return result
