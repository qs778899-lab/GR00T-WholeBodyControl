"""Small, tracker-neutral primitives for auditable residual finetuning."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def incremental_batch_count(start_step: int, final_step: int) -> int:
    """Return the number of new batches required by an incremental trainer run."""

    if type(start_step) is not int or type(final_step) is not int:
        raise TypeError("start_step and final_step must be integers")
    count = final_step - start_step
    if start_step < 0 or count <= 0:
        raise ValueError("steps must define a non-negative, increasing interval")
    return count


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def tensor_byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor metadata and storage bytes, including NaN payloads."""

    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and _tensor_bytes(left) == _tensor_bytes(right)
    )


def assert_state_dict_exact(
    reference: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    *,
    allow_additional_current: bool = False,
    label: str = "state_dict",
) -> None:
    """Require the requested state schema and every tensor byte to match."""

    reference_keys = set(reference)
    current_keys = set(current)
    missing = sorted(reference_keys - current_keys)
    unexpected = sorted(current_keys - reference_keys)
    if missing or (unexpected and not allow_additional_current):
        raise AssertionError(
            f"{label} schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(reference_keys):
        if not tensor_byte_equal(reference[name], current[name]):
            raise AssertionError(f"{label} tensor is not byte-exact: {name}")


def state_dict_digest(
    state_dict: Mapping[str, torch.Tensor],
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> str:
    """Hash ordered names, shapes, dtypes, and bytes without retaining a copy."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        if name.startswith(excluded_prefixes):
            continue
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def assert_nested_exact(reference: Any, current: Any, *, label: str) -> None:
    """Recursively compare optimizer/scheduler state with tensor-byte rigor."""

    if isinstance(reference, torch.Tensor) or isinstance(current, torch.Tensor):
        if not (
            isinstance(reference, torch.Tensor)
            and isinstance(current, torch.Tensor)
            and tensor_byte_equal(reference, current)
        ):
            raise AssertionError(f"{label} tensor mismatch")
        return
    if isinstance(reference, Mapping) or isinstance(current, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(current, Mapping):
            raise AssertionError(f"{label} mapping type mismatch")
        if set(reference) != set(current):
            raise AssertionError(f"{label} mapping keys mismatch")
        for key in reference:
            assert_nested_exact(
                reference[key],
                current[key],
                label=f"{label}.{key}",
            )
        return
    if isinstance(reference, (list, tuple)) or isinstance(current, (list, tuple)):
        if type(reference) is not type(current) or len(reference) != len(current):
            raise AssertionError(f"{label} sequence mismatch")
        for index, (left, right) in enumerate(zip(reference, current)):
            assert_nested_exact(left, right, label=f"{label}[{index}]")
        return
    if type(reference) is not type(current) or reference != current:
        raise AssertionError(f"{label} value mismatch: {reference!r} != {current!r}")


def optimizer_parameter_count(state_dict: Mapping[str, Any]) -> int:
    """Return the number of parameter slots declared by an optimizer state."""

    groups = state_dict.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer state must contain non-empty param_groups")
    parameter_ids: list[int] = []
    for group in groups:
        params = group.get("params") if isinstance(group, Mapping) else None
        if not isinstance(params, list):
            raise ValueError("optimizer param_groups entries must contain params lists")
        parameter_ids.extend(params)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer parameter ids must be unique across groups")
    return len(parameter_ids)


def finite_loss_metrics(logs: Mapping[str, Any]) -> dict[str, float]:
    """Extract scalar loss metrics and reject missing or non-finite values."""

    losses: dict[str, float] = {}
    for name, value in logs.items():
        if "loss" not in str(name).casefold():
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"loss metric {name!r} must be scalar")
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"loss metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"loss metric {name!r} is non-finite")
        losses[str(name)] = numeric
    if not losses:
        raise ValueError("training log contains no loss metrics")
    return losses


def directory_usage_bytes(root: str | Path) -> tuple[int, int]:
    """Return total file bytes and largest log bytes without following symlinks."""

    root = Path(root)
    total = 0
    largest_log = 0
    if not root.exists():
        return total, largest_log
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(directory) / filename
            try:
                size = path.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                # A concurrent atomic checkpoint replacement may briefly make
                # one directory entry stale.  The next audit sees the new file.
                continue
            total += size
            if path.suffix == ".log":
                largest_log = max(largest_log, size)
    return total, largest_log


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one bounded audit document atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
