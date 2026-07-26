"""Strict, opt-in migration of released SONIC weights into new branches."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CheckpointMigrationReport:
    """Auditable result of one legacy-to-compliance state migration."""

    legacy_keys: tuple[str, ...]
    initialized_new_keys: tuple[str, ...]


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def classify_checkpoint_state(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    new_key_prefixes: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return expected legacy, expected new, and source new key names."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    if not isinstance(source_state, Mapping):
        raise TypeError("source_state must be a mapping")
    if isinstance(new_key_prefixes, str | bytes):
        raise TypeError("new_key_prefixes must be a sequence")
    prefixes = tuple(new_key_prefixes)
    if not prefixes or any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise ValueError("new_key_prefixes must contain non-empty strings")

    target_names = tuple(module.state_dict().keys())
    expected_new = tuple(name for name in target_names if _matches_prefix(name, prefixes))
    if not expected_new:
        raise ValueError("new_key_prefixes do not match any target state keys")
    expected_legacy = tuple(name for name in target_names if name not in expected_new)
    source_new = tuple(name for name in source_state if _matches_prefix(name, prefixes))
    return expected_legacy, expected_new, source_new


def migrate_legacy_state_dict(
    module: nn.Module,
    source_state: Mapping[str, torch.Tensor],
    *,
    new_key_prefixes: Sequence[str],
    assign: bool = False,
) -> tuple[CheckpointMigrationReport, object]:
    """Load every legacy tensor exactly and retain only initialized new tensors.

    This function accepts a *complete* released state dict, never a partial
    checkpoint.  Any unknown, missing, shape-mismatched, or dtype-mismatched
    legacy tensor is rejected before the module is mutated.  The merged state
    is then loaded strictly, so a saved migrated checkpoint resumes strictly.
    """

    expected_legacy, expected_new, source_new = classify_checkpoint_state(
        module,
        source_state,
        new_key_prefixes=new_key_prefixes,
    )
    if source_new:
        raise ValueError("legacy migration received new-branch keys; use strict resume loading")

    target_state = module.state_dict()
    source_names = set(source_state)
    legacy_names = set(expected_legacy)
    missing = sorted(legacy_names - source_names)
    unexpected = sorted(source_names - legacy_names)
    if missing or unexpected:
        raise RuntimeError(
            "legacy checkpoint schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name in expected_legacy:
        source = source_state[name]
        target = target_state[name]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"checkpoint value {name!r} must be a torch.Tensor")
        if source.shape != target.shape:
            raise RuntimeError(
                f"checkpoint tensor {name!r} shape {tuple(source.shape)} "
                f"does not match {tuple(target.shape)}"
            )
        if source.dtype != target.dtype:
            raise RuntimeError(
                f"checkpoint tensor {name!r} dtype {source.dtype} does not match {target.dtype}"
            )

    merged = OrderedDict()
    for name in target_state:
        merged[name] = source_state[name] if name in source_state else target_state[name]
    result = nn.Module.load_state_dict(module, merged, strict=True, assign=assign)

    loaded_state = module.state_dict()
    for name in expected_legacy:
        source = source_state[name]
        loaded = loaded_state[name]
        if source.device != loaded.device:
            source = source.to(loaded.device)
        if not torch.equal(loaded, source):
            raise RuntimeError(f"legacy checkpoint tensor {name!r} did not load exactly")

    report = CheckpointMigrationReport(
        legacy_keys=tuple(sorted(expected_legacy)),
        initialized_new_keys=tuple(sorted(expected_new)),
    )
    return report, result
