"""Hard acceptance checks for trained motion-compliance residual checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    MOTION_COMPLIANCE_INITIALIZATION_KEY,
    OFFICIAL_POLICY_TENSOR_COUNT,
    OFFICIAL_SONIC_RELEASE_SHA256,
    OFFICIAL_VALUE_TENSOR_COUNT,
    RESIDUAL_DTYPE,
    VALUE_STATE_KEY,
    audit_residual_init_checkpoint,
    expected_residual_shapes,
    load_trl_checkpoint,
    tensor_bytes_equal,
    validate_checkpoint_sha256,
    validate_optimizer_parameter_group_hyperparameters,
    validate_strict_resume_payload,
)
from .paths import (
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
)


_OFFICIAL_OPTIMIZER_STEP = 831000
_PINNED_ADAMW_GROUP_KEYS = frozenset(
    {
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "initial_lr",
        "params",
    }
)
_PINNED_ADAMW_FIXED_VALUES = {
    "betas": (0.9, 0.999),
    "eps": 1.0e-8,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "decoupled_weight_decay": True,
}


@dataclass(frozen=True)
class TrainedCheckpointAuditReport:
    """Compact evidence emitted after the real training and resume smokes."""

    checkpoint_path: str
    initialization_checkpoint_path: str
    global_step: int
    changed_policy_residual_names: tuple[str, ...]
    changed_value_residual_names: tuple[str, ...]
    frozen_policy_tensor_count: int
    frozen_value_tensor_count: int
    optimizer_slot_count: int
    optimizer_steps: tuple[int, ...]


def _resolve_single_policy_state(
    checkpoint: Mapping[str, Any],
) -> tuple[str, Mapping[str, torch.Tensor]]:
    present = [
        key for key in ("actor_model_state_dict", "policy_state_dict") if key in checkpoint
    ]
    if len(present) != 1 or not isinstance(checkpoint[present[0]], Mapping):
        raise ValueError("checkpoint must contain exactly one policy state mapping")
    return present[0], checkpoint[present[0]]


def _require_tensor_mapping(value: Any, name: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a state mapping")
    if any(
        not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise ValueError(f"{name} must contain only string tensor entries")
    return value


def _require_schema_and_finite(
    official: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    residual_shapes: Mapping[str, tuple[int, ...]],
    *,
    group_name: str,
) -> None:
    expected_keys = set(official) | set(residual_shapes)
    if set(candidate) != expected_keys:
        raise ValueError(
            f"{group_name} keys differ: missing={sorted(expected_keys - set(candidate))}, "
            f"unexpected={sorted(set(candidate) - expected_keys)}"
        )
    for key, tensor in candidate.items():
        expected_tensor = official.get(key)
        expected_shape = (
            tuple(expected_tensor.shape) if expected_tensor is not None else residual_shapes[key]
        )
        expected_dtype = expected_tensor.dtype if expected_tensor is not None else RESIDUAL_DTYPE
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"{group_name}.{key} schema differs: expected "
                f"{expected_shape}/{expected_dtype}, got {tuple(tensor.shape)}/{tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{group_name}.{key} contains NaN or Inf")


def _require_official_bytes(
    official: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    group_name: str,
) -> None:
    for key, official_tensor in official.items():
        if not tensor_bytes_equal(official_tensor, candidate[key]):
            raise ValueError(f"frozen {group_name} tensor changed: {key}")


def _require_every_residual_changed(
    initialized: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    residual_shapes: Mapping[str, tuple[int, ...]],
    *,
    group_name: str,
) -> tuple[str, ...]:
    unchanged = tuple(
        key
        for key in residual_shapes
        if tensor_bytes_equal(initialized[key], trained[key])
    )
    if unchanged:
        raise ValueError(f"one or more {group_name} residual tensors did not change: {unchanged}")
    return tuple(residual_shapes)


def _optimizer_steps_and_ownership(
    optimizer_state: Mapping[str, Any],
    ordered_residual_tensors: tuple[torch.Tensor, ...],
) -> tuple[int, ...]:
    slots = optimizer_state.get("state")
    parameter_groups = optimizer_state.get("param_groups")
    if not isinstance(slots, Mapping) or not isinstance(parameter_groups, list):
        raise ValueError("optimizer state/parameter groups are malformed")
    expected_parameter_groups = (tuple(range(6)), tuple(range(6, 12)))
    parameter_ids: list[Any] = []
    for group_index, (group, expected_parameter_ids) in enumerate(
        zip(parameter_groups, expected_parameter_groups, strict=True)
    ):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("optimizer parameter group is malformed")
        if set(group) != _PINNED_ADAMW_GROUP_KEYS:
            raise ValueError(
                f"optimizer parameter group {group_index} differs from the pinned "
                "AdamW schema"
            )
        if tuple(group["params"]) != expected_parameter_ids:
            raise ValueError(
                f"optimizer parameter order differs in group {group_index}"
            )
        validate_optimizer_parameter_group_hyperparameters(
            group,
            group_index=group_index,
        )
        for key, expected_value in _PINNED_ADAMW_FIXED_VALUES.items():
            if group[key] != expected_value:
                raise ValueError(
                    f"optimizer fixed AdamW flag/value differs for group "
                    f"{group_index}: {key}"
                )
        # The pinned SONIC PPOConfig leaves weight_decay at its 0.0 default;
        # Hugging Face still creates decay/no-decay groups, but both serialize
        # zero weight decay for this Phase-4 workflow.
        expected_weight_decay = 0.0
        if group["weight_decay"] != expected_weight_decay:
            raise ValueError(
                f"optimizer fixed AdamW weight_decay differs for group {group_index}"
            )
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer contains duplicate parameter ids")
    if len(parameter_ids) != len(ordered_residual_tensors):
        raise ValueError(
            "optimizer parameter ownership differs from the residual-only schema: "
            f"expected {len(ordered_residual_tensors)}, got {len(parameter_ids)}"
        )
    if set(slots) != set(parameter_ids):
        raise ValueError("optimizer slots do not cover exactly every residual parameter")

    steps: list[int] = []
    for parameter_id, residual_tensor in zip(
        parameter_ids,
        ordered_residual_tensors,
        strict=True,
    ):
        slot = slots[parameter_id]
        if not isinstance(slot, Mapping) or "step" not in slot:
            raise ValueError("optimizer slot lacks a step counter")
        raw_step = slot["step"]
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1 or not torch.isfinite(raw_step).all():
                raise ValueError("optimizer step tensor must be a finite scalar")
            raw_step = raw_step.item()
        step = int(raw_step)
        if float(raw_step) != step or step <= 0:
            raise ValueError(f"optimizer step must be a positive integer; got {raw_step!r}")
        steps.append(step)
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = slot.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != residual_tensor.shape
                or moment.dtype != residual_tensor.dtype
            ):
                raise ValueError(
                    f"optimizer {moment_name} schema does not match its residual tensor"
                )
            if not torch.isfinite(moment).all():
                raise ValueError(f"optimizer slot tensor {moment_name} contains NaN or Inf")
            if torch.count_nonzero(moment).item() == 0:
                raise ValueError(
                    "optimizer evidence shows a residual tensor has a zero "
                    f"{moment_name} moment"
                )
    return tuple(sorted(set(steps)))


def audit_trained_motion_compliance_checkpoint(
    official_checkpoint_path: str | os.PathLike[str],
    initialization_checkpoint_path: str | os.PathLike[str],
    trained_checkpoint_path: str | os.PathLike[str],
    *,
    expected_global_step: int,
    num_sites: int = 2,
) -> TrainedCheckpointAuditReport:
    """Compare trained state independently against pinned official and init files."""

    if type(expected_global_step) is not int or expected_global_step <= 0:
        raise ValueError("expected_global_step must be a positive integer")
    initialized_path = validate_motion_compliance_run_path(initialization_checkpoint_path)
    trained_path = validate_motion_compliance_run_path(trained_checkpoint_path)
    validate_distinct_artifact_paths(initialized=initialized_path, trained=trained_path)
    validate_checkpoint_sha256(
        official_checkpoint_path,
        OFFICIAL_SONIC_RELEASE_SHA256,
    )
    official = load_trl_checkpoint(official_checkpoint_path, map_location="cpu")
    initialized = load_trl_checkpoint(initialized_path, map_location="cpu")
    trained = load_trl_checkpoint(trained_path, map_location="cpu")
    audit_residual_init_checkpoint(initialized)
    if MOTION_COMPLIANCE_INITIALIZATION_KEY in trained:
        raise ValueError("trained checkpoint must not masquerade as an initialization artifact")

    official_policy_key, official_policy_raw = _resolve_single_policy_state(official)
    initialized_policy_key, initialized_policy_raw = _resolve_single_policy_state(initialized)
    trained_policy_key, trained_policy_raw = _resolve_single_policy_state(trained)
    if official_policy_key != "policy_state_dict":
        raise ValueError("pinned official source_policy_key changed")
    if initialized_policy_key != "policy_state_dict" or trained_policy_key != "policy_state_dict":
        raise ValueError("motion-compliance policy state key must be policy_state_dict")
    official_policy = _require_tensor_mapping(official_policy_raw, "official policy")
    initialized_policy = _require_tensor_mapping(initialized_policy_raw, "initialized policy")
    trained_policy = _require_tensor_mapping(trained_policy_raw, "trained policy")
    official_value = _require_tensor_mapping(
        official.get(VALUE_STATE_KEY),
        "official value",
    )
    initialized_value = _require_tensor_mapping(
        initialized.get(VALUE_STATE_KEY),
        "initialized value",
    )
    trained_value = _require_tensor_mapping(trained.get(VALUE_STATE_KEY), "trained value")
    if len(official_policy) != OFFICIAL_POLICY_TENSOR_COUNT or len(
        official_value
    ) != OFFICIAL_VALUE_TENSOR_COUNT:
        raise ValueError("pinned official policy/value tensor counts changed")

    policy_residual_shapes, value_residual_shapes = expected_residual_shapes(num_sites)
    for state, group_name in (
        (initialized_policy, "initialized policy"),
        (trained_policy, "trained policy"),
    ):
        _require_schema_and_finite(
            official_policy,
            state,
            policy_residual_shapes,
            group_name=group_name,
        )
        _require_official_bytes(official_policy, state, group_name=group_name)
    for state, group_name in (
        (initialized_value, "initialized value"),
        (trained_value, "trained value"),
    ):
        _require_schema_and_finite(
            official_value,
            state,
            value_residual_shapes,
            group_name=group_name,
        )
        _require_official_bytes(official_value, state, group_name=group_name)

    changed_policy = _require_every_residual_changed(
        initialized_policy,
        trained_policy,
        policy_residual_shapes,
        group_name="policy",
    )
    changed_value = _require_every_residual_changed(
        initialized_value,
        trained_value,
        value_residual_shapes,
        group_name="value",
    )

    payload = validate_strict_resume_payload(trained)
    global_step = getattr(payload.state, "global_step")
    if global_step != expected_global_step:
        raise ValueError(
            f"trained global step differs: expected {expected_global_step}, got {global_step}"
        )
    policy_weight_keys = tuple(
        key for key in policy_residual_shapes if key.endswith(".weight")
    )
    value_weight_keys = tuple(
        key for key in value_residual_shapes if key.endswith(".weight")
    )
    policy_bias_keys = tuple(key for key in policy_residual_shapes if key.endswith(".bias"))
    value_bias_keys = tuple(key for key in value_residual_shapes if key.endswith(".bias"))
    ordered_residual_tensors = (
        tuple(trained_policy[key] for key in policy_weight_keys)
        + tuple(trained_value[key] for key in value_weight_keys)
        + tuple(trained_policy[key] for key in policy_bias_keys)
        + tuple(trained_value[key] for key in value_bias_keys)
    )
    optimizer_steps = _optimizer_steps_and_ownership(
        payload.optimizer_state_dict,
        ordered_residual_tensors,
    )
    if _OFFICIAL_OPTIMIZER_STEP in optimizer_steps:
        raise ValueError("trained optimizer retained the official step 831000")

    return TrainedCheckpointAuditReport(
        checkpoint_path=str(trained_path),
        initialization_checkpoint_path=str(initialized_path),
        global_step=global_step,
        changed_policy_residual_names=changed_policy,
        changed_value_residual_names=changed_value,
        frozen_policy_tensor_count=len(official_policy),
        frozen_value_tensor_count=len(official_value),
        optimizer_slot_count=len(payload.optimizer_state_dict["state"]),
        optimizer_steps=optimizer_steps,
    )


def audit_motion_compliance_exposure_report(
    report_path: str | os.PathLike[str],
    *,
    expected_global_step: int,
    num_sites: int,
) -> dict[str, Any]:
    """Require bounded per-site exposure evidence from the real PPO run."""

    resolved_path = validate_motion_compliance_run_path(report_path)
    with resolved_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if report.get("global_step") != expected_global_step:
        raise ValueError("exposure report global step differs")
    for key in ("active_site_samples_by_index", "nonzero_force_site_samples_by_index"):
        counts = report.get(key)
        if not isinstance(counts, list) or len(counts) != num_sites:
            raise ValueError(f"exposure report {key} site count differs")
        if any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError(f"exposure report {key} contains an unexposed site")
    if report.get("observed_batches", 0) <= 0 or report.get("peak_site_force_n", 0.0) <= 0:
        raise ValueError("exposure report lacks a nonzero physical batch")
    if report.get("observed_loss_logs", 0) <= 0 or report.get(
        "finite_loss_metric_samples", 0
    ) <= 0:
        raise ValueError("exposure report lacks finite PPO loss evidence")
    loss_metrics = report.get("last_loss_metrics")
    if not isinstance(loss_metrics, dict) or not loss_metrics:
        raise ValueError("exposure report lacks final loss metrics")
    if any(not math.isfinite(float(value)) for value in loss_metrics.values()):
        raise ValueError("exposure report contains a non-finite final loss")
    if report.get("iteration_timing_logs", 0) <= 0:
        raise ValueError("exposure report lacks iteration timing evidence")
    for key in (
        "iteration_collection_time_mean_s",
        "iteration_learn_time_mean_s",
        "min_fps",
        "max_fps",
    ):
        value = report.get(key)
        if not isinstance(value, int | float) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"exposure report contains invalid timing metric {key}")
    if report.get("process_peak_cuda_memory_allocated_bytes", 0) <= 0:
        raise ValueError("exposure report lacks CUDA peak-memory evidence")
    return report
