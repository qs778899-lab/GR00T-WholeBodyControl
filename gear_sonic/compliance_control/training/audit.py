"""Hard acceptance checks for trained motion-compliance checkpoints."""

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
    ACTOR_ADDED_COLUMNS,
    ACTOR_INPUT_WEIGHT_KEY,
    CRITIC_INPUT_WEIGHT_KEY,
    CRITIC_RUNNING_MEAN_KEY,
    CRITIC_RUNNING_VAR_KEY,
    OFFICIAL_ACTOR_INPUT_WIDTH,
    OFFICIAL_CRITIC_INPUT_WIDTH,
    OFFICIAL_INPUT_HIDDEN_WIDTH,
    OFFICIAL_SONIC_RELEASE_SHA256,
    VALUE_STATE_KEY,
    critic_added_columns,
    load_trl_checkpoint,
    validate_checkpoint_sha256,
    validate_strict_resume_payload,
)
from .paths import validate_motion_compliance_run_path


_DYNAMIC_DECODER_PREFIX = "actor_module.decoders.g1_dyn."
_FROZEN_REQUIRED_PREFIXES = (
    "actor_module.encoders.",
    "actor_module.decoders.g1_kin.",
)
_OFFICIAL_OPTIMIZER_STEP = 831000


@dataclass(frozen=True)
class TrainedCheckpointAuditReport:
    """Compact evidence emitted after the real training and resume smokes."""

    checkpoint_path: str
    global_step: int
    actor_added_columns_nonzero: tuple[bool, ...]
    critic_added_columns_nonzero: tuple[bool, ...]
    frozen_policy_tensor_count: int
    quantizer_state_tensor_count: int
    optimizer_slot_count: int
    optimizer_steps: tuple[int, ...]


def _resolve_single_policy_state(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    present = [
        key for key in ("actor_model_state_dict", "policy_state_dict") if key in checkpoint
    ]
    if len(present) != 1 or not isinstance(checkpoint[present[0]], Mapping):
        raise ValueError("checkpoint must contain exactly one policy state mapping")
    return checkpoint[present[0]]


def _require_tensor_mapping(value: Any, name: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a state mapping")
    if any(not isinstance(key, str) or not isinstance(tensor, torch.Tensor) for key, tensor in value.items()):
        raise ValueError(f"{name} must contain only string tensor entries")
    return value


def _require_equal_keys_and_expected_shapes(
    official: Mapping[str, torch.Tensor],
    trained: Mapping[str, torch.Tensor],
    *,
    expanded_shapes: Mapping[str, tuple[int, ...]],
    group_name: str,
) -> None:
    if set(official) != set(trained):
        raise ValueError(f"{group_name} keys differ from the official checkpoint")
    for key, official_tensor in official.items():
        trained_tensor = trained[key]
        expected_shape = expanded_shapes.get(key, tuple(official_tensor.shape))
        if tuple(trained_tensor.shape) != expected_shape:
            raise ValueError(
                f"{group_name}.{key} shape differs: expected {expected_shape}, "
                f"got {tuple(trained_tensor.shape)}"
            )
        if trained_tensor.dtype != official_tensor.dtype:
            raise ValueError(
                f"{group_name}.{key} dtype differs: expected {official_tensor.dtype}, "
                f"got {trained_tensor.dtype}"
            )
        if not torch.isfinite(trained_tensor).all():
            raise ValueError(f"{group_name}.{key} contains NaN or Inf")


def _column_nonzero_flags(tensor: torch.Tensor, added_columns: int) -> tuple[bool, ...]:
    tail = tensor[:, -added_columns:]
    return tuple(bool(value) for value in torch.any(tail != 0, dim=0).tolist())


def _tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor element representations, including the sign bit of zero."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _optimizer_steps(optimizer_state: Mapping[str, Any]) -> tuple[int, ...]:
    slots = optimizer_state["state"]
    steps: list[int] = []
    for slot in slots.values():
        if not isinstance(slot, Mapping) or "step" not in slot:
            raise ValueError("optimizer slot lacks a step counter")
        raw_step = slot["step"]
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError("optimizer step tensor must be scalar")
            raw_step = raw_step.item()
        step = int(raw_step)
        if float(raw_step) != step or step <= 0:
            raise ValueError(f"optimizer step must be a positive integer; got {raw_step!r}")
        steps.append(step)
        for key, value in slot.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"optimizer slot tensor {key} contains NaN or Inf")
    return tuple(sorted(set(steps)))


def audit_trained_motion_compliance_checkpoint(
    official_checkpoint_path: str | os.PathLike[str],
    trained_checkpoint_path: str | os.PathLike[str],
    *,
    expected_global_step: int,
    num_sites: int = 2,
) -> TrainedCheckpointAuditReport:
    """Compare a step checkpoint against the immutable release and finetune contract."""

    if type(expected_global_step) is not int or expected_global_step <= 0:
        raise ValueError("expected_global_step must be a positive integer")
    trained_path = validate_motion_compliance_run_path(trained_checkpoint_path)
    validate_checkpoint_sha256(
        official_checkpoint_path,
        OFFICIAL_SONIC_RELEASE_SHA256,
    )
    official = load_trl_checkpoint(official_checkpoint_path, map_location="cpu")
    trained = load_trl_checkpoint(trained_path, map_location="cpu")

    official_policy = _require_tensor_mapping(
        _resolve_single_policy_state(official),
        "official policy",
    )
    trained_policy = _require_tensor_mapping(
        _resolve_single_policy_state(trained),
        "trained policy",
    )
    official_value = _require_tensor_mapping(
        official.get(VALUE_STATE_KEY),
        "official value",
    )
    trained_value = _require_tensor_mapping(
        trained.get(VALUE_STATE_KEY),
        "trained value",
    )

    critic_columns = critic_added_columns(num_sites)
    _require_equal_keys_and_expected_shapes(
        official_policy,
        trained_policy,
        expanded_shapes={
            ACTOR_INPUT_WEIGHT_KEY: (
                OFFICIAL_INPUT_HIDDEN_WIDTH,
                OFFICIAL_ACTOR_INPUT_WIDTH + ACTOR_ADDED_COLUMNS,
            )
        },
        group_name="policy",
    )
    _require_equal_keys_and_expected_shapes(
        official_value,
        trained_value,
        expanded_shapes={
            CRITIC_INPUT_WEIGHT_KEY: (
                OFFICIAL_INPUT_HIDDEN_WIDTH,
                OFFICIAL_CRITIC_INPUT_WIDTH + critic_columns,
            ),
            CRITIC_RUNNING_MEAN_KEY: (OFFICIAL_CRITIC_INPUT_WIDTH + critic_columns,),
            CRITIC_RUNNING_VAR_KEY: (OFFICIAL_CRITIC_INPUT_WIDTH + critic_columns,),
        },
        group_name="value",
    )

    actor_nonzero = _column_nonzero_flags(
        trained_policy[ACTOR_INPUT_WEIGHT_KEY],
        ACTOR_ADDED_COLUMNS,
    )
    critic_nonzero = _column_nonzero_flags(
        trained_value[CRITIC_INPUT_WEIGHT_KEY],
        critic_columns,
    )
    if not all(actor_nonzero):
        raise ValueError(f"one or more added actor columns remained zero: {actor_nonzero}")
    if not all(critic_nonzero):
        raise ValueError(f"one or more added critic columns remained zero: {critic_nonzero}")

    frozen_keys = tuple(
        key for key in official_policy if not key.startswith(_DYNAMIC_DECODER_PREFIX)
    )
    for prefix in _FROZEN_REQUIRED_PREFIXES:
        if not any(key.startswith(prefix) for key in frozen_keys):
            raise ValueError(f"official checkpoint lacks frozen contract prefix {prefix}")
    noise_keys = tuple(key for key in frozen_keys if key in {"std", "log_std"})
    if len(noise_keys) != 1:
        raise ValueError("official checkpoint must contain exactly one frozen noise tensor")
    for key in frozen_keys:
        if not _tensor_bytes_equal(official_policy[key], trained_policy[key]):
            raise ValueError(f"frozen policy tensor changed during finetuning: {key}")
    quantizer_keys = tuple(
        key for key in official_policy if key.startswith("actor_module.quantizer.")
    )
    if set(quantizer_keys) != {
        key for key in trained_policy if key.startswith("actor_module.quantizer.")
    }:
        raise ValueError("quantizer state keys changed during finetuning")

    payload = validate_strict_resume_payload(trained)
    global_step = getattr(payload.state, "global_step")
    if global_step != expected_global_step:
        raise ValueError(
            f"trained global step differs: expected {expected_global_step}, got {global_step}"
        )
    expected_optimizer_slots = sum(
        key.startswith(_DYNAMIC_DECODER_PREFIX) for key in trained_policy
    ) + sum(key.startswith("critic_module.") for key in trained_value)
    optimizer_slot_count = len(payload.optimizer_state_dict["state"])
    if optimizer_slot_count != expected_optimizer_slots:
        raise ValueError(
            "optimizer slot ownership differs from dynamic-decoder + critic tensors: "
            f"expected {expected_optimizer_slots}, got {optimizer_slot_count}"
        )
    optimizer_steps = _optimizer_steps(payload.optimizer_state_dict)
    if _OFFICIAL_OPTIMIZER_STEP in optimizer_steps:
        raise ValueError("trained optimizer retained the official step 831000")

    return TrainedCheckpointAuditReport(
        checkpoint_path=str(trained_path),
        global_step=global_step,
        actor_added_columns_nonzero=actor_nonzero,
        critic_added_columns_nonzero=critic_nonzero,
        frozen_policy_tensor_count=len(frozen_keys),
        quantizer_state_tensor_count=len(quantizer_keys),
        optimizer_slot_count=optimizer_slot_count,
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
