"""Strict initialization and resume contracts for motion-compliance residuals."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


MOTION_COMPLIANCE_INITIALIZATION_KEY = "motion_compliance_residual_initialization"
LEGACY_MOTION_COMPLIANCE_MIGRATION_KEY = "motion_compliance_migration"
INITIALIZATION_SCHEMA_VERSION = 2
OFFICIAL_SONIC_RELEASE_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
OFFICIAL_SONIC_RELEASE_STEP = 41550
OFFICIAL_SONIC_RELEASE_REVISION = "7c90a56cfe04788c4f041daeef5b1e12930675ad"

POLICY_STATE_KEYS = ("actor_model_state_dict", "policy_state_dict")
VALUE_STATE_KEY = "value_state_dict"
ACTOR_INPUT_WEIGHT_KEY = "actor_module.decoders.g1_dyn.module.0.weight"
CRITIC_INPUT_WEIGHT_KEY = "critic_module.module.0.weight"
CRITIC_RUNNING_MEAN_KEY = "running_mean_std.running_mean"
CRITIC_RUNNING_VAR_KEY = "running_mean_std.running_var"
CRITIC_RUNNING_COUNT_KEY = "running_mean_std.count"
ACTION_RESIDUAL_PREFIX = "actor_module.motion_compliance_action_residual."
VALUE_RESIDUAL_PREFIX = "motion_compliance_value_residual."
DEFAULT_NUM_COMPLIANCE_SITES = 2
OFFICIAL_ACTOR_INPUT_WIDTH = 994
OFFICIAL_CRITIC_INPUT_WIDTH = 1645
OFFICIAL_INPUT_HIDDEN_WIDTH = 2048
OFFICIAL_POLICY_TENSOR_COUNT = 55
OFFICIAL_VALUE_TENSOR_COUNT = 17
COMPLIANCE_CONDITION_WIDTH = 3
COMPLIANCE_PRIVILEGED_FIXED_WIDTH = 1
COMPLIANCE_PRIVILEGED_WIDTH_PER_SITE = 4
ACTION_DIM = 29
RESIDUAL_HIDDEN_DIMS = (256, 256)
RESIDUAL_DTYPE = torch.float32

_INITIALIZATION_KEYS = frozenset(
    {
        "policy_state_dict",
        VALUE_STATE_KEY,
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "env_state_dict",
        "state",
        MOTION_COMPLIANCE_INITIALIZATION_KEY,
    }
)
_TRAINED_CHECKPOINT_KEYS = frozenset(
    {
        "policy_state_dict",
        VALUE_STATE_KEY,
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "env_state_dict",
        "state",
        "args",
    }
)
TRAINER_STATE_SAVED_KEYS = frozenset(
    {
        "epoch",
        "global_step",
        "max_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "train_batch_size",
        "num_train_epochs",
        "num_input_tokens_seen",
        "total_flos",
        "best_metric",
        "best_global_step",
        "best_model_checkpoint",
        "is_local_process_zero",
        "is_world_process_zero",
        "is_hyper_param_search",
        "trial_name",
        "trial_params",
        "stateful_callbacks",
        "episode",
        "rewbuffer",
        "lenbuffer",
        "cur_reward_sum",
        "cur_episode_length",
        "tot_timesteps",
        "tot_time",
        "eval_step",
        "eval_render_step",
    }
)
TRAINER_STATE_NOT_RESTORED_KEYS = frozenset(
    {
        "stateful_callbacks",
        "is_local_process_zero",
        "is_world_process_zero",
    }
)


@dataclass(frozen=True)
class CheckpointInitializationReport:
    """Auditable result of composing frozen release state and target residual state."""

    source_sha256: str
    source_revision: str
    source_global_step: int
    source_policy_key: str
    num_sites: int
    release_actor_input_width: int
    action_residual_context_width: int
    release_critic_input_width: int
    critic_running_width: int
    critic_running_count: float
    value_residual_context_width: int
    policy_residual_keys: tuple[str, ...]
    value_residual_keys: tuple[str, ...]
    frozen_policy_tensor_count: int
    frozen_value_tensor_count: int
    initial_trainer_global_step: int
    output_path: str | None = None


@dataclass(frozen=True)
class LoadStateReport:
    """Model-loading mode and key contract used by the trainer."""

    policy_key: str
    strict: bool
    residual_init: bool


@dataclass(frozen=True)
class ResumeStatePayload:
    """Complete training-state payload required for a strict branch resume."""

    optimizer_state_dict: Mapping[str, Any]
    lr_scheduler_state_dict: Mapping[str, Any]
    env_state_dict: Mapping[str, Any]
    state: Any


def compliance_privileged_width(num_sites: int) -> int:
    """Return critic-only compliance width: threshold plus force/mask per site."""

    if isinstance(num_sites, bool) or not isinstance(num_sites, int) or num_sites <= 0:
        raise ValueError("num_sites must be a positive integer")
    return COMPLIANCE_PRIVILEGED_FIXED_WIDTH + (
        COMPLIANCE_PRIVILEGED_WIDTH_PER_SITE * num_sites
    )


def action_residual_context_width() -> int:
    """Return the separate action residual context width; the base stays 994."""

    return OFFICIAL_ACTOR_INPUT_WIDTH + COMPLIANCE_CONDITION_WIDTH


def value_residual_context_width(num_sites: int) -> int:
    """Return the separate value residual context width; the base stays 1645."""

    return (
        OFFICIAL_CRITIC_INPUT_WIDTH
        + COMPLIANCE_CONDITION_WIDTH
        + compliance_privileged_width(num_sites)
    )


def expected_residual_shapes(
    num_sites: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Return the locked Phase-4 residual tensor schema."""

    action_context = action_residual_context_width()
    value_context = value_residual_context_width(num_sites)
    hidden0, hidden1 = RESIDUAL_HIDDEN_DIMS
    policy = {
        f"{ACTION_RESIDUAL_PREFIX}module.0.weight": (hidden0, action_context),
        f"{ACTION_RESIDUAL_PREFIX}module.0.bias": (hidden0,),
        f"{ACTION_RESIDUAL_PREFIX}module.2.weight": (hidden1, hidden0),
        f"{ACTION_RESIDUAL_PREFIX}module.2.bias": (hidden1,),
        f"{ACTION_RESIDUAL_PREFIX}module.4.weight": (ACTION_DIM, hidden1),
        f"{ACTION_RESIDUAL_PREFIX}module.4.bias": (ACTION_DIM,),
    }
    value = {
        f"{VALUE_RESIDUAL_PREFIX}module.0.weight": (hidden0, value_context),
        f"{VALUE_RESIDUAL_PREFIX}module.0.bias": (hidden0,),
        f"{VALUE_RESIDUAL_PREFIX}module.2.weight": (hidden1, hidden0),
        f"{VALUE_RESIDUAL_PREFIX}module.2.bias": (hidden1,),
        f"{VALUE_RESIDUAL_PREFIX}module.4.weight": (1, hidden1),
        f"{VALUE_RESIDUAL_PREFIX}module.4.bias": (1,),
    }
    return policy, value


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    """Hash a checkpoint incrementally without loading or mutating it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_sha256(
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> str:
    """Require an exact lowercase SHA-256 before checkpoint deserialization."""

    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("expected_sha256 must be hexadecimal") from error
    expected_sha256 = expected_sha256.lower()
    actual_sha256 = checkpoint_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def _validate_sha256_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must contain exactly 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} must be hexadecimal") from error
    return value.lower()


def load_trl_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Install the repository TRL compatibility aliases before unpickling."""

    from gear_sonic.trl.trainer import ppo_trainer as _trl_checkpoint_compat  # noqa: F401

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dict")
    return checkpoint


def _resolve_policy_state(
    checkpoint: Mapping[str, Any],
) -> tuple[str, Mapping[str, torch.Tensor]]:
    present = [key for key in POLICY_STATE_KEYS if key in checkpoint]
    if len(present) != 1:
        raise ValueError(
            "checkpoint must contain exactly one policy state key; "
            f"found {present or 'none'}"
        )
    state = checkpoint[present[0]]
    if not isinstance(state, Mapping):
        raise TypeError(f"{present[0]} must be a state-dict mapping")
    return present[0], state


def _require_tensor_state(state: Mapping[str, Any], group_name: str) -> None:
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError(f"{group_name} contains a non-string key")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{group_name}.{key} is not a tensor")


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor representations, including signed zero and NaN payloads."""

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(_tensor_bytes(left), _tensor_bytes(right))
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(tensor).numpy().tobytes()).hexdigest()


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_tensor_bytes(tensor).numpy().tobytes())
    return digest.hexdigest()


def _source_global_step(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("state")
    step = getattr(state, "global_step", None)
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint state.global_step must be a non-negative int")
    return step


def _validate_official_source_shapes(
    policy_state: Mapping[str, torch.Tensor],
    value_state: Mapping[str, torch.Tensor],
) -> tuple[int, int, float]:
    if len(policy_state) != OFFICIAL_POLICY_TENSOR_COUNT:
        raise ValueError(
            "official policy tensor count must remain "
            f"{OFFICIAL_POLICY_TENSOR_COUNT}; got {len(policy_state)}"
        )
    if len(value_state) != OFFICIAL_VALUE_TENSOR_COUNT:
        raise ValueError(
            "official value tensor count must remain "
            f"{OFFICIAL_VALUE_TENSOR_COUNT}; got {len(value_state)}"
        )
    actor_weight = policy_state.get(ACTOR_INPUT_WEIGHT_KEY)
    critic_weight = value_state.get(CRITIC_INPUT_WEIGHT_KEY)
    running_mean = value_state.get(CRITIC_RUNNING_MEAN_KEY)
    running_var = value_state.get(CRITIC_RUNNING_VAR_KEY)
    running_count = value_state.get(CRITIC_RUNNING_COUNT_KEY)
    for key, tensor in (
        (ACTOR_INPUT_WEIGHT_KEY, actor_weight),
        (CRITIC_INPUT_WEIGHT_KEY, critic_weight),
        (CRITIC_RUNNING_MEAN_KEY, running_mean),
        (CRITIC_RUNNING_VAR_KEY, running_var),
        (CRITIC_RUNNING_COUNT_KEY, running_count),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"official checkpoint lacks tensor {key}")
    if tuple(actor_weight.shape) != (
        OFFICIAL_INPUT_HIDDEN_WIDTH,
        OFFICIAL_ACTOR_INPUT_WIDTH,
    ):
        raise ValueError(
            "official actor input must have shape "
            f"({OFFICIAL_INPUT_HIDDEN_WIDTH}, {OFFICIAL_ACTOR_INPUT_WIDTH}); "
            f"got {tuple(actor_weight.shape)}"
        )
    if tuple(critic_weight.shape) != (
        OFFICIAL_INPUT_HIDDEN_WIDTH,
        OFFICIAL_CRITIC_INPUT_WIDTH,
    ):
        raise ValueError(
            "official critic input must have shape "
            f"({OFFICIAL_INPUT_HIDDEN_WIDTH}, {OFFICIAL_CRITIC_INPUT_WIDTH}); "
            f"got {tuple(critic_weight.shape)}"
        )
    if tuple(running_mean.shape) != (OFFICIAL_CRITIC_INPUT_WIDTH,):
        raise ValueError("official critic running mean width differs from critic input")
    if tuple(running_var.shape) != (OFFICIAL_CRITIC_INPUT_WIDTH,):
        raise ValueError("official critic running variance width differs from critic input")
    if running_count.ndim != 0 or not running_count.is_floating_point():
        raise ValueError("official critic running count must be a floating scalar tensor")
    return (
        int(actor_weight.shape[1]),
        int(critic_weight.shape[1]),
        float(running_count.item()),
    )


def _fresh_online_trainer_state() -> Any:
    from gear_sonic.trl.trainer import ppo_trainer as _trl_checkpoint_compat

    return _trl_checkpoint_compat.ppo_trainer.OnlineTrainerState(global_step=0)


def _compose_state(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    expected_residual: Mapping[str, tuple[int, ...]],
    *,
    group_name: str,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    _require_tensor_state(source, f"source_{group_name}")
    _require_tensor_state(target, f"target_{group_name}")
    missing_base = sorted(set(source) - set(target))
    residual_keys = tuple(sorted(set(target) - set(source)))
    unexpected_residual = sorted(set(residual_keys) - set(expected_residual))
    missing_residual = sorted(set(expected_residual) - set(residual_keys))
    if missing_base or unexpected_residual or missing_residual:
        raise ValueError(
            f"{group_name} target schema differs: missing_base={missing_base}, "
            f"missing_residual={missing_residual}, unexpected_residual={unexpected_residual}"
        )

    composed: dict[str, torch.Tensor] = {}
    for key, source_tensor in source.items():
        target_tensor = target[key]
        if (
            source_tensor.shape != target_tensor.shape
            or source_tensor.dtype != target_tensor.dtype
        ):
            raise ValueError(
                f"release {group_name}.{key} shape/dtype changed: "
                f"source={tuple(source_tensor.shape)}/{source_tensor.dtype}, "
                f"target={tuple(target_tensor.shape)}/{target_tensor.dtype}"
            )
        composed[key] = source_tensor.detach().cpu().clone()

    for key in residual_keys:
        target_tensor = target[key]
        expected_shape = expected_residual[key]
        if tuple(target_tensor.shape) != expected_shape:
            raise ValueError(
                f"{group_name}.{key} shape differs: expected {expected_shape}, "
                f"got {tuple(target_tensor.shape)}"
            )
        if target_tensor.dtype != RESIDUAL_DTYPE or not torch.isfinite(target_tensor).all():
            raise ValueError(
                f"{group_name}.{key} must be a finite {RESIDUAL_DTYPE} tensor"
            )
        composed[key] = target_tensor.detach().cpu().clone()

    # Lexicographic ordering puts module.4.bias before module.4.weight, so use
    # explicit suffix selection instead of relying on that ordering.
    final_keys = tuple(key for key in residual_keys if ".module.4." in key)
    if len(final_keys) != 2:
        raise ValueError(f"{group_name} residual lacks the zero-initialized output layer")
    for key in final_keys:
        if torch.count_nonzero(composed[key]).item() != 0:
            raise ValueError(f"{group_name}.{key} must be zero initialized")
    return composed, residual_keys


def initialize_motion_compliance_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source_sha256: str,
    expected_source_step: int = OFFICIAL_SONIC_RELEASE_STEP,
    source_revision: str = OFFICIAL_SONIC_RELEASE_REVISION,
    num_sites: int = DEFAULT_NUM_COMPLIANCE_SITES,
    target_policy_state: Mapping[str, torch.Tensor] | None = None,
    target_value_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], CheckpointInitializationReport]:
    """Compose an init checkpoint without changing any released tensor shape/value."""

    source_sha256 = _validate_sha256_text(source_sha256, "source_sha256")
    if source_sha256 != OFFICIAL_SONIC_RELEASE_SHA256:
        raise ValueError("source_sha256 does not identify the audited official SONIC release")
    if expected_source_step != OFFICIAL_SONIC_RELEASE_STEP:
        raise ValueError(f"expected_source_step must remain {OFFICIAL_SONIC_RELEASE_STEP}")
    if source_revision != OFFICIAL_SONIC_RELEASE_REVISION:
        raise ValueError(f"source_revision must remain {OFFICIAL_SONIC_RELEASE_REVISION}")
    if target_policy_state is None or target_value_state is None:
        raise ValueError("strict residual initialization requires target policy/value states")
    if MOTION_COMPLIANCE_INITIALIZATION_KEY in checkpoint:
        raise ValueError("checkpoint is already motion-compliance initialized")
    if LEGACY_MOTION_COMPLIANCE_MIGRATION_KEY in checkpoint:
        raise ValueError("legacy expanded motion-compliance artifacts are invalid inputs")

    policy_key, source_policy = _resolve_policy_state(checkpoint)
    if policy_key != "policy_state_dict":
        raise ValueError("audited official source_policy_key must be 'policy_state_dict'")
    source_value = checkpoint.get(VALUE_STATE_KEY)
    if not isinstance(source_value, Mapping):
        raise ValueError("checkpoint must contain a value_state_dict mapping")
    source_step = _source_global_step(checkpoint)
    if source_step != expected_source_step:
        raise ValueError(
            f"source global step mismatch: expected {expected_source_step}, got {source_step}"
        )
    actor_width, critic_width, critic_running_count = _validate_official_source_shapes(
        source_policy,
        source_value,
    )

    policy_shapes, value_shapes = expected_residual_shapes(num_sites)
    initialized_policy, policy_residual_keys = _compose_state(
        source_policy,
        target_policy_state,
        policy_shapes,
        group_name="policy",
    )
    initialized_value, value_residual_keys = _compose_state(
        source_value,
        target_value_state,
        value_shapes,
        group_name="value",
    )
    source_policy_cpu = {
        key: tensor.detach().cpu() for key, tensor in source_policy.items()
    }
    source_value_cpu = {key: tensor.detach().cpu() for key, tensor in source_value.items()}
    residual_digests = {
        key: _tensor_digest(initialized_policy[key]) for key in policy_residual_keys
    }
    residual_digests.update(
        {key: _tensor_digest(initialized_value[key]) for key in value_residual_keys}
    )
    metadata = {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "initialization_kind": "same_shape_release_plus_residual",
        "source_sha256": source_sha256,
        "source_revision": source_revision,
        "source_global_step": source_step,
        "source_policy_key": policy_key,
        "num_compliance_sites": num_sites,
        "release_actor_input_width": actor_width,
        "action_residual_context_width": action_residual_context_width(),
        "release_critic_input_width": critic_width,
        "value_residual_context_width": value_residual_context_width(num_sites),
        "critic_running_width": critic_width,
        "critic_running_count": critic_running_count,
        "policy_residual_keys": list(policy_residual_keys),
        "value_residual_keys": list(value_residual_keys),
        "residual_shapes": {
            key: list(shape) for key, shape in {**policy_shapes, **value_shapes}.items()
        },
        "residual_dtypes": {
            key: str(RESIDUAL_DTYPE) for key in {**policy_shapes, **value_shapes}
        },
        "residual_initial_sha256": residual_digests,
        "official_policy_state_sha256": _state_digest(source_policy_cpu),
        "official_value_state_sha256": _state_digest(source_value_cpu),
        "frozen_policy_tensor_count": len(source_policy),
        "frozen_value_tensor_count": len(source_value),
        "initial_trainer_global_step": 0,
        "old_training_state_discarded": True,
        "old_env_state_discarded": True,
    }
    initialized_checkpoint = {
        "policy_state_dict": initialized_policy,
        VALUE_STATE_KEY: initialized_value,
        "optimizer_state_dict": None,
        "lr_scheduler_state_dict": None,
        "env_state_dict": None,
        "state": _fresh_online_trainer_state(),
        MOTION_COMPLIANCE_INITIALIZATION_KEY: metadata,
    }
    report = CheckpointInitializationReport(
        source_sha256=source_sha256,
        source_revision=source_revision,
        source_global_step=source_step,
        source_policy_key=policy_key,
        num_sites=num_sites,
        release_actor_input_width=actor_width,
        action_residual_context_width=action_residual_context_width(),
        release_critic_input_width=critic_width,
        critic_running_width=critic_width,
        critic_running_count=critic_running_count,
        value_residual_context_width=value_residual_context_width(num_sites),
        policy_residual_keys=policy_residual_keys,
        value_residual_keys=value_residual_keys,
        frozen_policy_tensor_count=len(source_policy),
        frozen_value_tensor_count=len(source_value),
        initial_trainer_global_step=0,
    )
    audit_residual_init_checkpoint(initialized_checkpoint)
    return initialized_checkpoint, report


def audit_residual_init_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject malformed provenance, shape expansion, or training-state leakage."""

    keys = frozenset(checkpoint)
    if keys != _INITIALIZATION_KEYS:
        raise ValueError(
            f"residual init keys differ: missing={sorted(_INITIALIZATION_KEYS - keys)}, "
            f"unexpected={sorted(keys - _INITIALIZATION_KEYS)}"
        )
    metadata = checkpoint[MOTION_COMPLIANCE_INITIALIZATION_KEY]
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != (
        INITIALIZATION_SCHEMA_VERSION
    ):
        raise ValueError("invalid motion-compliance residual initialization metadata")
    if metadata.get("initialization_kind") != "same_shape_release_plus_residual":
        raise ValueError("invalid motion-compliance initialization kind")
    for key in ("optimizer_state_dict", "lr_scheduler_state_dict", "env_state_dict"):
        if checkpoint[key] is not None:
            raise ValueError(f"residual init must not carry {key}")
    if getattr(checkpoint["state"], "global_step", None) != 0:
        raise ValueError("residual init must contain a fresh global_step=0 trainer state")

    policy_state = checkpoint["policy_state_dict"]
    value_state = checkpoint[VALUE_STATE_KEY]
    if not isinstance(policy_state, Mapping) or not isinstance(value_state, Mapping):
        raise ValueError("residual init must contain policy/value state mappings")
    _require_tensor_state(policy_state, "policy")
    _require_tensor_state(value_state, "value")
    num_sites = metadata.get("num_compliance_sites")
    policy_shapes, value_shapes = expected_residual_shapes(num_sites)
    required_metadata = {
        "source_sha256": OFFICIAL_SONIC_RELEASE_SHA256,
        "source_revision": OFFICIAL_SONIC_RELEASE_REVISION,
        "source_global_step": OFFICIAL_SONIC_RELEASE_STEP,
        "source_policy_key": "policy_state_dict",
        "release_actor_input_width": OFFICIAL_ACTOR_INPUT_WIDTH,
        "action_residual_context_width": action_residual_context_width(),
        "release_critic_input_width": OFFICIAL_CRITIC_INPUT_WIDTH,
        "value_residual_context_width": value_residual_context_width(num_sites),
        "critic_running_width": OFFICIAL_CRITIC_INPUT_WIDTH,
        "frozen_policy_tensor_count": OFFICIAL_POLICY_TENSOR_COUNT,
        "frozen_value_tensor_count": OFFICIAL_VALUE_TENSOR_COUNT,
        "policy_residual_keys": sorted(policy_shapes),
        "value_residual_keys": sorted(value_shapes),
        "residual_shapes": {
            key: list(shape) for key, shape in {**policy_shapes, **value_shapes}.items()
        },
        "residual_dtypes": {
            key: str(RESIDUAL_DTYPE) for key in {**policy_shapes, **value_shapes}
        },
        "initial_trainer_global_step": 0,
        "old_training_state_discarded": True,
        "old_env_state_discarded": True,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"invalid motion-compliance initialization metadata {key}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )

    policy_residual_keys = set(policy_shapes)
    value_residual_keys = set(value_shapes)
    if not policy_residual_keys.issubset(policy_state) or not value_residual_keys.issubset(
        value_state
    ):
        raise ValueError("residual init lacks one or more residual tensors")
    if any(
        key.startswith(ACTION_RESIDUAL_PREFIX)
        for key in set(policy_state) - policy_residual_keys
    ):
        raise ValueError("residual init contains an unexpected policy residual tensor")
    if any(
        key.startswith(VALUE_RESIDUAL_PREFIX)
        for key in set(value_state) - value_residual_keys
    ):
        raise ValueError("residual init contains an unexpected value residual tensor")

    digests = metadata.get("residual_initial_sha256")
    if (
        not isinstance(digests, Mapping)
        or set(digests) != policy_residual_keys | value_residual_keys
    ):
        raise ValueError("residual init digest keys differ")
    for state, shapes in ((policy_state, policy_shapes), (value_state, value_shapes)):
        for key, expected_shape in shapes.items():
            tensor = state[key]
            if tuple(tensor.shape) != expected_shape or tensor.dtype != RESIDUAL_DTYPE:
                raise ValueError(f"residual init tensor schema differs for {key}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"residual init tensor is non-finite: {key}")
            if digests.get(key) != _tensor_digest(tensor):
                raise ValueError(f"residual init tensor digest differs: {key}")

    policy_base = {key: value for key, value in policy_state.items() if key not in policy_shapes}
    value_base = {key: value for key, value in value_state.items() if key not in value_shapes}
    if len(policy_base) != metadata.get("frozen_policy_tensor_count") or len(
        value_base
    ) != metadata.get("frozen_value_tensor_count"):
        raise ValueError("residual init frozen tensor count differs")
    if _state_digest(policy_base) != metadata.get("official_policy_state_sha256"):
        raise ValueError("residual init policy base digest differs")
    if _state_digest(value_base) != metadata.get("official_value_state_sha256"):
        raise ValueError("residual init value base digest differs")
    _validate_official_source_shapes(policy_base, value_base)


def _atomic_torch_save(checkpoint: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
        torch.save(dict(checkpoint), temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def initialize_motion_compliance_checkpoint_file(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    expected_sha256: str = OFFICIAL_SONIC_RELEASE_SHA256,
    expected_source_step: int = OFFICIAL_SONIC_RELEASE_STEP,
    source_revision: str = OFFICIAL_SONIC_RELEASE_REVISION,
    num_sites: int = DEFAULT_NUM_COMPLIANCE_SITES,
    target_policy_state: Mapping[str, torch.Tensor],
    target_value_state: Mapping[str, torch.Tensor],
    overwrite: bool = False,
) -> CheckpointInitializationReport:
    """Hash, compat-load, initialize, audit, and atomically save one artifact."""

    source = Path(source_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    if source == output:
        raise ValueError("initialization output must not overwrite the source checkpoint")
    if output.exists() and not overwrite:
        raise FileExistsError(f"initialization output already exists: {output}")
    source_sha256 = validate_checkpoint_sha256(source, expected_sha256)
    checkpoint = load_trl_checkpoint(source, map_location="cpu")
    initialized, report = initialize_motion_compliance_checkpoint(
        checkpoint,
        source_sha256=source_sha256,
        expected_source_step=expected_source_step,
        source_revision=source_revision,
        num_sites=num_sites,
        target_policy_state=target_policy_state,
        target_value_state=target_value_state,
    )
    _atomic_torch_save(initialized, output)
    loaded = load_trl_checkpoint(output, map_location="cpu")
    audit_residual_init_checkpoint(loaded)
    return CheckpointInitializationReport(**{**report.__dict__, "output_path": str(output)})


def initialize_official_sonic_release_checkpoint_file(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    num_sites: int = DEFAULT_NUM_COMPLIANCE_SITES,
    target_policy_state: Mapping[str, torch.Tensor],
    target_value_state: Mapping[str, torch.Tensor],
    overwrite: bool = False,
) -> CheckpointInitializationReport:
    """Initialize only from the pinned, audited SONIC release checkpoint."""

    return initialize_motion_compliance_checkpoint_file(
        source_path,
        output_path,
        expected_sha256=OFFICIAL_SONIC_RELEASE_SHA256,
        expected_source_step=OFFICIAL_SONIC_RELEASE_STEP,
        source_revision=OFFICIAL_SONIC_RELEASE_REVISION,
        num_sites=num_sites,
        target_policy_state=target_policy_state,
        target_value_state=target_value_state,
        overwrite=overwrite,
    )


def strict_load_policy_value_state(
    policy: torch.nn.Module,
    value_model: torch.nn.Module | None,
    checkpoint: Mapping[str, Any],
    *,
    resume: bool,
) -> LoadStateReport:
    """Load models with strict residual-init and strict branch-resume semantics."""

    if LEGACY_MOTION_COMPLIANCE_MIGRATION_KEY in checkpoint:
        raise ValueError("legacy expanded motion-compliance artifacts are invalid")
    policy_key, policy_state = _resolve_policy_state(checkpoint)
    residual_init = MOTION_COMPLIANCE_INITIALIZATION_KEY in checkpoint
    if residual_init:
        audit_residual_init_checkpoint(checkpoint)
        if resume:
            raise ValueError("a residual init artifact cannot be used with resume=true")
    elif not resume:
        raise ValueError(
            "resume=false requires a schema-v2 motion-compliance residual init artifact"
        )
    if resume:
        validate_strict_resume_payload(checkpoint)

    value_state = checkpoint.get(VALUE_STATE_KEY)
    if value_model is None:
        raise ValueError("strict compliance checkpoint load requires a value model")
    if not isinstance(value_state, Mapping):
        raise ValueError("strict checkpoint load requires value_state_dict")
    _preflight_model_state(policy, policy_state, "policy")
    _preflight_model_state(value_model, value_state, "value")
    policy.load_state_dict(policy_state, strict=True)
    value_model.load_state_dict(value_state, strict=True)
    return LoadStateReport(policy_key=policy_key, strict=True, residual_init=residual_init)


def _preflight_model_state(
    model: torch.nn.Module,
    incoming: Mapping[str, Any],
    group_name: str,
) -> None:
    """Validate a full model state before mutating either live model."""

    _require_tensor_state(incoming, group_name)
    current = model.state_dict()
    missing = sorted(set(current) - set(incoming))
    unexpected = sorted(set(incoming) - set(current))
    if missing or unexpected:
        raise ValueError(
            f"strict {group_name} keys differ: missing={missing}, unexpected={unexpected}"
        )
    for key, current_tensor in current.items():
        incoming_tensor = incoming[key]
        if (
            current_tensor.shape != incoming_tensor.shape
            or current_tensor.dtype != incoming_tensor.dtype
        ):
            raise ValueError(
                f"strict {group_name}.{key} shape/dtype differs: "
                f"current={tuple(current_tensor.shape)}/{current_tensor.dtype}, "
                f"checkpoint={tuple(incoming_tensor.shape)}/{incoming_tensor.dtype}"
            )
        if not torch.isfinite(incoming_tensor).all():
            raise ValueError(f"strict {group_name}.{key} contains NaN or Inf")


def _require_positive_finite_sequence(value: Any, field_name: str) -> None:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"strict resume requires non-empty {field_name}")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not math.isfinite(float(item))
        or float(item) <= 0.0
        for item in value
    ):
        raise ValueError(f"strict resume requires finite positive {field_name}")


def _require_finite_real(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"strict resume requires finite numeric {field_name}")
    return float(value)


def _require_finite_numeric_tree(value: Any, field_name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"strict resume requires finite {field_name}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_numeric_tree(item, f"{field_name}.{key}")
        return
    if isinstance(value, list | tuple | deque):
        for index, item in enumerate(value):
            _require_finite_numeric_tree(item, f"{field_name}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"strict resume requires finite {field_name}")


def _require_finite_buffer_tree(value: Any, field_name: str) -> None:
    if isinstance(value, list | tuple | deque):
        for index, item in enumerate(value):
            _require_finite_buffer_tree(item, f"{field_name}[{index}]")
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"strict resume requires numeric finite leaves in {field_name}")


def validate_optimizer_parameter_group_hyperparameters(
    group: Mapping[str, Any],
    *,
    group_index: int,
) -> None:
    """Validate the numeric and domain contract of one residual AdamW group."""

    field_prefix = f"optimizer param_groups[{group_index}]"
    required = {"lr", "weight_decay", "betas", "eps", "amsgrad"}
    missing = sorted(required - set(group))
    if missing:
        raise ValueError(f"strict resume {field_prefix} lacks {missing}")
    learning_rate = _require_finite_real(group["lr"], f"{field_prefix}.lr")
    if learning_rate <= 0.0:
        raise ValueError(f"strict resume requires positive {field_prefix}.lr")
    weight_decay = _require_finite_real(
        group["weight_decay"],
        f"{field_prefix}.weight_decay",
    )
    if weight_decay < 0.0:
        raise ValueError(f"strict resume requires non-negative {field_prefix}.weight_decay")
    epsilon = _require_finite_real(group["eps"], f"{field_prefix}.eps")
    if epsilon <= 0.0:
        raise ValueError(f"strict resume requires positive {field_prefix}.eps")
    betas = group["betas"]
    if not isinstance(betas, list | tuple) or len(betas) != 2:
        raise ValueError(f"strict resume requires two {field_prefix}.betas")
    for index, beta in enumerate(betas):
        beta_value = _require_finite_real(beta, f"{field_prefix}.betas[{index}]")
        if not 0.0 <= beta_value < 1.0:
            raise ValueError(
                f"strict resume requires 0 <= {field_prefix}.betas[{index}] < 1"
            )
    if group["amsgrad"] is not False:
        raise ValueError(f"strict resume requires {field_prefix}.amsgrad=false")
    if "initial_lr" in group:
        initial_lr = _require_finite_real(
            group["initial_lr"],
            f"{field_prefix}.initial_lr",
        )
        if initial_lr <= 0.0:
            raise ValueError(f"strict resume requires positive {field_prefix}.initial_lr")
    for key, value in group.items():
        if key != "params":
            _require_finite_numeric_tree(value, f"{field_prefix}.{key}")


def validate_strict_resume_payload(
    checkpoint: Mapping[str, Any],
) -> ResumeStatePayload:
    """Require every non-model state needed for an exact training resume."""

    if MOTION_COMPLIANCE_INITIALIZATION_KEY in checkpoint:
        raise ValueError("a residual init artifact cannot be resumed")
    if LEGACY_MOTION_COMPLIANCE_MIGRATION_KEY in checkpoint:
        raise ValueError("legacy expanded motion-compliance artifacts cannot be resumed")
    keys = frozenset(checkpoint)
    if keys != _TRAINED_CHECKPOINT_KEYS:
        raise ValueError(
            "strict resume checkpoint keys differ: "
            f"missing={sorted(_TRAINED_CHECKPOINT_KEYS - keys)}, "
            f"unexpected={sorted(keys - _TRAINED_CHECKPOINT_KEYS)}"
        )

    required_mappings: dict[str, Mapping[str, Any]] = {}
    for key in ("optimizer_state_dict", "lr_scheduler_state_dict", "env_state_dict"):
        value = checkpoint.get(key)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"strict resume requires a non-empty {key} mapping")
        required_mappings[key] = value

    optimizer_state = required_mappings["optimizer_state_dict"]
    optimizer_slots = optimizer_state.get("state")
    parameter_groups = optimizer_state.get("param_groups")
    if not isinstance(optimizer_slots, Mapping) or not optimizer_slots:
        raise ValueError("strict resume requires non-empty optimizer parameter state")
    if not isinstance(parameter_groups, list) or not parameter_groups:
        raise ValueError("strict resume requires non-empty optimizer parameter groups")
    if len(parameter_groups) != 2:
        raise ValueError("strict resume requires the two residual-only optimizer groups")
    optimizer_parameter_ids: list[Any] = []
    for group_index, group in enumerate(parameter_groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("strict resume optimizer parameter group is malformed")
        if len(group["params"]) != 6 or any(
            isinstance(parameter_id, bool)
            or not isinstance(parameter_id, int)
            or parameter_id < 0
            for parameter_id in group["params"]
        ):
            raise ValueError(
                "strict resume optimizer groups must each own six integer parameter ids"
            )
        validate_optimizer_parameter_group_hyperparameters(
            group,
            group_index=group_index,
        )
        optimizer_parameter_ids.extend(group["params"])
    if len(optimizer_parameter_ids) != 12 or len(set(optimizer_parameter_ids)) != 12:
        raise ValueError("strict resume optimizer must own exactly 12 unique residual tensors")
    if set(optimizer_slots) != set(optimizer_parameter_ids):
        raise ValueError("strict resume optimizer slots differ from residual parameter ids")
    for slot in optimizer_slots.values():
        if not isinstance(slot, Mapping):
            raise ValueError("strict resume optimizer slot must be a mapping")
        if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(
                "strict resume AdamW slot keys must be step/exp_avg/exp_avg_sq"
            )
        for key in ("step", "exp_avg", "exp_avg_sq"):
            tensor = slot.get(key)
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"strict resume optimizer slot requires finite tensor {key}")
        step = slot["step"]
        if (
            step.numel() != 1
            or step.dtype != torch.float32
            or float(step.item()) <= 0.0
            or not float(step.item()).is_integer()
        ):
            raise ValueError(
                "strict resume optimizer slot step must be a positive integer-valued "
                "float32 scalar"
            )

    scheduler_state = required_mappings["lr_scheduler_state_dict"]
    last_epoch = scheduler_state.get("last_epoch")
    if type(last_epoch) is not int or last_epoch < 0:
        raise ValueError("strict resume scheduler last_epoch must be a non-negative int")
    _require_positive_finite_sequence(scheduler_state.get("base_lrs"), "scheduler base_lrs")
    _require_positive_finite_sequence(scheduler_state.get("_last_lr"), "scheduler _last_lr")

    env_state = required_mappings["env_state_dict"]
    if not isinstance(env_state.get("motion_lib"), Mapping):
        raise ValueError("strict resume requires env_state_dict.motion_lib")

    args = checkpoint.get("args")
    learning_rate = getattr(args, "learning_rate", None)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, int | float)
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError("strict resume requires args.learning_rate to be finite and positive")

    state = checkpoint.get("state")
    state_values = getattr(state, "__dict__", None)
    global_step = getattr(state, "global_step", None)
    if not isinstance(state_values, dict) or not state_values:
        raise ValueError("strict resume requires a non-empty trainer state object")
    if set(state_values) != TRAINER_STATE_SAVED_KEYS:
        raise ValueError(
            "strict resume trainer state keys differ: "
            f"missing={sorted(TRAINER_STATE_SAVED_KEYS - set(state_values))}, "
            f"unexpected={sorted(set(state_values) - TRAINER_STATE_SAVED_KEYS)}"
        )
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("strict resume requires a positive integer global_step")
    cur_reward_sum = getattr(state, "cur_reward_sum", None)
    cur_episode_length = getattr(state, "cur_episode_length", None)
    if (
        not isinstance(cur_reward_sum, torch.Tensor)
        or cur_reward_sum.ndim != 2
        or not torch.isfinite(cur_reward_sum).all()
    ):
        raise ValueError("strict resume requires finite 2D state.cur_reward_sum")
    if (
        not isinstance(cur_episode_length, torch.Tensor)
        or cur_episode_length.ndim != 1
        or cur_episode_length.shape[0] != cur_reward_sum.shape[0]
        or not torch.isfinite(cur_episode_length).all()
    ):
        raise ValueError("strict resume requires aligned finite state.cur_episode_length")
    for key in (
        "max_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "num_input_tokens_seen",
        "total_flos",
        "episode",
        "tot_timesteps",
        "eval_step",
        "eval_render_step",
    ):
        value = getattr(state, key, None)
        if type(value) is not int or value < 0:
            raise ValueError(f"strict resume requires non-negative integer state.{key}")
    for key in ("epoch", "num_train_epochs"):
        value = getattr(state, key, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"strict resume requires finite non-negative state.{key}")
    tot_time = getattr(state, "tot_time", None)
    if (
        isinstance(tot_time, bool)
        or not isinstance(tot_time, int | float)
        or not math.isfinite(float(tot_time))
        or float(tot_time) < 0.0
    ):
        raise ValueError("strict resume requires finite non-negative state.tot_time")
    for key in (
        "train_batch_size",
        "best_metric",
        "best_global_step",
        "best_model_checkpoint",
        "trial_name",
        "trial_params",
    ):
        if getattr(state, key, None) is not None:
            raise ValueError(f"Phase-4 strict resume requires state.{key}=None")
    for key in (
        "is_local_process_zero",
        "is_world_process_zero",
        "is_hyper_param_search",
    ):
        if type(getattr(state, key, None)) is not bool:
            raise ValueError(f"strict resume requires boolean state.{key}")
    if state.is_hyper_param_search:
        raise ValueError("Phase-4 strict resume forbids hyperparameter-search state")
    if not isinstance(state.stateful_callbacks, Mapping):
        raise ValueError("strict resume requires mapping state.stateful_callbacks")
    for key in ("rewbuffer", "lenbuffer"):
        value = getattr(state, key, None)
        if not isinstance(value, deque) or value.maxlen != 100:
            raise ValueError(f"strict resume requires maxlen-100 deque state.{key}")
        _require_finite_buffer_tree(value, f"state.{key}")
    if state.episode != global_step * 16:
        raise ValueError("strict resume state.episode differs from 16-env global_step")
    if state.tot_timesteps != global_step * 16 * 24:
        raise ValueError("strict resume state.tot_timesteps differs from the 16x24 rollout")

    return ResumeStatePayload(
        optimizer_state_dict=optimizer_state,
        lr_scheduler_state_dict=required_mappings["lr_scheduler_state_dict"],
        env_state_dict=required_mappings["env_state_dict"],
        state=state,
    )
