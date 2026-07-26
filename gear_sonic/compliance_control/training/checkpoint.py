"""Strict, one-way migration of released SONIC weights for compliance inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


MOTION_COMPLIANCE_MIGRATION_KEY = "motion_compliance_migration"
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
ACTOR_ADDED_COLUMNS = 3
CRITIC_FIXED_ADDED_COLUMNS = 4
CRITIC_ADDED_COLUMNS_PER_SITE = 4
DEFAULT_NUM_COMPLIANCE_SITES = 2
OFFICIAL_ACTOR_INPUT_WIDTH = 994
OFFICIAL_CRITIC_INPUT_WIDTH = 1645
OFFICIAL_INPUT_HIDDEN_WIDTH = 2048

_MIGRATED_INIT_KEYS = frozenset(
    {
        "policy_state_dict",
        VALUE_STATE_KEY,
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "env_state_dict",
        "state",
        MOTION_COMPLIANCE_MIGRATION_KEY,
    }
)


@dataclass(frozen=True)
class _Expansion:
    key: str
    added_columns: int
    axis: int
    fill_value: float


_POLICY_EXPANSIONS = (
    _Expansion(ACTOR_INPUT_WEIGHT_KEY, ACTOR_ADDED_COLUMNS, 1, 0.0),
)


def critic_added_columns(num_sites: int) -> int:
    """Return critic additions: public condition/threshold plus force/mask per site."""

    if isinstance(num_sites, bool) or not isinstance(num_sites, int) or num_sites <= 0:
        raise ValueError("num_sites must be a positive integer")
    return CRITIC_FIXED_ADDED_COLUMNS + CRITIC_ADDED_COLUMNS_PER_SITE * num_sites


def _value_expansions(num_sites: int) -> tuple[_Expansion, ...]:
    added_columns = critic_added_columns(num_sites)
    return (
        _Expansion(CRITIC_INPUT_WEIGHT_KEY, added_columns, 1, 0.0),
        _Expansion(CRITIC_RUNNING_MEAN_KEY, added_columns, 0, 0.0),
        _Expansion(CRITIC_RUNNING_VAR_KEY, added_columns, 0, 1.0),
    )


@dataclass(frozen=True)
class CheckpointMigrationReport:
    """Auditable result of a strict released-checkpoint migration."""

    source_sha256: str
    source_global_step: int
    source_policy_key: str
    num_sites: int
    actor_input_width: tuple[int, int]
    critic_input_width: tuple[int, int]
    critic_running_count: float
    expanded_keys: tuple[str, ...]
    output_path: str | None = None


@dataclass(frozen=True)
class LoadStateReport:
    """Model-loading mode and key contract used by the trainer."""

    policy_key: str
    strict: bool
    migrated_init: bool


@dataclass(frozen=True)
class ResumeStatePayload:
    """Complete training-state payload required for a strict branch resume."""

    optimizer_state_dict: Mapping[str, Any]
    lr_scheduler_state_dict: Mapping[str, Any]
    env_state_dict: Mapping[str, Any]
    state: Any


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


def _expanded_shape(
    source: torch.Tensor,
    expansion: _Expansion,
) -> tuple[int, ...]:
    if source.ndim <= expansion.axis:
        raise ValueError(
            f"{expansion.key} must have axis {expansion.axis}; got shape {tuple(source.shape)}"
        )
    shape = list(source.shape)
    shape[expansion.axis] += expansion.added_columns
    return tuple(shape)


def _validate_target_keys(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor] | None,
    group_name: str,
) -> None:
    if target is None:
        return
    _require_tensor_state(target, f"target_{group_name}")
    missing = sorted(set(source) - set(target))
    unexpected = sorted(set(target) - set(source))
    if missing or unexpected:
        raise ValueError(
            f"{group_name} target keys differ: missing={missing}, unexpected={unexpected}"
        )


def _migrate_state_dict(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor] | None,
    expansions: tuple[_Expansion, ...],
    group_name: str,
) -> dict[str, torch.Tensor]:
    _require_tensor_state(source, group_name)
    _validate_target_keys(source, target, group_name)
    expansion_by_key = {expansion.key: expansion for expansion in expansions}
    missing_expansions = sorted(set(expansion_by_key) - set(source))
    if missing_expansions:
        raise ValueError(f"{group_name} lacks expansion keys: {missing_expansions}")

    migrated: dict[str, torch.Tensor] = {}
    for key, source_tensor in source.items():
        target_tensor = None if target is None else target[key]
        expansion = expansion_by_key.get(key)
        if expansion is None:
            if target_tensor is not None and (
                source_tensor.shape != target_tensor.shape
                or source_tensor.dtype != target_tensor.dtype
            ):
                raise ValueError(
                    f"unexpected {group_name} shape/dtype change for {key}: "
                    f"source={tuple(source_tensor.shape)}/{source_tensor.dtype}, "
                    f"target={tuple(target_tensor.shape)}/{target_tensor.dtype}"
                )
            migrated[key] = source_tensor.detach().cpu().clone()
            continue

        expected_shape = _expanded_shape(source_tensor, expansion)
        if target_tensor is not None and (
            tuple(target_tensor.shape) != expected_shape
            or target_tensor.dtype != source_tensor.dtype
        ):
            raise ValueError(
                f"invalid target expansion for {group_name}.{key}: expected "
                f"{expected_shape}/{source_tensor.dtype}, got "
                f"{tuple(target_tensor.shape)}/{target_tensor.dtype}"
            )
        new_tensor = torch.full(
            expected_shape,
            expansion.fill_value,
            dtype=source_tensor.dtype,
            device="cpu",
        )
        legacy_slice = [slice(None)] * source_tensor.ndim
        legacy_slice[expansion.axis] = slice(0, source_tensor.shape[expansion.axis])
        new_tensor[tuple(legacy_slice)].copy_(source_tensor.detach().cpu())
        migrated[key] = new_tensor
    return migrated


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
    if actor_weight.ndim != 2:
        raise ValueError(f"official actor input weight must be 2D; got {actor_weight.ndim}D")
    if critic_weight.ndim != 2:
        raise ValueError(f"official critic input weight must be 2D; got {critic_weight.ndim}D")
    actor_width = actor_weight.shape[1]
    critic_width = critic_weight.shape[1]
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
    if tuple(running_mean.shape) != (critic_width,):
        raise ValueError("official critic running mean width differs from critic input")
    if tuple(running_var.shape) != (critic_width,):
        raise ValueError("official critic running variance width differs from critic input")
    if running_count.ndim != 0 or not running_count.is_floating_point():
        raise ValueError("official critic running count must be a floating scalar tensor")
    return actor_width, critic_width, float(running_count.item())


def _fresh_online_trainer_state() -> Any:
    from gear_sonic.trl.trainer import ppo_trainer as _trl_checkpoint_compat

    return _trl_checkpoint_compat.ppo_trainer.OnlineTrainerState(global_step=0)


def migrate_motion_compliance_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source_sha256: str,
    expected_source_step: int = OFFICIAL_SONIC_RELEASE_STEP,
    source_revision: str = OFFICIAL_SONIC_RELEASE_REVISION,
    num_sites: int = DEFAULT_NUM_COMPLIANCE_SITES,
    target_policy_state: Mapping[str, torch.Tensor] | None = None,
    target_value_state: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], CheckpointMigrationReport]:
    """Migrate exactly the actor/critic input tails and discard old training state."""

    source_sha256 = _validate_sha256_text(source_sha256, "source_sha256")
    if source_sha256 != OFFICIAL_SONIC_RELEASE_SHA256:
        raise ValueError(
            "source_sha256 does not identify the audited official SONIC release"
        )
    if expected_source_step != OFFICIAL_SONIC_RELEASE_STEP:
        raise ValueError(
            f"expected_source_step must remain {OFFICIAL_SONIC_RELEASE_STEP}"
        )
    if source_revision != OFFICIAL_SONIC_RELEASE_REVISION:
        raise ValueError(
            f"source_revision must remain {OFFICIAL_SONIC_RELEASE_REVISION}"
        )
    critic_columns = critic_added_columns(num_sites)
    if target_policy_state is None or target_value_state is None:
        raise ValueError("strict migration requires target policy and value state dicts")
    if MOTION_COMPLIANCE_MIGRATION_KEY in checkpoint:
        raise ValueError("checkpoint is already motion-compliance migrated")
    policy_key, source_policy = _resolve_policy_state(checkpoint)
    source_value = checkpoint.get(VALUE_STATE_KEY)
    if not isinstance(source_value, Mapping):
        raise ValueError("checkpoint must contain a value_state_dict mapping")
    source_step = _source_global_step(checkpoint)
    if source_step != expected_source_step:
        raise ValueError(
            f"source global step mismatch: expected {expected_source_step}, got {source_step}"
        )
    actor_old_width, critic_old_width, critic_running_count = _validate_official_source_shapes(
        source_policy,
        source_value,
    )

    migrated_policy = _migrate_state_dict(
        source_policy,
        target_policy_state,
        _POLICY_EXPANSIONS,
        "policy",
    )
    migrated_value = _migrate_state_dict(
        source_value,
        target_value_state,
        _value_expansions(num_sites),
        "value",
    )
    metadata = {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "source_revision": source_revision,
        "source_global_step": source_step,
        "source_policy_key": policy_key,
        "num_compliance_sites": num_sites,
        "actor_added_input_columns": ACTOR_ADDED_COLUMNS,
        "critic_added_input_columns": critic_columns,
        "actor_input_width": [actor_old_width, actor_old_width + ACTOR_ADDED_COLUMNS],
        "critic_input_width": [critic_old_width, critic_old_width + critic_columns],
        "critic_running_count": critic_running_count,
        "expanded_keys": [
            ACTOR_INPUT_WEIGHT_KEY,
            CRITIC_INPUT_WEIGHT_KEY,
            CRITIC_RUNNING_MEAN_KEY,
            CRITIC_RUNNING_VAR_KEY,
        ],
        "initial_trainer_global_step": 0,
        "old_training_state_discarded": True,
        "old_env_state_discarded": True,
    }
    migrated_checkpoint = {
        "policy_state_dict": migrated_policy,
        VALUE_STATE_KEY: migrated_value,
        "optimizer_state_dict": None,
        "lr_scheduler_state_dict": None,
        "env_state_dict": None,
        "state": _fresh_online_trainer_state(),
        MOTION_COMPLIANCE_MIGRATION_KEY: metadata,
    }
    report = CheckpointMigrationReport(
        source_sha256=source_sha256,
        source_global_step=source_step,
        source_policy_key=policy_key,
        num_sites=num_sites,
        actor_input_width=(actor_old_width, actor_old_width + ACTOR_ADDED_COLUMNS),
        critic_input_width=(critic_old_width, critic_old_width + critic_columns),
        critic_running_count=critic_running_count,
        expanded_keys=tuple(metadata["expanded_keys"]),
    )
    return migrated_checkpoint, report


def audit_migrated_init_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject training-state leakage or malformed provenance in an init artifact."""

    keys = frozenset(checkpoint)
    if keys != _MIGRATED_INIT_KEYS:
        raise ValueError(
            f"migrated init keys differ: missing={sorted(_MIGRATED_INIT_KEYS - keys)}, "
            f"unexpected={sorted(keys - _MIGRATED_INIT_KEYS)}"
        )
    metadata = checkpoint[MOTION_COMPLIANCE_MIGRATION_KEY]
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != 1:
        raise ValueError("invalid motion-compliance migration metadata")
    if checkpoint["optimizer_state_dict"] is not None:
        raise ValueError("migrated init must not carry optimizer state")
    if checkpoint["lr_scheduler_state_dict"] is not None:
        raise ValueError("migrated init must not carry scheduler state")
    if checkpoint["env_state_dict"] is not None:
        raise ValueError("migrated init must not carry environment state")
    if getattr(checkpoint["state"], "global_step", None) != 0:
        raise ValueError("migrated init must contain a fresh global_step=0 trainer state")
    policy_state = checkpoint["policy_state_dict"]
    value_state = checkpoint[VALUE_STATE_KEY]
    if not isinstance(policy_state, Mapping) or not isinstance(value_state, Mapping):
        raise ValueError("migrated init must contain policy/value state mappings")
    _require_tensor_state(policy_state, "policy")
    _require_tensor_state(value_state, "value")
    num_sites = metadata.get("num_compliance_sites")
    added_critic_columns = critic_added_columns(num_sites)
    required_metadata = {
        "source_sha256": OFFICIAL_SONIC_RELEASE_SHA256,
        "source_revision": OFFICIAL_SONIC_RELEASE_REVISION,
        "source_global_step": OFFICIAL_SONIC_RELEASE_STEP,
        "source_policy_key": "policy_state_dict",
        "actor_added_input_columns": ACTOR_ADDED_COLUMNS,
        "critic_added_input_columns": added_critic_columns,
        "actor_input_width": [
            OFFICIAL_ACTOR_INPUT_WIDTH,
            OFFICIAL_ACTOR_INPUT_WIDTH + ACTOR_ADDED_COLUMNS,
        ],
        "critic_input_width": [
            OFFICIAL_CRITIC_INPUT_WIDTH,
            OFFICIAL_CRITIC_INPUT_WIDTH + added_critic_columns,
        ],
        "expanded_keys": [
            ACTOR_INPUT_WEIGHT_KEY,
            CRITIC_INPUT_WEIGHT_KEY,
            CRITIC_RUNNING_MEAN_KEY,
            CRITIC_RUNNING_VAR_KEY,
        ],
        "initial_trainer_global_step": 0,
        "old_training_state_discarded": True,
        "old_env_state_discarded": True,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"invalid motion-compliance migration metadata {key}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    required_tensor_keys = (
        (policy_state, ACTOR_INPUT_WEIGHT_KEY),
        (value_state, CRITIC_INPUT_WEIGHT_KEY),
        (value_state, CRITIC_RUNNING_MEAN_KEY),
        (value_state, CRITIC_RUNNING_VAR_KEY),
        (value_state, CRITIC_RUNNING_COUNT_KEY),
    )
    for state_dict, key in required_tensor_keys:
        if key not in state_dict:
            raise ValueError(f"migrated init lacks tensor {key}")
    actor_weight = policy_state[ACTOR_INPUT_WEIGHT_KEY]
    critic_weight = value_state[CRITIC_INPUT_WEIGHT_KEY]
    running_mean = value_state[CRITIC_RUNNING_MEAN_KEY]
    running_var = value_state[CRITIC_RUNNING_VAR_KEY]
    running_count = value_state[CRITIC_RUNNING_COUNT_KEY]
    if tuple(actor_weight.shape) != (
        OFFICIAL_INPUT_HIDDEN_WIDTH,
        OFFICIAL_ACTOR_INPUT_WIDTH + ACTOR_ADDED_COLUMNS,
    ):
        raise ValueError("migrated actor input width is invalid")
    expected_critic_width = OFFICIAL_CRITIC_INPUT_WIDTH + added_critic_columns
    if tuple(critic_weight.shape) != (OFFICIAL_INPUT_HIDDEN_WIDTH, expected_critic_width):
        raise ValueError("migrated critic input width is invalid")
    if tuple(running_mean.shape) != (expected_critic_width,) or tuple(
        running_var.shape
    ) != (expected_critic_width,):
        raise ValueError("migrated critic running-stat widths are invalid")
    if torch.count_nonzero(actor_weight[:, -ACTOR_ADDED_COLUMNS:]).item() != 0:
        raise ValueError("migrated actor input tail must be zero")
    if torch.count_nonzero(critic_weight[:, -added_critic_columns:]).item() != 0:
        raise ValueError("migrated critic input tail must be zero")
    if torch.count_nonzero(running_mean[-added_critic_columns:]).item() != 0:
        raise ValueError("migrated critic running-mean tail must be zero")
    if not torch.equal(
        running_var[-added_critic_columns:],
        torch.ones_like(running_var[-added_critic_columns:]),
    ):
        raise ValueError("migrated critic running-variance tail must be one")
    if (
        running_count.ndim != 0
        or not running_count.is_floating_point()
        or float(running_count.item()) != metadata.get("critic_running_count")
    ):
        raise ValueError("migrated critic running count differs from source")


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


def migrate_motion_compliance_checkpoint_file(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    expected_sha256: str = OFFICIAL_SONIC_RELEASE_SHA256,
    expected_source_step: int = OFFICIAL_SONIC_RELEASE_STEP,
    source_revision: str = OFFICIAL_SONIC_RELEASE_REVISION,
    num_sites: int = DEFAULT_NUM_COMPLIANCE_SITES,
    target_policy_state: Mapping[str, torch.Tensor] | None = None,
    target_value_state: Mapping[str, torch.Tensor] | None = None,
    overwrite: bool = False,
) -> CheckpointMigrationReport:
    """Hash, compat-load, strictly migrate, audit, and atomically save one artifact."""

    source = Path(source_path).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    if source == output:
        raise ValueError("migration output must not overwrite the source checkpoint")
    if output.exists() and not overwrite:
        raise FileExistsError(f"migration output already exists: {output}")
    source_sha256 = validate_checkpoint_sha256(source, expected_sha256)
    checkpoint = load_trl_checkpoint(source, map_location="cpu")
    migrated, report = migrate_motion_compliance_checkpoint(
        checkpoint,
        source_sha256=source_sha256,
        expected_source_step=expected_source_step,
        source_revision=source_revision,
        num_sites=num_sites,
        target_policy_state=target_policy_state,
        target_value_state=target_value_state,
    )
    audit_migrated_init_checkpoint(migrated)
    _atomic_torch_save(migrated, output)
    return CheckpointMigrationReport(**{**report.__dict__, "output_path": str(output)})


def migrate_official_sonic_release_checkpoint_file(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    num_sites: int = DEFAULT_NUM_COMPLIANCE_SITES,
    target_policy_state: Mapping[str, torch.Tensor],
    target_value_state: Mapping[str, torch.Tensor],
    overwrite: bool = False,
) -> CheckpointMigrationReport:
    """Migrate only the pinned, audited SONIC release checkpoint contract."""

    return migrate_motion_compliance_checkpoint_file(
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
    """Load models with strict migrated-init and strict branch-resume semantics."""

    policy_key, policy_state = _resolve_policy_state(checkpoint)
    migrated_init = MOTION_COMPLIANCE_MIGRATION_KEY in checkpoint
    if migrated_init:
        audit_migrated_init_checkpoint(checkpoint)
        if resume:
            raise ValueError("a migrated init artifact cannot be used with resume=true")
    strict = bool(resume or migrated_init or policy_key == "actor_model_state_dict")
    policy.load_state_dict(policy_state, strict=strict)

    value_state = checkpoint.get(VALUE_STATE_KEY)
    if strict and value_model is None:
        raise ValueError("strict compliance checkpoint load requires a value model")
    if value_model is not None:
        if value_state is None:
            if strict:
                raise ValueError("strict checkpoint load requires value_state_dict")
        else:
            value_model.load_state_dict(value_state, strict=strict)
    return LoadStateReport(policy_key=policy_key, strict=strict, migrated_init=migrated_init)


def validate_strict_resume_payload(
    checkpoint: Mapping[str, Any],
) -> ResumeStatePayload:
    """Require every non-model state needed for an exact training resume."""

    if MOTION_COMPLIANCE_MIGRATION_KEY in checkpoint:
        raise ValueError("a migrated init artifact cannot be resumed")

    required_mappings: dict[str, Mapping[str, Any]] = {}
    for key in (
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "env_state_dict",
    ):
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

    state = checkpoint.get("state")
    state_values = getattr(state, "__dict__", None)
    global_step = getattr(state, "global_step", None)
    if not isinstance(state_values, dict) or not state_values:
        raise ValueError("strict resume requires a non-empty trainer state object")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("strict resume requires a positive integer global_step")

    return ResumeStatePayload(
        optimizer_state_dict=optimizer_state,
        lr_scheduler_state_dict=required_mappings["lr_scheduler_state_dict"],
        env_state_dict=required_mappings["env_state_dict"],
        state=state,
    )
