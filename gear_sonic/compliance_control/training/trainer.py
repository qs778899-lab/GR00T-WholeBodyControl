"""Compliance-only PPO trainer with strict residual and checkpoint semantics."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math

import torch

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer

from .checkpoint import (
    TRAINER_STATE_NOT_RESTORED_KEYS,
    TRAINER_STATE_SAVED_KEYS,
    load_trl_checkpoint,
    strict_load_policy_value_state,
    validate_optimizer_parameter_group_hyperparameters,
    validate_strict_resume_payload,
)
from .finetune import validate_optimizer_parameter_set
from .residual_policy import (
    CONDITION_OBSERVATION_KEY,
    PRIVILEGED_OBSERVATION_KEY,
    motion_compliance_residual_parameters,
)


PHASE4_PPO_ROLLOUT_STEPS = 24
PHASE4_PPO_MICRO_BATCH_SIZE = 4
PHASE4_ACTION_DIM = 29
PHASE4_PRIVILEGED_DIM = 9


def _require_tensor_shape(
    value,
    expected: tuple[int, ...],
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape must be {expected}; got {tuple(value.shape)}")


def validate_motion_compliance_ppo_batch(
    mb_rollout_data: Mapping[str, object],
    *,
    rollout_steps: int = PHASE4_PPO_ROLLOUT_STEPS,
) -> tuple[int, int]:
    """Require the real temporal PPO contract before the combined forward pass."""

    if rollout_steps != PHASE4_PPO_ROLLOUT_STEPS:
        raise ValueError(
            f"Phase-4 PPO rollout_steps must be {PHASE4_PPO_ROLLOUT_STEPS}; "
            f"got {rollout_steps}"
        )
    obs_dict = mb_rollout_data.get("mb_obs_dict")
    if not isinstance(obs_dict, Mapping):
        raise TypeError("mb_obs_dict must be a mapping")
    required_widths = {
        "actor_obs": 930,
        "critic_obs": 1645,
        CONDITION_OBSERVATION_KEY: 3,
    }
    leading_shape: tuple[int, int] | None = None
    for key, width in required_widths.items():
        tensor = obs_dict.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"mb_obs_dict.{key} must have shape [B,24,{width}]")
        if tensor.shape[1:] != (rollout_steps, width):
            raise ValueError(
                f"mb_obs_dict.{key} must have shape [B,{rollout_steps},{width}]; "
                f"got {tuple(tensor.shape)}"
            )
        if leading_shape is None:
            leading_shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        elif tuple(tensor.shape[:2]) != leading_shape:
            raise ValueError("PPO observation leading shapes differ")
    privileged = obs_dict.get(PRIVILEGED_OBSERVATION_KEY)
    if (
        not isinstance(privileged, torch.Tensor)
        or privileged.ndim != 3
        or tuple(privileged.shape[:2]) != leading_shape
        or privileged.shape[-1] != PHASE4_PRIVILEGED_DIM
    ):
        raise ValueError("motion-compliance privileged PPO input must have shape [4,24,9]")
    tokenizer = obs_dict.get("tokenizer")
    if (
        not isinstance(tokenizer, torch.Tensor)
        or tokenizer.ndim < 3
        or tuple(tokenizer.shape[:2]) != leading_shape
    ):
        raise ValueError("tokenizer PPO input must preserve [4,24,...] leading shape")
    if leading_shape != (PHASE4_PPO_MICRO_BATCH_SIZE, PHASE4_PPO_ROLLOUT_STEPS):
        raise ValueError(
            "Phase-4 PPO micro-batch leading shape must be "
            f"[{PHASE4_PPO_MICRO_BATCH_SIZE},{PHASE4_PPO_ROLLOUT_STEPS}]; "
            f"got {leading_shape}"
        )
    batch_size, temporal_steps = leading_shape
    _require_tensor_shape(
        mb_rollout_data.get("mb_actions"),
        (batch_size, temporal_steps, PHASE4_ACTION_DIM),
        "mb_actions",
    )
    _require_tensor_shape(
        mb_rollout_data.get("episode_attnmask"),
        (batch_size, temporal_steps, temporal_steps),
        "episode_attnmask",
    )
    return leading_shape


def validate_motion_compliance_ppo_outputs(
    forward_results: Mapping[str, object],
    *,
    batch_size: int,
    rollout_steps: int,
) -> None:
    """Require action/value tensors to retain `[B,24,*]` through PPO."""

    policy_results = forward_results.get("policy_results")
    if not isinstance(policy_results, Mapping):
        raise TypeError("policy_results must be a mapping")
    action_shape = (batch_size, rollout_steps, PHASE4_ACTION_DIM)
    for key in ("action_mean", "action_std"):
        _require_tensor_shape(policy_results.get(key), action_shape, f"policy_results.{key}")
    for key in ("logprobs", "entropy"):
        _require_tensor_shape(
            policy_results.get(key),
            (batch_size, rollout_steps),
            f"policy_results.{key}",
        )
    _require_tensor_shape(
        forward_results.get("value_results"),
        (batch_size, rollout_steps, 1),
        "value_results",
    )


def residual_parameter_names(
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
) -> tuple[str, ...]:
    """Return the exact ordered residual parameter names used by the optimizer."""

    residuals = motion_compliance_residual_parameters(policy, value_model)
    residual_ids = {id(parameter) for parameter in residuals}
    names = tuple(
        [
            f"policy.{name}"
            for name, parameter in policy.named_parameters()
            if id(parameter) in residual_ids
        ]
        + [
            f"value.{name}"
            for name, parameter in value_model.named_parameters()
            if id(parameter) in residual_ids
        ]
    )
    if len(names) != len(residuals):
        raise RuntimeError("could not resolve every residual parameter name")
    return names


def _ordered_residual_named_parameters(
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    residuals = motion_compliance_residual_parameters(policy, value_model)
    residual_ids = {id(parameter) for parameter in residuals}
    named = tuple(
        [(f"policy.{name}", parameter) for name, parameter in policy.named_parameters()]
        + [(f"value.{name}", parameter) for name, parameter in value_model.named_parameters()]
    )
    selected = tuple(item for item in named if id(item[1]) in residual_ids)
    weights = tuple(item for item in selected if item[0].endswith(".weight"))
    biases = tuple(item for item in selected if item[0].endswith(".bias"))
    if len(weights) != 6 or len(biases) != 6:
        raise RuntimeError("residual optimizer schema must contain six weights and six biases")
    return weights + biases


def validate_optimizer_parameter_order(
    optimizer: torch.optim.Optimizer,
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
) -> tuple[str, ...]:
    """Match the two HF decay/no-decay groups and their exact residual order."""

    ordered = _ordered_residual_named_parameters(policy, value_model)
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("residual optimizer must contain two decay/no-decay groups")
    expected_groups = (ordered[:6], ordered[6:])
    for index, (actual_group, expected_group) in enumerate(
        zip(optimizer.param_groups, expected_groups, strict=True)
    ):
        actual_parameters = tuple(actual_group["params"])
        expected_parameters = tuple(parameter for _, parameter in expected_group)
        if len(actual_parameters) != len(expected_parameters) or any(
            actual is not expected
            for actual, expected in zip(actual_parameters, expected_parameters, strict=True)
        ):
            raise RuntimeError(f"residual optimizer parameter order differs in group {index}")
    return tuple(name for name, _ in ordered)


def validate_loaded_optimizer_slots(
    optimizer: torch.optim.Optimizer,
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
) -> None:
    """Validate all twelve loaded Adam slots against live residual parameters."""

    validate_optimizer_parameter_order(optimizer, policy, value_model)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            slot = optimizer.state.get(parameter)
            if not isinstance(slot, Mapping):
                raise RuntimeError("loaded optimizer lacks a residual parameter slot")
            step = slot.get("step")
            if (
                not isinstance(step, torch.Tensor)
                or step.numel() != 1
                or not torch.isfinite(step).all()
                or float(step.item()) <= 0.0
            ):
                raise RuntimeError("loaded optimizer residual step is invalid")
            for key in ("exp_avg", "exp_avg_sq"):
                moment = slot.get(key)
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or not torch.isfinite(moment).all()
                ):
                    raise RuntimeError(f"loaded optimizer residual {key} is invalid")


def preflight_optimizer_resume_state(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, object],
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
) -> None:
    """Validate saved group order and slot shapes before mutating the optimizer."""

    validate_optimizer_parameter_order(optimizer, policy, value_model)
    current = optimizer.state_dict()
    saved_groups = optimizer_state["param_groups"]
    if len(saved_groups) != len(current["param_groups"]):
        raise ValueError("resume optimizer group count differs from the live optimizer")
    parameter_ids: list[object] = []
    for index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current["param_groups"], strict=True)
    ):
        if set(saved_group) != set(current_group):
            raise ValueError(f"resume optimizer group {index} keys differ")
        if len(saved_group["params"]) != 6:
            raise ValueError(f"resume optimizer group {index} must contain six tensors")
        if saved_group["params"] != current_group["params"]:
            raise ValueError(f"resume optimizer group {index} parameter order differs")
        validate_optimizer_parameter_group_hyperparameters(
            saved_group,
            group_index=index,
        )
        for key in saved_group:
            if key != "params":
                _preflight_nested_structure(
                    current_group[key],
                    saved_group[key],
                    f"optimizer.param_groups[{index}].{key}",
                )
                if (
                    key not in {"lr", "initial_lr"}
                    and saved_group[key] != current_group[key]
                ):
                    raise ValueError(
                        f"resume optimizer group {index} changed fixed hyperparameter {key}"
                    )
        parameter_ids.extend(saved_group["params"])

    ordered = _ordered_residual_named_parameters(policy, value_model)
    slots = optimizer_state["state"]
    for parameter_id, (name, parameter) in zip(parameter_ids, ordered, strict=True):
        slot = slots[parameter_id]
        step = slot["step"]
        if step.numel() != 1:
            raise ValueError(f"resume optimizer step for {name} must be scalar")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = slot[key]
            if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                raise ValueError(
                    f"resume optimizer {key} for {name} shape/dtype differs"
                )


def _preflight_nested_structure(actual, expected, name: str) -> None:
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.isfinite(expected).all()
        ):
            raise ValueError(f"resume {name} tensor schema differs")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"resume {name} mapping keys differ")
        for key in expected:
            _preflight_nested_structure(actual[key], expected[key], f"{name}.{key}")
        return
    if isinstance(expected, deque):
        if not isinstance(actual, deque) or actual.maxlen != expected.maxlen:
            raise ValueError(f"resume {name} deque schema differs")
        _require_nested_finite(expected, name)
        return
    if isinstance(expected, list | tuple):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise ValueError(f"resume {name} sequence schema differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _preflight_nested_structure(actual_item, expected_item, f"{name}[{index}]")
        return
    if type(actual) is not type(expected):
        raise ValueError(f"resume {name} scalar type differs")
    if isinstance(expected, float) and not math.isfinite(expected):
        raise ValueError(f"resume {name} scalar must be finite")


def _require_nested_finite(value, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"resume {name} contains NaN or Inf")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_nested_finite(item, f"{name}.{key}")
        return
    if isinstance(value, list | tuple | deque):
        for index, item in enumerate(value):
            _require_nested_finite(item, f"{name}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"resume {name} contains NaN or Inf")


def _preflight_trainer_state(trainer, incoming_state) -> None:
    live_values = getattr(trainer.state, "__dict__", None)
    incoming_values = getattr(incoming_state, "__dict__", None)
    if not isinstance(live_values, dict) or not isinstance(incoming_values, dict):
        raise ValueError("strict resume trainer state objects are malformed")
    expected_live_keys = TRAINER_STATE_SAVED_KEYS | {"log_history"}
    if set(live_values) != expected_live_keys:
        raise ValueError(
            "live trainer state schema differs from the Phase-4 checkpoint contract"
        )
    if set(incoming_values) != TRAINER_STATE_SAVED_KEYS:
        raise ValueError("resume trainer state keys differ from the saved-state contract")

    dynamic_numeric = {"epoch", "num_train_epochs", "tot_time"}
    restored_keys = TRAINER_STATE_SAVED_KEYS - TRAINER_STATE_NOT_RESTORED_KEYS
    for key in restored_keys:
        incoming_value = incoming_values[key]
        if key in dynamic_numeric:
            _require_nested_finite(incoming_value, f"trainer_state.{key}")
        else:
            _preflight_nested_structure(
                live_values[key],
                incoming_value,
                f"trainer_state.{key}",
            )
    for key in ("cur_reward_sum", "cur_episode_length"):
        if live_values[key] is not getattr(trainer, key, None):
            raise ValueError(f"live trainer state.{key} is not its trainer tensor")


def preflight_trainer_resume_boundary(
    trainer,
    payload,
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
) -> None:
    """Preflight every non-model resume boundary before any live-state load."""

    preflight_optimizer_resume_state(
        trainer.optimizer,
        payload.optimizer_state_dict,
        policy,
        value_model,
    )
    _preflight_nested_structure(
        trainer.lr_scheduler.state_dict(),
        payload.lr_scheduler_state_dict,
        "scheduler",
    )
    get_env_state = getattr(trainer.env, "get_env_state_dict", None)
    if not callable(get_env_state):
        raise ValueError("strict resume environment lacks get_env_state_dict")
    live_env_state = get_env_state()
    _preflight_nested_structure(
        live_env_state,
        payload.env_state_dict,
        "environment",
    )
    _preflight_trainer_state(trainer, payload.state)


def _assert_nested_state_equal(actual, expected, name: str) -> None:
    """Recursively require exact checkpoint-boundary state after loading."""

    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.equal(actual.detach().cpu(), expected.detach().cpu())
        ):
            raise RuntimeError(f"loaded {name} tensor differs from checkpoint")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"loaded {name} mapping keys differ from checkpoint")
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key], f"{name}.{key}")
        return
    if isinstance(expected, deque):
        if (
            not isinstance(actual, deque)
            or actual.maxlen != expected.maxlen
            or len(actual) != len(expected)
        ):
            raise RuntimeError(f"loaded {name} deque differs from checkpoint")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_state_equal(actual_item, expected_item, f"{name}[{index}]")
        return
    if isinstance(expected, list | tuple):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise RuntimeError(f"loaded {name} sequence differs from checkpoint")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_state_equal(actual_item, expected_item, f"{name}[{index}]")
        return
    if actual != expected:
        raise RuntimeError(f"loaded {name} differs from checkpoint")


def validate_residual_gradients(
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
    *,
    require_nonzero: bool = False,
) -> tuple[str, ...]:
    """Require every residual tensor to receive a finite PPO gradient."""

    parameters = motion_compliance_residual_parameters(policy, value_model)
    names = residual_parameter_names(policy, value_model)
    missing: list[str] = []
    nonfinite: list[str] = []
    zero: list[str] = []
    for name, parameter in zip(names, parameters, strict=True):
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
        elif not torch.isfinite(gradient).all():
            nonfinite.append(name)
        elif require_nonzero and torch.count_nonzero(gradient).item() == 0:
            zero.append(name)
    if missing or nonfinite or zero:
        raise RuntimeError(
            "residual gradient contract failed: "
            f"missing={missing}, nonfinite={nonfinite}, zero={zero}"
        )
    return names


class MotionCompliancePPOTrainer(TRLAuxLossPPOTrainer):
    """Keep strict residual-only behavior isolated from SONIC's generic trainer."""

    _tag_names = ["trl", "aux_loss_ppo", "motion_compliance"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        validate_optimizer_parameter_set(
            self.optimizer,
            self.policy_model,
            self.value_model,
        )
        validate_optimizer_parameter_order(
            self.optimizer,
            self.policy_model,
            self.value_model,
        )
        if self.num_steps_per_env != PHASE4_PPO_ROLLOUT_STEPS:
            raise ValueError(
                "motion-compliance trainer requires "
                f"num_steps_per_env={PHASE4_PPO_ROLLOUT_STEPS}"
            )
        self._motion_compliance_gradient_seen_names: set[str] = set()

    def _forward_model(self, model, mb_rollout_data):
        batch_size, rollout_steps = validate_motion_compliance_ppo_batch(
            mb_rollout_data,
            rollout_steps=self.num_steps_per_env,
        )
        results = super()._forward_model(model, mb_rollout_data)
        validate_motion_compliance_ppo_outputs(
            results,
            batch_size=batch_size,
            rollout_steps=rollout_steps,
        )
        return results

    def _gradient_clipping(self):
        names = validate_residual_gradients(
            self.policy_model,
            self.value_model,
            require_nonzero=False,
        )
        self._motion_compliance_gradient_seen_names.update(names)
        return super()._gradient_clipping()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Report the effective clamped noise while preserving frozen raw state."""

        if "Policy/mean_noise_std" in logs:
            model = self.accelerator.unwrap_model(self.model)
            with torch.no_grad():
                logs["Policy/mean_noise_std"] = float(model.policy.get_std.mean().item())
        super().log(logs, start_time=start_time)

    def load_checkpoint(self, checkpoint_path, resume=False):  # noqa: D417
        """Strict-load residual initialization or a complete branch checkpoint."""

        print(f"Loading motion-compliance checkpoint from {checkpoint_path}")  # noqa: T201
        checkpoint = load_trl_checkpoint(
            checkpoint_path,
            map_location=self.accelerator.device,
        )

        payload = validate_strict_resume_payload(checkpoint) if resume else None

        model = self.accelerator.unwrap_model(self.model)
        if model.value_model is None:
            raise ValueError("motion-compliance trainer requires a critic value model")
        if payload is not None:
            preflight_trainer_resume_boundary(
                self,
                payload,
                model.policy,
                model.value_model,
            )
        load_report = strict_load_policy_value_state(
            model.policy,
            model.value_model,
            checkpoint,
            resume=resume,
        )

        if resume:
            assert payload is not None
            self.optimizer.load_state_dict(payload.optimizer_state_dict)
            self.lr_scheduler.load_state_dict(payload.lr_scheduler_state_dict)
            self.env.load_env_state_dict(payload.env_state_dict)

            if "args" in checkpoint and hasattr(checkpoint["args"], "learning_rate"):
                self.args.learning_rate = checkpoint["args"].learning_rate

            _assert_nested_state_equal(
                self.optimizer.state_dict(),
                payload.optimizer_state_dict,
                "optimizer",
            )
            _assert_nested_state_equal(
                self.lr_scheduler.state_dict(),
                payload.lr_scheduler_state_dict,
                "scheduler",
            )
            _assert_nested_state_equal(
                self.env.get_env_state_dict(),
                payload.env_state_dict,
                "environment",
            )
            validate_optimizer_parameter_set(
                self.optimizer,
                model.policy,
                model.value_model,
            )
            validate_loaded_optimizer_slots(
                self.optimizer,
                model.policy,
                model.value_model,
            )

            restored_state_keys = (
                TRAINER_STATE_SAVED_KEYS - TRAINER_STATE_NOT_RESTORED_KEYS
            )
            for key in restored_state_keys:
                value = payload.state.__dict__[key]
                if key in {"cur_reward_sum", "cur_episode_length"}:
                    current_value = getattr(self, key, None)
                    if current_value is None or current_value.shape != value.shape:
                        raise ValueError(
                            f"strict resume trainer tensor shape differs for {key}: "
                            f"current={getattr(current_value, 'shape', None)}, "
                            f"checkpoint={value.shape}"
                        )
                    setattr(self, key, value)
                setattr(self.state, key, value)
            for key in restored_state_keys:
                _assert_nested_state_equal(
                    getattr(self.state, key),
                    getattr(payload.state, key),
                    f"trainer_state.{key}",
                )
            if (
                self.state.cur_reward_sum is not self.cur_reward_sum
                or self.state.cur_episode_length is not self.cur_episode_length
            ):
                raise RuntimeError("loaded trainer state tensors are not trainer attributes")
            _assert_nested_state_equal(
                model.policy.state_dict(),
                checkpoint[load_report.policy_key],
                "policy",
            )
            _assert_nested_state_equal(
                model.value_model.state_dict(),
                checkpoint["value_state_dict"],
                "value",
            )

        loaded_step = getattr(checkpoint.get("state"), "global_step", None)
        print(  # noqa: T201
            "Loaded motion-compliance checkpoint "
            f"from step {loaded_step}: policy_key={load_report.policy_key}, "
            f"strict={load_report.strict}, residual_init={load_report.residual_init}"
        )
        return checkpoint
