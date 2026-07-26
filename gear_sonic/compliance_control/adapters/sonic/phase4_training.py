"""Strict SONIC residual-finetune audits and a thin trainer callback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer

from ...training import (
    assert_nested_exact,
    assert_state_dict_exact,
    atomic_write_json,
    directory_usage_bytes,
    finite_loss_metrics,
    incremental_batch_count,
    optimizer_parameter_count,
    state_dict_digest,
)


OFFICIAL_SONIC_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
ACTOR_RESIDUAL_PREFIX = "actor_module.compliance_residual."
CRITIC_RESIDUAL_PREFIX = "compliance_value_residual."
TRAINER_ACTOR_RESIDUAL_PREFIX = f"policy.{ACTOR_RESIDUAL_PREFIX}"
TRAINER_CRITIC_RESIDUAL_PREFIX = f"value_model.{CRITIC_RESIDUAL_PREFIX}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sonic_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted local SONIC checkpoint with its historical TRL symbols."""

    from trl.experimental.ppo import ppo_trainer
    import trl.trainer.utils

    trl.trainer.utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
    trl.trainer.utils.exact_div = ppo_trainer.exact_div
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _checkpoint_step(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint.get("state")
    if isinstance(state, Mapping):
        step = state.get("global_step")
    else:
        step = getattr(state, "global_step", None)
    if type(step) is not int or step < 0:
        raise AssertionError("checkpoint has no non-negative integer global_step")
    return step


def _residual_state(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in state_dict.items() if name.startswith(prefix)}


def _all_finite(state_dict: Mapping[str, torch.Tensor], *, label: str) -> None:
    for name, tensor in state_dict.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{label} contains a non-finite tensor: {name}")


def _assert_optimizer_resume_payload(
    checkpoint: Mapping[str, Any], *, expected_parameter_count: int
) -> None:
    optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer, Mapping):
        raise AssertionError("checkpoint optimizer state is missing")
    if optimizer_parameter_count(optimizer) != expected_parameter_count:
        raise AssertionError("checkpoint optimizer parameter count mismatch")
    slots = optimizer.get("state")
    if not isinstance(slots, Mapping) or len(slots) != expected_parameter_count:
        raise AssertionError("every residual parameter must have an optimizer slot")
    for parameter_id, slot in slots.items():
        if not isinstance(slot, Mapping) or not slot:
            raise AssertionError(f"optimizer slot is empty: {parameter_id}")
        for name, value in slot.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise AssertionError(
                    f"optimizer slot contains a non-finite tensor: {parameter_id}.{name}"
                )
    scheduler = checkpoint.get("lr_scheduler_state_dict")
    if not isinstance(scheduler, Mapping) or not scheduler:
        raise AssertionError("checkpoint scheduler state is missing or empty")
    if not isinstance(checkpoint.get("env_state_dict"), Mapping):
        raise AssertionError("checkpoint environment payload is missing")


def _residual_changed(
    before: Mapping[str, torch.Tensor], current: Mapping[str, torch.Tensor]
) -> bool:
    return state_dict_digest(before) != state_dict_digest(current)


def _head_is_nonzero(state_dict: Mapping[str, torch.Tensor]) -> bool:
    head = [
        tensor
        for name, tensor in state_dict.items()
        if name.endswith("output_layer.weight") or name.endswith("output_layer.bias")
    ]
    return bool(head) and any(torch.count_nonzero(tensor).item() > 0 for tensor in head)


def audit_sonic_phase4_checkpoint(
    *,
    checkpoint_path: str | Path,
    official_checkpoint_path: str | Path,
    audit_report_path: str | Path,
    expected_step: int,
    source_branch_checkpoint_path: str | Path | None = None,
    expected_optimizer_parameter_count: int = 12,
    expected_trainable_scalar_count: int = 770_753,
    max_run_bytes: int = 1_200_000_000,
    max_log_bytes: int = 64_000_000,
) -> dict[str, Any]:
    """Audit a saved step-5/step-6 checkpoint independently of the trainer."""

    checkpoint_path = Path(checkpoint_path).resolve()
    official_checkpoint_path = Path(official_checkpoint_path).resolve()
    audit_report_path = Path(audit_report_path).resolve()
    if _file_sha256(official_checkpoint_path) != OFFICIAL_SONIC_SHA256:
        raise AssertionError("official SONIC checkpoint SHA-256 mismatch")

    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise AssertionError("training audit report is not complete")
    if report.get("final_step") != expected_step:
        raise AssertionError("training audit report step mismatch")
    if report.get("optimizer_parameter_count") != expected_optimizer_parameter_count:
        raise AssertionError("training audit optimizer ownership mismatch")
    if report.get("trainable_scalar_count") != expected_trainable_scalar_count:
        raise AssertionError("training audit residual scalar count mismatch")
    if report.get("actor_residual_changed") is not True:
        raise AssertionError("training audit actor residual did not change")
    if report.get("critic_residual_changed") is not True:
        raise AssertionError("training audit critic residual did not change")
    losses = report.get("losses")
    if not isinstance(losses, list) or not losses:
        raise AssertionError("training audit contains no finite loss records")
    expected_start_step = expected_step - 1 if source_branch_checkpoint_path else 0
    expected_loss_steps = list(range(expected_start_step + 1, expected_step + 1))
    if [entry.get("step") for entry in losses] != expected_loss_steps:
        raise AssertionError("training audit loss-step sequence mismatch")
    for entry in losses:
        finite_loss_metrics(entry.get("values", {}))
    site_names = report.get("site_names")
    site_exposure_counts = report.get("site_exposure_counts")
    if (
        not isinstance(site_names, list)
        or not site_names
        or not isinstance(site_exposure_counts, list)
        or len(site_exposure_counts) != len(site_names)
    ):
        raise AssertionError("training audit site exposure schema mismatch")
    if any(count <= 0 for count in site_exposure_counts):
        raise AssertionError("at least one configured site has no true exposure")
    if report.get("peak_cuda_memory_bytes", 0) <= 0:
        raise AssertionError("training audit did not record peak CUDA memory")
    gradient_stats = report.get("gradient_stats")
    if not isinstance(gradient_stats, Mapping) or len(gradient_stats) != 12:
        raise AssertionError("training audit must contain all 12 residual gradients")
    for name, values in gradient_stats.items():
        if not isinstance(values, Mapping):
            raise AssertionError(f"invalid gradient audit entry: {name}")
        if values.get("seen_backward_count", 0) <= 0:
            raise AssertionError(f"residual gradient was never observed: {name}")
        if values.get("nonzero_backward_count", 0) <= 0:
            raise AssertionError(f"residual gradient was always zero: {name}")
        maximum = values.get("max_abs_gradient")
        if not isinstance(maximum, int | float) or not math.isfinite(float(maximum)):
            raise AssertionError(f"invalid residual gradient maximum: {name}")
    trainable_names = report.get("trainable_parameter_names")
    if not isinstance(trainable_names, list) or set(trainable_names) != set(gradient_stats):
        raise AssertionError("training audit gradient/trainable name mismatch")
    actor_gradient_names = [
        name for name in trainable_names if name.startswith(TRAINER_ACTOR_RESIDUAL_PREFIX)
    ]
    critic_gradient_names = [
        name for name in trainable_names if name.startswith(TRAINER_CRITIC_RESIDUAL_PREFIX)
    ]
    if len(actor_gradient_names) != 6 or len(critic_gradient_names) != 6:
        raise AssertionError("training audit names must be six actor plus six critic tensors")

    expected_mode = "branch_resume" if source_branch_checkpoint_path else "official_init"
    expected_source_step = expected_step - 1 if source_branch_checkpoint_path else 41_550
    if report.get("audit_mode") != expected_mode:
        raise AssertionError("training audit mode mismatch")
    if report.get("source_checkpoint_step") != expected_source_step:
        raise AssertionError("training audit source step mismatch")
    if report.get("start_step") != expected_start_step:
        raise AssertionError("training audit start step mismatch")
    if Path(report.get("checkpoint", "")).resolve() != checkpoint_path:
        raise AssertionError("training audit checkpoint path mismatch")

    official = load_sonic_checkpoint(official_checkpoint_path)
    trained = load_sonic_checkpoint(checkpoint_path)
    if _checkpoint_step(trained) != expected_step:
        raise AssertionError("saved checkpoint step mismatch")
    actor_state = trained["policy_state_dict"]
    critic_state = trained["value_state_dict"]
    assert_state_dict_exact(
        official["policy_state_dict"],
        actor_state,
        allow_additional_current=True,
        label="trained policy legacy state",
    )
    assert_state_dict_exact(
        official["value_state_dict"],
        critic_state,
        allow_additional_current=True,
        label="trained value legacy state",
    )
    actor_residual = _residual_state(actor_state, ACTOR_RESIDUAL_PREFIX)
    critic_residual = _residual_state(critic_state, CRITIC_RESIDUAL_PREFIX)
    if len(actor_residual) != 6 or len(critic_residual) != 6:
        raise AssertionError("saved checkpoint must contain six residual keys per model")
    actual_trainable_scalar_count = sum(
        tensor.numel() for tensor in (*actor_residual.values(), *critic_residual.values())
    )
    if actual_trainable_scalar_count != expected_trainable_scalar_count:
        raise AssertionError("saved residual scalar count does not match the smoke contract")
    _all_finite(actor_residual, label="actor residual")
    _all_finite(critic_residual, label="critic residual")
    if not _head_is_nonzero(actor_residual) or not _head_is_nonzero(critic_residual):
        raise AssertionError("both saved residual output heads must have trained")
    if set(actor_state) - set(official["policy_state_dict"]) != set(actor_residual):
        raise AssertionError("trained policy contains state outside the residual branch")
    if set(critic_state) - set(official["value_state_dict"]) != set(critic_residual):
        raise AssertionError("trained value contains state outside the residual branch")
    _assert_optimizer_resume_payload(
        trained,
        expected_parameter_count=expected_optimizer_parameter_count,
    )

    if source_branch_checkpoint_path is not None:
        source = load_sonic_checkpoint(Path(source_branch_checkpoint_path).resolve())
        if _checkpoint_step(source) != expected_step - 1:
            raise AssertionError("resume source checkpoint step mismatch")
        _assert_optimizer_resume_payload(
            source,
            expected_parameter_count=expected_optimizer_parameter_count,
        )
        source_actor = _residual_state(
            source["policy_state_dict"], ACTOR_RESIDUAL_PREFIX
        )
        source_critic = _residual_state(
            source["value_state_dict"], CRITIC_RESIDUAL_PREFIX
        )
        if not _residual_changed(source_actor, actor_residual):
            raise AssertionError("resumed actor residual did not change after step 5")
        if not _residual_changed(source_critic, critic_residual):
            raise AssertionError("resumed critic residual did not change after step 5")
        try:
            assert_nested_exact(
                source["optimizer_state_dict"],
                trained["optimizer_state_dict"],
                label="resumed optimizer advancement",
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("resumed optimizer state did not advance")

    total_bytes, largest_log = directory_usage_bytes(checkpoint_path.parent)
    if total_bytes > max_run_bytes:
        raise AssertionError(f"run directory is too large: {total_bytes} bytes")
    if largest_log > max_log_bytes:
        raise AssertionError(f"run log is too large: {largest_log} bytes")
    result = {
        "checkpoint": str(checkpoint_path),
        "step": expected_step,
        "legacy_policy_tensors": len(official["policy_state_dict"]),
        "legacy_value_tensors": len(official["value_state_dict"]),
        "actor_residual_tensors": len(actor_residual),
        "critic_residual_tensors": len(critic_residual),
        "optimizer_parameter_count": expected_optimizer_parameter_count,
        "trainable_scalar_count": actual_trainable_scalar_count,
        "run_bytes": total_bytes,
        "largest_log_bytes": largest_log,
        "site_names": site_names,
        "site_exposure_counts": site_exposure_counts,
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "finite_loss_steps": expected_loss_steps,
    }
    del official, trained
    gc.collect()
    return result


class SonicComplianceResidualPPOTrainer(TRLAuxLossPPOTrainer):
    """Phase-4-only PPO guard for exact residual optimizer/gradient ownership."""

    _EXPECTED_TRAINABLE_PREFIXES = (
        TRAINER_ACTOR_RESIDUAL_PREFIX,
        TRAINER_CRITIC_RESIDUAL_PREFIX,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if len(trainable) != 12 or not all(
            name.startswith(self._EXPECTED_TRAINABLE_PREFIXES) for name in trainable
        ):
            raise AssertionError("Phase-4 trainer requires exactly 12 residual tensors")
        expected_ids = {id(parameter) for parameter in trainable.values()}
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != expected_ids:
            raise AssertionError("Phase-4 optimizer ownership is not residual-only")
        model._chip_phase4_gradient_stats = {  # noqa: SLF001
            name: {
                "seen_backward_count": 0,
                "nonzero_backward_count": 0,
                "max_abs_gradient": 0.0,
            }
            for name in sorted(trainable)
        }

    def load_checkpoint(self, checkpoint_path, resume=False):  # noqa: D417
        """Restore the serialized optimizer boundary for branch resumes.

        The generic trainer first loads the optimizer, then reconciles every
        parameter group's learning rate from ``checkpoint["args"]``.  SONIC's
        adaptive-KL value can legitimately differ from the optimizer learning
        rate serialized after the scheduler step.  Reloading only the same
        optimizer payload preserves both values without changing the generic
        trainer or the already-restored scheduler/environment/trainer state.
        """

        checkpoint = super().load_checkpoint(checkpoint_path, resume=resume)
        if resume:
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if not isinstance(optimizer_state, Mapping):
                raise AssertionError(
                    "Phase-4 branch resume requires an optimizer state payload"
                )
            self.optimizer.load_state_dict(optimizer_state)
        return checkpoint

    def _gradient_clipping(self):
        model = self.accelerator.unwrap_model(self.model)
        stats = model._chip_phase4_gradient_stats  # noqa: SLF001
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if not parameter.requires_grad:
                if gradient is not None:
                    raise AssertionError(f"frozen parameter received a gradient: {name}")
                continue
            if name not in stats:
                raise AssertionError(f"unexpected trainable parameter: {name}")
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise AssertionError(f"residual gradient is non-finite: {name}")
            maximum = float(gradient.detach().abs().amax().cpu())
            stats[name]["seen_backward_count"] += 1
            stats[name]["max_abs_gradient"] = max(
                stats[name]["max_abs_gradient"], maximum
            )
            if maximum > 0.0:
                stats[name]["nonzero_backward_count"] += 1
        return super()._gradient_clipping()


class SonicPhase4TrainingAuditCallback(TrainerCallback):
    """Fail closed if residual-only PPO mutates any released SONIC state."""

    def __init__(
        self,
        *,
        report_path: str,
        source_checkpoint: str,
        audit_mode: str,
        expected_source_checkpoint_step: int,
        expected_start_step: int,
        expected_final_step: int,
        expected_site_names: Sequence[str],
        expected_num_envs: int = 16,
        expected_trainable_scalar_count: int | None = None,
        expected_pulse_interval_range_s: Sequence[float] = (0.02, 0.04),
        max_run_bytes: int = 1_200_000_000,
        max_log_bytes: int = 64_000_000,
    ) -> None:
        if audit_mode not in {"official_init", "branch_resume"}:
            raise ValueError("audit_mode must be 'official_init' or 'branch_resume'")
        if isinstance(expected_site_names, str | bytes) or not expected_site_names:
            raise ValueError("expected_site_names must be a non-empty sequence")
        self.report_path = Path(report_path)
        if self.report_path.suffix != ".json":
            raise ValueError("Phase-4 report_path must use a .json suffix")
        self.source_checkpoint = Path(source_checkpoint)
        self.audit_mode = audit_mode
        self.expected_source_checkpoint_step = expected_source_checkpoint_step
        self.expected_start_step = expected_start_step
        self.expected_final_step = expected_final_step
        self.expected_site_names = tuple(expected_site_names)
        self.expected_num_envs = expected_num_envs
        self.expected_trainable_scalar_count = expected_trainable_scalar_count
        self.expected_pulse_interval_range_s = tuple(
            float(value) for value in expected_pulse_interval_range_s
        )
        if type(expected_num_envs) is not int or expected_num_envs <= 0:
            raise ValueError("expected_num_envs must be a positive integer")
        if expected_trainable_scalar_count is not None and (
            type(expected_trainable_scalar_count) is not int
            or expected_trainable_scalar_count <= 0
        ):
            raise ValueError("expected_trainable_scalar_count must be positive or None")
        if len(self.expected_pulse_interval_range_s) != 2:
            raise ValueError("expected_pulse_interval_range_s must contain two values")
        self.max_run_bytes = max_run_bytes
        self.max_log_bytes = max_log_bytes
        self._policy_legacy_digest: str | None = None
        self._value_legacy_digest: str | None = None
        self._actor_residual_start: dict[str, torch.Tensor] = {}
        self._critic_residual_start: dict[str, torch.Tensor] = {}
        self._losses: list[dict[str, Any]] = []
        self._site_exposure_counts = [0 for _ in self.expected_site_names]
        self._optimizer_parameter_count = 0
        self._trainable_scalar_count = 0
        self._peak_cuda_memory_bytes = 0
        self._start_validated = False

    @staticmethod
    def _models(model, accelerator):
        unwrapped = accelerator.unwrap_model(model)
        return unwrapped.policy, unwrapped.value_model

    @staticmethod
    def _state_cpu(
        state_dict: Mapping[str, torch.Tensor], prefix: str
    ) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in state_dict.items()
            if name.startswith(prefix)
        }

    def _assert_release_frozen(self, policy, value_model) -> None:
        policy_digest = state_dict_digest(
            policy.state_dict(), excluded_prefixes=(ACTOR_RESIDUAL_PREFIX,)
        )
        value_digest = state_dict_digest(
            value_model.state_dict(), excluded_prefixes=(CRITIC_RESIDUAL_PREFIX,)
        )
        if policy_digest != self._policy_legacy_digest:
            raise AssertionError("released policy/std state changed during residual finetune")
        if value_digest != self._value_legacy_digest:
            raise AssertionError("released value/RMS state changed during residual finetune")

    def _report(self, *, state, policy, value_model, complete: bool) -> None:
        actor_changed = _residual_changed(
            self._actor_residual_start,
            _residual_state(policy.state_dict(), ACTOR_RESIDUAL_PREFIX),
        )
        critic_changed = _residual_changed(
            self._critic_residual_start,
            _residual_state(value_model.state_dict(), CRITIC_RESIDUAL_PREFIX),
        )
        run_root = self.report_path.parent
        total_bytes, largest_log = directory_usage_bytes(run_root)
        wrapper = getattr(self, "_model_wrapper", None)
        gradient_stats = (
            getattr(wrapper, "_chip_phase4_gradient_stats", {})
            if wrapper is not None
            else {}
        )
        payload = {
            "schema_version": 1,
            "complete": complete,
            "audit_mode": self.audit_mode,
            "source_checkpoint": str(self.source_checkpoint),
            "source_checkpoint_step": self.expected_source_checkpoint_step,
            "start_step": self.expected_start_step,
            "final_step": state.global_step,
            "trainable_parameter_names": self._trainable_parameter_names,
            "optimizer_parameter_count": self._optimizer_parameter_count,
            "trainable_scalar_count": self._trainable_scalar_count,
            "policy_legacy_sha256": self._policy_legacy_digest,
            "value_legacy_sha256": self._value_legacy_digest,
            "actor_residual_changed": actor_changed,
            "critic_residual_changed": critic_changed,
            "site_names": list(self.expected_site_names),
            "site_exposure_counts": self._site_exposure_counts,
            "losses": self._losses,
            "gradient_stats": gradient_stats,
            "peak_cuda_memory_bytes": self._peak_cuda_memory_bytes,
            "run_bytes_before_report": total_bytes,
            "largest_log_bytes": largest_log,
            "checkpoint": str(run_root / "last.pt"),
        }
        atomic_write_json(self.report_path, payload)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        lr_scheduler = kwargs["lr_scheduler"]
        accelerator = kwargs["accelerator"]
        env = kwargs["env"]
        if self.report_path.resolve().parent != Path(
            env.config.experiment_dir
        ).resolve():
            raise AssertionError("Phase-4 audit must stay in the experiment directory")
        policy, value_model = self._models(model, accelerator)
        self._model_wrapper = accelerator.unwrap_model(model)
        if state.global_step != self.expected_start_step:
            raise AssertionError(
                f"trainer start step {state.global_step} != {self.expected_start_step}"
            )
        expected_batches = incremental_batch_count(
            self.expected_start_step,
            self.expected_final_step,
        )
        if state.max_steps != expected_batches:
            raise AssertionError(
                f"trainer batch count {state.max_steps} != {expected_batches}"
            )
        if env.num_envs != self.expected_num_envs:
            raise AssertionError(
                f"Phase-4 smoke requires {self.expected_num_envs} environments"
            )
        if policy.use_log_std:
            raise AssertionError("Phase-4 SONIC residual finetune requires direct std")
        if not policy.algo_config.get("use_clampped_std", False):
            raise AssertionError("Phase-4 SONIC residual finetune requires release std clamp")
        if not math.isclose(float(policy.algo_config.std_clamp_min), 0.001):
            raise AssertionError("release std_clamp_min changed")
        if not math.isclose(float(policy.algo_config.std_clamp_max), 0.5):
            raise AssertionError("release std_clamp_max changed")
        if policy.clamp_noise_std:
            raise AssertionError("release clamp_noise_std semantics changed")
        if policy.std.requires_grad or policy.std.grad is not None:
            raise AssertionError("official action std must be exactly frozen")
        if value_model.running_mean_std is None or not value_model.running_mean_std.frozen:
            raise AssertionError("official critic running statistics must be frozen")

        actor_trainable = {
            name: parameter
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        critic_trainable = {
            name: parameter
            for name, parameter in value_model.named_parameters()
            if parameter.requires_grad
        }
        if len(actor_trainable) != 6 or not all(
            name.startswith(ACTOR_RESIDUAL_PREFIX) for name in actor_trainable
        ):
            raise AssertionError("actor must expose exactly six residual parameters")
        if len(critic_trainable) != 6 or not all(
            name.startswith(CRITIC_RESIDUAL_PREFIX) for name in critic_trainable
        ):
            raise AssertionError("critic must expose exactly six residual parameters")
        expected_parameters = {
            id(parameter) for parameter in (*actor_trainable.values(), *critic_trainable.values())
        }
        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameters != expected_parameters:
            raise AssertionError("optimizer owns parameters outside the 12 residual tensors")
        self._optimizer_parameter_count = len(optimizer_parameters)
        self._trainable_scalar_count = sum(
            parameter.numel()
            for parameter in (*actor_trainable.values(), *critic_trainable.values())
        )
        if (
            self.expected_trainable_scalar_count is not None
            and self._trainable_scalar_count != self.expected_trainable_scalar_count
        ):
            raise AssertionError(
                "Phase-4 residual scalar count does not match the configured site layout"
            )
        self._trainable_parameter_names = sorted(
            [f"policy.{name}" for name in actor_trainable]
            + [f"value_model.{name}" for name in critic_trainable]
        )

        source = load_sonic_checkpoint(self.source_checkpoint)
        if _checkpoint_step(source) != self.expected_source_checkpoint_step:
            raise AssertionError("source checkpoint step mismatch")
        source_policy = source["policy_state_dict"]
        source_value = source["value_state_dict"]
        if self.audit_mode == "official_init":
            if _file_sha256(self.source_checkpoint) != OFFICIAL_SONIC_SHA256:
                raise AssertionError("official SONIC checkpoint SHA-256 mismatch")
            assert_state_dict_exact(
                source_policy,
                policy.state_dict(),
                allow_additional_current=True,
                label="official policy initialization",
            )
            assert_state_dict_exact(
                source_value,
                value_model.state_dict(),
                allow_additional_current=True,
                label="official value initialization",
            )
            if policy.last_migration_report is None or value_model.last_migration_report is None:
                raise AssertionError("official checkpoint did not use explicit migration")
        else:
            assert_state_dict_exact(
                source_policy,
                policy.state_dict(),
                label="branch policy strict resume",
            )
            assert_state_dict_exact(
                source_value,
                value_model.state_dict(),
                label="branch value strict resume",
            )
            if (
                policy.last_migration_report is not None
                or value_model.last_migration_report is not None
            ):
                raise AssertionError("branch resume must not use legacy migration")
            assert_nested_exact(
                source["optimizer_state_dict"],
                optimizer.state_dict(),
                label="branch optimizer strict resume",
            )
            assert_nested_exact(
                source["lr_scheduler_state_dict"],
                lr_scheduler.state_dict(),
                label="branch scheduler strict resume",
            )

        self._policy_legacy_digest = state_dict_digest(
            policy.state_dict(), excluded_prefixes=(ACTOR_RESIDUAL_PREFIX,)
        )
        self._value_legacy_digest = state_dict_digest(
            value_model.state_dict(), excluded_prefixes=(CRITIC_RESIDUAL_PREFIX,)
        )
        self._actor_residual_start = self._state_cpu(
            policy.state_dict(), ACTOR_RESIDUAL_PREFIX
        )
        self._critic_residual_start = self._state_cpu(
            value_model.state_dict(), CRITIC_RESIDUAL_PREFIX
        )
        command = env.force_command
        if command is None or not command.operational_enabled:
            raise AssertionError("Phase-4 compliance command must be operationally enabled")
        if tuple(command.sites.spec.site_names) != self.expected_site_names:
            raise AssertionError("runtime compliance site ordering changed")
        if not math.isclose(float(command.cfg.enabled_probability), 1.0):
            raise AssertionError("Phase-4 smoke must expose every environment")
        if not math.isclose(float(command.cfg.site_probability), 1.0):
            raise AssertionError("Phase-4 smoke must expose every configured site")
        if command.cfg.max_active_sites != len(self.expected_site_names):
            raise AssertionError("Phase-4 smoke must allow every configured site")
        if tuple(command.cfg.pulse_interval_range_s) != self.expected_pulse_interval_range_s:
            raise AssertionError("Phase-4 smoke pulse interval changed")
        if not command.cfg.compliance_values_m_per_n or any(
            value <= 0.0 for value in command.cfg.compliance_values_m_per_n
        ):
            raise AssertionError("Phase-4 smoke requires nonzero compliance sampling")
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        self._start_validated = True
        self._report(state=state, policy=policy, value_model=value_model, complete=False)
        del source
        gc.collect()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._start_validated:
            raise AssertionError("training log arrived before Phase-4 start audit")
        losses = finite_loss_metrics({} if logs is None else logs)
        self._losses.append({"step": state.global_step, "values": losses})

    def _finalize(self, *, state, policy, value_model) -> None:
        if state.global_step != self.expected_final_step:
            raise AssertionError(
                f"trainer final step {state.global_step} != {self.expected_final_step}"
            )
        if any(count <= 0 for count in self._site_exposure_counts):
            raise AssertionError("every configured compliance site must be truly exposed")
        if not _residual_changed(
            self._actor_residual_start,
            _residual_state(policy.state_dict(), ACTOR_RESIDUAL_PREFIX),
        ):
            raise AssertionError("actor residual did not update")
        if not _residual_changed(
            self._critic_residual_start,
            _residual_state(value_model.state_dict(), CRITIC_RESIDUAL_PREFIX),
        ):
            raise AssertionError("critic residual did not update")
        gradient_stats = getattr(
            self._model_wrapper, "_chip_phase4_gradient_stats", {}
        )
        if set(gradient_stats) != set(self._trainable_parameter_names):
            raise AssertionError("gradient audit schema does not match trainable ownership")
        missing_gradient = [
            name
            for name, values in gradient_stats.items()
            if values["seen_backward_count"] <= 0
            or values["nonzero_backward_count"] <= 0
        ]
        if missing_gradient:
            raise AssertionError(
                f"residual tensors lack finite nonzero gradients: {missing_gradient}"
            )
        total_bytes, largest_log = directory_usage_bytes(self.report_path.parent)
        if total_bytes > self.max_run_bytes:
            raise AssertionError(f"Phase-4 run exceeds byte budget: {total_bytes}")
        if largest_log > self.max_log_bytes:
            raise AssertionError(f"Phase-4 log exceeds byte budget: {largest_log}")
        if not (self.report_path.parent / "last.pt").is_file():
            raise AssertionError("Phase-4 model callback did not save last.pt")
        self._report(
            state=state,
            policy=policy,
            value_model=value_model,
            complete=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        policy, value_model = self._models(kwargs["model"], kwargs["accelerator"])
        self._assert_release_frozen(policy, value_model)
        command = kwargs["env"].force_command
        enabled = command.state.enabled.unsqueeze(-1)
        compliance_active = (command.state.compliance > 0.0).any(dim=-1)
        force_active = torch.linalg.vector_norm(
            command.state.force_on_robot_w, dim=-1
        ) > 0.0
        exposed = enabled & command.state.site_mask & compliance_active & force_active
        counts = exposed.sum(dim=0).detach().cpu().tolist()
        self._site_exposure_counts = [
            total + int(count) for total, count in zip(self._site_exposure_counts, counts)
        ]
        self._peak_cuda_memory_bytes = max(
            self._peak_cuda_memory_bytes,
            int(torch.cuda.max_memory_allocated(kwargs["accelerator"].device)),
        )
        if not any(entry["step"] == state.global_step for entry in self._losses):
            raise AssertionError("no finite loss record for completed training step")
        if state.global_step == self.expected_final_step:
            self._finalize(
                state=state,
                policy=policy,
                value_model=value_model,
            )
        else:
            self._report(
                state=state,
                policy=policy,
                value_model=value_model,
                complete=False,
            )

    def on_train_end(self, args, state, control, **kwargs):
        policy, value_model = self._models(kwargs["model"], kwargs["accelerator"])
        self._assert_release_frozen(policy, value_model)
        if state.global_step == self.expected_final_step:
            report = json.loads(self.report_path.read_text(encoding="utf-8"))
            if report.get("complete") is not True:
                self._finalize(
                    state=state,
                    policy=policy,
                    value_model=value_model,
                )
