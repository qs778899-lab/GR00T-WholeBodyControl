"""Selective parameter ownership for staged compliance finetuning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .actor import MOTION_COMPLIANCE_ACTOR_TARGET
from .paths import (
    OFFICIAL_SAMPLE_ROBOT_MOTION,
    OFFICIAL_SAMPLE_SMPL_MOTION,
    OFFICIAL_SONIC_RELEASE_CHECKPOINT,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
)


MOTION_COMPLIANCE_TRAINER_TARGET = (
    "gear_sonic.compliance_control.training.trainer.MotionCompliancePPOTrainer"
)
_PHASE4_SITE_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


@dataclass(frozen=True)
class FinetuneStageReport:
    """Trainable parameter names selected for one finetune stage."""

    stage: str
    trainable_policy_names: tuple[str, ...]
    frozen_policy_names: tuple[str, ...]
    trainable_value_names: tuple[str, ...]


def validate_motion_compliance_workflow_config(config) -> None:
    """Validate path ownership and branch semantics before creating artifacts."""

    finetune_cfg = config.get("motion_compliance_finetune", None)
    if finetune_cfg is None:
        return

    experiment_dir = validate_motion_compliance_run_path(config.experiment_dir)
    trainer_target = config.trainer.get("_target_", None)
    if trainer_target != MOTION_COMPLIANCE_TRAINER_TARGET:
        raise ValueError(
            "motion-compliance finetuning requires the isolated strict trainer; "
            f"got {trainer_target!r}"
        )
    actor_target = config.algo.config.actor.get("_target_", None)
    if actor_target != MOTION_COMPLIANCE_ACTOR_TARGET:
        raise ValueError(
            "motion-compliance finetuning requires the frozen-noise actor; "
            f"got {actor_target!r}"
        )

    migration_cfg = config.get("motion_compliance_checkpoint_migration", None)
    migration_enabled = bool(
        migration_cfg is not None and migration_cfg.get("enabled", False)
    )
    if config.get("resume", False):
        if migration_enabled:
            raise ValueError("strict resume requires checkpoint migration to be disabled")
        if config.get("checkpoint", None) is None:
            raise ValueError("strict resume requires a checkpoint path")
        input_checkpoint = validate_motion_compliance_run_path(config.checkpoint)
        resume_output = finetune_cfg.get("resume_output_dir", None)
        if resume_output is None:
            raise ValueError("strict resume requires a separate resume_output_dir")
        resolved_resume_output = validate_motion_compliance_run_path(resume_output)
        if resolved_resume_output != experiment_dir:
            raise ValueError(
                "strict resume requires experiment_dir to equal resume_output_dir"
            )
    else:
        if not migration_enabled:
            raise ValueError("initial compliance finetuning requires official migration")
        source = Path(config.get("checkpoint", "")).expanduser().resolve(strict=False)
        official_source = OFFICIAL_SONIC_RELEASE_CHECKPOINT.resolve(strict=False)
        if source != official_source:
            raise ValueError(
                f"initial compliance checkpoint must be {official_source}; got {source}"
            )
        validate_motion_compliance_run_path(migration_cfg.output_path)
        input_checkpoint = source

    exposure_output = validate_motion_compliance_run_path(
        config.callbacks.motion_compliance_exposure.output_path
    )
    if exposure_output.suffix != ".json":
        raise ValueError("motion-compliance exposure output must use a .json suffix")
    model_save_dir = validate_motion_compliance_run_path(
        config.callbacks.model_save.save_dir
    )
    if model_save_dir != experiment_dir:
        raise ValueError("motion-compliance model_save.save_dir must equal experiment_dir")
    hydra_save_dir = validate_motion_compliance_run_path(config.save_dir)
    if hydra_save_dir != experiment_dir / ".hydra":
        raise ValueError("motion-compliance save_dir must equal experiment_dir/.hydra")
    artifact_paths = {
        "input_checkpoint": input_checkpoint,
        "last_checkpoint": experiment_dir / "last.pt",
        "exposure_report": exposure_output,
        "resolved_config": experiment_dir / "config.yaml",
        "run_metadata": experiment_dir / "meta.yaml",
    }
    if migration_enabled:
        migration_output = Path(migration_cfg.output_path).expanduser().resolve(strict=False)
        if migration_output.suffix != ".pt":
            raise ValueError("motion-compliance migration output must use a .pt suffix")
        artifact_paths["migration_output"] = migration_output
    validate_distinct_artifact_paths(**artifact_paths)

    if not finetune_cfg.get("enforce_phase4_smoke_contract", False):
        return

    expected_iterations = 1 if config.get("resume", False) else 5
    expected_save_last = 1 if config.get("resume", False) else 5
    exact_values = (
        ("motion_compliance_finetune.stage", finetune_cfg.get("stage"), "decoder_critic"),
        (
            "motion_compliance_finetune.trainable_decoder_names",
            tuple(finetune_cfg.get("trainable_decoder_names", ())),
            ("g1_dyn",),
        ),
        ("algo.config.use_log_std", config.algo.config.get("use_log_std", False), False),
        (
            "algo.config.use_clampped_std",
            config.algo.config.get("use_clampped_std", False),
            True,
        ),
        ("algo.config.std_clamp_min", config.algo.config.get("std_clamp_min"), 0.001),
        ("algo.config.std_clamp_max", config.algo.config.get("std_clamp_max"), 0.5),
        (
            "algo.config.clamp_noise_std",
            config.algo.config.get("clamp_noise_std", False),
            False,
        ),
        ("num_envs", config.get("num_envs"), 16),
        (
            "algo.config.num_learning_iterations",
            config.algo.config.get("num_learning_iterations"),
            expected_iterations,
        ),
        ("use_wandb", config.get("use_wandb"), False),
        (
            "callbacks.model_save.save_last_frequency",
            config.callbacks.model_save.get("save_last_frequency"),
            expected_save_last,
        ),
        (
            "manager_env.commands.motion.motion_lib_cfg.motion_file",
            Path(config.manager_env.commands.motion.motion_lib_cfg.motion_file)
            .expanduser()
            .resolve(strict=False),
            OFFICIAL_SAMPLE_ROBOT_MOTION.resolve(strict=False),
        ),
        (
            "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file",
            Path(config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file)
            .expanduser()
            .resolve(strict=False),
            OFFICIAL_SAMPLE_SMPL_MOTION.resolve(strict=False),
        ),
        (
            "manager_env.commands.motion.motion_lib_cfg.multi_thread",
            config.manager_env.commands.motion.motion_lib_cfg.get("multi_thread"),
            False,
        ),
        (
            "manager_env.commands.motion_compliance.site_body_names",
            tuple(config.manager_env.commands.motion_compliance.site_body_names),
            _PHASE4_SITE_NAMES,
        ),
        (
            "manager_env.commands.motion_compliance.enabled",
            config.manager_env.commands.motion_compliance.get("enabled"),
            True,
        ),
        (
            "manager_env.commands.motion_compliance.enable_probability",
            config.manager_env.commands.motion_compliance.get("enable_probability"),
            1.0,
        ),
        (
            "manager_env.commands.motion_compliance.site_activation_probability",
            config.manager_env.commands.motion_compliance.get("site_activation_probability"),
            1.0,
        ),
        (
            "manager_env.commands.motion_compliance.resampling_time_range",
            tuple(config.manager_env.commands.motion_compliance.resampling_time_range),
            (0.02, 0.02),
        ),
        (
            "manager_env.commands.motion_compliance.force_threshold_range_n",
            tuple(config.manager_env.commands.motion_compliance.force_threshold_range_n),
            (10.0, 10.0),
        ),
        (
            "manager_env.commands.motion_compliance.reference_offset_range_m",
            tuple(config.manager_env.commands.motion_compliance.reference_offset_range_m),
            (0.05, 0.05),
        ),
        (
            "manager_env.commands.motion_compliance.reference_displacement_m",
            config.manager_env.commands.motion_compliance.get("reference_displacement_m"),
            0.05,
        ),
    )
    for field_name, actual, expected in exact_values:
        if actual != expected:
            raise ValueError(
                f"Phase-4 smoke requires {field_name}={expected!r}; got {actual!r}"
            )


def _validate_decoder_names(decoders, names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, str | bytes):
        raise TypeError("trainable_decoder_names must be a sequence of names")
    resolved = tuple(names)
    if not resolved or any(not isinstance(name, str) or not name for name in resolved):
        raise ValueError("trainable_decoder_names must contain non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ValueError("trainable_decoder_names contains duplicates")
    missing = sorted(set(resolved) - set(decoders))
    if missing:
        raise ValueError(f"unknown trainable decoders: {missing}")
    return resolved


def configure_motion_compliance_finetune_stage(
    policy: torch.nn.Module,
    value_model: torch.nn.Module | None,
    *,
    stage: str,
    trainable_decoder_names: Sequence[str] = ("g1_dyn",),
) -> FinetuneStageReport:
    """Freeze all policy state except selected decoders, or explicitly unfreeze."""

    if stage not in {"decoder_critic", "full"}:
        raise ValueError("finetune stage must be 'decoder_critic' or 'full'")
    actor_module = getattr(policy, "actor_module", None)
    decoders = getattr(actor_module, "decoders", None)
    if decoders is None:
        raise TypeError("policy must expose actor_module.decoders")
    decoder_names = _validate_decoder_names(decoders, trainable_decoder_names)
    if value_model is None:
        raise TypeError("motion-compliance finetuning requires a critic value model")

    if stage == "decoder_critic":
        if decoder_names != ("g1_dyn",):
            raise ValueError("decoder_critic stage trains exactly the g1_dyn decoder")
        encoders = getattr(actor_module, "encoders", None)
        quantizer = getattr(actor_module, "quantizer", None)
        if encoders is None or not tuple(encoders):
            raise TypeError("policy must expose non-empty actor_module.encoders")
        if quantizer is None:
            raise TypeError("policy must expose actor_module.quantizer")
        if "g1_kin" not in decoders:
            raise TypeError("policy must expose the frozen g1_kin decoder")
        noise_names = tuple(
            name for name, _ in policy.named_parameters() if name in {"std", "log_std"}
        )
        if len(noise_names) != 1:
            raise TypeError("policy must expose exactly one std or log_std noise parameter")

    for parameter in policy.parameters():
        parameter.requires_grad = stage == "full"
    if stage == "decoder_critic":
        for decoder_name in decoder_names:
            for parameter in decoders[decoder_name].parameters():
                parameter.requires_grad = True
    for parameter in value_model.parameters():
        parameter.requires_grad = True

    trainable_policy = tuple(
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    )
    frozen_policy = tuple(
        name for name, parameter in policy.named_parameters() if not parameter.requires_grad
    )
    trainable_value = tuple(
        name for name, parameter in value_model.named_parameters() if parameter.requires_grad
    )
    if not trainable_policy:
        raise ValueError("finetune stage selected no trainable policy parameters")
    if stage == "decoder_critic":
        allowed_prefixes = tuple(f"actor_module.decoders.{name}." for name in decoder_names)
        if any(not name.startswith(allowed_prefixes) for name in trainable_policy):
            raise RuntimeError("decoder_critic stage leaked non-decoder policy parameters")
        required_frozen_prefixes = (
            "actor_module.encoders.",
            "actor_module.decoders.g1_kin.",
        )
        for prefix in required_frozen_prefixes:
            matching = tuple(name for name in frozen_policy if name.startswith(prefix))
            if not matching:
                raise RuntimeError(f"decoder_critic stage found no frozen parameters for {prefix}")
        quantizer_parameters = tuple(actor_module.quantizer.parameters())
        if any(parameter.requires_grad for parameter in quantizer_parameters):
            raise RuntimeError("decoder_critic stage left quantizer parameters trainable")
        noise_name = next(
            name for name, _ in policy.named_parameters() if name in {"std", "log_std"}
        )
        if noise_name not in frozen_policy:
            raise RuntimeError("decoder_critic stage left action noise trainable")
    if not trainable_value:
        raise ValueError("finetune stage selected no trainable critic parameters")
    return FinetuneStageReport(
        stage=stage,
        trainable_policy_names=trainable_policy,
        frozen_policy_names=frozen_policy,
        trainable_value_names=trainable_value,
    )


def validate_optimizer_parameter_set(
    optimizer: torch.optim.Optimizer,
    policy: torch.nn.Module,
    value_model: torch.nn.Module | None,
) -> None:
    """Require optimizer ownership to equal exactly all requires-grad parameters."""

    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if value_model is not None:
        expected.update(
            id(parameter) for parameter in value_model.parameters() if parameter.requires_grad
        )
    actual_list = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]
    actual = {id(parameter) for parameter in actual_list}
    if len(actual) != len(actual_list):
        raise RuntimeError("optimizer contains duplicate parameters")
    if actual != expected:
        raise RuntimeError(
            "optimizer parameter ownership differs from requires-grad set: "
            f"missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )
