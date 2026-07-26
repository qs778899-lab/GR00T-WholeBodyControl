"""Residual-only parameter ownership for motion-compliance finetuning."""

from __future__ import annotations

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
from .residual_policy import (
    MOTION_COMPLIANCE_BACKBONE_TARGET,
    MOTION_COMPLIANCE_CRITIC_TARGET,
    motion_compliance_residual_parameters,
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
    frozen_value_names: tuple[str, ...]


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

    initialization_cfg = config.get(
        "motion_compliance_checkpoint_initialization",
        None,
    )
    initialization_enabled = bool(
        initialization_cfg is not None and initialization_cfg.get("enabled", False)
    )
    if config.get("resume", False):
        if initialization_enabled:
            raise ValueError("strict resume requires checkpoint initialization to be disabled")
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
        if not initialization_enabled:
            raise ValueError(
                "initial compliance finetuning requires official residual initialization"
            )
        source = Path(config.get("checkpoint", "")).expanduser().resolve(strict=False)
        official_source = OFFICIAL_SONIC_RELEASE_CHECKPOINT.resolve(strict=False)
        if source != official_source:
            raise ValueError(
                f"initial compliance checkpoint must be {official_source}; got {source}"
            )
        validate_motion_compliance_run_path(initialization_cfg.output_path)
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
    if initialization_enabled:
        initialization_output = (
            Path(initialization_cfg.output_path).expanduser().resolve(strict=False)
        )
        if initialization_output.suffix != ".pt":
            raise ValueError("motion-compliance initialization output must use a .pt suffix")
        artifact_paths["initialization_output"] = initialization_output
    validate_distinct_artifact_paths(**artifact_paths)

    if not finetune_cfg.get("enforce_phase4_smoke_contract", False):
        return

    expected_iterations = 1 if config.get("resume", False) else 5
    expected_save_last = 1 if config.get("resume", False) else 5
    exact_values = (
        ("motion_compliance_finetune.stage", finetune_cfg.get("stage"), "residual_only"),
        (
            "algo.config.num_steps_per_env",
            config.algo.config.get("num_steps_per_env"),
            24,
        ),
        (
            "algo.config.num_learning_epochs",
            config.algo.config.get("num_learning_epochs"),
            5,
        ),
        (
            "algo.config.num_mini_batches",
            config.algo.config.get("num_mini_batches"),
            4,
        ),
        (
            "manager_env.config.use_symmetry",
            config.manager_env.config.get("use_symmetry", False),
            False,
        ),
        (
            "algo.config.freeze_noise_std",
            config.algo.config.get("freeze_noise_std", False),
            True,
        ),
        (
            "algo.config.actor.backbone._target_",
            config.algo.config.actor.backbone.get("_target_"),
            MOTION_COMPLIANCE_BACKBONE_TARGET,
        ),
        (
            "algo.config.critic._target_",
            config.algo.config.critic.get("_target_"),
            MOTION_COMPLIANCE_CRITIC_TARGET,
        ),
        (
            "algo.config.actor.backbone.motion_compliance_action_delta_limit",
            config.algo.config.actor.backbone.get(
                "motion_compliance_action_delta_limit"
            ),
            0.25,
        ),
        (
            "algo.config.actor.backbone.motion_compliance_residual_hidden_dims",
            tuple(
                config.algo.config.actor.backbone.get(
                    "motion_compliance_residual_hidden_dims",
                    (),
                )
            ),
            (256, 256),
        ),
        (
            "algo.config.critic.motion_compliance_residual_hidden_dims",
            tuple(
                config.algo.config.critic.get(
                    "motion_compliance_residual_hidden_dims",
                    (),
                )
            ),
            (256, 256),
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
    if "trainable_decoder_names" in finetune_cfg:
        raise ValueError("residual-only finetuning forbids trainable_decoder_names")


def configure_motion_compliance_finetune_stage(
    policy: torch.nn.Module,
    value_model: torch.nn.Module | None,
    *,
    stage: str,
) -> FinetuneStageReport:
    """Freeze every release parameter and enable only the two residual heads."""

    if stage != "residual_only":
        raise ValueError("motion-compliance finetune stage must be 'residual_only'")
    if value_model is None:
        raise TypeError("motion-compliance finetuning requires a critic value model")

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in value_model.parameters():
        parameter.requires_grad_(False)

    policy_backbone = getattr(policy, "actor_module", policy)
    action_residual = getattr(
        policy_backbone,
        "motion_compliance_action_residual",
        None,
    )
    value_residual = getattr(
        value_model,
        "motion_compliance_value_residual",
        None,
    )
    if not isinstance(action_residual, torch.nn.Module):
        raise TypeError("policy lacks a motion-compliance action residual module")
    if not isinstance(value_residual, torch.nn.Module):
        raise TypeError("value model lacks a motion-compliance value residual module")
    for parameter in action_residual.parameters():
        parameter.requires_grad_(True)
    for parameter in value_residual.parameters():
        parameter.requires_grad_(True)
    motion_compliance_residual_parameters(policy, value_model)

    trainable_policy = tuple(
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    )
    frozen_policy = tuple(
        name for name, parameter in policy.named_parameters() if not parameter.requires_grad
    )
    trainable_value = tuple(
        name for name, parameter in value_model.named_parameters() if parameter.requires_grad
    )
    frozen_value = tuple(
        name for name, parameter in value_model.named_parameters() if not parameter.requires_grad
    )
    if not trainable_policy or not trainable_value:
        raise ValueError("residual-only stage selected an empty residual module")
    if any(
        not name.startswith("actor_module.motion_compliance_action_residual.")
        for name in trainable_policy
    ):
        raise RuntimeError("residual-only stage leaked a released policy parameter")
    if any(
        not name.startswith("motion_compliance_value_residual.")
        for name in trainable_value
    ):
        raise RuntimeError("residual-only stage leaked a released value parameter")
    return FinetuneStageReport(
        stage=stage,
        trainable_policy_names=trainable_policy,
        frozen_policy_names=frozen_policy,
        trainable_value_names=trainable_value,
        frozen_value_names=frozen_value,
    )


def validate_optimizer_parameter_set(
    optimizer: torch.optim.Optimizer,
    policy: torch.nn.Module,
    value_model: torch.nn.Module | None,
) -> None:
    """Require optimizer ownership to equal exactly both residual modules."""

    if value_model is None:
        raise TypeError("motion-compliance optimizer validation requires a value model")
    residual_parameters = motion_compliance_residual_parameters(policy, value_model)
    expected = {id(parameter) for parameter in residual_parameters}
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
