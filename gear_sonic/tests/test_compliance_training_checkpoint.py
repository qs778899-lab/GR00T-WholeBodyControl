"""Phase-4 checkpoint, freezing, exposure, and Hydra contracts."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from gear_sonic.compliance_control.training import (
    MOTION_COMPLIANCE_ACTOR_TARGET,
    MOTION_COMPLIANCE_MIGRATION_KEY,
    OFFICIAL_SONIC_RELEASE_SHA256,
    MotionComplianceExposureCallback,
    MotionComplianceFrozenNoiseActor,
    audit_migrated_init_checkpoint,
    audit_trained_motion_compliance_checkpoint,
    configure_motion_compliance_finetune_stage,
    critic_added_columns,
    migrate_motion_compliance_checkpoint,
    strict_load_policy_value_state,
    validate_checkpoint_sha256,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
    validate_motion_compliance_workflow_config,
    validate_optimizer_parameter_set,
    validate_strict_resume_payload,
)
from gear_sonic.compliance_control.training import audit as audit_module
from gear_sonic.compliance_control.training.checkpoint import (
    ACTOR_ADDED_COLUMNS,
    ACTOR_INPUT_WEIGHT_KEY,
    CRITIC_INPUT_WEIGHT_KEY,
    CRITIC_RUNNING_MEAN_KEY,
    CRITIC_RUNNING_VAR_KEY,
    OFFICIAL_ACTOR_INPUT_WIDTH,
    OFFICIAL_CRITIC_INPUT_WIDTH,
    OFFICIAL_INPUT_HIDDEN_WIDTH,
)
from gear_sonic.compliance_control.training.finetune import (
    MOTION_COMPLIANCE_TRAINER_TARGET,
)
from gear_sonic.compliance_control.training.trainer import MotionCompliancePPOTrainer
from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer
from gear_sonic.utils.config_utils import register_rl_resolvers


ROOT = Path(__file__).parents[2]
CONFIG_DIR = str((ROOT / "gear_sonic" / "config").resolve())
OFFICIAL_CHECKPOINT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/"
    "sonic_release/last.pt"
)
RUNS_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion"
)


class _InputMLP(nn.Module):
    def __init__(self, input_width: int):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(input_width, OFFICIAL_INPUT_HIDDEN_WIDTH))


class _FakeBackbone(nn.Module):
    def __init__(self, actor_input_width: int):
        super().__init__()
        self.encoders = nn.ModuleDict({"g1": nn.Linear(2, 2)})
        self.quantizer = nn.Linear(1, 1, bias=False)
        self.decoders = nn.ModuleDict(
            {
                "g1_dyn": _InputMLP(actor_input_width),
                "g1_kin": _InputMLP(64),
            }
        )


class _ConstantActionBackbone(_FakeBackbone):
    def forward(self, inputs, **kwargs):
        del kwargs
        return torch.zeros(
            (*inputs.shape[:-1], 29),
            dtype=inputs.dtype,
            device=inputs.device,
        )


class _FakePolicy(nn.Module):
    def __init__(self, actor_input_width: int):
        super().__init__()
        self.std = nn.Parameter(torch.full((29,), 0.05))
        self.actor_module = _FakeBackbone(actor_input_width)


class _RunningStats(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(width))
        self.register_buffer("running_var", torch.ones(width))
        self.register_buffer("count", torch.tensor(69_574_656_000.0, dtype=torch.float32))


class _FakeValue(nn.Module):
    def __init__(self, critic_input_width: int):
        super().__init__()
        self.critic_module = _InputMLP(critic_input_width)
        self.running_mean_std = _RunningStats(critic_input_width)


def _source_checkpoint_from_targets(
    policy: _FakePolicy,
    value: _FakeValue,
) -> dict:
    policy_state = {key: tensor.detach().clone() for key, tensor in policy.state_dict().items()}
    value_state = {key: tensor.detach().clone() for key, tensor in value.state_dict().items()}
    policy_state[ACTOR_INPUT_WEIGHT_KEY] = torch.full(
        (OFFICIAL_INPUT_HIDDEN_WIDTH, OFFICIAL_ACTOR_INPUT_WIDTH),
        0.25,
    )
    value_state[CRITIC_INPUT_WEIGHT_KEY] = torch.full(
        (OFFICIAL_INPUT_HIDDEN_WIDTH, OFFICIAL_CRITIC_INPUT_WIDTH),
        -0.125,
    )
    value_state[CRITIC_RUNNING_MEAN_KEY] = torch.linspace(
        -1.0,
        1.0,
        OFFICIAL_CRITIC_INPUT_WIDTH,
    )
    value_state[CRITIC_RUNNING_VAR_KEY] = torch.linspace(
        0.5,
        1.5,
        OFFICIAL_CRITIC_INPUT_WIDTH,
    )
    return {
        "policy_state_dict": policy_state,
        "value_state_dict": value_state,
        "optimizer_state_dict": {"old": True},
        "lr_scheduler_state_dict": {"old": True},
        "env_state_dict": {"old": True},
        "state": SimpleNamespace(global_step=41550),
        "args": SimpleNamespace(learning_rate=1e-3),
    }


@pytest.fixture(scope="module")
def migrated_bundle():
    policy = _FakePolicy(OFFICIAL_ACTOR_INPUT_WIDTH + ACTOR_ADDED_COLUMNS)
    value = _FakeValue(OFFICIAL_CRITIC_INPUT_WIDTH + critic_added_columns(2))
    source = _source_checkpoint_from_targets(policy, value)
    migrated, report = migrate_motion_compliance_checkpoint(
        source,
        source_sha256=OFFICIAL_SONIC_RELEASE_SHA256,
        num_sites=2,
        target_policy_state=policy.state_dict(),
        target_value_state=value.state_dict(),
    )
    return policy, value, source, migrated, report


def _resume_checkpoint(policy, value, global_step: int = 5) -> dict:
    return {
        "policy_state_dict": {
            key: tensor.detach().clone() for key, tensor in policy.state_dict().items()
        },
        "value_state_dict": {
            key: tensor.detach().clone() for key, tensor in value.state_dict().items()
        },
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(global_step)}},
            "param_groups": [{"params": [0]}],
        },
        "lr_scheduler_state_dict": {"last_epoch": global_step},
        "env_state_dict": {"motion_lib": {"sample": 0}},
        "state": SimpleNamespace(global_step=global_step),
    }


def test_critic_width_is_derived_from_configured_site_count():
    assert {site_count: critic_added_columns(site_count) for site_count in (1, 2, 5)} == {
        1: 8,
        2: 12,
        5: 24,
    }
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            critic_added_columns(invalid)


def test_synthetic_migration_copies_every_legacy_value_and_initializes_only_tails(
    migrated_bundle,
):
    _, _, source, migrated, report = migrated_bundle
    audit_migrated_init_checkpoint(migrated)
    assert report.actor_input_width == (994, 997)
    assert report.critic_input_width == (1645, 1657)
    assert report.num_sites == 2
    assert migrated["optimizer_state_dict"] is None
    assert migrated["lr_scheduler_state_dict"] is None
    assert migrated["env_state_dict"] is None
    assert migrated["state"].global_step == 0

    migrated_actor = migrated["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY]
    source_actor = source["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY]
    assert torch.equal(migrated_actor[:, :994], source_actor)
    assert torch.count_nonzero(migrated_actor[:, 994:]).item() == 0

    migrated_critic = migrated["value_state_dict"][CRITIC_INPUT_WEIGHT_KEY]
    source_critic = source["value_state_dict"][CRITIC_INPUT_WEIGHT_KEY]
    assert torch.equal(migrated_critic[:, :1645], source_critic)
    assert torch.count_nonzero(migrated_critic[:, 1645:]).item() == 0
    assert torch.equal(
        migrated["value_state_dict"][CRITIC_RUNNING_MEAN_KEY][:1645],
        source["value_state_dict"][CRITIC_RUNNING_MEAN_KEY],
    )
    assert torch.count_nonzero(
        migrated["value_state_dict"][CRITIC_RUNNING_MEAN_KEY][1645:]
    ).item() == 0
    assert torch.equal(
        migrated["value_state_dict"][CRITIC_RUNNING_VAR_KEY][1645:],
        torch.ones(12),
    )
    for group in ("policy_state_dict", "value_state_dict"):
        expanded = {
            ACTOR_INPUT_WEIGHT_KEY,
            CRITIC_INPUT_WEIGHT_KEY,
            CRITIC_RUNNING_MEAN_KEY,
            CRITIC_RUNNING_VAR_KEY,
        }
        for key, source_tensor in source[group].items():
            if key not in expanded:
                assert torch.equal(migrated[group][key], source_tensor)


def test_migration_rejects_any_unpinned_official_contract(migrated_bundle):
    policy, value, source, _, _ = migrated_bundle
    kwargs = {
        "source_sha256": OFFICIAL_SONIC_RELEASE_SHA256,
        "target_policy_state": policy.state_dict(),
        "target_value_state": value.state_dict(),
    }
    with pytest.raises(ValueError, match="audited official"):
        migrate_motion_compliance_checkpoint(source, **{**kwargs, "source_sha256": "0" * 64})
    with pytest.raises(ValueError, match="expected_source_step"):
        migrate_motion_compliance_checkpoint(source, **kwargs, expected_source_step=41549)
    with pytest.raises(ValueError, match="source_revision"):
        migrate_motion_compliance_checkpoint(source, **kwargs, source_revision="main")
    missing_target = dict(policy.state_dict())
    missing_target.pop("std")
    with pytest.raises(ValueError, match="target keys differ"):
        migrate_motion_compliance_checkpoint(
            source,
            **{**kwargs, "target_policy_state": missing_target},
        )


def test_migrated_init_and_normal_resume_model_loading_are_strict(migrated_bundle):
    policy, value, _, migrated, _ = migrated_bundle
    report = strict_load_policy_value_state(policy, value, migrated, resume=False)
    assert report.strict and report.migrated_init
    with pytest.raises(ValueError, match="cannot be used with resume=true"):
        strict_load_policy_value_state(policy, value, migrated, resume=True)

    resume_checkpoint = _resume_checkpoint(policy, value)
    report = strict_load_policy_value_state(policy, value, resume_checkpoint, resume=True)
    assert report.strict and not report.migrated_init
    broken = dict(resume_checkpoint)
    broken["policy_state_dict"] = dict(resume_checkpoint["policy_state_dict"])
    broken["policy_state_dict"].pop("std")
    with pytest.raises(RuntimeError):
        strict_load_policy_value_state(policy, value, broken, resume=True)


def test_strict_resume_requires_every_nonempty_training_state(migrated_bundle):
    policy, value, _, _, _ = migrated_bundle
    checkpoint = _resume_checkpoint(policy, value)
    payload = validate_strict_resume_payload(checkpoint)
    assert payload.state.global_step == 5
    for key in (
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "env_state_dict",
        "state",
    ):
        broken = dict(checkpoint)
        broken[key] = None
        with pytest.raises(ValueError, match="strict resume"):
            validate_strict_resume_payload(broken)


def test_decoder_critic_freeze_and_optimizer_ownership_are_exact(migrated_bundle):
    policy, value, _, migrated, _ = migrated_bundle
    strict_load_policy_value_state(policy, value, migrated, resume=False)
    report = configure_motion_compliance_finetune_stage(
        policy,
        value,
        stage="decoder_critic",
        trainable_decoder_names=("g1_dyn",),
    )
    assert report.trainable_policy_names
    assert all(
        name.startswith("actor_module.decoders.g1_dyn.")
        for name in report.trainable_policy_names
    )
    assert "std" in report.frozen_policy_names
    assert any(name.startswith("actor_module.encoders.") for name in report.frozen_policy_names)
    assert any(
        name.startswith("actor_module.decoders.g1_kin.")
        for name in report.frozen_policy_names
    )
    assert report.trainable_value_names

    trainable = [
        parameter
        for module in (policy, value)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    validate_optimizer_parameter_set(optimizer, policy, value)
    frozen = next(parameter for parameter in policy.parameters() if not parameter.requires_grad)
    bad_optimizer = torch.optim.Adam([*trainable, frozen], lr=1e-3)
    with pytest.raises(RuntimeError, match="ownership differs"):
        validate_optimizer_parameter_set(bad_optimizer, policy, value)
    with pytest.raises(ValueError, match="exactly the g1_dyn"):
        configure_motion_compliance_finetune_stage(
            policy,
            value,
            stage="decoder_critic",
            trainable_decoder_names=("g1_kin",),
        )


def test_frozen_release_std_clamp_is_non_mutating_and_optimizer_excluded(monkeypatch):
    official = audit_module.load_trl_checkpoint(OFFICIAL_CHECKPOINT, map_location="cpu")
    policy_key = (
        "actor_model_state_dict"
        if "actor_model_state_dict" in official
        else "policy_state_dict"
    )
    official_std = official[policy_key]["std"].detach().clone()
    assert torch.any(official_std > 0.5)

    actor = object.__new__(MotionComplianceFrozenNoiseActor)
    nn.Module.__init__(actor)
    actor.algo_config = OmegaConf.create(
        {
            "use_clampped_std": True,
            "std_clamp_min": 0.001,
            "std_clamp_max": 0.5,
        }
    )
    actor.actor_module = _ConstantActionBackbone(OFFICIAL_ACTOR_INPUT_WIDTH + 3)
    actor.input_key = "actor_obs"
    actor.input_obs_dict = False
    actor.has_aux_loss = False
    actor.aux_losses = None
    actor.aux_loss_coef = None
    actor.output_original_obs_dict = False
    actor.use_batch_norm = False
    actor.use_running_mean_std = False
    actor.running_mean_std = None
    actor.use_log_std = False
    actor.std = nn.Parameter(official_std.clone())
    actor.clamp_noise_std = False
    actor.distribution = None

    value = _FakeValue(OFFICIAL_CRITIC_INPUT_WIDTH + critic_added_columns(2))
    configure_motion_compliance_finetune_stage(
        actor,
        value,
        stage="decoder_critic",
        trainable_decoder_names=("g1_dyn",),
    )
    trainable = [
        parameter
        for module in (actor, value)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    validate_optimizer_parameter_set(optimizer, actor, value)
    assert all(
        actor.std is not parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    before = actor.std.detach().clone()
    expected_effective = torch.clamp(before, min=0.001, max=0.5)
    for _ in range(3):
        actor.update_distribution({"actor_obs": torch.zeros(2, 4)})
        assert torch.equal(actor.action_std[0], expected_effective)
        assert torch.equal(actor.std, before)

    captured_logs = {}

    def capture_base_log(self, logs, start_time=None):
        del self, start_time
        captured_logs.update(logs)

    monkeypatch.setattr(TRLAuxLossPPOTrainer, "log", capture_base_log)
    trainer = object.__new__(MotionCompliancePPOTrainer)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.model = SimpleNamespace(policy=actor)
    MotionCompliancePPOTrainer.log(
        trainer,
        {"Policy/mean_noise_std": float(before.mean().item())},
    )
    assert captured_logs["Policy/mean_noise_std"] == float(expected_effective.mean().item())
    assert torch.equal(actor.std, before)


def test_custom_trainer_strict_resume_restores_all_state(tmp_path, migrated_bundle):
    policy, value, _, migrated, _ = migrated_bundle
    checkpoint = _resume_checkpoint(policy, value)
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(checkpoint, checkpoint_path)

    class Loader:
        def __init__(self):
            self.loaded = None
            self.param_groups = [{"lr": 0.1}]

        def load_state_dict(self, state):
            self.loaded = state

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def unwrap_model(model):
            return model

    optimizer = Loader()
    scheduler = Loader()
    env = SimpleNamespace(loaded=None)
    env.load_env_state_dict = lambda state: setattr(env, "loaded", state)
    trainer = object.__new__(MotionCompliancePPOTrainer)
    trainer.accelerator = Accelerator()
    trainer.model = SimpleNamespace(policy=policy, value_model=value)
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.env = env
    trainer.args = SimpleNamespace(learning_rate=0.1)
    trainer.state = SimpleNamespace(global_step=0)

    init_path = tmp_path / "init.pt"
    torch.save(migrated, init_path)
    init_loaded = MotionCompliancePPOTrainer.load_checkpoint(
        trainer,
        init_path,
        resume=False,
    )
    assert MOTION_COMPLIANCE_MIGRATION_KEY in init_loaded
    assert optimizer.loaded is None
    assert scheduler.loaded is None
    assert env.loaded is None
    assert trainer.state.global_step == 0

    loaded = MotionCompliancePPOTrainer.load_checkpoint(
        trainer,
        checkpoint_path,
        resume=True,
    )
    assert loaded["state"].global_step == 5
    assert trainer.state.global_step == 5
    assert optimizer.loaded == checkpoint["optimizer_state_dict"]
    assert scheduler.loaded == checkpoint["lr_scheduler_state_dict"]
    assert env.loaded == checkpoint["env_state_dict"]


def test_exposure_callback_writes_each_step_and_finalizes_per_site(tmp_path):
    runs_root = tmp_path / "runs"
    output_path = runs_root / "case" / "exposure.json"
    callback = MotionComplianceExposureCallback(
        str(output_path),
        runs_root=str(runs_root),
    )
    command = SimpleNamespace(
        state=SimpleNamespace(
            enabled=torch.tensor([True, True]),
            active_site_mask=torch.tensor([[True, True], [True, True]]),
            site_force_world=torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                ]
            ),
        )
    )
    command_manager = SimpleNamespace(get_term=lambda name: command)
    env = SimpleNamespace(command_manager=command_manager)
    state = SimpleNamespace(global_step=1, max_steps=2)
    control = SimpleNamespace()
    callback.on_log(
        None,
        state,
        control,
        logs={
            "loss/policy_avg": 0.1,
            "loss/value_avg": torch.tensor(0.2),
            "collection_time": 0.3,
            "learn_time": 0.4,
            "fps": 100.0,
        },
    )
    callback.on_step_end(None, state, control, env=env)
    first = json.loads(output_path.read_text(encoding="utf-8"))
    assert first["observed_batches"] == 1
    assert not callback._finalized

    state.global_step = 2
    callback.on_log(
        None,
        state,
        control,
        logs={
            "loss/policy_avg": 0.09,
            "loss/value_avg": 0.18,
            "collection_time": 0.31,
            "learn_time": 0.41,
            "fps": 99.0,
        },
    )
    callback.on_step_end(None, state, control, env=env)
    final = json.loads(output_path.read_text(encoding="utf-8"))
    assert callback._finalized
    assert final["active_site_samples_by_index"] == [4, 4]
    assert final["nonzero_force_site_samples_by_index"] == [4, 4]
    callback.on_train_end(None, state, control)

    missing_site = MotionComplianceExposureCallback(
        str(runs_root / "missing" / "exposure.json"),
        runs_root=str(runs_root),
    )
    command.state.site_force_world[:, 1] = 0.0
    state.global_step = state.max_steps = 1
    missing_site.on_log(
        None,
        state,
        control,
        logs={
            "loss/policy_avg": 0.1,
            "collection_time": 0.3,
            "learn_time": 0.4,
            "fps": 100.0,
        },
    )
    with pytest.raises(RuntimeError, match="no physical exposure"):
        missing_site.on_step_end(None, state, control, env=env)
    assert (runs_root / "missing" / "exposure.json").is_file()

    stale_force = MotionComplianceExposureCallback(
        str(runs_root / "stale" / "exposure.json"),
        runs_root=str(runs_root),
    )
    command.state.active_site_mask[:, 1] = False
    command.state.site_force_world[:, 1, 0] = 1.0
    with pytest.raises(RuntimeError, match="persisted"):
        stale_force.on_step_end(None, state, control, env=env)

    invalid_loss = MotionComplianceExposureCallback(
        str(runs_root / "invalid_loss" / "exposure.json"),
        runs_root=str(runs_root),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        invalid_loss.on_log(
            None,
            state,
            control,
            logs={
                "loss/policy_avg": float("nan"),
                "collection_time": 0.3,
                "learn_time": 0.4,
                "fps": 100.0,
            },
        )


def _compose_finetune_config():
    register_rl_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune",
                "exp_base=phase4_test",
                "timestamp=20260727_000000",
            ],
        )


def test_finetune_hydra_config_resolves_to_low_resource_owned_workflow():
    config = _compose_finetune_config()
    assert config.base_dir == str(RUNS_ROOT)
    assert Path(config.experiment_dir).is_relative_to(RUNS_ROOT)
    assert Path(config.checkpoint) == OFFICIAL_CHECKPOINT
    assert config.trainer._target_ == MOTION_COMPLIANCE_TRAINER_TARGET
    assert config.algo.config.actor._target_ == MOTION_COMPLIANCE_ACTOR_TARGET
    assert config.algo.config.get("use_log_std", False) is False
    assert config.algo.config.use_clampped_std is True
    assert config.algo.config.std_clamp_min == 0.001
    assert config.algo.config.std_clamp_max == 0.5
    assert config.algo.config.get("clamp_noise_std", False) is False
    assert config.num_envs == 16
    assert config.algo.config.num_learning_iterations == 5
    assert config.use_wandb is False
    assert config.callbacks.model_save.save_last_frequency == 5
    assert config.manager_env.commands.motion.motion_lib_cfg.multi_thread is False
    assert config.manager_env.commands.motion.motion_lib_cfg.motion_file.endswith(
        "walk_forward_amateur_001__A001.pkl"
    )
    assert config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file.endswith(
        "sample_data/smpl_filtered"
    )
    assert list(config.manager_env.commands.motion_compliance.site_body_names) == [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    assert list(config.manager_env.commands.motion_compliance.resampling_time_range) == [
        0.02,
        0.02,
    ]
    validate_motion_compliance_workflow_config(config)

    resume_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    resume_config.resume = True
    resume_config.motion_compliance_checkpoint_migration.enabled = False
    resume_config.checkpoint = str(RUNS_ROOT / "resume_case" / "last.pt")
    resume_config.experiment_dir = str(RUNS_ROOT / "resume_output")
    resume_config.motion_compliance_finetune.resume_output_dir = str(
        RUNS_ROOT / "resume_output"
    )
    resume_config.algo.config.num_learning_iterations = 1
    resume_config.callbacks.model_save.save_last_frequency = 1
    validate_motion_compliance_workflow_config(resume_config)

    bad_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    bad_config.base_dir = "logs_rl"
    bad_config.experiment_dir = "logs_rl/run"
    with pytest.raises(ValueError, match="artifacts must be written below"):
        validate_motion_compliance_workflow_config(bad_config)

    invalid_smoke_values = {
        "motion_compliance_finetune.stage": "full",
        "motion_compliance_finetune.trainable_decoder_names": ["g1_kin"],
        "algo.config.use_log_std": True,
        "algo.config.use_clampped_std": False,
        "algo.config.std_clamp_min": 0.002,
        "algo.config.std_clamp_max": 0.6,
        "algo.config.clamp_noise_std": True,
        "num_envs": 32,
        "algo.config.num_learning_iterations": 6,
        "use_wandb": True,
        "callbacks.model_save.save_last_frequency": 4,
        "manager_env.commands.motion.motion_lib_cfg.motion_file": "/tmp/other.pkl",
        "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file": "/tmp/smpl",
        "manager_env.commands.motion.motion_lib_cfg.multi_thread": True,
        "manager_env.commands.motion_compliance.site_body_names": ["left_wrist_yaw_link"],
        "manager_env.commands.motion_compliance.enabled": False,
        "manager_env.commands.motion_compliance.enable_probability": 0.5,
        "manager_env.commands.motion_compliance.site_activation_probability": 0.5,
        "manager_env.commands.motion_compliance.resampling_time_range": [2.0, 16.0],
        "manager_env.commands.motion_compliance.force_threshold_range_n": [5.0, 15.0],
        "manager_env.commands.motion_compliance.reference_offset_range_m": [0.0, 0.01],
        "manager_env.commands.motion_compliance.reference_displacement_m": 0.1,
    }
    for field_name, invalid_value in invalid_smoke_values.items():
        invalid = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        OmegaConf.update(invalid, field_name, invalid_value)
        with pytest.raises(ValueError, match="Phase-4 smoke requires"):
            validate_motion_compliance_workflow_config(invalid)

    wrong_actor = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    wrong_actor.algo.config.actor._target_ = (
        "gear_sonic.trl.modules.actor_critic_modules.Actor"
    )
    with pytest.raises(ValueError, match="frozen-noise actor"):
        validate_motion_compliance_workflow_config(wrong_actor)

    collision = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    collision.callbacks.motion_compliance_exposure.output_path = (
        "${experiment_dir}/last.pt"
    )
    with pytest.raises(ValueError, match="must use a .json suffix"):
        validate_motion_compliance_workflow_config(collision)

    resume_collision = OmegaConf.create(
        OmegaConf.to_container(resume_config, resolve=False)
    )
    resume_collision.experiment_dir = str(RUNS_ROOT / "resume_case")
    resume_collision.motion_compliance_finetune.resume_output_dir = str(
        RUNS_ROOT / "resume_case"
    )
    with pytest.raises(ValueError, match="must be distinct"):
        validate_motion_compliance_workflow_config(resume_collision)

    escaped_model_save = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    escaped_model_save.callbacks.model_save.save_dir = "/tmp/escaped"
    with pytest.raises(ValueError, match="artifacts must be written below"):
        validate_motion_compliance_workflow_config(escaped_model_save)

    mismatched_model_save = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    mismatched_model_save.callbacks.model_save.save_dir = str(RUNS_ROOT / "other")
    with pytest.raises(ValueError, match="must equal experiment_dir"):
        validate_motion_compliance_workflow_config(mismatched_model_save)


def test_compliance_resume_preserves_step5_and_routes_step6_to_separate_directory():
    from gear_sonic.train_agent_trl import resolve_checkpoint_and_experiment_dir

    step5 = RUNS_ROOT / "phase4_gpu_smoke" / "last.pt"
    resume_output = RUNS_ROOT / "phase4_gpu_resume"
    config = OmegaConf.create(
        {
            "resume": True,
            "checkpoint": str(step5),
            "experiment_dir": str(RUNS_ROOT / "hydra_initial"),
            "motion_compliance_finetune": {
                "resume_output_dir": str(resume_output),
            },
        }
    )
    resolve_checkpoint_and_experiment_dir(config)
    assert config.checkpoint == str(step5)
    assert config.experiment_dir == str(resume_output)


def test_training_artifact_paths_cannot_escape_the_central_root(tmp_path):
    owned = tmp_path / "runs"
    assert validate_motion_compliance_run_path(
        owned / "case" / "last.pt",
        runs_root=owned,
    ) == (owned / "case" / "last.pt").resolve()
    for invalid in (owned, tmp_path / "outside" / "last.pt"):
        with pytest.raises(ValueError, match="below"):
            validate_motion_compliance_run_path(invalid, runs_root=owned)
    with pytest.raises(ValueError, match="must be distinct"):
        validate_distinct_artifact_paths(
            checkpoint=owned / "case" / "last.pt",
            report=owned / "case" / "." / "last.pt",
        )


def test_post_train_audit_requires_nonzero_new_columns_and_exact_frozen_state(
    monkeypatch,
    migrated_bundle,
):
    _, _, source, migrated, _ = migrated_bundle
    trained = {
        "policy_state_dict": {
            key: tensor.detach().clone()
            for key, tensor in migrated["policy_state_dict"].items()
        },
        "value_state_dict": {
            key: tensor.detach().clone()
            for key, tensor in migrated["value_state_dict"].items()
        },
        "optimizer_state_dict": {
            "state": {
                index: {
                    "step": torch.tensor(5),
                    "exp_avg": torch.zeros(1),
                    "exp_avg_sq": torch.zeros(1),
                }
                for index in range(4)
            },
            "param_groups": [{"params": list(range(4))}],
        },
        "lr_scheduler_state_dict": {"last_epoch": 5},
        "env_state_dict": {"motion_lib": {"sample": 0}},
        "state": SimpleNamespace(global_step=5),
    }
    trained["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY][:, -3:] = 0.01
    trained["value_state_dict"][CRITIC_INPUT_WEIGHT_KEY][:, -12:] = 0.01
    monkeypatch.setattr(audit_module, "validate_checkpoint_sha256", lambda *args: "ok")
    monkeypatch.setattr(
        audit_module,
        "load_trl_checkpoint",
        lambda path, map_location="cpu": (
            source if str(path).endswith("official.pt") else trained
        ),
    )
    report = audit_trained_motion_compliance_checkpoint(
        "/tmp/official.pt",
        RUNS_ROOT / "unit" / "last.pt",
        expected_global_step=5,
        num_sites=2,
    )
    assert all(report.actor_added_columns_nonzero)
    assert all(report.critic_added_columns_nonzero)
    assert report.optimizer_slot_count == 4
    assert report.optimizer_steps == (5,)

    trained["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY][0, -1] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
            num_sites=2,
        )
    trained["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY][0, -1] = 0.01

    trained["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("inf")
    with pytest.raises(ValueError, match="optimizer slot tensor"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
            num_sites=2,
        )
    trained["optimizer_state_dict"]["state"][0]["exp_avg"].zero_()

    source_std = source["policy_state_dict"]["std"]
    trained_std = trained["policy_state_dict"]["std"]
    original_source_std = source_std[0].clone()
    original_trained_std = trained_std[0].clone()
    source_std[0] = 0.0
    trained_std[0] = -0.0
    assert torch.equal(source_std, trained_std)
    with pytest.raises(ValueError, match="frozen policy tensor changed"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
            num_sites=2,
        )
    source_std[0] = original_source_std
    trained_std[0] = original_trained_std

    trained["policy_state_dict"]["std"][0] += 1.0
    with pytest.raises(ValueError, match="frozen policy tensor changed"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
            num_sites=2,
        )


def test_official_checkpoint_digest_and_generic_trainer_is_unmodified():
    assert validate_checkpoint_sha256(
        OFFICIAL_CHECKPOINT,
        OFFICIAL_SONIC_RELEASE_SHA256,
    ) == OFFICIAL_SONIC_RELEASE_SHA256
    generic_trainer = (ROOT / "gear_sonic" / "trl" / "trainer" / "ppo_trainer.py").read_text(
        encoding="utf-8"
    )
    assert "compliance_control" not in generic_trainer
    train_source = inspect.getsource(TRLPPOTrainer.train)
    metrics_index = train_source.index("train_metrics = self._get_train_metrics()")
    log_index = train_source.index("self.log(metrics)")
    step_end_index = train_source.index("self.callback_handler.on_step_end")
    assert metrics_index < log_index < step_end_index
