"""Phase-4 residual initialization, optimizer, PPO, audit, and Hydra contracts."""

from __future__ import annotations

import copy
from collections import deque
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from trl import PPOConfig

from gear_sonic.compliance_control.training import (
    MOTION_COMPLIANCE_ACTOR_TARGET,
    MOTION_COMPLIANCE_INITIALIZATION_KEY,
    OFFICIAL_SONIC_RELEASE_SHA256,
    MotionComplianceExposureCallback,
    MotionComplianceFrozenNoiseActor,
    ZeroInitializedResidualMLP,
    action_residual_context_width,
    audit_residual_init_checkpoint,
    audit_trained_motion_compliance_checkpoint,
    compliance_privileged_width,
    configure_motion_compliance_finetune_stage,
    expected_residual_shapes,
    initialize_motion_compliance_checkpoint,
    strict_load_policy_value_state,
    tensor_bytes_equal,
    validate_checkpoint_sha256,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
    validate_motion_compliance_workflow_config,
    validate_optimizer_parameter_set,
    validate_strict_resume_payload,
    value_residual_context_width,
)
from gear_sonic.compliance_control.training import audit as audit_module
from gear_sonic.compliance_control.training import trainer as trainer_module
from gear_sonic.compliance_control.training.checkpoint import (
    ACTION_RESIDUAL_PREFIX,
    ACTOR_INPUT_WEIGHT_KEY,
    CRITIC_INPUT_WEIGHT_KEY,
    CRITIC_RUNNING_MEAN_KEY,
    CRITIC_RUNNING_VAR_KEY,
    OFFICIAL_ACTOR_INPUT_WIDTH,
    OFFICIAL_CRITIC_INPUT_WIDTH,
    OFFICIAL_INPUT_HIDDEN_WIDTH,
    OFFICIAL_POLICY_TENSOR_COUNT,
    OFFICIAL_VALUE_TENSOR_COUNT,
    VALUE_RESIDUAL_PREFIX,
)
from gear_sonic.compliance_control.training.finetune import (
    MOTION_COMPLIANCE_TRAINER_TARGET,
)
from gear_sonic.compliance_control.training.trainer import (
    MotionCompliancePPOTrainer,
    preflight_optimizer_resume_state,
    preflight_trainer_resume_boundary,
    residual_parameter_names,
    validate_motion_compliance_ppo_batch,
    validate_motion_compliance_ppo_outputs,
    validate_optimizer_parameter_order,
    validate_residual_gradients,
)
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
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({"g1": nn.Linear(2, 2)})
        self.quantizer = nn.Linear(1, 1, bias=False)
        self.decoders = nn.ModuleDict(
            {
                "g1_dyn": _InputMLP(OFFICIAL_ACTOR_INPUT_WIDTH),
                "g1_kin": _InputMLP(64),
            }
        )
        self.motion_compliance_action_residual = ZeroInitializedResidualMLP(
            action_residual_context_width(),
            29,
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
    def __init__(self):
        super().__init__()
        self.std = nn.Parameter(torch.full((29,), 0.05))
        self.actor_module = _FakeBackbone()
        base_count = len(
            [
                key
                for key in self.state_dict()
                if not key.startswith(ACTION_RESIDUAL_PREFIX)
            ]
        )
        for index in range(OFFICIAL_POLICY_TENSOR_COUNT - base_count):
            self.register_buffer(f"release_policy_buffer_{index:02d}", torch.tensor(float(index)))


class _RunningStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(OFFICIAL_CRITIC_INPUT_WIDTH))
        self.register_buffer("running_var", torch.ones(OFFICIAL_CRITIC_INPUT_WIDTH))
        self.register_buffer("count", torch.tensor(69_574_656_000.0, dtype=torch.float32))


class _FakeValue(nn.Module):
    def __init__(self, num_sites: int = 2):
        super().__init__()
        self.critic_module = _InputMLP(OFFICIAL_CRITIC_INPUT_WIDTH)
        self.running_mean_std = _RunningStats()
        self.motion_compliance_value_residual = ZeroInitializedResidualMLP(
            value_residual_context_width(num_sites),
            1,
        )
        base_count = len(
            [key for key in self.state_dict() if not key.startswith(VALUE_RESIDUAL_PREFIX)]
        )
        for index in range(OFFICIAL_VALUE_TENSOR_COUNT - base_count):
            self.register_buffer(f"release_value_buffer_{index:02d}", torch.tensor(float(index)))


def _clone_state(state):
    return {key: tensor.detach().clone() for key, tensor in state.items()}


def _assert_nested_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert tensor_bytes_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_exact(actual_item, expected_item)
    else:
        assert actual == expected


def _source_checkpoint_from_targets(policy: _FakePolicy, value: _FakeValue) -> dict:
    policy_state = {
        key: tensor
        for key, tensor in _clone_state(policy.state_dict()).items()
        if not key.startswith(ACTION_RESIDUAL_PREFIX)
    }
    value_state = {
        key: tensor
        for key, tensor in _clone_state(value.state_dict()).items()
        if not key.startswith(VALUE_RESIDUAL_PREFIX)
    }
    policy_state[ACTOR_INPUT_WEIGHT_KEY].fill_(0.25)
    value_state[CRITIC_INPUT_WEIGHT_KEY].fill_(-0.125)
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
    assert len(policy_state) == OFFICIAL_POLICY_TENSOR_COUNT
    assert len(value_state) == OFFICIAL_VALUE_TENSOR_COUNT
    return {
        "policy_state_dict": policy_state,
        "value_state_dict": value_state,
        "optimizer_state_dict": {"old": True},
        "lr_scheduler_state_dict": {"old": True},
        "env_state_dict": {"old": True},
        "state": SimpleNamespace(global_step=41550),
        "args": SimpleNamespace(learning_rate=1e-3),
    }


@pytest.fixture
def initialized_bundle():
    policy = _FakePolicy()
    value = _FakeValue()
    source = _source_checkpoint_from_targets(policy, value)
    initialized, report = initialize_motion_compliance_checkpoint(
        source,
        source_sha256=OFFICIAL_SONIC_RELEASE_SHA256,
        num_sites=2,
        target_policy_state=policy.state_dict(),
        target_value_state=value.state_dict(),
    )
    return policy, value, source, initialized, report


def _optimizer_state_for_residuals(policy_state, value_state, step: int = 5):
    policy_shapes, value_shapes = expected_residual_shapes(2)
    policy_weights = tuple(
        policy_state[key] for key in policy_shapes if key.endswith(".weight")
    )
    value_weights = tuple(
        value_state[key] for key in value_shapes if key.endswith(".weight")
    )
    policy_biases = tuple(
        policy_state[key] for key in policy_shapes if key.endswith(".bias")
    )
    value_biases = tuple(
        value_state[key] for key in value_shapes if key.endswith(".bias")
    )
    tensors = policy_weights + value_weights + policy_biases + value_biases
    return {
        "state": {
            index: {
                "step": torch.tensor(float(step), dtype=torch.float32),
                "exp_avg": torch.ones_like(tensor),
                "exp_avg_sq": torch.ones_like(tensor),
            }
            for index, tensor in enumerate(tensors)
        },
        "param_groups": [
            {
                "params": list(range(6)),
                "lr": 2e-5,
                "initial_lr": 2e-5,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
            {
                "params": list(range(6, 12)),
                "lr": 3e-5,
                "initial_lr": 3e-5,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": True,
            },
        ],
    }


def _resume_checkpoint(
    policy,
    value,
    global_step: int = 5,
    optimizer_state=None,
) -> dict:
    policy_state = _clone_state(policy.state_dict())
    value_state = _clone_state(value.state_dict())
    state = SimpleNamespace(
        epoch=float(global_step),
        global_step=global_step,
        max_steps=global_step,
        logging_steps=10,
        eval_steps=500,
        save_steps=500,
        train_batch_size=None,
        num_train_epochs=float(global_step),
        num_input_tokens_seen=0,
        total_flos=0,
        best_metric=None,
        best_global_step=None,
        best_model_checkpoint=None,
        is_local_process_zero=True,
        is_world_process_zero=True,
        is_hyper_param_search=False,
        trial_name=None,
        trial_params=None,
        stateful_callbacks={},
        episode=global_step * 16,
        rewbuffer=deque([[[0.25]]], maxlen=100),
        lenbuffer=deque([[24.0]], maxlen=100),
        cur_reward_sum=torch.zeros(16, 1),
        cur_episode_length=torch.zeros(16),
        tot_timesteps=global_step * 16 * 24,
        tot_time=1.25,
        eval_step=0,
        eval_render_step=0,
    )
    return {
        "policy_state_dict": policy_state,
        "value_state_dict": value_state,
        "optimizer_state_dict": (
            _optimizer_state_for_residuals(policy_state, value_state, global_step)
            if optimizer_state is None
            else optimizer_state
        ),
        "lr_scheduler_state_dict": {
            "last_epoch": global_step,
            "base_lrs": [1e-5, 1e-5],
            "_last_lr": [2e-5, 3e-5],
        },
        "env_state_dict": {"motion_lib": {"sample": 0}},
        "state": state,
        "args": SimpleNamespace(learning_rate=1e-5),
    }


def _make_hf_grouped_optimizer(policy, value, *, learning_rates=(2e-5, 3e-5)):
    selected = tuple(
        [
            (name, parameter)
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        ]
        + [
            (f"value.{name}", parameter)
            for name, parameter in value.named_parameters()
            if parameter.requires_grad
        ]
    )
    weights = [parameter for name, parameter in selected if name.endswith(".weight")]
    biases = [parameter for name, parameter in selected if name.endswith(".bias")]
    assert len(weights) == len(biases) == 6
    return torch.optim.AdamW(
        [
            {"params": weights, "lr": learning_rates[0], "weight_decay": 0.0},
            {"params": biases, "lr": learning_rates[1], "weight_decay": 0.0},
        ]
    )


class _ResumeScheduler:
    def __init__(self):
        self.loaded = {
            "last_epoch": 0,
            "base_lrs": [9e-5, 9e-5],
            "_last_lr": [9e-5, 9e-5],
        }

    def load_state_dict(self, state):
        self.loaded = copy.deepcopy(state)

    def state_dict(self):
        return self.loaded


class _ResumeEnvironment:
    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.load_calls = 0

    def load_env_state_dict(self, state):
        self.load_calls += 1
        self.state = copy.deepcopy(state)

    def get_env_state_dict(self):
        return self.state


class _ResumeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model


def _make_resume_trainer_boundaries(policy, value, *, env_state=None):
    configure_motion_compliance_finetune_stage(policy, value, stage="residual_only")
    optimizer = _make_hf_grouped_optimizer(policy, value)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter] = {
                "step": torch.tensor(5.0),
                "exp_avg": torch.ones_like(parameter),
                "exp_avg_sq": torch.full_like(parameter, 2.0),
            }
    checkpoint = _resume_checkpoint(
        policy,
        value,
        optimizer_state=optimizer.state_dict(),
    )
    if env_state is not None:
        checkpoint["env_state_dict"] = copy.deepcopy(env_state)
    for group in optimizer.param_groups:
        group["lr"] = 9e-5
    optimizer.state.clear()

    trainer = object.__new__(MotionCompliancePPOTrainer)
    trainer.accelerator = _ResumeAccelerator()
    trainer.model = SimpleNamespace(policy=policy, value_model=value)
    trainer.optimizer = optimizer
    trainer.lr_scheduler = _ResumeScheduler()
    trainer.env = _ResumeEnvironment(checkpoint["env_state_dict"])
    trainer.args = SimpleNamespace(learning_rate=0.1)
    live_state = copy.deepcopy(checkpoint["state"])
    live_state.log_history = []
    live_state.epoch = None
    live_state.global_step = 0
    live_state.num_train_epochs = 0
    live_state.rewbuffer = deque(maxlen=100)
    live_state.lenbuffer = deque(maxlen=100)
    live_state.cur_reward_sum = torch.zeros(16, 1)
    live_state.cur_episode_length = torch.zeros(16)
    live_state.tot_timesteps = 0
    live_state.tot_time = 0
    trainer.state = live_state
    trainer.cur_reward_sum = live_state.cur_reward_sum
    trainer.cur_episode_length = live_state.cur_episode_length
    return trainer, checkpoint


def test_residual_context_widths_do_not_expand_release_models():
    assert OFFICIAL_ACTOR_INPUT_WIDTH == 994
    assert OFFICIAL_CRITIC_INPUT_WIDTH == 1645
    assert action_residual_context_width() == 997
    assert {sites: compliance_privileged_width(sites) for sites in (1, 2, 5)} == {
        1: 5,
        2: 9,
        5: 21,
    }
    assert {sites: value_residual_context_width(sites) for sites in (1, 2, 5)} == {
        1: 1653,
        2: 1657,
        5: 1669,
    }
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            compliance_privileged_width(invalid)


def test_synthetic_initialization_preserves_all_base_bytes_and_adds_only_residuals(
    initialized_bundle,
):
    _, _, source, initialized, report = initialized_bundle
    audit_residual_init_checkpoint(initialized)
    assert report.release_actor_input_width == 994
    assert report.action_residual_context_width == 997
    assert report.release_critic_input_width == 1645
    assert report.value_residual_context_width == 1657
    assert report.frozen_policy_tensor_count == 55
    assert report.frozen_value_tensor_count == 17
    assert initialized["optimizer_state_dict"] is None
    assert initialized["lr_scheduler_state_dict"] is None
    assert initialized["env_state_dict"] is None
    assert initialized["state"].global_step == 0

    policy_shapes, value_shapes = expected_residual_shapes(2)
    assert set(initialized["policy_state_dict"]) == set(source["policy_state_dict"]) | set(
        policy_shapes
    )
    assert set(initialized["value_state_dict"]) == set(source["value_state_dict"]) | set(
        value_shapes
    )
    assert initialized["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY].shape == (
        OFFICIAL_INPUT_HIDDEN_WIDTH,
        994,
    )
    assert initialized["value_state_dict"][CRITIC_INPUT_WEIGHT_KEY].shape == (
        OFFICIAL_INPUT_HIDDEN_WIDTH,
        1645,
    )
    assert initialized["value_state_dict"][CRITIC_RUNNING_MEAN_KEY].shape == (1645,)
    for group in ("policy_state_dict", "value_state_dict"):
        for key, source_tensor in source[group].items():
            assert tensor_bytes_equal(source_tensor, initialized[group][key])


def test_initialization_rejects_unpinned_or_nonresidual_target_schema(initialized_bundle):
    policy, value, source, _, _ = initialized_bundle
    kwargs = {
        "source_sha256": OFFICIAL_SONIC_RELEASE_SHA256,
        "target_policy_state": policy.state_dict(),
        "target_value_state": value.state_dict(),
    }
    with pytest.raises(ValueError, match="audited official"):
        initialize_motion_compliance_checkpoint(
            source,
            **{**kwargs, "source_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="expected_source_step"):
        initialize_motion_compliance_checkpoint(source, **kwargs, expected_source_step=41549)
    with pytest.raises(ValueError, match="source_revision"):
        initialize_motion_compliance_checkpoint(source, **kwargs, source_revision="main")
    bad_target = dict(policy.state_dict())
    bad_target["unrelated_new_tensor"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected_residual"):
        initialize_motion_compliance_checkpoint(
            source,
            **{**kwargs, "target_policy_state": bad_target},
        )
    wrong_dtype = dict(policy.state_dict())
    residual_key = next(iter(expected_residual_shapes(2)[0]))
    wrong_dtype[residual_key] = wrong_dtype[residual_key].double()
    with pytest.raises(ValueError, match="torch.float32"):
        initialize_motion_compliance_checkpoint(
            source,
            **{**kwargs, "target_policy_state": wrong_dtype},
        )
    legacy = dict(source)
    legacy["motion_compliance_migration"] = {"schema_version": 1}
    with pytest.raises(ValueError, match="legacy expanded"):
        initialize_motion_compliance_checkpoint(legacy, **kwargs)


def test_residual_init_and_normal_resume_model_loading_are_strict(initialized_bundle):
    policy, value, _, initialized, _ = initialized_bundle
    report = strict_load_policy_value_state(policy, value, initialized, resume=False)
    assert report.strict and report.residual_init
    with pytest.raises(ValueError, match="cannot be used with resume=true"):
        strict_load_policy_value_state(policy, value, initialized, resume=True)

    resume_checkpoint = _resume_checkpoint(policy, value)
    report = strict_load_policy_value_state(policy, value, resume_checkpoint, resume=True)
    assert report.strict and not report.residual_init
    broken = dict(resume_checkpoint)
    broken["policy_state_dict"] = dict(resume_checkpoint["policy_state_dict"])
    broken["policy_state_dict"].pop(next(iter(expected_residual_shapes(2)[0])))
    with pytest.raises(ValueError, match="strict policy keys differ"):
        strict_load_policy_value_state(policy, value, broken, resume=True)


def test_strict_resume_requires_every_nonempty_training_state(initialized_bundle):
    policy, value, _, initialized, _ = initialized_bundle
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
    with pytest.raises(ValueError, match="cannot be resumed"):
        validate_strict_resume_payload(initialized)

    extra_root = dict(checkpoint)
    extra_root["unexpected"] = True
    with pytest.raises(ValueError, match="checkpoint keys differ"):
        validate_strict_resume_payload(extra_root)
    invalid_args = dict(checkpoint)
    invalid_args["args"] = SimpleNamespace(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="args.learning_rate"):
        validate_strict_resume_payload(invalid_args)
    invalid_scheduler = dict(checkpoint)
    invalid_scheduler["lr_scheduler_state_dict"] = {
        **checkpoint["lr_scheduler_state_dict"],
        "_last_lr": [float("nan")],
    }
    with pytest.raises(ValueError, match="scheduler _last_lr"):
        validate_strict_resume_payload(invalid_scheduler)
    invalid_env = dict(checkpoint)
    invalid_env["env_state_dict"] = {"wrong": {}}
    with pytest.raises(ValueError, match="motion_lib"):
        validate_strict_resume_payload(invalid_env)
    invalid_slot = dict(checkpoint)
    invalid_slot["optimizer_state_dict"] = {
        **checkpoint["optimizer_state_dict"],
        "state": dict(checkpoint["optimizer_state_dict"]["state"]),
    }
    first_slot = next(iter(invalid_slot["optimizer_state_dict"]["state"]))
    invalid_slot["optimizer_state_dict"]["state"][first_slot] = {
        **invalid_slot["optimizer_state_dict"]["state"][first_slot],
        "exp_avg": torch.tensor(float("inf")),
    }
    with pytest.raises(ValueError, match="finite tensor exp_avg"):
        validate_strict_resume_payload(invalid_slot)


def test_strict_resume_rejects_optimizer_group_and_slot_semantics(initialized_bundle):
    policy, value, _, _, _ = initialized_bundle
    checkpoint = _resume_checkpoint(policy, value)

    def invalid_optimizer():
        broken = dict(checkpoint)
        broken["optimizer_state_dict"] = copy.deepcopy(
            checkpoint["optimizer_state_dict"]
        )
        return broken

    invalid_group_values = (
        ("lr", float("nan"), "finite numeric"),
        ("weight_decay", -0.1, "non-negative"),
        ("betas", (1.0, 0.999), "0 <="),
        ("eps", 0.0, "positive"),
        ("amsgrad", True, "amsgrad=false"),
    )
    for field, value_to_set, match in invalid_group_values:
        broken = invalid_optimizer()
        broken["optimizer_state_dict"]["param_groups"][0][field] = value_to_set
        with pytest.raises(ValueError, match=match):
            validate_strict_resume_payload(broken)

    broken = invalid_optimizer()
    del broken["optimizer_state_dict"]["param_groups"][0]["betas"]
    with pytest.raises(ValueError, match="lacks"):
        validate_strict_resume_payload(broken)

    first_id = checkpoint["optimizer_state_dict"]["param_groups"][0]["params"][0]
    slot_variants = (
        ({"unexpected": torch.tensor(1.0)}, "slot keys"),
        ({"step": None}, "slot keys"),
        ({"step": torch.tensor(5, dtype=torch.int64)}, "float32 scalar"),
        ({"step": torch.tensor(0.0)}, "positive integer-valued"),
    )
    for updates, match in slot_variants:
        broken = invalid_optimizer()
        slot = broken["optimizer_state_dict"]["state"][first_id]
        if updates.get("step", object()) is None:
            del slot["step"]
        else:
            slot.update(updates)
        with pytest.raises(ValueError, match=match):
            validate_strict_resume_payload(broken)


def test_strict_resume_preflights_both_models_before_mutation(initialized_bundle):
    policy, value, _, initialized, _ = initialized_bundle
    strict_load_policy_value_state(policy, value, initialized, resume=False)
    checkpoint = _resume_checkpoint(policy, value)
    checkpoint["policy_state_dict"]["std"].add_(0.25)
    checkpoint["value_state_dict"].pop("critic_module.module.0.bias")
    policy_before = _clone_state(policy.state_dict())
    value_before = _clone_state(value.state_dict())
    with pytest.raises(ValueError, match="strict value keys differ"):
        strict_load_policy_value_state(policy, value, checkpoint, resume=True)
    assert all(
        tensor_bytes_equal(policy_before[key], tensor)
        for key, tensor in policy.state_dict().items()
    )
    assert all(
        tensor_bytes_equal(value_before[key], tensor)
        for key, tensor in value.state_dict().items()
    )


def test_residual_only_freeze_optimizer_gradients_and_two_step_changes(initialized_bundle):
    policy, value, _, initialized, _ = initialized_bundle
    strict_load_policy_value_state(policy, value, initialized, resume=False)
    report = configure_motion_compliance_finetune_stage(
        policy,
        value,
        stage="residual_only",
    )
    assert len(report.trainable_policy_names) == 6
    assert len(report.trainable_value_names) == 6
    assert all(ACTION_RESIDUAL_PREFIX in name for name in report.trainable_policy_names)
    assert all(name.startswith(VALUE_RESIDUAL_PREFIX) for name in report.trainable_value_names)
    assert "std" in report.frozen_policy_names
    assert "critic_module.module.0.weight" in report.frozen_value_names
    assert "running_mean_std.running_mean" not in dict(value.named_parameters())

    trainable = [
        parameter
        for module in (policy, value)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = _make_hf_grouped_optimizer(policy, value, learning_rates=(1e-3, 1e-3))
    validate_optimizer_parameter_set(optimizer, policy, value)
    validate_optimizer_parameter_order(optimizer, policy, value)
    assert len(residual_parameter_names(policy, value)) == 12
    initial = [parameter.detach().clone() for parameter in trainable]
    assert len(initial) == 12
    torch.manual_seed(7)
    batch_size = 4
    rollout_steps = 24
    action_context = torch.randn(
        batch_size,
        rollout_steps,
        action_residual_context_width(),
    )
    value_context = torch.randn(
        batch_size,
        rollout_steps,
        value_residual_context_width(2),
    )
    action_target = torch.randn(batch_size, rollout_steps, 29)
    value_target = torch.randn(batch_size, rollout_steps, 1)
    assert action_context.shape == (4, 24, 997)
    assert value_context.shape == (4, 24, 1657)
    assert torch.count_nonzero(action_context[:, 1:] - action_context[:, :1]).item() > 0
    assert torch.count_nonzero(value_context[:, 1:] - value_context[:, :1]).item() > 0
    for update in range(2):
        optimizer.zero_grad()
        action = policy.actor_module.motion_compliance_action_residual(action_context)
        predicted_value = value.motion_compliance_value_residual(value_context)
        assert action.shape == (4, 24, 29)
        assert predicted_value.shape == (4, 24, 1)
        if update > 0:
            assert torch.count_nonzero(action[:, 1:] - action[:, :1]).item() > 0
            assert (
                torch.count_nonzero(predicted_value[:, 1:] - predicted_value[:, :1]).item()
                > 0
            )
        loss = (action - action_target).square().mean() + (
            predicted_value - value_target
        ).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradient_names = validate_residual_gradients(
            policy,
            value,
            require_nonzero=update > 0,
        )
        assert len(gradient_names) == 12
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in trainable
        )
        if update > 0:
            assert all(
                torch.count_nonzero(parameter.grad).item() > 0
                for parameter in trainable
            )
        optimizer.step()
    assert all(
        not tensor_bytes_equal(before, after)
        for before, after in zip(initial, trainable, strict=True)
    )

    frozen = next(parameter for parameter in policy.parameters() if not parameter.requires_grad)
    bad_optimizer = torch.optim.Adam([*trainable, frozen], lr=1e-3)
    with pytest.raises(RuntimeError, match="ownership differs"):
        validate_optimizer_parameter_set(bad_optimizer, policy, value)
    with pytest.raises(ValueError, match="residual_only"):
        configure_motion_compliance_finetune_stage(policy, value, stage="full")


def test_optimizer_resume_moment_shapes_are_preflighted_before_load(initialized_bundle):
    policy, value, _, initialized, _ = initialized_bundle
    strict_load_policy_value_state(policy, value, initialized, resume=False)
    configure_motion_compliance_finetune_stage(policy, value, stage="residual_only")
    optimizer = _make_hf_grouped_optimizer(policy, value)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter] = {
                "step": torch.tensor(5.0),
                "exp_avg": torch.ones_like(parameter),
                "exp_avg_sq": torch.ones_like(parameter),
            }
    saved = optimizer.state_dict()
    bad = {
        "state": {
            key: dict(value) for key, value in saved["state"].items()
        },
        "param_groups": [dict(group) for group in saved["param_groups"]],
    }
    first_parameter_id = bad["param_groups"][0]["params"][0]
    bad["state"][first_parameter_id]["exp_avg"] = torch.zeros(1)
    live_before = optimizer.state_dict()
    with pytest.raises(ValueError, match="shape/dtype differs"):
        preflight_optimizer_resume_state(optimizer, bad, policy, value)
    assert optimizer.state_dict()["param_groups"] == live_before["param_groups"]
    assert all(
        torch.equal(
            optimizer.state_dict()["state"][key]["exp_avg"],
            live_before["state"][key]["exp_avg"],
        )
        for key in live_before["state"]
    )

    bad_dtype = copy.deepcopy(saved)
    first_parameter_id = bad_dtype["param_groups"][0]["params"][0]
    bad_dtype["state"][first_parameter_id]["exp_avg"] = bad_dtype["state"][
        first_parameter_id
    ]["exp_avg"].double()
    with pytest.raises(ValueError, match="shape/dtype differs"):
        preflight_optimizer_resume_state(optimizer, bad_dtype, policy, value)
    _assert_nested_exact(optimizer.state_dict(), live_before)

    bad_order = copy.deepcopy(saved)
    parameters = bad_order["param_groups"][0]["params"]
    parameters[1], parameters[4] = parameters[4], parameters[1]
    assert (
        bad_order["state"][parameters[1]]["exp_avg"].shape
        == bad_order["state"][parameters[4]]["exp_avg"].shape
    )
    with pytest.raises(ValueError, match="parameter order differs"):
        preflight_optimizer_resume_state(optimizer, bad_order, policy, value)
    _assert_nested_exact(optimizer.state_dict(), live_before)


def test_real_ppo_temporal_contract_is_b_by_24():
    batch_size = 4
    rollout_steps = 24
    mb = {
        "mb_obs_dict": {
            "actor_obs": torch.zeros(batch_size, rollout_steps, 930),
            "critic_obs": torch.zeros(batch_size, rollout_steps, 1645),
            "motion_compliance_condition": torch.zeros(batch_size, rollout_steps, 3),
            "motion_compliance_privileged": torch.zeros(batch_size, rollout_steps, 9),
            "tokenizer": torch.zeros(batch_size, rollout_steps, 10, 64),
        },
        "mb_actions": torch.zeros(batch_size, rollout_steps, 29),
        "episode_attnmask": torch.zeros(
            batch_size,
            rollout_steps,
            rollout_steps,
            dtype=torch.bool,
        ),
    }
    assert validate_motion_compliance_ppo_batch(mb) == (batch_size, rollout_steps)
    outputs = {
        "policy_results": {
            "action_mean": torch.zeros(batch_size, rollout_steps, 29),
            "action_std": torch.ones(batch_size, rollout_steps, 29),
            "logprobs": torch.zeros(batch_size, rollout_steps),
            "entropy": torch.zeros(batch_size, rollout_steps),
        },
        "value_results": torch.zeros(batch_size, rollout_steps, 1),
    }
    validate_motion_compliance_ppo_outputs(
        outputs,
        batch_size=batch_size,
        rollout_steps=rollout_steps,
    )
    mb["mb_actions"] = torch.zeros(batch_size, 23, 29)
    with pytest.raises(ValueError, match="mb_actions shape"):
        validate_motion_compliance_ppo_batch(mb)
    with pytest.raises(ValueError, match="rollout_steps"):
        validate_motion_compliance_ppo_batch(mb, rollout_steps=23)


def test_frozen_release_std_clamp_is_non_mutating_and_optimizer_excluded(monkeypatch):
    official = audit_module.load_trl_checkpoint(OFFICIAL_CHECKPOINT, map_location="cpu")
    official_std = official["policy_state_dict"]["std"].detach().clone()
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
    actor.actor_module = _ConstantActionBackbone()
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

    value = _FakeValue()
    configure_motion_compliance_finetune_stage(actor, value, stage="residual_only")
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
    actor_inputs = {
        "actor_obs": torch.zeros(2, 4),
        "tokenizer": torch.zeros(2, 1),
        "motion_compliance_condition": torch.zeros(2, 3),
    }
    for _ in range(3):
        actor.update_distribution(actor_inputs)
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


def test_custom_trainer_strict_resume_restores_all_state(tmp_path, initialized_bundle):
    policy, value, _, initialized, _ = initialized_bundle
    trainer, checkpoint = _make_resume_trainer_boundaries(policy, value)
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(checkpoint, checkpoint_path)

    init_path = tmp_path / "init.pt"
    torch.save(initialized, init_path)
    init_loaded = MotionCompliancePPOTrainer.load_checkpoint(
        trainer,
        init_path,
        resume=False,
    )
    assert MOTION_COMPLIANCE_INITIALIZATION_KEY in init_loaded
    assert trainer.optimizer.state == {}
    assert trainer.lr_scheduler.loaded["last_epoch"] == 0
    assert trainer.env.load_calls == 0
    assert trainer.state.global_step == 0

    loaded = MotionCompliancePPOTrainer.load_checkpoint(
        trainer,
        checkpoint_path,
        resume=True,
    )
    assert loaded["state"].global_step == 5
    assert trainer.state.global_step == 5
    assert trainer.args.learning_rate == 1e-5
    assert [group["lr"] for group in trainer.optimizer.param_groups] == [2e-5, 3e-5]
    assert trainer.lr_scheduler.loaded == checkpoint["lr_scheduler_state_dict"]
    assert trainer.env.load_calls == 1
    _assert_nested_exact(trainer.env.state, checkpoint["env_state_dict"])
    for key, value in checkpoint["state"].__dict__.items():
        if key not in {
            "stateful_callbacks",
            "is_local_process_zero",
            "is_world_process_zero",
        }:
            _assert_nested_exact(getattr(trainer.state, key), value)


def test_resume_preflight_failures_leave_every_live_boundary_unchanged(
    monkeypatch,
    initialized_bundle,
):
    policy, value, _, _, _ = initialized_bundle
    env_state = {
        "motion_lib": {
            "adp_samp_num_episodes": torch.zeros(3),
            "adp_samp_num_failures": torch.ones(3),
        }
    }
    trainer, checkpoint = _make_resume_trainer_boundaries(
        policy,
        value,
        env_state=env_state,
    )

    def candidate():
        result = dict(checkpoint)
        result["policy_state_dict"] = dict(checkpoint["policy_state_dict"])
        result["policy_state_dict"]["std"] = (
            checkpoint["policy_state_dict"]["std"].clone() + 0.25
        )
        return result

    cases = []
    for name, invalid_motion_lib in (
        (
            "env_keys",
            {
                "wrong": torch.zeros(3),
                "adp_samp_num_failures": torch.ones(3),
            },
        ),
        (
            "env_shape",
            {
                "adp_samp_num_episodes": torch.zeros(4),
                "adp_samp_num_failures": torch.ones(3),
            },
        ),
        (
            "env_dtype",
            {
                "adp_samp_num_episodes": torch.zeros(3, dtype=torch.float64),
                "adp_samp_num_failures": torch.ones(3),
            },
        ),
        (
            "env_nonfinite",
            {
                "adp_samp_num_episodes": torch.full((3,), float("nan")),
                "adp_samp_num_failures": torch.ones(3),
            },
        ),
    ):
        broken = candidate()
        broken["env_state_dict"] = {"motion_lib": invalid_motion_lib}
        cases.append((name, broken, "environment"))

    broken = candidate()
    broken["optimizer_state_dict"] = copy.deepcopy(checkpoint["optimizer_state_dict"])
    broken["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
    cases.append(("optimizer_nonfinite", broken, "finite numeric"))

    broken = candidate()
    broken["optimizer_state_dict"] = copy.deepcopy(checkpoint["optimizer_state_dict"])
    broken["optimizer_state_dict"]["param_groups"][0]["maximize"] = True
    cases.append(("optimizer_fixed_flag", broken, "fixed hyperparameter maximize"))

    broken = candidate()
    broken["optimizer_state_dict"] = copy.deepcopy(checkpoint["optimizer_state_dict"])
    parameter_ids = broken["optimizer_state_dict"]["param_groups"][0]["params"]
    parameter_ids[1], parameter_ids[4] = parameter_ids[4], parameter_ids[1]
    cases.append(("optimizer_same_shape_swap", broken, "parameter order differs"))

    broken = candidate()
    broken["state"] = copy.deepcopy(checkpoint["state"])
    broken["state"].unexpected = 1
    cases.append(("trainer_extra", broken, "trainer state keys differ"))

    broken = candidate()
    broken["state"] = copy.deepcopy(checkpoint["state"])
    del broken["state"].__dict__["eval_render_step"]
    cases.append(("trainer_missing", broken, "trainer state keys differ"))

    broken = candidate()
    broken["state"] = copy.deepcopy(checkpoint["state"])
    broken["state"].episode = 1.5
    cases.append(("trainer_scalar", broken, "state.episode"))

    broken = candidate()
    broken["state"] = copy.deepcopy(checkpoint["state"])
    broken["state"].cur_reward_sum = torch.zeros(15, 1)
    broken["state"].cur_episode_length = torch.zeros(15)
    cases.append(("trainer_tensor_shape", broken, "trainer_state.cur_"))

    active_checkpoint = {"value": None}

    def fake_load(*_args, **_kwargs):
        return active_checkpoint["value"]

    monkeypatch.setattr(trainer_module, "load_trl_checkpoint", fake_load)
    for name, broken, match in cases:
        policy_before = _clone_state(policy.state_dict())
        value_before = _clone_state(value.state_dict())
        optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
        scheduler_before = copy.deepcopy(trainer.lr_scheduler.state_dict())
        env_before = copy.deepcopy(trainer.env.get_env_state_dict())
        state_before = copy.deepcopy(trainer.state.__dict__)
        args_learning_rate_before = trainer.args.learning_rate
        active_checkpoint["value"] = broken
        with pytest.raises(ValueError, match=match):
            MotionCompliancePPOTrainer.load_checkpoint(
                trainer,
                f"unused-{name}.pt",
                resume=True,
            )
        _assert_nested_exact(policy.state_dict(), policy_before)
        _assert_nested_exact(value.state_dict(), value_before)
        _assert_nested_exact(trainer.optimizer.state_dict(), optimizer_before)
        _assert_nested_exact(trainer.lr_scheduler.state_dict(), scheduler_before)
        _assert_nested_exact(trainer.env.get_env_state_dict(), env_before)
        _assert_nested_exact(trainer.state.__dict__, state_before)
        assert trainer.state.cur_reward_sum is trainer.cur_reward_sum
        assert trainer.state.cur_episode_length is trainer.cur_episode_length
        assert trainer.args.learning_rate == args_learning_rate_before
        assert trainer.env.load_calls == 0


def test_exposure_callback_writes_each_step_and_finalizes_per_site(tmp_path):
    runs_root = tmp_path / "runs"
    output_path = runs_root / "case" / "exposure.json"
    callback = MotionComplianceExposureCallback(str(output_path), runs_root=str(runs_root))
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
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    state = SimpleNamespace(global_step=1, max_steps=2)
    control = SimpleNamespace()
    logs = {
        "loss/policy_avg": 0.1,
        "loss/value_avg": torch.tensor(0.2),
        "collection_time": 0.3,
        "learn_time": 0.4,
        "fps": 100.0,
    }
    callback.on_log(None, state, control, logs=logs)
    callback.on_step_end(None, state, control, env=env)
    state.global_step = 2
    callback.on_log(None, state, control, logs=logs)
    callback.on_step_end(None, state, control, env=env)
    final = json.loads(output_path.read_text(encoding="utf-8"))
    assert callback._finalized
    assert final["active_site_samples_by_index"] == [4, 4]
    assert final["nonzero_force_site_samples_by_index"] == [4, 4]

    command.state.active_site_mask[:, 1] = False
    command.state.site_force_world[:, 1, 0] = 1.0
    stale = MotionComplianceExposureCallback(
        str(runs_root / "stale" / "exposure.json"),
        runs_root=str(runs_root),
    )
    with pytest.raises(RuntimeError, match="persisted"):
        stale.on_step_end(None, state, control, env=env)


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


def test_finetune_hydra_config_resolves_to_residual_only_owned_workflow():
    config = _compose_finetune_config()
    assert config.base_dir == str(RUNS_ROOT)
    assert Path(config.experiment_dir).is_relative_to(RUNS_ROOT)
    assert Path(config.checkpoint) == OFFICIAL_CHECKPOINT
    assert config.trainer._target_ == MOTION_COMPLIANCE_TRAINER_TARGET
    assert config.algo.config.actor._target_ == MOTION_COMPLIANCE_ACTOR_TARGET
    assert config.motion_compliance_finetune.stage == "residual_only"
    assert "trainable_decoder_names" not in config.motion_compliance_finetune
    assert config.algo.config.num_steps_per_env == 24
    assert config.algo.config.num_learning_epochs == 5
    assert config.algo.config.num_mini_batches == 4
    assert "weight_decay" not in config.algo.trl
    assert PPOConfig.__dataclass_fields__["weight_decay"].default == 0.0
    assert config.manager_env.config.get("use_symmetry", False) is False
    assert config.algo.config.freeze_noise_std is True
    assert list(
        config.algo.config.actor.backbone.motion_compliance_residual_hidden_dims
    ) == [256, 256]
    assert list(config.algo.config.critic.motion_compliance_residual_hidden_dims) == [256, 256]
    assert config.num_envs == 16
    assert config.algo.config.num_learning_iterations == 5
    assert config.use_wandb is False
    assert config.callbacks.model_save.save_last_frequency == 5
    validate_motion_compliance_workflow_config(config)

    resume_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    resume_config.resume = True
    resume_config.motion_compliance_checkpoint_initialization.enabled = False
    resume_config.checkpoint = str(RUNS_ROOT / "residual_gpu_smoke" / "last.pt")
    resume_config.experiment_dir = str(RUNS_ROOT / "residual_gpu_resume")
    resume_config.motion_compliance_finetune.resume_output_dir = str(
        RUNS_ROOT / "residual_gpu_resume"
    )
    resume_config.algo.config.num_learning_iterations = 1
    resume_config.callbacks.model_save.save_last_frequency = 1
    validate_motion_compliance_workflow_config(resume_config)

    invalid_smoke_values = {
        "motion_compliance_finetune.stage": "full",
        "algo.config.num_steps_per_env": 23,
        "algo.config.num_learning_epochs": 4,
        "algo.config.num_mini_batches": 2,
        "manager_env.config.use_symmetry": True,
        "algo.config.freeze_noise_std": False,
        "algo.config.actor.backbone._target_": "torch.nn.Identity",
        "algo.config.critic._target_": "torch.nn.Identity",
        "algo.config.actor.backbone.motion_compliance_action_delta_limit": 0.5,
        "algo.config.actor.backbone.motion_compliance_residual_hidden_dims": [128, 128],
        "algo.config.critic.motion_compliance_residual_hidden_dims": [128, 128],
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

    forbidden_decoder = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    forbidden_decoder.motion_compliance_finetune.trainable_decoder_names = ["g1_dyn"]
    with pytest.raises(ValueError, match="forbids trainable_decoder_names"):
        validate_motion_compliance_workflow_config(forbidden_decoder)


def test_compliance_resume_routes_step6_to_separate_directory():
    from gear_sonic.train_agent_trl import resolve_checkpoint_and_experiment_dir

    step5 = RUNS_ROOT / "phase4_residual_gpu_smoke_tensordict_fix" / "last.pt"
    resume_output = RUNS_ROOT / "phase4_residual_gpu_resume_tensordict_fix"
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


def test_post_train_audit_uses_independent_official_and_init_files(
    monkeypatch,
    initialized_bundle,
):
    policy, value, source, initialized, _ = initialized_bundle
    trained = _resume_checkpoint(policy, value)
    trained["policy_state_dict"] = _clone_state(initialized["policy_state_dict"])
    trained["value_state_dict"] = _clone_state(initialized["value_state_dict"])
    policy_shapes, value_shapes = expected_residual_shapes(2)
    for key in policy_shapes:
        trained["policy_state_dict"][key].add_(0.01)
    for key in value_shapes:
        trained["value_state_dict"][key].add_(0.01)
    trained["optimizer_state_dict"] = _optimizer_state_for_residuals(
        trained["policy_state_dict"],
        trained["value_state_dict"],
    )
    monkeypatch.setattr(audit_module, "validate_checkpoint_sha256", lambda *args: "ok")

    def fake_load(path, map_location="cpu"):
        del map_location
        path = str(path)
        if path.endswith("official.pt"):
            return source
        if path.endswith("init.pt"):
            return initialized
        return trained

    monkeypatch.setattr(audit_module, "load_trl_checkpoint", fake_load)
    report = audit_trained_motion_compliance_checkpoint(
        "/tmp/official.pt",
        RUNS_ROOT / "unit" / "init.pt",
        RUNS_ROOT / "unit" / "last.pt",
        expected_global_step=5,
        num_sites=2,
    )
    assert len(report.changed_policy_residual_names) == 6
    assert len(report.changed_value_residual_names) == 6
    assert report.frozen_policy_tensor_count == 55
    assert report.frozen_value_tensor_count == 17
    assert report.optimizer_slot_count == 12

    first_slot = next(iter(trained["optimizer_state_dict"]["state"].values()))
    saved_exp_avg_sq = first_slot["exp_avg_sq"].clone()
    first_slot["exp_avg_sq"].zero_()
    with pytest.raises(ValueError, match="zero exp_avg_sq moment"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "init.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
        )
    first_slot["exp_avg_sq"].copy_(saved_exp_avg_sq)

    first_group = trained["optimizer_state_dict"]["param_groups"][0]
    first_group["params"][1], first_group["params"][4] = (
        first_group["params"][4],
        first_group["params"][1],
    )
    slots = trained["optimizer_state_dict"]["state"]
    assert slots[1]["exp_avg"].shape == slots[4]["exp_avg"].shape
    with pytest.raises(ValueError, match="optimizer parameter order differs"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "init.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
        )
    first_group["params"][1], first_group["params"][4] = (
        first_group["params"][4],
        first_group["params"][1],
    )

    first_group["maximize"] = True
    with pytest.raises(ValueError, match="fixed AdamW flag/value differs"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "init.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
        )
    first_group["maximize"] = False

    trained["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY][0, 0] += 1.0
    with pytest.raises(ValueError, match="frozen trained policy tensor changed"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "init.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
        )
    trained["policy_state_dict"][ACTOR_INPUT_WEIGHT_KEY][0, 0] -= 1.0
    first_residual = next(iter(policy_shapes))
    trained["policy_state_dict"][first_residual].copy_(
        initialized["policy_state_dict"][first_residual]
    )
    with pytest.raises(ValueError, match="did not change"):
        audit_trained_motion_compliance_checkpoint(
            "/tmp/official.pt",
            RUNS_ROOT / "unit" / "init.pt",
            RUNS_ROOT / "unit" / "last.pt",
            expected_global_step=5,
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
