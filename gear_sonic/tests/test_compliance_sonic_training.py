"""Phase-3 contracts for additive SONIC observations, rewards, and config."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from gear_sonic.compliance_control.adapters.sonic.contracts import (
    condition_from_command,
    current_endpoint_position_errors_from_command,
    current_site_force_from_command,
    endpoint_position_errors_from_state,
    gated_mean_gaussian_reward,
    quaternion_error_magnitude_wxyz,
    select_yielded_site_reference,
    site_mask_from_command,
    threshold_from_command,
)
from gear_sonic.compliance_control.adapters.sonic.state import (
    ComplianceCommandState,
    ComplianceSamplingSpec,
)
from gear_sonic.compliance_control.core import hard_gate_residual
from gear_sonic.compliance_control.training import (
    MotionComplianceFrozenNoiseActor,
    MotionComplianceResidualCritic,
    MotionComplianceUniversalTokenModule,
    ZeroInitializedResidualMLP,
    motion_compliance_residual_parameters,
)
from gear_sonic.trl.modules.actor_critic_modules import Critic
from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule


ROOT = Path(__file__).parents[2]
CONFIG_DIR = str((ROOT / "gear_sonic" / "config").resolve())
ADAPTER_DIR = ROOT / "gear_sonic" / "compliance_control" / "adapters" / "sonic"
CONFIG_METADATA = {"_target_", "enable_corruption", "concatenate_terms"}


def _compose_release_pair():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        baseline = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                "num_envs=1",
            ],
        )
        compliance = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release_motion_compliance",
                "num_envs=1",
            ],
        )
    return baseline, compliance


def _term_names(group) -> list[str]:
    return [name for name in group.keys() if name not in CONFIG_METADATA]


def _class_assignment_order(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return [
        target.id
        for node in class_node.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == function_name
    )
    return "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])


def _tiny_base_module_params(*, temporal_input=None, temporal_output=None):
    params = {
        "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
        "input_dim": 0,
        "output_dim": 0,
        "module_config_dict": {
            "input_dim": [0],
            "output_dim": [0],
            "layer_config": {
                "type": "MLP",
                "hidden_dims": [8],
                "activation": "SiLU",
            },
        },
    }
    if temporal_input is not None:
        params["num_input_temporal_dims"] = temporal_input
    if temporal_output is not None:
        params["num_output_temporal_dims"] = temporal_output
    return params


def _build_tiny_residual_models():
    obs_dims = {
        "actor_obs": 930,
        "critic_obs": 1645,
        "motion_compliance_condition": 3,
        "motion_compliance_privileged": 9,
    }
    env_config = SimpleNamespace(
        robot=SimpleNamespace(algo_obs_dim_dict=obs_dims, actions_dim=29),
        obs=SimpleNamespace(
            group_obs_dims={
                "tokenizer": {
                    "g1_input": (10, 4),
                    "probe_feature": (1,),
                }
            },
            group_obs_names={"tokenizer": ["g1_input", "probe_feature"]},
        ),
    )
    algo_config = OmegaConf.create(
        {
            "init_noise_std": 0.1,
            "freeze_noise_std": True,
            "use_log_std": False,
            "use_clampped_std": True,
            "std_clamp_min": 0.001,
            "std_clamp_max": 0.5,
        }
    )
    backbone = OmegaConf.create(
        {
            "_target_": (
                "gear_sonic.compliance_control.training.residual_policy."
                "MotionComplianceUniversalTokenModule"
            ),
            "proprioception_features": ["actor_obs"],
            "num_fsq_levels": 32,
            "fsq_level_list": 32,
            "max_num_tokens": 2,
            "num_future_frames": 10,
            "encoder_sample_probs": {"g1": 1.0},
            "quantizer": None,
            "encoders": {
                "g1": {
                    "inputs": ["g1_input"],
                    "outputs": [],
                    "params": _tiny_base_module_params(
                        temporal_input=10,
                        temporal_output=2,
                    ),
                }
            },
            "decoders": {
                "g1_dyn": {
                    "inputs": ["token_flattened", "proprioception"],
                    "outputs": ["action"],
                    "conds": [],
                    "mask": [],
                    "has_temporal_dim": False,
                    "params": _tiny_base_module_params(),
                },
                # A non-g1 decoder that deliberately consumes proprioception;
                # its hook proves the wrapper never forwards 933 columns.
                "probe": {
                    "inputs": ["proprioception"],
                    "outputs": ["probe_feature"],
                    "conds": [],
                    "mask": [],
                    "has_temporal_dim": False,
                    "params": _tiny_base_module_params(),
                },
            },
            "aux_loss_func": {},
            "aux_loss_coef": {},
            "motion_compliance_residual_hidden_dims": [8, 8],
            "motion_compliance_action_delta_limit": 0.25,
        }
    )
    policy = MotionComplianceFrozenNoiseActor(
        env_config,
        algo_config,
        backbone,
        input_obs_dict=True,
        has_aux_loss=True,
    )
    critic_backbone = OmegaConf.create(
        {
            "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
            "process_output_dim": True,
            "module_config_dict": {
                "input_dim": ["critic_obs"],
                "output_dim": [1],
                "layer_config": {
                    "type": "MLP",
                    "hidden_dims": [8],
                    "activation": "SiLU",
                },
            },
        }
    )
    value = MotionComplianceResidualCritic(
        env_config,
        algo_config,
        critic_backbone,
        running_mean_std=True,
        motion_compliance_residual_hidden_dims=(8,),
    )
    return policy, value


def test_release_tokenizer_subtree_and_robot_motion_shapes_are_exactly_preserved():
    baseline, compliance = _compose_release_pair()

    baseline_tokenizer = OmegaConf.to_container(
        baseline.manager_env.observations.tokenizer,
        resolve=True,
    )
    compliance_tokenizer = OmegaConf.to_container(
        compliance.manager_env.observations.tokenizer,
        resolve=True,
    )
    assert compliance_tokenizer == baseline_tokenizer
    baseline_terminations = OmegaConf.to_container(
        baseline.manager_env.terminations,
        resolve=True,
    )
    compliance_terminations = OmegaConf.to_container(
        compliance.manager_env.terminations,
        resolve=True,
    )
    assert compliance_terminations == baseline_terminations

    baseline_inputs = list(baseline.algo.config.actor.backbone.encoders.g1.inputs)
    compliance_inputs = list(compliance.algo.config.actor.backbone.encoders.g1.inputs)
    assert compliance_inputs == baseline_inputs == [
        "command_multi_future_nonflat",
        "motion_anchor_ori_b_mf_nonflat",
    ]
    assert baseline.manager_env.commands.motion.num_future_frames == 10
    assert compliance.manager_env.commands.motion.num_future_frames == 10
    baseline_shapes = {
        "command_multi_future_nonflat": (10, 58),
        "motion_anchor_ori_b_mf_nonflat": (10, 6),
    }
    compliance_shapes = dict(baseline_shapes)
    assert compliance_shapes == baseline_shapes
    assert sum(shape[-1] for shape in compliance_shapes.values()) == 64
    assert "motion_compliance_condition" not in compliance_tokenizer


def test_release_policy_and_critic_stay_exact_while_compliance_groups_are_separate():
    baseline, compliance = _compose_release_pair()
    baseline_policy_terms = _term_names(baseline.manager_env.observations.policy)
    compliance_policy_terms = _term_names(compliance.manager_env.observations.policy)
    assert compliance_policy_terms == baseline_policy_terms
    assert OmegaConf.to_container(
        compliance.manager_env.observations.policy,
        resolve=True,
    ) == OmegaConf.to_container(
        baseline.manager_env.observations.policy,
        resolve=True,
    )
    assert OmegaConf.to_container(
        compliance.manager_env.observations.critic,
        resolve=True,
    ) == OmegaConf.to_container(
        baseline.manager_env.observations.critic,
        resolve=True,
    )
    for privileged_name in (
        "motion_compliance_threshold",
        "motion_compliance_site_force",
        "motion_compliance_site_mask",
    ):
        assert privileged_name not in compliance_policy_terms

    class ActorCommand:
        command = torch.tensor([[1.0, 12.0, 240.0], [0.0, 0.0, 0.0]])

        @property
        def state(self):
            raise AssertionError("actor condition must not read privileged command state")

    condition = condition_from_command(ActorCommand())
    assert condition.shape == (2, 3)
    assert _term_names(
        compliance.manager_env.observations.motion_compliance_condition
    ) == ["motion_compliance_condition"]
    condition_fields = _class_assignment_order(
        ADAPTER_DIR / "observation.py",
        "MotionComplianceConditionCfg",
    )
    assert condition_fields == ["motion_compliance_condition"]

    privileged_terms = _term_names(
        compliance.manager_env.observations.motion_compliance_privileged
    )
    assert privileged_terms == [
        "motion_compliance_threshold",
        "motion_compliance_site_force",
        "motion_compliance_site_mask",
    ]
    for num_sites in (1, 2, 5):
        state = ComplianceCommandState(
            3,
            num_sites,
            4,
            ComplianceSamplingSpec(enable_probability=1.0),
            seed=3,
        )
        state.reset()
        state.force_common_future[:, 0] = torch.arange(
            3 * num_sites * 3,
            dtype=state.dtype,
        ).reshape(3, num_sites, 3)
        state.force_common_future[:, 1:] = -999.0
        command = SimpleNamespace(command=state.condition, state=state, num_envs=3)
        critic_additions = (
            condition_from_command(command),
            threshold_from_command(command),
            current_site_force_from_command(command),
            site_mask_from_command(command),
        )
        expected_width = 3 + 1 + 3 * num_sites + num_sites
        assert sum(value.shape[-1] for value in critic_additions) == expected_width
        assert current_site_force_from_command(command).shape == (3, 3 * num_sites)
        assert not torch.any(current_site_force_from_command(command) == -999.0)
    assert condition.shape[-1] == 3
    assert 1 + 3 * 2 + 2 == 9


def test_zero_initialized_residual_has_mixed_batch_hard_gate_and_exact_ownership():
    torch.manual_seed(7)
    rng_before = torch.random.get_rng_state().clone()
    action_residual = ZeroInitializedResidualMLP(11, 4, hidden_dims=(8, 6))
    value_residual = ZeroInitializedResidualMLP(13, 1, hidden_dims=(7,))
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_module = nn.Module()
            self.actor_module.release = nn.Linear(5, 4)
            self.actor_module.motion_compliance_action_residual = action_residual
            self.std = nn.Parameter(torch.ones(4))

    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            self.release = nn.Linear(5, 1)
            self.motion_compliance_value_residual = value_residual

    policy = Policy()
    value = Value()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in value.parameters():
        parameter.requires_grad_(False)
    for parameter in action_residual.parameters():
        parameter.requires_grad_(True)
    for parameter in value_residual.parameters():
        parameter.requires_grad_(True)

    owned = motion_compliance_residual_parameters(policy, value)
    assert {id(parameter) for parameter in owned} == {
        id(parameter)
        for parameter in (*action_residual.parameters(), *value_residual.parameters())
    }
    optimizer = torch.optim.Adam(owned, lr=1.0e-3)
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == {id(parameter) for parameter in owned}

    action_context = torch.randn(2, 11)
    value_context = torch.randn(2, 13)
    base_action = torch.randn(2, 4)
    base_value = torch.randn(2, 1)
    enabled = torch.tensor([False, True])
    action_delta = action_residual(action_context)
    value_delta = value_residual(value_context)
    assert torch.equal(action_delta, torch.zeros_like(action_delta))
    assert torch.equal(value_delta, torch.zeros_like(value_delta))
    assert torch.equal(
        hard_gate_residual(base_action, action_delta, enabled),
        base_action,
    )
    assert torch.equal(
        hard_gate_residual(base_value, value_delta, enabled),
        base_value,
    )

    # A poisoned rejected row cannot contaminate hard-off output.  The enabled
    # row remains differentiable and updates only residual output layers on the
    # zero-initialized first optimizer step.
    poisoned_delta = action_delta.clone()
    poisoned_delta[0].fill_(float("nan"))
    mixed_action = hard_gate_residual(base_action, poisoned_delta, enabled)
    assert torch.equal(mixed_action[0], base_action[0])
    loss = mixed_action[1].sum() + hard_gate_residual(
        base_value,
        value_delta,
        enabled,
    )[1].sum()
    loss.backward()
    assert action_residual.module[-1].weight.grad is not None
    assert torch.count_nonzero(action_residual.module[-1].weight.grad) > 0
    assert value_residual.module[-1].weight.grad is not None
    assert torch.count_nonzero(value_residual.module[-1].weight.grad) > 0
    assert all(parameter.grad is None for parameter in policy.actor_module.release.parameters())
    assert all(parameter.grad is None for parameter in value.release.parameters())

    policy.std.requires_grad_(True)
    with pytest.raises(RuntimeError, match="only motion-compliance residual"):
        motion_compliance_residual_parameters(policy, value)


def test_full_residual_actor_and_value_preserve_release_paths_and_mixed_gradients():
    torch.manual_seed(23)
    policy, value = _build_tiny_residual_models()
    assert isinstance(policy.actor_module, MotionComplianceUniversalTokenModule)
    assert isinstance(value, MotionComplianceResidualCritic)
    assert policy.actor_module.decoders["g1_dyn"].module[0].in_features == 994
    assert policy.actor_module.decoders["probe"].module[0].in_features == 930
    assert value.critic_module.module[0].in_features == 1645
    assert value.running_mean_std.running_mean.shape == (1645,)
    assert value.running_mean_std.frozen is True
    assert policy.std.requires_grad is False

    batch = 3
    condition = torch.tensor(
        [
            [[0.0, float("nan"), float("nan")]],
            [[1.0, 12.0, 240.0]],
            [[0.0, float("nan"), float("nan")]],
        ]
    )
    actor_obs = {
        "actor_obs": torch.randn(batch, 1, 930),
        "tokenizer": torch.randn(batch, 1, 41),
        "motion_compliance_condition": condition,
    }
    privileged = torch.randn(batch, 1, 9)
    privileged[[0, 2]].fill_(float("nan"))
    full_obs = {
        **actor_obs,
        "critic_obs": torch.randn(batch, 1, 1645),
        "motion_compliance_privileged": privileged,
    }

    probe_widths = []
    hook = policy.actor_module.decoders["probe"].register_forward_pre_hook(
        lambda _module, args: probe_widths.append(args[0].shape[-1])
    )
    try:
        base_action = UniversalTokenModule.forward(
            policy.actor_module,
            actor_obs,
        )
        initial_action = policy(full_obs)
        assert torch.equal(initial_action.view(torch.uint8), base_action.view(torch.uint8))

        rich_output = policy.actor_module(
            actor_obs,
            compute_aux_loss=True,
        )
        assert rich_output["action_mean"].shape == (batch, 1, 29)
        assert rich_output["aux_losses"] == {}
        assert probe_widths and set(probe_widths) == {930}
    finally:
        hook.remove()

    with torch.no_grad():
        policy.actor_module.motion_compliance_action_residual.module[-1].bias.fill_(2.0)
        value.motion_compliance_value_residual.module[-1].bias.fill_(0.5)

    mixed_action = policy(full_obs)
    assert torch.equal(
        mixed_action[[0, 2]].view(torch.uint8),
        base_action[[0, 2]].view(torch.uint8),
    )
    assert not torch.equal(mixed_action[1], base_action[1])
    assert torch.max(torch.abs(mixed_action[1] - base_action[1])) <= 0.25

    # PPO supplies the whole observation dict.  The actor's explicit boundary
    # drops privileged groups before direct forward and before temporal history.
    privileged_poisoned = dict(full_obs)
    privileged_poisoned["critic_obs"] = torch.full_like(full_obs["critic_obs"], float("nan"))
    privileged_poisoned["motion_compliance_privileged"] = torch.full_like(
        privileged,
        float("inf"),
    )
    poisoned_action = policy(privileged_poisoned)
    assert torch.equal(poisoned_action.view(torch.uint8), mixed_action.view(torch.uint8))
    with pytest.raises(ValueError, match="non-allowlisted"):
        policy.actor_module(privileged_poisoned)
    policy.init_rollout()
    policy._update_obs_buffer(
        {key: value_.squeeze(1) for key, value_ in privileged_poisoned.items()}
    )
    assert set(policy.obs_dict_buffer.keys()) == {
        "actor_obs",
        "tokenizer",
        "motion_compliance_condition",
    }

    external_tokens = torch.randn(batch, 2, 32)
    base_external = UniversalTokenModule.forward_with_external_tokens(
        policy.actor_module,
        actor_obs,
        external_tokens,
    )
    mixed_external = policy.actor_module.forward_with_external_tokens(
        actor_obs,
        external_tokens,
    )
    assert mixed_external.shape == (batch, 29)
    assert torch.equal(
        mixed_external[[0, 2]].view(torch.uint8),
        base_external[[0, 2]].view(torch.uint8),
    )
    assert not torch.equal(mixed_external[1], base_external[1])

    policy.init_rollout()
    with torch.no_grad():
        policy.std[0] = 0.75
    raw_std = policy.std.detach().clone()
    token_rollout = policy.rollout_with_tokens(
        {key: value_.squeeze(1) for key, value_ in privileged_poisoned.items()},
        external_tokens,
    )
    assert torch.equal(policy.std.detach().view(torch.uint8), raw_std.view(torch.uint8))
    assert token_rollout["action_sigma"].max().item() <= 0.5
    assert token_rollout["action_sigma"].min().item() >= 0.001

    base_value = Critic.evaluate(value, {"critic_obs": full_obs["critic_obs"]})
    mixed_value = value.evaluate(full_obs)
    assert torch.equal(
        mixed_value[[0, 2]].view(torch.uint8),
        base_value[[0, 2]].view(torch.uint8),
    )
    assert not torch.equal(mixed_value[1], base_value[1])

    owned = motion_compliance_residual_parameters(policy, value)
    loss = mixed_action.sum() + mixed_value.sum() + mixed_external.sum()
    loss.backward()
    for parameter in owned:
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.actor_module.motion_compliance_action_residual.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in value.motion_compliance_value_residual.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in policy.named_parameters()
        if "motion_compliance_action_residual" not in name
    )
    assert all(
        parameter.grad is None
        for name, parameter in value.named_parameters()
        if "motion_compliance_value_residual" not in name
    )


def test_host_off_reference_and_new_reward_contributions_are_exact_golden():
    baseline, compliance = _compose_release_pair()
    baseline_reward_terms = _term_names(baseline.manager_env.rewards)
    compliance_reward_terms = _term_names(compliance.manager_env.rewards)
    assert compliance_reward_terms[:-2] == baseline_reward_terms
    for term_name in baseline_reward_terms:
        baseline_term = OmegaConf.to_container(
            baseline.manager_env.rewards[term_name],
            resolve=True,
        )
        compliance_term = OmegaConf.to_container(
            compliance.manager_env.rewards[term_name],
            resolve=True,
        )
        assert compliance_term == baseline_term
    assert baseline.manager_env.rewards.feet_acc.weight == pytest.approx(-2.5e-6)
    assert compliance.manager_env.rewards.feet_acc.weight == pytest.approx(-2.5e-6)
    assert compliance.manager_env.commands.motion_compliance.enabled is False

    original = torch.randn(2, 3, 5, 3)
    original_before = original.clone()
    compliant_candidate = original + torch.randn_like(original)
    stale_all_active = torch.ones(2, 5, dtype=torch.bool)
    disabled = torch.zeros(2, dtype=torch.bool)
    selected = select_yielded_site_reference(
        original,
        compliant_candidate,
        stale_all_active,
        disabled,
    )
    assert torch.equal(selected, original_before)
    assert torch.equal(original, original_before)

    disabled_state = ComplianceCommandState(
        2,
        5,
        3,
        ComplianceSamplingSpec(enable_probability=1.0),
        seed=5,
    )
    disabled_state.reset()
    disabled_state.disable()
    assert not disabled_state.active_site_mask.any()
    torch.testing.assert_close(disabled_state.condition, torch.zeros((2, 3)), rtol=0.0, atol=0.0)

    arbitrary_orientation_error = torch.tensor([[1.0, 0.2], [0.4, 0.9]])
    position_reward = gated_mean_gaussian_reward(
        torch.tensor([[float("nan"), float("nan")], [3.0, 4.0]]),
        disabled,
        0.1,
    )
    orientation_reward = gated_mean_gaussian_reward(
        arbitrary_orientation_error,
        disabled,
        0.4,
    )
    torch.testing.assert_close(position_reward, torch.zeros(2), rtol=0.0, atol=0.0)
    torch.testing.assert_close(orientation_reward, torch.zeros(2), rtol=0.0, atol=0.0)


def test_active_selection_uses_future_zero_and_preserves_every_inactive_site_bitwise():
    num_envs, num_future, num_sites = 2, 3, 3
    original = torch.zeros((num_envs, num_future, num_sites, 3))
    original[0, 0, :, 0] = torch.tensor([1.0, 10.0, 20.0])
    original[1, 0, :, 0] = torch.tensor([2.0, 11.0, 21.0])
    original[:, 1, :, 0] = 1000.0
    original[:, 2, :, 0] = -1000.0
    compliant = original.clone()
    compliant[:, :, 0, 1] += 0.3
    compliant[:, :, 2, 2] -= 0.4
    active_mask = torch.tensor([[True, False, True], [True, True, True]])
    enabled = torch.tensor([True, False])

    selected = select_yielded_site_reference(
        original,
        compliant,
        active_mask,
        enabled,
    )
    assert torch.equal(selected[0, :, 0], compliant[0, :, 0])
    assert torch.equal(selected[0, :, 2], compliant[0, :, 2])
    assert torch.equal(selected[0, :, 1], original[0, :, 1])
    assert torch.equal(selected[1], original[1])

    current = selected[:, 0].clone()
    current[0, 2, 0] += 0.2
    state = SimpleNamespace(
        original_reference_common=original,
        compliant_reference_common=compliant,
        current_reference_common=current,
        active_site_mask=active_mask,
        enabled=enabled,
    )
    selected_error, original_error = endpoint_position_errors_from_state(state)
    torch.testing.assert_close(selected_error[0], torch.tensor([0.0, 0.0, 0.2]))
    assert original_error[0, 0] == pytest.approx(0.3)
    assert original_error[0, 1] == pytest.approx(0.0)
    assert original_error[0, 2] > selected_error[0, 2]
    assert torch.linalg.vector_norm(current[0] - selected[0, 1], dim=-1).min() > 900.0

    reward = gated_mean_gaussian_reward(selected_error, enabled, 0.1)
    assert reward[0] > 0.0
    assert reward[1].item() == 0.0

    stale_state = SimpleNamespace(
        original_reference_common=torch.full_like(original, -50.0),
        compliant_reference_common=torch.full_like(compliant, -40.0),
        current_reference_common=torch.full_like(current, -30.0),
        active_site_mask=active_mask,
        enabled=enabled,
    )
    fresh_state = SimpleNamespace(
        original_reference_common=original,
        compliant_reference_common=compliant,
        current_reference_common=current,
    )

    class FreshCommand:
        state = stale_state
        reads = 0

        def _site_tracking_state(self):
            self.reads += 1
            return fresh_state

    fresh_command = FreshCommand()
    fresh_selected_error, fresh_original_error = (
        current_endpoint_position_errors_from_command(fresh_command)
    )
    assert fresh_command.reads == 1
    torch.testing.assert_close(fresh_selected_error, selected_error)
    torch.testing.assert_close(fresh_original_error, original_error)

    reward_source = _function_source(
        ADAPTER_DIR / "reward.py",
        "endpoint_position_errors_per_site",
    )
    assert "current_endpoint_position_errors_from_command(command)" in reward_source
    assert "refresh_reference_cache" not in reward_source
    assert "command.state.original_reference_common" not in reward_source


def test_orientation_stays_on_original_and_reports_sites_independently():
    half_sqrt = math.sqrt(0.5)
    original_quaternion = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )
    current_quaternion = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [half_sqrt, 0.0, 0.0, half_sqrt]]]
    )
    error = quaternion_error_magnitude_wxyz(original_quaternion, current_quaternion)
    torch.testing.assert_close(error, torch.tensor([[0.0, math.pi / 2]]), atol=1.0e-6, rtol=0.0)

    reward_source = _function_source(
        ADAPTER_DIR / "reward.py",
        "endpoint_orientation_error_per_site",
    )
    assert "[:, 0," in reward_source
    assert "compliant" not in reward_source
    assert "active_site_mask" not in reward_source
    full_reward_source = (ADAPTER_DIR / "reward.py").read_text(encoding="utf-8")
    assert "tracking_vr_3point_error_pos_force" not in full_reward_source
    tree = ast.parse(full_reward_source)
    assert not any(isinstance(node, (ast.AugAssign, ast.NamedExpr)) for node in ast.walk(tree))


def test_opt_in_resolved_groups_are_additive_and_default_physically_off():
    _, compliance = _compose_release_pair()
    assert compliance.manager_env.commands._target_.endswith("ComplianceCommandsCfg")
    assert compliance.manager_env.events._target_.endswith("ComplianceEventsCfg")
    assert compliance.manager_env.observations._target_.endswith(
        "MotionComplianceObservationsCfg"
    )
    assert compliance.manager_env.observations.motion_compliance_condition._target_.endswith(
        "MotionComplianceConditionCfg"
    )
    assert compliance.manager_env.observations.motion_compliance_privileged._target_.endswith(
        "MotionCompliancePrivilegedCfg"
    )
    assert compliance.manager_env.rewards._target_.endswith("MotionComplianceRewardsCfg")
    assert compliance.algo.config.actor._target_.endswith(
        "MotionComplianceFrozenNoiseActor"
    )
    assert compliance.algo.config.actor.backbone._target_.endswith(
        "MotionComplianceUniversalTokenModule"
    )
    assert compliance.algo.config.actor.backbone.motion_compliance_action_delta_limit == 0.25
    assert compliance.algo.config.critic._target_.endswith(
        "MotionComplianceResidualCritic"
    )
    assert compliance.algo.config.freeze_noise_std is True
    assert compliance.manager_env.commands.motion_compliance.enabled is False
    assert compliance.manager_env.rewards.tracking_compliant_endpoint_pos.weight == 2.0
    assert compliance.manager_env.rewards.tracking_compliant_endpoint_pos.params.std == 0.1
    assert compliance.manager_env.rewards.tracking_endpoint_ori.weight == 0.5
    assert compliance.manager_env.rewards.tracking_endpoint_ori.params.std == 0.4
