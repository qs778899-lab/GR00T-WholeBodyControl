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


def test_actor_adds_only_three_public_values_and_critic_width_tracks_site_count():
    baseline, compliance = _compose_release_pair()
    baseline_policy_terms = _term_names(baseline.manager_env.observations.policy)
    compliance_policy_terms = _term_names(compliance.manager_env.observations.policy)
    assert compliance_policy_terms == [*baseline_policy_terms, "motion_compliance_condition"]
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
    assert 930 + condition.shape[-1] == 933

    policy_fields = _class_assignment_order(
        ADAPTER_DIR / "observation.py",
        "MotionCompliancePolicyCfg",
    )
    assert policy_fields == [
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
        "gravity_dir",
        "motion_compliance_condition",
    ]

    critic_terms = _term_names(compliance.manager_env.observations.critic)
    assert critic_terms[-4:] == [
        "motion_compliance_condition",
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
    assert 1645 + (3 + 1 + 3 * 2 + 2) == 1657


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
    assert compliance.manager_env.observations.policy._target_.endswith(
        "MotionCompliancePolicyCfg"
    )
    assert compliance.manager_env.observations.critic._target_.endswith(
        "MotionCompliancePrivilegedCfg"
    )
    assert compliance.manager_env.rewards._target_.endswith("MotionComplianceRewardsCfg")
    assert compliance.manager_env.commands.motion_compliance.enabled is False
    assert compliance.manager_env.rewards.tracking_compliant_endpoint_pos.weight == 2.0
    assert compliance.manager_env.rewards.tracking_compliant_endpoint_pos.params.std == 0.1
    assert compliance.manager_env.rewards.tracking_endpoint_ori.weight == 0.5
    assert compliance.manager_env.rewards.tracking_endpoint_ori.params.std == 0.4
