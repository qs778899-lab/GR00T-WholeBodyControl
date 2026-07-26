#!/usr/bin/env python3
"""One-environment resolved-shape smoke for the Phase-3 opt-in composition."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2_isaaclab_smoke import DEFAULT_ASSET_ROOT, _compose_config
from isaaclab.app import AppLauncher


def _run() -> None:
    app_launcher = AppLauncher(headless=True, device="cuda:0")
    simulation_app = app_launcher.app
    env = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        from gear_sonic.compliance_control.adapters.sonic.command import (
            _articulation_body_data,
        )
        from gear_sonic.compliance_control.adapters.sonic.contracts import (
            select_yielded_site_reference,
        )
        from gear_sonic.compliance_control.training import (
            OFFICIAL_SONIC_RELEASE_CHECKPOINT,
            motion_compliance_residual_parameters,
        )
        from gear_sonic.compliance_control.training.checkpoint import (
            POLICY_STATE_KEYS,
            VALUE_STATE_KEY,
            load_trl_checkpoint,
        )
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from gear_sonic.trl.modules.actor_critic_modules import Critic
        from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.utils.obs_utils import get_group_term_obs_shape

        cfg = _compose_config(
            REPO_ROOT,
            DEFAULT_ASSET_ROOT.resolve(),
            experiment_name="sonic_release_motion_compliance",
            experiment_dir="/tmp/motion_compliance_phase3_smoke",
        )
        env_cfg = custom_instantiate(cfg.manager_env)
        env_cfg.seed = 0
        env_cfg.sim.device = "cuda:0"
        env_cfg.config["headless"] = True
        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env.reset(seed=0)

        observation = env.observation_manager.compute()
        if observation["policy"].shape != (1, 930):
            raise AssertionError(f"unexpected policy shape: {observation['policy'].shape}")
        if observation["critic"].shape != (1, 1645):
            raise AssertionError(f"unexpected critic shape: {observation['critic'].shape}")
        if observation["motion_compliance_condition"].shape != (1, 3):
            raise AssertionError(
                "unexpected condition shape: "
                f"{observation['motion_compliance_condition'].shape}"
            )
        if observation["motion_compliance_privileged"].shape != (1, 9):
            raise AssertionError(
                "unexpected privileged shape: "
                f"{observation['motion_compliance_privileged'].shape}"
            )
        expected_tokenizer_shapes = {
            "command_multi_future_nonflat": (1, 10, 58),
            "motion_anchor_ori_b_mf_nonflat": (1, 10, 6),
        }
        for term_name, expected_shape in expected_tokenizer_shapes.items():
            actual_shape = tuple(observation["tokenizer"][term_name].shape)
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"unexpected tokenizer shape for {term_name}: {actual_shape}"
                )

        # Instantiate the resolved release-size actor/value models and load the
        # immutable official checkpoint without expanding either release path.
        wrapped_env = ManagerEnvWrapper(env, env_cfg.config)
        model_env_config = wrapped_env.config
        model_env_config["obs"]["obs_dims"]["actor_obs"] = 930
        model_env_config["obs"]["obs_dims"]["critic_obs"] = 1645
        model_env_config["robot"]["algo_obs_dim_dict"]["actor_obs"] = 930
        model_env_config["robot"]["algo_obs_dim_dict"]["critic_obs"] = 1645
        for key in env.observation_space:
            if key in ("policy", "critic"):
                continue
            group_dims, group_names, group_total = get_group_term_obs_shape(
                observation,
                key,
            )
            model_env_config["obs"]["group_obs_dims"][key] = group_dims
            model_env_config["obs"]["group_obs_names"][key] = group_names
            model_env_config["obs"]["obs_dims"][key] = group_total
            model_env_config["robot"]["algo_obs_dim_dict"][key] = group_total
        model_env_config["robot"]["actions_dim"] = env.action_space.shape[-1]

        policy = custom_instantiate(
            cfg.algo.config.actor,
            env_config=model_env_config,
            algo_config=cfg.algo.config,
            module_dim_dict=getattr(cfg.algo.config, "module_dim", {}),
            backbone_kwargs={},
            _resolve=False,
        ).to("cuda:0")
        value_model = custom_instantiate(
            cfg.algo.config.critic,
            env_config=model_env_config,
            algo_config=cfg.algo.config,
            module_dim_dict=getattr(cfg.algo.config, "module_dim", {}),
            backbone_kwargs={},
            _resolve=False,
        ).to("cuda:0")
        if policy.actor_module.decoders["g1_dyn"].module[0].in_features != 994:
            raise AssertionError("resolved g1_dyn no longer has official input width 994")
        if value_model.critic_module.module[0].in_features != 1645:
            raise AssertionError("resolved critic no longer has official input width 1645")
        if tuple(value_model.running_mean_std.running_mean.shape) != (1645,):
            raise AssertionError("resolved critic RMS no longer has official width 1645")

        official = load_trl_checkpoint(
            OFFICIAL_SONIC_RELEASE_CHECKPOINT,
            map_location="cuda:0",
        )
        present_policy_keys = [key for key in POLICY_STATE_KEYS if key in official]
        if len(present_policy_keys) != 1:
            raise AssertionError(f"unexpected official policy keys: {present_policy_keys}")
        official_policy = official[present_policy_keys[0]]
        official_value = official[VALUE_STATE_KEY]
        policy_load = policy.load_state_dict(official_policy, strict=False)
        value_load = value_model.load_state_dict(official_value, strict=False)
        if policy_load.unexpected_keys or value_load.unexpected_keys:
            raise AssertionError(
                "official checkpoint produced unexpected keys: "
                f"policy={policy_load.unexpected_keys}, value={value_load.unexpected_keys}"
            )
        if not policy_load.missing_keys or not all(
            key.startswith("actor_module.motion_compliance_action_residual.")
            for key in policy_load.missing_keys
        ):
            raise AssertionError(
                f"official policy missing-key contract changed: {policy_load.missing_keys}"
            )
        if not value_load.missing_keys or not all(
            key.startswith("motion_compliance_value_residual.")
            for key in value_load.missing_keys
        ):
            raise AssertionError(
                f"official value missing-key contract changed: {value_load.missing_keys}"
            )
        loaded_policy = policy.state_dict()
        loaded_value = value_model.state_dict()
        if not all(
            torch.equal(loaded_policy[key], tensor)
            for key, tensor in official_policy.items()
        ):
            raise AssertionError("an official policy tensor changed during residual load")
        if not all(
            torch.equal(loaded_value[key], tensor)
            for key, tensor in official_value.items()
        ):
            raise AssertionError("an official value tensor changed during residual load")

        flat_observation = wrapped_env.process_raw_obs(observation, flatten_dict_obs=True)
        policy_input = {
            key: value.unsqueeze(1)
            for key, value in flat_observation.items()
        }
        actor_only = policy._policy_only_observations(policy_input)
        base_action = UniversalTokenModule.forward(policy.actor_module, actor_only)
        initial_action = policy(policy_input)
        if not torch.equal(initial_action.view(torch.uint8), base_action.view(torch.uint8)):
            raise AssertionError("zero residual changed official actor output")
        base_value = Critic.evaluate(
            value_model,
            {"critic_obs": policy_input["critic_obs"]},
        )
        initial_value = value_model.evaluate(policy_input)
        if not torch.equal(initial_value.view(torch.uint8), base_value.view(torch.uint8)):
            raise AssertionError("zero residual changed official critic output")

        mixed_input = {
            key: value.repeat((3,) + (1,) * (value.ndim - 1))
            for key, value in policy_input.items()
        }
        mixed_input["motion_compliance_condition"].copy_(
            torch.tensor(
                [
                    [[0.0, float("nan"), float("nan")]],
                    [[1.0, 12.0, 240.0]],
                    [[0.0, float("nan"), float("nan")]],
                ],
                device="cuda:0",
            )
        )
        mixed_input["motion_compliance_privileged"][[0, 2]].fill_(float("nan"))
        mixed_actor_only = policy._policy_only_observations(mixed_input)
        mixed_base_action = UniversalTokenModule.forward(
            policy.actor_module,
            mixed_actor_only,
        )
        mixed_base_value = Critic.evaluate(
            value_model,
            {"critic_obs": mixed_input["critic_obs"]},
        )
        with torch.no_grad():
            policy.actor_module.motion_compliance_action_residual.module[-1].bias.fill_(
                2.0
            )
            value_model.motion_compliance_value_residual.module[-1].bias.fill_(0.5)
        mixed_action = policy(mixed_input)
        mixed_value = value_model.evaluate(mixed_input)
        if not torch.equal(
            mixed_action[[0, 2]].view(torch.uint8),
            mixed_base_action[[0, 2]].view(torch.uint8),
        ):
            raise AssertionError("mixed-batch hard-off actor rows changed")
        if not torch.equal(
            mixed_value[[0, 2]].view(torch.uint8),
            mixed_base_value[[0, 2]].view(torch.uint8),
        ):
            raise AssertionError("mixed-batch hard-off critic rows changed")
        if torch.equal(mixed_action[1], mixed_base_action[1]):
            raise AssertionError("enabled actor residual did not change action")
        if torch.equal(mixed_value[1], mixed_base_value[1]):
            raise AssertionError("enabled value residual did not change value")
        if torch.max(torch.abs(mixed_action[1] - mixed_base_action[1])).item() > 0.25:
            raise AssertionError("enabled action residual exceeded its configured bound")

        privileged_poison = dict(mixed_input)
        privileged_poison["critic_obs"] = torch.full_like(
            mixed_input["critic_obs"],
            float("nan"),
        )
        privileged_poison["motion_compliance_privileged"] = torch.full_like(
            mixed_input["motion_compliance_privileged"],
            float("inf"),
        )
        if not torch.equal(
            policy(privileged_poison).view(torch.uint8),
            mixed_action.view(torch.uint8),
        ):
            raise AssertionError("privileged actor poison changed policy output")
        try:
            policy.actor_module(privileged_poison)
        except ValueError as error:
            if "non-allowlisted" not in str(error):
                raise
        else:
            raise AssertionError("direct backbone call accepted privileged groups")
        policy.init_rollout()
        policy._update_obs_buffer(
            {key: value.squeeze(1) for key, value in privileged_poison.items()}
        )
        expected_policy_keys = {
            "actor_obs",
            "tokenizer",
            "motion_compliance_condition",
        }
        if set(policy.obs_dict_buffer.keys()) != expected_policy_keys:
            raise AssertionError("privileged observation leaked into policy history")

        rich_output = policy.actor_module(
            mixed_actor_only,
            compute_aux_loss=True,
        )
        if rich_output["action_mean"].shape != (3, 1, 29):
            raise AssertionError("compute_aux_loss residual path changed action shape")
        external_tokens = policy.actor_module._last_full_latent_flat.reshape(3, 1, 2, 32)
        external_base = UniversalTokenModule.forward_with_external_tokens(
            policy.actor_module,
            mixed_actor_only,
            external_tokens,
        )
        external_action = policy.actor_module.forward_with_external_tokens(
            mixed_actor_only,
            external_tokens,
        )
        if not torch.equal(
            external_action[[0, 2]].view(torch.uint8),
            external_base[[0, 2]].view(torch.uint8),
        ):
            raise AssertionError("external-token hard-off actor rows changed")
        if torch.equal(external_action[1], external_base[1]):
            raise AssertionError("external-token enabled residual did not change action")

        raw_std = policy.std.detach().clone()
        policy.init_rollout()
        token_rollout = policy.rollout_with_tokens(
            {key: value.squeeze(1) for key, value in privileged_poison.items()},
            external_tokens[:, 0],
        )
        if not torch.equal(policy.std.detach().view(torch.uint8), raw_std.view(torch.uint8)):
            raise AssertionError("external-token rollout mutated official std")
        if token_rollout["action_sigma"].max().item() > cfg.algo.config.std_clamp_max:
            raise AssertionError("external-token rollout bypassed release std clamp")

        owned_parameters = motion_compliance_residual_parameters(policy, value_model)
        policy.zero_grad(set_to_none=True)
        value_model.zero_grad(set_to_none=True)
        gradient_action = policy(mixed_input)
        gradient_value = value_model.evaluate(mixed_input)
        (gradient_action.sum() + gradient_value.sum()).backward()
        finite_nonzero_gradients = 0
        for parameter in owned_parameters:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise AssertionError("mixed disabled NaN poisoned residual gradients")
                finite_nonzero_gradients += int(torch.count_nonzero(parameter.grad).item() > 0)
        if finite_nonzero_gradients == 0:
            raise AssertionError("enabled residual produced no trainable gradient")

        model_contract = {
            "actor_base_input": 994,
            "actor_off_uint8_equal": True,
            "critic_base_input": 1645,
            "critic_off_uint8_equal": True,
            "external_token_off_uint8_equal": True,
            "official_policy_missing": sorted(policy_load.missing_keys),
            "official_value_missing": sorted(value_load.missing_keys),
            "residual_gradient_tensors": finite_nonzero_gradients,
        }
        del official, official_policy, official_value
        del policy, value_model, wrapped_env
        torch.cuda.empty_cache()

        command = env.command_manager.get_term("motion_compliance")
        if command.operational_enabled or command.state.active_site_mask.any():
            raise AssertionError("Phase-3 opt-in composition must default physically off")

        reward_names = (
            "tracking_compliant_endpoint_pos",
            "tracking_endpoint_ori",
        )
        episode_before = {
            name: value.clone()
            for name, value in env.reward_manager._episode_sums.items()
        }
        total_reward = env.reward_manager.compute(dt=env.step_dt)
        shared_reward = torch.zeros_like(total_reward)
        reward_values = {}
        reward_term_cfgs = {}
        for name, term_cfg in zip(
            env.reward_manager._term_names,
            env.reward_manager._term_cfgs,
        ):
            reward_term_cfgs[name] = term_cfg
            contribution = env.reward_manager._episode_sums[name] - episode_before[name]
            if name in reward_names:
                if torch.count_nonzero(contribution).item() != 0:
                    raise AssertionError(f"hard-off reward {name} was nonzero")
                reward_values[name] = float(
                    (contribution / (term_cfg.weight * env.step_dt)).item()
                )
            else:
                shared_reward += contribution
        if not torch.equal(total_reward, shared_reward):
            raise AssertionError("hard-off total reward differs from released shared terms")

        tracking = command._tracking_term()
        num_reference_bodies = len(command.cfg.reference_body_names)
        tracking_position = tracking.body_pos_w_multi_future.reshape(
            command.num_envs,
            command.state.num_future_frames,
            num_reference_bodies,
            3,
        )[:, 0, command.body_map.reference_site_indices]
        tracking_quaternion = tracking.body_quat_w_multi_future.reshape(
            command.num_envs,
            command.state.num_future_frames,
            num_reference_bodies,
            4,
        )[:, 0, command.body_map.reference_site_indices]
        expected_world = tracking_position + quat_apply(
            tracking_quaternion,
            command.site_body_offsets.unsqueeze(0).expand(command.num_envs, -1, -1),
        )
        robot_position = _articulation_body_data(command.robot, "pos")
        robot_quaternion = _articulation_body_data(command.robot, "quat")
        anchor_index = command.body_map.articulation_anchor_index
        expected_original_common = quat_apply_inverse(
            robot_quaternion[:, anchor_index, None].expand(-1, command.state.num_sites, -1),
            expected_world - robot_position[:, anchor_index, None],
        )
        current_site_state = command._site_tracking_state()
        if not torch.equal(
            current_site_state.original_reference_common[:, 0],
            expected_original_common,
        ):
            raise AssertionError("future-zero reference differs from tracking command")

        selected_reference = select_yielded_site_reference(
            current_site_state.original_reference_common[:, 0],
            current_site_state.compliant_reference_common[:, 0],
            command.state.active_site_mask,
            command.state.enabled,
        )
        if not torch.equal(
            selected_reference,
            expected_original_common,
        ):
            raise AssertionError("hard-off selected reference changed the original")

        # Exercise the production enabled path against deliberately stale
        # command caches.  This models IsaacLab's reward-before-command-update
        # lifecycle without stepping physics or activating the wrench composer.
        initial_site_state = command._site_tracking_state()
        command.state.enabled.fill_(True)
        command.state.active_site_mask.fill_(True)
        command.state.reference_offset_common.copy_(
            initial_site_state.current_reference_common
            - initial_site_state.original_reference_common[:, 0]
        )
        enabled_site_state = command._site_tracking_state()
        expected_selected = torch.where(
            command.state.active_site_mask.unsqueeze(-1),
            enabled_site_state.compliant_reference_common[:, 0],
            enabled_site_state.original_reference_common[:, 0],
        )
        expected_error = torch.linalg.vector_norm(
            enabled_site_state.current_reference_common - expected_selected,
            dim=-1,
        )
        position_cfg = reward_term_cfgs["tracking_compliant_endpoint_pos"]
        expected_enabled_reward = torch.exp(
            -expected_error.square().mean(dim=-1)
            / (position_cfg.params["std"] * position_cfg.params["std"])
        )
        command.state.original_reference_common.fill_(100.0)
        command.state.compliant_reference_common.fill_(-100.0)
        command.state.current_reference_common.fill_(50.0)
        stale_original = command.state.original_reference_common.clone()
        stale_compliant = command.state.compliant_reference_common.clone()
        stale_current = command.state.current_reference_common.clone()
        enabled_reward = position_cfg.func(env, **position_cfg.params)
        torch.testing.assert_close(
            enabled_reward,
            expected_enabled_reward,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        if enabled_reward.min().item() < 0.99:
            raise AssertionError(f"unexpected enabled freshness reward: {enabled_reward}")
        for actual, stale, label in (
            (command.state.original_reference_common, stale_original, "original"),
            (command.state.compliant_reference_common, stale_compliant, "compliant"),
            (command.state.current_reference_common, stale_current, "current"),
        ):
            if not torch.equal(actual, stale):
                raise AssertionError(f"reward mutated the {label} command cache")
        if (
            torch.count_nonzero(command.application_force_world).item() != 0
            or torch.count_nonzero(command.application_torque_world).item() != 0
            or command.wrench_dirty
        ):
            raise AssertionError("enabled reward-only check activated a wrench")
        command.state.disable()

        print(
            json.dumps(
                {
                    "critic_shape": list(observation["critic"].shape),
                    "condition_shape": list(
                        observation["motion_compliance_condition"].shape
                    ),
                    "enabled_fresh_reward": float(enabled_reward.item()),
                    "model_contract": model_contract,
                    "policy_shape": list(observation["policy"].shape),
                    "privileged_shape": list(
                        observation["motion_compliance_privileged"].shape
                    ),
                    "shared_total_reward_exact": True,
                    "reward_values": reward_values,
                    "tokenizer_shapes": {
                        name: list(shape) for name, shape in expected_tokenizer_shapes.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    _run()
