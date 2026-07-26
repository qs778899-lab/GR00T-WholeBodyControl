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
        from gear_sonic.trl.utils.common import custom_instantiate

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
        if observation["policy"].shape != (1, 933):
            raise AssertionError(f"unexpected policy shape: {observation['policy'].shape}")
        if observation["critic"].shape != (1, 1657):
            raise AssertionError(f"unexpected critic shape: {observation['critic'].shape}")
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
                    "enabled_fresh_reward": float(enabled_reward.item()),
                    "policy_shape": list(observation["policy"].shape),
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
