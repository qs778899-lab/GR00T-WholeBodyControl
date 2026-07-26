#!/usr/bin/env python3
"""Resolve Phase-3 observations and models in one real Isaac Lab environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--smpl-motion-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _assert_finite(name, value, torch) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains a non-finite value")


def main() -> int:
    args = _parse_args()
    for path in (args.motion_file, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.smpl_motion_dir.is_dir():
        raise NotADirectoryError(args.smpl_motion_dir)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    raw_env = None
    try:
        from hydra import compose, initialize_config_dir
        from isaaclab.envs import ManagerBasedRLEnv
        import torch

        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.utils.config_utils import register_rl_resolvers
        from gear_sonic.utils.obs_utils import get_group_term_obs_shape

        register_rl_resolvers()
        experiment_dir = Path("/tmp/chip_compliance_phase3_shape_smoke")
        overrides = [
            "+exp=manager/universal_token/all_modes/sonic_release_compliance_residual",
            "num_envs=1",
            "headless=true",
            "use_wandb=false",
            "exp_base=chip_compliance_phase3_shape_smoke",
            "experiment_name=chip_compliance_phase3_shape_smoke",
            f"experiment_dir={experiment_dir}",
            f"output_dir={experiment_dir}/output",
            f"manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
            (
                "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
                f"{args.smpl_motion_dir}"
            ),
            "manager_env.commands.force.enabled=false",
            "manager_env.commands.force.target_damper_enabled=false",
        ]
        config_dir = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            config = compose(config_name="base", overrides=overrides)

        env_cfg = custom_instantiate(config.manager_env)
        env_cfg.seed = config.seed
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = True
        raw_env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env = ManagerEnvWrapper(raw_env, env_cfg.config)

        env.config.obs.obs_dims.actor_obs = raw_env.observation_space["policy"].shape[-1]
        env.config.obs.obs_dims.critic_obs = raw_env.observation_space["critic"].shape[-1]
        env.config.robot.algo_obs_dim_dict.actor_obs = raw_env.observation_space[
            "policy"
        ].shape[-1]
        env.config.robot.algo_obs_dim_dict.critic_obs = raw_env.observation_space[
            "critic"
        ].shape[-1]
        example_obs = env.reset(flatten_dict_obs=False)
        for key in raw_env.observation_space:
            if key in ("policy", "critic"):
                continue
            group_dims, group_names, group_total_dim = get_group_term_obs_shape(
                example_obs, key
            )
            env.config.obs.group_obs_dims[key] = group_dims
            env.config.obs.group_obs_names[key] = group_names
            env.config.obs.obs_dims[key] = group_total_dim
            env.config.robot.algo_obs_dim_dict[key] = group_total_dim
        env.config.robot.actions_dim = raw_env.action_space.shape[-1]

        expected_dims = {
            "actor_obs": 930,
            "critic_obs": 1645,
            "tokenizer": 1761,
            "compliance_target": 60,
            "compliance_command": 9,
            "compliance_force": 6,
        }
        resolved_dims = {
            name: int(env.config.robot.algo_obs_dim_dict[name]) for name in expected_dims
        }
        if resolved_dims != expected_dims:
            raise AssertionError(
                f"unexpected resolved observation dimensions: {resolved_dims}"
            )

        actor = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            _resolve=False,
        ).to(args.device)
        critic = custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            _resolve=False,
        ).to(args.device)

        from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
        import trl.trainer.utils

        trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
        trl.trainer.utils.exact_div = exact_div
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["policy_state_dict"])
        critic.load_state_dict(checkpoint["value_state_dict"])
        actor.eval()
        critic.eval()

        observations = env.reset(flatten_dict_obs=True)
        model_obs = {
            name: value.unsqueeze(1) if value.ndim == 2 else value
            for name, value in observations.items()
        }
        for name, width in expected_dims.items():
            actual = tuple(model_obs[name].shape)
            if actual != (1, 1, width):
                raise AssertionError(f"unexpected {name} shape: {actual}")
            _assert_finite(name, model_obs[name], torch)
        if model_obs["compliance_command"].count_nonzero():
            raise AssertionError("default-off actor command is non-zero")
        if model_obs["compliance_force"].count_nonzero():
            raise AssertionError("default-off privileged force is non-zero")

        public = {
            name: model_obs[name] for name in actor.allowed_observation_keys
        }
        with torch.no_grad():
            for _ in range(3):
                actor.update_distribution(model_obs)
            release_action = UniversalTokenModule.forward(actor.actor_module, public)
            compliance_action = actor(model_obs)
            value = critic.evaluate(model_obs)
        official_std = checkpoint["policy_state_dict"]["std"]
        loaded_std = actor.std.detach().cpu()
        if not torch.equal(
            official_std.contiguous().reshape(-1).view(torch.uint8),
            loaded_std.contiguous().reshape(-1).view(torch.uint8),
        ):
            raise AssertionError("distribution update mutated the frozen official std")
        expected_effective_std = torch.clamp(
            official_std,
            min=actor.algo_config.std_clamp_min,
            max=actor.algo_config.std_clamp_max,
        )
        if not torch.equal(actor.get_std.detach().cpu(), expected_effective_std):
            raise AssertionError("effective frozen std does not match release clamp")
        if not torch.equal(
            release_action.contiguous().reshape(-1).view(torch.uint8),
            compliance_action.contiguous().reshape(-1).view(torch.uint8),
        ):
            raise AssertionError("default-off real-observation action is not byte exact")
        if tuple(compliance_action.shape) != (1, 1, 29):
            raise AssertionError(f"unexpected action shape: {compliance_action.shape}")
        if tuple(value.shape) != (1, 1, 1):
            raise AssertionError(f"unexpected value shape: {value.shape}")
        _assert_finite("action", compliance_action, torch)
        _assert_finite("value", value, torch)

        print(
            "CHIP_PHASE3_SHAPE_SMOKE_PASS",
            f"obs_dims={resolved_dims}",
            "action_shape=(1,1,29)",
            "value_shape=(1,1,1)",
            "disabled_action_byte_exact=true",
            "frozen_std_byte_exact=true",
            flush=True,
        )
        result = 0
    except BaseException:
        traceback.print_exc()
        result = 1
    try:
        if raw_env is not None:
            raw_env.close()
    except BaseException:
        traceback.print_exc()
        result = 1
    try:
        simulation_app.close()
    except BaseException:
        traceback.print_exc()
        result = 1
    return result


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
