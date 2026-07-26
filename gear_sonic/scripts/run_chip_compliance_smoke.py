#!/usr/bin/env python3
"""One-environment, zero-action Isaac Lab smoke for Phase-2 compliance wiring."""

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
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--smpl-motion-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _assert_finite_tree(value, torch) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError("non-finite observation encountered")
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite_tree(child, torch)
    elif isinstance(value, tuple | list):
        for child in value:
            _assert_finite_tree(child, torch)


def main() -> int:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not args.motion_file.is_file():
        raise FileNotFoundError(args.motion_file)
    if not args.smpl_motion_dir.is_dir():
        raise NotADirectoryError(args.smpl_motion_dir)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    try:
        from hydra import compose, initialize_config_dir
        from isaaclab.envs import ManagerBasedRLEnv
        from omegaconf import OmegaConf
        import torch

        from gear_sonic.compliance_control.adapters.sonic.isaaclab.command import (
            SonicComplianceCommand,
        )
        from gear_sonic.compliance_control.adapters.sonic.frames import (
            quaternion_rotate_wxyz,
        )
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.utils.config_utils import register_rl_resolvers

        register_rl_resolvers()
        config_dir = Path(__file__).resolve().parents[1] / "config"
        experiment_dir = Path("/tmp/chip_compliance_phase2_smoke")
        overrides = [
            "+exp=manager/universal_token/all_modes/sonic_release_compliance",
            "num_envs=1",
            "headless=true",
            "use_wandb=false",
            "exp_base=chip_compliance_phase2_smoke",
            "experiment_name=chip_compliance_phase2_smoke",
            f"experiment_dir={experiment_dir}",
            f"output_dir={experiment_dir}/output",
            f"manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
            (
                "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
                f"{args.smpl_motion_dir}"
            ),
            f"manager_env.commands.force.enabled={str(args.enabled).lower()}",
            "manager_env.commands.force.enabled_probability=1.0",
            "manager_env.commands.force.site_probability=1.0",
            "manager_env.commands.force.force_magnitude_range_n=[5.0,5.0]",
            "manager_env.commands.force.compliance_values_m_per_n=[0.02]",
            "manager_env.commands.force.force_duration_range_s=[1.0,1.0]",
            "manager_env.events.compliance_force_push.interval_range_s=[0.02,0.02]",
        ]
        with initialize_config_dir(
            version_base="1.1",
            config_dir=str(config_dir),
        ):
            config = compose(config_name="base", overrides=overrides)
        OmegaConf.resolve(config.manager_env)
        env_cfg = custom_instantiate(config.manager_env)
        env_cfg.seed = config.seed
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = True
        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)

        observations, _ = env.reset()
        _assert_finite_tree(observations, torch)
        command = env.command_manager.get_term("force")
        if not isinstance(command, SonicComplianceCommand):
            raise AssertionError("force command did not instantiate SonicComplianceCommand")
        composer = command.robot.permanent_wrench_composer
        if not args.enabled and composer.active:
            raise AssertionError("disabled command touched permanent wrench composer")
        action = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            dtype=torch.float32,
            device=env.device,
        )
        saw_nonzero_force = False
        peak_net_force_n = torch.zeros((), dtype=torch.float32, device=env.device)
        peak_net_torque_nm = torch.zeros((), dtype=torch.float32, device=env.device)
        for _ in range(args.steps):
            step_result = env.step(action)
            _assert_finite_tree(step_result, torch)
            force = command.state.force_on_robot_w
            nonzero_force = bool((torch.linalg.vector_norm(force, dim=-1) > 0.0).any())
            saw_nonzero_force |= nonzero_force
            if nonzero_force:
                application_positions_w = command.current_site_positions_w()
                wrench_origin_w = command.robot.data.body_pos_w[
                    :,
                    command.anchor_body_index,
                ]
                net_force = force.sum(dim=1)
                net_torque = torch.linalg.cross(
                    application_positions_w - wrench_origin_w.unsqueeze(1),
                    force,
                    dim=-1,
                ).sum(dim=1)
                peak_net_force_n = torch.maximum(
                    peak_net_force_n,
                    torch.linalg.vector_norm(net_force, dim=-1).max(),
                )
                peak_net_torque_nm = torch.maximum(
                    peak_net_torque_nm,
                    torch.linalg.vector_norm(net_torque, dim=-1).max(),
                )
                if bool(
                    (
                        torch.linalg.vector_norm(net_force, dim=-1)
                        > command.cfg.max_net_force_n + 1.0e-4
                    ).any()
                ):
                    raise AssertionError("per-step net force cap was exceeded")
                if bool(
                    (
                        torch.linalg.vector_norm(net_torque, dim=-1)
                        > command.cfg.max_net_torque_nm + 1.0e-4
                    ).any()
                ):
                    raise AssertionError("per-step net torque cap was exceeded")
                body_ids = torch.tensor(
                    command.sites.articulation_indices,
                    dtype=torch.long,
                    device=env.device,
                )
                local_force = composer.composed_force_as_torch.index_select(1, body_ids)
                reconstructed_world_force = quaternion_rotate_wxyz(
                    command.current_site_quaternions_wxyz(),
                    local_force,
                )
                torch.testing.assert_close(
                    reconstructed_world_force,
                    force,
                    atol=2.0e-4,
                    rtol=2.0e-4,
                )
                local_torque = composer.composed_torque_as_torch.index_select(1, body_ids)
                expected_local_torque = torch.linalg.cross(
                    command.application_offsets_local(),
                    local_force,
                    dim=-1,
                )
                torch.testing.assert_close(
                    local_torque,
                    expected_local_torque,
                    atol=2.0e-4,
                    rtol=2.0e-4,
                )

        if args.enabled and not saw_nonzero_force:
            raise AssertionError("enabled smoke never observed a non-zero compliance wrench")
        if not args.enabled and saw_nonzero_force:
            raise AssertionError("disabled smoke observed a non-zero compliance wrench")
        if not args.enabled and composer.active:
            raise AssertionError("disabled command touched permanent wrench composer during steps")

        if args.enabled:
            command.cfg.enabled = False
            body_ids = torch.tensor(
                command.sites.articulation_indices,
                dtype=torch.long,
                device=env.device,
            )
            for disabled_step in range(2):
                step_result = env.step(action)
                _assert_finite_tree(step_result, torch)
                if command._wrench_write_gate.was_written:  # noqa: SLF001
                    raise AssertionError(
                        f"disabled step {disabled_step + 1} retained wrench ownership"
                    )
                if command.state.enabled.any() or command.state.site_mask.any():
                    raise AssertionError(
                        f"disabled step {disabled_step + 1} retained compliance gates"
                    )
                selected_force = composer.composed_force_as_torch[0].index_select(
                    0,
                    body_ids,
                )
                selected_torque = composer.composed_torque_as_torch[0].index_select(
                    0,
                    body_ids,
                )
                if selected_force.count_nonzero() or selected_torque.count_nonzero():
                    raise AssertionError(
                        f"disabled step {disabled_step + 1} retained selected composer rows"
                    )

        env.reset(env_ids=torch.tensor([0], dtype=torch.long, device=env.device))
        if command.state.enabled.any() or command.state.site_mask.any():
            raise AssertionError("reset left stale compliance gates")
        if command.state.force_on_robot_w.count_nonzero():
            raise AssertionError("reset left stale command force")
        if command.state.compliance.count_nonzero():
            raise AssertionError("reset left stale compliance")
        torch.testing.assert_close(
            command.state.damped_target_common,
            command.current_eef_common_future(),
        )
        composer_force = composer.composed_force_as_torch
        body_ids = torch.tensor(
            command.sites.articulation_indices,
            dtype=torch.long,
            device=composer_force.device,
        )
        if composer_force[0].index_select(0, body_ids).count_nonzero():
            raise AssertionError("reset left stale permanent wrench-composer force")

        print(
            "CHIP_PHASE2_SMOKE_PASS",
            f"enabled={args.enabled}",
            f"steps={args.steps}",
            f"nonzero_force_seen={saw_nonzero_force}",
            f"peak_net_force_n={float(peak_net_force_n):.6f}",
            f"force_limit_n={command.cfg.max_net_force_n:.6f}",
            f"peak_net_torque_nm={float(peak_net_torque_nm):.6f}",
            f"torque_limit_nm={command.cfg.max_net_torque_nm:.6f}",
            flush=True,
        )
        result = 0
    except BaseException:
        traceback.print_exc()
        result = 1
    try:
        if env is not None:
            env.close()
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
