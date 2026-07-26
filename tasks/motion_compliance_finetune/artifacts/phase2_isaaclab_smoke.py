#!/usr/bin/env python3
"""One-environment IsaacLab smoke for the Phase-2 compliance adapter.

This intentionally exercises the real manager lifecycle and the modern
``permanent_wrench_composer`` path.  It is not a PPO/training substitute.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_ASSET_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data"
)


def _stage(name: str) -> None:
    print(f"PHASE2_SMOKE_STAGE={name}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps-per-mode", type=int, default=100)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    if args.steps_per_mode != 100:
        raise ValueError("Phase-2 acceptance requires exactly 100 policy steps per mode")
    return args


def _assert_finite_tree(value: Any, torch_module: Any, path: str = "result") -> None:
    if isinstance(value, torch_module.Tensor):
        if not torch_module.isfinite(value).all():
            raise AssertionError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, torch_module, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, torch_module, f"{path}[{index}]")


def _assert_command_finite(command: Any, torch_module: Any) -> None:
    for name, value in vars(command.state).items():
        if isinstance(value, torch_module.Tensor):
            _assert_finite_tree(value, torch_module, f"command.state.{name}")
    _assert_finite_tree(command.application_force_world, torch_module, "application_force")
    _assert_finite_tree(command.application_torque_world, torch_module, "application_torque")
    _assert_finite_tree(command.application_force_body, torch_module, "application_force_body")
    _assert_finite_tree(command.application_torque_body, torch_module, "application_torque_body")


def _compose_config(
    repo_root: Path,
    asset_root: Path,
    *,
    experiment_name: str = "sonic_release",
    experiment_dir: str = "/tmp/motion_compliance_phase2_smoke",
):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from gear_sonic.utils.config_utils import register_rl_resolvers

    robot_motion = asset_root / "robot_filtered/210531/walk_forward_amateur_001__A001.pkl"
    smpl_motion = asset_root / "smpl_filtered"
    if not robot_motion.is_file() or not smpl_motion.is_dir():
        raise FileNotFoundError(
            f"audited smoke assets missing: robot={robot_motion}, smpl={smpl_motion}"
        )

    register_rl_resolvers()
    config_dir = str((repo_root / "gear_sonic/config").resolve())
    overrides = [
        f"+exp=manager/universal_token/all_modes/{experiment_name}",
        "manager_env/commands=tracking/motion_compliance",
        "manager_env/events=tracking/motion_compliance",
        "num_envs=1",
        "headless=true",
        "seed=0",
        f"experiment_dir={experiment_dir}",
        "manager_env.config.episode_length_s=100.0",
        "manager_env.commands.motion.debug_vis=false",
        "manager_env.commands.motion.motion_lib_cfg.multi_thread=false",
        "manager_env.commands.motion_compliance.enabled=false",
        "manager_env.commands.motion_compliance.enable_probability=0.0",
        "manager_env.commands.motion_compliance.site_activation_probability=1.0",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={robot_motion}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={smpl_motion}",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base", overrides=overrides)

    # Keep this smoke focused on the adapter.  Explicit reset is tested below,
    # so unrelated randomization and task termination cannot mask its result.
    with open_dict(cfg):
        for name in list(cfg.manager_env.events):
            if name not in {"_target_", "motion_compliance_reset"}:
                cfg.manager_env.events[name] = None
        for name in list(cfg.manager_env.terminations):
            if name != "_target_":
                cfg.manager_env.terminations[name] = None
    return cfg


def _assert_disabled_rng_neutral(command: Any, env: Any, torch_module: Any) -> None:
    """Compare the next CPU/CUDA samples across real disabled lifecycle calls."""

    cpu_state = torch_module.random.get_rng_state()
    cuda_state = torch_module.cuda.get_rng_state(env.device)
    expected_cpu = torch_module.rand(8)
    expected_cuda = torch_module.rand(8, device=env.device)
    torch_module.random.set_rng_state(cpu_state)
    torch_module.cuda.set_rng_state(cuda_state, env.device)

    env_ids = torch_module.arange(env.num_envs, device=env.device, dtype=torch_module.long)
    command.reset(env_ids)
    for _ in range(8):
        command.compute(dt=env.step_dt)
    env.event_manager.apply(
        mode="reset",
        env_ids=env_ids,
        global_env_step_count=env.common_step_counter,
    )

    actual_cpu = torch_module.rand(8)
    actual_cuda = torch_module.rand(8, device=env.device)
    torch_module.testing.assert_close(actual_cpu, expected_cpu, rtol=0.0, atol=0.0)
    torch_module.testing.assert_close(actual_cuda, expected_cuda, rtol=0.0, atol=0.0)


def _assert_command_compute_has_no_internal_sync_ops(
    command: Any,
    torch_module: Any,
) -> None:
    """Audit the complete added CUDA command compute path with two traces."""

    from torch.utils._python_dispatch import TorchDispatchMode

    class OperationRecorder(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.operations.add(str(func))
            return func(*args, **(kwargs or {}))

    command.operational_enabled = True
    command.time_left.zero_()
    recorder = OperationRecorder()
    with recorder:
        command.compute(dt=0.02)

    command.time_left.zero_()
    activities = [
        torch_module.profiler.ProfilerActivity.CPU,
        torch_module.profiler.ProfilerActivity.CUDA,
    ]
    with torch_module.profiler.profile(activities=activities) as profiler:
        command.compute(dt=0.02)
    torch_module.cuda.synchronize(command.device)
    command.set_operational_enabled(False)

    synchronization_tokens = ("local_scalar", "nonzero")
    dispatch_sync_ops = sorted(
        operation
        for operation in recorder.operations
        if any(token in operation for token in synchronization_tokens)
    )
    profiler_sync_ops = sorted(
        event.key
        for event in profiler.key_averages()
        if any(token in event.key for token in synchronization_tokens)
    )
    if dispatch_sync_ops or profiler_sync_ops:
        raise AssertionError(
            "CUDA command compute dispatched a host-synchronizing operation: "
            f"dispatch={dispatch_sync_ops}, profiler={profiler_sync_ops}"
        )


def _run() -> None:
    args = _parse_args()
    _stage("before_app_launcher")
    app_launcher = AppLauncher(
        headless=True,
        device=args.device,
    )
    simulation_app = app_launcher.app
    _stage("after_app_launcher")
    env = None
    try:
        # Isaac/Omni-dependent modules must be imported only after the app starts.
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        from gear_sonic.trl.utils.common import custom_instantiate

        _stage("before_hydra_compose")
        cfg = _compose_config(REPO_ROOT, args.asset_root.resolve())
        _stage("after_hydra_compose")
        env_cfg = custom_instantiate(cfg.manager_env)
        _stage("after_config_instantiate")
        env_cfg.seed = 0
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = True
        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        _stage("after_env_construct")
        env.reset(seed=0)
        _stage("after_env_reset")

        command = env.command_manager.get_term("motion_compliance")
        composer = getattr(command.robot, "permanent_wrench_composer", None)
        if composer is None or not hasattr(composer, "set_forces_and_torques"):
            raise AssertionError("smoke requires the modern permanent_wrench_composer API")
        if composer.active:
            raise AssertionError("operationally disabled adapter activated the composer at reset")
        _assert_disabled_rng_neutral(command, env, torch)
        if composer.active:
            raise AssertionError("disabled RNG lifecycle audit activated the composer")
        action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )

        disabled_peak = 0.0
        with torch.no_grad():
            for _ in range(args.steps_per_mode):
                result = env.step(action)
                _assert_finite_tree(result, torch, "disabled_step")
                _assert_command_finite(command, torch)
                disabled_peak = max(
                    disabled_peak,
                    float(command.application_force_world.abs().max().item()),
                )
                if torch.count_nonzero(command.application_force_world).item() != 0:
                    raise AssertionError("disabled mode produced a command wrench")
                if torch.count_nonzero(command.application_force_body).item() != 0:
                    raise AssertionError("disabled mode produced a body-frame wrench")
                if torch.count_nonzero(composer.composed_force_as_torch).item() != 0:
                    raise AssertionError("disabled mode left force in the PhysX composer")
                if torch.count_nonzero(composer.composed_torque_as_torch).item() != 0:
                    raise AssertionError("disabled mode left torque in the PhysX composer")
                if composer.active:
                    raise AssertionError("disabled adapter touched/activated the composer")
            _stage("disabled_100_complete")

            _assert_command_compute_has_no_internal_sync_ops(command, torch)
            if torch.count_nonzero(composer.composed_force_as_torch).item() != 0:
                raise AssertionError("compute trace left force in the PhysX composer")
            if torch.count_nonzero(composer.composed_torque_as_torch).item() != 0:
                raise AssertionError("compute trace left torque in the PhysX composer")

            command.state.sampling = replace(
                command.state.sampling,
                enable_probability=1.0,
                site_activation_probability=1.0,
                force_threshold_range_n=(10.0, 10.0),
            )
            command.set_operational_enabled(True)
            command.state.reference_offset_common.zero_()
            command.state.reference_offset_common[..., 0] = 0.05
            command.time_left.fill_(1.0e9)

            forced_peak = 0.0
            composer_peak = 0.0
            for _ in range(args.steps_per_mode):
                result = env.step(action)
                _assert_finite_tree(result, torch, "forced_step")
                _assert_command_finite(command, torch)
                forced_peak = max(
                    forced_peak,
                    float(command.state.site_force_world.norm(dim=-1).max().item()),
                )
                composer_peak = max(
                    composer_peak,
                    float(composer.composed_force_as_torch.norm(dim=-1).max().item()),
                )
            if forced_peak <= 0.0 or composer_peak <= 0.0:
                raise AssertionError("forced-on mode never produced/applied a nonzero wrench")
            _stage("forced_100_complete")

            command.set_operational_enabled(False)
            if command.wrench_dirty:
                raise AssertionError("disable transition retained compliance wrench ownership")
            if torch.count_nonzero(composer.composed_force_as_torch).item() != 0:
                raise AssertionError("disable transition did not immediately clear composer force")
            if torch.count_nonzero(composer.composed_torque_as_torch).item() != 0:
                raise AssertionError("disable transition did not immediately clear composer torque")

            command.set_operational_enabled(True)
            command.state.reference_offset_common.zero_()
            command.state.reference_offset_common[..., 0] = 0.05
            command.time_left.fill_(1.0e9)
            command.compute(dt=env.step_dt)
            stale_force = composer.composed_force_as_torch.clone()
            if torch.count_nonzero(stale_force).item() == 0:
                raise AssertionError("reset check requires a pre-existing composer wrench")
            env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
            env.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                global_env_step_count=env.common_step_counter,
            )
            if torch.count_nonzero(command.application_force_world).item() != 0:
                raise AssertionError("reset left stale command force")
            if torch.count_nonzero(command.application_torque_world).item() != 0:
                raise AssertionError("reset left stale command torque")
            if torch.count_nonzero(command.application_force_body).item() != 0:
                raise AssertionError("reset left stale body-frame force")
            if torch.count_nonzero(command.application_torque_body).item() != 0:
                raise AssertionError("reset left stale body-frame torque")
            if torch.count_nonzero(composer.composed_force_as_torch).item() != 0:
                raise AssertionError("reset left stale composer force")
            if torch.count_nonzero(composer.composed_torque_as_torch).item() != 0:
                raise AssertionError("reset left stale composer torque")
            _stage("reset_check_complete")

        print(
            json.dumps(
                {
                    "composer_api": "permanent_wrench_composer",
                    "disabled_peak_force_n": disabled_peak,
                    "disabled_rng_neutral": True,
                    "command_compute_host_scalar_free": True,
                    "command_compute_internal_nonzero_free": True,
                    "disabled_composer_active": False,
                    "disabled_steps": args.steps_per_mode,
                    "forced_composer_peak_force_n": composer_peak,
                    "forced_site_peak_force_n": forced_peak,
                    "forced_steps": args.steps_per_mode,
                    "reset_zero": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _stage("success_json_emitted")
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
