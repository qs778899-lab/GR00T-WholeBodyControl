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
    motion_file = parser.add_argument("--motion-file", type=Path)
    smpl_motion_dir = parser.add_argument("--smpl-motion-dir", type=Path)
    # AppLauncher performs a preliminary parse before adding its options.  Make
    # the application arguments required only after that parse so bare help is
    # complete and successful while a real launch still validates both paths.
    AppLauncher.add_app_launcher_args(parser)
    motion_file.required = True
    smpl_motion_dir.required = True
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


def _profile_real_bound_compute(command, composer, env, torch) -> None:
    """Trace one forced-due update through the real Isaac-bound command/writer."""

    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except ImportError as error:
        raise RuntimeError("TorchDispatchMode is required for the real bound trace") from error

    from gear_sonic.compliance_control.adapters.sonic.frames import (
        quaternion_rotate_wxyz,
    )

    forbidden_dispatches = []

    class RejectDynamicCudaSync(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            function_name = str(func)
            if "_local_scalar_dense" in function_name or "nonzero" in function_name:
                forbidden_dispatches.append(function_name)
                raise AssertionError(f"dynamic CUDA sync operation: {func}")
            return func(*args, **(kwargs or {}))

    if command.cfg.enabled:
        raise AssertionError("real bound trace must follow the disabled baseline")
    body_ids = torch.tensor(
        command.sites.articulation_indices,
        dtype=torch.long,
        device=env.device,
    )
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(env.device).clone()
    private_rng_before = command._sampling_generator.get_state().clone()  # noqa: SLF001

    command.set_operational_enabled(True)
    if not command.operational_enabled:
        raise AssertionError("failed to enable real bound profiler trace")
    command._time_to_next_pulse.zero_()  # noqa: SLF001

    torch.cuda.synchronize(env.device)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as hot_path_profile:
        with RejectDynamicCudaSync():
            with torch.profiler.record_function("chip_real_bound_command_compute"):
                command.compute(env.step_dt)

    profiled_force_w = command.state.force_on_robot_w.clone()
    selected_force_before_off = composer.composed_force_as_torch.index_select(
        1,
        body_ids,
    ).clone()
    selected_torque_before_off = composer.composed_torque_as_torch.index_select(
        1,
        body_ids,
    ).clone()
    private_rng_after = command._sampling_generator.get_state().clone()  # noqa: SLF001

    # Disable before any environment step or profiler-result inspection. This is
    # the lifecycle assertion that the real composer never retains owned rows.
    command.set_operational_enabled(False)
    torch.cuda.synchronize(env.device)

    if forbidden_dispatches:
        raise AssertionError(f"forbidden dispatches: {forbidden_dispatches}")
    profile_keys = {event.key for event in hot_path_profile.key_averages()}
    forbidden_profile_events = [
        (
            event.name,
            event.cpu_parent.name if event.cpu_parent is not None else None,
        )
        for event in hot_path_profile.events()
        if "_local_scalar_dense" in event.name or "nonzero" in event.name
    ]
    if any(
        "_local_scalar_dense" in key or "nonzero" in key for key in profile_keys
    ):
        raise AssertionError(
            f"forbidden profiler operations: {forbidden_profile_events}"
        )
    if "chip_real_bound_command_compute" not in profile_keys:
        raise AssertionError("profiler did not record the real bound compute region")

    if not torch.equal(torch.random.get_rng_state(), cpu_rng_before):
        raise AssertionError("real bound trace consumed process-global CPU RNG")
    if not torch.equal(torch.cuda.get_rng_state(env.device), cuda_rng_before):
        raise AssertionError("real bound trace consumed process-global CUDA RNG")
    if torch.equal(private_rng_after, private_rng_before):
        raise AssertionError("forced-due trace did not consume the private RNG")
    if not profiled_force_w.count_nonzero():
        raise AssertionError("forced-due trace did not produce a compliance force")
    if not selected_force_before_off.count_nonzero():
        raise AssertionError("real wrench composer did not receive the profiled force")

    reconstructed_world_force = quaternion_rotate_wxyz(
        command.current_site_quaternions_wxyz(),
        selected_force_before_off,
    )
    torch.testing.assert_close(
        reconstructed_world_force,
        profiled_force_w,
        atol=2.0e-4,
        rtol=2.0e-4,
    )
    expected_local_torque = torch.linalg.cross(
        command.application_offsets_local(),
        selected_force_before_off,
        dim=-1,
    )
    torch.testing.assert_close(
        selected_torque_before_off,
        expected_local_torque,
        atol=2.0e-4,
        rtol=2.0e-4,
    )

    if command.operational_enabled:
        raise AssertionError("real bound trace did not return to disabled mode")
    if command.cfg.enabled:
        raise AssertionError("real bound trace mutated static cfg.enabled")
    if command._wrench_write_gate.was_written:  # noqa: SLF001
        raise AssertionError("real bound trace retained wrench ownership")
    if command.state.enabled.any() or command.state.site_mask.any():
        raise AssertionError("real bound trace retained compliance gates")
    if command.state.force_on_robot_w.count_nonzero():
        raise AssertionError("real bound trace retained command force")
    if not torch.isinf(command.time_to_next_pulse).all():
        raise AssertionError("real bound trace retained a pulse countdown")
    selected_force_after_off = composer.composed_force_as_torch.index_select(
        1,
        body_ids,
    )
    selected_torque_after_off = composer.composed_torque_as_torch.index_select(
        1,
        body_ids,
    )
    if selected_force_after_off.count_nonzero():
        raise AssertionError("real bound trace retained owned composer force rows")
    if selected_torque_after_off.count_nonzero():
        raise AssertionError("real bound trace retained owned composer torque rows")

    print(
        "CHIP_PHASE2_REAL_BOUND_PROFILE_PASS",
        "due=forced",
        "dispatch=clean",
        "profiler_cpu_cuda=clean",
        "owned_rows_after_off=zero",
        flush=True,
    )


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
            "manager_env.commands.force.pulse_interval_range_s=[0.02,0.02]",
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
        if not args.enabled:
            reset_ids = torch.tensor([0], dtype=torch.long, device=env.device)
            cpu_rng_before = torch.random.get_rng_state().clone()
            cuda_rng_before = torch.cuda.get_rng_state(env.device).clone()
            command.reset_envs(reset_ids)
            command._update_command()  # noqa: SLF001
            if not torch.equal(torch.random.get_rng_state(), cpu_rng_before):
                raise AssertionError("disabled command reset/update consumed global CPU RNG")
            if not torch.equal(torch.cuda.get_rng_state(env.device), cuda_rng_before):
                raise AssertionError("disabled command reset/update consumed global CUDA RNG")
            if not torch.isinf(command.time_to_next_pulse).all():
                raise AssertionError("disabled command scheduled a compliance pulse")
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

        # Only after the complete disabled baseline is accepted may the process
        # exercise the real bound enabled path. Immediate-off cleanup happens
        # inside the helper, before the separate normal enabled smoke is run.
        if not args.enabled:
            _profile_real_bound_compute(command, composer, env, torch)

        if args.enabled:
            body_ids = torch.tensor(
                command.sites.articulation_indices,
                dtype=torch.long,
                device=env.device,
            )
            unrelated_body_id = next(
                body_id
                for body_id in range(composer.composed_force_as_torch.shape[1])
                if body_id not in command.sites.articulation_indices
            )
            reset_ids = torch.tensor([0], dtype=torch.long, device=env.device)
            sentinel_force = torch.tensor(
                [[[0.125, -0.25, 0.375]]],
                dtype=command.state.dtype,
                device=env.device,
            )
            sentinel_torque = torch.tensor(
                [[[0.03125, -0.0625, 0.09375]]],
                dtype=command.state.dtype,
                device=env.device,
            )
            composer.set_forces_and_torques(
                forces=sentinel_force,
                torques=sentinel_torque,
                positions=None,
                body_ids=[unrelated_body_id],
                env_ids=reset_ids,
                is_global=False,
            )
            unrelated_force_before = composer.composed_force_as_torch[
                0,
                unrelated_body_id,
            ].clone()
            unrelated_torque_before = composer.composed_torque_as_torch[
                0,
                unrelated_body_id,
            ].clone()

            owned_force_w = torch.zeros(
                1,
                command.sites.spec.num_sites,
                3,
                dtype=command.state.dtype,
                device=env.device,
            )
            owned_force_w[0, 0, 0] = 1.25
            owned_site_mask = torch.zeros(
                1,
                command.sites.spec.num_sites,
                dtype=torch.bool,
                device=env.device,
            )
            owned_site_mask[0, 0] = True
            owned_compliance = torch.zeros_like(owned_force_w)
            owned_compliance[0, 0] = 0.02
            command.state.start_pulses(
                reset_ids,
                enabled=torch.ones(1, dtype=torch.bool, device=env.device),
                site_mask=owned_site_mask,
                compliance=owned_compliance,
                peak_force_on_robot_w=owned_force_w,
                duration_s=torch.ones(1, dtype=command.state.dtype, device=env.device),
            )
            command.wrench.set_world_forces_prevalidated(
                owned_force_w,
                body_quaternions_wxyz=command.current_site_quaternions_wxyz(),
                application_offsets_local=command.application_offsets_local(),
            )
            command._wrench_write_gate.mark_written()  # noqa: SLF001
            selected_force_before = composer.composed_force_as_torch[0].index_select(
                0,
                body_ids,
            )
            if not selected_force_before.count_nonzero():
                raise AssertionError("failed to preseed owned non-zero composer rows")
            if not command.state.pulse_active.any():
                raise AssertionError("failed to preseed active compliance state")

            cpu_rng_before = torch.random.get_rng_state().clone()
            cuda_rng_before = torch.cuda.get_rng_state(env.device).clone()
            private_rng_before = command._sampling_generator.get_state().clone()  # noqa: SLF001
            command.set_operational_enabled(False)
            if command.operational_enabled:
                raise AssertionError("operational setter did not disable compliance")
            if not command.cfg.enabled:
                raise AssertionError("operational setter mutated the static enabled config")
            if not torch.equal(torch.random.get_rng_state(), cpu_rng_before):
                raise AssertionError("operational disable consumed global CPU RNG")
            if not torch.equal(torch.cuda.get_rng_state(env.device), cuda_rng_before):
                raise AssertionError("operational disable consumed global CUDA RNG")
            if not torch.equal(
                command._sampling_generator.get_state(),  # noqa: SLF001
                private_rng_before,
            ):
                raise AssertionError("operational disable consumed private RNG")
            if command._wrench_write_gate.was_written:  # noqa: SLF001
                raise AssertionError("operational disable retained wrench ownership")
            if command.state.enabled.any() or command.state.site_mask.any():
                raise AssertionError("operational disable retained compliance gates")
            if command.state.pulse_active.any():
                raise AssertionError("operational disable retained an active pulse")
            if (
                command.state.compliance.count_nonzero()
                or command.state.force_on_robot_w.count_nonzero()
                or command.state.peak_force_on_robot_w.count_nonzero()
                or command.state.pulse_elapsed_s.count_nonzero()
                or command.state.pulse_duration_s.count_nonzero()
            ):
                raise AssertionError("operational disable retained compliance state")
            if not torch.isinf(command.time_to_next_pulse).all():
                raise AssertionError("operational disable retained a pulse countdown")
            selected_force = composer.composed_force_as_torch[0].index_select(0, body_ids)
            selected_torque = composer.composed_torque_as_torch[0].index_select(0, body_ids)
            if selected_force.count_nonzero() or selected_torque.count_nonzero():
                raise AssertionError("operational disable retained owned composer rows")
            torch.testing.assert_close(
                composer.composed_force_as_torch[0, unrelated_body_id],
                unrelated_force_before,
            )
            torch.testing.assert_close(
                composer.composed_torque_as_torch[0, unrelated_body_id],
                unrelated_torque_before,
            )

            cpu_rng_before = torch.random.get_rng_state().clone()
            cuda_rng_before = torch.cuda.get_rng_state(env.device).clone()
            private_rng_before = command._sampling_generator.get_state().clone()  # noqa: SLF001
            command.set_operational_enabled(True)
            if not command.operational_enabled:
                raise AssertionError("operational setter did not enable compliance")
            if torch.equal(
                command._sampling_generator.get_state(),  # noqa: SLF001
                private_rng_before,
            ):
                raise AssertionError("operational enable did not use the private RNG")
            if not torch.equal(torch.random.get_rng_state(), cpu_rng_before):
                raise AssertionError("operational enable consumed global CPU RNG")
            if not torch.equal(torch.cuda.get_rng_state(env.device), cuda_rng_before):
                raise AssertionError("operational enable consumed global CUDA RNG")
            torch.testing.assert_close(
                command.time_to_next_pulse,
                torch.full_like(command.time_to_next_pulse, 0.02),
            )
            command.set_operational_enabled(False)
            if not torch.isinf(command.time_to_next_pulse).all():
                raise AssertionError("second operational disable retained countdowns")

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
            composer.set_forces_and_torques(
                forces=torch.zeros_like(sentinel_force),
                torques=torch.zeros_like(sentinel_torque),
                positions=None,
                body_ids=[unrelated_body_id],
                env_ids=reset_ids,
                is_global=False,
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
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
