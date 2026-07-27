#!/usr/bin/env python3
"""Run one deterministic stiff or compliant Phase-5 evaluation trajectory."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import traceback


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_argument("--mode", choices=("stiff", "compliant"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    motion_file = parser.add_argument("--motion-file", type=Path)
    smpl_motion_dir = parser.add_argument("--smpl-motion-dir", type=Path)
    checkpoint = parser.add_argument("--checkpoint", type=Path)
    trace = parser.add_argument("--trace", type=Path)
    summary = parser.add_argument("--summary", type=Path)
    # AppLauncher temporarily parses application options to detect collisions.
    # Required flags are enabled only for the final parse so help remains a
    # complete, zero-exit, non-writing operation.
    AppLauncher.add_app_launcher_args(parser)
    for action in (mode, motion_file, smpl_motion_dir, checkpoint, trace, summary):
        action.required = True
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack(values, torch):
    return torch.stack(values, dim=0)


def _resolve_new_rollout_outputs(
    trace: Path,
    summary: Path,
) -> tuple[Path, Path, Path, Path]:
    """Resolve new output paths only after rejecting final-component symlinks."""

    requested_trace = Path(trace)
    requested_summary = Path(summary)
    requested_metadata = requested_trace.with_suffix(".json")
    requested_runtime = requested_trace.parent / "runtime"
    requested = (
        requested_trace,
        requested_metadata,
        requested_summary,
        requested_runtime,
    )
    if any(path.is_symlink() or os.path.lexists(path) for path in requested):
        raise FileExistsError("Phase-5 rollout outputs must not already exist")

    resolved_trace = requested_trace.resolve()
    resolved_summary = requested_summary.resolve()
    trace_metadata = resolved_trace.with_suffix(".json")
    runtime_root = resolved_trace.parent / "runtime"
    if resolved_trace.parent != resolved_summary.parent:
        raise ValueError("--trace and --summary must share one output directory")
    if resolved_summary == trace_metadata:
        raise ValueError("--summary must not collide with trace metadata")
    if any(
        path.is_symlink() or os.path.lexists(path)
        for path in (resolved_trace, trace_metadata, resolved_summary, runtime_root)
    ):
        raise FileExistsError("Phase-5 rollout outputs must not already exist")
    return resolved_trace, resolved_summary, trace_metadata, runtime_root


def main() -> int:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    for path in (args.motion_file, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.smpl_motion_dir.is_dir():
        raise NotADirectoryError(args.smpl_motion_dir)
    if args.trace.suffix != ".npz" or args.summary.suffix != ".json":
        raise ValueError("--trace must be .npz and --summary must be .json")
    args.trace, args.summary, trace_metadata, runtime_root = (
        _resolve_new_rollout_outputs(args.trace, args.summary)
    )
    if not args.trace.parent.is_dir():
        raise NotADirectoryError(args.trace.parent)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    raw_env = None
    try:
        from hydra import compose, initialize_config_dir
        from isaaclab.envs import ManagerBasedRLEnv
        from omegaconf import OmegaConf, open_dict
        import torch

        from gear_sonic.compliance_control.adapters.sonic.frames import (
            quaternion_rotate_wxyz_prevalidated,
            world_positions_to_frame_prevalidated,
        )
        from gear_sonic.compliance_control.adapters.sonic.contracts import (
            require_sonic_release_tracking_body_names,
        )
        from gear_sonic.compliance_control.adapters.sonic.isaaclab.command import (
            SonicComplianceCommand,
        )
        from gear_sonic.compliance_control.postprocess import (
            save_tracking_trace,
            write_json_new_atomic,
        )
        from gear_sonic.compliance_control.core import AlignedTrackingTrace
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.utils.config_utils import register_rl_resolvers
        from gear_sonic.utils.obs_utils import get_group_term_obs_shape

        register_rl_resolvers()
        experiment_dir = runtime_root
        overrides = [
            "+exp=manager/universal_token/all_modes/sonic_release_compliance_eval",
            f"seed={args.seed}",
            "num_envs=1",
            "headless=true",
            "use_wandb=false",
            f"experiment_dir={experiment_dir}",
            f"output_dir={experiment_dir}/output",
            f"manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
            (
                "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
                f"{args.smpl_motion_dir}"
            ),
            f"manager_env.commands.force.sampling_seed={args.seed}",
        ]
        config_dir = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            config = compose(config_name="base", overrides=overrides)
        with open_dict(config.manager_env.events):
            config.manager_env.events.pop("push_robot", None)
        OmegaConf.resolve(config.manager_env)

        env_cfg = custom_instantiate(config.manager_env)
        env_cfg.seed = args.seed
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
                example_obs,
                key,
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
            name: int(env.config.robot.algo_obs_dim_dict[name])
            for name in expected_dims
        }
        if resolved_dims != expected_dims:
            raise AssertionError(f"unexpected evaluation observation dimensions: {resolved_dims}")

        actor = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            _resolve=False,
        ).to(args.device)
        try:
            from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
            import trl.trainer.utils

            trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
            trl.trainer.utils.exact_div = exact_div
        except ImportError:
            pass
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        actor.eval()
        actor.init_rollout()

        env.set_is_evaluating(True)
        observations = env.reset(flatten_dict_obs=True)
        motion = env.motion_command
        force_command = env.force_command
        if not isinstance(force_command, SonicComplianceCommand):
            raise AssertionError("evaluation force term is not SonicComplianceCommand")
        if not force_command.operational_enabled:
            raise AssertionError("evaluation force command is not operationally enabled")
        body_names = require_sonic_release_tracking_body_names(
            motion.cfg.body_names,
        )
        site_names = tuple(force_command.sites.spec.site_names)
        reference_site_indices = torch.tensor(
            force_command.sites.reference_indices,
            dtype=torch.long,
            device=env.device,
        )
        offsets = force_command.application_offsets_local()
        frame = force_command.sites.spec.common_frame
        target_fps = float(motion.motion_lib.m_cfg.target_fps)
        dones = torch.zeros(1, dtype=torch.long, device=env.device)
        alive = True
        termination_sample = None
        peak_latent_residual = 0.0
        records = {
            name: []
            for name in (
                "episode_id",
                "motion_id",
                "reference_frame",
                "reference_positions_w",
                "actual_positions_w",
                "reference_positions_local",
                "actual_positions_local",
                "reference_site_positions_w",
                "actual_site_positions_w",
                "reference_site_quaternions_wxyz",
                "actual_site_quaternions_wxyz",
                "force_on_robot_w",
                "enabled",
                "site_mask",
                "compliance_m_per_n",
                "valid",
            )
        }

        for sample_index in range(args.steps):
            reference_positions_w = motion.body_pos_w
            actual_positions_w = motion.robot_body_pos_w
            reference_positions_local = world_positions_to_frame_prevalidated(
                reference_positions_w,
                frame=frame,
                anchor_position_w=motion.anchor_pos_w,
                anchor_quaternion_wxyz=motion.anchor_quat_w,
            )
            actual_positions_local = world_positions_to_frame_prevalidated(
                actual_positions_w,
                frame=frame,
                anchor_position_w=motion.robot_anchor_pos_w,
                anchor_quaternion_wxyz=motion.robot_anchor_quat_w,
            )
            reference_site_positions_w = reference_positions_w.index_select(
                1,
                reference_site_indices,
            ) + quaternion_rotate_wxyz_prevalidated(
                motion.body_quat_w.index_select(1, reference_site_indices),
                offsets,
            )
            actual_site_positions_w = force_command.current_site_positions_w()
            reference_site_quaternions_wxyz = motion.body_quat_w.index_select(
                1,
                reference_site_indices,
            )
            actual_site_quaternions_wxyz = (
                force_command.current_site_quaternions_wxyz()
            )
            normalized_site_quaternions = []
            for label, quaternions in (
                ("reference", reference_site_quaternions_wxyz),
                ("actual", actual_site_quaternions_wxyz),
            ):
                if not torch.isfinite(quaternions).all():
                    raise ValueError(f"{label} site quaternions are not finite")
                norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
                if (norms <= torch.finfo(quaternions.dtype).eps).any():
                    raise ValueError(f"{label} site quaternions contain a zero norm")
                normalized_site_quaternions.append(quaternions / norms)
            (
                reference_site_quaternions_wxyz,
                actual_site_quaternions_wxyz,
            ) = normalized_site_quaternions
            dataset_motion_ids = motion.motion_lib.get_motion_ids_in_dataset(
                motion.motion_ids
            )
            absolute_frame = motion.motion_start_time_steps + motion.time_steps

            records["episode_id"].append(torch.zeros((), dtype=torch.int64))
            records["motion_id"].append(dataset_motion_ids[0].detach().cpu())
            records["reference_frame"].append(absolute_frame[0].detach().cpu())
            records["reference_positions_w"].append(
                reference_positions_w[0].detach().cpu()
            )
            records["actual_positions_w"].append(actual_positions_w[0].detach().cpu())
            records["reference_positions_local"].append(
                reference_positions_local[0].detach().cpu()
            )
            records["actual_positions_local"].append(
                actual_positions_local[0].detach().cpu()
            )
            records["reference_site_positions_w"].append(
                reference_site_positions_w[0].detach().cpu()
            )
            records["actual_site_positions_w"].append(
                actual_site_positions_w[0].detach().cpu()
            )
            records["reference_site_quaternions_wxyz"].append(
                reference_site_quaternions_wxyz[0].detach().cpu()
            )
            records["actual_site_quaternions_wxyz"].append(
                actual_site_quaternions_wxyz[0].detach().cpu()
            )
            records["force_on_robot_w"].append(
                force_command.state.force_on_robot_w[0].detach().cpu()
            )
            records["enabled"].append(force_command.state.enabled[0].detach().cpu())
            records["site_mask"].append(force_command.state.site_mask[0].detach().cpu())
            records["compliance_m_per_n"].append(
                force_command.state.compliance[0].detach().cpu()
            )
            records["valid"].append(torch.tensor(alive, dtype=torch.bool))

            policy_observations = dict(observations)
            if args.mode == "stiff":
                policy_observations["compliance_command"] = torch.zeros_like(
                    observations["compliance_command"]
                )
            with torch.no_grad():
                actions = actor.act_inference(
                    policy_observations,
                    cur_dones=dones,
                    skip_episode_attnmask=True,
                )
            residual = actor.actor_module._last_compliance_residual  # noqa: SLF001
            if residual is None:
                raise AssertionError("evaluation actor did not expose residual output")
            peak_latent_residual = max(
                peak_latent_residual,
                float(residual.abs().max().item()),
            )
            observations, _, dones, extras = env.step({"actions": actions})
            timed_out = bool(extras["time_outs"][0].item())
            if timed_out and sample_index + 1 < args.steps:
                raise AssertionError("evaluation motion timed out before the fixed horizon")
            fell_now = bool(dones[0].item()) and not timed_out
            if alive and fell_now:
                alive = False
                termination_sample = sample_index

        valid = _stack(records["valid"], torch)
        trace = AlignedTrackingTrace(
            mode=args.mode,
            body_names=body_names,
            site_names=site_names,
            local_frame=frame,
            sample_index=torch.arange(args.steps, dtype=torch.int64),
            episode_id=_stack(records["episode_id"], torch),
            motion_id=_stack(records["motion_id"], torch).to(torch.int64),
            reference_frame=_stack(records["reference_frame"], torch).to(torch.int64),
            time_s=torch.arange(args.steps, dtype=torch.float64) / target_fps,
            valid=valid,
            reference_positions_w=_stack(records["reference_positions_w"], torch),
            actual_positions_w=_stack(records["actual_positions_w"], torch),
            reference_positions_local=_stack(records["reference_positions_local"], torch),
            actual_positions_local=_stack(records["actual_positions_local"], torch),
            reference_site_positions_w=_stack(
                records["reference_site_positions_w"],
                torch,
            ),
            actual_site_positions_w=_stack(records["actual_site_positions_w"], torch),
            reference_site_quaternions_wxyz=_stack(
                records["reference_site_quaternions_wxyz"],
                torch,
            ),
            actual_site_quaternions_wxyz=_stack(
                records["actual_site_quaternions_wxyz"],
                torch,
            ),
            force_on_robot_w=_stack(records["force_on_robot_w"], torch),
            enabled=_stack(records["enabled"], torch),
            site_mask=_stack(records["site_mask"], torch),
            compliance_m_per_n=_stack(records["compliance_m_per_n"], torch),
            fell=termination_sample is not None,
            horizon_reached=termination_sample is None,
            termination_sample=termination_sample,
        )
        if args.mode == "stiff" and peak_latent_residual != 0.0:
            raise AssertionError("stiff evaluation residual was not exact zero")
        if args.mode == "compliant" and peak_latent_residual <= 0.0:
            raise AssertionError("compliant evaluation never activated the trained residual")
        save_tracking_trace(trace, args.trace)
        write_json_new_atomic(
            args.summary,
            {
                "schema_version": 1,
                "mode": args.mode,
                "policy_semantics": (
                    "matched_force_release_zero_residual"
                    if args.mode == "stiff"
                    else "matched_force_trained_residual"
                ),
                "comparison_semantics": (
                    "both modes use the same reference motion, external force, site mask, "
                    "and compliance schedule; stiff zeros only the actor residual command"
                ),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": _sha256(args.checkpoint),
                "trace": str(args.trace.resolve()),
                "steps": args.steps,
                "valid_frames": int(valid.sum().item()),
                "fell": trace.fell,
                "termination_sample": termination_sample,
                "peak_latent_residual": peak_latent_residual,
                "observation_dims": resolved_dims,
                "body_names": list(body_names),
                "tracking_body_contract": {
                    "source": "sonic_release_motion_body_names",
                    "count": len(body_names),
                    "ordered_names": list(body_names),
                },
                "site_names": list(site_names),
                "common_frame": {
                    "kind": frame.kind.value,
                    "anchor": frame.anchor,
                    "rotation": frame.rotation.value,
                },
                "site_index_contract": {
                    "ordered_names": list(site_names),
                    "reference_indices": list(force_command.sites.reference_indices),
                    "articulation_indices": list(
                        force_command.sites.articulation_indices
                    ),
                },
                "site_orientation_contract": {
                    "convention": "normalized_finite_wxyz",
                    "reference_index_space": "motion_reference_body_names",
                    "actual_index_space": "articulation_body_names",
                },
            },
        )
        print(
            "CHIP_PHASE5_ROLLOUT_PASS",
            f"mode={args.mode}",
            f"valid_frames={int(valid.sum().item())}",
            f"fell={str(trace.fell).lower()}",
            f"peak_residual={peak_latent_residual:.9g}",
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
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
