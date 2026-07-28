#!/usr/bin/env python3
"""Collect one real SONIC simulator trial into the portable Phase-6 trace."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import traceback


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    trial_name = parser.add_argument("--trial-name")
    protocol = parser.add_argument(
        "--protocol",
        choices=("baseline", "off", "no_contact", "single_site", "multi_site"),
    )
    parser.add_argument(
        "--active-site",
        action="append",
        default=[],
        help="Configured SONIC compliance site; repeat for multi-site trials.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2500,
        help="Fail-safe only; publication requires the earlier natural motion timeout.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-threshold-n", type=float, default=10.0)
    parser.add_argument(
        "--reference-offset-common-m",
        type=float,
        nargs=3,
        default=(0.05, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument(
        "--expected-motion-key",
        default="walk_forward_amateur_001__A001",
    )
    motion_file = parser.add_argument("--motion-file", type=Path)
    smpl_motion_dir = parser.add_argument("--smpl-motion-dir", type=Path)
    checkpoint = parser.add_argument("--checkpoint", type=Path)
    trace = parser.add_argument("--trace", type=Path)
    summary = parser.add_argument("--summary", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    for action in (
        trial_name,
        protocol,
        motion_file,
        smpl_motion_dir,
        checkpoint,
        trace,
        summary,
    ):
        action.required = True
    return parser.parse_args()


def _resolve_new_outputs(trace: Path, summary: Path) -> tuple[Path, Path, Path]:
    if trace.suffix != ".npz" or summary.suffix != ".json":
        raise ValueError("--trace must be .npz and --summary must be .json")
    if trace.parent != summary.parent:
        raise ValueError("--trace and --summary must share one output directory")
    runtime_root = trace.parent / f".{trace.stem}.runtime"
    requested = (trace, summary, runtime_root)
    if any(path.is_symlink() or os.path.lexists(path) for path in requested):
        raise FileExistsError("Phase-6 collector outputs must not already exist")
    output_parent = trace.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    resolved_trace = output_parent / trace.name
    resolved_summary = output_parent / summary.name
    resolved_runtime = output_parent / runtime_root.name
    if any(path.is_symlink() or os.path.lexists(path) for path in requested):
        raise FileExistsError("Phase-6 collector outputs must not already exist")
    return resolved_trace, resolved_summary, resolved_runtime


def _protocol_from_args(args: argparse.Namespace):
    from gear_sonic.compliance_control.adapters.sonic.evaluation import (
        SonicEvaluationProtocol,
    )

    active_sites = tuple(args.active_site)
    if args.protocol in {"baseline", "off", "no_contact"} and active_sites:
        raise ValueError(f"{args.protocol} does not accept --active-site")
    if args.protocol == "single_site" and len(active_sites) != 1:
        raise ValueError("single_site requires exactly one --active-site")
    if args.protocol == "multi_site" and len(active_sites) < 2:
        raise ValueError("multi_site requires at least two --active-site values")
    enabled = args.protocol not in {"baseline", "off"}
    operational_enabled = args.protocol != "baseline"
    return SonicEvaluationProtocol(
        enabled=enabled,
        operational_enabled=operational_enabled,
        active_site_ids=active_sites,
        force_threshold_n=args.force_threshold_n,
        reference_offset_common_m=tuple(args.reference_offset_common_m),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not args.expected_motion_key:
        raise ValueError("--expected-motion-key must not be empty")
    if args.max_rows < args.max_steps + 1:
        raise ValueError("--max-rows must accommodate reset plus every requested step")
    for path in (args.motion_file, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.smpl_motion_dir.is_dir():
        raise NotADirectoryError(args.smpl_motion_dir)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _configure_env_observation_dims(env, raw_env, example_obs, get_group_term_obs_shape) -> None:
    env.config.obs.obs_dims.actor_obs = raw_env.observation_space["policy"].shape[-1]
    env.config.obs.obs_dims.critic_obs = raw_env.observation_space["critic"].shape[-1]
    env.config.robot.algo_obs_dim_dict.actor_obs = raw_env.observation_space[
        "policy"
    ].shape[-1]
    env.config.robot.algo_obs_dim_dict.critic_obs = raw_env.observation_space[
        "critic"
    ].shape[-1]
    for key in raw_env.observation_space:
        if key in ("policy", "critic"):
            continue
        group_dims, group_names, group_total_dim = get_group_term_obs_shape(example_obs, key)
        env.config.obs.group_obs_dims[key] = group_dims
        env.config.obs.group_obs_names[key] = group_names
        env.config.obs.obs_dims[key] = group_total_dim
        env.config.robot.algo_obs_dim_dict[key] = group_total_dim
    env.config.robot.actions_dim = raw_env.action_space.shape[-1]
    expected = {
        "actor_obs": 930,
        "critic_obs": 1645,
        "motion_compliance_condition": 3,
        "motion_compliance_privileged": 9,
    }
    actual = {
        key: int(env.config.robot.algo_obs_dim_dict[key])
        for key in expected
    }
    if actual != expected:
        raise AssertionError(f"unexpected Phase-6 observation dimensions: {actual}")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    protocol = _protocol_from_args(args)
    args.trace, args.summary, runtime_root = _resolve_new_outputs(args.trace, args.summary)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    raw_env = None
    termination_observer = None
    result = 1
    try:
        from hydra import compose, initialize_config_dir
        from isaaclab.envs import ManagerBasedRLEnv
        from omegaconf import OmegaConf, open_dict
        import torch

        from gear_sonic.compliance_control.adapters.sonic.evaluation_recorder import (
            SonicEvaluationTraceRecorderTerm,
            make_sonic_evaluation_recorders_cfg,
        )
        from gear_sonic.compliance_control.adapters.sonic.evaluation import (
            NaturalMotionTimeoutObserver,
            PolicyActionByteEvidence,
            SONIC_ACTION_RESIDUAL_PREFIX,
            SONIC_RELEASE_CHECKPOINT_SHA256,
            SONIC_RELEASE_CHECKPOINT_STEP,
            SONIC_RELEASE_TRACKING_BODY_NAMES,
            SONIC_TRAINED_CHECKPOINT_SHA256,
            SONIC_TRAINED_CHECKPOINT_STEP,
            assert_g1_only_encoder_selection,
            clear_and_assert_owned_composer_wrench,
            validate_sonic_evaluation_checkpoint_role,
            validate_sonic_evaluation_event_names,
        )
        from gear_sonic.compliance_control.evaluation import (
            alignment_digest,
            evaluate_trace,
            write_report_json_atomic,
            write_trace_npz_atomic,
        )
        from gear_sonic.compliance_control.training.checkpoint import (
            load_trl_checkpoint,
            validate_checkpoint_sha256,
        )
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.utils.config_utils import register_rl_resolvers
        from gear_sonic.utils.obs_utils import get_group_term_obs_shape

        expected_checkpoint_sha256 = (
            SONIC_RELEASE_CHECKPOINT_SHA256
            if args.protocol == "baseline"
            else SONIC_TRAINED_CHECKPOINT_SHA256
        )
        checkpoint_sha256 = validate_checkpoint_sha256(
            args.checkpoint,
            expected_checkpoint_sha256,
        )
        register_rl_resolvers()
        overrides = [
            "+exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune",
            "manager_env/terminations=tracking/eval",
            f"seed={args.seed}",
            "num_envs=1",
            "headless=true",
            "force_flat_terrain=true",
            "use_wandb=false",
            f"experiment_dir={runtime_root}",
            f"output_dir={runtime_root}/output",
            f"manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
            (
                "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
                f"{args.smpl_motion_dir}"
            ),
            "manager_env.commands.motion.motion_lib_cfg.multi_thread=false",
            "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
        ]
        config_dir = _REPOSITORY_ROOT / "gear_sonic" / "config"
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            config = compose(config_name="base", overrides=overrides)
        with open_dict(config.manager_env.commands.motion):
            config.manager_env.commands.motion.encoder_sample_probs = {
                "g1": 1.0,
                "teleop": 0.0,
                "smpl": 0.0,
            }
            config.manager_env.commands.motion.cat_upper_body_poses = False
            config.manager_env.commands.motion.cat_upper_body_poses_prob = 0.0
            config.manager_env.commands.motion.freeze_frame_aug = False
            config.manager_env.commands.motion.freeze_frame_aug_prob = 0.0
            config.manager_env.commands.motion.teleop_sample_prob_when_smpl = 0.0
            config.manager_env.commands.motion.start_from_first_frame = True
            config.manager_env.commands.motion.sample_from_n_initial_frames = None
            config.manager_env.commands.motion.sample_before_contact = False
            config.manager_env.commands.motion.sample_unique_motions = False
            config.manager_env.commands.motion.motion_lib_cfg.filter_motion_keys = [
                args.expected_motion_key
            ]
        with open_dict(config.manager_env.events):
            for event_name in tuple(config.manager_env.events):
                if event_name not in {"_target_", "motion_compliance_reset"}:
                    config.manager_env.events.pop(event_name)
        with open_dict(config.manager_env.config):
            config.manager_env.config.terrain_type = "plane"
        configured_event_names = tuple(
            name for name in config.manager_env.events if name != "_target_"
        )
        validate_sonic_evaluation_event_names(configured_event_names)
        OmegaConf.resolve(config.manager_env)
        if config.manager_env.config.terrain_type != "plane" or not config.force_flat_terrain:
            raise AssertionError("Phase-6 paired evaluation requires flat plane terrain")

        env_cfg = custom_instantiate(config.manager_env)
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = True
        env_cfg.recorders = make_sonic_evaluation_recorders_cfg(
            trial_name=args.trial_name,
            seed_id=args.seed,
            protocol=protocol,
            max_rows=args.max_rows,
        )
        raw_env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env = ManagerEnvWrapper(raw_env, env_cfg.config)
        active_event_names = tuple(
            term_name
            for mode_names in raw_env.event_manager.active_terms.values()
            for term_name in mode_names
        )
        validate_sonic_evaluation_event_names(active_event_names)
        if env_cfg.config["terrain_type"] != "plane":
            raise AssertionError("resolved simulator terrain must be a plane")
        policy_step_dt_s = float(raw_env.step_dt)
        if not math.isclose(policy_step_dt_s, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise AssertionError("Phase-6 motion frames require an exact 50 Hz policy step")

        example_obs = env.reset(flatten_dict_obs=False)
        _configure_env_observation_dims(
            env,
            raw_env,
            example_obs,
            get_group_term_obs_shape,
        )
        actor = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            _resolve=False,
        ).to(args.device)
        checkpoint = load_trl_checkpoint(args.checkpoint, map_location="cpu")
        checkpoint_state = checkpoint.get("state")
        state_step = getattr(checkpoint_state, "global_step", None)
        if state_step is None and isinstance(checkpoint_state, dict):
            state_step = checkpoint_state.get("global_step")
        expected_state_step = (
            SONIC_RELEASE_CHECKPOINT_STEP
            if args.protocol == "baseline"
            else SONIC_TRAINED_CHECKPOINT_STEP
        )
        expected_residual_keys = tuple(
            key for key in actor.state_dict() if key.startswith(SONIC_ACTION_RESIDUAL_PREFIX)
        )
        if args.protocol == "baseline":
            load_result = actor.load_state_dict(
                checkpoint["policy_state_dict"],
                strict=False,
            )
            missing_keys = tuple(load_result.missing_keys)
            unexpected_keys = tuple(load_result.unexpected_keys)
        else:
            actor.load_state_dict(checkpoint["policy_state_dict"], strict=True)
            missing_keys = ()
            unexpected_keys = ()
        validate_sonic_evaluation_checkpoint_role(
            protocol_role=args.protocol,
            checkpoint_sha256=checkpoint_sha256,
            global_step=state_step,
            missing_policy_keys=missing_keys,
            unexpected_policy_keys=unexpected_keys,
            expected_action_residual_keys=expected_residual_keys,
        )
        if state_step != expected_state_step:
            raise AssertionError("checkpoint role validation failed to pin its step")
        actor.eval()
        actor.init_rollout()

        env.set_is_evaluating(True)
        observations = env.reset(flatten_dict_obs=True)
        tracking = raw_env.command_manager.get_term("motion")
        compliance_command = raw_env.command_manager.get_term("motion_compliance")
        if compliance_command.body_map.anchor_name != "torso_link":
            raise AssertionError("SONIC compliance evidence requires torso_link anchor")
        dataset_motion_ids = tracking.motion_lib.get_motion_ids_in_dataset(
            tracking.motion_ids
        )
        if dataset_motion_ids.numel() != 1:
            raise AssertionError("Phase-6 collector requires exactly one motion stream")
        dataset_motion_id = int(dataset_motion_ids.item())
        motion_id = int(tracking.motion_ids.item())
        motion_key = str(tracking.motion_lib._motion_data_keys[dataset_motion_id])
        motion_start_frame = int(tracking.motion_start_time_steps.item())
        motion_time_step = int(tracking.time_steps.item())
        motion_total_steps = int(
            tracking.motion_lib.get_time_step_total(tracking.motion_ids).item()
        )
        target_fps = int(config.manager_env.commands.motion.motion_lib_cfg.target_fps)
        tracking_body_layout = tuple(tracking.cfg.body_names)
        if (
            dataset_motion_id != 0
            or motion_id != 0
            or motion_key != args.expected_motion_key
            or motion_start_frame != 0
            or motion_time_step != 0
            or target_fps != 50
            or tracking_body_layout != SONIC_RELEASE_TRACKING_BODY_NAMES
        ):
            raise AssertionError(
                "Phase-6 reset motion must be dataset/internal id 0, expected key, "
                "start/time 0, target 50 Hz, and the release 14-point layout"
            )
        assert_g1_only_encoder_selection(tracking)
        observed_host_operational = bool(compliance_command.operational_enabled)
        observed_logical_enabled = bool(compliance_command.state.enabled.item())
        observed_active_site_ids = [
            site_id
            for site_id, is_active in zip(
                compliance_command.cfg.site_body_names,
                compliance_command.state.active_site_mask[0].detach().cpu().tolist(),
                strict=True,
            )
            if is_active
        ]
        if (
            observed_host_operational != protocol.operational_enabled
            or observed_logical_enabled != protocol.enabled
            or observed_active_site_ids != list(protocol.active_site_ids)
        ):
            raise AssertionError("resolved compliance protocol state differs from request")
        initial_condition = (
            compliance_command.state.condition[0].detach().cpu().tolist()
        )
        expected_condition = (
            [
                1.0,
                float(protocol.force_threshold_n),
                float(protocol.force_threshold_n)
                / float(compliance_command.cfg.reference_displacement_m),
            ]
            if protocol.enabled
            else [0.0, 0.0, 0.0]
        )
        if initial_condition != expected_condition:
            raise AssertionError("resolved compliance condition differs from protocol")
        termination_observer = NaturalMotionTimeoutObserver(raw_env.termination_manager)
        termination_observer.install()
        recorder = raw_env.recorder_manager._terms.get(  # noqa: SLF001
            "motion_compliance_trace"
        )
        if not isinstance(recorder, SonicEvaluationTraceRecorderTerm):
            raise RuntimeError("Phase-6 SONIC trace recorder is not active")
        dones = torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
        action_evidence = PolicyActionByteEvidence()
        executed_steps = 0
        for _ in range(args.max_steps):
            assert_g1_only_encoder_selection(tracking)
            with torch.no_grad():
                actions = actor.act_inference(
                    observations,
                    cur_dones=dones,
                    skip_episode_attnmask=True,
                )
            action_evidence.update(actions)
            observations, _, dones, _ = env.step({"actions": actions})
            executed_steps += 1
            if bool(dones.any().item()):
                raise RuntimeError("termination observer failed to suppress automatic reset")
            if bool(termination_observer.sticky_time_out.all().item()):
                break
        else:
            raise RuntimeError(
                "natural motion timeout was not observed before --max-steps; "
                "trace publication is forbidden"
            )
        termination_observer.assert_natural_timeout_completion(executed_steps)
        if executed_steps != motion_total_steps:
            raise RuntimeError(
                "natural timeout did not cover exactly the audited full motion clip"
            )
        post_timeout_clear = clear_and_assert_owned_composer_wrench(compliance_command)
        natural_timeout_env_ids = tuple(
            int(value)
            for value in torch.nonzero(
                termination_observer.sticky_time_out,
                as_tuple=False,
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        failed_env_ids = tuple(
            int(value)
            for value in torch.nonzero(
                termination_observer.sticky_terminated,
                as_tuple=False,
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        trace = recorder.finalize_trace(
            natural_timeout_env_ids=natural_timeout_env_ids,
            failed_env_ids=failed_env_ids,
        )
        if len(trace.motion_ids) != executed_steps + 1:
            raise RuntimeError("trace must contain one reset row plus every physics step")
        composer_evidence = recorder.adapter_evidence_report()
        write_trace_npz_atomic(trace, args.trace)
        trace_sha256 = _sha256_file(args.trace)
        metrics = evaluate_trace(trace)
        summary = {
            "schema_version": "sonic_phase6_collection_v2",
            "evidence_kind": "real_sonic_simulator_trace",
            "trial_name": args.trial_name,
            "protocol": args.protocol,
            "active_site_ids": list(protocol.active_site_ids),
            "protocol_parameters": {
                "force_threshold_n": float(protocol.force_threshold_n),
                "reference_offset_common_m": list(protocol.reference_offset_common_m),
                "derived_stiffness_n_per_m": (
                    float(protocol.force_threshold_n)
                    / float(compliance_command.cfg.reference_displacement_m)
                ),
                "resolved_initial_condition": initial_condition,
            },
            "seed": args.seed,
            "max_steps_fail_safe": args.max_steps,
            "executed_steps": executed_steps,
            "natural_motion_timeout_observed": True,
            "motion": {
                "file": str(args.motion_file.resolve()),
                "file_sha256": _sha256_file(args.motion_file),
                "dataset_motion_id": dataset_motion_id,
                "internal_motion_id": motion_id,
                "key": motion_key,
                "start_frame": motion_start_frame,
                "initial_time_step": motion_time_step,
                "total_target_50hz_steps": motion_total_steps,
                "target_fps": target_fps,
            },
            "tracking_body_layout": list(tracking_body_layout),
            "policy_step_dt_s": policy_step_dt_s,
            "smpl_motion_dir": str(args.smpl_motion_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_global_step": state_step,
            "checkpoint_role": (
                "official_release" if args.protocol == "baseline" else "accepted_step6"
            ),
            "checkpoint_load": {
                "missing_policy_keys": list(missing_keys),
                "unexpected_policy_keys": list(unexpected_keys),
                "expected_action_residual_keys": list(expected_residual_keys),
            },
            "trace": str(args.trace.resolve()),
            "trace_sha256": trace_sha256,
            "alignment_sha256": alignment_digest(trace),
            "policy_action_evidence": action_evidence.report(),
            "termination_evidence": termination_observer.report(),
            "actual_composer_evidence": composer_evidence,
            "post_timeout_clear_evidence": post_timeout_clear,
            "deterministic_environment": {
                "hydra_termination_override": "/manager_env/terminations=tracking/eval",
                "event_names": list(active_event_names),
                "terrain_type": "plane",
                "force_flat_terrain": True,
                "robot_motion_encoder": "g1",
                "encoder_sample_probs": {"g1": 1.0, "teleop": 0.0, "smpl": 0.0},
                "cat_upper_body_poses": False,
                "freeze_frame_aug": False,
                "teleop_sample_prob_when_smpl": 0.0,
                "host_operational_enabled": observed_host_operational,
                "logical_condition_enabled": observed_logical_enabled,
            },
            "frame_semantics": (
                "episode-local simulator transition index; reset snapshot is frame 0; "
                "motion identity embeds dataset motion and reference start frame"
            ),
            "coordinate_convention": {
                "world": "right-handed, Z-up, X-forward",
                "input_quaternion": "WXYZ",
                "persisted_quaternion": "XYZW",
                "site_and_force_frame": (
                    "reference torso_link full-pose frame; direct reference-world targets "
                    "avoid candidate robot-anchor round trips"
                ),
                "tracking_local_frame": (
                    "reference/robot pelvis-relative translations expressed in the same "
                    "reference-pelvis orientation basis"
                ),
            },
            "metrics": metrics,
        }
        write_report_json_atomic(summary, args.summary)
        print(
            "MOTION_PHASE6_TRACE_PASS",
            f"trial={args.trial_name}",
            f"rows={len(trace.motion_ids)}",
            f"steps={executed_steps}",
            f"fall_count={metrics['lifecycle']['fall_count']}",
            flush=True,
        )
        result = 0
    except BaseException:
        traceback.print_exc()
        result = 1
    try:
        if termination_observer is not None:
            termination_observer.restore()
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
