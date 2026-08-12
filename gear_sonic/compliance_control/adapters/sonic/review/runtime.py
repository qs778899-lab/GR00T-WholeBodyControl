"""Real one-role SONIC collector, imported only after AppLauncher starts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from ....review import write_report_json_atomic, write_trace_npz_atomic
from ..contracts import require_sonic_release_tracking_body_names
from .camera import (
    AtomicReviewVideoWriter,
    ReviewFrameMetadata,
    capture_review_frame,
)
from .config import ReviewArtifactPaths
from .driver import (
    SonicReviewProtocolDriver,
    gate_actor_observations,
    refresh_compliance_observations,
)
from .roles import REVIEW_SITE_NAMES, ReviewRole
from .snapshot import capture_sonic_review_snapshot
from .trace import ReviewTraceAccumulator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_load_semantics(actor: object, role: ReviewRole) -> str:
    """Prove official migration versus native trained-checkpoint restoration."""

    if not isinstance(role, ReviewRole):
        raise TypeError("role must be a ReviewRole")
    report = getattr(actor, "last_migration_report", None)
    if role.checkpoint_kind == "official":
        initialized = getattr(report, "initialized_new_keys", ())
        if report is None or not initialized:
            raise AssertionError(
                "official checkpoint did not use strict legacy migration"
            )
        return "legacy_migration_strict"
    if report is not None:
        raise AssertionError("trained checkpoint unexpectedly used legacy migration")
    return "native_strict_resume"


def _prepare_observation_contract(env, raw_env, config, device: str):
    """Instantiate the accepted actor after deriving manager observation shapes."""

    import torch

    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.utils.obs_utils import get_group_term_obs_shape

    env.config.obs.obs_dims.actor_obs = raw_env.observation_space["policy"].shape[-1]
    env.config.obs.obs_dims.critic_obs = raw_env.observation_space["critic"].shape[-1]
    env.config.robot.algo_obs_dim_dict.actor_obs = raw_env.observation_space["policy"].shape[-1]
    env.config.robot.algo_obs_dim_dict.critic_obs = raw_env.observation_space["critic"].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in raw_env.observation_space:
        if key in ("policy", "critic"):
            continue
        group_dims, group_names, group_total_dim = get_group_term_obs_shape(example_obs, key)
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
    observed_dims = {
        name: int(env.config.robot.algo_obs_dim_dict[name]) for name in expected_dims
    }
    if observed_dims != expected_dims:
        raise AssertionError(f"unexpected review observation dimensions: {observed_dims}")
    actor = custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        _resolve=False,
    ).to(device)
    try:
        from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
        import trl.trainer.utils

        trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
        trl.trainer.utils.exact_div = exact_div
    except ImportError:
        pass
    return actor, observed_dims, torch


def collect_sonic_review_role(
    *,
    config: object,
    role: ReviewRole,
    motion_id: str,
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    branch_commit: str,
    paths: ReviewArtifactPaths,
    device: str,
) -> dict[str, Any]:
    """Run one complete natural-timeout role and publish trace/video/summary."""

    from isaaclab.envs import ManagerBasedRLEnv
    from omegaconf import open_dict

    from gear_sonic.compliance_control.adapters.sonic.isaaclab.command import (
        SonicComplianceCommand,
    )
    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
    from gear_sonic.trl.utils.common import custom_instantiate

    if paths.role_name != role.name or paths.motion_id != motion_id:
        raise ValueError("artifact layout does not match role/motion")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(checkpoint)
    if _sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 changed before collection")
    if paths.directory.exists():
        raise FileExistsError(paths.directory)
    paths.directory.mkdir(parents=True, exist_ok=False)
    runtime_root = paths.directory / "runtime"
    with open_dict(config.manager_env.config):
        config.manager_env.config.experiment_dir = str(runtime_root)
    env_cfg = custom_instantiate(config.manager_env)
    env_cfg.seed = seed
    env_cfg.sim.device = device
    env_cfg.config["headless"] = True
    raw_env = None
    started_at = time.monotonic()
    try:
        raw_env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        env = ManagerEnvWrapper(raw_env, env_cfg.config)
        actor, observation_dims, torch = _prepare_observation_contract(
            env,
            raw_env,
            config,
            device,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(payload["policy_state_dict"], strict=True)
        checkpoint_load_semantics = validate_checkpoint_load_semantics(actor, role)
        actor.eval()
        actor.init_rollout()

        env.set_is_evaluating(True)
        observations = env.reset(flatten_dict_obs=True)
        motion = env.motion_command
        command = env.force_command
        if not isinstance(command, SonicComplianceCommand):
            raise AssertionError("review force term is not SonicComplianceCommand")
        if not command.operational_enabled:
            raise AssertionError("review force command is not operationally enabled")
        body_names = require_sonic_release_tracking_body_names(motion.cfg.body_names)
        if tuple(command.sites.spec.site_names) != REVIEW_SITE_NAMES:
            raise AssertionError("review site order changed at runtime")
        if int(motion.motion_start_time_steps[0].item()) != 0:
            raise AssertionError("review motion did not start at frame zero")
        expected_frame_count = int(
            motion.motion_lib.get_time_step_total(motion.motion_ids)[0].item()
        )
        if expected_frame_count < 200:
            raise AssertionError("formal review motion is unexpectedly short")
        driver = SonicReviewProtocolDriver(command, role)
        driver.reset()
        accumulator = ReviewTraceAccumulator(
            role=role,
            motion_id=motion_id,
            seed=seed,
            point_ids=body_names,
        )
        dones = torch.zeros(1, dtype=torch.long, device=env.device)
        peak_residual = 0.0
        peak_world_force = 0.0
        natural_timeout_seen = False
        with AtomicReviewVideoWriter(paths.panel_video) as video:
            for sample_index in range(expected_frame_count):
                applied = driver.apply(sample_index, expected_frame_count)
                observations = refresh_compliance_observations(raw_env, observations)
                policy_observations = gate_actor_observations(observations, role)
                snapshot = capture_sonic_review_snapshot(motion, command)
                with torch.no_grad():
                    actions = actor.act_inference(
                        policy_observations,
                        cur_dones=dones,
                        skip_episode_attnmask=True,
                    )
                if not torch.isfinite(actions).all():
                    raise ValueError("review actor produced non-finite actions")
                residual = actor.actor_module._last_compliance_residual  # noqa: SLF001
                if residual is None or not torch.isfinite(residual).all():
                    raise AssertionError("review actor did not expose a finite residual")
                residual_peak_now = float(residual.abs().max().item())
                peak_residual = max(peak_residual, residual_peak_now)
                if role.actor_hard_off and residual_peak_now != 0.0:
                    raise AssertionError("hard-off actor residual was not exact zero")
                force_norms = torch.linalg.vector_norm(
                    applied.force_on_robot_world_n[0],
                    dim=-1,
                )
                peak_world_force = max(peak_world_force, float(force_norms.max().item()))
                active_names = tuple(
                    REVIEW_SITE_NAMES[index]
                    for index, active in enumerate(
                        applied.active_site_mask[0].detach().cpu().tolist()
                    )
                    if active
                )
                frame = capture_review_frame(raw_env, motion)
                video.append(
                    frame,
                    ReviewFrameMetadata(
                        role=role.name,
                        branch_commit=branch_commit,
                        checkpoint_sha256=checkpoint_sha256,
                        motion_id=motion_id,
                        seed=seed,
                        frame_index=sample_index,
                        timestamp_s=sample_index / 50.0,
                        active_site_names=active_names,
                        force_norms_n=tuple(
                            float(value) for value in force_norms.detach().cpu().tolist()
                        ),
                        compliance_m_per_n=float(
                            applied.compliance_m_per_n.max().detach().cpu().item()
                        ),
                    ),
                )
                observations, _, dones, extras = env.step({"actions": actions})
                timed_out = bool(extras["time_outs"][0].item())
                terminal = bool(dones[0].item())
                fall = terminal and not timed_out
                accumulator.append(
                    snapshot,
                    policy_action=np.asarray(actions[0].detach().cpu().numpy()),
                    reset=sample_index == 0,
                    terminal=terminal,
                    success=timed_out,
                    fall=fall,
                )
                if terminal:
                    if fall:
                        raise RuntimeError(f"review role fell at sample {sample_index}")
                    if sample_index + 1 != expected_frame_count:
                        raise RuntimeError("review role timed out before the natural final frame")
                    natural_timeout_seen = True
                    break
        if not natural_timeout_seen:
            raise RuntimeError("review role did not reach its natural timeout")
        trace = accumulator.finish(expected_frame_count=expected_frame_count)
        if role.external_force_enabled and peak_world_force < 4.999:
            raise AssertionError("active review role never reached the pinned 5 N force")
        if not role.external_force_enabled and peak_world_force != 0.0:
            raise AssertionError("non-contact review role observed external force")
        if role.residual_enabled and role.external_force_enabled and peak_residual <= 0.0:
            raise AssertionError("compliant review role never activated its trained residual")
        driver.reset()
        write_trace_npz_atomic(trace, paths.trace)
        elapsed_s = time.monotonic() - started_at
        summary = {
            "schema_version": "sonic_chip_review_role_v1",
            "role": role.name,
            "checkpoint_kind": role.checkpoint_kind,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_load_semantics": checkpoint_load_semantics,
            "branch_commit": branch_commit,
            "motion_id": motion_id,
            "seed": seed,
            "frame_count": expected_frame_count,
            "natural_timeout_count": 1,
            "fall_count": 0,
            "reset_count": 1,
            "trace": str(paths.trace.resolve()),
            "trace_sha256": _sha256(paths.trace),
            "panel_video": str(paths.panel_video.resolve()),
            "panel_video_sha256": _sha256(paths.panel_video),
            "body_names": list(body_names),
            "site_names": list(REVIEW_SITE_NAMES),
            "force_evaluation_frame": "world",
            "force_common_frame": command.sites.spec.common_frame.kind.value,
            "peak_world_force_n": peak_world_force,
            "peak_latent_residual": peak_residual,
            "observation_dims": observation_dims,
            "elapsed_s": elapsed_s,
            "simulation_fps": expected_frame_count / elapsed_s,
            "frame_contract": "video frame k equals pre-transition trace sample k",
        }
        write_report_json_atomic(summary, paths.summary)
        return summary
    finally:
        if raw_env is not None:
            raw_env.close()
