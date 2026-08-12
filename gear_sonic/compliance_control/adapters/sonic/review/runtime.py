"""Real one-role SONIC collector, imported only after AppLauncher starts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np

from ....review import (
    probe_video_with_sha256,
    write_report_json_atomic,
    write_trace_npz_atomic,
)
from ..contracts import require_sonic_release_tracking_body_names
from .camera import (
    REVIEW_PANEL_HEIGHT,
    REVIEW_PANEL_WIDTH,
    REVIEW_VIDEO_FPS,
    AtomicReviewVideoWriter,
    ReviewFrameMetadata,
    capture_review_frame,
)
from .config import ReviewArtifactPaths
from .diagnostic import ReviewDiagnosticAccumulator, write_diagnostic_trace_atomic
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


def validate_finite_observations(observations: object) -> None:
    """Fail before inference if any flattened actor observation is invalid."""

    from collections.abc import Mapping

    import torch

    if not isinstance(observations, Mapping) or not observations:
        raise TypeError("observations must be a non-empty mapping")
    for name, value in observations.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("observation groups must map string names to tensors")
        if not torch.isfinite(value).all():
            raise ValueError(f"observation group contains non-finite values: {name}")


def validate_owned_composer_rows_cleared(command: object) -> None:
    """Prove the real permanent-wrench composer cleared only owned body rows."""

    import torch

    composer = getattr(getattr(command, "robot", None), "permanent_wrench_composer", None)
    force = getattr(composer, "composed_force_as_torch", None)
    torque = getattr(composer, "composed_torque_as_torch", None)
    indices = tuple(getattr(getattr(command, "sites", None), "articulation_indices", ()))
    if not isinstance(force, torch.Tensor) or not isinstance(torque, torch.Tensor):
        raise TypeError("command does not expose real composer force/torque tensors")
    if not indices:
        raise ValueError("command does not expose owned articulation indices")
    body_ids = torch.tensor(indices, dtype=torch.long, device=force.device)
    owned_force = force.index_select(1, body_ids)
    owned_torque = torque.index_select(1, body_ids)
    if torch.count_nonzero(owned_force).item() != 0:
        raise AssertionError("reset left nonzero owned composer force rows")
    if torch.count_nonzero(owned_torque).item() != 0:
        raise AssertionError("reset left nonzero owned composer torque rows")


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
    motion_file: Path,
    motion_sha256: str,
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    branch_commit: str,
    paths: ReviewArtifactPaths,
    device: str,
    diagnostic_frame_limit: int | None = None,
) -> dict[str, Any]:
    """Run one formal full role or an explicitly non-formal rendered smoke."""

    from isaaclab.envs import ManagerBasedRLEnv
    from omegaconf import open_dict

    from gear_sonic.compliance_control.adapters.sonic.isaaclab.command import (
        SonicComplianceCommand,
    )
    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
    from gear_sonic.trl.utils.common import custom_instantiate

    if paths.role_name != role.name or paths.motion_id != motion_id:
        raise ValueError("artifact layout does not match role/motion")
    if not motion_file.is_file() or motion_file.is_symlink():
        raise FileNotFoundError(motion_file)
    if _sha256(motion_file) != motion_sha256:
        raise ValueError("motion SHA-256 changed before collection")
    if diagnostic_frame_limit is not None and (
        type(diagnostic_frame_limit) is not int or diagnostic_frame_limit < 8
    ):
        raise ValueError("diagnostic_frame_limit must be an integer of at least eight")
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
        if (
            diagnostic_frame_limit is not None
            and diagnostic_frame_limit >= expected_frame_count
        ):
            raise ValueError("diagnostic cutoff must precede the natural timeout")
        run_frame_count = diagnostic_frame_limit or expected_frame_count
        driver = SonicReviewProtocolDriver(command, role)
        driver.reset()
        validate_owned_composer_rows_cleared(command)
        accumulator = (
            ReviewTraceAccumulator(
                role=role,
                motion_id=motion_id,
                seed=seed,
                point_ids=body_names,
            )
            if diagnostic_frame_limit is None
            else None
        )
        diagnostic_accumulator = (
            ReviewDiagnosticAccumulator(
                role=role.name,
                motion_id=motion_id,
                seed=seed,
            )
            if diagnostic_frame_limit is not None
            else None
        )
        dones = torch.zeros(1, dtype=torch.long, device=env.device)
        peak_residual = 0.0
        peak_world_force = 0.0
        natural_timeout_seen = False
        with AtomicReviewVideoWriter(paths.panel_video) as video:
            for sample_index in range(run_frame_count):
                applied = driver.apply(sample_index, run_frame_count)
                observations = refresh_compliance_observations(raw_env, observations)
                policy_observations = gate_actor_observations(observations, role)
                validate_finite_observations(policy_observations)
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
                        timestamp_s=sample_index / REVIEW_VIDEO_FPS,
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
                action_numpy = np.asarray(actions[0].detach().cpu().numpy())
                if accumulator is not None:
                    accumulator.append(
                        snapshot,
                        policy_action=action_numpy,
                        reset=sample_index == 0,
                        terminal=terminal,
                        success=timed_out,
                        fall=fall,
                    )
                else:
                    assert diagnostic_accumulator is not None
                    diagnostic_accumulator.append(
                        snapshot,
                        policy_action=action_numpy,
                        terminal=terminal,
                        timed_out=timed_out,
                        fall=fall,
                    )
                if terminal:
                    if diagnostic_frame_limit is not None:
                        raise RuntimeError(
                            f"diagnostic role terminated at sample {sample_index}"
                        )
                    if fall:
                        raise RuntimeError(f"review role fell at sample {sample_index}")
                    if sample_index + 1 != expected_frame_count:
                        raise RuntimeError("review role timed out before the natural final frame")
                    natural_timeout_seen = True
                    break
        if diagnostic_frame_limit is None and not natural_timeout_seen:
            raise RuntimeError("review role did not reach its natural timeout")
        if role.external_force_enabled and peak_world_force < 4.999:
            raise AssertionError("active review role never reached the pinned 5 N force")
        if not role.external_force_enabled and peak_world_force != 0.0:
            raise AssertionError("non-contact review role observed external force")
        if role.residual_enabled and role.external_force_enabled and peak_residual <= 0.0:
            raise AssertionError("compliant review role never activated its trained residual")
        driver.reset()
        validate_owned_composer_rows_cleared(command)
        if accumulator is not None:
            trace = accumulator.finish(expected_frame_count=expected_frame_count)
            write_trace_npz_atomic(trace, paths.trace)
            trace_kind = "formal_natural_timeout"
        else:
            assert diagnostic_accumulator is not None
            diagnostic_arrays = diagnostic_accumulator.finish(
                expected_frame_count=run_frame_count
            )
            write_diagnostic_trace_atomic(diagnostic_arrays, paths.trace)
            trace_kind = "diagnostic_fixed_cutoff_nonformal"
        video_probe, panel_video_sha256 = probe_video_with_sha256(paths.panel_video)
        expected_probe = {
            "codec_name": "h264",
            "pixel_format": "yuv420p",
            "width": REVIEW_PANEL_WIDTH,
            "height": REVIEW_PANEL_HEIGHT,
            "frame_rate": str(REVIEW_VIDEO_FPS),
            "frame_count": run_frame_count,
        }
        for key, expected_value in expected_probe.items():
            if video_probe[key] != expected_value:
                raise AssertionError(
                    f"review panel video {key} changed: {video_probe[key]!r}"
                )
        expected_duration_s = run_frame_count / REVIEW_VIDEO_FPS
        if abs(video_probe["duration_s"] - expected_duration_s) > (
            0.5 / REVIEW_VIDEO_FPS
        ):
            raise AssertionError("review panel duration differs from the trace")
        elapsed_s = time.monotonic() - started_at
        summary = {
            "schema_version": (
                "sonic_chip_review_role_v1"
                if diagnostic_frame_limit is None
                else "sonic_chip_review_diagnostic_v1"
            ),
            "role": role.name,
            "checkpoint_kind": role.checkpoint_kind,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_load_semantics": checkpoint_load_semantics,
            "branch_commit": branch_commit,
            "motion_id": motion_id,
            "motion_file": str(motion_file.resolve()),
            "motion_sha256": motion_sha256,
            "seed": seed,
            "frame_count": run_frame_count,
            "source_motion_frame_count": expected_frame_count,
            "trace_kind": trace_kind,
            "natural_timeout_count": int(natural_timeout_seen),
            "fall_count": 0,
            "finite_observations": True,
            "finite_actions": True,
            "trace_reset_count": 1,
            "command_reset_count": 2,
            "composer_owned_reset_force_peak_n": 0.0,
            "composer_owned_reset_torque_peak_nm": 0.0,
            "trace": str(paths.trace.resolve()),
            "trace_sha256": _sha256(paths.trace),
            "panel_video": str(paths.panel_video.resolve()),
            "panel_video_sha256": panel_video_sha256,
            "panel_video_probe": video_probe,
            "body_names": list(body_names),
            "site_names": list(REVIEW_SITE_NAMES),
            "force_evaluation_frame": "world",
            "force_common_frame": command.sites.spec.common_frame.kind.value,
            "peak_world_force_n": peak_world_force,
            "peak_latent_residual": peak_residual,
            "observation_dims": observation_dims,
            "elapsed_s": elapsed_s,
            "simulation_fps": run_frame_count / elapsed_s,
            "frame_contract": "video frame k equals pre-transition trace sample k",
        }
        write_report_json_atomic(summary, paths.summary)
        return summary
    finally:
        if raw_env is not None:
            raw_env.close()
