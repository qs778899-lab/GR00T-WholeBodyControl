"""Hydra composition and no-write planning for deterministic SONIC review runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import SONIC_RELEASE_TRACKING_BODY_NAMES
from .camera import (
    REVIEW_CAMERA_EYE_OFFSET_M,
    REVIEW_CAMERA_LOOKAT_HEIGHT_M,
    REVIEW_PANEL_HEIGHT,
    REVIEW_PANEL_WIDTH,
    REVIEW_VIDEO_FPS,
)
from .roles import REVIEW_SITE_NAMES, ReviewRole, assert_role_config


@dataclass(frozen=True, slots=True)
class ReviewArtifactPaths:
    """One role's strict artifact layout; construction performs no I/O."""

    root: Path
    motion_id: str
    role_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for field_name in ("motion_id", "role_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value in {".", ".."}:
                raise ValueError(f"{field_name} must be a safe non-empty name")
            if "/" in value or "\\" in value:
                raise ValueError(f"{field_name} must not contain path separators")

    @property
    def directory(self) -> Path:
        return self.root / self.motion_id / self.role_name

    @property
    def trace(self) -> Path:
        return self.directory / "trace.npz"

    @property
    def summary(self) -> Path:
        return self.directory / "summary.json"

    @property
    def panel_video(self) -> Path:
        return self.directory / "panel.mp4"


def compose_review_config(
    role: ReviewRole,
    *,
    motion_file: Path,
    smpl_motion_dir: Path,
    seed: int,
    experiment_dir: Path,
):
    """Compose one role without instantiating a simulator or creating paths."""

    if not isinstance(role, ReviewRole):
        raise TypeError("role must be a ReviewRole")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from gear_sonic.utils.config_utils import register_rl_resolvers

    register_rl_resolvers()
    config_dir = Path(__file__).resolve().parents[4] / "config"
    overrides = [
        "+exp=manager/universal_token/all_modes/sonic_release_compliance_review",
        f"compliance_review_role={role.name}",
        f"seed={seed}",
        "num_envs=1",
        "headless=true",
        "use_wandb=false",
        f"experiment_dir={experiment_dir}",
        f"output_dir={experiment_dir}/output",
        f"manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}",
        (
            "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
            f"{smpl_motion_dir}"
        ),
        f"manager_env.commands.force.sampling_seed={seed}",
    ]
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        config = compose(config_name="base", overrides=overrides)
    # The root contains `${hydra:runtime...}` bookkeeping that is unavailable
    # under the no-run compose API. Resolve only the simulator subtree used by
    # this workflow, matching the existing rollout entrypoint.
    OmegaConf.resolve(config.manager_env)
    validate_review_config(config, role)
    return config


def validate_review_config(config: object, role: ReviewRole) -> None:
    """Fail closed if any deterministic release/review pin was lost."""

    from omegaconf import OmegaConf

    if not isinstance(role, ReviewRole):
        raise TypeError("role must be a ReviewRole")
    cfg = config
    if int(cfg.num_envs) != 1 or int(cfg.manager_env.config.num_envs) != 1:
        raise AssertionError("review config must use exactly one environment")
    env = cfg.manager_env.config
    if env.terrain_type != "plane" or not bool(env.render_results):
        raise AssertionError("review requires rendered plane terrain")
    if (int(env.render_width), int(env.render_height)) != (
        REVIEW_PANEL_WIDTH,
        REVIEW_PANEL_HEIGHT,
    ):
        raise AssertionError("review camera dimensions changed")
    if tuple(float(value) for value in env.eval_camera_offset) != REVIEW_CAMERA_EYE_OFFSET_M:
        raise AssertionError("review camera pose changed")
    if float(env.eval_camera_lookat_height) != REVIEW_CAMERA_LOOKAT_HEIGHT_M:
        raise AssertionError("review camera look-at height changed")
    if float(env.sim_dt) * int(env.decimation) != 1.0 / REVIEW_VIDEO_FPS:
        raise AssertionError("review policy clock must be exactly 50 Hz")
    if float(env.episode_length_s) < 60.0:
        raise AssertionError("episode limit may truncate the audited full clip")

    motion = cfg.manager_env.commands.motion
    if tuple(motion.body_names) != SONIC_RELEASE_TRACKING_BODY_NAMES:
        raise AssertionError("review body order differs from the 14-body release contract")
    if not bool(motion.start_from_first_frame) or motion.sample_from_n_initial_frames is not None:
        raise AssertionError("review motion must start deterministically from frame zero")
    for field_name in (
        "randomize_heading",
        "freeze_frame_aug",
        "cat_upper_body_poses",
        "randomize_wrist_poses",
    ):
        if bool(getattr(motion, field_name)):
            raise AssertionError(f"review motion augmentation must be disabled: {field_name}")
    if int(motion.motion_lib_cfg.target_fps) != REVIEW_VIDEO_FPS:
        raise AssertionError("motion library target_fps must be 50")
    if bool(motion.motion_lib_cfg.adaptive_sampling.enable):
        raise AssertionError("adaptive sampling must be disabled")

    force = cfg.manager_env.commands.force
    if tuple(force.site_names) != REVIEW_SITE_NAMES:
        raise AssertionError("review wrist site order changed")
    expected_offsets = ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0))
    if tuple(tuple(float(value) for value in row) for row in force.site_offsets_local_xyz) != (
        expected_offsets
    ):
        raise AssertionError("review wrist offsets changed")
    if not bool(force.enabled) or bool(force.target_damper_enabled):
        raise AssertionError("review driver requires enabled undamped command ownership")
    if tuple(float(value) for value in force.pulse_interval_range_s) != (120.0, 120.0):
        raise AssertionError("stochastic force sampler is not parked")
    event_keys = set(cfg.manager_env.events.keys())
    if event_keys != {"_target_", "compliance_force_reset"}:
        raise AssertionError(f"review events contain stochastic terms: {sorted(event_keys)}")
    if bool(cfg.manager_env.observations.policy.enable_corruption):
        raise AssertionError("policy observation corruption must be disabled")
    if bool(cfg.manager_env.observations.tokenizer.enable_corruption):
        raise AssertionError("tokenizer corruption must be disabled")

    role_config = OmegaConf.to_container(cfg.compliance_review_role, resolve=True)
    if not isinstance(role_config, dict):
        raise TypeError("compliance_review_role must compose to a mapping")
    assert_role_config(role, role_config)


def build_review_dry_run_plan(
    role: ReviewRole,
    *,
    motion_id: str,
    motion_file: Path,
    smpl_motion_dir: Path,
    checkpoint: Path,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    """Return a JSON-safe plan after Hydra validation and without writing output."""

    from omegaconf import OmegaConf

    paths = ReviewArtifactPaths(output_root, motion_id, role.name)
    config = compose_review_config(
        role,
        motion_file=motion_file,
        smpl_motion_dir=smpl_motion_dir,
        seed=seed,
        experiment_dir=paths.directory / "runtime",
    )
    return {
        "schema_version": "sonic_chip_review_dry_run_v1",
        "role": role.name,
        "checkpoint_kind": role.checkpoint_kind,
        "checkpoint": str(checkpoint),
        "motion_id": motion_id,
        "motion_file": str(motion_file),
        "smpl_motion_dir": str(smpl_motion_dir),
        "seed": seed,
        "output_root": str(output_root),
        "would_write": [str(paths.trace), str(paths.summary), str(paths.panel_video)],
        "body_names": list(config.manager_env.commands.motion.body_names),
        "site_names": list(config.manager_env.commands.force.site_names),
        "site_offsets_local_xyz": OmegaConf.to_container(
            config.manager_env.commands.force.site_offsets_local_xyz,
            resolve=True,
        ),
        "fps": REVIEW_VIDEO_FPS,
        "panel_dimensions": [REVIEW_PANEL_WIDTH, REVIEW_PANEL_HEIGHT],
        "terrain_type": config.manager_env.config.terrain_type,
        "events": sorted(config.manager_env.events.keys()),
        "app_launcher_started": False,
    }
