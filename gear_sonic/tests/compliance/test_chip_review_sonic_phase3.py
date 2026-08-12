"""CPU/fake-manager gates for the deterministic SONIC CHIP review workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.compliance_control.adapters.sonic.contracts import (
    SONIC_RELEASE_TRACKING_BODY_NAMES,
)
from gear_sonic.compliance_control.adapters.sonic.review.camera import (
    REVIEW_CAMERA_EYE_OFFSET_M,
    REVIEW_PANEL_HEIGHT,
    REVIEW_PANEL_WIDTH,
    AtomicReviewVideoWriter,
    ReviewFrameMetadata,
    capture_review_frame,
    normalize_rgb_frame,
    overlay_review_metadata,
)
from gear_sonic.compliance_control.adapters.sonic.review.config import (
    compose_review_config,
)
from gear_sonic.compliance_control.adapters.sonic.review.driver import (
    SonicReviewProtocolDriver,
    gate_actor_observations,
)
from gear_sonic.compliance_control.adapters.sonic.review.protocol import (
    DeterministicForceProtocol,
    chip_selected_target,
)
from gear_sonic.compliance_control.adapters.sonic.review.roles import (
    REVIEW_COMPARISONS,
    REVIEW_ROLE_NAMES,
    REVIEW_SITE_NAMES,
    get_review_role,
)
from gear_sonic.compliance_control.adapters.sonic.review.runtime import (
    validate_checkpoint_load_semantics,
)
from gear_sonic.compliance_control.adapters.sonic.review.trace import (
    ReviewTraceAccumulator,
    SonicReviewSnapshot,
)
from gear_sonic.compliance_control.adapters.sonic.wrench import WrenchWriteGate
from gear_sonic.compliance_control.core import CartesianFrameSpec
from gear_sonic.compliance_control.review import (
    compose_review_panels,
    probe_video_with_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = (
    REPOSITORY_ROOT / "gear_sonic/scripts/run_chip_review_collect.py",
    REPOSITORY_ROOT / "gear_sonic/scripts/evaluate_chip_review.py",
    REPOSITORY_ROOT / "gear_sonic/scripts/validate_chip_review.py",
)
ASSET_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control"
)
ORIGINAL_MOTION = (
    ASSET_ROOT
    / "official_assets/sample_data/robot_filtered/210531/"
    "walk_forward_amateur_001__A001.pkl"
)
MIRRORED_MOTION = ORIGINAL_MOTION.with_name("walk_forward_amateur_001__A001_M.pkl")
SMPL_DIR = ASSET_ROOT / "official_assets/sample_data/smpl_filtered"
OFFICIAL_CHECKPOINT = ASSET_ROOT / "official_assets/sonic_release/last.pt"
TRAINED_CHECKPOINT = (
    ASSET_ROOT
    / "runs/chip/phase4_acceptance_resume_fix/"
    "compliance_residual_step6_resume/last.pt"
)


def test_nine_roles_and_five_comparisons_are_exact_and_unaliased():
    assert REVIEW_ROLE_NAMES == (
        "release_baseline",
        "chip_hard_off",
        "enabled_no_contact",
        "single_left_stiff",
        "single_left_compliant",
        "single_right_stiff",
        "single_right_compliant",
        "simultaneous_stiff",
        "simultaneous_compliant",
    )
    assert len(REVIEW_COMPARISONS) == 5
    compared_roles = {role for _, left, right in REVIEW_COMPARISONS for role in (left, right)}
    assert compared_roles == set(REVIEW_ROLE_NAMES)
    assert get_review_role("release_baseline").checkpoint_kind == "official"
    assert all(
        get_review_role(name).checkpoint_kind == "trained"
        for name in REVIEW_ROLE_NAMES[1:]
    )
    with pytest.raises(ValueError, match="unsupported"):
        get_review_role("implicit_alias")


def test_checkpoint_load_semantics_distinguish_official_migration_and_resume():
    migrated = SimpleNamespace(
        last_migration_report=SimpleNamespace(
            initialized_new_keys=("actor_module.compliance_residual.weight",),
        )
    )
    resumed = SimpleNamespace(last_migration_report=None)
    assert (
        validate_checkpoint_load_semantics(
            migrated,
            get_review_role("release_baseline"),
        )
        == "legacy_migration_strict"
    )
    assert (
        validate_checkpoint_load_semantics(
            resumed,
            get_review_role("chip_hard_off"),
        )
        == "native_strict_resume"
    )
    with pytest.raises(AssertionError, match="did not use"):
        validate_checkpoint_load_semantics(
            resumed,
            get_review_role("release_baseline"),
        )
    with pytest.raises(AssertionError, match="unexpectedly"):
        validate_checkpoint_load_semantics(
            migrated,
            get_review_role("chip_hard_off"),
        )


def test_protocol_pins_masks_matched_force_bytes_and_separate_chip_sign():
    protocol = DeterministicForceProtocol()
    frame_count = 100
    active_frame = 50
    release = protocol.sample(get_review_role("release_baseline"), active_frame, frame_count)
    no_contact = protocol.sample(
        get_review_role("enabled_no_contact"), active_frame, frame_count
    )
    assert release.compliance_enabled is False
    assert no_contact.compliance_enabled is True
    assert np.count_nonzero(release.force_on_robot_world_n) == 0
    assert np.count_nonzero(no_contact.force_on_robot_world_n) == 0
    assert np.count_nonzero(no_contact.active_site_mask) == 0

    for prefix, expected_mask in (
        ("single_left", [True, False]),
        ("single_right", [False, True]),
        ("simultaneous", [True, True]),
    ):
        stiff = protocol.sample(get_review_role(f"{prefix}_stiff"), active_frame, frame_count)
        compliant = protocol.sample(
            get_review_role(f"{prefix}_compliant"), active_frame, frame_count
        )
        assert stiff.force_on_robot_world_n.tobytes() == (
            compliant.force_on_robot_world_n.tobytes()
        )
        assert stiff.compliance_m_per_n.tobytes() == compliant.compliance_m_per_n.tobytes()
        assert stiff.active_site_mask.tolist() == expected_mask
        assert np.max(np.linalg.norm(stiff.force_on_robot_world_n, axis=-1)) == 5.0
        original = np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0
        selected = chip_selected_target(original, compliant)
        expected = original - compliant.compliance_m_per_n * compliant.force_on_robot_world_n
        np.testing.assert_array_equal(selected, expected)
        displacement = selected - original
        active = compliant.active_site_mask
        assert np.all(
            np.sum(displacement[active] * compliant.force_on_robot_world_n[active], axis=-1)
            < 0.0
        )

    start, stop, _ = protocol.active_bounds(frame_count)
    for frame in (0, start - 1, stop, frame_count - 1):
        sample = protocol.sample(get_review_role("simultaneous_compliant"), frame, frame_count)
        assert np.count_nonzero(sample.force_on_robot_world_n) == 0
        assert np.count_nonzero(sample.active_site_mask) == 0


@pytest.mark.parametrize("motion_file", (ORIGINAL_MOTION, MIRRORED_MOTION))
@pytest.mark.parametrize("role_name", REVIEW_ROLE_NAMES)
def test_hydra_composes_every_role_for_both_audited_motions(role_name, motion_file):
    role = get_review_role(role_name)
    cfg = compose_review_config(
        role,
        motion_file=motion_file,
        smpl_motion_dir=SMPL_DIR,
        seed=0,
        experiment_dir=Path("/tmp/chip_review_hydra_no_write"),
    )
    assert cfg.compliance_review_role.name == role.name
    assert cfg.compliance_review_role.checkpoint_kind == role.checkpoint_kind
    assert int(cfg.num_envs) == 1
    assert tuple(cfg.manager_env.commands.motion.body_names) == SONIC_RELEASE_TRACKING_BODY_NAMES
    assert tuple(cfg.manager_env.commands.force.site_names) == REVIEW_SITE_NAMES
    assert cfg.manager_env.commands.motion.motion_lib_cfg.motion_file == str(motion_file)
    assert cfg.manager_env.config.terrain_type == "plane"
    assert cfg.manager_env.config.render_results is True
    assert cfg.manager_env.config.render_width == REVIEW_PANEL_WIDTH
    assert cfg.manager_env.config.render_height == REVIEW_PANEL_HEIGHT
    assert tuple(cfg.manager_env.config.eval_camera_offset) == REVIEW_CAMERA_EYE_OFFSET_M
    assert float(cfg.manager_env.config.eval_camera_lookat_height) == 0.9
    assert sorted(cfg.manager_env.events.keys()) == ["_target_", "compliance_force_reset"]


class _FakeState:
    def __init__(self):
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._enabled = torch.zeros(1, dtype=torch.bool)
        self._site_mask = torch.zeros(1, 2, dtype=torch.bool)
        self._compliance = torch.zeros(1, 2, 3)
        self._force = torch.zeros(1, 2, 3)
        self._peak = torch.zeros(1, 2, 3)
        self._pulse = torch.zeros(1, dtype=torch.bool)

    @property
    def enabled(self):
        return self._enabled.clone()

    @property
    def site_mask(self):
        return self._site_mask.clone()

    @property
    def compliance(self):
        return self._compliance.clone()

    @property
    def force_on_robot_w(self):
        return self._force.clone()

    @property
    def peak_force_on_robot_w(self):
        return self._peak.clone()

    @property
    def pulse_active(self):
        return self._pulse.clone()

    def set_samples(self, env_ids, *, enabled, site_mask, compliance, force_on_robot_w):
        assert env_ids.tolist() == [0]
        requested = enabled[:, None] & site_mask
        self._enabled[:] = enabled
        self._site_mask[:] = site_mask
        self._compliance[:] = torch.where(requested[..., None], compliance, 0.0)
        self._force[:] = torch.where(requested[..., None], force_on_robot_w, 0.0)
        self._peak[:] = self._force
        self._pulse.zero_()

    def zero(self):
        self._enabled.zero_()
        self._site_mask.zero_()
        self._compliance.zero_()
        self._force.zero_()
        self._peak.zero_()
        self._pulse.zero_()


class _FakeWrench:
    def __init__(self):
        self.forces = torch.zeros(1, 2, 3)
        self.torques = torch.zeros(1, 2, 3)
        self.clear_count = 0

    def set_world_forces_prevalidated(
        self,
        forces,
        *,
        body_quaternions_wxyz,
        application_offsets_local,
    ):
        assert body_quaternions_wxyz.shape == (1, 2, 4)
        assert application_offsets_local.shape == (1, 2, 3)
        self.forces.copy_(forces)
        self.torques.zero_()

    def clear(self):
        self.forces.zero_()
        self.torques.zero_()
        self.clear_count += 1


class _FakeCommand:
    def __init__(self):
        self.state = _FakeState()
        self.sites = SimpleNamespace(
            spec=SimpleNamespace(
                site_names=REVIEW_SITE_NAMES,
                common_frame=CartesianFrameSpec.world(),
            )
        )
        self.robot = SimpleNamespace(
            data=SimpleNamespace(body_pos_w=torch.zeros(1, 1, 3))
        )
        self.anchor_body_index = 0
        self.cfg = SimpleNamespace(max_net_force_n=30.0, max_net_torque_nm=20.0)
        self.wrench = _FakeWrench()
        self._application_offsets_local = torch.zeros(1, 2, 3)
        self._wrench_write_gate = WrenchWriteGate()

    def _anchor_pose_w(self):
        return None, None

    def current_site_positions_w(self):
        return torch.tensor([[[0.2, 0.3, 1.0], [0.2, -0.3, 1.0]]])

    def current_site_quaternions_wxyz(self):
        result = torch.zeros(1, 2, 4)
        result[..., 0] = 1.0
        return result

    def reset_envs(self, env_ids):
        assert env_ids is None
        self.state.zero()
        self.wrench.clear()


def _fake_action(observations):
    base = observations["base"]
    command = observations["compliance_command"]
    enabled = command[:, :1]
    active = (command[:, 1:3].sum(dim=-1, keepdim=True) > 0).to(command.dtype)
    residual = enabled * active * command[:, 3:].sum(dim=-1, keepdim=True)
    return base + residual


def test_fake_manager_applies_exact_roles_actions_and_clears_reset_wrench():
    frame_count = 100
    active_frame = 50
    applied = {}
    for role_name in REVIEW_ROLE_NAMES:
        command = _FakeCommand()
        role = get_review_role(role_name)
        driver = SonicReviewProtocolDriver(command, role)
        driver.reset()
        assert command.wrench.clear_count == 1
        result = driver.apply(active_frame, frame_count)
        applied[role_name] = result
        command_obs = torch.cat(
            (
                result.command_enabled.to(torch.float32).unsqueeze(-1),
                result.active_site_mask.to(torch.float32),
                result.compliance_m_per_n.reshape(1, -1),
            ),
            dim=-1,
        )
        observations = {"base": torch.ones(1, 1), "compliance_command": command_obs}
        action = _fake_action(gate_actor_observations(observations, role))
        if role.actor_hard_off or not role.active_site_names:
            torch.testing.assert_close(action, torch.ones_like(action), rtol=0.0, atol=0.0)
        else:
            assert float(action.item()) > 1.0
        if role.external_force_enabled:
            assert int(torch.count_nonzero(command.wrench.forces)) > 0
            driver.reset()
            assert int(torch.count_nonzero(command.wrench.forces)) == 0
            assert int(torch.count_nonzero(command.wrench.torques)) == 0
            assert int(torch.count_nonzero(command.state.force_on_robot_w)) == 0
            assert int(torch.count_nonzero(command.state.compliance)) == 0
            assert int(torch.count_nonzero(command.state.site_mask)) == 0

    for prefix in ("single_left", "single_right", "simultaneous"):
        stiff = applied[f"{prefix}_stiff"]
        compliant = applied[f"{prefix}_compliant"]
        assert torch.equal(stiff.force_on_robot_world_n, compliant.force_on_robot_world_n)
        assert torch.equal(stiff.active_site_mask, compliant.active_site_mask)
        assert torch.equal(stiff.compliance_m_per_n, compliant.compliance_m_per_n)


def _snapshot(frame: int, *, active: bool = False) -> SonicReviewSnapshot:
    sites = len(REVIEW_SITE_NAMES)
    points = len(SONIC_RELEASE_TRACKING_BODY_NAMES)
    original = np.zeros((sites, 3), dtype=np.float32)
    force = np.zeros_like(original)
    compliance = np.zeros_like(original)
    mask = np.zeros(sites, dtype=np.bool_)
    if active:
        force[0, 1] = 5.0
        compliance[0, :] = 0.02
        mask[0] = True
    selected = original - compliance * force
    quaternions = np.zeros((sites, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    points_array = np.zeros((points, 3), dtype=np.float32)
    return SonicReviewSnapshot(
        reference_frame=frame,
        original_site_positions_m=original,
        selected_site_positions_m=selected,
        measured_site_positions_m=original.copy(),
        original_site_orientations_xyzw=quaternions,
        measured_site_orientations_xyzw=quaternions.copy(),
        reference_points_global_m=points_array,
        measured_points_global_m=points_array.copy(),
        reference_points_local_m=points_array.copy(),
        measured_points_local_m=points_array.copy(),
        force_on_robot_n=force,
        force_on_robot_world_n=force.copy(),
        force_on_robot_common_n=force.copy(),
        compliance_m_per_n=compliance,
        active_site_mask=mask,
    )


def test_trace_accumulator_requires_exact_natural_timeout_and_no_reset_suffix():
    accumulator = ReviewTraceAccumulator(
        role=get_review_role("single_left_compliant"),
        motion_id="original",
        seed=0,
        point_ids=SONIC_RELEASE_TRACKING_BODY_NAMES,
    )
    for frame in range(8):
        accumulator.append(
            _snapshot(frame, active=2 <= frame <= 5),
            policy_action=np.zeros(29, dtype=np.float32),
            reset=frame == 0,
            terminal=frame == 7,
            success=frame == 7,
            fall=False,
        )
    trace = accumulator.finish(expected_frame_count=8)
    assert trace.frame_indices.tolist() == list(range(8))
    assert trace.terminal_mask.tolist() == [False] * 7 + [True]
    assert trace.reset_mask.tolist() == [True] + [False] * 7
    np.testing.assert_array_equal(
        trace.selected_site_positions_m,
        trace.original_site_positions_m
        - trace.compliance_m_per_n * trace.force_on_robot_n,
    )
    with pytest.raises(RuntimeError, match="auto-reset suffix"):
        accumulator.append(
            _snapshot(8),
            policy_action=np.zeros(29, dtype=np.float32),
            reset=False,
            terminal=False,
            success=False,
            fall=False,
        )


def _metadata(frame_index: int) -> ReviewFrameMetadata:
    return ReviewFrameMetadata(
        role="single_left_compliant",
        branch_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        motion_id="original",
        seed=0,
        frame_index=frame_index,
        timestamp_s=frame_index / 50.0,
        active_site_names=(REVIEW_SITE_NAMES[0],),
        force_norms_n=(5.0, 0.0),
        compliance_m_per_n=0.02,
    )


def test_camera_rgba_overlay_pose_and_frame_sample_order(tmp_path: Path):
    rgba = np.zeros((64, 96, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgb = normalize_rgb_frame(rgba, width=96, height=64)
    assert rgb.shape == (64, 96, 3)
    overlaid = overlay_review_metadata(rgb, _metadata(0))
    assert overlaid.shape == rgb.shape
    assert np.count_nonzero(overlaid) > 0

    calls = []

    class Camera:
        def __init__(self):
            self.data = SimpleNamespace(
                output={"rgb": torch.zeros(1, 64, 96, 4, dtype=torch.uint8)}
            )

        def set_world_poses_from_view(self, eye, lookat):
            calls.append(("pose", eye.clone(), lookat.clone()))

        def update(self, dt):
            calls.append(("update", dt))

    camera = Camera()

    class Scene:
        sensors = {"eval_camera": camera}

        def __getitem__(self, name):
            assert name == "eval_camera"
            return camera

    raw_env = SimpleNamespace(
        scene=Scene(),
        sim=SimpleNamespace(render=lambda: calls.append(("render",))),
    )
    motion = SimpleNamespace(robot_body_pos_w=torch.zeros(1, 14, 3))
    captured = capture_review_frame(raw_env, motion, width=96, height=64)
    assert captured.shape == (64, 96, 3)
    assert [entry[0] for entry in calls] == ["pose", "render", "update"]
    np.testing.assert_allclose(calls[0][1][0].numpy(), REVIEW_CAMERA_EYE_OFFSET_M)
    np.testing.assert_allclose(calls[0][2][0].numpy(), (0.0, 0.0, 0.9))

    camera.data.output["rgb"] = torch.zeros(1, 64, 96, 4, dtype=torch.float32)
    with pytest.raises(TypeError, match="uint8"):
        capture_review_frame(raw_env, motion, width=96, height=64)

    video = tmp_path / "ordered.mp4"
    with AtomicReviewVideoWriter(video, width=96, height=64) as writer:
        writer.append(rgb, _metadata(0))
        with pytest.raises(AssertionError, match="frame index"):
            writer.append(rgb, _metadata(2))
        writer.append(rgb, _metadata(1))
    probe, _ = probe_video_with_sha256(video)
    assert probe["codec_name"] == "h264"
    assert probe["pixel_format"] == "yuv420p"
    assert probe["frame_rate"] == "50"
    assert probe["frame_count"] == 2


def test_atomic_camera_writer_refuses_collisions_and_cleans_partial(tmp_path: Path):
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        AtomicReviewVideoWriter(existing, width=96, height=64)
    assert existing.read_bytes() == b"owned"

    output = tmp_path / "aborted.mp4"
    with pytest.raises(RuntimeError, match="stop"):
        with AtomicReviewVideoWriter(output, width=96, height=64) as writer:
            writer.append(frame, _metadata(0))
            raise RuntimeError("stop")
    assert not output.exists()
    assert not (tmp_path / ".aborted.partial.mp4").exists()


def test_actual_compositor_is_aligned_h264_yuv420p_and_atomic(tmp_path: Path):
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    panels = []
    for name in ("stiff", "compliant"):
        panel = tmp_path / f"{name}.mp4"
        with AtomicReviewVideoWriter(panel, width=96, height=64) as writer:
            writer.append(frame, _metadata(0))
            writer.append(frame, _metadata(1))
        panels.append(panel)

    composite = tmp_path / "comparison.mp4"
    compose_review_panels(panels[0], panels[1], composite)
    probe, _ = probe_video_with_sha256(composite)
    assert probe["codec_name"] == "h264"
    assert probe["pixel_format"] == "yuv420p"
    assert probe["frame_rate"] == "50"
    assert probe["frame_count"] == 2
    assert probe["width"] == 192
    assert probe["height"] == 64
    with pytest.raises(FileExistsError):
        compose_review_panels(panels[0], panels[1], composite)

    oversized = tmp_path / "oversized.mp4"
    with pytest.raises(ValueError, match="max_output_bytes"):
        compose_review_panels(
            panels[0],
            panels[1],
            oversized,
            max_output_bytes=1,
        )
    assert not oversized.exists()
    assert not (tmp_path / ".oversized.partial.mp4").exists()


def test_video_writer_settings_are_pinned(monkeypatch, tmp_path: Path):
    captured = {}

    class Writer:
        def append_data(self, frame):
            pass

        def close(self):
            pass

    def fake_get_writer(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return Writer()

    import imageio.v2 as imageio

    from gear_sonic.compliance_control.adapters.sonic.review import camera as camera_module

    monkeypatch.setattr(imageio, "get_writer", fake_get_writer)
    writer = camera_module._open_imageio_writer(tmp_path / "settings.mp4", fps=50)
    writer.close()
    assert captured["fps"] == 50
    assert captured["codec"] == "libx264"
    assert captured["pixelformat"] == "yuv420p"
    assert captured["macro_block_size"] is None


@pytest.mark.parametrize("script", SCRIPTS)
def test_review_entrypoint_help_is_zero_exit_and_has_no_isaac_top_level(script):
    completed = subprocess.run(
        (sys.executable, "-B", str(script), "--help"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in completed.stdout.lower()
    tree = ast.parse(script.read_text(encoding="utf-8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("isaaclab" not in ast.unparse(node) for node in top_imports)


def test_all_three_dry_runs_are_no_write_and_do_not_launch_app(tmp_path: Path):
    output_root = tmp_path / "must_remain_absent"
    collector = subprocess.run(
        (
            sys.executable,
            "-B",
            str(SCRIPTS[0]),
            "--role",
            "release_baseline",
            "--motion-id",
            "original",
            "--motion-file",
            str(ORIGINAL_MOTION),
            "--smpl-motion-dir",
            str(SMPL_DIR),
            "--official-checkpoint",
            str(OFFICIAL_CHECKPOINT),
            "--trained-checkpoint",
            str(TRAINED_CHECKPOINT),
            "--output-root",
            str(output_root),
            "--dry-run",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    collector_plan = json.loads(collector.stdout)
    assert collector_plan["app_launcher_started"] is False
    assert not output_root.exists()

    motion_root = output_root / "original"
    for script, extra in (
        (SCRIPTS[1], ("--motion-root", str(motion_root))),
        (
            SCRIPTS[2],
            ("--motion-root", str(motion_root), "--branch-commit", "a" * 40),
        ),
    ):
        completed = subprocess.run(
            (sys.executable, "-B", str(script), *extra, "--dry-run"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(completed.stdout)
        assert plan["simulator_imported"] is False
        assert not output_root.exists()
