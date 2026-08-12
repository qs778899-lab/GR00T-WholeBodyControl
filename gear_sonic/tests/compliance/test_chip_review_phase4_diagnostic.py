"""CPU gates for the explicitly non-formal 32-frame rendered smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np

try:
    import pytest
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - unittest portability gate
    raise unittest.SkipTest("pytest and torch are required for diagnostic tests") from error

from gear_sonic.compliance_control.adapters.sonic.contracts import (
    SONIC_RELEASE_TRACKING_BODY_NAMES,
)
from gear_sonic.compliance_control.adapters.sonic.review.camera import (
    AtomicReviewVideoWriter,
    ReviewFrameMetadata,
)
from gear_sonic.compliance_control.adapters.sonic.review.config import (
    ReviewArtifactPaths,
)
from gear_sonic.compliance_control.adapters.sonic.review.diagnostic import (
    DIAGNOSTIC_TRACE_FIELDS,
    DIAGNOSTIC_TRACE_SCHEMA,
    ReviewDiagnosticAccumulator,
    write_diagnostic_trace_atomic,
)
from gear_sonic.compliance_control.adapters.sonic.review.protocol import (
    DeterministicForceProtocol,
)
from gear_sonic.compliance_control.adapters.sonic.review.roles import (
    REVIEW_SITE_NAMES,
    get_review_role,
)
from gear_sonic.compliance_control.adapters.sonic.review.runtime import (
    collect_sonic_review_role,
    validate_finite_observations,
    validate_owned_composer_rows_cleared,
)
from gear_sonic.compliance_control.adapters.sonic.review.trace import (
    SonicReviewSnapshot,
)
from gear_sonic.compliance_control.review import (
    probe_video_with_sha256,
    write_report_json_atomic,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control"
)
COLLECTOR = REPOSITORY_ROOT / "gear_sonic/scripts/run_chip_review_collect.py"
DIAGNOSTIC_AUDIT = (
    REPOSITORY_ROOT
    / "tasks/chip_runtime_video_validation/artifacts/phase4_rendered_smoke_audit.py"
)
MOTION = (
    ASSET_ROOT
    / "official_assets/sample_data/robot_filtered/210531/"
    "walk_forward_amateur_001__A001.pkl"
)
SMPL_DIR = ASSET_ROOT / "official_assets/sample_data/smpl_filtered"
OFFICIAL_CHECKPOINT = ASSET_ROOT / "official_assets/sonic_release/last.pt"
TRAINED_CHECKPOINT = (
    ASSET_ROOT
    / "runs/chip/phase4_acceptance_resume_fix/"
    "compliance_residual_step6_resume/last.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(frame: int) -> SonicReviewSnapshot:
    sites = len(REVIEW_SITE_NAMES)
    points = len(SONIC_RELEASE_TRACKING_BODY_NAMES)
    site_vectors = np.zeros((sites, 3), dtype=np.float32)
    point_vectors = np.zeros((points, 3), dtype=np.float32)
    quaternions = np.zeros((sites, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    return SonicReviewSnapshot(
        reference_frame=frame,
        original_site_positions_m=site_vectors,
        selected_site_positions_m=site_vectors.copy(),
        measured_site_positions_m=site_vectors.copy(),
        original_site_orientations_xyzw=quaternions,
        measured_site_orientations_xyzw=quaternions.copy(),
        reference_points_global_m=point_vectors,
        measured_points_global_m=point_vectors.copy(),
        reference_points_local_m=point_vectors.copy(),
        measured_points_local_m=point_vectors.copy(),
        force_on_robot_n=site_vectors.copy(),
        force_on_robot_world_n=site_vectors.copy(),
        force_on_robot_common_n=site_vectors.copy(),
        compliance_m_per_n=site_vectors.copy(),
        active_site_mask=np.zeros(sites, dtype=np.bool_),
    )


def _active_snapshot(frame: int) -> SonicReviewSnapshot:
    snapshot = _snapshot(frame)
    active = 7 <= frame < 25
    if not active:
        return snapshot
    force = snapshot.force_on_robot_world_n.copy()
    force[0, 1] = 5.0
    force[1, 1] = -5.0
    compliance = np.full_like(force, 0.02)
    selected = snapshot.original_site_positions_m - compliance * force
    return SonicReviewSnapshot(
        reference_frame=frame,
        original_site_positions_m=snapshot.original_site_positions_m,
        selected_site_positions_m=selected,
        measured_site_positions_m=selected.copy(),
        original_site_orientations_xyzw=snapshot.original_site_orientations_xyzw,
        measured_site_orientations_xyzw=snapshot.measured_site_orientations_xyzw,
        reference_points_global_m=snapshot.reference_points_global_m,
        measured_points_global_m=snapshot.measured_points_global_m,
        reference_points_local_m=snapshot.reference_points_local_m,
        measured_points_local_m=snapshot.measured_points_local_m,
        force_on_robot_n=force.copy(),
        force_on_robot_world_n=force,
        force_on_robot_common_n=force.copy(),
        compliance_m_per_n=compliance,
        active_site_mask=np.ones(2, dtype=np.bool_),
    )


def test_diagnostic_trace_is_explicit_fixed_cutoff_and_atomic(tmp_path: Path):
    accumulator = ReviewDiagnosticAccumulator(
        role="simultaneous_compliant",
        motion_id="original",
        seed=0,
    )
    for frame in range(8):
        accumulator.append(
            _snapshot(frame),
            policy_action=np.zeros(29, dtype=np.float32),
            terminal=False,
            timed_out=False,
            fall=False,
        )
    arrays = accumulator.finish(expected_frame_count=8)
    assert set(arrays) == set(DIAGNOSTIC_TRACE_FIELDS)
    assert arrays["schema_version"].item() == DIAGNOSTIC_TRACE_SCHEMA
    assert arrays["frame_indices"].tolist() == list(range(8))
    assert arrays["reference_frames"].tolist() == list(range(8))
    assert arrays["reset_mask"].tolist() == [True] + [False] * 7
    assert not np.any(arrays["terminal_mask"])
    output = tmp_path / "trace.npz"
    write_diagnostic_trace_atomic(arrays, output)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == DIAGNOSTIC_TRACE_SCHEMA
        assert archive["policy_actions"].shape == (8, 29)
    with pytest.raises(FileExistsError):
        write_diagnostic_trace_atomic(arrays, output)
    assert not list(tmp_path.glob(".*.tmp"))


def test_diagnostic_trace_rejects_nonfinite_and_early_terminal():
    accumulator = ReviewDiagnosticAccumulator(
        role="simultaneous_compliant",
        motion_id="original",
        seed=0,
    )
    invalid = _snapshot(0)
    invalid.original_site_orientations_xyzw[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.append(
            invalid,
            policy_action=np.zeros(29, dtype=np.float32),
            terminal=False,
            timed_out=False,
            fall=False,
        )

    for frame in range(8):
        terminal = frame == 7
        accumulator.append(
            _snapshot(frame),
            policy_action=np.zeros(29, dtype=np.float32),
            terminal=terminal,
            timed_out=terminal,
            fall=False,
        )
    with pytest.raises(AssertionError, match="terminated"):
        accumulator.finish(expected_frame_count=8)


def test_diagnostic_observation_finiteness_is_fail_closed():
    validate_finite_observations({"actor_obs": torch.zeros(1, 4)})
    with pytest.raises(ValueError, match="actor_obs"):
        validate_finite_observations(
            {"actor_obs": torch.tensor([[0.0, float("nan")]])}
        )
    with pytest.raises(TypeError, match="string names"):
        validate_finite_observations({"actor_obs": np.zeros((1, 4))})


def test_real_composer_reset_evidence_checks_only_owned_rows():
    force = torch.zeros(1, 5, 3)
    torque = torch.zeros_like(force)
    force[0, 4, 0] = 7.0
    torque[0, 4, 1] = 3.0
    command = type(
        "Command",
        (),
        {
            "robot": type(
                "Robot",
                (),
                {
                    "permanent_wrench_composer": type(
                        "Composer",
                        (),
                        {
                            "composed_force_as_torch": force,
                            "composed_torque_as_torch": torque,
                        },
                    )()
                },
            )(),
            "sites": type("Sites", (), {"articulation_indices": (1, 3)})(),
        },
    )()
    validate_owned_composer_rows_cleared(command)
    force[0, 1, 0] = 1.0
    with pytest.raises(AssertionError, match="force rows"):
        validate_owned_composer_rows_cleared(command)


def test_collector_diagnostic_dry_run_is_labelled_and_no_write(tmp_path: Path):
    output_root = tmp_path / "must_remain_absent"
    completed = subprocess.run(
        (
            sys.executable,
            "-B",
            str(COLLECTOR),
            "--role",
            "simultaneous_compliant",
            "--motion-id",
            "original",
            "--motion-file",
            str(MOTION),
            "--smpl-motion-dir",
            str(SMPL_DIR),
            "--official-checkpoint",
            str(OFFICIAL_CHECKPOINT),
            "--trained-checkpoint",
            str(TRAINED_CHECKPOINT),
            "--output-root",
            str(output_root),
            "--diagnostic-frames",
            "32",
            "--dry-run",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["diagnostic_frames"] == 32
    assert plan["publication_kind"] == "diagnostic_fixed_cutoff_nonformal"
    assert len(plan["motion_sha256"]) == 64
    assert plan["app_launcher_started"] is False
    assert not output_root.exists()


def test_independent_diagnostic_auditor_accepts_trace_bound_video(tmp_path: Path):
    branch_commit = "a" * 40
    run_root = tmp_path / "diagnostic"
    role_root = run_root / "original" / "simultaneous_compliant"
    role_root.mkdir(parents=True)
    accumulator = ReviewDiagnosticAccumulator(
        role="simultaneous_compliant",
        motion_id="original",
        seed=0,
    )
    checkpoint_sha256 = _sha256(TRAINED_CHECKPOINT)
    motion_sha256 = _sha256(MOTION)
    video_path = role_root / "panel.mp4"
    base_frame = np.zeros((720, 960, 3), dtype=np.uint8)
    with AtomicReviewVideoWriter(video_path) as writer:
        for frame in range(32):
            snapshot = _active_snapshot(frame)
            accumulator.append(
                snapshot,
                policy_action=np.zeros(29, dtype=np.float32),
                terminal=False,
                timed_out=False,
                fall=False,
            )
            writer.append(
                base_frame,
                ReviewFrameMetadata(
                    role="simultaneous_compliant",
                    branch_commit=branch_commit,
                    checkpoint_sha256=checkpoint_sha256,
                    motion_id="original",
                    seed=0,
                    frame_index=frame,
                    timestamp_s=frame / 50.0,
                    active_site_names=("left_wrist", "right_wrist")
                    if 7 <= frame < 25
                    else (),
                    force_norms_n=(5.0, 5.0) if 7 <= frame < 25 else (0.0, 0.0),
                    compliance_m_per_n=0.02 if 7 <= frame < 25 else 0.0,
                ),
            )
    trace_path = role_root / "trace.npz"
    write_diagnostic_trace_atomic(
        accumulator.finish(expected_frame_count=32),
        trace_path,
    )
    video_probe, video_sha256 = probe_video_with_sha256(video_path)
    summary = {
        "schema_version": "sonic_chip_review_diagnostic_v1",
        "role": "simultaneous_compliant",
        "checkpoint_kind": "trained",
        "checkpoint": str(TRAINED_CHECKPOINT.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load_semantics": "native_strict_resume",
        "branch_commit": branch_commit,
        "motion_id": "original",
        "motion_file": str(MOTION.resolve()),
        "motion_sha256": motion_sha256,
        "seed": 0,
        "frame_count": 32,
        "source_motion_frame_count": 300,
        "trace_kind": "diagnostic_fixed_cutoff_nonformal",
        "natural_timeout_count": 0,
        "fall_count": 0,
        "finite_observations": True,
        "finite_actions": True,
        "trace_reset_count": 1,
        "command_reset_count": 2,
        "composer_owned_reset_force_peak_n": 0.0,
        "composer_owned_reset_torque_peak_nm": 0.0,
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "panel_video": str(video_path.resolve()),
        "panel_video_sha256": video_sha256,
        "panel_video_probe": video_probe,
        "body_names": list(SONIC_RELEASE_TRACKING_BODY_NAMES),
        "site_names": list(REVIEW_SITE_NAMES),
        "force_evaluation_frame": "world",
        "force_common_frame": "heading_local",
        "peak_world_force_n": 5.0,
        "peak_latent_residual": 0.1,
        "observation_dims": {
            "actor_obs": 930,
            "critic_obs": 1645,
            "tokenizer": 1761,
            "compliance_target": 60,
            "compliance_command": 9,
            "compliance_force": 6,
        },
    }
    write_report_json_atomic(summary, role_root / "summary.json")
    completed = subprocess.run(
        (
            sys.executable,
            "-B",
            str(DIAGNOSTIC_AUDIT),
            "--run-root",
            str(run_root),
            "--branch-commit",
            branch_commit,
            "--trained-checkpoint",
            str(TRAINED_CHECKPOINT),
            "--motion-file",
            str(MOTION),
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CHIP_PHASE4_RENDERED_SMOKE_AUDIT_PASS" in completed.stdout


def test_collector_runtime_orchestration_publishes_complete_diagnostic(
    tmp_path: Path,
    monkeypatch,
):
    from omegaconf import OmegaConf

    import gear_sonic.compliance_control.adapters.sonic.review.runtime as runtime

    expected_dims = {
        "actor_obs": 930,
        "critic_obs": 1645,
        "tokenizer": 1761,
        "compliance_target": 60,
        "compliance_command": 9,
        "compliance_force": 6,
    }
    observations = {
        name: torch.zeros(1, width, dtype=torch.float32)
        for name, width in expected_dims.items()
    }
    protocol = DeterministicForceProtocol()
    runtime_state = SimpleNamespace(frame=0, raw_env=None, driver=None)

    class FakeSonicComplianceCommand:
        def __init__(self):
            self.operational_enabled = True
            self.sites = SimpleNamespace(
                articulation_indices=(1, 2),
                spec=SimpleNamespace(
                    site_names=REVIEW_SITE_NAMES,
                    common_frame=SimpleNamespace(
                        kind=SimpleNamespace(value="heading_local")
                    ),
                ),
            )
            composer = SimpleNamespace(
                composed_force_as_torch=torch.zeros(1, 3, 3),
                composed_torque_as_torch=torch.zeros(1, 3, 3),
            )
            self.robot = SimpleNamespace(permanent_wrench_composer=composer)
            self.current = None

    class FakeMotionLibrary:
        @staticmethod
        def get_time_step_total(motion_ids):
            assert motion_ids.tolist() == [0]
            return torch.tensor([300], dtype=torch.int64)

    class FakeRawEnvironment:
        def __init__(self, *, cfg, render_mode):
            assert cfg.sim.device == "cpu"
            assert render_mode is None
            self.motion = SimpleNamespace(
                cfg=SimpleNamespace(body_names=SONIC_RELEASE_TRACKING_BODY_NAMES),
                motion_start_time_steps=torch.zeros(1, dtype=torch.int64),
                motion_ids=torch.zeros(1, dtype=torch.int64),
                motion_lib=FakeMotionLibrary(),
            )
            self.command = FakeSonicComplianceCommand()
            self.closed = False
            runtime_state.raw_env = self

        def close(self):
            self.closed = True

    class FakeWrappedEnvironment:
        def __init__(self, raw_env, config):
            assert config["headless"] is True
            self.raw_env = raw_env
            self.motion_command = raw_env.motion
            self.force_command = raw_env.command
            self.device = torch.device("cpu")
            self.evaluating = False

        def set_is_evaluating(self, value):
            self.evaluating = value

        def reset(self, *, flatten_dict_obs):
            assert flatten_dict_obs is True
            return {name: value.clone() for name, value in observations.items()}

        def step(self, action):
            assert tuple(action["actions"].shape) == (1, 29)
            return (
                {name: value.clone() for name, value in observations.items()},
                torch.zeros(1),
                torch.zeros(1, dtype=torch.long),
                {"time_outs": torch.zeros(1, dtype=torch.bool)},
            )

    class FakeActor:
        def __init__(self):
            self.last_migration_report = None
            self.actor_module = SimpleNamespace(
                _last_compliance_residual=torch.full((1, 1, 64), 0.125)
            )
            self.loaded = False

        def load_state_dict(self, state, *, strict):
            assert state == {"fixture": "trained"}
            assert strict is True
            self.loaded = True

        def eval(self):
            return self

        def init_rollout(self):
            assert self.loaded

        def act_inference(self, policy_observations, **kwargs):
            assert kwargs["skip_episode_attnmask"] is True
            assert kwargs["cur_dones"].tolist() == [0]
            assert set(policy_observations) == set(expected_dims)
            return torch.zeros(1, 29, dtype=torch.float32)

    actor = FakeActor()

    class FakeDriver:
        def __init__(self, command, role):
            self.command = command
            self.role = role
            self.reset_count = 0
            runtime_state.driver = self

        def reset(self):
            self.reset_count += 1
            composer = self.command.robot.permanent_wrench_composer
            composer.composed_force_as_torch[:, (1, 2)].zero_()
            composer.composed_torque_as_torch[:, (1, 2)].zero_()
            self.command.current = None

        def apply(self, frame_index, frame_count):
            sample = protocol.sample(self.role, frame_index, frame_count)
            force = torch.tensor(
                sample.force_on_robot_world_n,
                dtype=torch.float32,
            ).unsqueeze(0)
            compliance = torch.tensor(
                sample.compliance_m_per_n,
                dtype=torch.float32,
            ).unsqueeze(0)
            active = torch.tensor(sample.active_site_mask).unsqueeze(0)
            enabled = torch.tensor([sample.compliance_enabled])
            self.command.current = SimpleNamespace(
                frame=frame_index,
                force=force,
                compliance=compliance,
                active=active,
            )
            composer = self.command.robot.permanent_wrench_composer
            composer.composed_force_as_torch[:, (1, 2)] = force
            runtime_state.frame = frame_index
            return SimpleNamespace(
                force_on_robot_world_n=force,
                compliance_m_per_n=compliance,
                active_site_mask=active,
                command_enabled=enabled,
            )

    def fake_snapshot(motion, command):
        assert motion is runtime_state.raw_env.motion
        current = command.current
        assert current is not None
        original = np.zeros((2, 3), dtype=np.float32)
        force = current.force[0].numpy().copy()
        compliance = current.compliance[0].numpy().copy()
        selected = original - compliance * force
        points = np.zeros((14, 3), dtype=np.float32)
        orientations = np.zeros((2, 4), dtype=np.float32)
        orientations[:, 3] = 1.0
        return SonicReviewSnapshot(
            reference_frame=current.frame,
            original_site_positions_m=original,
            selected_site_positions_m=selected,
            measured_site_positions_m=selected.copy(),
            original_site_orientations_xyzw=orientations,
            measured_site_orientations_xyzw=orientations.copy(),
            reference_points_global_m=points,
            measured_points_global_m=points.copy(),
            reference_points_local_m=points.copy(),
            measured_points_local_m=points.copy(),
            force_on_robot_n=force,
            force_on_robot_world_n=force.copy(),
            force_on_robot_common_n=force.copy(),
            compliance_m_per_n=compliance,
            active_site_mask=current.active[0].numpy().copy(),
        )

    env_cfg = SimpleNamespace(
        seed=None,
        sim=SimpleNamespace(device=None),
        config={},
    )
    actor_payload = {"policy_state_dict": {"fixture": "trained"}}
    fake_isaac_envs = ModuleType("isaaclab.envs")
    fake_isaac_envs.ManagerBasedRLEnv = FakeRawEnvironment
    fake_command_module = ModuleType(
        "gear_sonic.compliance_control.adapters.sonic.isaaclab.command"
    )
    fake_command_module.SonicComplianceCommand = FakeSonicComplianceCommand
    fake_wrapper_module = ModuleType("gear_sonic.envs.wrapper.manager_env_wrapper")
    fake_wrapper_module.ManagerEnvWrapper = FakeWrappedEnvironment
    fake_common_module = ModuleType("gear_sonic.trl.utils.common")
    fake_common_module.custom_instantiate = lambda config: env_cfg
    monkeypatch.setitem(sys.modules, "isaaclab.envs", fake_isaac_envs)
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.compliance_control.adapters.sonic.isaaclab.command",
        fake_command_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.envs.wrapper.manager_env_wrapper",
        fake_wrapper_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "gear_sonic.trl.utils.common",
        fake_common_module,
    )
    monkeypatch.setattr(runtime, "SonicReviewProtocolDriver", FakeDriver)
    monkeypatch.setattr(runtime, "capture_sonic_review_snapshot", fake_snapshot)
    monkeypatch.setattr(
        runtime,
        "capture_review_frame",
        lambda raw_env, motion: np.full(
            (720, 960, 3),
            runtime_state.frame,
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "refresh_compliance_observations",
        lambda raw_env, values: dict(values),
    )
    monkeypatch.setattr(
        runtime,
        "_prepare_observation_contract",
        lambda env, raw_env, config, device: (actor, expected_dims, torch),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: actor_payload)

    motion_file = tmp_path / "motion.pkl"
    checkpoint = tmp_path / "trained.pt"
    motion_file.write_bytes(b"fixed-motion-fixture")
    checkpoint.write_bytes(b"fixed-trained-checkpoint-fixture")
    paths = ReviewArtifactPaths(
        tmp_path / "result",
        "original",
        "simultaneous_compliant",
    )
    config = OmegaConf.create(
        {"manager_env": {"config": {"experiment_dir": "unassigned"}}}
    )
    summary = collect_sonic_review_role(
        config=config,
        role=get_review_role("simultaneous_compliant"),
        motion_id="original",
        motion_file=motion_file,
        motion_sha256=_sha256(motion_file),
        seed=0,
        checkpoint=checkpoint,
        checkpoint_sha256=_sha256(checkpoint),
        branch_commit="b" * 40,
        paths=paths,
        device="cpu",
        diagnostic_frame_limit=8,
    )

    assert actor.loaded is True
    assert runtime_state.raw_env.closed is True
    assert runtime_state.driver.reset_count == 2
    assert summary["frame_count"] == 8
    assert summary["trace_kind"] == "diagnostic_fixed_cutoff_nonformal"
    assert summary["checkpoint_load_semantics"] == "native_strict_resume"
    assert summary["peak_world_force_n"] == 5.0
    assert summary["peak_latent_residual"] == 0.125
    assert summary["panel_video_probe"]["frame_count"] == 8
    assert paths.trace.is_file()
    assert paths.summary.is_file()
    assert paths.panel_video.is_file()
    with np.load(paths.trace, allow_pickle=False) as archive:
        assert archive["frame_indices"].tolist() == list(range(8))
        np.testing.assert_allclose(
            archive["selected_site_positions_m"],
            archive["original_site_positions_m"]
            - archive["compliance_m_per_n"]
            * archive["force_on_robot_world_n"],
            rtol=0.0,
            atol=0.0,
        )
    assert not list(paths.directory.rglob("*.tmp"))
    assert not list(paths.directory.rglob("*.part"))
