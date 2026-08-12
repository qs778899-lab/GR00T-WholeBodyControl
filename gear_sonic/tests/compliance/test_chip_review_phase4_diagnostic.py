"""CPU gates for the explicitly non-formal 32-frame rendered smoke."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
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
from gear_sonic.compliance_control.adapters.sonic.review.diagnostic import (
    DIAGNOSTIC_TRACE_FIELDS,
    DIAGNOSTIC_TRACE_SCHEMA,
    ReviewDiagnosticAccumulator,
    write_diagnostic_trace_atomic,
)
from gear_sonic.compliance_control.adapters.sonic.review.roles import REVIEW_SITE_NAMES
from gear_sonic.compliance_control.adapters.sonic.review.runtime import (
    validate_finite_observations,
    validate_owned_composer_rows_cleared,
)
from gear_sonic.compliance_control.adapters.sonic.review.trace import (
    SonicReviewSnapshot,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = Path(
    "/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control"
)
COLLECTOR = REPOSITORY_ROOT / "gear_sonic/scripts/run_chip_review_collect.py"
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
