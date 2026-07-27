"""Portable Phase-5 alignment, tracking, compliance, and artifact contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from gear_sonic.compliance_control.postprocess import evaluation_io

from gear_sonic.compliance_control import (
    AlignedTrackingTrace,
    CartesianFrameSpec,
    PairedEvaluationThresholds,
    compare_aligned_tracking_traces,
    summarize_tracking_trace,
)
from gear_sonic.compliance_control.postprocess import (
    load_tracking_trace,
    paired_result_to_dict,
    save_tracking_trace,
)


class Phase5EvaluationTest(unittest.TestCase):
    def _trace(
        self,
        mode: str,
        *,
        samples: int = 5,
        bodies: int = 3,
        sites: int = 2,
        position_error_m: float = 0.01,
        endpoint_error_m: float = 0.01,
    ) -> AlignedTrackingTrace:
        reference = torch.zeros(samples, bodies, 3)
        actual = reference.clone()
        actual[..., 0] = position_error_m
        reference_sites = torch.zeros(samples, sites, 3)
        actual_sites = reference_sites.clone()
        actual_sites[..., 0] = endpoint_error_m
        reference_site_quaternions = torch.zeros(samples, sites, 4)
        reference_site_quaternions[..., 0] = 1.0
        actual_site_quaternions = reference_site_quaternions.clone()
        force = torch.zeros(samples, sites, 3)
        force[1:-1, :, 0] = torch.tensor([2.0, 4.0, 4.0])[: samples - 2, None]
        enabled = torch.zeros(samples, dtype=torch.bool)
        enabled[1:-1] = True
        site_mask = enabled.unsqueeze(-1).expand(samples, sites).clone()
        compliance = torch.zeros(samples, sites, 3)
        compliance[1:-1] = 0.02
        return AlignedTrackingTrace(
            mode=mode,
            body_names=tuple(f"body_{index}" for index in range(bodies)),
            site_names=tuple(f"site_{index}" for index in range(sites)),
            local_frame=CartesianFrameSpec.heading_local("pelvis"),
            sample_index=torch.arange(samples),
            episode_id=torch.zeros(samples, dtype=torch.int64),
            motion_id=torch.full((samples,), 7, dtype=torch.int64),
            reference_frame=torch.arange(10, 10 + samples),
            time_s=torch.arange(samples, dtype=torch.float64) * 0.02,
            valid=torch.ones(samples, dtype=torch.bool),
            reference_positions_w=reference,
            actual_positions_w=actual,
            reference_positions_local=reference.clone(),
            actual_positions_local=actual.clone(),
            reference_site_positions_w=reference_sites,
            actual_site_positions_w=actual_sites,
            reference_site_quaternions_wxyz=reference_site_quaternions,
            actual_site_quaternions_wxyz=actual_site_quaternions,
            force_on_robot_w=force,
            enabled=enabled,
            site_mask=site_mask,
            compliance_m_per_n=compliance,
            fell=False,
            horizon_reached=True,
        )

    def test_metrics_cover_tracking_endpoints_displacement_and_force(self) -> None:
        metrics = summarize_tracking_trace(self._trace("compliant"))
        self.assertEqual(metrics.valid_frames, 5)
        self.assertAlmostEqual(metrics.global_mpjpe_m, 0.01, places=6)
        self.assertAlmostEqual(metrics.local_mpjpe_m, 0.01, places=6)
        self.assertAlmostEqual(metrics.upper_endpoint_mpjpe_m, 0.01, places=6)
        self.assertAlmostEqual(metrics.exposed_upper_endpoint_mpjpe_m, 0.01, places=6)
        self.assertAlmostEqual(metrics.peak_force_n, 4.0, places=6)
        self.assertAlmostEqual(metrics.steady_force_mean_n, 4.0, places=6)
        self.assertEqual(metrics.per_site_exposed_frames, (3, 3))
        for values in (
            metrics.per_site_endpoint_rmse_m,
            metrics.per_site_endpoint_p95_m,
            metrics.per_site_exposed_endpoint_rmse_m,
            metrics.per_site_unexposed_endpoint_rmse_m,
        ):
            for value in values:
                self.assertAlmostEqual(value, 0.01)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.fall_rate, 0.0)
        self.assertEqual(metrics.upper_endpoint_orientation_rmse_rad, 0.0)

    def test_pair_is_exactly_keyed_and_tracking_budgeted(self) -> None:
        stiff = self._trace("stiff")
        compliant = self._trace(
            "compliant",
            position_error_m=0.02,
            endpoint_error_m=0.03,
        )
        result = compare_aligned_tracking_traces(
            stiff,
            compliant,
            thresholds=PairedEvaluationThresholds(
                min_aligned_frames=5,
                min_exposed_frames_per_site=3,
                max_upper_endpoint_regression_m=0.02,
                max_global_mpjpe_regression_m=0.01,
                max_local_mpjpe_regression_m=0.01,
                min_peak_force_n=2.0,
                max_peak_force_n=5.0,
            ),
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(
            result.compliance_response.displacement_mean_m,
            0.02,
            places=6,
        )
        self.assertAlmostEqual(
            result.compliance_response.displacement_along_force_mean_m,
            0.02,
            places=6,
        )
        for value in result.compliance_response.per_site_displacement_mean_m:
            self.assertAlmostEqual(value, 0.02)
        payload = paired_result_to_dict(result)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["aligned_frames"], 5)
        self.assertEqual(payload["compliant"]["per_site_exposed_frames"], (3, 3))
        self.assertIn("compliance_response", payload)

    def test_endpoint_rmse_p95_are_per_site_and_split_by_exposure(self) -> None:
        trace = self._trace("compliant")
        actual = trace.actual_site_positions_w.clone()
        actual[:, 0, 0] = torch.tensor([0.01, 0.02, 0.02, 0.02, 0.01])
        actual[:, 1, 0] = torch.tensor([0.03, 0.04, 0.04, 0.04, 0.03])
        metrics = summarize_tracking_trace(
            replace(trace, actual_site_positions_w=actual)
        )
        self.assertAlmostEqual(metrics.per_site_exposed_endpoint_rmse_m[0], 0.02)
        self.assertAlmostEqual(metrics.per_site_unexposed_endpoint_rmse_m[0], 0.01)
        self.assertAlmostEqual(metrics.per_site_exposed_endpoint_p95_m[1], 0.04)
        self.assertAlmostEqual(metrics.per_site_unexposed_endpoint_p95_m[1], 0.03)
        self.assertGreater(metrics.per_site_endpoint_rmse_m[1], 0.03)

    def test_steady_force_is_contiguous_pulse_tail_not_peak_threshold(self) -> None:
        trace = self._trace("compliant")
        force = trace.force_on_robot_w.clone()
        force[1:4, :, 0] = torch.tensor([10.0, 1.0, 1.0])[:, None]
        metrics = summarize_tracking_trace(
            replace(trace, force_on_robot_w=force),
            steady_tail_fraction=0.2,
        )
        self.assertAlmostEqual(metrics.peak_force_n, 10.0)
        self.assertAlmostEqual(metrics.steady_force_mean_n, 1.0)

    def test_orientation_error_is_sign_invariant_and_split_by_exposure(self) -> None:
        trace = self._trace("compliant")
        angles = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
        actual = trace.actual_site_quaternions_wxyz.clone()
        actual[:, 0, 0] = torch.cos(angles / 2.0)
        actual[:, 0, 1] = torch.sin(angles / 2.0)
        actual[:, 1] *= -1.0
        metrics = summarize_tracking_trace(
            replace(trace, actual_site_quaternions_wxyz=actual)
        )
        expected_all = torch.sqrt(torch.mean(torch.square(angles))).item()
        expected_exposed = torch.sqrt(torch.mean(torch.square(angles[1:4]))).item()
        expected_unexposed = torch.sqrt(
            torch.mean(torch.square(angles[torch.tensor([0, 4])]))
        ).item()
        self.assertAlmostEqual(
            metrics.per_site_orientation_rmse_rad[0],
            expected_all,
            places=5,
        )
        self.assertAlmostEqual(
            metrics.per_site_exposed_orientation_rmse_rad[0],
            expected_exposed,
            places=5,
        )
        self.assertAlmostEqual(
            metrics.per_site_unexposed_orientation_rmse_rad[0],
            expected_unexposed,
            places=5,
        )
        self.assertAlmostEqual(
            metrics.per_site_exposed_orientation_p95_rad[0],
            0.29,
            places=5,
        )
        self.assertEqual(metrics.per_site_orientation_rmse_rad[1], 0.0)
        self.assertAlmostEqual(
            metrics.upper_endpoint_orientation_rmse_rad,
            expected_all / (2.0**0.5),
            places=5,
        )

    def test_pair_rejects_nearest_frame_or_force_substitution(self) -> None:
        stiff = self._trace("stiff")
        compliant = self._trace("compliant")
        bad_frame = compliant.reference_frame.clone()
        bad_frame[2] += 1
        with self.assertRaisesRegex(ValueError, "reference_frame"):
            compare_aligned_tracking_traces(
                stiff,
                replace(compliant, reference_frame=bad_frame),
            )
        bad_force = compliant.force_on_robot_w.clone()
        bad_force[2, 0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "force_on_robot_w"):
            compare_aligned_tracking_traces(
                stiff,
                replace(compliant, force_on_robot_w=bad_force),
            )
        with self.assertRaisesRegex(ValueError, "structured local frame"):
            compare_aligned_tracking_traces(
                stiff,
                replace(compliant, local_frame=CartesianFrameSpec.world()),
            )

    def test_pair_has_an_overall_upper_endpoint_orientation_gate(self) -> None:
        stiff = self._trace("stiff")
        compliant = self._trace("compliant", endpoint_error_m=0.02)
        actual = compliant.actual_site_quaternions_wxyz.clone()
        angle = torch.tensor(0.3)
        actual[..., 0] = torch.cos(angle / 2.0)
        actual[..., 1] = torch.sin(angle / 2.0)
        result = compare_aligned_tracking_traces(
            stiff,
            replace(compliant, actual_site_quaternions_wxyz=actual),
            thresholds=PairedEvaluationThresholds(
                min_aligned_frames=5,
                min_exposed_frames_per_site=3,
                max_upper_endpoint_orientation_regression_rad=0.1,
                min_peak_force_n=2.0,
                max_peak_force_n=5.0,
            ),
        )
        self.assertFalse(result.passed)
        self.assertFalse(dict(result.checks)["upper_endpoint_orientation_budget"])

    def test_pair_requires_nonzero_displacement_only_as_a_chain_gate(self) -> None:
        result = compare_aligned_tracking_traces(
            self._trace("stiff"),
            self._trace("compliant"),
            thresholds=PairedEvaluationThresholds(
                min_aligned_frames=5,
                min_exposed_frames_per_site=3,
                min_paired_displacement_m=1.0e-6,
                min_peak_force_n=2.0,
                max_peak_force_n=5.0,
            ),
        )
        self.assertFalse(dict(result.checks)["paired_displacement_activation"])
        self.assertEqual(result.compliance_response.displacement_mean_m, 0.0)

    def test_pair_gates_each_site_so_one_wrist_cannot_hide_in_the_mean(self) -> None:
        stiff = self._trace("stiff")
        compliant = self._trace("compliant")
        actual_positions = compliant.actual_site_positions_w.clone()
        actual_positions[:, 0, 0] = 0.08
        actual_positions[:, 1, 0] = 0.0
        actual_quaternions = compliant.actual_site_quaternions_wxyz.clone()
        angle = torch.tensor(0.3)
        actual_quaternions[:, 0, 0] = torch.cos(angle / 2.0)
        actual_quaternions[:, 0, 1] = torch.sin(angle / 2.0)
        result = compare_aligned_tracking_traces(
            stiff,
            replace(
                compliant,
                actual_site_positions_w=actual_positions,
                actual_site_quaternions_wxyz=actual_quaternions,
            ),
            thresholds=PairedEvaluationThresholds(
                min_aligned_frames=5,
                min_exposed_frames_per_site=3,
                min_peak_force_n=2.0,
                max_peak_force_n=5.0,
            ),
        )
        checks = dict(result.checks)
        self.assertTrue(checks["upper_endpoint_tracking_budget"])
        self.assertTrue(checks["upper_endpoint_orientation_budget"])
        self.assertFalse(checks["site/site_0/position_rmse_budget"])
        self.assertFalse(checks["site/site_0/position_p95_budget"])
        self.assertFalse(checks["site/site_0/orientation_rmse_budget"])
        self.assertFalse(checks["site/site_0/orientation_p95_budget"])

    def test_transition_termination_keeps_current_sample_and_locks_suffix(self) -> None:
        trace = self._trace("compliant")
        valid = torch.tensor([True, True, True, False, False])
        failed = replace(
            trace,
            valid=valid,
            fell=True,
            horizon_reached=False,
            termination_sample=2,
        )
        metrics = summarize_tracking_trace(failed)
        self.assertEqual(metrics.valid_frames, 3)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.fall_rate, 1.0)
        auto_reset_suffix = valid.clone()
        auto_reset_suffix[4] = True
        with self.assertRaisesRegex(ValueError, "after termination"):
            replace(failed, valid=auto_reset_suffix)
        with self.assertRaisesRegex(ValueError, "complete horizon"):
            replace(trace, valid=valid)

    def test_pair_tracking_metrics_use_only_the_common_valid_prefix(self) -> None:
        stiff = self._trace("stiff")
        stiff_actual = stiff.actual_positions_w.clone()
        stiff_actual[2:, :, 0] = 1.0
        stiff = replace(stiff, actual_positions_w=stiff_actual)
        compliant = self._trace("compliant")
        compliant = replace(
            compliant,
            valid=torch.tensor([True, True, False, False, False]),
            fell=True,
            horizon_reached=False,
            termination_sample=1,
        )
        result = compare_aligned_tracking_traces(
            stiff,
            compliant,
            thresholds=PairedEvaluationThresholds(
                min_aligned_frames=2,
                min_exposed_frames_per_site=1,
                min_compliant_success_rate=0.0,
                max_compliant_fall_rate=1.0,
                min_peak_force_n=1.0,
                max_peak_force_n=5.0,
            ),
        )
        self.assertEqual(result.aligned_frames, 2)
        self.assertEqual(result.stiff.valid_frames, 2)
        self.assertEqual(result.compliant.valid_frames, 2)
        self.assertAlmostEqual(result.stiff.global_mpjpe_m, 0.01, places=6)

    def test_variable_site_contract_has_no_three_point_assumption(self) -> None:
        trace = self._trace("compliant", samples=5, sites=17)
        metrics = summarize_tracking_trace(trace)
        self.assertEqual(len(metrics.per_site_exposed_frames), 17)
        self.assertTrue(all(count == 3 for count in metrics.per_site_exposed_frames))

    def test_non_finite_and_non_monotonic_trace_is_rejected(self) -> None:
        trace = self._trace("stiff")
        bad = trace.actual_positions_w.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(trace, actual_positions_w=bad)
        time_s = trace.time_s.clone()
        time_s[2] = time_s[1]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            replace(trace, time_s=time_s)
        non_unit = trace.actual_site_quaternions_wxyz.clone()
        non_unit[0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "normalized"):
            replace(trace, actual_site_quaternions_wxyz=non_unit)
        bad_quaternion = trace.reference_site_quaternions_wxyz.clone()
        bad_quaternion[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(trace, reference_site_quaternions_wxyz=bad_quaternion)
        with self.assertRaisesRegex(TypeError, "one dtype"):
            replace(trace, actual_site_positions_w=trace.actual_site_positions_w.double())

    def test_npz_json_round_trip_uses_no_pickle(self) -> None:
        trace = self._trace("compliant")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            save_tracking_trace(trace, path)
            loaded = load_tracking_trace(path)
            self.assertEqual(loaded.mode, trace.mode)
            self.assertEqual(loaded.body_names, trace.body_names)
            self.assertEqual(loaded.local_frame, trace.local_frame)
            self.assertTrue(torch.equal(loaded.force_on_robot_w, trace.force_on_robot_w))
            self.assertTrue(
                torch.equal(
                    loaded.actual_site_quaternions_wxyz,
                    trace.actual_site_quaternions_wxyz,
                )
            )
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["local_frame"], {
                "anchor": "pelvis",
                "kind": "heading_local",
                "rotation": "yaw_only",
            })
            self.assertNotIn("pickle", metadata)
            with self.assertRaisesRegex(ValueError, "uncompressed byte cap"):
                load_tracking_trace(path, max_uncompressed_bytes=1)

    def test_trace_writer_refuses_collisions_and_symlinks(self) -> None:
        trace = self._trace("compliant")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trace.npz"
            path.write_bytes(b"preserve-existing")
            with self.assertRaises(FileExistsError):
                save_tracking_trace(trace, path)
            self.assertEqual(path.read_bytes(), b"preserve-existing")
            path.unlink()
            metadata = path.with_suffix(".json")
            metadata.write_text("preserve-metadata\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_tracking_trace(trace, path)
            self.assertFalse(path.exists())
            self.assertEqual(
                metadata.read_text(encoding="utf-8"),
                "preserve-metadata\n",
            )
            metadata.unlink()
            path.symlink_to(root / "missing-target")
            with self.assertRaises(FileExistsError):
                save_tracking_trace(trace, path)
            path.unlink()

            original_publish = evaluation_io._publish_new

            def fail_metadata_publish(temporary_path, destination):
                if destination.suffix == ".json":
                    raise RuntimeError("injected metadata publication failure")
                original_publish(temporary_path, destination)

            with mock.patch.object(
                evaluation_io,
                "_publish_new",
                side_effect=fail_metadata_publish,
            ):
                with self.assertRaisesRegex(RuntimeError, "injected metadata"):
                    save_tracking_trace(trace, path)
            self.assertFalse(path.exists())
            self.assertFalse(metadata.exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
