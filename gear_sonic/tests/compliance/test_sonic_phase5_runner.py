"""CPU-only safety and command contracts for the Phase-5 workflow runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gear_sonic.compliance_control.adapters.sonic.contracts import (
    SONIC_RELEASE_TRACKING_BODY_NAMES,
    require_sonic_release_tracking_body_names,
)
from gear_sonic.scripts.run_chip_phase5_eval_export import (
    _assert_artifact_tree_safe,
)
from gear_sonic.scripts.run_chip_phase5_rollout import (
    _resolve_new_rollout_outputs,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RUNNER = _REPOSITORY_ROOT / "gear_sonic/scripts/run_chip_phase5_eval_export.py"


class SonicPhase5RunnerTest(unittest.TestCase):
    def _dry_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        runs_root = root / "runs"
        runs_root.mkdir()
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"dry-run-checkpoint")
        motion = root / "motion.pkl"
        motion.write_bytes(b"dry-run-motion")
        smpl = root / "smpl"
        smpl.mkdir()
        return runs_root, checkpoint, motion, smpl

    def test_rollout_uses_distinct_reference_and_articulation_quaternions(self) -> None:
        self.assertEqual(
            require_sonic_release_tracking_body_names(
                SONIC_RELEASE_TRACKING_BODY_NAMES,
            ),
            SONIC_RELEASE_TRACKING_BODY_NAMES,
        )
        reordered = list(SONIC_RELEASE_TRACKING_BODY_NAMES)
        reordered[-2:] = reversed(reordered[-2:])
        with self.assertRaisesRegex(AssertionError, "ordered release"):
            require_sonic_release_tracking_body_names(reordered)
        source = (
            _REPOSITORY_ROOT / "gear_sonic/scripts/run_chip_phase5_rollout.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "motion.body_quat_w.index_select(\n                1,\n"
            "                reference_site_indices,\n            )",
            source,
        )
        self.assertIn("force_command.current_site_quaternions_wxyz()", source)
        self.assertIn('"convention": "normalized_finite_wxyz"', source)
        self.assertIn('"reference_indices": list(force_command.sites.reference_indices)', source)
        self.assertIn('force_command.sites.articulation_indices', source)
        self.assertIn('"rotation": frame.rotation.value', source)
        self.assertIn('experiment_dir = runtime_root', source)
        self.assertIn('require_sonic_release_tracking_body_names(', source)
        self.assertNotIn('/tmp/chip_phase5_', source)
        self.assertIn('resolved_summary == trace_metadata', source)

    def test_dry_run_is_non_writing_and_builds_two_serial_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root, checkpoint, motion, smpl = self._dry_fixture(root)
            run_root = runs_root / "phase5_dry_run"
            command = [
                sys.executable,
                "-B",
                str(_RUNNER),
                "--runs-root",
                str(runs_root),
                "--run-root",
                str(run_root),
                "--checkpoint",
                str(checkpoint),
                "--motion-file",
                str(motion),
                "--smpl-motion-dir",
                str(smpl),
                "--steps",
                "300",
                "--dry-run",
            ]
            completed = subprocess.run(
                command,
                cwd=_REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(run_root.exists())
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["runs_root"], str(runs_root))
            self.assertEqual(set(payload["commands"]), {"stiff", "compliant"})
            self.assertEqual(
                payload["onnx_runtime"]["python"],
                str(Path(sys.executable).resolve()),
            )
            self.assertEqual(payload["onnx_runtime"]["expected_version"], "1.25.0")
            parity_command = payload["onnx_runtime"]["command"]
            self.assertIn("verify_chip_phase5_onnx.py", " ".join(parity_command))
            self.assertIn("--expected-version", parity_command)
            self.assertEqual(payload["export"]["spec"], {
                "condition_dim": 60,
                "command_dim": 9,
                "context_dim": 930,
                "output_dim": 64,
            })
            self.assertEqual(payload["thresholds"]["min_paired_displacement_m"], 1.0e-6)
            for mode, rollout in payload["commands"].items():
                self.assertIn("--mode", rollout)
                self.assertEqual(rollout[rollout.index("--mode") + 1], mode)
                self.assertEqual(rollout[rollout.index("--steps") + 1], "300")
                self.assertIn("--headless", rollout)
                self.assertEqual(rollout[rollout.index("--device") + 1], "cuda:0")
                trace = Path(rollout[rollout.index("--trace") + 1])
                self.assertEqual(trace.parent, run_root / mode)

    def test_runner_rejects_paths_outside_bounded_root_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root, checkpoint, motion, smpl = self._dry_fixture(root)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(_RUNNER),
                    "--runs-root",
                    str(runs_root),
                    "--run-root",
                    str(root / "outside"),
                    "--checkpoint",
                    str(checkpoint),
                    "--motion-file",
                    str(motion),
                    "--smpl-motion-dir",
                    str(smpl),
                    "--dry-run",
                ],
                cwd=_REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("strict child", completed.stderr)

            run_root = runs_root / "wrong_ort_version"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(_RUNNER),
                    "--runs-root",
                    str(runs_root),
                    "--run-root",
                    str(run_root),
                    "--checkpoint",
                    str(checkpoint),
                    "--motion-file",
                    str(motion),
                    "--smpl-motion-dir",
                    str(smpl),
                    "--onnxruntime-version",
                    "0.0.0",
                    "--dry-run",
                ],
                cwd=_REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("pinned Phase-5", completed.stderr)
            self.assertFalse(run_root.exists())

    def test_artifact_gate_rejects_symlinks_and_byte_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"1234")
            self.assertEqual(
                _assert_artifact_tree_safe(
                    root,
                    max_workflow_bytes=4,
                    max_log_bytes=4,
                ),
                4,
            )
            with self.assertRaisesRegex(AssertionError, "byte cap"):
                _assert_artifact_tree_safe(
                    root,
                    max_workflow_bytes=3,
                    max_log_bytes=4,
                )
            link = root / "outside-link"
            link.symlink_to(Path(directory).parent)
            with self.assertRaisesRegex(AssertionError, "symlink"):
                _assert_artifact_tree_safe(root)

    def test_rollout_output_leaf_symlinks_fail_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            trace = output / "trace.npz"
            summary = output / "rollout.json"
            escaped = root / "escaped-trace.npz"
            trace.symlink_to(escaped)
            with self.assertRaises(FileExistsError):
                _resolve_new_rollout_outputs(trace, summary)
            self.assertFalse(escaped.exists())
            trace.unlink()

            escaped_summary = root / "escaped-summary.json"
            summary.symlink_to(escaped_summary)
            with self.assertRaises(FileExistsError):
                _resolve_new_rollout_outputs(trace, summary)
            self.assertFalse(escaped_summary.exists())
            summary.unlink()

            metadata = trace.with_suffix(".json")
            metadata.symlink_to(root / "escaped-metadata.json")
            with self.assertRaises(FileExistsError):
                _resolve_new_rollout_outputs(trace, summary)


if __name__ == "__main__":
    unittest.main()
