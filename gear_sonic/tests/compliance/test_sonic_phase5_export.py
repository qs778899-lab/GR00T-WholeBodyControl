"""Phase-5 separate residual ONNX contract and inference parity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from gear_sonic.compliance_control.postprocess import evaluation_io

from gear_sonic.compliance_control import ComplianceResidualMLP
from gear_sonic.compliance_control.adapters.sonic.export import (
    ACTOR_RESIDUAL_STATE_PREFIX,
    PHASE5_ACCEPTED_ONNXRUNTIME_VERSION,
    SonicResidualExportSpec,
    export_sonic_actor_residual_onnx,
    extract_actor_residual_state,
    verify_sonic_actor_residual_onnx,
)


_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None


class SonicPhase5ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = SonicResidualExportSpec(
            site_names=("left_wrist", "right_wrist"),
            num_future_frames=10,
            cartesian_dim=3,
            context_dim=930,
            output_dim=64,
        )
        self.module = ComplianceResidualMLP(
            condition_dim=self.spec.condition_dim,
            num_sites=len(self.spec.site_names),
            cartesian_dim=self.spec.cartesian_dim,
            context_dim=self.spec.context_dim,
            output_dim=self.spec.output_dim,
            hidden_dims=self.spec.hidden_dims,
            residual_limit=self.spec.residual_limit,
        )
        with torch.no_grad():
            self.module.output_layer.weight.fill_(1.0e-3)
            self.module.output_layer.bias.fill_(2.0e-3)

    def _checkpoint(self, directory: str) -> Path:
        checkpoint = Path(directory) / "branch.pt"
        policy_state = {
            f"{ACTOR_RESIDUAL_STATE_PREFIX}{name}": tensor.clone()
            for name, tensor in self.module.state_dict().items()
        }
        torch.save({"policy_state_dict": policy_state}, checkpoint)
        return checkpoint

    def test_extract_requires_exact_six_tensor_schema(self) -> None:
        policy_state = {
            f"{ACTOR_RESIDUAL_STATE_PREFIX}{name}": tensor
            for name, tensor in self.module.state_dict().items()
        }
        extracted = extract_actor_residual_state(policy_state)
        self.assertEqual(set(extracted), set(self.module.state_dict()))
        policy_state.pop(f"{ACTOR_RESIDUAL_STATE_PREFIX}output_layer.bias")
        with self.assertRaisesRegex(ValueError, "exactly six"):
            extract_actor_residual_state(policy_state)

    def test_spec_exposes_release_composition_dimensions(self) -> None:
        self.assertEqual(self.spec.condition_dim, 60)
        self.assertEqual(self.spec.command_dim, 9)
        self.assertEqual(self.spec.context_dim, 930)
        self.assertEqual(self.spec.output_dim, 64)

    def test_export_refuses_output_or_manifest_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(directory)
            output = Path(directory) / "compliance_residual.onnx"
            output.write_bytes(b"preserve-onnx")
            with self.assertRaises(FileExistsError):
                export_sonic_actor_residual_onnx(
                    checkpoint_path=checkpoint,
                    output_path=output,
                    spec=self.spec,
                )
            self.assertEqual(output.read_bytes(), b"preserve-onnx")
            output.unlink()
            escaped_output = Path(directory) / "escaped-output-target.onnx"
            output.symlink_to(escaped_output)
            with self.assertRaises(FileExistsError):
                export_sonic_actor_residual_onnx(
                    checkpoint_path=checkpoint,
                    output_path=output,
                    spec=self.spec,
                )
            self.assertFalse(escaped_output.exists())
            output.unlink()
            manifest = output.with_suffix(".json")
            manifest.symlink_to(Path(directory) / "missing-manifest-target")
            with self.assertRaises(FileExistsError):
                export_sonic_actor_residual_onnx(
                    checkpoint_path=checkpoint,
                    output_path=output,
                    spec=self.spec,
                )
            self.assertFalse(output.exists())
            manifest.unlink()

            def write_fake_onnx(_model, _inputs, destination, **_kwargs):
                Path(destination).write_bytes(b"valid-enough-for-publication-test")

            with mock.patch.object(
                torch.onnx,
                "export",
                side_effect=write_fake_onnx,
            ), mock.patch.object(
                evaluation_io,
                "_publish_new",
                side_effect=RuntimeError("injected manifest publication failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected manifest"):
                    export_sonic_actor_residual_onnx(
                        checkpoint_path=checkpoint,
                        output_path=output,
                        spec=self.spec,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual(
                sorted(path.name for path in Path(directory).iterdir()),
                ["branch.pt"],
            )

    @unittest.skipUnless(_HAS_ONNX, "onnx is unavailable in the portable interpreter")
    def test_separate_onnx_dynamic_and_hard_off_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(directory)
            output = Path(directory) / "compliance_residual.onnx"
            manifest = export_sonic_actor_residual_onnx(
                checkpoint_path=checkpoint,
                output_path=output,
                spec=self.spec,
            )
            report = verify_sonic_actor_residual_onnx(
                checkpoint_path=checkpoint,
                onnx_path=output,
                spec=self.spec,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(report["onnx_checker"])
            self.assertTrue(report["hard_off_exact"])
            self.assertTrue(report["zero_compliance_exact"])
            self.assertTrue(report["mixed_row_exact"])
            expected_runtime = (
                "onnxruntime.InferenceSession"
                if _HAS_ONNXRUNTIME
                else "onnx.reference.ReferenceEvaluator"
            )
            self.assertEqual(report["runtime"], expected_runtime)
            if _HAS_ONNXRUNTIME:
                self.assertEqual(report["providers"], ["CPUExecutionProvider"])
                self.assertEqual(
                    report["runtime_version"],
                    PHASE5_ACCEPTED_ONNXRUNTIME_VERSION,
                )
                strict_report = verify_sonic_actor_residual_onnx(
                    checkpoint_path=checkpoint,
                    onnx_path=output,
                    spec=self.spec,
                    runtime="onnxruntime",
                )
                self.assertEqual(
                    strict_report["runtime"],
                    "onnxruntime.InferenceSession",
                )
            self.assertEqual(len(report["dynamic_shape_cases"]), 5)
            self.assertEqual(report["dynamic_shape_cases"][-1]["active_rows"], 4)
            self.assertEqual(report["dynamic_shape_cases"][-1]["inactive_rows"], 4)
            self.assertLessEqual(report["maximum_absolute_error"], report["atol"])
            self.assertFalse(manifest["release_models_modified"])
            self.assertEqual(
                [item["name"] for item in manifest["inputs"]],
                ["compliance_target", "compliance_command", "actor_context"],
            )
            self.assertEqual(manifest["outputs"][0]["name"], "latent_residual")
            saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved["onnx_sha256"], manifest["onnx_sha256"])


if __name__ == "__main__":
    unittest.main()
