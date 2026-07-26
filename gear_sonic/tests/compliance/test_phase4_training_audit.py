"""Portable tests for tracker-neutral Phase-4 training audit primitives."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from gear_sonic.compliance_control.training import (
    assert_nested_exact,
    assert_state_dict_exact,
    atomic_write_json,
    directory_usage_bytes,
    finite_loss_metrics,
    incremental_batch_count,
    optimizer_parameter_count,
    state_dict_digest,
    tensor_byte_equal,
)


class TensorAuditTest(unittest.TestCase):
    def test_byte_comparison_rejects_signed_zero_and_schema_changes(self) -> None:
        positive_zero = torch.tensor([0.0], dtype=torch.float32)
        negative_zero = torch.tensor([-0.0], dtype=torch.float32)
        self.assertFalse(tensor_byte_equal(positive_zero, negative_zero))
        with self.assertRaisesRegex(AssertionError, "byte-exact"):
            assert_state_dict_exact({"weight": positive_zero}, {"weight": negative_zero})
        with self.assertRaisesRegex(AssertionError, "schema mismatch"):
            assert_state_dict_exact(
                {"weight": positive_zero},
                {"weight": positive_zero, "extra": positive_zero},
            )
        assert_state_dict_exact(
            {"weight": positive_zero},
            {"weight": positive_zero, "extra": positive_zero},
            allow_additional_current=True,
        )

    def test_digest_exclusion_and_nested_optimizer_comparison(self) -> None:
        original = {
            "base": torch.tensor([1.0]),
            "residual.weight": torch.tensor([0.0]),
        }
        changed = {
            "base": torch.tensor([1.0]),
            "residual.weight": torch.tensor([2.0]),
        }
        self.assertNotEqual(state_dict_digest(original), state_dict_digest(changed))
        self.assertEqual(
            state_dict_digest(original, excluded_prefixes=("residual.",)),
            state_dict_digest(changed, excluded_prefixes=("residual.",)),
        )
        nested = {
            "state": {0: {"step": torch.tensor(4), "exp_avg": torch.tensor([1.0])}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
        }
        assert_nested_exact(nested, nested, label="optimizer")
        modified = {
            "state": {0: {"step": torch.tensor(5), "exp_avg": torch.tensor([1.0])}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
        }
        with self.assertRaisesRegex(AssertionError, "tensor mismatch"):
            assert_nested_exact(nested, modified, label="optimizer")


class ScalarAndFilesystemAuditTest(unittest.TestCase):
    def test_command_lifecycle_toggle_is_portable_and_strict(self) -> None:
        from gear_sonic.compliance_control.adapters.sonic.operational import (
            ComplianceOperationalControl,
        )

        command = object.__new__(ComplianceOperationalControl)
        self.assertFalse(command.is_evaluating)
        command.set_is_evaluating(True)
        self.assertTrue(command.is_evaluating)
        command.set_is_evaluating(False)
        self.assertFalse(command.is_evaluating)
        with self.assertRaises(TypeError):
            command.set_is_evaluating(1)

    def test_incremental_batch_count_distinguishes_init_and_resume(self) -> None:
        self.assertEqual(incremental_batch_count(0, 5), 5)
        self.assertEqual(incremental_batch_count(5, 6), 1)
        for start, final in ((5, 5), (6, 5), (-1, 1)):
            with self.assertRaises(ValueError):
                incremental_batch_count(start, final)

    def test_loss_and_optimizer_contracts_fail_closed(self) -> None:
        self.assertEqual(
            finite_loss_metrics(
                {"loss/policy": torch.tensor(1.25), "fps": 100.0}
            ),
            {"loss/policy": 1.25},
        )
        for invalid in ({}, {"loss/value": float("nan")}, {"loss/value": [1.0]}):
            with self.assertRaises((TypeError, ValueError)):
                finite_loss_metrics(invalid)
        optimizer = {"param_groups": [{"params": list(range(12))}]}
        self.assertEqual(optimizer_parameter_count(optimizer), 12)
        with self.assertRaisesRegex(ValueError, "unique"):
            optimizer_parameter_count(
                {"param_groups": [{"params": [0, 1]}, {"params": [1]}]}
            )

    def test_atomic_json_and_bounded_usage_do_not_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            payload_path = root / "audit.json"
            atomic_write_json(payload_path, {"complete": True, "value": 3})
            self.assertEqual(
                json.loads(payload_path.read_text(encoding="utf-8")),
                {"complete": True, "value": 3},
            )
            log_path = root / "train.log"
            log_path.write_text("bounded\n", encoding="utf-8")
            external = root.parent / f"{root.name}_external.bin"
            external.write_bytes(b"x" * 4096)
            try:
                (root / "external-link.bin").symlink_to(external)
                total, largest_log = directory_usage_bytes(root)
                expected_without_link = payload_path.stat().st_size + log_path.stat().st_size
                self.assertGreaterEqual(total, expected_without_link)
                self.assertLess(total, expected_without_link + external.stat().st_size)
                self.assertEqual(largest_log, log_path.stat().st_size)
            finally:
                external.unlink(missing_ok=True)
            with self.assertRaises(ValueError):
                atomic_write_json(root / "invalid.json", {"value": float("nan")})
            self.assertFalse(any(root.glob(".invalid.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
