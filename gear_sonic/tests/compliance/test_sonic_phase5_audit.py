"""Negative provenance contracts for the independent Phase-5 audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from gear_sonic.scripts.audit_chip_phase5 import (
    _require_rollout_checkpoint_provenance,
    _require_workflow_provenance,
)


class SonicPhase5AuditTest(unittest.TestCase):
    def test_workflow_rejects_wrong_run_and_runs_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = (Path(directory) / "runs").resolve()
            run_root = (runs_root / "accepted").resolve()
            checkpoint = (Path(directory) / "checkpoint.pt").resolve()
            workflow = {
                "complete": True,
                "marker": "CHIP_PHASE5_EVAL_EXPORT_PASS",
                "run_root": str(run_root),
                "runs_root": str(runs_root),
                "checkpoint": str(checkpoint),
                "evaluation_claim": "chain_validation_not_performance_proof",
            }
            _require_workflow_provenance(
                workflow,
                run_root=run_root,
                runs_root=runs_root,
                checkpoint=checkpoint,
            )
            for key, wrong in (
                ("run_root", runs_root / "other-run"),
                ("runs_root", Path(directory) / "other-runs"),
            ):
                corrupted = dict(workflow)
                corrupted[key] = str(wrong.resolve())
                with self.assertRaisesRegex(AssertionError, "mismatch"):
                    _require_workflow_provenance(
                        corrupted,
                        run_root=run_root,
                        runs_root=runs_root,
                        checkpoint=checkpoint,
                    )

    def test_rollout_rejects_checkpoint_alias_path_and_wrong_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (root / "checkpoint.pt").resolve()
            checkpoint.write_bytes(b"phase-5-checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            summary = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
            }
            _require_rollout_checkpoint_provenance(
                summary,
                mode="stiff",
                checkpoint=checkpoint,
                checkpoint_sha256=digest,
            )

            alias = root / "checkpoint-alias.pt"
            alias.symlink_to(checkpoint)
            wrong_path = dict(summary, checkpoint=str(alias))
            with self.assertRaisesRegex(AssertionError, "checkpoint mismatch"):
                _require_rollout_checkpoint_provenance(
                    wrong_path,
                    mode="stiff",
                    checkpoint=checkpoint,
                    checkpoint_sha256=digest,
                )

            wrong_sha = dict(summary, checkpoint_sha256="0" * 64)
            with self.assertRaisesRegex(AssertionError, "SHA-256 mismatch"):
                _require_rollout_checkpoint_provenance(
                    wrong_sha,
                    mode="compliant",
                    checkpoint=checkpoint,
                    checkpoint_sha256=digest,
                )


if __name__ == "__main__":
    unittest.main()
