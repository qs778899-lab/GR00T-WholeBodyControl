"""Phase-4 SONIC config, runner, checkpoint, and source-boundary tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from gear_sonic.scripts.run_chip_phase4_finetune import (
    CENTRAL_RUNS_ROOT,
    INITIAL_FINAL_STEP,
    NUM_ENVIRONMENTS,
    RESUME_FINAL_STEP,
    Phase4Assets,
    build_commands,
    make_layout,
)


HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
TRL_AVAILABLE = importlib.util.find_spec("trl") is not None


class Phase4RunnerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = make_layout(CENTRAL_RUNS_ROOT / "unit_test_phase4_layout")
        self.assets = Phase4Assets(
            checkpoint=Path("/tmp/official.pt"),
            robot_motion=Path("/tmp/motion.pkl"),
            smpl_motion=Path("/tmp/smpl"),
        )
        self.commands = build_commands(
            self.layout,
            self.assets,
            python_executable=Path("/pinned/python"),
        )

    def test_layout_rejects_outside_or_broad_run_paths(self) -> None:
        for invalid in (Path("/tmp/outside"), CENTRAL_RUNS_ROOT):
            with self.assertRaisesRegex(ValueError, "must be a child"):
                make_layout(invalid)
        self.assertEqual(self.layout.root.parent, CENTRAL_RUNS_ROOT)
        self.assertEqual(len({self.layout.stiff, self.layout.initial, self.layout.resume}), 3)

    def test_exact_stiff_warm_start_and_incremental_resume_commands(self) -> None:
        stiff = set(self.commands.stiff)
        self.assertIn("+exp=manager/universal_token/all_modes/sonic_release", stiff)
        self.assertIn("+resume=false", stiff)
        self.assertIn(f"num_envs={NUM_ENVIRONMENTS}", stiff)
        self.assertIn(
            f"++algo.config.num_learning_iterations={INITIAL_FINAL_STEP}", stiff
        )
        self.assertIn("use_wandb=false", stiff)
        self.assertIn(f"experiment_dir={self.layout.stiff}", stiff)

        initial = set(self.commands.initial)
        self.assertIn("+resume=false", initial)
        self.assertIn("chip_phase4.expected_start_step=0", initial)
        self.assertIn("chip_phase4.expected_final_step=5", initial)
        self.assertIn("callbacks.model_save.save_last_frequency=5", initial)
        self.assertIn(
            "manager_env.commands.force.pulse_interval_range_s=[0.02,0.04]",
            initial,
        )
        self.assertFalse(any("${" in argument for argument in self.commands.initial))

        resume = set(self.commands.resume)
        self.assertIn("+resume=true", resume)
        self.assertIn("++algo.config.num_learning_iterations=1", resume)
        self.assertNotIn(
            f"++algo.config.num_learning_iterations={RESUME_FINAL_STEP}", resume
        )
        self.assertIn("chip_phase4.expected_start_step=5", resume)
        self.assertIn("chip_phase4.expected_final_step=6", resume)
        self.assertIn("callbacks.model_save.save_last_frequency=1", resume)
        self.assertIn(f"experiment_dir={self.layout.resume}", resume)
        self.assertTrue(
            any(
                argument.endswith("/resume_input_step5.pt")
                for argument in self.commands.resume
            )
        )

    def test_runner_documents_non_bitwise_resume_boundary(self) -> None:
        source = Path(
            "gear_sonic/scripts/run_chip_phase4_finetune.py"
        ).read_text(encoding="utf-8")
        self.assertIn("strict model/optimizer/scheduler/global-step restoration", source)
        self.assertIn("trajectory-bitwise", source)
        self.assertIn("if layout.root.exists()", source)
        self.assertNotIn("shutil.rmtree", source)

    def test_runner_names_pre_manifest_size_and_rechecks_final_budget(self) -> None:
        source = Path(
            "gear_sonic/scripts/run_chip_phase4_finetune.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"workflow_bytes_before_final_manifest": total_bytes', source)
        self.assertNotIn('"workflow_bytes": total_bytes', source)
        self.assertIn(
            "final_total_bytes, final_largest_log = _directory_usage_bytes(layout.root)",
            source,
        )
        self.assertIn("if final_total_bytes > MAX_WORKFLOW_BYTES", source)

    def test_phase4_docs_separate_accepted_evidence_from_fresh_reruns(self) -> None:
        plan = Path("tasks/chip_compliance_finetune/plan.md").read_text(
            encoding="utf-8"
        )
        matrix = Path("tasks/chip_compliance_finetune/test_matrix.md").read_text(
            encoding="utf-8"
        )
        accepted = "phase4_acceptance_resume_fix"
        for document in (plan, matrix):
            self.assertIn(accepted, document)
            self.assertIn("<fresh-run-root>", document)
            self.assertIn("must not exist", document)
        self.assertIn("--run-root <fresh-run-root>", plan)

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE and TRL_AVAILABLE,
        "phase4_training dependencies unavailable",
    )
    def test_file_path_runner_bootstraps_lazy_audit_import_outside_repo(self) -> None:
        script = Path(
            "gear_sonic/scripts/run_chip_phase4_finetune.py"
        ).resolve()
        probe = (
            "import importlib, runpy\n"
            f"runpy.run_path({str(script)!r}, run_name='phase4_runner_probe')\n"
            "module = importlib.import_module("
            "'gear_sonic.compliance_control.adapters.sonic.phase4_training')\n"
            "print(module.__name__)\n"
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = ""
        with tempfile.TemporaryDirectory(dir="/tmp") as outside_repo:
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=outside_repo,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn(
            "gear_sonic.compliance_control.adapters.sonic.phase4_training",
            completed.stdout,
        )


@unittest.skipUnless(HYDRA_AVAILABLE, "hydra-core unavailable")
class Phase4HydraContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from hydra import compose, initialize_config_dir

        from gear_sonic.utils.config_utils import register_rl_resolvers

        register_rl_resolvers()
        config_dir = Path(__file__).resolve().parents[2] / "config"
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            cls.config = compose(
                config_name="base",
                overrides=[
                    (
                        "+exp=manager/universal_token/all_modes/"
                        "sonic_release_compliance_finetune_smoke"
                    ),
                    "+checkpoint=/tmp/official.pt",
                    "experiment_dir=/tmp/chip_phase4_hydra",
                    "save_dir=/tmp/chip_phase4_hydra/.hydra",
                ],
            )

    def test_smoke_owns_only_residual_trainer_and_forces_exposure(self) -> None:
        cfg = self.config
        self.assertEqual(cfg.num_envs, 16)
        self.assertFalse(cfg.use_wandb)
        self.assertEqual(cfg.algo.config.num_learning_iterations, 5)
        self.assertEqual(cfg.algo.config.num_steps_per_env, 24)
        self.assertEqual(cfg.algo.config.num_mini_batches, 4)
        self.assertEqual(cfg.num_envs // cfg.algo.config.num_mini_batches, 4)
        self.assertEqual(
            cfg.trainer._target_,
            (
                "gear_sonic.compliance_control.adapters.sonic.phase4_training."
                "SonicComplianceResidualPPOTrainer"
            ),
        )
        force = cfg.manager_env.commands.force
        self.assertTrue(force.enabled)
        self.assertEqual(force.enabled_probability, 1.0)
        self.assertEqual(force.site_probability, 1.0)
        self.assertEqual(force.max_active_sites, len(force.site_names))
        self.assertEqual(list(force.pulse_interval_range_s), [0.02, 0.04])
        self.assertTrue(all(value > 0.0 for value in force.compliance_values_m_per_n))
        self.assertEqual(cfg.callbacks.model_save.save_last_frequency, 5)
        self.assertEqual(
            list(cfg.callbacks),
            ["model_save", "wandb", "read_eval", "im_resample", "chip_phase4_audit"],
        )

    def test_two_site_smoke_has_twelve_tensors_and_770753_scalars(self) -> None:
        cfg = self.config
        num_sites = len(cfg.manager_env.commands.force.site_names)
        future = cfg.manager_env.commands.force.num_future_frames
        actor_input = future * num_sites * 3 + (1 + num_sites + num_sites * 3) + 930
        critic_input = (
            future * num_sites * 3
            + num_sites * 3
            + (1 + num_sites + num_sites * 3)
            + 1645
        )

        def mlp_scalars(input_width: int, output_width: int) -> int:
            return (
                input_width * 256
                + 256
                + 256 * 128
                + 128
                + 128 * output_width
                + output_width
            )

        total = mlp_scalars(actor_input, 64) + mlp_scalars(critic_input, 1)
        self.assertEqual(num_sites, 2)
        self.assertEqual(total, 770_753)
        self.assertEqual(cfg.chip_phase4.expected_trainable_scalar_count, total)
        self.assertEqual(cfg.chip_phase4.expected_num_envs, 16)


@unittest.skipUnless(
    TRANSFORMERS_AVAILABLE and TRL_AVAILABLE,
    "transformers/TRL unavailable",
)
class Phase4CheckpointAuditTest(unittest.TestCase):
    @staticmethod
    def _residual(prefix: str, offset: float) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}trunk.0.weight": torch.tensor([[offset + 1.0]]),
            f"{prefix}trunk.0.bias": torch.tensor([offset + 2.0]),
            f"{prefix}trunk.2.weight": torch.tensor([[offset + 3.0]]),
            f"{prefix}trunk.2.bias": torch.tensor([offset + 4.0]),
            f"{prefix}output_layer.weight": torch.tensor([[offset + 5.0]]),
            f"{prefix}output_layer.bias": torch.tensor([offset + 6.0]),
        }

    def test_independent_audit_accepts_only_exact_legacy_plus_residual(self) -> None:
        from gear_sonic.compliance_control.adapters.sonic import phase4_training

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            official_path = root / "official.pt"
            trained_path = root / "last.pt"
            report_path = root / "phase4_audit.json"
            official = {
                "policy_state_dict": {
                    "std": torch.tensor([0.50001]),
                    "actor_module.base": torch.tensor([1.0]),
                },
                "value_state_dict": {
                    "running_mean_std.running_mean": torch.tensor([2.0]),
                },
                "state": {"global_step": 41_550},
            }
            actor_residual = self._residual(
                phase4_training.ACTOR_RESIDUAL_PREFIX, 0.0
            )
            critic_residual = self._residual(
                phase4_training.CRITIC_RESIDUAL_PREFIX, 10.0
            )
            trained = {
                "policy_state_dict": {
                    **official["policy_state_dict"],
                    **actor_residual,
                },
                "value_state_dict": {
                    **official["value_state_dict"],
                    **critic_residual,
                },
                "optimizer_state_dict": {
                    "state": {
                        index: {
                            "step": torch.tensor(1),
                            "exp_avg": torch.tensor([0.0]),
                        }
                        for index in range(12)
                    },
                    "param_groups": [{"params": list(range(12))}],
                },
                "lr_scheduler_state_dict": {"last_epoch": 5},
                "env_state_dict": {},
                "state": {"global_step": 5},
            }
            torch.save(official, official_path)
            torch.save(trained, trained_path)
            trainable_names = [f"policy.{name}" for name in actor_residual]
            trainable_names.extend(
                f"value_model.{name}" for name in critic_residual
            )
            report = {
                "complete": True,
                "audit_mode": "official_init",
                "source_checkpoint_step": 41_550,
                "start_step": 0,
                "final_step": 5,
                "checkpoint": str(trained_path),
                "optimizer_parameter_count": 12,
                "trainable_scalar_count": 12,
                "trainable_parameter_names": trainable_names,
                "actor_residual_changed": True,
                "critic_residual_changed": True,
                "losses": [
                    {"step": step, "values": {"loss/policy": 1.0 / step}}
                    for step in range(1, 6)
                ],
                "site_names": ["left_wrist", "right_wrist"],
                "site_exposure_counts": [80, 80],
                "gradient_stats": {
                    name: {
                        "seen_backward_count": 20,
                        "nonzero_backward_count": 19,
                        "max_abs_gradient": 0.25,
                    }
                    for name in trainable_names
                },
                "peak_cuda_memory_bytes": 1024,
            }
            report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
            with mock.patch.object(
                phase4_training,
                "_file_sha256",
                return_value=phase4_training.OFFICIAL_SONIC_SHA256,
            ):
                result = phase4_training.audit_sonic_phase4_checkpoint(
                    checkpoint_path=trained_path,
                    official_checkpoint_path=official_path,
                    audit_report_path=report_path,
                    expected_step=5,
                    expected_trainable_scalar_count=12,
                )
            self.assertEqual(result["actor_residual_tensors"], 6)
            self.assertEqual(result["critic_residual_tensors"], 6)
            self.assertEqual(result["optimizer_parameter_count"], 12)
            self.assertEqual(result["finite_loss_steps"], [1, 2, 3, 4, 5])

            corrupted = torch.load(trained_path, weights_only=False)
            corrupted["policy_state_dict"]["std"] = torch.tensor([0.5])
            torch.save(corrupted, trained_path)
            with mock.patch.object(
                phase4_training,
                "_file_sha256",
                return_value=phase4_training.OFFICIAL_SONIC_SHA256,
            ), self.assertRaisesRegex(AssertionError, "byte-exact"):
                phase4_training.audit_sonic_phase4_checkpoint(
                    checkpoint_path=trained_path,
                    official_checkpoint_path=official_path,
                    audit_report_path=report_path,
                    expected_step=5,
                    expected_trainable_scalar_count=12,
                )

    def test_final_audit_is_on_step_end_before_trainer_early_return(self) -> None:
        source = Path(
            "gear_sonic/compliance_control/adapters/sonic/phase4_training.py"
        ).read_text(encoding="utf-8")
        step_end = source[source.index("    def on_step_end"):source.index("    def on_train_end")]
        self.assertIn("self._finalize", step_end)
        trainer_source = Path("gear_sonic/trl/trainer/ppo_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            trainer_source.index("if self.control.should_training_stop:\n            return"),
            trainer_source.index("self.callback_handler.on_train_end"),
        )

    def test_resume_restores_serialized_optimizer_after_args_lr_reconciliation(
        self,
    ) -> None:
        from gear_sonic.compliance_control.adapters.sonic.phase4_training import (
            SonicComplianceResidualPPOTrainer,
        )
        from gear_sonic.compliance_control.training import assert_nested_exact
        from gear_sonic.trl.trainer.ppo_trainer_aux_loss import (
            TRLAuxLossPPOTrainer,
        )

        source_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        source_optimizer = torch.optim.AdamW([source_parameter], lr=2e-5)
        source_scheduler = torch.optim.lr_scheduler.LambdaLR(
            source_optimizer,
            lr_lambda=lambda _: 1.0,
        )
        source_parameter.grad = torch.tensor([0.25, -0.5])
        source_optimizer.step()
        source_scheduler.step()
        checkpoint = {
            "optimizer_state_dict": copy.deepcopy(source_optimizer.state_dict()),
            "lr_scheduler_state_dict": copy.deepcopy(source_scheduler.state_dict()),
            "args": SimpleNamespace(learning_rate=1e-5),
        }

        target_parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        target_optimizer = torch.optim.AdamW([target_parameter], lr=9e-4)
        target_scheduler = torch.optim.lr_scheduler.LambdaLR(
            target_optimizer,
            lr_lambda=lambda _: 1.0,
        )
        trainer = object.__new__(SonicComplianceResidualPPOTrainer)
        trainer.optimizer = target_optimizer
        trainer.lr_scheduler = target_scheduler
        trainer.args = SimpleNamespace(learning_rate=9e-4)

        def emulate_generic_load(self, checkpoint_path, resume=False):
            if resume:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.args.learning_rate = checkpoint["args"].learning_rate
                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = self.args.learning_rate
                self.lr_scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"]
                )
            return checkpoint

        with mock.patch.object(
            TRLAuxLossPPOTrainer,
            "load_checkpoint",
            autospec=True,
            side_effect=emulate_generic_load,
        ):
            loaded = SonicComplianceResidualPPOTrainer.load_checkpoint(
                trainer,
                "/tmp/synthetic_phase4_resume.pt",
                resume=True,
            )

        self.assertIs(loaded, checkpoint)
        self.assertEqual(trainer.args.learning_rate, 1e-5)
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 2e-5)
        assert_nested_exact(
            checkpoint["optimizer_state_dict"],
            trainer.optimizer.state_dict(),
            label="restored optimizer boundary",
        )
        assert_nested_exact(
            checkpoint["lr_scheduler_state_dict"],
            trainer.lr_scheduler.state_dict(),
            label="restored scheduler boundary",
        )


if __name__ == "__main__":
    unittest.main()
