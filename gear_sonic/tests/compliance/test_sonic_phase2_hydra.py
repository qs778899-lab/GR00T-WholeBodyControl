"""Hydra composition checks for the isolated Phase-2 compliance experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(HYDRA_AVAILABLE, "hydra-core unavailable in portable CPU environment")
class ComplianceHydraCompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from hydra import compose, initialize_config_dir

        from gear_sonic.utils.config_utils import register_rl_resolvers

        register_rl_resolvers()
        cls.config_dir = Path(__file__).resolve().parents[2] / "config"
        with initialize_config_dir(version_base="1.1", config_dir=str(cls.config_dir)):
            cls.config = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release_compliance",
                    "num_envs=1",
                    "exp_base=chip_phase2_test",
                    "experiment_name=chip_phase2_test",
                    "experiment_dir=/tmp/chip_phase2_hydra_test",
                ],
            )

    def test_only_derived_experiment_composes_compliance_groups(self) -> None:
        cfg = self.config
        self.assertEqual(
            cfg.manager_env.commands.force._target_,
            (
                "gear_sonic.compliance_control.adapters.sonic.isaaclab.command."
                "SonicComplianceCommandCfg"
            ),
        )
        self.assertEqual(
            cfg.manager_env.events._target_,
            (
                "gear_sonic.compliance_control.adapters.sonic.isaaclab.configs."
                "ComplianceEventsCfg"
            ),
        )
        self.assertIn("compliance_force_push", cfg.manager_env.events)
        self.assertIn("compliance_force_reset", cfg.manager_env.events)
        self.assertIn("chip_compliance_target", cfg.manager_env.observations.tokenizer)
        self.assertFalse(cfg.manager_env.commands.force.enabled)
        self.assertFalse(cfg.manager_env.commands.force.target_damper_enabled)
        self.assertEqual(
            list(cfg.manager_env.commands.force.force_magnitude_range_n),
            [0.0, 40.0],
        )
        self.assertEqual(
            list(cfg.manager_env.commands.force.compliance_values_m_per_n),
            [0.0, 0.02, 0.05],
        )
        self.assertEqual(
            list(cfg.manager_env.commands.force.force_duration_range_s),
            [1.0, 3.0],
        )
        self.assertEqual(cfg.manager_env.commands.force.max_net_force_n, 30.0)
        self.assertEqual(cfg.manager_env.commands.force.max_net_torque_nm, 20.0)

        release_path = (
            self.config_dir
            / "exp"
            / "manager"
            / "universal_token"
            / "all_modes"
            / "sonic_release.yaml"
        )
        release_text = release_path.read_text(encoding="utf-8")
        self.assertNotIn("chip_compliance", release_text)
        self.assertNotIn("commands.force", release_text)

    def test_hydra_runtime_names_resolve_full_and_partial_without_fixed_indices(self) -> None:
        from gear_sonic.compliance_control import CartesianFrameSpec
        from gear_sonic.compliance_control.adapters.sonic import resolve_compliance_sites

        reference_names = tuple(self.config.manager_env.commands.motion.body_names)
        partial_names = tuple(self.config.manager_env.commands.force.site_names)
        articulation_names = tuple(reversed(reference_names))
        common_frame = CartesianFrameSpec.heading_local(
            self.config.manager_env.commands.force.anchor_body
        )
        full = resolve_compliance_sites(
            reference_names,
            articulation_names,
            reference_names,
            target_frame=common_frame,
            force_frame=common_frame,
        )
        partial = resolve_compliance_sites(
            reference_names,
            articulation_names,
            partial_names,
            target_frame=common_frame,
            force_frame=common_frame,
        )
        self.assertEqual(full.reference_indices, tuple(range(len(reference_names))))
        self.assertEqual(
            full.articulation_indices,
            tuple(reversed(range(len(reference_names)))),
        )
        self.assertEqual(partial.reference.site_names, partial_names)
        self.assertEqual(partial.articulation.site_names, partial_names)
        self.assertNotEqual(partial.reference_indices, partial.articulation_indices)

if __name__ == "__main__":
    unittest.main()
