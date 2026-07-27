"""Hydra contract for deterministic matched-force Phase-5 evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from gear_sonic.compliance_control.adapters.sonic.contracts import (
    SONIC_RELEASE_TRACKING_BODY_NAMES,
)


_HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(_HYDRA_AVAILABLE, "hydra-core unavailable in portable CPU environment")
class SonicPhase5HydraTest(unittest.TestCase):
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
                    "+exp=manager/universal_token/all_modes/sonic_release_compliance_eval",
                    "experiment_dir=/tmp/chip_phase5_hydra_test",
                    "num_envs=1",
                ],
            )
            cls.release_config = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release",
                    "experiment_dir=/tmp/chip_phase5_release_contract_test",
                    "num_envs=1",
                ],
            )

    def test_robot_encoder_and_initial_frame_are_deterministic(self) -> None:
        motion = self.config.manager_env.commands.motion
        self.assertTrue(motion.use_paired_motions)
        self.assertTrue(motion.start_from_first_frame)
        self.assertIsNone(motion.sample_from_n_initial_frames)
        self.assertFalse(motion.randomize_heading)
        self.assertFalse(motion.freeze_frame_aug)
        self.assertFalse(motion.cat_upper_body_poses)
        self.assertFalse(motion.randomize_wrist_poses)
        self.assertEqual(
            dict(motion.encoder_sample_probs),
            {"g1": 1.0, "teleop": 0.0, "smpl": 0.0},
        )
        self.assertFalse(motion.motion_lib_cfg.adaptive_sampling.enable)

    def test_tracking_body_contract_is_the_complete_ordered_release_14(self) -> None:
        evaluated = list(self.config.manager_env.commands.motion.body_names)
        released = list(self.release_config.manager_env.commands.motion.body_names)
        self.assertEqual(evaluated, list(SONIC_RELEASE_TRACKING_BODY_NAMES))
        self.assertEqual(evaluated, released)

    def test_matched_force_schedule_is_two_site_and_nonzero(self) -> None:
        force = self.config.manager_env.commands.force
        self.assertTrue(force.enabled)
        self.assertEqual(force.enabled_probability, 1.0)
        self.assertEqual(force.site_probability, 1.0)
        self.assertEqual(force.max_active_sites, 2)
        self.assertEqual(list(force.force_magnitude_range_n), [5.0, 5.0])
        self.assertEqual(list(force.compliance_values_m_per_n), [0.02])
        self.assertEqual(list(force.force_duration_range_s), [1.0, 1.0])
        self.assertEqual(list(force.pulse_interval_range_s), [0.02, 0.02])
        self.assertEqual(
            list(force.site_names),
            ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        )

    def test_observation_noise_is_off_and_eval_termination_is_explicit(self) -> None:
        observations = self.config.manager_env.observations
        self.assertFalse(observations.policy.enable_corruption)
        self.assertFalse(observations.tokenizer.enable_corruption)
        terminations = self.config.manager_env.terminations
        self.assertEqual(terminations.anchor_pos.params.threshold, 0.25)
        self.assertEqual(terminations.ee_body_pos.params.threshold, 0.25)
        self.assertEqual(terminations.anchor_ori_full.params.threshold, 1.0)
        self.assertNotIn("foot_pos_xyz", terminations)


if __name__ == "__main__":
    unittest.main()
