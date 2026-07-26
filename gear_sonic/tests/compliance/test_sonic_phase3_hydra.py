"""Hydra and release-path invariants for the opt-in Phase-3 experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest


HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None

RELEASE_SHARED_FILES = (
    "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml",
    "gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml",
    "gear_sonic/config/actor_critic/critics/mlp.yaml",
    "gear_sonic/config/actor_critic/quantizers/fsq.yaml",
    "gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml",
    "gear_sonic/config/actor_critic/encoders/teleop_mlp.yaml",
    "gear_sonic/config/actor_critic/encoders/smpl_mlp.yaml",
    "gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml",
    "gear_sonic/config/actor_critic/decoders/g1_kin_mf_mlp.yaml",
    "gear_sonic/config/aux_losses/universal_token/g1_recon_and_all_latent.yaml",
    "gear_sonic/config/manager_env/observations/tokenizer/unitoken_all_noz.yaml",
    "gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml",
    "gear_sonic/config/manager_env/observations/critic/privileged_mf_hist.yaml",
    "gear_sonic/config/manager_env/observations/terms/actions.yaml",
    "gear_sonic/config/manager_env/observations/terms/base_ang_vel.yaml",
    "gear_sonic/config/manager_env/observations/terms/base_lin_vel.yaml",
    "gear_sonic/config/manager_env/observations/terms/body_ori.yaml",
    "gear_sonic/config/manager_env/observations/terms/body_pos.yaml",
    "gear_sonic/config/manager_env/observations/terms/command_multi_future.yaml",
    "gear_sonic/config/manager_env/observations/terms/command_multi_future_lower_body.yaml",
    "gear_sonic/config/manager_env/observations/terms/command_multi_future_nonflat.yaml",
    "gear_sonic/config/manager_env/observations/terms/command_z.yaml",
    "gear_sonic/config/manager_env/observations/terms/command_z_multi_future_nonflat.yaml",
    "gear_sonic/config/manager_env/observations/terms/encoder_index.yaml",
    "gear_sonic/config/manager_env/observations/terms/gravity_dir.yaml",
    "gear_sonic/config/manager_env/observations/terms/joint_pos.yaml",
    "gear_sonic/config/manager_env/observations/terms/joint_pos_multi_future_wrist_for_smpl.yaml",
    "gear_sonic/config/manager_env/observations/terms/joint_vel.yaml",
    "gear_sonic/config/manager_env/observations/terms/motion_anchor_ori_b.yaml",
    "gear_sonic/config/manager_env/observations/terms/motion_anchor_ori_b_mf_nonflat.yaml",
    "gear_sonic/config/manager_env/observations/terms/motion_anchor_pos_b.yaml",
    "gear_sonic/config/manager_env/observations/terms/smpl_joints_multi_future_local_nonflat.yaml",
    "gear_sonic/config/manager_env/observations/terms/smpl_root_ori_b_multi_future.yaml",
    "gear_sonic/config/manager_env/observations/terms/vr_3point_local_orn_target.yaml",
    "gear_sonic/config/manager_env/observations/terms/vr_3point_local_target.yaml",
    "gear_sonic/config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml",
    "gear_sonic/config/manager_env/rewards/terms/action_rate_l2.yaml",
    "gear_sonic/config/manager_env/rewards/terms/anti_shake_ang_vel.yaml",
    "gear_sonic/config/manager_env/rewards/terms/feet_acc.yaml",
    "gear_sonic/config/manager_env/rewards/terms/joint_limit.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_anchor_ori.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_anchor_pos.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_body_angvel.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_body_linvel.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_relative_body_ori.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_relative_body_pos.yaml",
    "gear_sonic/config/manager_env/rewards/terms/tracking_vr_5point_local.yaml",
    "gear_sonic/config/manager_env/rewards/terms/undesired_contacts.yaml",
)


@unittest.skipUnless(HYDRA_AVAILABLE, "hydra-core unavailable in portable CPU environment")
class CompliancePhase3HydraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from hydra import compose, initialize_config_dir

        from gear_sonic.utils.config_utils import register_rl_resolvers

        register_rl_resolvers()
        cls.config_dir = Path(__file__).resolve().parents[2] / "config"
        common = [
            "num_envs=1",
            "exp_base=chip_phase3_test",
            "experiment_name=chip_phase3_test",
            "experiment_dir=/tmp/chip_phase3_hydra_test",
        ]
        with initialize_config_dir(version_base="1.1", config_dir=str(cls.config_dir)):
            cls.release = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release",
                    *common,
                ],
            )
            cls.phase3 = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release_compliance_residual",
                    *common,
                ],
            )
            cls.three_site = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release_compliance_residual",
                    *common,
                    (
                        "manager_env.commands.force.site_names="
                        "[left_wrist_yaw_link,right_wrist_yaw_link,torso_link]"
                    ),
                ],
            )

    def test_opt_in_targets_and_isolated_observation_groups(self) -> None:
        cfg = self.phase3
        self.assertEqual(
            cfg.algo.config.actor._target_,
            "gear_sonic.compliance_control.adapters.sonic.policy.SonicComplianceActor",
        )
        self.assertEqual(
            cfg.algo.config.actor.backbone._target_,
            (
                "gear_sonic.compliance_control.adapters.sonic.policy."
                "SonicComplianceUniversalTokenModule"
            ),
        )
        self.assertEqual(
            cfg.algo.config.critic._target_,
            "gear_sonic.compliance_control.adapters.sonic.critic.SonicComplianceCritic",
        )
        self.assertEqual(
            cfg.manager_env.observations._target_,
            (
                "gear_sonic.compliance_control.adapters.sonic.isaaclab.configs."
                "ComplianceObservationsCfg"
            ),
        )
        self.assertFalse(cfg.manager_env.commands.force.enabled)
        self.assertFalse(cfg.manager_env.commands.force.target_damper_enabled)
        self.assertIsNone(cfg.manager_env.commands.force.site_offsets_local_xyz)
        self.assertFalse(
            cfg.manager_env.observations.compliance_target.chip_compliance_target.params.non_flatten
        )
        self.assertEqual(
            cfg.algo.config.actor.backbone.compliance_target_key,
            "compliance_target",
        )
        self.assertEqual(
            cfg.algo.config.actor.backbone.compliance_command_key,
            "compliance_command",
        )
        self.assertEqual(
            cfg.algo.config.actor.backbone.compliance_num_future_frames,
            cfg.manager_env.commands.force.num_future_frames,
        )
        self.assertNotIn("compliance_force_key", cfg.algo.config.actor)
        self.assertNotIn("compliance_force_key", cfg.algo.config.actor.backbone)
        self.assertEqual(cfg.algo.config.critic.compliance_force_key, "compliance_force")
        self.assertEqual(
            list(cfg.algo.config.actor.allowed_observation_keys),
            ["actor_obs", "tokenizer", "compliance_target", "compliance_command"],
        )
        self.assertNotIn(
            "compliance_force", cfg.algo.config.actor.allowed_observation_keys
        )

    def test_finetune_ownership_is_explicit_and_minimal(self) -> None:
        cfg = self.phase3
        self.assertTrue(cfg.algo.config.freeze_noise_std)
        self.assertTrue(cfg.algo.config.actor.backbone.freeze_encoders)
        self.assertTrue(cfg.algo.config.actor.backbone.freeze_decoders)
        self.assertTrue(cfg.algo.config.actor.backbone.freeze_quantizer)
        self.assertTrue(cfg.algo.config.critic.freeze_base_critic)

    def test_release_g1_640_fsq_encoder_and_aux_contract_is_unchanged(self) -> None:
        from omegaconf import OmegaConf

        release_backbone = self.release.algo.config.actor.backbone
        phase3_backbone = self.phase3.algo.config.actor.backbone
        self.assertEqual(
            list(phase3_backbone.encoders.g1.inputs),
            ["command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"],
        )
        self.assertEqual(10 * (58 + 6), 640)
        for field in (
            "num_future_frames",
            "num_fsq_levels",
            "fsq_level_list",
            "max_num_tokens",
            "encoder_sample_probs",
            "encoders",
            "decoders",
            "aux_loss_func",
            "aux_loss_coef",
            "reencode_smpl_g1_recon",
        ):
            self.assertEqual(
                OmegaConf.to_container(release_backbone[field], resolve=True)
                if OmegaConf.is_config(release_backbone[field])
                else release_backbone[field],
                OmegaConf.to_container(phase3_backbone[field], resolve=True)
                if OmegaConf.is_config(phase3_backbone[field])
                else phase3_backbone[field],
                msg=field,
            )
        self.assertEqual(phase3_backbone.num_fsq_levels, 32)
        self.assertEqual(phase3_backbone.max_num_tokens, 2)

    def test_release_tokenizer_dense_observations_and_rewards_are_identical(self) -> None:
        from omegaconf import OmegaConf

        for field in ("policy", "critic", "tokenizer"):
            self.assertEqual(
                OmegaConf.to_container(
                    self.release.manager_env.observations[field],
                    resolve=True,
                ),
                OmegaConf.to_container(
                    self.phase3.manager_env.observations[field],
                    resolve=True,
                ),
                msg=field,
            )
        self.assertNotIn(
            "chip_compliance_target",
            self.phase3.manager_env.observations.tokenizer,
        )
        self.assertEqual(
            OmegaConf.to_container(self.release.manager_env.rewards, resolve=True),
            OmegaConf.to_container(self.phase3.manager_env.rewards, resolve=True),
        )
        self.assertEqual(
            list(self.phase3.algo.config.critic.backbone.module_config_dict.input_dim),
            ["critic_obs"],
        )

    def test_site_count_is_config_driven(self) -> None:
        expected = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
        self.assertEqual(list(self.three_site.manager_env.commands.force.site_names), expected)
        self.assertEqual(
            list(self.three_site.algo.config.actor.backbone.compliance_site_names),
            expected,
        )
        self.assertEqual(
            list(self.three_site.algo.config.critic.compliance_site_names),
            expected,
        )
        self.assertIsNone(self.three_site.manager_env.commands.force.site_offsets_local_xyz)

    def test_every_claimed_release_shared_file_matches_baseline_commit(self) -> None:
        repository_root = self.config_dir.parents[1]
        for relative in RELEASE_SHARED_FILES:
            current = (repository_root / relative).read_bytes()
            baseline = subprocess.run(
                [
                    "git",
                    "show",
                    f"4141c34280abb67c82e115342a8720f4a83d750d:{relative}",
                ],
                cwd=repository_root,
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(current, baseline, msg=relative)

    def test_gpu_compatibility_blocker_is_recorded_as_resolved(self) -> None:
        plan = (
            self.config_dir.parents[1]
            / "tasks/chip_compliance_finetune/plan.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("this command is blocked before launch", plan)
        self.assertIn(
            "compatibility-driver workaround is validated",
            " ".join(plan.split()),
        )


if __name__ == "__main__":
    unittest.main()
