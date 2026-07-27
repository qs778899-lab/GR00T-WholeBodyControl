"""Lightweight, IsaacLab-free audit tests for the compliance design assumptions.

These tests intentionally inspect the released SONIC contracts rather than importing
the simulation stack. A failure means either the repository evolved or the analysis
in compliance_control/ must be updated before implementation continues.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TRACKING_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def _yaml(path: str):
    with (ROOT / path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _literal_assignment(path: str, name: str):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class SonicReleaseContractTests(unittest.TestCase):
    def test_fourteen_bodies_are_tracking_skeleton(self):
        motion_cfg = _yaml("gear_sonic/config/manager_env/commands/terms/motion.yaml")["motion"]
        self.assertEqual(motion_cfg["body_names"], EXPECTED_TRACKING_BODIES)
        self.assertEqual(len(motion_cfg["body_names"]), 14)

    def test_g1_encoder_is_joint_motion_not_fourteen_keypoints(self):
        encoder_cfg = _yaml("gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml")["g1"]
        self.assertEqual(
            encoder_cfg["inputs"],
            ["command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"],
        )

        release_cfg = _yaml(
            "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml"
        )
        num_frames = release_cfg["manager_env"]["commands"]["motion"]["num_future_frames"]
        self.assertEqual(num_frames, 10)

        num_dof = 29
        orientation_6d = 6
        per_frame_dim = 2 * num_dof + orientation_6d
        self.assertEqual(per_frame_dim, 64)
        self.assertEqual(num_frames * per_frame_dim, 640)

        deploy_cfg = _yaml("gear_sonic_deploy/policy/release/observation_config.yaml")
        g1_mode = next(mode for mode in deploy_cfg["encoder"]["encoder_modes"] if mode["name"] == "g1")
        self.assertEqual(
            g1_mode["required_observations"],
            [
                "encoder_mode_4",
                "motion_joint_positions_10frame_step5",
                "motion_joint_velocities_10frame_step5",
                "motion_anchor_orientation_10frame_step5",
            ],
        )
        self.assertEqual(deploy_cfg["encoder"]["dimension"], 64)

    def test_bfs_dfs_dof_mappings_are_consistent_and_shared(self):
        robot_path = "gear_sonic/envs/manager_env/robots/g1.py"
        eval_path = "tools/sonic_eval/motionlib_provider.py"
        i2m = _literal_assignment(robot_path, "G1_ISAACLAB_TO_MUJOCO_DOF")
        m2i = _literal_assignment(robot_path, "G1_MUJOCO_TO_ISAACLAB_DOF")

        self.assertEqual(sorted(i2m), list(range(29)))
        self.assertEqual(sorted(m2i), list(range(29)))
        self.assertTrue(all(m2i[i2m[mujoco_i]] == mujoco_i for mujoco_i in range(29)))
        self.assertTrue(all(i2m[m2i[isaac_i]] == isaac_i for isaac_i in range(29)))

        self.assertEqual(i2m, _literal_assignment(eval_path, "G1_ISAACLAB_TO_MUJOCO_DOF"))
        self.assertEqual(m2i, _literal_assignment(eval_path, "G1_MUJOCO_TO_ISAACLAB_DOF"))

    def test_release_model_does_not_enable_dormant_compliance_interface(self):
        deploy_text = (
            ROOT / "gear_sonic_deploy/policy/release/observation_config.yaml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("vr_3point_compliance", deploy_text)

        command_files = list((ROOT / "gear_sonic/config/manager_env/commands").rglob("*.yaml"))
        self.assertFalse(any("force" in path.name.lower() for path in command_files))

        event_terms = ROOT / "gear_sonic/config/manager_env/events/terms"
        self.assertFalse((event_terms / "compliance_force_push.yaml").exists())
        self.assertFalse(any(event_terms.glob("chip_change_compliance*.yaml")))

        command_source = (ROOT / "gear_sonic/envs/manager_env/mdp/commands.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class ForceTrackingCommand", command_source)

    def test_existing_latent_residual_hook_can_preserve_base_encoder_contract(self):
        module_source = (
            ROOT / "gear_sonic/trl/modules/universal_token_modules.py"
        ).read_text(encoding="utf-8")
        wrapper_source = (
            ROOT / "gear_sonic/envs/wrapper/manager_env_wrapper.py"
        ).read_text(encoding="utf-8")
        self.assertIn('latent_residual_mode="post_quantization"', module_source)
        self.assertIn("all_tokens = all_tokens + residual_reshaped", module_source)
        self.assertIn('self._use_latent_residual = self.config.get("use_latent_residual", False)', wrapper_source)


class DormantComplianceShapeReproducerTests(unittest.TestCase):
    def test_current_multi_future_broadcast_and_two_vs_three_point_shapes_fail(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is part of this repository's runtime
            self.skipTest(str(exc))

        envs, future_frames, force_bodies = 8, 5, 2
        magnitudes = torch.zeros(envs, force_bodies)
        phases = torch.zeros(envs, future_frames)
        directions = torch.zeros(envs, force_bodies, 3)

        # Reproduces observations.py's current `[:, None, None, None]` expression.
        with self.assertRaises(RuntimeError):
            _ = (
                magnitudes[:, None, None, None]
                * phases[:, :, None, None]
                * directions[:, None, :, :]
            )

        three_point_target = torch.zeros(envs, future_frames, 3, 3)
        two_wrist_displacement = torch.zeros(envs, future_frames, 2, 3)
        with self.assertRaises(RuntimeError):
            _ = three_point_target - two_wrist_displacement


if __name__ == "__main__":
    unittest.main()
