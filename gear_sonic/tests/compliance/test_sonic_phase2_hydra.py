"""Hydra composition checks for the isolated Phase-2 compliance experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


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
            cls.release_config = compose(
                config_name="base",
                overrides=[
                    "+exp=manager/universal_token/all_modes/sonic_release",
                    "num_envs=1",
                    "exp_base=chip_phase2_release_test",
                    "experiment_name=chip_phase2_release_test",
                    "experiment_dir=/tmp/chip_phase2_release_hydra_test",
                ],
            )
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
        self.assertNotIn("compliance_force_push", cfg.manager_env.events)
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
        self.assertEqual(
            list(cfg.manager_env.commands.force.pulse_interval_range_s),
            [3.5, 6.0],
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

    def test_interval_event_set_and_ranges_match_release_exactly(self) -> None:
        def interval_terms(events):
            return {
                name: tuple(term.interval_range_s)
                for name, term in events.items()
                if name != "_target_" and term is not None and term.get("mode") == "interval"
            }

        release_intervals = interval_terms(self.release_config.manager_env.events)
        compliance_intervals = interval_terms(self.config.manager_env.events)
        self.assertEqual(compliance_intervals, release_intervals)
        self.assertEqual(compliance_intervals, {"push_robot": (4.0, 6.0)})
        self.assertEqual(
            self.config.manager_env.events.compliance_force_reset.mode,
            "reset",
        )

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


class OperationalEnableTest(unittest.TestCase):
    def test_disable_is_immediate_scoped_and_enable_uses_only_private_rng(self) -> None:
        from gear_sonic.compliance_control import CartesianFrameSpec
        from gear_sonic.compliance_control.adapters.sonic import (
            ArticulationWrenchAdapter,
            ComplianceOperationalControl,
            SonicComplianceCommandState,
            WrenchWriteGate,
            resolve_compliance_sites,
        )

        class FakeComposer:
            def __init__(self) -> None:
                self.composed_force_as_torch = torch.zeros(2, 4, 3)
                self.composed_torque_as_torch = torch.zeros(2, 4, 3)

            def set_forces_and_torques(
                self,
                *,
                forces,
                torques,
                positions,
                body_ids,
                env_ids,
                is_global,
            ) -> None:
                self.assertFalse(is_global)
                for row, env_id in enumerate(env_ids.tolist()):
                    self.composed_force_as_torch[env_id, body_ids] = forces[row]
                    composed_torque = torques[row]
                    if positions is not None:
                        composed_torque = composed_torque + torch.linalg.cross(
                            positions[row],
                            forces[row],
                            dim=-1,
                        )
                    self.composed_torque_as_torch[env_id, body_ids] = composed_torque

            def assertFalse(self, value) -> None:
                if value:
                    raise AssertionError("test composer requires body-local rows")

        class FakeArticulation:
            def __init__(self) -> None:
                self.permanent_wrench_composer = FakeComposer()

        frame = CartesianFrameSpec.world()
        sites = resolve_compliance_sites(
            ("root", "left", "right"),
            ("root", "right", "left", "tail"),
            ("left", "right"),
            target_frame=frame,
            force_frame=frame,
        )
        state = SonicComplianceCommandState(
            sites=sites,
            num_envs=2,
            num_future_frames=1,
            device="cpu",
            dtype=torch.float32,
            target_damper_alpha=0.1,
        )
        state.reset(torch.zeros(2, 1, 2, 3))
        enabled = torch.ones(2, dtype=torch.bool)
        site_mask = torch.tensor([[True, False], [False, True]])
        compliance = torch.full((2, 2, 3), 0.02)
        peak_force = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        state.start_pulses(
            None,
            enabled=enabled,
            site_mask=site_mask,
            compliance=compliance,
            peak_force_on_robot_w=peak_force,
            duration_s=torch.ones(2),
        )

        articulation = FakeArticulation()
        wrench = ArticulationWrenchAdapter(
            articulation,
            body_selection=sites.articulation,
            num_envs=2,
            device="cpu",
            dtype=torch.float32,
        )
        identity = torch.zeros(2, 2, 4)
        identity[..., 0] = 1.0
        wrench.set_world_forces_prevalidated(
            peak_force,
            body_quaternions_wxyz=identity,
            application_offsets_local=torch.zeros(2, 2, 3),
        )
        composer = articulation.permanent_wrench_composer
        sentinel_force = torch.tensor([0.125, -0.25, 0.375])
        sentinel_torque = torch.tensor([0.03125, -0.0625, 0.09375])
        composer.composed_force_as_torch[:, 0] = sentinel_force
        composer.composed_torque_as_torch[:, 0] = sentinel_torque

        command = ComplianceOperationalControl()
        command.state = state
        command.wrench = wrench
        command._env = type("FakeEnv", (), {"step_dt": 0.02})()  # noqa: SLF001
        command._sampling_generator = torch.Generator().manual_seed(983)  # noqa: SLF001
        command._all_env_ids = torch.arange(2)  # noqa: SLF001
        command._time_to_next_pulse = torch.tensor([0.1, 0.2])  # noqa: SLF001
        command._operational_enabled = True  # noqa: SLF001
        command._operational_enabled_last_update = True  # noqa: SLF001
        command._wrench_write_gate = WrenchWriteGate()  # noqa: SLF001
        command._wrench_write_gate.mark_written()  # noqa: SLF001
        command.cfg = type(
            "FakeCfg",
            (),
            {"enabled": True, "pulse_interval_range_s": (3.5, 6.0)},
        )()

        cpu_rng_before = torch.random.get_rng_state().clone()
        private_rng_before = command._sampling_generator.get_state().clone()  # noqa: SLF001
        command.set_operational_enabled(False)
        self.assertFalse(command.operational_enabled)
        self.assertTrue(command.cfg.enabled)
        self.assertFalse(command._wrench_write_gate.was_written)  # noqa: SLF001
        self.assertTrue(torch.isinf(command.time_to_next_pulse).all())
        self.assertFalse(command.state.enabled.any())
        self.assertFalse(command.state.site_mask.any())
        self.assertFalse(command.state.pulse_active.any())
        self.assertEqual(command.state.compliance.count_nonzero(), 0)
        self.assertEqual(command.state.force_on_robot_w.count_nonzero(), 0)
        self.assertEqual(command.state.peak_force_on_robot_w.count_nonzero(), 0)
        self.assertEqual(command.state.pulse_elapsed_s.count_nonzero(), 0)
        self.assertEqual(command.state.pulse_duration_s.count_nonzero(), 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng_before))
        self.assertTrue(
            torch.equal(
                command._sampling_generator.get_state(),  # noqa: SLF001
                private_rng_before,
            )
        )
        body_ids = torch.tensor(sites.articulation_indices)
        self.assertEqual(
            composer.composed_force_as_torch.index_select(1, body_ids).count_nonzero(),
            0,
        )
        self.assertEqual(
            composer.composed_torque_as_torch.index_select(1, body_ids).count_nonzero(),
            0,
        )
        torch.testing.assert_close(
            composer.composed_force_as_torch[:, 0],
            sentinel_force.expand(2, 3),
        )
        torch.testing.assert_close(
            composer.composed_torque_as_torch[:, 0],
            sentinel_torque.expand(2, 3),
        )

        cpu_rng_before = torch.random.get_rng_state().clone()
        private_rng_before = command._sampling_generator.get_state().clone()  # noqa: SLF001
        command.set_operational_enabled(True)
        self.assertTrue(command.operational_enabled)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng_before))
        self.assertFalse(
            torch.equal(
                command._sampling_generator.get_state(),  # noqa: SLF001
                private_rng_before,
            )
        )
        self.assertTrue((command.time_to_next_pulse >= 3.5).all())
        self.assertTrue((command.time_to_next_pulse <= 6.0).all())
        torch.testing.assert_close(
            composer.composed_force_as_torch[:, 0],
            sentinel_force.expand(2, 3),
        )
        torch.testing.assert_close(
            composer.composed_torque_as_torch[:, 0],
            sentinel_torque.expand(2, 3),
        )

        with self.assertRaisesRegex(TypeError, "enabled must be a bool"):
            command.set_operational_enabled(1)

if __name__ == "__main__":
    unittest.main()
