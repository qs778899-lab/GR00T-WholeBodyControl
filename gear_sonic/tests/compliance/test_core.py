import ast
from pathlib import Path
import unittest

import torch

from gear_sonic.compliance_control import (
    FORCE_ON_ROBOT,
    CartesianFrameKind,
    CartesianFrameSpec,
    CartesianRotation,
    ComplianceTargetSpec,
    TargetDamper,
    apply_hindsight_target,
    pyramid_phase_weight,
    summarize_compliance_response,
)
from gear_sonic.compliance_control.adapters.sonic import (
    NamedSiteIndices,
    SiteIndexSpace,
    SonicComplianceSites,
    resolve_compliance_sites,
    resolve_site_indices,
)


TEST_FRAME = CartesianFrameSpec.heading_local("tracking_anchor")


def _spec(site_names, **kwargs):
    return ComplianceTargetSpec(
        site_names=site_names,
        target_frame=kwargs.pop("target_frame", TEST_FRAME),
        force_frame=kwargs.pop("force_frame", TEST_FRAME),
        **kwargs,
    )


class ComplianceSchemaTest(unittest.TestCase):
    def test_spec_normalizes_order_and_validates_metadata(self):
        spec = _spec(["hand_b", "hand_a"])
        self.assertEqual(spec.site_names, ("hand_b", "hand_a"))
        self.assertEqual(spec.num_sites, 2)
        self.assertEqual(spec.common_frame, TEST_FRAME)
        self.assertIs(spec.force_sign_convention, FORCE_ON_ROBOT)

        invalid_specs = (
            {"site_names": []},
            {"site_names": ["hand", "hand"]},
            {"site_names": ["hand"], "compliance_unit": "normalized"},
            {"site_names": ["hand"], "max_displacement_m": 0.0},
        )
        for kwargs in invalid_specs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _spec(**kwargs)

        for invalid_names in ("left_hand", b"left_hand"):
            with self.subTest(invalid_names=invalid_names), self.assertRaises(TypeError):
                _spec(invalid_names)
        with self.assertRaises(TypeError):
            _spec(["hand"], force_sign_convention="force_on_robot")

    def test_structured_frames_reject_invalid_and_mismatched_contracts(self):
        self.assertEqual(CartesianFrameSpec.world().rotation, CartesianRotation.IDENTITY)
        self.assertEqual(
            CartesianFrameSpec.anchor_local("reference_anchor").kind,
            CartesianFrameKind.ANCHOR_LOCAL,
        )
        self.assertEqual(TEST_FRAME.rotation, CartesianRotation.YAW_ONLY)

        with self.assertRaises(TypeError):
            CartesianFrameSpec(
                kind="heading_local",
                anchor="anchor",
                rotation=CartesianRotation.YAW_ONLY,
            )
        with self.assertRaises(ValueError):
            CartesianFrameSpec(
                kind=CartesianFrameKind.WORLD,
                anchor="anchor",
                rotation=CartesianRotation.IDENTITY,
            )
        with self.assertRaises(ValueError):
            _spec(
                ["hand"],
                target_frame=CartesianFrameSpec.world(),
                force_frame=TEST_FRAME,
            )


class HindsightTargetTest(unittest.TestCase):
    def test_disabled_mode_is_exact_non_aliased_identity(self):
        spec = _spec(["site"])
        reference = torch.randn(2, 4, 1, 3)
        result = apply_hindsight_target(reference, None, None, spec=spec, enabled=False)
        self.assertTrue(torch.equal(result, reference))
        self.assertNotEqual(result.data_ptr(), reference.data_ptr())

    def test_global_hard_off_ignores_nan_operands_and_has_clean_backward(self):
        spec = _spec(["site"])
        reference = torch.randn(2, 2, 1, 3, requires_grad=True)
        forces = torch.full((2, 1, 3), torch.nan, requires_grad=True)
        compliance = torch.full((2, 1), torch.nan, requires_grad=True)

        result = apply_hindsight_target(
            reference,
            forces,
            compliance,
            spec=spec,
            enabled=False,
        )
        result.sum().backward()

        self.assertTrue(torch.equal(result, reference.detach()))
        torch.testing.assert_close(reference.grad, torch.ones_like(reference))
        self.assertIsNone(forces.grad)
        self.assertIsNone(compliance.grad)

    def test_zero_compliance_is_exact_and_inputs_are_not_mutated(self):
        spec = _spec(["left", "right"])
        reference = torch.randn(2, 3, 2, 3)
        forces = torch.randn(2, 2, 3)
        compliance = torch.zeros(2, 2)
        snapshots = tuple(tensor.clone() for tensor in (reference, forces, compliance))

        result = apply_hindsight_target(reference, forces, compliance, spec=spec)

        self.assertTrue(torch.equal(result, reference))
        for tensor, snapshot in zip((reference, forces, compliance), snapshots, strict=True):
            self.assertTrue(torch.equal(tensor, snapshot))

    def test_mixed_batch_and_per_site_hard_gates_are_exact(self):
        spec = _spec(["left", "right"])
        reference = torch.randn(3, 2, 2, 3)
        forces = torch.ones(3, 2, 3)
        compliance = torch.full((3, 2), 0.2)

        batch_gate = torch.tensor([True, False, True])
        batch_result = apply_hindsight_target(
            reference,
            forces,
            compliance,
            spec=spec,
            enabled=batch_gate,
        )
        self.assertTrue(torch.equal(batch_result[1], reference[1]))
        torch.testing.assert_close(batch_result[[0, 2]], reference[[0, 2]] - 0.2)

        granular_gate = torch.tensor([[[True, False], [False, True]]])
        granular_result = apply_hindsight_target(
            reference[:1],
            forces[:1],
            compliance[:1],
            spec=spec,
            enabled=granular_gate,
        )
        expected = torch.where(
            granular_gate.unsqueeze(-1),
            reference[:1] - 0.2,
            reference[:1],
        )
        self.assertTrue(torch.equal(granular_result, expected))

    def test_mixed_gate_backward_is_zero_for_disabled_force_and_compliance(self):
        spec = _spec(["contact"])
        reference = torch.zeros(2, 1, 1, 3, requires_grad=True)
        forces = torch.ones(2, 1, 3, requires_grad=True)
        compliance = torch.full((2, 1), 0.2, requires_grad=True)
        result = apply_hindsight_target(
            reference,
            forces,
            compliance,
            spec=spec,
            enabled=torch.tensor([False, True]),
        )
        result.sum().backward()

        self.assertTrue(torch.equal(result[0], reference.detach()[0]))
        torch.testing.assert_close(forces.grad[0], torch.zeros_like(forces.grad[0]))
        torch.testing.assert_close(compliance.grad[0], torch.zeros_like(compliance.grad[0]))
        torch.testing.assert_close(forces.grad[1], torch.full_like(forces.grad[1], -0.2))
        torch.testing.assert_close(compliance.grad[1], torch.tensor([-3.0]))

    def test_chip_sign_and_static_future_broadcast(self):
        spec = _spec(["left", "right"])
        reference = torch.zeros(1, 2, 2, 3)
        forces = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        compliance = torch.tensor([[0.1, 0.2]])

        result = apply_hindsight_target(reference, forces, compliance, spec=spec)
        expected_frame = torch.tensor([[-0.1, 0.0, 0.0], [0.0, -0.4, 0.0]])
        torch.testing.assert_close(result[0, 0], expected_frame)
        torch.testing.assert_close(result[0, 1], expected_frame)

    def test_future_specific_force_and_compliance(self):
        spec = _spec(["contact"])
        reference = torch.zeros(1, 2, 1, 3)
        forces = torch.tensor([[[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]])
        compliance = torch.tensor([[[0.1], [0.2]]])
        result = apply_hindsight_target(reference, forces, compliance, spec=spec)
        torch.testing.assert_close(result[..., 0].flatten(), torch.tensor([-0.1, -0.4]))

    def test_anisotropic_compliance_preserves_unselected_axes(self):
        spec = _spec(["left_wrist", "right_wrist"])
        reference = torch.zeros(1, 4, 2, 3)
        forces = torch.ones(1, 2, 3)
        anisotropic_static = torch.tensor([[[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]]])
        result = apply_hindsight_target(
            reference,
            forces,
            anisotropic_static,
            spec=spec,
        )
        expected = -anisotropic_static.view(1, 1, 2, 3).expand_as(reference)
        torch.testing.assert_close(result, expected)

        anisotropic_future = torch.zeros(1, 4, 2, 3)
        anisotropic_future[:, :, :, 2] = 0.05
        future_result = apply_hindsight_target(
            reference,
            forces,
            anisotropic_future,
            spec=spec,
        )
        torch.testing.assert_close(future_result, -anisotropic_future)

    def test_arbitrary_site_count_and_mask(self):
        names = [f"tracker_site_{index}" for index in range(17)]
        spec = _spec(names)
        reference = torch.zeros(2, 3, len(names), 3)
        forces = torch.ones(2, len(names), 3)
        compliance = torch.full((2, len(names)), 0.1)
        mask = torch.zeros(len(names), dtype=torch.bool)
        mask[[3, 16]] = True

        result = apply_hindsight_target(
            reference,
            forces,
            compliance,
            spec=spec,
            site_mask=mask,
        )
        self.assertTrue(torch.equal(result[:, :, ~mask], reference[:, :, ~mask]))
        torch.testing.assert_close(result[:, :, mask], torch.full_like(result[:, :, mask], -0.1))

    def test_displacement_limit_is_vector_norm_based(self):
        spec = _spec(["contact"], max_displacement_m=0.2)
        reference = torch.zeros(1, 1, 1, 3)
        forces = torch.tensor([[[3.0, 4.0, 0.0]]])
        compliance = torch.ones(1, 1)
        result = apply_hindsight_target(reference, forces, compliance, spec=spec)
        torch.testing.assert_close(result.flatten(), torch.tensor([-0.12, -0.16, 0.0]))

    def test_gradients_flow_through_adapter(self):
        spec = _spec(["contact"])
        reference = torch.zeros(1, 1, 1, 3, requires_grad=True)
        forces = torch.ones(1, 1, 3, requires_grad=True)
        compliance = torch.full((1, 1), 0.2, requires_grad=True)
        result = apply_hindsight_target(reference, forces, compliance, spec=spec)
        result.sum().backward()
        torch.testing.assert_close(reference.grad, torch.ones_like(reference))
        torch.testing.assert_close(forces.grad, torch.full_like(forces, -0.2))
        torch.testing.assert_close(compliance.grad, torch.tensor([[-3.0]]))

    def test_nonfinite_and_negative_values_are_rejected_when_enabled(self):
        spec = _spec(["contact"])
        reference = torch.zeros(2, 1, 1, 3)
        force = torch.zeros(2, 1, 3)
        compliance = torch.zeros(2, 1)

        invalid_reference = reference.clone()
        invalid_reference[0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "reference_positions.*finite"):
            apply_hindsight_target(invalid_reference, force, compliance, spec=spec, enabled=False)
        invalid_force = force.clone()
        invalid_force[0, 0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "external_forces.*finite"):
            apply_hindsight_target(reference, invalid_force, compliance, spec=spec)
        invalid_compliance = compliance.clone()
        invalid_compliance[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "compliance.*finite"):
            apply_hindsight_target(reference, force, invalid_compliance, spec=spec)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            apply_hindsight_target(reference, force, compliance - 0.1, spec=spec)

        inactive_nan_force = force.clone()
        inactive_nan_force[0] = torch.nan
        with self.assertRaisesRegex(ValueError, "external_forces.*finite"):
            apply_hindsight_target(
                reference,
                inactive_nan_force,
                compliance,
                spec=spec,
                enabled=torch.tensor([False, True]),
            )

    def test_invalid_and_known_five_dimensional_shapes_are_rejected(self):
        spec = _spec(["left", "right"])
        reference = torch.zeros(2, 3, 2, 3)
        compliance = torch.zeros(2, 2)
        malformed_force = torch.zeros(2, 2, 1, 1, 3)
        with self.assertRaisesRegex(ValueError, "external_forces"):
            apply_hindsight_target(reference, malformed_force, compliance, spec=spec)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            apply_hindsight_target(
                torch.zeros(2, 3, 1, 3),
                None,
                None,
                spec=spec,
                enabled=False,
            )
        with self.assertRaisesRegex(TypeError, "dtype"):
            apply_hindsight_target(
                reference,
                torch.zeros(2, 2, 3, dtype=torch.float64),
                compliance,
                spec=spec,
            )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            apply_hindsight_target(
                torch.zeros(1, 3, 3, 3),
                torch.zeros(1, 3, 3),
                torch.zeros(1, 3, 3),
                spec=_spec(["a", "b", "c"]),
            )
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            apply_hindsight_target(
                reference,
                torch.zeros(2, 2, 3),
                compliance,
                spec=spec,
                enabled=torch.ones(2),
            )


class TargetDamperTest(unittest.TestCase):
    def test_update_matches_chip_target_damper_and_preserves_gradients(self):
        damper = TargetDamper(alpha=0.25)
        initial = torch.zeros(2, 2, 3)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            damper.update(initial)
        damper.reset(initial)

        current = torch.full((2, 2, 3), 2.0, requires_grad=True)
        first = damper.update(current)
        torch.testing.assert_close(first, torch.full_like(first, 0.5))
        first.sum().backward()
        torch.testing.assert_close(current.grad, torch.full_like(current, 0.25))

        second = damper.update(torch.full((2, 2, 3), 2.0))
        torch.testing.assert_close(second, torch.full_like(second, 0.875))

    def test_full_and_partial_reset_are_non_aliasing(self):
        damper = TargetDamper(alpha=0.5)
        initial = torch.zeros(2, 1, 3)
        damper.reset(initial)
        initial.add_(100.0)
        torch.testing.assert_close(damper.previous_target, torch.zeros(2, 1, 3))

        reset_values = torch.tensor([[[10.0, 10.0, 10.0]], [[20.0, 20.0, 20.0]]])
        state = damper.reset(reset_values, reset_mask=torch.tensor([True, False]))
        torch.testing.assert_close(state[0], reset_values[0])
        torch.testing.assert_close(state[1], torch.zeros_like(state[1]))
        state.add_(5.0)
        torch.testing.assert_close(damper.previous_target[1], torch.zeros_like(state[1]))

    def test_damper_rejects_invalid_state_and_reset_contracts(self):
        with self.assertRaises((TypeError, ValueError)):
            TargetDamper(alpha=float("nan"))
        with self.assertRaises(ValueError):
            TargetDamper(alpha=1.1)
        damper = TargetDamper(alpha=0.5)
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            damper.reset(torch.zeros(2, 1, 3), reset_mask=torch.tensor([True, False]))
        with self.assertRaisesRegex(ValueError, "finite"):
            damper.reset(torch.full((2, 1, 3), torch.nan))


class ScheduleAndMetricsTest(unittest.TestCase):
    def test_pyramid_schedule(self):
        phase = torch.tensor([-0.1, 0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0, 1.1])
        phase_snapshot = phase.clone()
        result = pyramid_phase_weight(phase)
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.equal(phase, phase_snapshot))

    def test_response_metrics_measure_true_exposure(self):
        spec = _spec(["active", "ignored"])
        reference = torch.zeros(1, 2, 2, 3)
        compliant = reference.clone()
        compliant[:, :, 0, 0] = torch.tensor([1.0, 3.0])
        compliant[:, :, 1, 0] = 100.0
        mask = torch.tensor([True, False])
        compliance = torch.tensor([[0.2, 0.2]])

        metrics = summarize_compliance_response(
            reference,
            compliant,
            spec=spec,
            compliance=compliance,
            site_mask=mask,
        )
        torch.testing.assert_close(metrics.mean_displacement_m, torch.tensor(2.0))
        torch.testing.assert_close(metrics.max_displacement_m, torch.tensor(3.0))
        torch.testing.assert_close(
            metrics.per_site_mean_displacement_m,
            torch.tensor([2.0, 0.0]),
        )
        torch.testing.assert_close(metrics.active_fraction, torch.tensor(0.5))

        for enabled, zero_compliance in ((False, compliance), (True, torch.zeros_like(compliance))):
            disabled_metrics = summarize_compliance_response(
                reference,
                compliant,
                spec=spec,
                compliance=zero_compliance,
                enabled=enabled,
                site_mask=torch.ones(2, dtype=torch.bool),
            )
            torch.testing.assert_close(disabled_metrics.mean_displacement_m, torch.tensor(0.0))
            torch.testing.assert_close(disabled_metrics.max_displacement_m, torch.tensor(0.0))
            torch.testing.assert_close(disabled_metrics.active_fraction, torch.tensor(0.0))


class SonicAdapterBoundaryTest(unittest.TestCase):
    def test_dual_runtime_name_resolution_keeps_index_spaces_separate(self):
        reference_order = ["pelvis", "left_hand", "right_hand", "head"]
        articulation_order = ["head", "right_hand", "pelvis", "left_hand"]
        requested = ["left_hand", "head"]
        resolved = resolve_compliance_sites(
            reference_order,
            articulation_order,
            requested,
            target_frame=TEST_FRAME,
            force_frame=TEST_FRAME,
        )

        self.assertEqual(resolved.spec.site_names, tuple(requested))
        self.assertEqual(resolved.reference_indices, (1, 3))
        self.assertEqual(resolved.articulation_indices, (3, 0))
        self.assertIs(resolved.reference.index_space, SiteIndexSpace.REFERENCE)
        self.assertIs(resolved.articulation.index_space, SiteIndexSpace.ARTICULATION)

    def test_resolvers_reject_strings_missing_names_and_inconsistent_site_order(self):
        for bad_names in ("left_hand", b"left_hand"):
            with self.subTest(bad_names=bad_names), self.assertRaises(TypeError):
                resolve_site_indices(
                    bad_names,
                    ["left_hand"],
                    index_space=SiteIndexSpace.REFERENCE,
                )
            with self.subTest(bad_names=bad_names), self.assertRaises(TypeError):
                resolve_site_indices(
                    ["left_hand"],
                    bad_names,
                    index_space=SiteIndexSpace.REFERENCE,
                )
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_site_indices(
                ["left_hand"],
                ["unknown_body"],
                index_space=SiteIndexSpace.ARTICULATION,
            )

        spec = _spec(["left_hand"])
        wrong_reference = NamedSiteIndices(
            index_space=SiteIndexSpace.REFERENCE,
            site_names=("right_hand",),
            indices=(0,),
        )
        articulation = NamedSiteIndices(
            index_space=SiteIndexSpace.ARTICULATION,
            site_names=spec.site_names,
            indices=(0,),
        )
        with self.assertRaisesRegex(ValueError, "reference site_names"):
            SonicComplianceSites(
                spec=spec,
                reference=wrong_reference,
                articulation=articulation,
            )

    def test_core_imports_and_production_literals_are_portable(self):
        compliance_root = Path(__file__).parents[2] / "compliance_control"
        core_root = compliance_root / "core"
        forbidden_import_prefixes = ("isaaclab", "gear_sonic")

        for path in core_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.append(node.module)
            self.assertFalse(
                any(
                    module.startswith(prefix)
                    for module in imported_modules
                    for prefix in forbidden_import_prefixes
                ),
                msg=f"tracker-specific import found in {path}",
            )

        for path in compliance_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            integer_literals = {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and type(node.value) is int
            }
            self.assertTrue(
                integer_literals.isdisjoint({14, 29}),
                msg=f"fixed SONIC body/DoF contract found in {path}",
            )


if __name__ == "__main__":
    unittest.main()
