"""Phase-2 tensor, state, frame, offset, schedule, and wrench tests."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.compliance_control.adapters.sonic import (
    ArticulationWrenchAdapter,
    ComplianceOperationalControl,
    SiteIndexSpace,
    SonicComplianceCommandState,
    WrenchWriteGate,
    advance_pulse_countdown_prevalidated,
    build_sonic_compliance_targets,
    build_sonic_compliance_targets_prevalidated,
    frame_positions_to_world,
    limit_peak_forces_by_net_wrench,
    limit_peak_forces_by_net_wrench_prevalidated,
    mask_requested_peak_forces,
    mask_requested_peak_forces_prevalidated,
    resolve_compliance_sites,
    resolve_site_indices,
    reschedule_pulse_countdown_mask_prevalidated,
    reschedule_pulse_countdown_prevalidated,
    sample_compliance_pulses,
    sample_compliance_pulses_prevalidated,
    select_articulation_sites,
    select_reference_sites,
    world_positions_to_frame,
    world_vectors_to_frame,
)
from gear_sonic.compliance_control import CartesianFrameSpec


def _quaternion(axis: str, angle_rad: float, *, device: torch.device) -> torch.Tensor:
    half = 0.5 * angle_rad
    components = {
        "x": (math.cos(half), math.sin(half), 0.0, 0.0),
        "y": (math.cos(half), 0.0, math.sin(half), 0.0),
        "z": (math.cos(half), 0.0, 0.0, math.sin(half)),
    }
    return torch.tensor(components[axis], dtype=torch.float32, device=device)


def _make_state(
    *,
    device: torch.device,
    num_envs: int = 2,
    num_future: int = 3,
    site_names: tuple[str, ...] = ("left", "right"),
    frame: CartesianFrameSpec | None = None,
    offsets: torch.Tensor | None = None,
) -> SonicComplianceCommandState:
    reference_names = ("root", *site_names, "tail")
    articulation_names = ("tail", *reversed(site_names), "root")
    common_frame = frame or CartesianFrameSpec.world()
    sites = resolve_compliance_sites(
        reference_names,
        articulation_names,
        site_names,
        target_frame=common_frame,
        force_frame=common_frame,
    )
    return SonicComplianceCommandState(
        sites=sites,
        num_envs=num_envs,
        num_future_frames=num_future,
        device=device,
        dtype=torch.float32,
        target_damper_alpha=0.5,
        site_offsets_local=offsets,
    )


class StructuredFrameAndObservationTest(unittest.TestCase):
    def test_nonzero_yaw_common_frame_chip_sign_and_nonmutation(self) -> None:
        device = torch.device("cpu")
        frame = CartesianFrameSpec.heading_local("root")
        state = _make_state(
            device=device,
            num_envs=1,
            num_future=2,
            site_names=("wrist",),
            frame=frame,
        )
        anchor_position = torch.tensor([[1.0, 2.0, 0.0]])
        anchor_quaternion = _quaternion("z", math.pi / 2.0, device=device).view(1, 4)
        reference_position = torch.zeros(1, 2, 3, 3)
        reference_position[:, 0, 1] = anchor_position + torch.tensor([[0.0, 2.0, 0.0]])
        reference_position[:, 1, 1] = anchor_position + torch.tensor([[0.0, 3.0, 0.0]])
        reference_quaternion = torch.zeros(1, 2, 3, 4)
        reference_quaternion[..., 0] = 1.0
        articulation_position = torch.zeros(1, 3, 3)
        articulation_position[:, 1] = anchor_position + torch.tensor([[0.0, 0.5, 0.0]])
        articulation_quaternion = torch.zeros(1, 3, 4)
        articulation_quaternion[..., 0] = 1.0
        current_common = world_positions_to_frame(
            articulation_position[:, 1:2],
            frame=frame,
            anchor_position_w=anchor_position,
            anchor_quaternion_wxyz=anchor_quaternion,
        ).unsqueeze(1).expand(1, 2, 1, 3)
        state.reset(current_common)
        state.set_samples(
            None,
            enabled=torch.tensor([True]),
            site_mask=torch.tensor([[True]]),
            compliance=torch.tensor([[0.1]]),
            force_on_robot_w=torch.tensor([[[0.0, 10.0, 0.0]]]),
        )

        reference_before = reference_position.clone()
        state_before = (
            state.enabled,
            state.site_mask,
            state.compliance,
            state.force_on_robot_w,
            state.damped_target_common,
        )
        result = build_sonic_compliance_targets(
            reference_positions_w=reference_position,
            reference_quaternions_wxyz=reference_quaternion,
            articulation_positions_w=articulation_position,
            articulation_quaternions_wxyz=articulation_quaternion,
            anchor_position_w=anchor_position,
            anchor_quaternion_wxyz=anchor_quaternion,
            state=state,
        )

        torch.testing.assert_close(
            result.reference_target_common[0, :, 0],
            torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            result.force_on_robot_common,
            torch.tensor([[[[10.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]]]),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            result.observed_target_common[0, :, 0],
            torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(reference_position, reference_before)
        for actual, expected in zip(
            (
                state.enabled,
                state.site_mask,
                state.compliance,
                state.force_on_robot_w,
                state.damped_target_common,
            ),
            state_before,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)

    def test_full_rotation_round_trip_is_not_yaw_only(self) -> None:
        device = torch.device("cpu")
        frame = CartesianFrameSpec.anchor_local("root")
        anchor_position = torch.tensor([[0.4, -0.2, 1.0]])
        anchor_quaternion = _quaternion("x", math.pi / 2.0, device=device).view(1, 4)
        local_position = torch.tensor([[[0.0, 2.0, 0.0]]])
        world_position = frame_positions_to_world(
            local_position,
            frame=frame,
            anchor_position_w=anchor_position,
            anchor_quaternion_wxyz=anchor_quaternion,
        )
        torch.testing.assert_close(
            world_position,
            torch.tensor([[[0.4, -0.2, 3.0]]]),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            world_positions_to_frame(
                world_position,
                frame=frame,
                anchor_position_w=anchor_position,
                anchor_quaternion_wxyz=anchor_quaternion,
            ),
            local_position,
            atol=1.0e-6,
            rtol=0.0,
        )
        world_force = torch.tensor([[[0.0, 0.0, 7.0]]])
        torch.testing.assert_close(
            world_vectors_to_frame(
                world_force,
                frame=frame,
                anchor_quaternion_wxyz=anchor_quaternion,
            ),
            torch.tensor([[[0.0, 7.0, 0.0]]]),
            atol=2.0e-6,
            rtol=0.0,
        )

    def test_nonzero_local_site_offset_uses_each_space_quaternion(self) -> None:
        device = torch.device("cpu")
        offsets = torch.tensor([[0.2, 0.0, 0.0]])
        state = _make_state(
            device=device,
            num_envs=1,
            num_future=1,
            site_names=("wrist",),
            offsets=offsets,
        )
        reference_position = torch.zeros(1, 1, 3, 3)
        reference_quaternion = torch.zeros(1, 1, 3, 4)
        reference_quaternion[..., 0] = 1.0
        reference_quaternion[:, :, 1] = _quaternion(
            "z", math.pi / 2.0, device=device
        )
        articulation_position = torch.zeros(1, 3, 3)
        articulation_quaternion = torch.zeros(1, 3, 4)
        articulation_quaternion[..., 0] = 1.0
        articulation_quaternion[:, 1] = _quaternion(
            "z", -math.pi / 2.0, device=device
        )
        state.reset(torch.tensor([[[[0.0, -0.2, 0.0]]]]))
        result = build_sonic_compliance_targets(
            reference_positions_w=reference_position,
            reference_quaternions_wxyz=reference_quaternion,
            articulation_positions_w=articulation_position,
            articulation_quaternions_wxyz=articulation_quaternion,
            anchor_position_w=None,
            anchor_quaternion_wxyz=None,
            state=state,
        )
        torch.testing.assert_close(
            result.reference_target_common,
            torch.tensor([[[[0.0, 0.2, 0.0]]]]),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            result.damped_target_common,
            torch.tensor([[[[0.0, -0.2, 0.0]]]]),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_damper_is_optional_nominal_source_and_disabled_rows_are_exact(self) -> None:
        state = _make_state(device=torch.device("cpu"), num_envs=2, num_future=1)
        reference_position = torch.zeros(2, 1, 4, 3)
        reference_position[:, :, 1] = torch.tensor([1.0, 0.0, 0.0])
        reference_position[:, :, 2] = torch.tensor([2.0, 0.0, 0.0])
        reference_quaternion = torch.zeros(2, 1, 4, 4)
        reference_quaternion[..., 0] = 1.0
        articulation_position = torch.zeros(2, 4, 3)
        articulation_quaternion = torch.zeros(2, 4, 4)
        articulation_quaternion[..., 0] = 1.0
        current = torch.zeros(2, 1, 2, 3)
        state.reset(current)
        state.set_samples(
            None,
            enabled=torch.tensor([True, False]),
            site_mask=torch.tensor([[True, True], [True, True]]),
            compliance=torch.tensor([[0.1, 0.1], [0.1, 0.1]]),
            force_on_robot_w=torch.tensor(
                [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]] * 2
            ),
        )
        moved = current.clone()
        moved[0, :, :, 0] = 0.4
        state.update_damper(moved)
        common_kwargs = dict(
            reference_positions_w=reference_position,
            reference_quaternions_wxyz=reference_quaternion,
            articulation_positions_w=articulation_position,
            articulation_quaternions_wxyz=articulation_quaternion,
            anchor_position_w=None,
            anchor_quaternion_wxyz=None,
            state=state,
        )
        default_result = build_sonic_compliance_targets(**common_kwargs)
        damped_result = build_sonic_compliance_targets(
            **common_kwargs,
            use_target_damper=True,
        )
        torch.testing.assert_close(
            default_result.nominal_target_common,
            default_result.reference_target_common,
        )
        torch.testing.assert_close(
            damped_result.nominal_target_common[0],
            state.damped_target_common[0],
        )
        torch.testing.assert_close(
            damped_result.observed_target_common[1],
            damped_result.reference_target_common[1],
            atol=0.0,
            rtol=0.0,
        )
        expected = damped_result.nominal_target_common[0] - torch.tensor(
            [[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]
        )
        torch.testing.assert_close(damped_result.observed_target_common[0], expected)


class TypedIndexAndShapeTest(unittest.TestCase):
    def test_each_consumer_rejects_the_other_index_space(self) -> None:
        reference = resolve_site_indices(
            ("root", "left", "right"),
            ("right", "left"),
            index_space=SiteIndexSpace.REFERENCE,
        )
        articulation = resolve_site_indices(
            ("right", "root", "left"),
            ("right", "left"),
            index_space=SiteIndexSpace.ARTICULATION,
        )
        self.assertEqual(reference.indices, (2, 1))
        self.assertEqual(articulation.indices, (0, 2))
        reference_tensor = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
        articulation_tensor = torch.arange(9, dtype=torch.float32).view(1, 3, 3)
        torch.testing.assert_close(
            select_reference_sites(reference_tensor, reference),
            reference_tensor[:, :, [2, 1]],
        )
        torch.testing.assert_close(
            select_articulation_sites(articulation_tensor, articulation),
            articulation_tensor[:, [0, 2]],
        )
        with self.assertRaisesRegex(ValueError, "reference-space"):
            select_reference_sites(reference_tensor, articulation)
        with self.assertRaisesRegex(ValueError, "articulation-space"):
            select_articulation_sites(articulation_tensor, reference)

    def _run_future_site_alignment(self, device: torch.device) -> None:
        site_names = tuple(f"site_{index}" for index in range(5))
        state = _make_state(
            device=device,
            num_envs=2,
            num_future=4,
            site_names=site_names,
        )
        reference = torch.arange(
            2 * 4 * 7 * 3,
            device=device,
            dtype=torch.float32,
        ).view(2, 4, 7, 3) / 100.0
        reference_quaternion = torch.zeros(2, 4, 7, 4, device=device)
        reference_quaternion[..., 0] = 1.0
        articulation = torch.zeros(2, 7, 3, device=device)
        articulation_quaternion = torch.zeros(2, 7, 4, device=device)
        articulation_quaternion[..., 0] = 1.0
        state.reset(torch.zeros(2, 4, 5, 3, device=device))
        enabled = torch.tensor([True, True], device=device)
        mask = torch.tensor(
            [[True, True, False, False, False], [True, True, True, True, True]],
            device=device,
        )
        compliance = torch.full((2, 5), 0.01, device=device)
        force = torch.arange(2 * 5 * 3, device=device, dtype=torch.float32).view(2, 5, 3)
        state.set_samples(
            None,
            enabled=enabled,
            site_mask=mask,
            compliance=compliance,
            force_on_robot_w=force,
        )
        before = reference.clone()
        targets = build_sonic_compliance_targets(
            reference_positions_w=reference,
            reference_quaternions_wxyz=reference_quaternion,
            articulation_positions_w=articulation,
            articulation_quaternions_wxyz=articulation_quaternion,
            anchor_position_w=None,
            anchor_quaternion_wxyz=None,
            state=state,
        )
        self.assertEqual(tuple(targets.observed_target_common.shape), (2, 4, 5, 3))
        expected = targets.reference_target_common - 0.01 * force[:, None]
        expected = torch.where(mask[:, None, :, None], expected, targets.reference_target_common)
        torch.testing.assert_close(targets.observed_target_common, expected)
        torch.testing.assert_close(reference, before)

    def test_cpu_future_site_alignment_and_simultaneous_sites(self) -> None:
        self._run_future_site_alignment(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_future_site_alignment_and_simultaneous_sites(self) -> None:
        self._run_future_site_alignment(torch.device("cuda:0"))

    def test_full_and_partial_name_sets_are_arbitrary(self) -> None:
        names = tuple(f"body_{index}" for index in range(17))
        articulation = tuple(reversed(names))
        full = resolve_compliance_sites(
            names,
            articulation,
            names,
            target_frame=CartesianFrameSpec.world(),
            force_frame=CartesianFrameSpec.world(),
        )
        partial_names = (names[2], names[9], names[16])
        partial = resolve_compliance_sites(
            names,
            articulation,
            partial_names,
            target_frame=CartesianFrameSpec.world(),
            force_frame=CartesianFrameSpec.world(),
        )
        self.assertEqual(full.reference_indices, tuple(range(17)))
        self.assertEqual(full.articulation_indices, tuple(reversed(range(17))))
        self.assertEqual(partial.reference_indices, (2, 9, 16))
        self.assertEqual(partial.articulation_indices, (14, 7, 0))


class PulseScheduleAndResetTest(unittest.TestCase):
    def test_rise_hold_fall_async_completion_and_partial_reset(self) -> None:
        state = _make_state(device=torch.device("cpu"), num_envs=2, num_future=2)
        initial = torch.zeros(2, 2, 2, 3)
        initial[1] = 3.0
        state.reset(initial)
        state.start_pulses(
            None,
            enabled=torch.tensor([True, True]),
            site_mask=torch.tensor([[True, True], [True, False]]),
            compliance=torch.full((2, 2), 0.02),
            peak_force_on_robot_w=torch.full((2, 2, 3), 10.0),
            duration_s=torch.tensor([1.0, 2.0]),
        )
        first = state.advance_force_schedule(0.1)
        torch.testing.assert_close(first[0, 0], torch.full((3,), 5.0))
        torch.testing.assert_close(first[1, 0], torch.full((3,), 2.5))
        torch.testing.assert_close(first[1, 1], torch.zeros(3))
        second = state.advance_force_schedule(0.1)
        torch.testing.assert_close(second[0, 0], torch.full((3,), 10.0))
        torch.testing.assert_close(second[1, 0], torch.full((3,), 5.0))
        for _ in range(8):
            state.advance_force_schedule(0.1)
        self.assertFalse(state.pulse_active[0])
        self.assertTrue(state.pulse_active[1])
        torch.testing.assert_close(state.force_on_robot_w[0], torch.zeros(2, 3))

        moved = initial + 4.0
        state.update_damper(moved)
        state.reset(initial, env_ids=torch.tensor([1], dtype=torch.int64))
        self.assertFalse(state.pulse_active[1])
        torch.testing.assert_close(state.force_on_robot_w[1], torch.zeros(2, 3))
        torch.testing.assert_close(state.compliance[1], torch.zeros(2, 3))
        torch.testing.assert_close(state.damped_target_common[1], initial[1])

    def test_tensor_env_ids_and_slice_validation(self) -> None:
        state = _make_state(device=torch.device("cpu"), num_envs=3, num_future=1)
        current = torch.zeros(3, 1, 2, 3)
        state.reset(current)
        state.reset(current, env_ids=torch.tensor([2, 0], dtype=torch.int32))
        state.reset(current, env_ids=torch.tensor([False, True, False]))
        state.reset(current, env_ids=slice(None))
        with self.assertRaises(TypeError):
            state.reset(current, env_ids=slice(0, 1))
        with self.assertRaises(ValueError):
            state.reset(current, env_ids=torch.tensor([[0]], dtype=torch.int64))

    def test_damper_activation_edge_seeds_current_eef_after_stiff_motion(self) -> None:
        state = _make_state(device=torch.device("cpu"), num_envs=1, num_future=2)
        initial = torch.zeros(1, 2, 2, 3)
        state.reset(initial)
        moved = initial.clone()
        moved[:, :, 0, 0] = 3.0
        moved[:, :, 1, 0] = 4.0
        state.update_damper(moved)
        torch.testing.assert_close(state.damped_target_common, initial)

        state.seed_damper_sites(
            moved,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[True, False]]),
        )
        torch.testing.assert_close(state.damped_target_common[:, :, 0], moved[:, :, 0])
        torch.testing.assert_close(state.damped_target_common[:, :, 1], initial[:, :, 1])
        state.start_pulses(
            None,
            enabled=torch.tensor([True]),
            site_mask=torch.tensor([[True, False]]),
            compliance=torch.tensor([[0.1, 0.1]]),
            peak_force_on_robot_w=torch.zeros(1, 2, 3),
            duration_s=torch.tensor([1.0]),
        )

        reference_position = torch.zeros(1, 2, 4, 3)
        reference_position[:, :, 1, 0] = 10.0
        reference_position[:, :, 2, 0] = 20.0
        reference_quaternion = torch.zeros(1, 2, 4, 4)
        reference_quaternion[..., 0] = 1.0
        articulation_position = torch.zeros(1, 4, 3)
        articulation_quaternion = torch.zeros(1, 4, 4)
        articulation_quaternion[..., 0] = 1.0
        targets = build_sonic_compliance_targets(
            reference_positions_w=reference_position,
            reference_quaternions_wxyz=reference_quaternion,
            articulation_positions_w=articulation_position,
            articulation_quaternions_wxyz=articulation_quaternion,
            anchor_position_w=None,
            anchor_quaternion_wxyz=None,
            state=state,
            use_target_damper=True,
        )
        torch.testing.assert_close(targets.nominal_target_common[:, :, 0], moved[:, :, 0])
        torch.testing.assert_close(
            targets.nominal_target_common[:, :, 1],
            targets.reference_target_common[:, :, 1],
        )


class SamplingAndWrenchTest(unittest.TestCase):
    def _assert_private_countdown_rng_and_partial_reset(self, device: torch.device) -> None:
        private_generator = torch.Generator(device=device).manual_seed(731)
        private_before = private_generator.get_state().clone()
        countdown = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        selected = torch.tensor([1, 3], dtype=torch.long, device=device)
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        )

        reschedule_pulse_countdown_prevalidated(
            countdown,
            selected,
            globally_enabled=False,
            interval_range_s=(3.5, 6.0),
            generator=private_generator,
        )
        due = advance_pulse_countdown_prevalidated(
            countdown,
            0.02,
            globally_enabled=False,
        )
        self.assertFalse(due.any())
        self.assertTrue(torch.isinf(countdown).all())
        self.assertTrue(torch.equal(private_generator.get_state(), private_before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng_before))
        if cuda_rng_before is not None:
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before))

        generator_a = torch.Generator(device=device).manual_seed(99)
        generator_b = torch.Generator(device=device).manual_seed(99)
        countdown_a = torch.full((4,), 9.0, device=device)
        countdown_b = countdown_a.clone()
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        )
        for values, generator in (
            (countdown_a, generator_a),
            (countdown_b, generator_b),
        ):
            reschedule_pulse_countdown_prevalidated(
                values,
                selected,
                globally_enabled=True,
                interval_range_s=(3.5, 6.0),
                generator=generator,
            )
        torch.testing.assert_close(countdown_a, countdown_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(countdown_a[[0, 2]], torch.tensor([9.0, 9.0], device=device))
        self.assertTrue((countdown_a[selected] >= 3.5).all())
        self.assertTrue((countdown_a[selected] <= 6.0).all())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng_before))
        if cuda_rng_before is not None:
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before))

        countdown_a.copy_(torch.tensor([0.1, 0.3, 0.1, 0.5], device=device))
        due = advance_pulse_countdown_prevalidated(
            countdown_a,
            0.2,
            globally_enabled=True,
        )
        torch.testing.assert_close(
            due,
            torch.tensor([True, False, True, False], device=device),
        )
        unaffected_before = countdown_a[[1, 3]].clone()
        reschedule_pulse_countdown_mask_prevalidated(
            countdown_a,
            due,
            interval_range_s=(3.5, 6.0),
            generator=generator_a,
        )
        torch.testing.assert_close(countdown_a[[1, 3]], unaffected_before)

    def test_private_countdown_cpu_rng_and_partial_reset(self) -> None:
        self._assert_private_countdown_rng_and_partial_reset(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_private_countdown_cuda_rng_and_partial_reset(self) -> None:
        self._assert_private_countdown_rng_and_partial_reset(torch.device("cuda:0"))

    def test_disabled_wrench_gate_never_writes_and_clears_active_once(self) -> None:
        class FakeWrench:
            def __init__(self) -> None:
                self.clear_env_ids = []

            def clear(self, env_ids=None) -> None:
                self.clear_env_ids.append(env_ids)

        writer = FakeWrench()
        gate = WrenchWriteGate()
        self.assertFalse(gate.was_written)
        self.assertFalse(gate.consume_clear_on_disable())
        gate.mark_written()
        self.assertTrue(gate.was_written)
        if gate.consume_clear_on_reset(globally_enabled=False):
            writer.clear()
        self.assertFalse(gate.was_written)
        self.assertFalse(gate.consume_clear_on_disable())
        self.assertEqual(writer.clear_env_ids, [None])

        gate.mark_written()
        self.assertTrue(gate.consume_clear_on_reset(globally_enabled=True))
        writer.clear([1])
        self.assertTrue(gate.was_written)
        self.assertTrue(gate.consume_clear_on_disable())
        writer.clear()
        self.assertEqual(writer.clear_env_ids, [None, [1], None])
        self.assertFalse(gate.consume_clear_on_disable())

        with self.assertRaisesRegex(TypeError, "globally_enabled"):
            gate.consume_clear_on_reset(globally_enabled=1)

    def test_seeded_fourteen_site_sampling_is_bounded_and_reproducible(self) -> None:
        kwargs = dict(
            num_envs=32,
            num_sites=14,
            device=torch.device("cpu"),
            dtype=torch.float32,
            globally_enabled=True,
            enabled_probability=1.0,
            site_probability=1.0,
            force_magnitude_range_n=(20.0, 20.0),
            compliance_values_m_per_n=(0.0, 0.02, 0.05),
            duration_range_s=(1.0, 3.0),
            max_active_sites=3,
        )
        generator_a = torch.Generator().manual_seed(123)
        generator_b = torch.Generator().manual_seed(123)
        sample_a = sample_compliance_pulses(generator=generator_a, **kwargs)
        sample_b = sample_compliance_pulses(generator=generator_b, **kwargs)
        for field in (
            "enabled",
            "site_mask",
            "compliance",
            "peak_force_on_robot_w",
            "duration_s",
        ):
            torch.testing.assert_close(getattr(sample_a, field), getattr(sample_b, field))
        self.assertTrue((sample_a.site_mask.sum(dim=-1) <= 3).all())
        self.assertTrue((sample_a.site_mask.sum(dim=-1) >= 1).all())
        torch.testing.assert_close(
            sample_a.compliance.unique(),
            torch.tensor([0.0, 0.02, 0.05]),
        )

        positions = torch.randn(32, 14, 3)
        origin = torch.zeros(32, 3)
        masked_force = torch.where(
            sample_a.site_mask.unsqueeze(-1),
            sample_a.peak_force_on_robot_w,
            0.0,
        )
        limited = limit_peak_forces_by_net_wrench(
            masked_force,
            positions,
            origin,
            max_net_force_n=25.0,
            max_net_torque_nm=10.0,
        )
        net_force = limited.sum(dim=1)
        net_torque = torch.linalg.cross(positions, limited, dim=-1).sum(dim=1)
        self.assertTrue(torch.isfinite(limited).all())
        self.assertTrue((torch.linalg.vector_norm(net_force, dim=-1) <= 25.0001).all())
        self.assertTrue((torch.linalg.vector_norm(net_torque, dim=-1) <= 10.0001).all())

    def test_prevalidated_sampling_and_start_match_strict_public_boundaries(self) -> None:
        device = torch.device("cpu")
        sampling_kwargs = dict(
            num_envs=4,
            num_sites=3,
            device=device,
            dtype=torch.float32,
            globally_enabled=True,
            enabled_probability=1.0,
            site_probability=0.75,
            force_magnitude_range_n=(5.0, 20.0),
            duration_range_s=(1.0, 3.0),
            max_active_sites=2,
        )
        public_samples = sample_compliance_pulses(
            generator=torch.Generator().manual_seed(321),
            compliance_values_m_per_n=(0.0, 0.02, 0.05),
            **sampling_kwargs,
        )
        fast_samples = sample_compliance_pulses_prevalidated(
            generator=torch.Generator().manual_seed(321),
            compliance_values_m_per_n=torch.tensor([0.0, 0.02, 0.05]),
            **sampling_kwargs,
        )
        for field in (
            "enabled",
            "site_mask",
            "compliance",
            "peak_force_on_robot_w",
            "duration_s",
        ):
            torch.testing.assert_close(
                getattr(public_samples, field),
                getattr(fast_samples, field),
            )

        positions = torch.arange(36, dtype=torch.float32).view(4, 3, 3) / 10.0
        origins = torch.zeros(4, 3)
        public_requested = mask_requested_peak_forces(
            public_samples.peak_force_on_robot_w,
            public_samples.enabled,
            public_samples.site_mask,
        )
        fast_requested = mask_requested_peak_forces_prevalidated(
            fast_samples.peak_force_on_robot_w,
            fast_samples.enabled,
            fast_samples.site_mask,
        )
        public_peak = limit_peak_forces_by_net_wrench(
            public_requested,
            positions,
            origins,
            max_net_force_n=25.0,
            max_net_torque_nm=10.0,
        )
        fast_peak = limit_peak_forces_by_net_wrench_prevalidated(
            fast_requested,
            positions,
            origins,
            max_net_force_n=25.0,
            max_net_torque_nm=10.0,
        )
        torch.testing.assert_close(public_peak, fast_peak)

        public_state = _make_state(
            device=device,
            num_envs=4,
            num_future=1,
            site_names=("a", "b", "c"),
        )
        fast_state = _make_state(
            device=device,
            num_envs=4,
            num_future=1,
            site_names=("a", "b", "c"),
        )
        current = torch.zeros(4, 1, 3, 3)
        public_state.reset(current)
        fast_state.reset(current)
        compliance = public_samples.compliance.unsqueeze(-1).expand(4, 3, 3)
        public_state.start_pulses(
            None,
            enabled=public_samples.enabled,
            site_mask=public_samples.site_mask,
            compliance=compliance,
            peak_force_on_robot_w=public_peak,
            duration_s=public_samples.duration_s,
        )
        fast_state.start_pulses_masked_prevalidated(
            torch.ones(4, dtype=torch.bool),
            enabled=fast_samples.enabled,
            site_mask=fast_samples.site_mask,
            compliance=compliance,
            peak_force_on_robot_w=fast_peak,
            duration_s=fast_samples.duration_s,
        )
        for attribute in (
            "enabled",
            "site_mask",
            "compliance",
            "force_on_robot_w",
            "peak_force_on_robot_w",
            "pulse_active",
            "pulse_elapsed_s",
            "pulse_duration_s",
        ):
            torch.testing.assert_close(
                getattr(public_state, attribute),
                getattr(fast_state, attribute),
            )

        invalid_duration = public_samples.duration_s.clone()
        invalid_duration[0] = 0.0
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            public_state.start_pulses(
                None,
                enabled=public_samples.enabled,
                site_mask=public_samples.site_mask,
                compliance=compliance,
                peak_force_on_robot_w=public_peak,
                duration_s=invalid_duration,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            public_state.start_pulses(
                torch.tensor([0, 0]),
                enabled=torch.ones(2, dtype=torch.bool),
                site_mask=torch.ones(2, 3, dtype=torch.bool),
                compliance=torch.ones(2, 3, 3),
                peak_force_on_robot_w=torch.ones(2, 3, 3),
                duration_s=torch.ones(2),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            public_state.set_samples(
                torch.tensor([0]),
                enabled=torch.ones(1, dtype=torch.bool),
                site_mask=torch.ones(1, 3, dtype=torch.bool),
                compliance=torch.full((1, 3, 3), float("nan")),
                force_on_robot_w=torch.ones(1, 3, 3),
            )

    def test_inactive_random_peak_does_not_shrink_requested_site(self) -> None:
        peak = torch.tensor([[[5.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]])
        requested = mask_requested_peak_forces(
            peak,
            torch.tensor([True]),
            torch.tensor([[True, False]]),
        )
        limited = limit_peak_forces_by_net_wrench(
            requested,
            torch.zeros(1, 2, 3),
            torch.zeros(1, 3),
            max_net_force_n=10.0,
            max_net_torque_nm=10.0,
        )
        torch.testing.assert_close(limited, torch.tensor([[[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))

    def test_moving_lever_is_relimited_and_state_exposes_applied_force(self) -> None:
        state = _make_state(
            device=torch.device("cpu"),
            num_envs=1,
            num_future=2,
            site_names=("wrist",),
        )
        state.reset(torch.zeros(1, 2, 1, 3))
        state.set_samples(
            None,
            enabled=torch.tensor([True]),
            site_mask=torch.tensor([[True]]),
            compliance=torch.tensor([[0.1]]),
            force_on_robot_w=torch.tensor([[[0.0, 10.0, 0.0]]]),
        )
        force = state.force_on_robot_w
        origin = torch.zeros(1, 3)
        short_lever = limit_peak_forces_by_net_wrench(
            force,
            torch.tensor([[[0.1, 0.0, 0.0]]]),
            origin,
            max_net_force_n=100.0,
            max_net_torque_nm=2.0,
        )
        long_lever = limit_peak_forces_by_net_wrench(
            force,
            torch.tensor([[[1.0, 0.0, 0.0]]]),
            origin,
            max_net_force_n=100.0,
            max_net_torque_nm=2.0,
        )
        torch.testing.assert_close(short_lever, force)
        torch.testing.assert_close(long_lever, torch.tensor([[[0.0, 2.0, 0.0]]]))
        state.set_applied_force_prevalidated(long_lever)

        reference_position = torch.zeros(1, 2, 3, 3)
        reference_quaternion = torch.zeros(1, 2, 3, 4)
        reference_quaternion[..., 0] = 1.0
        articulation_position = torch.zeros(1, 3, 3)
        articulation_quaternion = torch.zeros(1, 3, 4)
        articulation_quaternion[..., 0] = 1.0
        targets = build_sonic_compliance_targets(
            reference_positions_w=reference_position,
            reference_quaternions_wxyz=reference_quaternion,
            articulation_positions_w=articulation_position,
            articulation_quaternions_wxyz=articulation_quaternion,
            anchor_position_w=None,
            anchor_quaternion_wxyz=None,
            state=state,
        )
        torch.testing.assert_close(
            targets.force_on_robot_common,
            long_lever.unsqueeze(1).expand(1, 2, 1, 3),
        )
        torch.testing.assert_close(
            targets.observed_target_common,
            torch.tensor([[[[0.0, -0.2, 0.0]], [[0.0, -0.2, 0.0]]]]),
        )

    def test_wrench_writer_uses_current_body_frame_offset_and_clears(self) -> None:
        class FakeComposer:
            def __init__(self) -> None:
                self.calls = []

            def set_forces_and_torques(self, **kwargs) -> None:
                self.calls.append(kwargs)

        class FakeArticulation:
            def __init__(self) -> None:
                self.permanent_wrench_composer = FakeComposer()

        selection = resolve_site_indices(
            ("root", "right", "left"),
            ("left", "right"),
            index_space=SiteIndexSpace.ARTICULATION,
        )
        fake = FakeArticulation()
        adapter = ArticulationWrenchAdapter(
            fake,
            body_selection=selection,
            num_envs=2,
            device="cpu",
            dtype=torch.float32,
        )
        all_ids_first = adapter._env_ids_tensor(None)  # noqa: SLF001
        all_ids_second = adapter._env_ids_tensor(slice(None))  # noqa: SLF001
        self.assertIs(all_ids_first, all_ids_second)
        torch.testing.assert_close(all_ids_first, torch.tensor([0, 1]))
        force = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        offset = torch.tensor([[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]])
        quaternion = torch.stack(
            (
                _quaternion("z", math.pi / 2.0, device=torch.device("cpu")),
                _quaternion("z", -math.pi / 2.0, device=torch.device("cpu")),
            ),
        ).unsqueeze(0)
        adapter.set_world_forces(
            force,
            env_ids=torch.tensor([1], dtype=torch.int64),
            body_quaternions_wxyz=quaternion,
            application_offsets_local=offset,
        )
        call = fake.permanent_wrench_composer.calls[-1]
        self.assertEqual(call["body_ids"], [2, 1])
        self.assertFalse(call["is_global"])
        torch.testing.assert_close(
            call["forces"],
            torch.tensor([[[0.0, -1.0, 0.0], [-2.0, 0.0, 0.0]]]),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(call["positions"], offset)
        adapter.clear(torch.tensor([1]))
        clear_call = fake.permanent_wrench_composer.calls[-1]
        torch.testing.assert_close(clear_call["forces"], torch.zeros_like(force))
        self.assertIsNone(clear_call["positions"])

    def test_wrench_writer_rejects_reference_indices_and_unsafe_legacy_api(self) -> None:
        reference_selection = resolve_site_indices(
            ("left",),
            ("left",),
            index_space=SiteIndexSpace.REFERENCE,
        )
        with self.assertRaisesRegex(ValueError, "articulation-space"):
            ArticulationWrenchAdapter(
                object(),
                body_selection=reference_selection,
                num_envs=1,
                device="cpu",
                dtype=torch.float32,
            )

        class UnsafeLegacy:
            def set_external_force_and_torque(self, forces, torques, body_ids, env_ids):
                del forces, torques, body_ids, env_ids

        articulation_selection = resolve_site_indices(
            ("left",),
            ("left",),
            index_space=SiteIndexSpace.ARTICULATION,
        )
        adapter = ArticulationWrenchAdapter(
            UnsafeLegacy(),
            body_selection=articulation_selection,
            num_envs=1,
            device="cpu",
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "body-local"):
            adapter.set_world_forces(
                torch.zeros(1, 1, 3),
                body_quaternions_wxyz=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            )


class HotPathHostSyncTest(unittest.TestCase):
    def test_isaac_production_source_routes_only_through_fast_boundaries(self) -> None:
        root = Path(__file__).resolve().parents[2]
        command_tree = ast.parse(
            (root / "compliance_control/adapters/sonic/isaaclab/command.py").read_text()
        )
        operational_tree = ast.parse(
            (root / "compliance_control/adapters/sonic/operational.py").read_text()
        )
        sampling_tree = ast.parse(
            (root / "compliance_control/adapters/sonic/sampling.py").read_text()
        )
        observation_tree = ast.parse(
            (root / "compliance_control/adapters/sonic/isaaclab/observations.py").read_text()
        )
        smoke_tree = ast.parse(
            (root / "scripts/run_chip_compliance_smoke.py").read_text()
        )

        command_class = next(
            node
            for node in command_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SonicComplianceCommand"
        )
        self.assertIsInstance(command_class.bases[0], ast.Name)
        self.assertEqual(command_class.bases[0].id, "ComplianceOperationalControl")
        self.assertNotIn(
            "compute",
            {
                node.name
                for node in command_class.body
                if isinstance(node, ast.FunctionDef)
            },
        )

        def calls_in(function_name: str, tree: ast.AST) -> set[str]:
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            names = set()
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    names.add(node.func.id)
            return names

        update_calls = calls_in("_update_command", command_tree)
        compute_calls = calls_in("compute", operational_tree)
        production_update_calls = calls_in(
            "_update_command_prevalidated",
            operational_tree,
        )
        operational_setter_calls = calls_in("set_operational_enabled", operational_tree)
        resample_calls = calls_in("_resample", command_tree)
        schedule_calls = calls_in("_schedule_next_pulse", operational_tree)
        due_update_calls = calls_in("_update_due_pulses_prevalidated", operational_tree)
        due_sample_calls = calls_in(
            "_sample_and_start_masked_prevalidated",
            operational_tree,
        )
        reset_calls = calls_in("reset_envs", command_tree)
        sample_calls = calls_in("sample_and_apply", command_tree)
        current_position_calls = calls_in("current_site_positions_w", command_tree)
        current_common_calls = calls_in("current_eef_common_future", command_tree)
        observation_calls = calls_in("sonic_compliance_target", observation_tree)
        real_profile_calls = calls_in("_profile_real_bound_compute", smoke_tree)
        self.assertEqual(update_calls, {"_update_command_prevalidated"})
        self.assertIn("_update_metrics", compute_calls)
        self.assertIn("_update_command_prevalidated", compute_calls)
        self.assertIn("set_world_forces_prevalidated", production_update_calls)
        self.assertIn("update_damper_prevalidated", production_update_calls)
        self.assertIn(
            "limit_peak_forces_by_net_wrench_prevalidated",
            production_update_calls,
        )
        self.assertIn("set_applied_force_prevalidated", production_update_calls)
        self.assertIn("consume_clear_on_disable", production_update_calls)
        self.assertIn(
            "advance_pulse_countdown_prevalidated",
            production_update_calls,
        )
        self.assertIn("_update_due_pulses_prevalidated", production_update_calls)
        self.assertNotIn("sample_and_apply", update_calls)
        self.assertIn("mark_written", production_update_calls)
        self.assertIn("cancel_all_prevalidated", operational_setter_calls)
        self.assertIn("advance_pulse_countdown_prevalidated", operational_setter_calls)
        self.assertIn("consume_clear_on_disable", operational_setter_calls)
        self.assertIn("clear", operational_setter_calls)
        self.assertIn("_schedule_next_pulse", operational_setter_calls)
        self.assertNotIn("reset", operational_setter_calls)
        self.assertIn("_resample_command", resample_calls)
        self.assertTrue({"uniform_", "rand", "random_"}.isdisjoint(resample_calls))
        self.assertIn("reschedule_pulse_countdown_prevalidated", schedule_calls)
        self.assertIn("advance_pulse_countdown_prevalidated", due_update_calls)
        self.assertIn("startable_pulse_mask_prevalidated", due_update_calls)
        self.assertIn("_sample_and_start_masked_prevalidated", due_update_calls)
        self.assertIn(
            "reschedule_pulse_countdown_mask_prevalidated",
            due_update_calls,
        )
        self.assertIn("sample_compliance_pulses_prevalidated", due_sample_calls)
        self.assertIn("mask_requested_peak_forces_prevalidated", due_sample_calls)
        self.assertIn("limit_peak_forces_by_net_wrench_prevalidated", due_sample_calls)
        self.assertIn("seed_damper_sites_masked_prevalidated", due_sample_calls)
        self.assertIn("start_pulses_masked_prevalidated", due_sample_calls)
        self.assertTrue(
            {
                "_env_ids_tensor",
                "nonzero",
                "sample_compliance_pulses",
                "mask_requested_peak_forces",
                "limit_peak_forces_by_net_wrench",
                "seed_damper_sites",
                "set_samples",
                "start_pulses",
            }.isdisjoint(due_sample_calls)
        )
        self.assertIn("consume_clear_on_reset", reset_calls)
        self.assertIn("_env_ids_tensor", sample_calls)
        self.assertIn("startable_pulse_mask_prevalidated", sample_calls)
        self.assertIn("_sample_and_start_masked_prevalidated", sample_calls)
        self.assertIn("quaternion_rotate_wxyz_prevalidated", current_position_calls)
        self.assertIn("index_select", current_position_calls)
        self.assertIn("world_positions_to_frame_prevalidated", current_common_calls)
        self.assertIn("build_sonic_compliance_targets_prevalidated", observation_calls)
        self.assertTrue(
            {
                "compute",
                "profile",
                "record_function",
                "set_operational_enabled",
                "synchronize",
                "index_select",
            }.issubset(real_profile_calls)
        )
        for calls in (
            update_calls,
            compute_calls,
            production_update_calls,
            reset_calls,
            sample_calls,
            current_position_calls,
            current_common_calls,
            observation_calls,
            operational_setter_calls,
            resample_calls,
            schedule_calls,
            due_update_calls,
            due_sample_calls,
        ):
            self.assertTrue({"item", "tolist", "bool", "nonzero"}.isdisjoint(calls))

        real_profile_function = next(
            node
            for node in ast.walk(smoke_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_profile_real_bound_compute"
        )
        profile_attributes = {
            node.attr
            for node in ast.walk(real_profile_function)
            if isinstance(node, ast.Attribute)
        }
        self.assertTrue({"CPU", "CUDA", "ProfilerActivity"}.issubset(profile_attributes))
        self.assertTrue(
            any(
                isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id == "TorchDispatchMode"
                    for base in node.bases
                )
                for node in ast.walk(real_profile_function)
            )
        )
        operational_switches = [
            node
            for node in ast.walk(real_profile_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_operational_enabled"
        ]
        enable_call = next(
            node
            for node in operational_switches
            if node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is True
        )
        disable_call = next(
            node
            for node in operational_switches
            if node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is False
        )
        bound_compute_call = next(
            node
            for node in ast.walk(real_profile_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute"
        )
        self.assertLess(enable_call.lineno, bound_compute_call.lineno)
        self.assertLess(bound_compute_call.lineno, disable_call.lineno)

        main_function = next(
            node
            for node in smoke_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        disabled_baseline_loop = next(
            node
            for node in ast.walk(main_function)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and any(
                isinstance(child, ast.Attribute) and child.attr == "steps"
                for child in ast.walk(node.iter)
            )
        )
        real_profile_call = next(
            node
            for node in ast.walk(main_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_profile_real_bound_compute"
        )
        self.assertLess(disabled_baseline_loop.end_lineno, real_profile_call.lineno)
        self.assertTrue(
            any(
                isinstance(node, ast.If)
                and ast.unparse(node.test) == "not args.enabled"
                and real_profile_call in set(ast.walk(node))
                for node in ast.walk(main_function)
            )
        )

        for tree in (command_tree, operational_tree, sampling_tree):
            self.assertFalse(
                any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "nonzero"
                    for node in ast.walk(tree)
                )
            )

        def dotted_attribute(node: ast.AST) -> str | None:
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if not isinstance(node, ast.Name):
                return None
            parts.append(node.id)
            return ".".join(reversed(parts))

        assigned_attributes = {
            name
            for node in ast.walk(smoke_tree)
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign)
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if (name := dotted_attribute(target)) is not None
        }
        self.assertNotIn("command.cfg.enabled", assigned_attributes)

    def _run_prevalidated_without_scalar_extraction(self, device: torch.device) -> None:
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except ImportError:
            self.skipTest("TorchDispatchMode unavailable")

        seen_local_scalar = []

        class RejectLocalScalar(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                del types
                if "_local_scalar_dense" in str(func):
                    seen_local_scalar.append(str(func))
                    raise AssertionError(f"host scalar extraction: {func}")
                return func(*args, **(kwargs or {}))

        state = _make_state(
            device=device,
            num_envs=1,
            num_future=2,
            site_names=("wrist",),
        )
        state.reset(torch.zeros(1, 2, 1, 3, device=device))
        state.set_samples(
            None,
            enabled=torch.tensor([True], device=device),
            site_mask=torch.tensor([[True]], device=device),
            compliance=torch.tensor([[0.1]], device=device),
            force_on_robot_w=torch.tensor([[[1.0, 0.0, 0.0]]], device=device),
        )
        reference_position = torch.zeros(1, 2, 3, 3, device=device)
        reference_quaternion = torch.zeros(1, 2, 3, 4, device=device)
        reference_quaternion[..., 0] = 1.0
        articulation_position = torch.zeros(1, 3, 3, device=device)
        articulation_quaternion = torch.zeros(1, 3, 4, device=device)
        articulation_quaternion[..., 0] = 1.0

        class FakeComposer:
            def set_forces_and_torques(self, **kwargs) -> None:
                self.last = kwargs

        class FakeArticulation:
            def __init__(self) -> None:
                self.permanent_wrench_composer = FakeComposer()

        adapter = ArticulationWrenchAdapter(
            FakeArticulation(),
            body_selection=state.sites.articulation,
            num_envs=1,
            device=device,
            dtype=torch.float32,
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as hot_path_profile:
            with RejectLocalScalar():
                build_sonic_compliance_targets_prevalidated(
                    reference_positions_w=reference_position,
                    reference_quaternions_wxyz=reference_quaternion,
                    articulation_positions_w=articulation_position,
                    articulation_quaternions_wxyz=articulation_quaternion,
                    anchor_position_w=None,
                    anchor_quaternion_wxyz=None,
                    state=state,
                )
                adapter.set_world_forces_prevalidated(
                    state.force_on_robot_w,
                    body_quaternions_wxyz=torch.tensor(
                        [[[1.0, 0.0, 0.0, 0.0]]],
                        device=device,
                    ),
                    application_offsets_local=torch.zeros(1, 1, 3, device=device),
                )
                state.update_damper_prevalidated(
                    torch.zeros(1, 2, 1, 3, device=device)
                )
        self.assertEqual(seen_local_scalar, [])
        profile_keys = {event.key for event in hot_path_profile.key_averages()}
        self.assertFalse(
            any("_local_scalar_dense" in key for key in profile_keys),
            profile_keys,
        )

    def _run_bound_compute_without_dynamic_indices(self, device: torch.device) -> None:
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except ImportError:
            self.skipTest("TorchDispatchMode unavailable")

        seen_forbidden = []

        class RejectDynamicCudaSync(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                del types
                function_name = str(func)
                if "_local_scalar_dense" in function_name or "nonzero" in function_name:
                    seen_forbidden.append(function_name)
                    raise AssertionError(f"dynamic CUDA sync operation: {func}")
                return func(*args, **(kwargs or {}))

        num_envs = 4096
        site_names = tuple(f"site_{index}" for index in range(14))
        state = _make_state(
            device=device,
            num_envs=num_envs,
            num_future=2,
            site_names=site_names,
        )
        application_positions_w = torch.zeros(num_envs, 14, 3, device=device)
        application_positions_w[:, :, 0] = torch.linspace(
            -0.65,
            0.65,
            14,
            device=device,
        )
        current_eef_common = application_positions_w.unsqueeze(1).expand(
            num_envs,
            2,
            14,
            3,
        ).clone()
        state.reset(current_eef_common)

        body_positions_w = torch.zeros(num_envs, 16, 3, device=device)
        body_quaternions_wxyz = torch.zeros(num_envs, 14, 4, device=device)
        body_quaternions_wxyz[..., 0] = 1.0
        application_offsets_local = torch.zeros(num_envs, 14, 3, device=device)

        class FakeComposer:
            def set_forces_and_torques(self, **kwargs) -> None:
                self.last = kwargs

        class FakeArticulation:
            def __init__(self) -> None:
                self.permanent_wrench_composer = FakeComposer()

        fake_articulation = FakeArticulation()
        wrench = ArticulationWrenchAdapter(
            fake_articulation,
            body_selection=state.sites.articulation,
            num_envs=num_envs,
            device=device,
            dtype=torch.float32,
        )

        runtime = ComplianceOperationalControl()
        runtime.state = state
        runtime.sites = state.sites
        runtime.cfg = SimpleNamespace(
            enabled_probability=1.0,
            site_probability=1.0,
            force_magnitude_range_n=(0.0, 40.0),
            force_duration_range_s=(1.0, 3.0),
            pulse_interval_range_s=(3.5, 6.0),
            max_active_sites=3,
            max_net_force_n=30.0,
            max_net_torque_nm=20.0,
            target_damper_enabled=True,
            force_rise_end=0.2,
            force_fall_start=0.8,
        )
        runtime.robot = SimpleNamespace(
            data=SimpleNamespace(body_pos_w=body_positions_w),
        )
        runtime.anchor_body_index = 15
        runtime.current_site_positions_w = lambda: application_positions_w
        runtime.current_eef_common_future = lambda positions=None: current_eef_common
        runtime.current_site_quaternions_wxyz = lambda: body_quaternions_wxyz
        runtime._application_offsets_local = application_offsets_local
        runtime.wrench = wrench
        runtime._wrench_write_gate = WrenchWriteGate()
        runtime._update_metrics = lambda: None
        runtime._operational_enabled = False
        runtime._operational_enabled_last_update = True
        runtime._sampling_generator = torch.Generator(device=device).manual_seed(937)
        runtime._compliance_values_m_per_n = torch.tensor(
            [0.0, 0.02, 0.05],
            device=device,
        )
        runtime._time_to_next_pulse = torch.zeros(num_envs, device=device)
        runtime._all_env_ids = torch.arange(num_envs, device=device)

        disabled_private_rng = runtime._sampling_generator.get_state().clone()
        disabled_cpu_rng = torch.random.get_rng_state().clone()
        disabled_cuda_rng = (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        )
        with RejectDynamicCudaSync():
            runtime.compute(0.02)
        self.assertEqual(seen_forbidden, [])
        self.assertTrue(torch.isinf(runtime._time_to_next_pulse).all())
        self.assertTrue(
            torch.equal(runtime._sampling_generator.get_state(), disabled_private_rng)
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), disabled_cpu_rng))
        if disabled_cuda_rng is not None:
            self.assertTrue(
                torch.equal(torch.cuda.get_rng_state(device), disabled_cuda_rng)
            )

        runtime._operational_enabled = True
        runtime._operational_enabled_last_update = True
        runtime._time_to_next_pulse.zero_()
        private_rng_before = runtime._sampling_generator.get_state().clone()
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as hot_path_profile:
            with RejectDynamicCudaSync():
                runtime.compute(0.02)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        self.assertTrue(state.pulse_active.all())
        torch.testing.assert_close(
            state.site_mask.sum(dim=-1),
            torch.full((num_envs,), 3, device=device),
        )
        self.assertTrue(torch.isfinite(runtime._time_to_next_pulse).all())
        self.assertTrue((runtime._time_to_next_pulse >= 3.5).all())
        self.assertTrue((runtime._time_to_next_pulse <= 6.0).all())
        self.assertFalse(
            torch.equal(runtime._sampling_generator.get_state(), private_rng_before)
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_rng_before))
        if cuda_rng_before is not None:
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before))
        self.assertTrue(runtime._wrench_write_gate.was_written)
        self.assertEqual(
            tuple(fake_articulation.permanent_wrench_composer.last["forces"].shape),
            (num_envs, 14, 3),
        )
        self.assertEqual(seen_forbidden, [])
        profile_keys = {event.key for event in hot_path_profile.key_averages()}
        forbidden_events = [
            (
                event.name,
                event.cpu_parent.name if event.cpu_parent is not None else None,
                (
                    event.cpu_parent.cpu_parent.name
                    if event.cpu_parent is not None
                    and event.cpu_parent.cpu_parent is not None
                    else None
                ),
            )
            for event in hot_path_profile.events()
            if "_local_scalar_dense" in event.name or "nonzero" in event.name
        ]
        self.assertFalse(
            any(
                "_local_scalar_dense" in key or "nonzero" in key
                for key in profile_keys
            ),
            (profile_keys, forbidden_events),
        )

    def test_cpu_prevalidated_hot_path_does_not_extract_scalars(self) -> None:
        self._run_prevalidated_without_scalar_extraction(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_prevalidated_hot_path_does_not_extract_scalars(self) -> None:
        self._run_prevalidated_without_scalar_extraction(torch.device("cuda:0"))

    def test_cpu_bound_compute_has_no_dynamic_indices_or_scalar_extraction(self) -> None:
        self._run_bound_compute_without_dynamic_indices(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bound_compute_has_no_dynamic_indices_or_scalar_extraction(self) -> None:
        self._run_bound_compute_without_dynamic_indices(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
