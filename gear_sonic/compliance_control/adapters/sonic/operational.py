"""Portable runtime gate shared by the Isaac Lab compliance command and tests."""

from __future__ import annotations

import torch

from .sampling import (
    advance_pulse_countdown_prevalidated,
    limit_peak_forces_by_net_wrench_prevalidated,
    mask_requested_peak_forces_prevalidated,
    reschedule_pulse_countdown_mask_prevalidated,
    reschedule_pulse_countdown_prevalidated,
    sample_compliance_pulses_prevalidated,
)


class ComplianceOperationalControl:
    """Mixin implementing immediate, command-owned compliance mode switches."""

    is_evaluating: bool = False

    def set_is_evaluating(self, is_evaluating: bool = True) -> None:
        """Honor the manager-wrapper lifecycle without changing force ownership."""

        if type(is_evaluating) is not bool:
            raise TypeError("is_evaluating must be a bool")
        self.is_evaluating = is_evaluating

    def compute(self, dt: float) -> None:
        """Override ``CommandTerm.compute`` without dynamic CUDA due indices."""

        self._update_metrics()
        self._update_command_prevalidated(dt)

    @property
    def time_to_next_pulse(self) -> torch.Tensor:
        """Return a non-aliasing snapshot of the private pulse countdown."""

        return self._time_to_next_pulse.clone()

    @property
    def operational_enabled(self) -> bool:
        """Return the runtime gate without mutating the static Hydra config."""

        return self._operational_enabled

    def set_operational_enabled(self, enabled: bool) -> None:
        """Switch compliance immediately while preserving unrelated composer rows."""

        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        was_enabled = self._operational_enabled
        self._operational_enabled = enabled
        if enabled:
            if not was_enabled:
                self._schedule_next_pulse(self._all_env_ids)
            self._operational_enabled_last_update = True
            return

        self.state.cancel_all_prevalidated()
        advance_pulse_countdown_prevalidated(
            self._time_to_next_pulse,
            self._env.step_dt,
            globally_enabled=False,
        )
        self._operational_enabled_last_update = False
        if self._wrench_write_gate.consume_clear_on_disable():
            self.wrench.clear()

    def _schedule_next_pulse(self, env_ids: torch.Tensor) -> None:
        """Reset selected countdowns without touching the process-global RNG."""

        reschedule_pulse_countdown_prevalidated(
            self._time_to_next_pulse,
            env_ids,
            globally_enabled=self._operational_enabled,
            interval_range_s=self.cfg.pulse_interval_range_s,
            generator=self._sampling_generator,
        )

    def _sample_and_start_masked_prevalidated(
        self,
        start_mask: torch.Tensor,
        *,
        application_positions_w: torch.Tensor,
        wrench_origin_w: torch.Tensor,
        current_eef_common: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample fixed-size candidates and start only trusted masked rows."""

        samples = sample_compliance_pulses_prevalidated(
            num_envs=self.state.num_envs,
            num_sites=self.sites.spec.num_sites,
            device=self.state.device,
            dtype=self.state.dtype,
            generator=self._sampling_generator,
            globally_enabled=self._operational_enabled,
            enabled_probability=self.cfg.enabled_probability,
            site_probability=self.cfg.site_probability,
            force_magnitude_range_n=self.cfg.force_magnitude_range_n,
            compliance_values_m_per_n=self._compliance_values_m_per_n,
            duration_range_s=self.cfg.force_duration_range_s,
            max_active_sites=self.cfg.max_active_sites,
        )
        candidate_enabled = samples.enabled & start_mask
        candidate_site_mask = samples.site_mask & start_mask.unsqueeze(-1)
        requested_peak_force = mask_requested_peak_forces_prevalidated(
            samples.peak_force_on_robot_w,
            candidate_enabled,
            candidate_site_mask,
        )
        peak_force = limit_peak_forces_by_net_wrench_prevalidated(
            requested_peak_force,
            application_positions_w,
            wrench_origin_w,
            max_net_force_n=self.cfg.max_net_force_n,
            max_net_torque_nm=self.cfg.max_net_torque_nm,
        )
        if self.cfg.target_damper_enabled:
            if current_eef_common is None:
                raise RuntimeError("target damper requires current_eef_common")
            self.state.seed_damper_sites_masked_prevalidated(
                current_eef_common,
                candidate_site_mask,
            )
        compliance = samples.compliance.unsqueeze(-1).expand(
            self.state.num_envs,
            self.sites.spec.num_sites,
            3,
        )
        self.state.start_pulses_masked_prevalidated(
            start_mask,
            enabled=samples.enabled,
            site_mask=samples.site_mask,
            compliance=compliance,
            peak_force_on_robot_w=peak_force,
            duration_s=samples.duration_s,
        )
        return candidate_enabled & candidate_site_mask.any(dim=-1)

    def _update_due_pulses_prevalidated(
        self,
        dt_s: float,
        *,
        application_positions_w: torch.Tensor,
        wrench_origin_w: torch.Tensor,
        current_eef_common: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance/sample/start fixed-size masks and privately reschedule."""

        due_mask = advance_pulse_countdown_prevalidated(
            self._time_to_next_pulse,
            dt_s,
            globally_enabled=True,
        )
        start_mask = self.state.startable_pulse_mask_prevalidated(due_mask)
        started_mask = self._sample_and_start_masked_prevalidated(
            start_mask,
            application_positions_w=application_positions_w,
            wrench_origin_w=wrench_origin_w,
            current_eef_common=current_eef_common,
        )
        reschedule_pulse_countdown_mask_prevalidated(
            self._time_to_next_pulse,
            due_mask,
            interval_range_s=self.cfg.pulse_interval_range_s,
            generator=self._sampling_generator,
        )
        return due_mask, started_mask

    def _update_command_prevalidated(self, dt_s: float) -> None:
        """Run the full fixed-shape command and writer path for one control step."""

        if not self._operational_enabled:
            self.state.cancel_all_prevalidated()
            advance_pulse_countdown_prevalidated(
                self._time_to_next_pulse,
                dt_s,
                globally_enabled=False,
            )
            self._operational_enabled_last_update = False
            if self._wrench_write_gate.consume_clear_on_disable():
                self.wrench.clear()
            return

        if not self._operational_enabled_last_update:
            self._schedule_next_pulse(self._all_env_ids)
        self._operational_enabled_last_update = True
        application_positions_w = self.current_site_positions_w()
        wrench_origin_w = self.robot.data.body_pos_w[:, self.anchor_body_index]
        current_eef_common = (
            self.current_eef_common_future(application_positions_w)
            if self.cfg.target_damper_enabled
            else None
        )
        self._update_due_pulses_prevalidated(
            dt_s,
            application_positions_w=application_positions_w,
            wrench_origin_w=wrench_origin_w,
            current_eef_common=current_eef_common,
        )

        scheduled_force_on_robot_w = self.state.advance_force_schedule(
            dt_s,
            rise_end=self.cfg.force_rise_end,
            fall_start=self.cfg.force_fall_start,
        )
        applied_force_on_robot_w = limit_peak_forces_by_net_wrench_prevalidated(
            scheduled_force_on_robot_w,
            application_positions_w,
            wrench_origin_w,
            max_net_force_n=self.cfg.max_net_force_n,
            max_net_torque_nm=self.cfg.max_net_torque_nm,
        )
        self.state.set_applied_force_prevalidated(applied_force_on_robot_w)
        self.wrench.set_world_forces_prevalidated(
            applied_force_on_robot_w,
            body_quaternions_wxyz=self.current_site_quaternions_wxyz(),
            application_offsets_local=self._application_offsets_local,
        )
        self._wrench_write_gate.mark_written()
        if self.cfg.target_damper_enabled:
            assert current_eef_common is not None
            if not self.state.damper_initialized:
                self.state.reset_damper_prevalidated(current_eef_common)
            self.state.update_damper_prevalidated(current_eef_common)
