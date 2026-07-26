"""Small bounded audit callback for real compliance-finetune exposure."""

from __future__ import annotations

import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import torch
from transformers import TrainerCallback

from .paths import MOTION_COMPLIANCE_RUNS_ROOT, validate_motion_compliance_run_path


class MotionComplianceExposureCallback(TrainerCallback):
    """Count sampled compliant environments/sites and nonzero force batches."""

    def __init__(
        self,
        output_path: str,
        command_name: str = "motion_compliance",
        require_exposure: bool = True,
        runs_root: str = str(MOTION_COMPLIANCE_RUNS_ROOT),
    ) -> None:
        self.output_path = validate_motion_compliance_run_path(
            output_path,
            runs_root=runs_root,
        )
        self.command_name = command_name
        self.require_exposure = require_exposure
        self.observed_batches = 0
        self.enabled_env_samples = 0
        self.active_site_samples = 0
        self.nonzero_force_batches = 0
        self.peak_site_force_n = 0.0
        self.observed_loss_logs = 0
        self.finite_loss_metric_samples = 0
        self.last_loss_metrics: dict[str, float] = {}
        self.iteration_timing_logs = 0
        self.collection_time_total_s = 0.0
        self.learn_time_total_s = 0.0
        self.min_fps: float | None = None
        self.max_fps: float | None = None
        self.process_peak_cuda_memory_allocated_bytes = 0
        self.active_site_samples_by_index: list[int] | None = None
        self.nonzero_force_site_samples_by_index: list[int] | None = None
        self._finalized = False

    def _command(self, env):
        manager_env = getattr(env, "env", env)
        command_manager = getattr(manager_env, "command_manager", None)
        if command_manager is None:
            raise RuntimeError("exposure callback requires a manager environment")
        return command_manager.get_term(self.command_name)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        loss_metrics = {
            key: value for key, value in (logs or {}).items() if key.startswith("loss/")
        }
        if not loss_metrics:
            raise RuntimeError("compliance finetuning log contained no loss/* metrics")
        finite_metrics: dict[str, float] = {}
        for key, value in loss_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise RuntimeError(f"non-scalar compliance loss metric: {key}")
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError(f"non-numeric compliance loss metric: {key}={value!r}")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise RuntimeError(f"non-finite compliance loss metric: {key}={numeric_value}")
            finite_metrics[key] = numeric_value
        self.observed_loss_logs += 1
        self.finite_loss_metric_samples += len(finite_metrics)
        self.last_loss_metrics = finite_metrics
        timing_values: dict[str, float] = {}
        for key in ("collection_time", "learn_time", "fps"):
            value = (logs or {}).get(key)
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise RuntimeError(f"non-scalar compliance timing metric: {key}")
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError(f"missing/non-numeric compliance timing metric: {key}")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise RuntimeError(f"invalid compliance timing metric: {key}={numeric_value}")
            timing_values[key] = numeric_value
        self.iteration_timing_logs += 1
        self.collection_time_total_s += timing_values["collection_time"]
        self.learn_time_total_s += timing_values["learn_time"]
        fps = timing_values["fps"]
        self.min_fps = fps if self.min_fps is None else min(self.min_fps, fps)
        self.max_fps = fps if self.max_fps is None else max(self.max_fps, fps)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ARG002
        command = self._command(kwargs["env"])
        enabled = int(torch.count_nonzero(command.state.enabled).item())
        exposed_site_mask = (
            command.state.active_site_mask & command.state.enabled.unsqueeze(-1)
        )
        active_sites = int(torch.count_nonzero(exposed_site_mask).item())
        site_force_norm = torch.linalg.vector_norm(command.state.site_force_world, dim=-1)
        peak_force = float(site_force_norm.max().item())
        active_by_index = (
            torch.count_nonzero(exposed_site_mask, dim=0).cpu().tolist()
        )
        nonzero_force_mask = site_force_norm > 0.0
        if torch.any(nonzero_force_mask & ~exposed_site_mask).item():
            raise RuntimeError("force persisted at a disabled or inactive compliance site")
        nonzero_by_index = torch.count_nonzero(
            nonzero_force_mask & exposed_site_mask,
            dim=0,
        ).cpu().tolist()
        if self.active_site_samples_by_index is None:
            self.active_site_samples_by_index = [0] * len(active_by_index)
            self.nonzero_force_site_samples_by_index = [0] * len(nonzero_by_index)
        if len(active_by_index) != len(self.active_site_samples_by_index):
            raise RuntimeError("compliance site count changed during finetuning")
        for site_index, count in enumerate(active_by_index):
            self.active_site_samples_by_index[site_index] += int(count)
        for site_index, count in enumerate(nonzero_by_index):
            self.nonzero_force_site_samples_by_index[site_index] += int(count)
        if not math.isfinite(peak_force):
            raise RuntimeError("non-finite compliance force observed during finetuning")
        force_device = command.state.site_force_world.device
        if force_device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(force_device)
            self.process_peak_cuda_memory_allocated_bytes = max(
                self.process_peak_cuda_memory_allocated_bytes,
                peak_memory,
            )
        self.observed_batches += 1
        self.enabled_env_samples += enabled
        self.active_site_samples += active_sites
        self.nonzero_force_batches += int(peak_force > 0.0)
        self.peak_site_force_n = max(self.peak_site_force_n, peak_force)
        report = self._report(state.global_step)
        self._write_report(report)
        if state.global_step >= state.max_steps:
            self._finalize(report)
        return control

    def _report(self, global_step: int) -> dict:
        return {
            "active_site_samples": self.active_site_samples,
            "active_site_samples_by_index": self.active_site_samples_by_index or [],
            "enabled_env_samples": self.enabled_env_samples,
            "finite_loss_metric_samples": self.finite_loss_metric_samples,
            "global_step": global_step,
            "iteration_collection_time_mean_s": (
                self.collection_time_total_s / self.iteration_timing_logs
                if self.iteration_timing_logs
                else 0.0
            ),
            "iteration_learn_time_mean_s": (
                self.learn_time_total_s / self.iteration_timing_logs
                if self.iteration_timing_logs
                else 0.0
            ),
            "iteration_timing_logs": self.iteration_timing_logs,
            "last_loss_metrics": self.last_loss_metrics,
            "max_fps": self.max_fps or 0.0,
            "min_fps": self.min_fps or 0.0,
            "nonzero_force_batches": self.nonzero_force_batches,
            "nonzero_force_site_samples_by_index": (
                self.nonzero_force_site_samples_by_index or []
            ),
            "observed_batches": self.observed_batches,
            "observed_loss_logs": self.observed_loss_logs,
            "peak_site_force_n": self.peak_site_force_n,
            "process_peak_cuda_memory_allocated_bytes": (
                self.process_peak_cuda_memory_allocated_bytes
            ),
        }

    def _write_report(self, report: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                json.dump(report, temporary_file, sort_keys=True)
                temporary_file.write("\n")
            os.replace(temporary_path, self.output_path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _finalize(self, report: dict) -> None:
        if self._finalized:
            return
        self._write_report(report)
        active_by_index = report["active_site_samples_by_index"]
        nonzero_by_index = report["nonzero_force_site_samples_by_index"]
        if self.require_exposure and (
            self.observed_batches == 0
            or self.observed_loss_logs == 0
            or self.finite_loss_metric_samples == 0
            or self.iteration_timing_logs == 0
            or self.observed_loss_logs != self.observed_batches
            or self.iteration_timing_logs != self.observed_batches
            or self.enabled_env_samples == 0
            or self.active_site_samples == 0
            or self.nonzero_force_batches == 0
            or not active_by_index
            or not nonzero_by_index
            or any(count == 0 for count in active_by_index)
            or any(count == 0 for count in nonzero_by_index)
        ):
            raise RuntimeError(f"compliance training had no physical exposure: {report}")
        self._finalized = True

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        report = self._report(state.global_step)
        self._finalize(report)
        return control
