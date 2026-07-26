#!/usr/bin/env python3
"""Measure the fixed-shape candidate scheduler used by the Phase-4 smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gear_sonic.compliance_control.adapters.sonic.state import (
    ComplianceCommandState,
    ComplianceSamplingSpec,
)
from gear_sonic.compliance_control.training import validate_motion_compliance_run_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", default=100, type=int)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--num-envs", default=16, type=int)
    args = parser.parse_args()
    if args.num_envs != 16:
        raise ValueError("Phase-4 scheduler acceptance requires exactly 16 environments")
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    return args


def _measure_cuda_loop(operation, *, warmup: int, iterations: int, device: torch.device):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    peak_increment = torch.cuda.max_memory_allocated(device) - allocated_before
    return elapsed_ms * 1000.0 / iterations, peak_increment


def _atomic_json_dump(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    args = _parse_args()
    output_path = validate_motion_compliance_run_path(args.output)
    if output_path.suffix != ".json":
        raise ValueError("scheduler benchmark output must use a .json suffix")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase-4 scheduler benchmark")
    device = torch.device(args.device)
    state = ComplianceCommandState(
        args.num_envs,
        2,
        10,
        ComplianceSamplingSpec(
            enable_probability=1.0,
            site_activation_probability=1.0,
            force_threshold_range_n=(10.0, 10.0),
            reference_displacement_m=0.05,
            reference_offset_range_m=(0.05, 0.05),
        ),
        device=device,
        seed=0,
    )
    due_mask = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    time_left = torch.ones(args.num_envs, dtype=torch.float32, device=device)
    disabled_timer = torch.finfo(time_left.dtype).max

    def host_off_operation() -> None:
        time_left.fill_(disabled_timer)

    def enabled_candidate_operation() -> None:
        candidate_time = state.sample_resampling_time(
            args.num_envs,
            (0.02, 0.02),
        )
        time_left.copy_(torch.where(due_mask, candidate_time, time_left))
        state._resample_masked_prevalidated(due_mask)

    host_off_us, host_off_peak = _measure_cuda_loop(
        host_off_operation,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    enabled_us, enabled_peak = _measure_cuda_loop(
        enabled_candidate_operation,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    payload = {
        "algorithm": "fixed_shape_all_environment_candidates",
        "device": torch.cuda.get_device_name(device),
        "enabled_candidate_peak_increment_bytes": enabled_peak,
        "enabled_candidate_time_us": enabled_us,
        "host_off_peak_increment_bytes": host_off_peak,
        "host_off_time_us": host_off_us,
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "num_sites": 2,
        "warmup": args.warmup,
    }
    _atomic_json_dump(payload, output_path)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
