#!/usr/bin/env python3
"""Benchmark the fixed-shape compliance scheduler at the Phase-6 scale gate."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", default=100, type=int)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--num-envs", default=4096, type=int)
    parser.add_argument("--num-sites", default=2, type=int)
    args = parser.parse_args()
    if args.num_envs != 4096:
        raise ValueError("Phase-6 scheduler acceptance requires exactly 4096 environments")
    if args.num_sites <= 0:
        raise ValueError("--num-sites must be positive")
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    return args


def _measure_cuda(operation, *, warmup: int, iterations: int, device: torch.device):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize(device)
    return {
        "time_us_per_policy_update": start.elapsed_time(end) * 1000.0 / iterations,
        "peak_allocated_increment_bytes": (
            torch.cuda.max_memory_allocated(device) - allocated_before
        ),
        "peak_reserved_increment_bytes": (
            torch.cuda.max_memory_reserved(device) - reserved_before
        ),
    }


def _write_json_new_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path, follow_symlinks=False)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    args = _parse_args()
    output = validate_motion_compliance_run_path(args.output)
    if output.suffix != ".json":
        raise ValueError("benchmark output must use a .json suffix")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase-6 scheduler benchmark")

    device = torch.device(args.device)
    state = ComplianceCommandState(
        args.num_envs,
        args.num_sites,
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

    def host_off_policy_update() -> None:
        time_left.fill_(disabled_timer)

    def enabled_policy_update() -> None:
        candidates = state.sample_resampling_time(args.num_envs, (0.02, 0.02))
        time_left.copy_(torch.where(due_mask, candidates, time_left))
        state._resample_masked_prevalidated(due_mask)

    host_off = _measure_cuda(
        host_off_policy_update,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    enabled = _measure_cuda(
        enabled_policy_update,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    payload = {
        "schema_version": 1,
        "claim": "scheduler_only_not_end_to_end_policy_latency",
        "algorithm": "fixed_shape_all_environment_candidates",
        "device": torch.cuda.get_device_name(device),
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "num_sites": args.num_sites,
        "warmup": args.warmup,
        "host_off": host_off,
        "enabled": enabled,
        "enabled_minus_host_off_time_us": (
            enabled["time_us_per_policy_update"]
            - host_off["time_us_per_policy_update"]
        ),
    }
    _write_json_new_atomic(output, payload)
    print("MOTION_COMPLIANCE_PHASE6_SCHEDULER_PASS", json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
