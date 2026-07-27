#!/usr/bin/env python3
"""Verify one Phase-5 residual export with the pinned ONNX Runtime CPU path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from gear_sonic.compliance_control.adapters.sonic.export import (
    PHASE5_ACCEPTED_ONNXRUNTIME_VERSION,
    SonicResidualExportSpec,
    verify_sonic_actor_residual_onnx,
)
from gear_sonic.compliance_control.postprocess import write_json_new_atomic


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--expected-version",
        default=PHASE5_ACCEPTED_ONNXRUNTIME_VERSION,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = SonicResidualExportSpec(
        site_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
        num_future_frames=10,
        cartesian_dim=3,
        context_dim=930,
        output_dim=64,
    )
    report = verify_sonic_actor_residual_onnx(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        spec=spec,
        runtime="onnxruntime",
    )
    if report["runtime_version"] != args.expected_version:
        raise AssertionError(
            "ONNX Runtime version mismatch: "
            f"{report['runtime_version']} != {args.expected_version}"
        )
    if report["providers"] != ["CPUExecutionProvider"]:
        raise AssertionError(
            f"unexpected ONNX Runtime providers: {report['providers']}"
        )
    report["expected_runtime_version"] = args.expected_version
    write_json_new_atomic(args.report, report)
    print(
        "CHIP_PHASE5_ONNXRUNTIME_PARITY_PASS",
        f"version={report['runtime_version']}",
        f"max_error={report['maximum_absolute_error']:.9g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
