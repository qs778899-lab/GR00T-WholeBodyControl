#!/usr/bin/env python3
"""Audit one real motion-compliance checkpoint and its exposure record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.compliance_control.training import (
    audit_motion_compliance_exposure_report,
    audit_trained_motion_compliance_checkpoint,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--exposure", required=True)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--num-sites", default=2, type=int)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


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
    output_path = validate_motion_compliance_run_path(args.output_json)
    if output_path.suffix != ".json":
        raise ValueError("checkpoint audit output must use a .json suffix")
    validate_distinct_artifact_paths(
        official_checkpoint=args.official,
        trained_checkpoint=args.trained,
        exposure_report=args.exposure,
        audit_output=output_path,
    )
    checkpoint_report = audit_trained_motion_compliance_checkpoint(
        args.official,
        args.trained,
        expected_global_step=args.expected_step,
        num_sites=args.num_sites,
    )
    exposure_report = audit_motion_compliance_exposure_report(
        args.exposure,
        expected_global_step=args.expected_step,
        num_sites=args.num_sites,
    )
    payload = {
        "checkpoint": asdict(checkpoint_report),
        "exposure": exposure_report,
    }
    _atomic_json_dump(payload, output_path)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
