#!/usr/bin/env python3
"""CPU-only strict migration smoke for the pinned official SONIC checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.compliance_control.training import (
    OFFICIAL_SONIC_RELEASE_CHECKPOINT,
    audit_migrated_init_checkpoint,
    critic_added_columns,
    migrate_official_sonic_release_checkpoint_file,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
)
from gear_sonic.compliance_control.training.checkpoint import (
    ACTOR_ADDED_COLUMNS,
    ACTOR_INPUT_WEIGHT_KEY,
    CRITIC_INPUT_WEIGHT_KEY,
    CRITIC_RUNNING_MEAN_KEY,
    CRITIC_RUNNING_VAR_KEY,
    load_trl_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--num-sites", default=2, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _meta_target_state(
    source_state: dict[str, torch.Tensor],
    expansions: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    target: dict[str, torch.Tensor] = {}
    for key, tensor in source_state.items():
        if key in expansions:
            axis, added = expansions[key]
            shape = list(tensor.shape)
            shape[axis] += added
            target[key] = torch.empty(tuple(shape), dtype=tensor.dtype, device="meta")
        else:
            target[key] = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta")
    return target


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
    report_path = validate_motion_compliance_run_path(args.report)
    if output_path.suffix != ".pt" or report_path.suffix != ".json":
        raise ValueError("migration output must be .pt and report output must be .json")
    validate_distinct_artifact_paths(
        official_checkpoint=OFFICIAL_SONIC_RELEASE_CHECKPOINT,
        migration_output=output_path,
        report_output=report_path,
    )
    critic_columns = critic_added_columns(args.num_sites)

    official = load_trl_checkpoint(OFFICIAL_SONIC_RELEASE_CHECKPOINT, map_location="cpu")
    source_policy = official["policy_state_dict"]
    source_value = official["value_state_dict"]
    target_policy = _meta_target_state(
        source_policy,
        {ACTOR_INPUT_WEIGHT_KEY: (1, ACTOR_ADDED_COLUMNS)},
    )
    target_value = _meta_target_state(
        source_value,
        {
            CRITIC_INPUT_WEIGHT_KEY: (1, critic_columns),
            CRITIC_RUNNING_MEAN_KEY: (0, critic_columns),
            CRITIC_RUNNING_VAR_KEY: (0, critic_columns),
        },
    )
    del official, source_policy, source_value
    gc.collect()

    migration_report = migrate_official_sonic_release_checkpoint_file(
        OFFICIAL_SONIC_RELEASE_CHECKPOINT,
        output_path,
        num_sites=args.num_sites,
        target_policy_state=target_policy,
        target_value_state=target_value,
        overwrite=args.overwrite,
    )
    del target_policy, target_value
    gc.collect()

    migrated = load_trl_checkpoint(output_path, map_location="cpu")
    audit_migrated_init_checkpoint(migrated)
    payload = asdict(migration_report)
    _atomic_json_dump(payload, report_path)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
