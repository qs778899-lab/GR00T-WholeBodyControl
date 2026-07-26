#!/usr/bin/env python3
"""CPU-only residual initialization smoke for the pinned SONIC checkpoint."""

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
    ZeroInitializedResidualMLP,
    action_residual_context_width,
    audit_residual_init_checkpoint,
    expected_residual_shapes,
    initialize_official_sonic_release_checkpoint_file,
    tensor_bytes_equal,
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
    value_residual_context_width,
)
from gear_sonic.compliance_control.training.checkpoint import (
    ACTION_DIM,
    ACTION_RESIDUAL_PREFIX,
    OFFICIAL_POLICY_TENSOR_COUNT,
    OFFICIAL_VALUE_TENSOR_COUNT,
    VALUE_RESIDUAL_PREFIX,
    load_trl_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--num-sites", default=2, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _meta_state(source_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta")
        for key, tensor in source_state.items()
    }


def _add_residual_state(
    target: dict[str, torch.Tensor],
    residual: torch.nn.Module,
    prefix: str,
) -> None:
    target.update(
        {
            f"{prefix}{key}": value.detach().cpu().clone()
            for key, value in residual.state_dict().items()
        }
    )


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
        raise ValueError("initialization output must be .pt and report output must be .json")
    validate_distinct_artifact_paths(
        official_checkpoint=OFFICIAL_SONIC_RELEASE_CHECKPOINT,
        initialization_output=output_path,
        report_output=report_path,
    )

    official = load_trl_checkpoint(OFFICIAL_SONIC_RELEASE_CHECKPOINT, map_location="cpu")
    source_policy = official["policy_state_dict"]
    source_value = official["value_state_dict"]
    if len(source_policy) != OFFICIAL_POLICY_TENSOR_COUNT or len(
        source_value
    ) != OFFICIAL_VALUE_TENSOR_COUNT:
        raise ValueError("official policy/value tensor counts changed")
    target_policy = _meta_state(source_policy)
    target_value = _meta_state(source_value)
    _add_residual_state(
        target_policy,
        ZeroInitializedResidualMLP(action_residual_context_width(), ACTION_DIM),
        ACTION_RESIDUAL_PREFIX,
    )
    _add_residual_state(
        target_value,
        ZeroInitializedResidualMLP(value_residual_context_width(args.num_sites), 1),
        VALUE_RESIDUAL_PREFIX,
    )
    del official, source_policy, source_value
    gc.collect()

    initialization_report = initialize_official_sonic_release_checkpoint_file(
        OFFICIAL_SONIC_RELEASE_CHECKPOINT,
        output_path,
        num_sites=args.num_sites,
        target_policy_state=target_policy,
        target_value_state=target_value,
        overwrite=args.overwrite,
    )
    del target_policy, target_value
    gc.collect()

    initialized = load_trl_checkpoint(output_path, map_location="cpu")
    audit_residual_init_checkpoint(initialized)
    official = load_trl_checkpoint(OFFICIAL_SONIC_RELEASE_CHECKPOINT, map_location="cpu")
    policy_shapes, value_shapes = expected_residual_shapes(args.num_sites)
    policy_base_exact = all(
        tensor_bytes_equal(tensor, initialized["policy_state_dict"][key])
        for key, tensor in official["policy_state_dict"].items()
    )
    value_base_exact = all(
        tensor_bytes_equal(tensor, initialized["value_state_dict"][key])
        for key, tensor in official["value_state_dict"].items()
    )
    if not policy_base_exact or not value_base_exact:
        raise ValueError("initialized release base differs byte-for-byte from official")
    if set(initialized["policy_state_dict"]) != set(official["policy_state_dict"]) | set(
        policy_shapes
    ):
        raise ValueError("initialized policy schema differs")
    if set(initialized["value_state_dict"]) != set(official["value_state_dict"]) | set(
        value_shapes
    ):
        raise ValueError("initialized value schema differs")

    payload = {
        **asdict(initialization_report),
        "official_policy_base_byte_exact": policy_base_exact,
        "official_value_base_byte_exact": value_base_exact,
        "official_policy_tensor_count": len(official["policy_state_dict"]),
        "official_value_tensor_count": len(official["value_state_dict"]),
    }
    _atomic_json_dump(payload, report_path)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
