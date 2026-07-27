#!/usr/bin/env python3
"""Export the accepted same-shape action residual as a standalone ONNX bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from gear_sonic.compliance_control.deployment import (
    ResidualExportSpec,
    export_action_residual_bundle,
)
from gear_sonic.compliance_control.training.checkpoint import (
    ACTION_RESIDUAL_PREFIX,
    checkpoint_sha256,
    expected_residual_shapes,
    load_trl_checkpoint,
)
from gear_sonic.compliance_control.training.paths import (
    validate_motion_compliance_run_path,
)
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER


def _global_step(checkpoint: dict[str, Any]) -> int:
    state = checkpoint.get("state")
    if isinstance(state, dict):
        value = state.get("global_step")
    else:
        value = getattr(state, "global_step", None)
    if type(value) is not int or value <= 0:
        raise ValueError("trained checkpoint lacks a positive integer global step")
    return value


def _load_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"motion_compliance_action_residual"}:
        raise ValueError("deployment overlay root schema differs")
    contract = value["motion_compliance_action_residual"]
    if not isinstance(contract, dict):
        raise ValueError("deployment overlay contract must be a mapping")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--num-sites", type=int, required=True)
    parser.add_argument("--deployment-overlay", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    output_directory = validate_motion_compliance_run_path(args.output_directory)
    actual_sha = checkpoint_sha256(args.checkpoint)
    if actual_sha != args.expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, "
            f"got {actual_sha}"
        )
    checkpoint = load_trl_checkpoint(args.checkpoint, map_location="cpu")
    if _global_step(checkpoint) != args.expected_global_step:
        raise ValueError("trained checkpoint global step differs from export contract")
    policy_keys = [
        key for key in ("policy_state_dict", "actor_model_state_dict") if key in checkpoint
    ]
    if len(policy_keys) != 1 or not isinstance(checkpoint[policy_keys[0]], dict):
        raise ValueError("checkpoint must contain exactly one policy state mapping")
    policy_key = policy_keys[0]
    policy_state = checkpoint[policy_key]
    expected_policy_shapes, _ = expected_residual_shapes(args.num_sites)
    residual_names = tuple(expected_policy_shapes)
    actual_residual_names = tuple(
        key for key in policy_state if key.startswith(ACTION_RESIDUAL_PREFIX)
    )
    if set(actual_residual_names) != set(residual_names):
        raise ValueError("checkpoint action-residual key schema differs")
    tensors: dict[str, torch.Tensor] = {}
    for name in residual_names:
        tensor = policy_state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_policy_shapes[name]
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"checkpoint action-residual tensor is incompatible: {name}")
        tensors[name] = tensor

    contract = _load_contract(args.deployment_overlay)
    if contract.get("checkpoint_sha256") != actual_sha:
        raise ValueError("deployment overlay checkpoint digest differs")
    if contract.get("checkpoint_global_step") != args.expected_global_step:
        raise ValueError("deployment overlay checkpoint step differs")
    context_layout = contract.get("context_layout")
    if not isinstance(context_layout, list) or context_layout != [
        {"name": "robot_motion_token", "width": 64},
        {"name": "actor_observation", "width": 930},
    ]:
        raise ValueError("SONIC release context layout must remain token then actor")
    token_width = context_layout[0]["width"]
    actor_width = context_layout[1]["width"]
    condition_width = contract.get("condition_width")
    if (token_width, actor_width, condition_width) != (64, 930, 3):
        raise ValueError("SONIC release context must remain 64 + 930 with condition width 3")
    if contract.get("default_enabled_condition") != [1.0, 10.0, 200.0]:
        raise ValueError("SONIC deployment condition must remain [1, 10 N, 200 N/m]")
    site_layout = tuple(contract.get("site_layout", ()))
    action_layout = tuple(contract.get("action_layout", ()))
    if len(site_layout) != args.num_sites:
        raise ValueError("deployment overlay site count differs from --num-sites")
    if action_layout != tuple(G1_ISAACLab_ORDER):
        raise ValueError(
            "SONIC residual action layout must equal the release decoder's "
            "IsaacLab/BFS output order"
        )
    spec = ResidualExportSpec(
        checkpoint_sha256=actual_sha,
        checkpoint_global_step=args.expected_global_step,
        policy_state_key=policy_key,
        residual_tensor_names=residual_names,
        release_context_width=token_width + actor_width,
        condition_width=condition_width,
        action_width=len(action_layout),
        hidden_dims=(256, 256),
        max_abs_delta=float(contract.get("max_abs_delta")),
        context_layout=(("robot_motion_token", token_width), ("actor_observation", actor_width)),
        site_layout=site_layout,
        action_layout=action_layout,
    )
    metadata = export_action_residual_bundle(tensors, output_directory, spec)
    print(
        json.dumps(
            {
                "status": "MOTION_COMPLIANCE_PHASE5_EXPORT_PASS",
                "output_directory": str(output_directory),
                "checkpoint_sha256": actual_sha,
                "checkpoint_global_step": args.expected_global_step,
                "model_sha256": metadata["model"]["sha256"],
                "metadata_sha256": metadata["metadata_sha256"],
                "input_widths": [token_width + actor_width, condition_width],
                "residual_context_width": token_width + actor_width + condition_width,
                "action_width": len(action_layout),
                "site_count": len(site_layout),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
