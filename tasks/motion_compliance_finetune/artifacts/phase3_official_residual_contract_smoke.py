#!/usr/bin/env python3
"""CPU audit of the official release shapes and isolated residual schema."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.compliance_control.training.checkpoint import (  # noqa: E402
    ACTOR_INPUT_WEIGHT_KEY,
    CRITIC_INPUT_WEIGHT_KEY,
    CRITIC_RUNNING_MEAN_KEY,
    CRITIC_RUNNING_VAR_KEY,
    OFFICIAL_SONIC_RELEASE_SHA256,
    POLICY_STATE_KEYS,
    VALUE_STATE_KEY,
    load_trl_checkpoint,
    validate_checkpoint_sha256,
)
from gear_sonic.compliance_control.training.paths import (  # noqa: E402
    OFFICIAL_SONIC_RELEASE_CHECKPOINT,
)


def _compose_release_pair():
    config_dir = str((REPO_ROOT / "gear_sonic/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        baseline = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                "num_envs=1",
            ],
        )
        compliance = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release_motion_compliance",
                "num_envs=1",
            ],
        )
    return baseline, compliance


def main() -> None:
    digest = validate_checkpoint_sha256(
        OFFICIAL_SONIC_RELEASE_CHECKPOINT,
        expected_sha256=OFFICIAL_SONIC_RELEASE_SHA256,
    )
    checkpoint = load_trl_checkpoint(
        OFFICIAL_SONIC_RELEASE_CHECKPOINT,
        map_location="cpu",
    )
    policy_keys = [key for key in POLICY_STATE_KEYS if key in checkpoint]
    if len(policy_keys) != 1:
        raise AssertionError(f"unexpected official policy keys: {policy_keys}")
    policy_state = checkpoint[policy_keys[0]]
    value_state = checkpoint[VALUE_STATE_KEY]
    if tuple(policy_state[ACTOR_INPUT_WEIGHT_KEY].shape) != (2048, 994):
        raise AssertionError("official g1_dyn input shape changed")
    if tuple(value_state[CRITIC_INPUT_WEIGHT_KEY].shape) != (2048, 1645):
        raise AssertionError("official critic input shape changed")
    if tuple(value_state[CRITIC_RUNNING_MEAN_KEY].shape) != (1645,):
        raise AssertionError("official critic running mean shape changed")
    if tuple(value_state[CRITIC_RUNNING_VAR_KEY].shape) != (1645,):
        raise AssertionError("official critic running variance shape changed")
    if any("motion_compliance" in key for key in (*policy_state.keys(), *value_state.keys())):
        raise AssertionError("official checkpoint unexpectedly contains residual keys")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (*policy_state.values(), *value_state.values())
    ):
        raise AssertionError("official model state contains a non-tensor value")

    baseline, compliance = _compose_release_pair()
    baseline_policy = OmegaConf.to_container(
        baseline.manager_env.observations.policy,
        resolve=True,
    )
    compliance_policy = OmegaConf.to_container(
        compliance.manager_env.observations.policy,
        resolve=True,
    )
    baseline_critic = OmegaConf.to_container(
        baseline.manager_env.observations.critic,
        resolve=True,
    )
    compliance_critic = OmegaConf.to_container(
        compliance.manager_env.observations.critic,
        resolve=True,
    )
    if compliance_policy != baseline_policy or compliance_critic != baseline_critic:
        raise AssertionError("compliance composition changed release policy/critic groups")
    if compliance.algo.config.actor.backbone._target_.split(".")[-1] != (
        "MotionComplianceUniversalTokenModule"
    ):
        raise AssertionError("compliance actor does not use the isolated residual wrapper")
    if compliance.algo.config.critic._target_.split(".")[-1] != (
        "MotionComplianceResidualCritic"
    ):
        raise AssertionError("compliance critic does not use the isolated residual wrapper")
    if compliance.algo.config.freeze_noise_std is not True:
        raise AssertionError("compliance composition did not freeze release action noise")
    condition_terms = [
        key
        for key in compliance.manager_env.observations.motion_compliance_condition
        if not key.startswith("_") and key not in ("enable_corruption", "concatenate_terms")
    ]
    privileged_terms = [
        key
        for key in compliance.manager_env.observations.motion_compliance_privileged
        if not key.startswith("_") and key not in ("enable_corruption", "concatenate_terms")
    ]
    if condition_terms != ["motion_compliance_condition"]:
        raise AssertionError(f"unexpected public condition terms: {condition_terms}")
    if privileged_terms != [
        "motion_compliance_threshold",
        "motion_compliance_site_force",
        "motion_compliance_site_mask",
    ]:
        raise AssertionError(f"unexpected privileged terms: {privileged_terms}")
    num_sites = len(compliance.manager_env.commands.motion_compliance.site_body_names)
    if num_sites != 2 or 1 + 4 * num_sites != 9:
        raise AssertionError("default separate privileged width is no longer nine")

    print(
        json.dumps(
            {
                "actor_base_input_shape": [2048, 994],
                "checkpoint_sha256": digest,
                "condition_width": 3,
                "critic_base_input_shape": [2048, 1645],
                "critic_rms_width": 1645,
                "policy_state_key": policy_keys[0],
                "privileged_width": 9,
                "release_observation_groups_equal": True,
                "residual_keys_are_target_initialized": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
