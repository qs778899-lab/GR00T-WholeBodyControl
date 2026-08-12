#!/usr/bin/env python3
"""Bind real SONIC collection evidence to a tracker-neutral paired report."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import stat
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gear_sonic.compliance_control.adapters.sonic.evaluation import (  # noqa: E402
    SONIC_ACTION_RESIDUAL_PREFIX,
    SONIC_EVALUATION_MANAGER_PROVENANCE,
    SONIC_EVALUATION_RESET_EVENT,
    SONIC_EVALUATION_TERMINATION_NAMES,
    SONIC_RELEASE_CHECKPOINT_SHA256,
    SONIC_RELEASE_CHECKPOINT_STEP,
    SONIC_RELEASE_TRACKING_BODY_NAMES,
    SONIC_TRAINED_CHECKPOINT_SHA256,
    SONIC_TRAINED_CHECKPOINT_STEP,
    validate_policy_action_byte_parity,
)
from gear_sonic.compliance_control.evaluation import (  # noqa: E402
    EvaluationTrace,
    RegressionCriteria,
    TrialMode,
    TrialSpec,
    evaluate_trial_suite,
    load_trace_npz_with_sha256,
    write_report_json_atomic,
)


SONIC_AUDITED_MOTION_SHA256 = (
    "005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b"
)
SONIC_PHASE6_PORTABLE_CRITERIA = {
    "endpoint_site_ids": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    "endpoint_tracking_point_ids": [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ],
    "max_success_rate_drop": 0.01,
    "local_mpjpe_absolute_regression_m": 0.003,
    "local_mpjpe_relative_regression": 0.10,
    "endpoint_rmse_regression_m": 0.005,
    "no_contact_endpoint_delta_m": 0.005,
    "reset_wrench_tolerance_n": 1.0e-6,
    "inactive_force_tolerance_n": 1.0e-6,
    "inactive_yield_tolerance_m": 1.0e-7,
    "minimum_active_force_peak_n": 5.0,
    "minimum_active_yield_peak_m": 0.049,
    "minimum_active_measured_yield_peak_m": 0.0005,
    "minimum_active_measured_yield_along_force_peak_m": 0.0001,
    "inactive_cross_coupling_rmse_m": 0.005,
    "inactive_cross_coupling_p95_m": 0.010,
    "active_selected_endpoint_rmse_regression_m": 0.005,
    "active_selected_endpoint_p95_regression_m": 0.010,
    "active_orientation_rmse_regression_rad": 0.05,
    "active_orientation_p95_regression_rad": 0.10,
    "active_invariant_local_mpjpe_absolute_regression_m": 0.003,
    "active_invariant_local_mpjpe_relative_regression": 0.10,
    "active_invariant_global_mpjpe_absolute_regression_m": 0.005,
    "active_invariant_global_mpjpe_relative_regression": 0.10,
}
SONIC_PHASE6_ENVIRONMENT_INVARIANTS = {
    "hydra_termination_override": "/manager_env/terminations=tracking/eval",
    "event_names": ["motion_compliance_reset"],
    "terrain_type": "plane",
    "force_flat_terrain": True,
    "robot_motion_encoder": "g1",
    "encoder_sample_probs": {"g1": 1.0, "teleop": 0.0, "smpl": 0.0},
    "cat_upper_body_poses": False,
    "freeze_frame_aug": False,
    "teleop_sample_prob_when_smpl": 0.0,
}
_ENVIRONMENT_GATE_KEYS = {"host_operational_enabled", "logical_condition_enabled"}


def _read_regular_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"evidence path must be a regular non-symlink file: {path}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_size > max_bytes:
            raise ValueError(f"evidence file is not regular or exceeds its bound: {path}")
        payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"evidence file exceeds its bound: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _read_regular_json(path: Path, *, max_bytes: int) -> tuple[dict[str, object], str]:
    payload, digest = _read_regular_bytes(path, max_bytes=max_bytes)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evidence JSON root must be an object: {path}")
    return value, digest


def _typed_equal(value: object, expected: object) -> bool:
    """Compare JSON-like evidence without Python's bool/int equality aliasing."""

    if isinstance(expected, Mapping):
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(_typed_equal(value[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _typed_equal(observed_item, expected_item)
                for observed_item, expected_item in zip(value, expected, strict=True)
            )
        )
    return type(value) is type(expected) and value == expected


def _eq_check(name: str, value: object, expected: object) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "limit": expected,
        "passed": _typed_equal(value, expected),
    }


def _upper_check(name: str, value: object, limit: float) -> dict[str, object]:
    valid = (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return {
        "name": name,
        "value": value,
        "limit": limit,
        "passed": bool(valid and float(value) <= limit),
    }


def _strict_positive_check(name: str, value: object) -> dict[str, object]:
    valid = (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )
    return {
        "name": name,
        "value": value,
        "limit": "finite and > 0",
        "passed": bool(valid),
    }


def _finite_nonnegative_check(name: str, value: object) -> dict[str, object]:
    valid = (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
    return {
        "name": name,
        "value": value,
        "limit": "finite and >= 0",
        "passed": bool(valid),
    }


def _bounded_positive_int_check(
    name: str,
    value: object,
    upper: object,
) -> dict[str, object]:
    valid = (
        type(value) is int
        and type(upper) is int
        and 0 < value <= upper
    )
    return {
        "name": name,
        "value": value,
        "limit": f"integer in [1, {upper}]" if type(upper) is int else None,
        "passed": valid,
    }


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sonic_collection_suite(
    paired_report: Mapping[str, object],
    collection_reports: Mapping[str, Mapping[str, object]],
    *,
    paired_report_sha256: str,
    collection_report_sha256: Mapping[str, str],
    observed_trace_sha256: Mapping[str, str],
    observed_traces: Mapping[str, EvaluationTrace],
    reset_owned_force_tolerance_n: float = 1.0e-6,
    reset_owned_torque_tolerance_nm: float = 1.0e-6,
    owned_wrench_buffer_difference_tolerance: float = 1.0e-6,
) -> dict[str, object]:
    """Return a fail-closed SONIC-specific acceptance layer over portable metrics."""

    trial_order = paired_report.get("trial_order")
    trial_specs = paired_report.get("trial_specs")
    if not isinstance(trial_order, list) or not isinstance(trial_specs, list):
        raise ValueError("paired report lacks trial order/specs")
    if len(trial_order) != 6 or len(trial_specs) != 6:
        raise ValueError("SONIC Phase-6 requires exactly six paired trials")
    if set(trial_order) != set(collection_reports):
        raise ValueError("collection report names must match paired trial names")
    if (
        set(collection_report_sha256) != set(trial_order)
        or set(observed_trace_sha256) != set(trial_order)
        or set(observed_traces) != set(trial_order)
    ):
        raise ValueError("input hash names must match paired trial names")
    specs: dict[str, Mapping[str, object]] = {}
    for raw_spec in trial_specs:
        spec = _mapping(raw_spec, "trial spec")
        name = spec.get("name")
        if not isinstance(name, str) or name in specs:
            raise ValueError("trial specs must have unique string names")
        expected_active = spec.get("expected_active_site_ids")
        if (
            not isinstance(expected_active, list)
            or any(not isinstance(site, str) or not site for site in expected_active)
            or len(set(expected_active)) != len(expected_active)
        ):
            raise ValueError("expected_active_site_ids must be a unique string list")
        specs[name] = spec
    if set(specs) != set(trial_order):
        raise ValueError("trial order and specs differ")
    mode_counts = Counter(spec.get("mode") for spec in specs.values())
    if mode_counts != Counter(
        {"baseline": 1, "off": 1, "no_contact": 1, "single_site": 2, "multi_site": 1}
    ):
        raise ValueError("SONIC Phase-6 protocol multiplicities changed")
    specs_by_mode: dict[str, list[Mapping[str, object]]] = {}
    for spec in specs.values():
        specs_by_mode.setdefault(str(spec.get("mode")), []).append(spec)
    for mode in ("baseline", "off", "no_contact"):
        if specs_by_mode[mode][0].get("expected_active_site_ids") != []:
            raise ValueError(f"{mode} trial must have no expected active site")
    single_site_layouts = {
        tuple(spec.get("expected_active_site_ids", ()))
        for spec in specs_by_mode["single_site"]
    }
    if single_site_layouts != {
        ("left_wrist_yaw_link",),
        ("right_wrist_yaw_link",),
    }:
        raise ValueError("single-site trials must cover each wrist exactly once")
    if specs_by_mode["multi_site"][0].get("expected_active_site_ids") != [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]:
        raise ValueError("multi-site trial must use both ordered wrist sites")
    baseline_name = paired_report.get("baseline_trial")
    off_name = paired_report.get("off_trial")
    if not isinstance(baseline_name, str) or not isinstance(off_name, str):
        raise ValueError("paired report lacks baseline/off names")
    if specs.get(baseline_name, {}).get("mode") != "baseline":
        raise ValueError("baseline_trial must identify the baseline-mode trial")
    if specs.get(off_name, {}).get("mode") != "off":
        raise ValueError("off_trial must identify the off-mode trial")

    recomputed_specs = []
    for name in trial_order:
        spec = specs[name]
        active_ids = spec.get("expected_active_site_ids")
        if not isinstance(active_ids, list):
            raise ValueError("expected_active_site_ids must be a list")
        recomputed_specs.append(
            TrialSpec(
                name=name,
                mode=TrialMode(spec.get("mode")),
                expected_active_site_ids=tuple(active_ids),
            )
        )
    recomputed_report = evaluate_trial_suite(
        observed_traces,
        recomputed_specs,
        baseline_name=baseline_name,
        criteria=RegressionCriteria(
            endpoint_site_ids=tuple(
                SONIC_PHASE6_PORTABLE_CRITERIA["endpoint_site_ids"]
            ),
            endpoint_tracking_point_ids=tuple(
                SONIC_PHASE6_PORTABLE_CRITERIA["endpoint_tracking_point_ids"]
            ),
            max_success_rate_drop=SONIC_PHASE6_PORTABLE_CRITERIA[
                "max_success_rate_drop"
            ],
            local_mpjpe_absolute_regression_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "local_mpjpe_absolute_regression_m"
            ],
            local_mpjpe_relative_regression=SONIC_PHASE6_PORTABLE_CRITERIA[
                "local_mpjpe_relative_regression"
            ],
            endpoint_rmse_regression_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "endpoint_rmse_regression_m"
            ],
            no_contact_endpoint_delta_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "no_contact_endpoint_delta_m"
            ],
            reset_wrench_tolerance_n=SONIC_PHASE6_PORTABLE_CRITERIA[
                "reset_wrench_tolerance_n"
            ],
            inactive_force_tolerance_n=SONIC_PHASE6_PORTABLE_CRITERIA[
                "inactive_force_tolerance_n"
            ],
            inactive_yield_tolerance_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "inactive_yield_tolerance_m"
            ],
            minimum_active_force_peak_n=SONIC_PHASE6_PORTABLE_CRITERIA[
                "minimum_active_force_peak_n"
            ],
            minimum_active_yield_peak_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "minimum_active_yield_peak_m"
            ],
            minimum_active_measured_yield_peak_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "minimum_active_measured_yield_peak_m"
            ],
            minimum_active_measured_yield_along_force_peak_m=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "minimum_active_measured_yield_along_force_peak_m"
                ]
            ),
            inactive_cross_coupling_rmse_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "inactive_cross_coupling_rmse_m"
            ],
            inactive_cross_coupling_p95_m=SONIC_PHASE6_PORTABLE_CRITERIA[
                "inactive_cross_coupling_p95_m"
            ],
            active_selected_endpoint_rmse_regression_m=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_selected_endpoint_rmse_regression_m"
                ]
            ),
            active_selected_endpoint_p95_regression_m=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_selected_endpoint_p95_regression_m"
                ]
            ),
            active_orientation_rmse_regression_rad=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_orientation_rmse_regression_rad"
                ]
            ),
            active_orientation_p95_regression_rad=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_orientation_p95_regression_rad"
                ]
            ),
            active_invariant_local_mpjpe_absolute_regression_m=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_invariant_local_mpjpe_absolute_regression_m"
                ]
            ),
            active_invariant_local_mpjpe_relative_regression=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_invariant_local_mpjpe_relative_regression"
                ]
            ),
            active_invariant_global_mpjpe_absolute_regression_m=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_invariant_global_mpjpe_absolute_regression_m"
                ]
            ),
            active_invariant_global_mpjpe_relative_regression=(
                SONIC_PHASE6_PORTABLE_CRITERIA[
                    "active_invariant_global_mpjpe_relative_regression"
                ]
            ),
        ),
    )
    paired_metrics_payload = dict(paired_report)
    paired_metrics_payload.pop("trace_sha256_by_trial", None)
    paired_metrics_sha256 = _canonical_json_sha256(paired_metrics_payload)
    recomputed_metrics_sha256 = _canonical_json_sha256(recomputed_report)

    checks: list[dict[str, object]] = []
    checks.extend(
        (
            _eq_check(
                "reset_owned_force_tolerance_pinned",
                reset_owned_force_tolerance_n,
                1.0e-6,
            ),
            _eq_check(
                "reset_owned_torque_tolerance_pinned",
                reset_owned_torque_tolerance_nm,
                1.0e-6,
            ),
            _eq_check(
                "owned_wrench_buffer_tolerance_pinned",
                owned_wrench_buffer_difference_tolerance,
                1.0e-6,
            ),
        )
    )
    checks.append(
        _eq_check(
            "portable_report_recomputed_from_bound_traces",
            paired_metrics_sha256,
            recomputed_metrics_sha256,
        )
    )
    checks.append(
        _eq_check(
            "portable_report_schema",
            paired_report.get("schema_version"),
            "compliance_evaluation_v2",
        )
    )
    paired_acceptance = _mapping(paired_report.get("acceptance"), "paired acceptance")
    portable_criteria = paired_acceptance.get("criteria")
    portable_checks = paired_acceptance.get("checks")
    portable_checks_valid = (
        isinstance(portable_checks, list)
        and bool(portable_checks)
        and all(
            isinstance(check, Mapping)
            and isinstance(check.get("name"), str)
            and check.get("passed") is True
            for check in portable_checks
        )
        and len({check["name"] for check in portable_checks}) == len(portable_checks)
    )
    checks.append(
        _eq_check(
            "portable_paired_acceptance",
            paired_acceptance.get("passed"),
            True,
        )
    )
    checks.append(
        _eq_check(
            "portable_phase6_criteria",
            portable_criteria,
            SONIC_PHASE6_PORTABLE_CRITERIA,
        )
    )
    checks.append(
        {
            "name": "portable_checks_all_passed",
            "value": len(portable_checks) if isinstance(portable_checks, list) else None,
            "limit": "non-empty, unique names, and every passed field exactly true",
            "passed": portable_checks_valid,
        }
    )
    paired_trace_hashes = _mapping(
        paired_report.get("trace_sha256_by_trial"),
        "paired trace SHA-256 mapping",
    )
    if set(paired_trace_hashes) != set(trial_order):
        raise ValueError("paired trace SHA-256 names must match the six trials")
    baseline_collection = collection_reports[baseline_name]
    common_seed = baseline_collection.get("seed")
    common_motion = baseline_collection.get("motion")
    common_coordinates = baseline_collection.get("coordinate_convention")
    paired_trials = _mapping(paired_report.get("trials"), "paired trials")
    evidence_summary: dict[str, object] = {}

    for name in trial_order:
        spec = specs[name]
        report = collection_reports[name]
        mode = spec.get("mode")
        expected_active = spec.get("expected_active_site_ids")
        checks.extend(
            (
                _eq_check(f"collection_schema:{name}", report.get("schema_version"), "sonic_phase6_collection_v3"),
                _eq_check(f"evidence_kind:{name}", report.get("evidence_kind"), "real_sonic_simulator_trace"),
                _eq_check(f"trial_name:{name}", report.get("trial_name"), name),
                _eq_check(f"protocol:{name}", report.get("protocol"), mode),
                _eq_check(f"active_site_ids:{name}", report.get("active_site_ids"), expected_active),
                _eq_check(f"seed_identity:{name}", report.get("seed"), common_seed),
                _eq_check(f"seed_pinned_zero:{name}", report.get("seed"), 0),
                _eq_check(f"motion_identity:{name}", report.get("motion"), common_motion),
                _eq_check(f"coordinate_identity:{name}", report.get("coordinate_convention"), common_coordinates),
            )
        )
        portable_trial = _mapping(paired_trials.get(name), f"portable trial {name}")
        checks.append(
            _eq_check(
                f"alignment_binding:{name}",
                report.get("alignment_sha256"),
                portable_trial.get("alignment_sha256"),
            )
        )
        checks.append(
            _eq_check(
                f"collection_vs_observed_trace_sha256:{name}",
                report.get("trace_sha256"),
                observed_trace_sha256[name],
            )
        )
        checks.append(
            _eq_check(
                f"portable_vs_observed_trace_sha256:{name}",
                paired_trace_hashes[name],
                observed_trace_sha256[name],
            )
        )

        motion = _mapping(report.get("motion"), f"motion {name}")
        executed_steps = report.get("executed_steps")
        checks.extend(
            (
                _eq_check(f"natural_timeout:{name}", report.get("natural_motion_timeout_observed"), True),
                _eq_check(f"motion_dataset_id:{name}", motion.get("dataset_motion_id"), 0),
                _eq_check(f"motion_internal_id:{name}", motion.get("internal_motion_id"), 0),
                _eq_check(f"motion_key:{name}", motion.get("key"), "walk_forward_amateur_001__A001"),
                _eq_check(f"motion_start_frame:{name}", motion.get("start_frame"), 0),
                _eq_check(f"motion_initial_time:{name}", motion.get("initial_time_step"), 0),
                _eq_check(f"motion_target_fps:{name}", motion.get("target_fps"), 50),
                _eq_check(f"full_clip_steps:{name}", executed_steps, motion.get("total_target_50hz_steps")),
                _eq_check(f"tracking_14point_layout:{name}", report.get("tracking_body_layout"), list(SONIC_RELEASE_TRACKING_BODY_NAMES)),
            )
        )
        checks.append(
            _eq_check(
                f"motion_file_sha256:{name}",
                motion.get("file_sha256"),
                SONIC_AUDITED_MOTION_SHA256,
            )
        )
        motion_hash = motion.get("file_sha256")
        expected_condition = (
            [1.0, 10.0, 200.0]
            if mode not in {"baseline", "off"}
            else [0.0, 0.0, 0.0]
        )
        protocol_parameters = _mapping(
            report.get("protocol_parameters"),
            f"protocol parameters {name}",
        )
        checks.extend(
            (
                _eq_check(
                    f"protocol_parameters:{name}",
                    protocol_parameters,
                    {
                        "force_threshold_n": 10.0,
                        "reference_offset_common_m": [0.05, 0.0, 0.0],
                        "derived_stiffness_n_per_m": 200.0,
                        "resolved_initial_condition": expected_condition,
                    },
                ),
                _eq_check(
                    f"policy_step_dt_s:{name}",
                    report.get("policy_step_dt_s"),
                    0.02,
                ),
            )
        )

        environment = _mapping(report.get("deterministic_environment"), f"environment {name}")
        invariant_environment = {
            key: value
            for key, value in environment.items()
            if key not in _ENVIRONMENT_GATE_KEYS
        }
        expected_operational = mode != "baseline"
        expected_logical = mode not in {"baseline", "off"}
        checks.extend(
            (
                _eq_check(
                    f"environment_invariants:{name}",
                    invariant_environment,
                    SONIC_PHASE6_ENVIRONMENT_INVARIANTS,
                ),
                _eq_check(f"eval_termination_override:{name}", environment.get("hydra_termination_override"), "/manager_env/terminations=tracking/eval"),
                _eq_check(f"eval_events:{name}", environment.get("event_names"), ["motion_compliance_reset"]),
                _eq_check(f"plane_terrain:{name}", environment.get("terrain_type"), "plane"),
                _eq_check(f"force_flat_terrain:{name}", environment.get("force_flat_terrain"), True),
                _eq_check(f"g1_encoder:{name}", environment.get("robot_motion_encoder"), "g1"),
                _eq_check(f"g1_encoder_probs:{name}", environment.get("encoder_sample_probs"), {"g1": 1.0, "teleop": 0.0, "smpl": 0.0}),
                _eq_check(f"host_operational_gate:{name}", environment.get("host_operational_enabled"), expected_operational),
                _eq_check(f"logical_condition_gate:{name}", environment.get("logical_condition_enabled"), expected_logical),
            )
        )
        checks.append(
            _eq_check(
                f"manager_provenance:{name}",
                report.get("manager_provenance"),
                SONIC_EVALUATION_MANAGER_PROVENANCE,
            )
        )

        termination = _mapping(report.get("termination_evidence"), f"termination {name}")
        term_counts = termination.get("term_observation_counts")
        first_terms = termination.get("first_term_step")
        expected_timeout_count = [[0, 0, 0, 1]]
        expected_first_timeout = [[-1, -1, -1, executed_steps]]
        checks.extend(
            (
                _eq_check(f"termination_schema:{name}", termination.get("schema_version"), "natural_motion_timeout_observer_v1"),
                _eq_check(f"termination_terms:{name}", termination.get("term_names"), list(SONIC_EVALUATION_TERMINATION_NAMES)),
                _eq_check(f"termination_compute_count:{name}", termination.get("compute_count"), executed_steps),
                _eq_check(f"sticky_timeout:{name}", termination.get("sticky_time_out"), [True]),
                _eq_check(f"first_timeout_final_step:{name}", termination.get("first_time_out_step"), [executed_steps]),
                _eq_check(f"timeout_count_once:{name}", term_counts, expected_timeout_count),
                _eq_check(f"timeout_term_first_final:{name}", first_terms, expected_first_timeout),
            )
        )

        expected_sha = SONIC_RELEASE_CHECKPOINT_SHA256 if mode == "baseline" else SONIC_TRAINED_CHECKPOINT_SHA256
        expected_step = SONIC_RELEASE_CHECKPOINT_STEP if mode == "baseline" else SONIC_TRAINED_CHECKPOINT_STEP
        expected_role = "official_release" if mode == "baseline" else "accepted_step6"
        load = _mapping(report.get("checkpoint_load"), f"checkpoint load {name}")
        residual_keys = load.get("expected_action_residual_keys")
        valid_residual_keys = (
            isinstance(residual_keys, list)
            and len(residual_keys) == 6
            and len(set(residual_keys)) == 6
            and all(isinstance(key, str) and key.startswith(SONIC_ACTION_RESIDUAL_PREFIX) for key in residual_keys)
        )
        expected_missing = residual_keys if mode == "baseline" and valid_residual_keys else []
        checks.extend(
            (
                _eq_check(f"checkpoint_sha256:{name}", report.get("checkpoint_sha256"), expected_sha),
                _eq_check(f"checkpoint_step:{name}", report.get("checkpoint_global_step"), expected_step),
                _eq_check(f"checkpoint_role:{name}", report.get("checkpoint_role"), expected_role),
                {"name": f"six_action_residual_keys:{name}", "value": residual_keys, "limit": "six unique residual keys", "passed": valid_residual_keys},
                _eq_check(f"checkpoint_missing_keys:{name}", load.get("missing_policy_keys"), expected_missing),
                _eq_check(f"checkpoint_unexpected_keys:{name}", load.get("unexpected_policy_keys"), []),
            )
        )

        action = report.get("policy_action_evidence")
        action_valid = isinstance(action, Mapping)
        action_error = None
        if action_valid:
            try:
                validate_policy_action_byte_parity(action, action)
            except ValueError as exc:
                action_valid = False
                action_error = str(exc)
        checks.extend(
            (
                {"name": f"action_byte_evidence_valid:{name}", "value": action_error, "limit": None, "passed": action_valid},
                _eq_check(f"action_step_count:{name}", action.get("step_count") if isinstance(action, Mapping) else None, executed_steps),
                _eq_check(f"action_shape:{name}", action.get("shape_per_step") if isinstance(action, Mapping) else None, [1, 29]),
            )
        )

        composer = _mapping(report.get("actual_composer_evidence"), f"composer {name}")
        cleanup = _mapping(report.get("post_timeout_clear_evidence"), f"cleanup {name}")
        checks.extend(
            (
                _eq_check(f"composer_source:{name}", composer.get("source"), "permanent_wrench_composer_body_local_owned_rows"),
                _upper_check(f"reset_owned_force_peak_n:{name}", composer.get("reset_owned_force_peak_n"), reset_owned_force_tolerance_n),
                _upper_check(f"reset_owned_torque_peak_nm:{name}", composer.get("reset_owned_torque_peak_nm"), reset_owned_torque_tolerance_nm),
                _upper_check(f"owned_force_buffer_difference_n:{name}", composer.get("owned_force_buffer_max_abs_difference_n"), owned_wrench_buffer_difference_tolerance),
                _upper_check(f"owned_torque_buffer_difference_nm:{name}", composer.get("owned_torque_buffer_max_abs_difference_nm"), owned_wrench_buffer_difference_tolerance),
                _eq_check(f"post_timeout_owned_force_zero:{name}", cleanup.get("owned_force_peak_n"), 0.0),
                _eq_check(f"post_timeout_owned_torque_zero:{name}", cleanup.get("owned_torque_peak_nm"), 0.0),
            )
        )
        reset_event = report.get("reset_event_evidence")
        if mode in {"single_site", "multi_site"}:
            reset_event = _mapping(reset_event, f"reset event {name}")
            pre_reset = _mapping(reset_event.get("pre_reset"), f"pre-reset wrench {name}")
            post_reset = _mapping(reset_event.get("post_reset"), f"post-reset wrench {name}")
            checks.extend(
                (
                    _eq_check(
                        f"reset_event_fields:{name}",
                        sorted(reset_event),
                        sorted(
                            (
                                "schema_version",
                                "event_name",
                                "resolved_func_target",
                                "mode",
                                "global_env_step_count",
                                "pre_reset",
                                "post_reset",
                            )
                        ),
                    ),
                    _eq_check(
                        f"pre_reset_fields:{name}",
                        sorted(pre_reset),
                        sorted(
                            (
                                "command_force_peak_n",
                                "command_torque_peak_nm",
                                "composer_force_peak_n",
                                "composer_torque_peak_nm",
                                "force_max_abs_difference_n",
                                "torque_max_abs_difference_nm",
                            )
                        ),
                    ),
                    _eq_check(
                        f"post_reset_fields:{name}",
                        sorted(post_reset),
                        sorted(
                            (
                                "command_force_peak_n",
                                "command_torque_peak_nm",
                                "composer_force_peak_n",
                                "composer_torque_peak_nm",
                                "force_max_abs_difference_n",
                                "torque_max_abs_difference_nm",
                            )
                        ),
                    ),
                    _eq_check(
                        f"reset_event_schema:{name}",
                        reset_event.get("schema_version"),
                        "sonic_phase6_reset_event_evidence_v1",
                    ),
                    _eq_check(
                        f"reset_event_name:{name}",
                        reset_event.get("event_name"),
                        SONIC_EVALUATION_RESET_EVENT,
                    ),
                    _eq_check(
                        f"reset_event_func:{name}",
                        reset_event.get("resolved_func_target"),
                        SONIC_EVALUATION_MANAGER_PROVENANCE["runtime"]["events"][0][
                            "resolved_func_target"
                        ],
                    ),
                    _eq_check(
                        f"reset_event_mode:{name}",
                        reset_event.get("mode"),
                        "reset",
                    ),
                    _bounded_positive_int_check(
                        f"reset_event_step:{name}",
                        reset_event.get("global_env_step_count"),
                        executed_steps,
                    ),
                    _strict_positive_check(
                        f"pre_reset_command_force_nonzero:{name}",
                        pre_reset.get("command_force_peak_n"),
                    ),
                    _strict_positive_check(
                        f"pre_reset_composer_force_nonzero:{name}",
                        pre_reset.get("composer_force_peak_n"),
                    ),
                    _finite_nonnegative_check(
                        f"pre_reset_command_torque_finite:{name}",
                        pre_reset.get("command_torque_peak_nm"),
                    ),
                    _finite_nonnegative_check(
                        f"pre_reset_composer_torque_finite:{name}",
                        pre_reset.get("composer_torque_peak_nm"),
                    ),
                    _upper_check(
                        f"pre_reset_force_buffer_match:{name}",
                        pre_reset.get("force_max_abs_difference_n"),
                        owned_wrench_buffer_difference_tolerance,
                    ),
                    _upper_check(
                        f"pre_reset_torque_buffer_match:{name}",
                        pre_reset.get("torque_max_abs_difference_nm"),
                        owned_wrench_buffer_difference_tolerance,
                    ),
                )
            )
            for field_name in (
                "command_force_peak_n",
                "command_torque_peak_nm",
                "composer_force_peak_n",
                "composer_torque_peak_nm",
                "force_max_abs_difference_n",
                "torque_max_abs_difference_nm",
            ):
                checks.append(
                    _eq_check(
                        f"post_reset_exact_zero_{field_name}:{name}",
                        post_reset.get(field_name),
                        0.0,
                    )
                )
        else:
            checks.append(
                _eq_check(
                    f"inactive_protocol_no_reset_event:{name}",
                    reset_event,
                    None,
                )
            )
        metrics = _mapping(report.get("metrics"), f"metrics {name}")
        lifecycle = _mapping(metrics.get("lifecycle"), f"lifecycle {name}")
        checks.extend(
            (
                _eq_check(f"trace_row_count:{name}", metrics.get("row_count"), executed_steps + 1 if isinstance(executed_steps, int) else None),
                _eq_check(f"trace_zero_falls:{name}", lifecycle.get("fall_count"), 0),
                _eq_check(f"trace_success_one:{name}", lifecycle.get("success_rate"), 1.0),
            )
        )

        evidence_summary[name] = {
            "collection_report_sha256": collection_report_sha256[name],
            "trace_sha256": observed_trace_sha256[name],
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "checkpoint_global_step": report.get("checkpoint_global_step"),
            "motion_file_sha256": motion_hash,
            "action_aggregate_sha256": (
                action.get("aggregate_sha256") if isinstance(action, Mapping) else None
            ),
            "reset_event_global_env_step_count": (
                reset_event.get("global_env_step_count")
                if isinstance(reset_event, Mapping)
                else None
            ),
            "executed_steps": executed_steps,
        }

    baseline_action = collection_reports[baseline_name].get("policy_action_evidence")
    off_action = collection_reports[off_name].get("policy_action_evidence")
    parity_error = None
    parity_passed = isinstance(baseline_action, Mapping) and isinstance(off_action, Mapping)
    if parity_passed:
        try:
            validate_policy_action_byte_parity(baseline_action, off_action)
        except ValueError as exc:
            parity_passed = False
            parity_error = str(exc)
    checks.append(
        {
            "name": "official_baseline_vs_trained_hard_off_action_byte_parity",
            "value": parity_error,
            "limit": "exact dtype/shape/per-row/aggregate parity",
            "passed": parity_passed,
        }
    )

    return {
        "schema_version": "sonic_phase6_suite_acceptance_v1",
        "paired_report_sha256": paired_report_sha256,
        "paired_metrics_payload_sha256": paired_metrics_sha256,
        "recomputed_metrics_payload_sha256": recomputed_metrics_sha256,
        "baseline_trial": baseline_name,
        "off_trial": off_name,
        "trial_order": trial_order,
        "collection_evidence": evidence_summary,
        "acceptance": {
            "passed": all(bool(check["passed"]) for check in checks),
            "criteria": {
                "reset_owned_force_tolerance_n": reset_owned_force_tolerance_n,
                "reset_owned_torque_tolerance_nm": reset_owned_torque_tolerance_nm,
                "owned_wrench_buffer_difference_tolerance": owned_wrench_buffer_difference_tolerance,
                "baseline_off_action_bytes": "exact",
                "post_timeout_owned_wrench": "exact_zero",
                "interaction_reset_event": (
                    "each single-site and multi-site trial: configured event after "
                    "nonzero force, command/composer exact zero afterward"
                ),
            },
            "checks": checks,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-report", type=Path, required=True)
    parser.add_argument(
        "--collection-report",
        action="append",
        nargs=2,
        required=True,
        metavar=("NAME", "COLLECTION_JSON"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-json-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-trace-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--reset-owned-force-tolerance-n", type=float, default=1.0e-6)
    parser.add_argument("--reset-owned-torque-tolerance-nm", type=float, default=1.0e-6)
    parser.add_argument(
        "--owned-wrench-buffer-difference-tolerance",
        type=float,
        default=1.0e-6,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paired_report, paired_sha = _read_regular_json(
        args.paired_report,
        max_bytes=args.max_json_bytes,
    )
    collections: dict[str, Mapping[str, object]] = {}
    collection_hashes: dict[str, str] = {}
    trace_hashes: dict[str, str] = {}
    traces: dict[str, EvaluationTrace] = {}
    for name, path_value in args.collection_report:
        if name in collections:
            raise ValueError(f"duplicate collection report name: {name}")
        report, report_sha = _read_regular_json(
            Path(path_value),
            max_bytes=args.max_json_bytes,
        )
        trace_path = report.get("trace")
        if not isinstance(trace_path, str) or not trace_path:
            raise ValueError(f"collection report lacks trace path: {name}")
        trace, trace_sha = load_trace_npz_with_sha256(
            Path(trace_path),
            max_bytes=args.max_trace_bytes,
        )
        collections[name] = report
        collection_hashes[name] = report_sha
        trace_hashes[name] = trace_sha
        traces[name] = trace
    final_report = validate_sonic_collection_suite(
        paired_report,
        collections,
        paired_report_sha256=paired_sha,
        collection_report_sha256=collection_hashes,
        observed_trace_sha256=trace_hashes,
        observed_traces=traces,
        reset_owned_force_tolerance_n=args.reset_owned_force_tolerance_n,
        reset_owned_torque_tolerance_nm=args.reset_owned_torque_tolerance_nm,
        owned_wrench_buffer_difference_tolerance=(
            args.owned_wrench_buffer_difference_tolerance
        ),
    )
    write_report_json_atomic(
        final_report,
        args.output,
        max_bytes=args.max_json_bytes,
        overwrite=args.overwrite,
    )
    print(
        "MOTION_COMPLIANCE_PHASE6_SONIC_ACCEPTANCE_COMPLETE "
        f"passed={str(final_report['acceptance']['passed']).lower()} "
        f"output={args.output}"
    )
    return 0 if final_report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
