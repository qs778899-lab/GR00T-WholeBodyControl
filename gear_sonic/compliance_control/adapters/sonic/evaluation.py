"""SONIC-to-portable aligned-trace adapter and bounded lifecycle collector.

The portable :mod:`gear_sonic.compliance_control.evaluation` package owns the
trace schema and metrics.  This module is the only layer that translates
SONIC command/body fields, WXYZ quaternions, and reset semantics into that
schema.  The lifecycle collector itself consumes plain NumPy snapshots so it
can be unit-tested without launching Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import math
from numbers import Integral, Real

import numpy as np
import torch

from ...evaluation import EvaluationTrace
from .contracts import _select_yielded_site_reference_unchecked
from .frames import (
    _rotate_vectors_wxyz_unchecked,
    _world_to_body_vectors_unchecked,
    _world_to_common_positions_unchecked,
)


SONIC_RELEASE_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
SONIC_RELEASE_CHECKPOINT_STEP = 41_550
SONIC_TRAINED_CHECKPOINT_SHA256 = (
    "42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba"
)
SONIC_TRAINED_CHECKPOINT_STEP = 6
SONIC_ACTION_RESIDUAL_PREFIX = "actor_module.motion_compliance_action_residual."
SONIC_EVALUATION_RESET_EVENT = "motion_compliance_reset"
SONIC_EVALUATION_TERMINATION_NAMES = (
    "anchor_pos",
    "anchor_ori_full",
    "ee_body_pos",
    "time_out",
)
SONIC_EVALUATION_MANAGER_PROVENANCE = {
    "schema_version": "sonic_phase6_manager_provenance_v1",
    "configured": {
        "terminations": {
            "_target_": "gear_sonic.envs.manager_env.mdp.terminations.TerminationsCfg",
            "anchor_pos": {
                "_target_": "isaaclab.managers.TerminationTermCfg",
                "func": "gear_sonic.envs.manager_env.mdp:exceeded_anchor_height",
                "params": {
                    "command_name": "motion",
                    "threshold": 0.25,
                    "threshold_adaptive": False,
                    "down_threshold": 0.25,
                },
            },
            "anchor_ori_full": {
                "_target_": "isaaclab.managers.TerminationTermCfg",
                "func": "gear_sonic.envs.manager_env.mdp:exceeded_anchor_ori",
                "params": {
                    "asset_cfg": {
                        "_target_": "isaaclab.managers.SceneEntityCfg",
                        "name": "robot",
                    },
                    "command_name": "motion",
                    "threshold": 1.0,
                },
            },
            "ee_body_pos": {
                "_target_": "isaaclab.managers.TerminationTermCfg",
                "func": "gear_sonic.envs.manager_env.mdp:exceeded_body_height",
                "params": {
                    "command_name": "motion",
                    "threshold": 0.25,
                    "body_names": [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    "threshold_adaptive": False,
                    "down_threshold": 0.25,
                },
            },
            "time_out": {
                "_target_": "isaaclab.managers.TerminationTermCfg",
                "func": "gear_sonic.envs.manager_env.mdp:tracking_time_out",
                "time_out": True,
                "params": {"command_name": "motion"},
            },
        },
        "events": {
            "_target_": (
                "gear_sonic.compliance_control.adapters.sonic.manager_cfg."
                "ComplianceEventsCfg"
            ),
            "motion_compliance_reset": {
                "_target_": "isaaclab.managers.EventTermCfg",
                "func": (
                    "gear_sonic.compliance_control.adapters.sonic.mdp:"
                    "reset_compliance_wrench"
                ),
                "mode": "reset",
                "min_step_count_between_reset": 0,
                "params": {"command_name": "motion_compliance"},
            },
        },
    },
    "runtime": {
        "terminations": [
            {
                "name": "anchor_pos",
                "resolved_func_target": (
                    "gear_sonic.envs.manager_env.mdp.terminations:"
                    "exceeded_anchor_height"
                ),
                "time_out": False,
                "effective_params": {
                    "command_name": "motion",
                    "threshold": 0.25,
                    "threshold_adaptive": False,
                    "down_threshold": 0.25,
                    "root_height_threshold": 1.0,
                },
            },
            {
                "name": "anchor_ori_full",
                "resolved_func_target": (
                    "gear_sonic.envs.manager_env.mdp.terminations:"
                    "exceeded_anchor_ori"
                ),
                "time_out": False,
                "effective_params": {
                    "asset_cfg": {
                        "config_type": "isaaclab.managers.SceneEntityCfg",
                        "name": "robot",
                    },
                    "command_name": "motion",
                    "threshold": 1.0,
                },
            },
            {
                "name": "ee_body_pos",
                "resolved_func_target": (
                    "gear_sonic.envs.manager_env.mdp.terminations:"
                    "exceeded_body_height"
                ),
                "time_out": False,
                "effective_params": {
                    "command_name": "motion",
                    "threshold": 0.25,
                    "threshold_adaptive": False,
                    "down_threshold": 0.25,
                    "body_names": [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    "root_height_threshold": 0.5,
                },
            },
            {
                "name": "time_out",
                "resolved_func_target": (
                    "gear_sonic.envs.manager_env.mdp.terminations:tracking_time_out"
                ),
                "time_out": True,
                "effective_params": {"command_name": "motion"},
            },
        ],
        "events": [
            {
                "name": "motion_compliance_reset",
                "resolved_func_target": (
                    "gear_sonic.compliance_control.adapters.sonic.event:"
                    "reset_compliance_wrench"
                ),
                "mode": "reset",
                "min_step_count_between_reset": 0,
                "effective_params": {"command_name": "motion_compliance"},
            }
        ],
    },
}
SONIC_RELEASE_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


_FLOAT_FIELDS = (
    "original_site_positions_m",
    "selected_site_positions_m",
    "measured_site_positions_m",
    "original_site_orientations_xyzw",
    "measured_site_orientations_xyzw",
    "reference_points_global_m",
    "measured_points_global_m",
    "reference_points_local_m",
    "measured_points_local_m",
    "force_on_robot_n",
)

_ADAPTER_EVIDENCE_FIELDS = (
    "owned_wrench_force_peak_n",
    "owned_wrench_torque_peak_nm",
    "owned_force_buffer_max_abs_difference_n",
    "owned_torque_buffer_max_abs_difference_nm",
)


def _name_tuple(name: str, values: object, *, unique: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a scalar string")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _float_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind not in "fc" or array.dtype.kind == "c":
        raise TypeError(f"{name} must have a real floating dtype")
    return np.array(array, dtype=np.float32, copy=True)


def _bool_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind != "b":
        raise TypeError(f"{name} must have a boolean dtype")
    return np.array(array, dtype=np.bool_, copy=True)


@dataclass(frozen=True)
class SonicEvaluationSnapshot:
    """One batched SONIC simulator snapshot in the portable trace convention."""

    motion_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    point_ids: tuple[str, ...]
    original_site_positions_m: np.ndarray
    selected_site_positions_m: np.ndarray
    measured_site_positions_m: np.ndarray
    original_site_orientations_xyzw: np.ndarray
    measured_site_orientations_xyzw: np.ndarray
    reference_points_global_m: np.ndarray
    measured_points_global_m: np.ndarray
    reference_points_local_m: np.ndarray
    measured_points_local_m: np.ndarray
    force_on_robot_n: np.ndarray
    owned_wrench_force_peak_n: np.ndarray
    owned_wrench_torque_peak_nm: np.ndarray
    owned_force_buffer_max_abs_difference_n: np.ndarray
    owned_torque_buffer_max_abs_difference_nm: np.ndarray
    compliance_enabled: np.ndarray
    active_site_mask: np.ndarray

    def __post_init__(self) -> None:
        motion_ids = _name_tuple("motion_ids", self.motion_ids, unique=False)
        site_ids = _name_tuple("site_ids", self.site_ids)
        point_ids = _name_tuple("point_ids", self.point_ids)
        object.__setattr__(self, "motion_ids", motion_ids)
        object.__setattr__(self, "site_ids", site_ids)
        object.__setattr__(self, "point_ids", point_ids)
        batch_size = len(motion_ids)
        site_vector_shape = (batch_size, len(site_ids), 3)
        site_quaternion_shape = (batch_size, len(site_ids), 4)
        point_shape = (batch_size, len(point_ids), 3)
        shapes = {
            "original_site_positions_m": site_vector_shape,
            "selected_site_positions_m": site_vector_shape,
            "measured_site_positions_m": site_vector_shape,
            "original_site_orientations_xyzw": site_quaternion_shape,
            "measured_site_orientations_xyzw": site_quaternion_shape,
            "reference_points_global_m": point_shape,
            "measured_points_global_m": point_shape,
            "reference_points_local_m": point_shape,
            "measured_points_local_m": point_shape,
            "force_on_robot_n": site_vector_shape,
        }
        for field_name in _FLOAT_FIELDS:
            object.__setattr__(
                self,
                field_name,
                _float_array(field_name, getattr(self, field_name), shapes[field_name]),
            )
        object.__setattr__(
            self,
            "compliance_enabled",
            _bool_array("compliance_enabled", self.compliance_enabled, (batch_size,)),
        )
        object.__setattr__(
            self,
            "active_site_mask",
            _bool_array(
                "active_site_mask",
                self.active_site_mask,
                (batch_size, len(site_ids)),
            ),
        )
        for field_name in (
            "owned_wrench_force_peak_n",
            "owned_wrench_torque_peak_nm",
            "owned_force_buffer_max_abs_difference_n",
            "owned_torque_buffer_max_abs_difference_nm",
        ):
            object.__setattr__(
                self,
                field_name,
                _float_array(field_name, getattr(self, field_name), (batch_size,)),
            )


@dataclass(frozen=True)
class SonicEvaluationProtocol:
    """Deterministic logical command state applied only at a reset boundary."""

    enabled: bool
    operational_enabled: bool | None = None
    active_site_ids: tuple[str, ...] = ()
    force_threshold_n: float = 10.0
    reference_offset_common_m: tuple[float, float, float] = (0.05, 0.0, 0.0)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        operational_enabled = self.operational_enabled
        if operational_enabled is None:
            operational_enabled = self.enabled
        if type(operational_enabled) is not bool:
            raise TypeError("operational_enabled must be bool or None")
        if self.enabled and not operational_enabled:
            raise ValueError("logical compliance cannot be enabled while host is disabled")
        object.__setattr__(self, "operational_enabled", operational_enabled)
        active_site_ids = tuple(self.active_site_ids)
        if any(not isinstance(value, str) or not value for value in active_site_ids):
            raise ValueError("active_site_ids must contain non-empty strings")
        if len(set(active_site_ids)) != len(active_site_ids):
            raise ValueError("active_site_ids must not contain duplicates")
        if not self.enabled and active_site_ids:
            raise ValueError("disabled protocol cannot activate a site")
        object.__setattr__(self, "active_site_ids", active_site_ids)
        if (
            isinstance(self.force_threshold_n, bool)
            or not isinstance(self.force_threshold_n, (int, float))
            or not math.isfinite(float(self.force_threshold_n))
            or self.force_threshold_n <= 0.0
        ):
            raise ValueError("force_threshold_n must be finite and positive")
        offset = tuple(self.reference_offset_common_m)
        if len(offset) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in offset
        ):
            raise ValueError("reference_offset_common_m must contain three finite values")
        if active_site_ids and math.sqrt(sum(float(value) ** 2 for value in offset)) == 0.0:
            raise ValueError("active protocol requires a nonzero reference offset")
        object.__setattr__(
            self,
            "reference_offset_common_m",
            tuple(float(value) for value in offset),
        )


def _plain_json_value(value: object) -> object:
    """Normalize config/runtime values into a type-strict JSON representation."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("provenance mapping keys must be strings")
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_json_value(item) for item in value]
    if value is None or type(value) in (bool, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("provenance numbers must be finite")
        return result
    value_type = type(value)
    if (
        value_type.__name__ == "SceneEntityCfg"
        and value_type.__module__.startswith("isaaclab.managers")
    ):
        name = getattr(value, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("runtime SceneEntityCfg must name one scene entity")
        return {
            "config_type": "isaaclab.managers.SceneEntityCfg",
            "name": name,
        }
    raise TypeError(f"unsupported provenance value: {value_type.__module__}.{value_type.__name__}")


def _typed_json_equal(value: object, expected: object) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(_typed_json_equal(value[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _typed_json_equal(item, expected_item)
                for item, expected_item in zip(value, expected, strict=True)
            )
        )
    return type(value) is type(expected) and value == expected


def _resolved_callable_target(func: object) -> str:
    if not callable(func):
        raise TypeError("manager function must be callable")
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise TypeError("manager function must expose module and qualified name")
    return f"{module}:{qualname}"


def _effective_callable_params(
    func: object,
    configured_params: object,
    *,
    injected_names: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(configured_params, Mapping):
        raise TypeError("manager term params must be a mapping")
    if any(not isinstance(key, str) for key in configured_params):
        raise TypeError("manager term param names must be strings")
    signature = inspect.signature(func)
    parameters = signature.parameters
    unknown = set(configured_params) - set(parameters)
    if unknown:
        raise ValueError(f"manager term has unknown configured params: {sorted(unknown)}")
    effective: dict[str, object] = {}
    for name, parameter in parameters.items():
        if name in injected_names:
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError("Phase-6 manager functions may not use variadic parameters")
        if name in configured_params:
            value = configured_params[name]
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        else:
            raise ValueError(f"manager term lacks required parameter: {name}")
        effective[name] = _plain_json_value(value)
    return effective


def validate_sonic_evaluation_config_provenance(
    terminations_config: Mapping[str, object],
    events_config: Mapping[str, object],
) -> dict[str, object]:
    """Pin the exact composed Hydra function targets and declared parameters."""

    observed = {
        "terminations": _plain_json_value(terminations_config),
        "events": _plain_json_value(events_config),
    }
    expected = SONIC_EVALUATION_MANAGER_PROVENANCE["configured"]
    if not _typed_json_equal(observed, expected):
        raise ValueError("Phase-6 composed termination/event provenance changed")
    return deepcopy(expected)


def _termination_runtime_provenance(termination_manager) -> list[dict[str, object]]:
    names = tuple(termination_manager.active_terms)
    if names != SONIC_EVALUATION_TERMINATION_NAMES:
        raise ValueError(
            "Phase-6 requires /manager_env/terminations=tracking/eval with "
            f"terms {SONIC_EVALUATION_TERMINATION_NAMES}; got {names}"
        )
    cfgs = tuple(getattr(termination_manager, "_term_cfgs", ()))
    if len(cfgs) != len(names):
        raise ValueError("termination manager does not expose one config per term")
    result: list[dict[str, object]] = []
    for name, cfg in zip(names, cfgs, strict=True):
        result.append(
            {
                "name": name,
                "resolved_func_target": _resolved_callable_target(cfg.func),
                "time_out": _plain_json_value(cfg.time_out),
                "effective_params": _effective_callable_params(
                    cfg.func,
                    cfg.params,
                    injected_names=("env",),
                ),
            }
        )
    expected = SONIC_EVALUATION_MANAGER_PROVENANCE["runtime"]["terminations"]
    if not _typed_json_equal(result, expected):
        raise ValueError("Phase-6 runtime termination functions or parameters changed")
    return result


def _event_runtime_provenance(event_manager) -> list[dict[str, object]]:
    active_terms = _plain_json_value(event_manager.active_terms)
    if not _typed_json_equal(
        active_terms,
        {"reset": [SONIC_EVALUATION_RESET_EVENT]},
    ):
        raise ValueError("Phase-6 runtime permits only the reset-mode compliance event")
    get_term_cfg = getattr(event_manager, "get_term_cfg", None)
    if not callable(get_term_cfg):
        raise TypeError("event manager must expose get_term_cfg()")
    cfg = get_term_cfg(SONIC_EVALUATION_RESET_EVENT)
    result = [
        {
            "name": SONIC_EVALUATION_RESET_EVENT,
            "resolved_func_target": _resolved_callable_target(cfg.func),
            "mode": _plain_json_value(cfg.mode),
            "min_step_count_between_reset": _plain_json_value(
                cfg.min_step_count_between_reset
            ),
            "effective_params": _effective_callable_params(
                cfg.func,
                cfg.params,
                injected_names=("env", "env_ids"),
            ),
        }
    ]
    expected = SONIC_EVALUATION_MANAGER_PROVENANCE["runtime"]["events"]
    if not _typed_json_equal(result, expected):
        raise ValueError("Phase-6 runtime reset event function, mode, or parameters changed")
    return result


def validate_sonic_evaluation_manager_provenance(
    termination_manager,
    event_manager,
    *,
    configured_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Bind exact composed configuration to the functions active in IsaacLab."""

    expected_configured = SONIC_EVALUATION_MANAGER_PROVENANCE["configured"]
    normalized_configured = _plain_json_value(configured_provenance)
    if not _typed_json_equal(normalized_configured, expected_configured):
        raise ValueError("configured manager provenance is not the pinned Phase-6 contract")
    observed = {
        "schema_version": "sonic_phase6_manager_provenance_v1",
        "configured": normalized_configured,
        "runtime": {
            "terminations": _termination_runtime_provenance(termination_manager),
            "events": _event_runtime_provenance(event_manager),
        },
    }
    if not _typed_json_equal(observed, SONIC_EVALUATION_MANAGER_PROVENANCE):
        raise ValueError("Phase-6 manager provenance changed")
    return deepcopy(observed)


def validate_sonic_evaluation_event_names(event_names: Sequence[str]) -> None:
    """Require a deterministic eval boundary with only wrench cleanup on reset."""

    names = tuple(event_names)
    if names != (SONIC_EVALUATION_RESET_EVENT,):
        raise ValueError(
            "Phase-6 paired evaluation permits only motion_compliance_reset; "
            f"got {names}"
        )


def validate_sonic_evaluation_termination_manager(termination_manager) -> None:
    """Pin exact relaxed eval functions, effective params, and timeout flags."""

    _termination_runtime_provenance(termination_manager)


def validate_sonic_evaluation_event_manager(event_manager) -> None:
    """Pin the sole runtime event function, reset mode, and effective params."""

    _event_runtime_provenance(event_manager)


def validate_sonic_evaluation_checkpoint_role(
    *,
    protocol_role: str,
    checkpoint_sha256: str,
    global_step: int,
    missing_policy_keys: Sequence[str],
    unexpected_policy_keys: Sequence[str],
    expected_action_residual_keys: Sequence[str],
) -> None:
    """Reject mislabeled official-baseline or trained-overlay policy loads."""

    allowed_roles = {"baseline", "off", "no_contact", "single_site", "multi_site"}
    if protocol_role not in allowed_roles:
        raise ValueError(f"unsupported Phase-6 protocol role: {protocol_role}")
    expected_residual = set(expected_action_residual_keys)
    if len(expected_residual) != len(tuple(expected_action_residual_keys)):
        raise ValueError("expected action-residual keys must be unique")
    if len(expected_residual) != 6 or any(
        not key.startswith(SONIC_ACTION_RESIDUAL_PREFIX) for key in expected_residual
    ):
        raise ValueError("SONIC action residual must contain exactly six expected keys")
    missing = set(missing_policy_keys)
    unexpected = set(unexpected_policy_keys)
    if protocol_role == "baseline":
        if checkpoint_sha256 != SONIC_RELEASE_CHECKPOINT_SHA256:
            raise ValueError("baseline must load the pinned official SONIC checkpoint")
        if global_step != SONIC_RELEASE_CHECKPOINT_STEP:
            raise ValueError("baseline official checkpoint global step differs")
        if missing != expected_residual or unexpected:
            raise ValueError(
                "baseline official load may miss only the six action-residual tensors"
            )
        return
    if checkpoint_sha256 != SONIC_TRAINED_CHECKPOINT_SHA256:
        raise ValueError("overlay trial must load the accepted step-6 checkpoint")
    if global_step != SONIC_TRAINED_CHECKPOINT_STEP:
        raise ValueError("overlay checkpoint global step differs")
    if missing or unexpected:
        raise ValueError("overlay checkpoint policy load must be strict")


def policy_action_row_sha256(action: torch.Tensor) -> str:
    """Hash dtype, shape, and exact contiguous bytes for one policy action batch."""

    if not isinstance(action, torch.Tensor):
        raise TypeError("action must be a tensor")
    if action.ndim < 2 or action.shape[0] <= 0 or action.shape[-1] <= 0:
        raise ValueError("action must have non-empty batch and action dimensions")
    contiguous = action.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PolicyActionByteEvidence:
    """Accumulate per-transition action hashes without retaining large tensors."""

    def __init__(self) -> None:
        self._row_hashes: list[str] = []
        self._dtype: str | None = None
        self._shape: tuple[int, ...] | None = None

    def update(self, action: torch.Tensor) -> None:
        row_hash = policy_action_row_sha256(action)
        dtype = str(action.dtype)
        shape = tuple(action.shape)
        if self._dtype is None:
            self._dtype = dtype
            self._shape = shape
        elif dtype != self._dtype or shape != self._shape:
            raise ValueError("policy action dtype/shape changed during collection")
        self._row_hashes.append(row_hash)

    def report(self) -> dict[str, object]:
        if not self._row_hashes or self._dtype is None or self._shape is None:
            raise RuntimeError("no policy action was recorded")
        digest = hashlib.sha256()
        for row_hash in self._row_hashes:
            digest.update(bytes.fromhex(row_hash))
        return {
            "schema_version": "policy_action_bytes_v1",
            "dtype": self._dtype,
            "shape_per_step": list(self._shape),
            "step_count": len(self._row_hashes),
            "row_sha256": list(self._row_hashes),
            "aggregate_sha256": digest.hexdigest(),
        }


class NaturalMotionTimeoutObserver:
    """Observe original termination terms while returning no auto-reset signal."""

    def __init__(self, termination_manager) -> None:
        if not hasattr(termination_manager, "compute"):
            raise TypeError("termination manager must expose compute()")
        validate_sonic_evaluation_termination_manager(termination_manager)
        self.manager = termination_manager
        self._original_compute = termination_manager.compute
        self._installed = False
        self.compute_count = 0
        self.sticky_terminated = torch.zeros_like(termination_manager.terminated)
        self.sticky_time_out = torch.zeros_like(termination_manager.time_outs)
        self.first_terminated_step = torch.full(
            self.sticky_terminated.shape,
            -1,
            dtype=torch.int64,
            device=self.sticky_terminated.device,
        )
        self.first_time_out_step = torch.full_like(self.first_terminated_step, -1)
        self.term_names = tuple(termination_manager.active_terms)
        self.sticky_terms = torch.zeros_like(termination_manager._term_dones)
        self.term_observation_counts = torch.zeros_like(
            termination_manager._term_dones,
            dtype=torch.int64,
        )
        self.first_term_step = torch.full_like(self.term_observation_counts, -1)

    @staticmethod
    def _record_first_step(
        first_step: torch.Tensor,
        observed: torch.Tensor,
        step: int,
    ) -> None:
        first_step.copy_(
            torch.where(
                observed & (first_step < 0),
                torch.full_like(first_step, step),
                first_step,
            )
        )

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("natural-timeout observer is already installed")

        def compute_without_auto_reset():
            original_done = self._original_compute()
            self.compute_count += 1
            terminated = self.manager.terminated.clone()
            timed_out = self.manager.time_outs.clone()
            term_dones = self.manager._term_dones.clone()
            self._record_first_step(
                self.first_terminated_step,
                terminated,
                self.compute_count,
            )
            self._record_first_step(
                self.first_time_out_step,
                timed_out,
                self.compute_count,
            )
            self.sticky_terminated |= terminated
            self.sticky_time_out |= timed_out
            self.sticky_terms |= term_dones
            self.term_observation_counts += term_dones.to(dtype=torch.int64)
            first_term = term_dones & (self.first_term_step < 0)
            self.first_term_step.copy_(
                torch.where(
                    first_term,
                    torch.full_like(self.first_term_step, self.compute_count),
                    self.first_term_step,
                )
            )
            self.manager._terminated_buf.zero_()
            self.manager._truncated_buf.zero_()
            self.manager._term_dones.zero_()
            return torch.zeros_like(original_done)

        self.manager.compute = compute_without_auto_reset
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            self.manager.compute = self._original_compute
            self._installed = False

    def report(self) -> dict[str, object]:
        return {
            "schema_version": "natural_motion_timeout_observer_v1",
            "compute_count": self.compute_count,
            "auto_reset_suppressed": True,
            "sticky_terminated": self.sticky_terminated.detach().cpu().tolist(),
            "sticky_time_out": self.sticky_time_out.detach().cpu().tolist(),
            "first_terminated_step": self.first_terminated_step.detach().cpu().tolist(),
            "first_time_out_step": self.first_time_out_step.detach().cpu().tolist(),
            "term_names": list(self.term_names),
            "sticky_terms": self.sticky_terms.detach().cpu().tolist(),
            "term_observation_counts": (
                self.term_observation_counts.detach().cpu().tolist()
            ),
            "first_term_step": self.first_term_step.detach().cpu().tolist(),
        }

    def assert_natural_timeout_completion(self, executed_steps: int) -> None:
        """Require one first timeout on the final executed physics transition."""

        if isinstance(executed_steps, bool) or not isinstance(executed_steps, int):
            raise TypeError("executed_steps must be an integer")
        if executed_steps <= 0 or self.compute_count != executed_steps:
            raise RuntimeError("termination compute count must equal executed policy steps")
        timeout_index = self.term_names.index("time_out")
        timeout_counts = self.term_observation_counts[:, timeout_index]
        timeout_first = self.first_term_step[:, timeout_index]
        if not bool(torch.all(self.sticky_time_out).item()):
            raise RuntimeError("natural motion timeout was not observed for every stream")
        if not bool(torch.all(timeout_counts == 1).item()):
            raise RuntimeError("natural motion timeout must occur exactly once")
        if not bool(torch.all(timeout_first == executed_steps).item()):
            raise RuntimeError("natural motion timeout must first occur on the final step")


def clear_and_assert_owned_composer_wrench(command) -> dict[str, float | str]:
    """Explicitly zero command-owned composer rows after the timeout transition."""

    ids = torch.arange(command.num_envs, device=command.device, dtype=torch.long)
    command.clear_wrench(ids)
    zero_force, zero_torque, writer_ids = command.body_wrench_for_envs(ids)
    composer = getattr(command.robot, "permanent_wrench_composer", None)
    setter = getattr(composer, "set_forces_and_torques", None)
    if setter is None:
        raise RuntimeError("Phase-6 cleanup requires permanent wrench composer writes")
    setter(
        forces=zero_force,
        torques=zero_torque,
        body_ids=command.application_body_ids,
        env_ids=writer_ids,
        is_global=False,
    )
    owned_ids = command.application_body_ids
    force = composer.composed_force_as_torch[:, owned_ids]
    torque = composer.composed_torque_as_torch[:, owned_ids]
    force_peak = float(torch.linalg.vector_norm(force, dim=-1).max().item())
    torque_peak = float(torch.linalg.vector_norm(torque, dim=-1).max().item())
    if force_peak != 0.0 or torque_peak != 0.0:
        raise RuntimeError("post-timeout owned composer wrench was not cleared exactly")
    return {
        "schema_version": "sonic_post_timeout_owned_wrench_clear_v1",
        "source": "permanent_wrench_composer_body_local_owned_rows",
        "owned_force_peak_n": force_peak,
        "owned_torque_peak_nm": torque_peak,
    }


def _owned_command_composer_wrench_stats(
    command,
    env_ids: torch.Tensor,
) -> dict[str, float]:
    command_force, command_torque, writer_ids = command.body_wrench_for_envs(env_ids)
    if not isinstance(writer_ids, torch.Tensor) or not torch.equal(writer_ids, env_ids):
        raise RuntimeError("reset evidence requires exact tensor environment IDs")
    composer = getattr(command.robot, "permanent_wrench_composer", None)
    composer_force = getattr(composer, "composed_force_as_torch", None)
    composer_torque = getattr(composer, "composed_torque_as_torch", None)
    if not isinstance(composer_force, torch.Tensor) or not isinstance(
        composer_torque,
        torch.Tensor,
    ):
        raise RuntimeError("reset evidence requires permanent composer torch buffers")
    owned_ids = command.application_body_ids
    observed_force = composer_force.index_select(0, env_ids).index_select(1, owned_ids)
    observed_torque = composer_torque.index_select(0, env_ids).index_select(1, owned_ids)
    if observed_force.shape != command_force.shape or observed_torque.shape != command_torque.shape:
        raise RuntimeError("command/composer owned-wrench layouts differ")

    def vector_peak(value: torch.Tensor) -> float:
        if value.numel() == 0 or value.shape[-1] != 3:
            raise RuntimeError("owned wrench evidence must contain non-empty 3-vectors")
        peak = float(torch.linalg.vector_norm(value, dim=-1).max().item())
        if not math.isfinite(peak):
            raise RuntimeError("owned wrench evidence contains a non-finite vector")
        return peak

    def max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
        difference = float(torch.abs(left - right).max().item())
        if not math.isfinite(difference):
            raise RuntimeError("owned wrench evidence contains a non-finite difference")
        return difference

    return {
        "command_force_peak_n": vector_peak(command_force),
        "command_torque_peak_nm": vector_peak(command_torque),
        "composer_force_peak_n": vector_peak(observed_force),
        "composer_torque_peak_nm": vector_peak(observed_torque),
        "force_max_abs_difference_n": max_difference(observed_force, command_force),
        "torque_max_abs_difference_nm": max_difference(observed_torque, command_torque),
    }


def exercise_sonic_evaluation_reset_event(
    event_manager,
    command,
    *,
    global_env_step_count: int,
) -> dict[str, object]:
    """Invoke the configured reset event after real force and prove exact cleanup."""

    validate_sonic_evaluation_event_manager(event_manager)
    if (
        isinstance(global_env_step_count, bool)
        or not isinstance(global_env_step_count, Integral)
        or int(global_env_step_count) <= 0
    ):
        raise ValueError("global_env_step_count must be a positive integer")
    ids = torch.arange(command.num_envs, device=command.device, dtype=torch.long)
    pre_reset = _owned_command_composer_wrench_stats(command, ids)
    if (
        pre_reset["command_force_peak_n"] <= 0.0
        or pre_reset["composer_force_peak_n"] <= 0.0
    ):
        raise RuntimeError("configured reset event must follow a nonzero owned force")
    if (
        pre_reset["force_max_abs_difference_n"] > 1.0e-6
        or pre_reset["torque_max_abs_difference_nm"] > 1.0e-6
    ):
        raise RuntimeError("command and composer differ before the reset event")

    event_manager.apply(
        mode="reset",
        env_ids=ids,
        global_env_step_count=int(global_env_step_count),
    )
    post_reset = _owned_command_composer_wrench_stats(command, ids)
    if any(value != 0.0 for value in post_reset.values()):
        raise RuntimeError("configured reset event did not clear command/composer exactly")
    return {
        "schema_version": "sonic_phase6_reset_event_evidence_v1",
        "event_name": SONIC_EVALUATION_RESET_EVENT,
        "resolved_func_target": SONIC_EVALUATION_MANAGER_PROVENANCE["runtime"][
            "events"
        ][0]["resolved_func_target"],
        "mode": "reset",
        "global_env_step_count": int(global_env_step_count),
        "pre_reset": pre_reset,
        "post_reset": post_reset,
    }


def validate_policy_action_byte_parity(
    baseline_report: Mapping[str, object],
    candidate_report: Mapping[str, object],
) -> None:
    """Require exact dtype/shape/count/per-row/aggregate policy action evidence."""

    def validate(report: Mapping[str, object]) -> None:
        if report.get("schema_version") != "policy_action_bytes_v1":
            raise ValueError("invalid policy action evidence schema")
        dtype = report.get("dtype")
        shape = report.get("shape_per_step")
        step_count = report.get("step_count")
        row_hashes = report.get("row_sha256")
        aggregate = report.get("aggregate_sha256")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("invalid policy action dtype evidence")
        if (
            not isinstance(shape, list)
            or len(shape) < 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError("invalid policy action shape evidence")
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
            raise ValueError("invalid policy action step count evidence")
        if not isinstance(row_hashes, list) or len(row_hashes) != step_count:
            raise ValueError("invalid policy action row hash evidence")
        digest = hashlib.sha256()
        try:
            for row_hash in row_hashes:
                if not isinstance(row_hash, str) or len(row_hash) != 64:
                    raise ValueError
                digest.update(bytes.fromhex(row_hash))
        except ValueError as exc:
            raise ValueError("invalid policy action row hash evidence") from exc
        if aggregate != digest.hexdigest():
            raise ValueError("invalid policy action aggregate hash evidence")

    validate(baseline_report)
    validate(candidate_report)
    fields = (
        "schema_version",
        "dtype",
        "shape_per_step",
        "step_count",
        "row_sha256",
        "aggregate_sha256",
    )
    for field_name in fields:
        if baseline_report.get(field_name) != candidate_report.get(field_name):
            raise ValueError(f"policy action byte parity mismatch: {field_name}")


def _resolved_env_ids(command, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(command.num_envs, device=command.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=command.device, dtype=torch.long)
    return torch.as_tensor(tuple(env_ids), device=command.device, dtype=torch.long)


def apply_sonic_evaluation_protocol(
    command,
    protocol: SonicEvaluationProtocol,
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Install one deterministic protocol after reset without applying a new wrench.

    Reset events own physical composer clearing.  This function updates logical
    command state before reset observations are computed, clears command-owned
    dynamics, and deliberately leaves force at zero on the reset snapshot.
    """

    site_ids = tuple(command.cfg.site_body_names)
    unknown = set(protocol.active_site_ids) - set(site_ids)
    if unknown:
        raise ValueError(f"protocol references unknown SONIC sites: {sorted(unknown)}")
    command.set_operational_enabled(protocol.operational_enabled)
    ids = _resolved_env_ids(command, env_ids)
    if ids.ndim != 1:
        raise ValueError("env_ids must be one-dimensional")
    state = command.state
    if not protocol.enabled:
        state._disable_prevalidated(ids)
        command._clear_application_buffers_prevalidated(ids)
        command.time_left[ids] = torch.finfo(command.time_left.dtype).max
        return

    state._clear_dynamic_prevalidated(ids)
    state.enabled[ids] = True
    state.active_site_mask[ids] = False
    for site_id in protocol.active_site_ids:
        state.active_site_mask[ids, site_ids.index(site_id)] = True
    threshold = float(protocol.force_threshold_n)
    stiffness = threshold / float(command.cfg.reference_displacement_m)
    state.force_threshold_n[ids] = threshold
    state.stiffness_n_per_m[ids] = stiffness
    state._condition[ids, 0] = 1.0
    state._condition[ids, 1] = threshold
    state._condition[ids, 2] = stiffness
    state.reference_offset_common[ids] = 0.0
    offset = torch.as_tensor(
        protocol.reference_offset_common_m,
        dtype=state.dtype,
        device=state.device,
    )
    for site_id in protocol.active_site_ids:
        state.reference_offset_common[ids, site_ids.index(site_id)] = offset
    command._clear_application_buffers_prevalidated(ids)
    command.time_left[ids] = torch.finfo(command.time_left.dtype).max


def _wxyz_to_xyzw(value: torch.Tensor) -> torch.Tensor:
    return value[..., (1, 2, 3, 0)]


def _to_numpy_float32(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float32).numpy().copy()


def _motion_identity_strings(tracking) -> tuple[str, ...]:
    dataset_ids = tracking.motion_lib.get_motion_ids_in_dataset(tracking.motion_ids)
    dataset_values = dataset_ids.detach().to(device="cpu").reshape(-1).tolist()
    start_values = (
        tracking.motion_start_time_steps.detach().to(device="cpu").reshape(-1).tolist()
    )
    if len(dataset_values) != len(start_values):
        raise ValueError("SONIC motion identity tensors have different batch sizes")
    return tuple(
        f"dataset_motion:{int(dataset_id)}:start_frame:{int(start_frame)}"
        for dataset_id, start_frame in zip(dataset_values, start_values, strict=True)
    )


def assert_g1_only_encoder_selection(tracking) -> None:
    """Fail closed unless every row uses only the robot-motion G1 encoder."""

    expected_probs = {"g1": 1.0, "teleop": 0.0, "smpl": 0.0}
    if dict(tracking.encoder_sample_probs_dict) != expected_probs:
        raise RuntimeError("Phase-6 robot-motion encoder probabilities changed")
    names = tuple(tracking.encoder_sample_probs_dict)
    if names != ("g1", "teleop", "smpl"):
        raise RuntimeError("Phase-6 robot-motion encoder order changed")
    expected = torch.zeros_like(tracking.encoder_index)
    expected[:, names.index("g1")] = 1
    if not torch.equal(tracking.encoder_index, expected):
        raise RuntimeError("Phase-6 observed a non-G1 encoder selection")


def _actual_composer_wrench_evidence(
    command,
    site_state,
    output_frame_quaternion_world_wxyz: torch.Tensor,
):
    """Read every command-owned composer row and recover site force in common."""

    composer = getattr(command.robot, "permanent_wrench_composer", None)
    if composer is None:
        raise RuntimeError("Phase-6 evidence requires the permanent wrench composer")
    force_body = getattr(composer, "composed_force_as_torch", None)
    torque_body = getattr(composer, "composed_torque_as_torch", None)
    if not isinstance(force_body, torch.Tensor) or not isinstance(torque_body, torch.Tensor):
        raise RuntimeError("permanent wrench composer lacks torch force/torque buffers")
    owned_ids = command.application_body_ids
    owned_force_body = force_body[:, owned_ids]
    owned_torque_body = torque_body[:, owned_ids]
    expected_rows = command.body_map.num_sites + 1
    if owned_force_body.shape[1:] != (expected_rows, 3) or owned_torque_body.shape[1:] != (
        expected_rows,
        3,
    ):
        raise RuntimeError("command-owned composer row layout changed")
    site_force_body = owned_force_body[:, : command.body_map.num_sites]
    site_force_world = _rotate_vectors_wxyz_unchecked(
        site_state.site_quaternion_world,
        site_force_body,
    )
    site_force_common = _world_to_body_vectors_unchecked(
        site_force_world,
        output_frame_quaternion_world_wxyz[:, None, :],
    )
    owned_force_peak = torch.linalg.vector_norm(owned_force_body, dim=-1).max(dim=-1).values
    owned_torque_peak = torch.linalg.vector_norm(owned_torque_body, dim=-1).max(dim=-1).values
    expected_force_body = command._application_force_body
    expected_torque_body = command._application_torque_body
    if expected_force_body.shape != owned_force_body.shape or expected_torque_body.shape != (
        owned_torque_body.shape
    ):
        raise RuntimeError("command body-local wrench buffer layout changed")
    force_buffer_difference = torch.abs(owned_force_body - expected_force_body).reshape(
        command.num_envs,
        -1,
    )
    torque_buffer_difference = torch.abs(owned_torque_body - expected_torque_body).reshape(
        command.num_envs,
        -1,
    )
    return (
        site_force_common,
        owned_force_peak,
        owned_torque_peak,
        force_buffer_difference.max(dim=-1).values,
        torque_buffer_difference.max(dim=-1).values,
    )


def snapshot_from_sonic_commands(command) -> SonicEvaluationSnapshot:
    """Read fresh SONIC reference/articulation state into the portable convention."""

    tracking = command._tracking_term()
    site_state = command._site_tracking_state()
    reference_anchor_index = command.body_map.reference_anchor_index
    reference_anchor_position_world = tracking.body_pos_w[:, reference_anchor_index]
    reference_anchor_quaternion_world = tracking.body_quat_w[:, reference_anchor_index]
    original_site_world = command._reference_world_state()[:, 0]
    original_site = _world_to_common_positions_unchecked(
        original_site_world,
        reference_anchor_position_world[:, None, :],
        reference_anchor_quaternion_world[:, None, :],
    )
    compliant_site_robot_anchor = site_state.compliant_reference_common[:, 0]

    def robot_anchor_to_world(positions: torch.Tensor) -> torch.Tensor:
        return site_state.anchor_position_world[:, None, :] + _rotate_vectors_wxyz_unchecked(
            site_state.anchor_quaternion_world[:, None, :],
            positions,
        )

    compliant_site_world = robot_anchor_to_world(compliant_site_robot_anchor)
    compliant_site_reference_anchor = _world_to_common_positions_unchecked(
        compliant_site_world,
        reference_anchor_position_world[:, None, :],
        reference_anchor_quaternion_world[:, None, :],
    )
    selected_site = _select_yielded_site_reference_unchecked(
        original_site,
        compliant_site_reference_anchor,
        command.state.active_site_mask,
        command.state.enabled,
    )
    measured_site_world = site_state.site_body_position_world + site_state.site_offset_world
    measured_site = _world_to_common_positions_unchecked(
        measured_site_world,
        reference_anchor_position_world[:, None, :],
        reference_anchor_quaternion_world[:, None, :],
    )

    reference_points_global = tracking.body_pos_w
    measured_points_global = tracking.robot_body_pos_w
    reference_points_local = _world_to_common_positions_unchecked(
        reference_points_global,
        tracking.anchor_pos_w[:, None, :],
        tracking.anchor_quat_w[:, None, :],
    )
    measured_points_local = _world_to_common_positions_unchecked(
        measured_points_global,
        tracking.robot_anchor_pos_w[:, None, :],
        tracking.anchor_quat_w[:, None, :],
    )
    reference_site_quaternion = tracking.body_quat_w[
        :, command.body_map.reference_site_indices
    ]
    (
        force_common,
        owned_force_peak,
        owned_torque_peak,
        force_buffer_max_abs_difference,
        torque_buffer_max_abs_difference,
    ) = _actual_composer_wrench_evidence(
        command,
        site_state,
        reference_anchor_quaternion_world,
    )
    return SonicEvaluationSnapshot(
        motion_ids=_motion_identity_strings(tracking),
        site_ids=tuple(command.cfg.site_body_names),
        point_ids=tuple(tracking.cfg.body_names),
        original_site_positions_m=_to_numpy_float32(original_site),
        selected_site_positions_m=_to_numpy_float32(selected_site),
        measured_site_positions_m=_to_numpy_float32(measured_site),
        original_site_orientations_xyzw=_to_numpy_float32(
            _wxyz_to_xyzw(reference_site_quaternion)
        ),
        measured_site_orientations_xyzw=_to_numpy_float32(
            _wxyz_to_xyzw(site_state.site_quaternion_world)
        ),
        reference_points_global_m=_to_numpy_float32(reference_points_global),
        measured_points_global_m=_to_numpy_float32(measured_points_global),
        reference_points_local_m=_to_numpy_float32(reference_points_local),
        measured_points_local_m=_to_numpy_float32(measured_points_local),
        force_on_robot_n=_to_numpy_float32(force_common),
        owned_wrench_force_peak_n=_to_numpy_float32(owned_force_peak),
        owned_wrench_torque_peak_nm=_to_numpy_float32(owned_torque_peak),
        owned_force_buffer_max_abs_difference_n=_to_numpy_float32(
            force_buffer_max_abs_difference
        ),
        owned_torque_buffer_max_abs_difference_nm=_to_numpy_float32(
            torque_buffer_max_abs_difference
        ),
        compliance_enabled=command.state.enabled.detach().to(device="cpu").numpy().copy(),
        active_site_mask=(
            command.state.active_site_mask.detach().to(device="cpu").numpy().copy()
        ),
    )


@dataclass
class _CollectedRow:
    order: int
    motion_id: str
    sequence_id: str
    seed_id: int
    frame_index: int
    timestamp_s: float
    values: dict[str, np.ndarray | np.bool_]
    terminal: bool = False
    success: bool = False
    fall: bool = False
    reset: bool = False


@dataclass
class _SequenceBuffer:
    env_id: int
    episode_index: int
    motion_id: str
    sequence_id: str
    rows: list[_CollectedRow]
    transition_count: int = 0
    included: bool = True


class SonicEvaluationTraceCollector:
    """Bounded lifecycle collector producing one strict portable trace."""

    def __init__(
        self,
        *,
        trial_name: str,
        seed_id: int,
        step_dt_s: float,
        site_ids: Sequence[str],
        point_ids: Sequence[str],
        max_rows: int = 100_000,
    ) -> None:
        if not isinstance(trial_name, str) or not trial_name:
            raise ValueError("trial_name must be a non-empty string")
        if isinstance(seed_id, bool) or not isinstance(seed_id, Integral) or seed_id < 0:
            raise ValueError("seed_id must be a non-negative integer")
        if not isinstance(step_dt_s, (int, float)) or not math.isfinite(step_dt_s) or step_dt_s <= 0:
            raise ValueError("step_dt_s must be finite and positive")
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")
        self.trial_name = trial_name
        self.seed_id = int(seed_id)
        self.step_dt_s = float(step_dt_s)
        self.site_ids = _name_tuple("site_ids", site_ids)
        self.point_ids = _name_tuple("point_ids", point_ids)
        self.max_rows = max_rows
        self._sequences: list[_SequenceBuffer] = []
        self._current: dict[int, _SequenceBuffer] = {}
        self._next_episode: dict[int, int] = {}
        self._row_count = 0
        self._sealed = False
        self._final_rows: list[_CollectedRow] | None = None

    def _validate_snapshot(self, snapshot: SonicEvaluationSnapshot) -> None:
        if snapshot.site_ids != self.site_ids:
            raise ValueError("SONIC snapshot site layout changed")
        if snapshot.point_ids != self.point_ids:
            raise ValueError("SONIC snapshot tracking-point layout changed")

    def _reserve_row(self) -> int:
        if self._sealed:
            raise RuntimeError("collector is already finalized")
        if self._row_count >= self.max_rows:
            raise RuntimeError("collector exceeded max_rows")
        order = self._row_count
        self._row_count += 1
        return order

    @staticmethod
    def _snapshot_values(
        snapshot: SonicEvaluationSnapshot,
        row: int,
    ) -> dict[str, np.ndarray | np.bool_]:
        values: dict[str, np.ndarray | np.bool_] = {
            field_name: np.array(getattr(snapshot, field_name)[row], copy=True)
            for field_name in _FLOAT_FIELDS
        }
        for field_name in _ADAPTER_EVIDENCE_FIELDS:
            values[field_name] = np.array(getattr(snapshot, field_name)[row], copy=True)
        values["compliance_enabled"] = np.bool_(snapshot.compliance_enabled[row])
        values["active_site_mask"] = np.array(snapshot.active_site_mask[row], copy=True)
        return values

    @staticmethod
    def _batch_env_ids(
        snapshot: SonicEvaluationSnapshot,
        env_ids: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if env_ids is None:
            return tuple(range(len(snapshot.motion_ids)))
        result = tuple(env_ids)
        if any(
            isinstance(env_id, bool)
            or not isinstance(env_id, Integral)
            or env_id < 0
            or env_id >= len(snapshot.motion_ids)
            for env_id in result
        ):
            raise IndexError("env_ids are out of snapshot range")
        if len(set(result)) != len(result):
            raise ValueError("env_ids must not contain duplicates")
        return tuple(int(env_id) for env_id in result)

    def record_post_reset(
        self,
        snapshot: SonicEvaluationSnapshot,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Record the sole reset row for each new sequence."""

        self._validate_snapshot(snapshot)
        for env_id in self._batch_env_ids(snapshot, env_ids):
            prior = self._current.get(env_id)
            if prior is not None:
                if prior.transition_count != 0:
                    raise RuntimeError("reset arrived before the prior sequence terminated")
                prior.included = False
                episode_index = prior.episode_index
            else:
                episode_index = self._next_episode.get(env_id, 0)
                self._next_episode[env_id] = episode_index + 1
            sequence_id = f"env_{env_id:04d}:episode_{episode_index:04d}"
            row = _CollectedRow(
                order=self._reserve_row(),
                motion_id=snapshot.motion_ids[env_id],
                sequence_id=sequence_id,
                seed_id=self.seed_id,
                frame_index=0,
                timestamp_s=0.0,
                values=self._snapshot_values(snapshot, env_id),
                reset=True,
            )
            sequence = _SequenceBuffer(
                env_id=env_id,
                episode_index=episode_index,
                motion_id=row.motion_id,
                sequence_id=sequence_id,
                rows=[row],
            )
            self._sequences.append(sequence)
            self._current[env_id] = sequence

    @staticmethod
    def _event_mask(name: str, value: object, batch_size: int) -> np.ndarray:
        return _bool_array(name, value, (batch_size,))

    def record_post_step(
        self,
        snapshot: SonicEvaluationSnapshot,
        *,
        terminal_mask: object,
        success_mask: object,
        fall_mask: object,
    ) -> None:
        """Record physics output before IsaacLab performs any automatic reset."""

        self._validate_snapshot(snapshot)
        batch_size = len(snapshot.motion_ids)
        terminal = self._event_mask("terminal_mask", terminal_mask, batch_size)
        success = self._event_mask("success_mask", success_mask, batch_size)
        fall = self._event_mask("fall_mask", fall_mask, batch_size)
        if np.any(success & ~terminal) or np.any(fall & ~terminal):
            raise ValueError("success/fall events require terminal_mask")
        if np.any(success & fall):
            raise ValueError("success and fall masks must be disjoint")
        for env_id in range(batch_size):
            sequence = self._current.get(env_id)
            if sequence is None:
                raise RuntimeError("post-step snapshot has no preceding post-reset snapshot")
            if snapshot.motion_ids[env_id] != sequence.motion_id:
                raise ValueError("SONIC motion identity changed within a sequence")
            sequence.transition_count += 1
            row = _CollectedRow(
                order=self._reserve_row(),
                motion_id=sequence.motion_id,
                sequence_id=sequence.sequence_id,
                seed_id=self.seed_id,
                frame_index=sequence.transition_count,
                timestamp_s=sequence.transition_count * self.step_dt_s,
                values=self._snapshot_values(snapshot, env_id),
                terminal=bool(terminal[env_id]),
                success=bool(success[env_id]),
                fall=bool(fall[env_id]),
            )
            sequence.rows.append(row)
            if row.terminal:
                self._current.pop(env_id)

    def finalize(
        self,
        *,
        natural_timeout_env_ids: Sequence[int],
        failed_env_ids: Sequence[int] = (),
    ) -> EvaluationTrace:
        """Close only streams that reached the observed natural motion timeout."""

        if self._sealed:
            raise RuntimeError("collector is already finalized")
        natural_timeouts = tuple(natural_timeout_env_ids)
        failed = tuple(failed_env_ids)
        if any(
            isinstance(env_id, bool) or not isinstance(env_id, Integral) or env_id < 0
            for env_id in (*natural_timeouts, *failed)
        ):
            raise ValueError("outcome env IDs must contain non-negative integers")
        if len(set(natural_timeouts)) != len(natural_timeouts) or len(set(failed)) != len(failed):
            raise ValueError("outcome env IDs must not contain duplicates")
        open_with_physics = {
            env_id
            for env_id, sequence in self._current.items()
            if sequence.transition_count > 0
        }
        if set(natural_timeouts) != open_with_physics:
            raise RuntimeError(
                "every published sequence must reach the observed natural motion timeout"
            )
        if not set(failed).issubset(open_with_physics):
            raise ValueError("failed_env_ids must identify naturally timed-out streams")
        for env_id, sequence in tuple(self._current.items()):
            if sequence.transition_count == 0:
                sequence.included = False
            else:
                final_row = sequence.rows[-1]
                final_row.terminal = True
                final_row.success = env_id not in failed
                final_row.fall = env_id in failed
            self._current.pop(env_id)
        self._sealed = True
        rows = sorted(
            (
                row
                for sequence in self._sequences
                if sequence.included
                for row in sequence.rows
            ),
            key=lambda row: row.order,
        )
        if not rows:
            raise RuntimeError("collector has no completed sequence")
        self._final_rows = rows

        def stack(field_name: str) -> np.ndarray:
            return np.stack([row.values[field_name] for row in rows], axis=0)

        return EvaluationTrace(
            trial_name=self.trial_name,
            motion_ids=tuple(row.motion_id for row in rows),
            sequence_ids=tuple(row.sequence_id for row in rows),
            seed_ids=np.asarray([row.seed_id for row in rows], dtype=np.int64),
            frame_indices=np.asarray([row.frame_index for row in rows], dtype=np.int64),
            timestamps_s=np.asarray([row.timestamp_s for row in rows], dtype=np.float64),
            site_ids=self.site_ids,
            point_ids=self.point_ids,
            original_site_positions_m=stack("original_site_positions_m"),
            selected_site_positions_m=stack("selected_site_positions_m"),
            measured_site_positions_m=stack("measured_site_positions_m"),
            original_site_orientations_xyzw=stack("original_site_orientations_xyzw"),
            measured_site_orientations_xyzw=stack("measured_site_orientations_xyzw"),
            reference_points_global_m=stack("reference_points_global_m"),
            measured_points_global_m=stack("measured_points_global_m"),
            reference_points_local_m=stack("reference_points_local_m"),
            measured_points_local_m=stack("measured_points_local_m"),
            force_on_robot_n=stack("force_on_robot_n"),
            compliance_enabled=np.asarray(
                [row.values["compliance_enabled"] for row in rows],
                dtype=np.bool_,
            ),
            active_site_mask=stack("active_site_mask"),
            terminal_mask=np.asarray([row.terminal for row in rows], dtype=np.bool_),
            success_mask=np.asarray([row.success for row in rows], dtype=np.bool_),
            fall_mask=np.asarray([row.fall for row in rows], dtype=np.bool_),
            reset_mask=np.asarray([row.reset for row in rows], dtype=np.bool_),
        )

    def adapter_evidence_report(self) -> dict[str, object]:
        """Summarize actual composer rows after finalization, including reset."""

        if self._final_rows is None:
            raise RuntimeError("collector must be finalized before evidence reporting")
        reset_rows = [row for row in self._final_rows if row.reset]
        if not reset_rows:
            raise RuntimeError("collector has no reset evidence row")

        def maximum(field_name: str, rows: Sequence[_CollectedRow]) -> float:
            values = np.asarray([row.values[field_name] for row in rows], dtype=np.float64)
            if not np.isfinite(values).all():
                return float("nan")
            return float(np.max(values))

        return {
            "schema_version": "sonic_actual_composer_wrench_v1",
            "source": "permanent_wrench_composer_body_local_owned_rows",
            "owned_row_semantics": "ordered compliance sites followed by anchor",
            "reset_owned_force_peak_n": maximum(
                "owned_wrench_force_peak_n",
                reset_rows,
            ),
            "reset_owned_torque_peak_nm": maximum(
                "owned_wrench_torque_peak_nm",
                reset_rows,
            ),
            "owned_force_buffer_max_abs_difference_n": maximum(
                "owned_force_buffer_max_abs_difference_n",
                self._final_rows,
            ),
            "owned_torque_buffer_max_abs_difference_nm": maximum(
                "owned_torque_buffer_max_abs_difference_nm",
                self._final_rows,
            ),
        }
