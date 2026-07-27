"""Export the opt-in SONIC actor residual without rewriting release ONNX files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ...core import ComplianceResidualMLP
from ...postprocess import write_json_new_atomic


ACTOR_RESIDUAL_STATE_PREFIX = "actor_module.compliance_residual."
PHASE5_ACCEPTED_ONNXRUNTIME_VERSION = "1.25.0"
_EXPECTED_RESIDUAL_KEYS = (
    "trunk.0.weight",
    "trunk.0.bias",
    "trunk.2.weight",
    "trunk.2.bias",
    "output_layer.weight",
    "output_layer.bias",
)


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SonicResidualExportSpec:
    """Explicit deployment contract for a separately composed residual model."""

    site_names: tuple[str, ...]
    num_future_frames: int
    cartesian_dim: int
    context_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)
    residual_limit: float = 0.25
    common_frame: str = "heading_local:pelvis"

    def __post_init__(self) -> None:
        if not isinstance(self.site_names, tuple):
            raise TypeError("site_names must be a tuple")
        if not self.site_names or any(
            not isinstance(name, str) or not name.strip() for name in self.site_names
        ):
            raise ValueError("site_names must contain non-empty strings")
        if len(self.site_names) != len(set(self.site_names)):
            raise ValueError("site_names must be unique and ordered")
        for name in (
            "num_future_frames",
            "cartesian_dim",
            "context_dim",
            "output_dim",
        ):
            _positive_integer(getattr(self, name), name=name)
        if not isinstance(self.hidden_dims, tuple) or not self.hidden_dims:
            raise ValueError("hidden_dims must be a non-empty tuple")
        for width in self.hidden_dims:
            _positive_integer(width, name="hidden_dims entry")
        if (
            isinstance(self.residual_limit, bool)
            or not math.isfinite(self.residual_limit)
            or self.residual_limit <= 0.0
        ):
            raise ValueError("residual_limit must be finite and positive")
        if not isinstance(self.common_frame, str) or not self.common_frame.strip():
            raise ValueError("common_frame must be a non-empty string")

    @property
    def condition_dim(self) -> int:
        return (
            self.num_future_frames
            * len(self.site_names)
            * self.cartesian_dim
        )

    @property
    def command_dim(self) -> int:
        return 1 + len(self.site_names) + len(self.site_names) * self.cartesian_dim


def extract_actor_residual_state(
    policy_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip the SONIC actor prefix and reject incomplete/extra branch state."""

    if not isinstance(policy_state, Mapping):
        raise TypeError("policy_state must be a mapping")
    residual = {
        name[len(ACTOR_RESIDUAL_STATE_PREFIX) :]: tensor
        for name, tensor in policy_state.items()
        if name.startswith(ACTOR_RESIDUAL_STATE_PREFIX)
    }
    if tuple(sorted(residual)) != tuple(sorted(_EXPECTED_RESIDUAL_KEYS)):
        raise ValueError("policy state must contain exactly six actor residual tensors")
    for name, tensor in residual.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"residual state tensor is not floating point: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"residual state tensor is non-finite: {name}")
    return {name: tensor.detach().cpu().clone() for name, tensor in residual.items()}


def load_sonic_policy_state(checkpoint_path: str | Path) -> Mapping[str, torch.Tensor]:
    """Load a trusted local branch checkpoint with historical TRL aliases."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
        import trl.trainer.utils

        trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
        trl.trainer.utils.exact_div = exact_div
    except ImportError:
        pass
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    policy_state = checkpoint.get("policy_state_dict")
    if not isinstance(policy_state, Mapping):
        raise ValueError("checkpoint has no policy_state_dict mapping")
    return policy_state


def build_export_residual(
    state: Mapping[str, torch.Tensor],
    spec: SonicResidualExportSpec,
) -> ComplianceResidualMLP:
    """Instantiate the portable residual and strictly load extracted weights."""

    residual = ComplianceResidualMLP(
        condition_dim=spec.condition_dim,
        num_sites=len(spec.site_names),
        cartesian_dim=spec.cartesian_dim,
        context_dim=spec.context_dim,
        output_dim=spec.output_dim,
        hidden_dims=spec.hidden_dims,
        residual_limit=spec.residual_limit,
    ).to(dtype=torch.float32, device="cpu")
    normalized = extract_actor_residual_state(
        {
            f"{ACTOR_RESIDUAL_STATE_PREFIX}{name}": tensor
            for name, tensor in state.items()
        }
        if set(state) == set(_EXPECTED_RESIDUAL_KEYS)
        else state
    )
    residual.load_state_dict(normalized, strict=True)
    residual.eval()
    return residual


def _model_contract(spec: SonicResidualExportSpec) -> dict[str, Any]:
    sites = len(spec.site_names)
    compliance_start = 1 + sites
    return {
        "schema_version": 1,
        "model_role": "optional_post_fsq_latent_residual",
        "inputs": [
            {
                "name": "compliance_target",
                "shape": ["batch", "sequence", spec.condition_dim],
                "dtype": "float32",
                "frame": spec.common_frame,
            },
            {
                "name": "compliance_command",
                "shape": ["batch", "sequence", spec.command_dim],
                "dtype": "float32",
                "layout": {
                    "enable": [0, 1],
                    "ordered_site_mask": [1, compliance_start],
                    "ordered_cartesian_compliance_m_per_n": [
                        compliance_start,
                        spec.command_dim,
                    ],
                },
            },
            {
                "name": "actor_context",
                "shape": ["batch", "sequence", spec.context_dim],
                "dtype": "float32",
                "meaning": "release actor proprioceptive context",
            },
        ],
        "outputs": [
            {
                "name": "latent_residual",
                "shape": ["batch", "sequence", spec.output_dim],
                "dtype": "float32",
                "bound": [-spec.residual_limit, spec.residual_limit],
            }
        ],
        "site_names": list(spec.site_names),
        "num_future_frames": spec.num_future_frames,
        "cartesian_dim": spec.cartesian_dim,
        "composition": (
            "release_decoder_latent = release_encoder_latent + latent_residual"
        ),
        "hard_off": (
            "enable<=0.5, no selected site, or zero selected compliance produces "
            "an exact all-zero latent_residual"
        ),
        "release_models_modified": False,
    }


def export_sonic_actor_residual_onnx(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    spec: SonicResidualExportSpec,
    opset_version: int = 17,
) -> dict[str, Any]:
    """Export only the trained residual and write an adjacent JSON contract."""

    requested_output_path = Path(output_path)
    requested_manifest_path = requested_output_path.with_suffix(".json")
    if requested_output_path.suffix != ".onnx":
        raise ValueError("output_path must end in .onnx")
    if (
        requested_output_path.is_symlink()
        or requested_manifest_path.is_symlink()
        or os.path.lexists(requested_output_path)
        or os.path.lexists(requested_manifest_path)
    ):
        raise FileExistsError("residual ONNX or manifest path already exists")
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = requested_output_path.resolve()
    if type(opset_version) is not int or opset_version < 17:
        raise ValueError("opset_version must be an integer of at least 17")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".json")
    if os.path.lexists(output_path) or os.path.lexists(manifest_path):
        raise FileExistsError("residual ONNX or manifest path already exists")
    policy_state = load_sonic_policy_state(checkpoint_path)
    residual_state = extract_actor_residual_state(policy_state)
    residual = build_export_residual(residual_state, spec)
    example = (
        torch.zeros(2, 3, spec.condition_dim, dtype=torch.float32),
        torch.zeros(2, 3, spec.command_dim, dtype=torch.float32),
        torch.zeros(2, 3, spec.context_dim, dtype=torch.float32),
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".onnx",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    published_output = False
    try:
        torch.onnx.export(
            residual,
            example,
            temporary_path,
            input_names=("compliance_target", "compliance_command", "actor_context"),
            output_names=("latent_residual",),
            dynamic_axes={
                "compliance_target": {0: "batch", 1: "sequence"},
                "compliance_command": {0: "batch", 1: "sequence"},
                "actor_context": {0: "batch", 1: "sequence"},
                "latent_residual": {0: "batch", 1: "sequence"},
            },
            opset_version=opset_version,
            do_constant_folding=False,
            dynamo=False,
        )
        manifest = _model_contract(spec)
        manifest.update(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "residual_state_sha256": _state_sha256(residual_state),
                "onnx": str(output_path),
                "onnx_sha256": _file_sha256(temporary_path),
                "opset_version": opset_version,
                "spec": asdict(spec),
            }
        )
        os.link(temporary_path, output_path, follow_symlinks=False)
        published_output = True
        write_json_new_atomic(manifest_path, manifest)
        return manifest
    except BaseException:
        if published_output and output_path.exists():
            if output_path.stat().st_ino == temporary_path.stat().st_ino:
                output_path.unlink()
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_sonic_actor_residual_onnx(
    *,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    spec: SonicResidualExportSpec,
    atol: float = 2.0e-5,
    rtol: float = 2.0e-5,
    runtime: str = "auto",
) -> dict[str, Any]:
    """Run hard-off and dynamic-shape parity through a selected ONNX runtime.

    ``auto`` prefers the deployment-grade ONNX Runtime CPU provider and falls
    back to ONNX's portable reference evaluator. Independent artifact audits
    use ``onnxruntime`` explicitly so a fallback cannot be mistaken for runtime
    validation.
    """

    import onnx

    if not math.isfinite(atol) or atol < 0.0 or not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError("ONNX parity tolerances must be finite and non-negative")
    if runtime not in {"auto", "onnxruntime", "reference"}:
        raise ValueError("runtime must be auto, onnxruntime, or reference")
    onnx_path = Path(onnx_path).resolve()
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model, full_check=True)
    if [value.name for value in model.graph.input] != [
        "compliance_target",
        "compliance_command",
        "actor_context",
    ]:
        raise AssertionError("ONNX residual input order/name contract mismatch")
    if [value.name for value in model.graph.output] != ["latent_residual"]:
        raise AssertionError("ONNX residual output contract mismatch")

    policy_state = load_sonic_policy_state(checkpoint_path)
    residual = build_export_residual(extract_actor_residual_state(policy_state), spec)
    runtime_name: str | None = None
    runtime_version = None
    providers: list[str] = []
    evaluator: Any
    if runtime in {"auto", "onnxruntime"}:
        try:
            import onnxruntime
        except ImportError as error:
            if runtime == "onnxruntime":
                raise RuntimeError(
                    "onnxruntime is required for the independent artifact audit"
                ) from error
        else:
            evaluator = onnxruntime.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            runtime_name = "onnxruntime.InferenceSession"
            runtime_version = onnxruntime.__version__
            providers = list(evaluator.get_providers())

            def run_onnx(inputs: dict[str, np.ndarray]) -> np.ndarray:
                return evaluator.run(["latent_residual"], inputs)[0]

    if runtime == "reference" or (runtime == "auto" and runtime_name is None):
        from onnx.reference import ReferenceEvaluator

        evaluator = ReferenceEvaluator(model)
        runtime_name = "onnx.reference.ReferenceEvaluator"
        runtime_version = onnx.__version__

        def run_onnx(inputs: dict[str, np.ndarray]) -> np.ndarray:
            return evaluator.run(None, inputs)[0]

    if runtime_name is None:
        raise AssertionError("ONNX evaluator selection did not produce a runtime")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260727)
    maximum_error = 0.0
    cases = []
    for batch, sequence in ((1, 1), (2, 4)):
        target = torch.randn(
            batch,
            sequence,
            spec.condition_dim,
            generator=generator,
            dtype=torch.float32,
        )
        context = torch.randn(
            batch,
            sequence,
            spec.context_dim,
            generator=generator,
            dtype=torch.float32,
        )
        command = torch.zeros(
            batch,
            sequence,
            spec.command_dim,
            dtype=torch.float32,
        )
        command[..., 0] = 1.0
        command[..., 1 : 1 + len(spec.site_names)] = 1.0
        command[..., 1 + len(spec.site_names) :] = 0.02
        with torch.no_grad():
            expected = residual(target, command, context).numpy()
        actual = run_onnx(
            {
                "compliance_target": target.numpy(),
                "compliance_command": command.numpy(),
                "actor_context": context.numpy(),
            },
        )
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        cases.append({"batch": batch, "sequence": sequence, "enabled": True})

        off_target = target.clone()
        off_context = context.clone()
        off_target[..., 0] = float("nan")
        off_context[..., 0] = float("nan")
        off_command = torch.zeros_like(command)
        off_actual = run_onnx(
            {
                "compliance_target": off_target.numpy(),
                "compliance_command": off_command.numpy(),
                "actor_context": off_context.numpy(),
            },
        )
        if np.count_nonzero(off_actual) != 0:
            raise AssertionError("ONNX hard-off residual is not exact zero")
        zero_compliance = off_command.clone()
        zero_compliance[..., 0] = 1.0
        zero_compliance[..., 1 : 1 + len(spec.site_names)] = 1.0
        zero_actual = run_onnx(
            {
                "compliance_target": target.numpy(),
                "compliance_command": zero_compliance.numpy(),
                "actor_context": context.numpy(),
            },
        )
        if np.count_nonzero(zero_actual) != 0:
            raise AssertionError("ONNX zero-compliance residual is not exact zero")
        cases.append({"batch": batch, "sequence": sequence, "enabled": False})

    batch, sequence = 2, 4
    mixed_target = torch.randn(
        batch,
        sequence,
        spec.condition_dim,
        generator=generator,
        dtype=torch.float32,
    )
    mixed_context = torch.randn(
        batch,
        sequence,
        spec.context_dim,
        generator=generator,
        dtype=torch.float32,
    )
    mixed_command = torch.zeros(
        batch,
        sequence,
        spec.command_dim,
        dtype=torch.float32,
    )
    flat_command = mixed_command.reshape(-1, spec.command_dim)
    sites = len(spec.site_names)
    compliance_start = 1 + sites
    active_rows = torch.tensor([True, False, False, False, True, True, False, True])
    flat_command[[0, 7], 0] = 1.0
    flat_command[[0, 7], 1:compliance_start] = 1.0
    flat_command[[0, 7], compliance_start:] = 0.02
    flat_command[2, 0] = 1.0
    flat_command[2, compliance_start:] = 0.02
    flat_command[3, 0] = 1.0
    flat_command[3, 1:compliance_start] = 1.0
    flat_command[4, 0] = 1.0
    flat_command[4, 1] = 1.0
    flat_command[4, compliance_start : compliance_start + spec.cartesian_dim] = 0.02
    flat_command[5, 0] = 1.0
    flat_command[5, sites] = 1.0
    last_site_start = compliance_start + (sites - 1) * spec.cartesian_dim
    flat_command[5, last_site_start : last_site_start + spec.cartesian_dim] = 0.02
    flat_command[6, 0] = 1.0
    flat_command[6, 1] = 1.0
    if sites > 1:
        flat_command[6, last_site_start : last_site_start + spec.cartesian_dim] = 0.02
    flat_target = mixed_target.reshape(-1, spec.condition_dim)
    flat_context = mixed_context.reshape(-1, spec.context_dim)
    flat_target[~active_rows, 0] = float("nan")
    flat_context[~active_rows, 0] = float("nan")
    with torch.no_grad():
        mixed_expected = residual(
            mixed_target,
            mixed_command,
            mixed_context,
        ).numpy()
    mixed_actual = run_onnx(
        {
            "compliance_target": mixed_target.numpy(),
            "compliance_command": mixed_command.numpy(),
            "actor_context": mixed_context.numpy(),
        },
    )
    np.testing.assert_allclose(mixed_actual, mixed_expected, atol=atol, rtol=rtol)
    flat_actual = mixed_actual.reshape(-1, spec.output_dim)
    inactive_rows = ~active_rows.numpy()
    if np.count_nonzero(flat_actual[inactive_rows]) != 0:
        raise AssertionError("ONNX mixed inactive rows are not exact zero")
    mixed_error = np.max(
        np.abs(
            flat_actual[active_rows.numpy()]
            - mixed_expected.reshape(-1, spec.output_dim)[active_rows.numpy()]
        )
    )
    maximum_error = max(maximum_error, float(mixed_error))
    cases.append(
        {
            "batch": batch,
            "sequence": sequence,
            "mixed_rows": True,
            "active_rows": int(active_rows.sum().item()),
            "inactive_rows": int((~active_rows).sum().item()),
        }
    )

    return {
        "runtime": runtime_name,
        "runtime_version": runtime_version,
        "providers": providers,
        "onnx_checker": True,
        "hard_off_exact": True,
        "zero_compliance_exact": True,
        "mixed_row_exact": True,
        "dynamic_shape_cases": cases,
        "maximum_absolute_error": maximum_error,
        "atol": atol,
        "rtol": rtol,
    }
