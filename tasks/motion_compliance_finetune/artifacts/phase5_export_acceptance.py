#!/usr/bin/env python3
"""Independent PT/ORT and hard-switch acceptance for a Phase-5 export."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from gear_sonic.compliance_control.deployment import (
    ActionResidualRuntime,
    ArtifactExpectation,
    ExportableActionResidual,
    ReleaseArtifactPin,
    load_artifact_metadata,
    load_onnxruntime_session,
)
from gear_sonic.compliance_control.training.checkpoint import (
    checkpoint_sha256,
    expected_residual_shapes,
    load_trl_checkpoint,
)
from gear_sonic.compliance_control.training.paths import (
    validate_distinct_artifact_paths,
    validate_motion_compliance_run_path,
)
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER


class CountingSession:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.calls = 0

    def run(self, output_names, input_feed):
        self.calls += 1
        return self.wrapped.run(output_names, input_feed)


def _atomic_json(path: Path, value: dict, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary_name, path)
        else:
            # Hard-link publication is atomic and fails closed if a concurrent
            # process creates the report after the preflight check.
            os.link(temporary_name, path)
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_regular_input(path: Path, name: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular, non-symlink file")
    return path.resolve(strict=True)


def _validate_report_destination(
    *,
    report: Path,
    artifact_directory: Path,
    checkpoint: Path,
    deployment_overlay: Path,
    model: Path,
    metadata: Path,
    release_artifacts: Sequence[Path],
    overwrite: bool,
) -> Path:
    """Reject every read/write collision before publishing acceptance evidence."""

    bundle_path = artifact_directory.resolve(strict=True)
    for name, path in (("model", model), ("metadata", metadata)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"artifact {name} must be a regular, non-symlink file")
        if not path.resolve(strict=True).is_relative_to(bundle_path):
            raise ValueError(f"artifact {name} must remain inside the bundle")
    named_paths = dict(
        report=report,
        checkpoint=checkpoint,
        deployment_overlay=deployment_overlay,
        model=model,
        metadata=metadata,
    )
    named_paths.update(
        {
            f"release_artifact_{index}": path
            for index, path in enumerate(release_artifacts)
        }
    )
    resolved = validate_distinct_artifact_paths(**named_paths)
    report_path = resolved["report"]
    if report_path.is_relative_to(bundle_path):
        raise ValueError("acceptance report must be outside the artifact bundle")
    if report_path.is_symlink() or (report_path.exists() and not report_path.is_file()):
        raise ValueError("acceptance report target must be a regular file or absent")
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            "acceptance report already exists; pass --overwrite-report explicitly"
        )
    return report_path


def _checkpoint_global_step(checkpoint) -> int:
    state = checkpoint["state"]
    value = state.get("global_step") if isinstance(state, dict) else state.global_step
    if type(value) is not int:
        raise ValueError("checkpoint global step must be an integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--num-sites", type=int, required=True)
    parser.add_argument("--deployment-overlay", type=Path, required=True)
    parser.add_argument(
        "--release-artifact",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "PATH", "SHA256"),
        help="repeatable caller-owned release pin",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite-report", action="store_true")
    args = parser.parse_args()
    artifact_directory = validate_motion_compliance_run_path(args.artifact_directory)
    report_candidate = validate_motion_compliance_run_path(args.report)
    checkpoint_path = _require_regular_input(args.checkpoint, "checkpoint")
    overlay_path = _require_regular_input(args.deployment_overlay, "deployment overlay")
    release_artifacts = tuple(
        ReleaseArtifactPin(
            name=name,
            path=_require_regular_input(Path(path), f"release artifact {name}"),
            sha256=sha256,
        )
        for name, path, sha256 in args.release_artifact
    )
    if checkpoint_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("independent checkpoint digest mismatch")
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))[
        "motion_compliance_action_residual"
    ]
    site_layout = tuple(overlay["site_layout"])
    action_layout = tuple(overlay["action_layout"])
    declared_release = overlay.get("release_artifacts")
    if declared_release != [
        {"name": pin.name, "sha256": pin.sha256} for pin in release_artifacts
    ]:
        raise ValueError("overlay release artifacts differ from caller-owned pins")
    if len(site_layout) != args.num_sites:
        raise ValueError("overlay site count differs")
    if action_layout != tuple(G1_ISAACLab_ORDER):
        raise ValueError("overlay action layout differs from release decoder output order")
    expectation = ArtifactExpectation(
        metadata_sha256=args.metadata_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_global_step=args.global_step,
        release_context_width=994,
        condition_width=3,
        action_layout=action_layout,
        site_layout=site_layout,
        max_abs_delta=0.25,
        release_artifacts=release_artifacts,
    )
    metadata = load_artifact_metadata(artifact_directory, expectation)
    metadata_path = artifact_directory / "action_residual.metadata.json"
    model_path = artifact_directory / metadata["model"]["file"]
    report_path = _validate_report_destination(
        report=report_candidate,
        artifact_directory=artifact_directory,
        checkpoint=checkpoint_path,
        deployment_overlay=overlay_path,
        model=model_path,
        metadata=metadata_path,
        release_artifacts=tuple(Path(pin.path) for pin in release_artifacts),
        overwrite=args.overwrite_report,
    )
    graph = onnx.load(model_path).graph
    if len(graph.initializer) != 6:
        raise AssertionError("standalone graph must contain only six residual tensors")
    if {value.name for value in graph.input} != {
        "release_action_context",
        "motion_compliance_condition",
    } or [value.name for value in graph.output] != ["action_delta"]:
        raise AssertionError("standalone graph I/O names differ from the export contract")

    checkpoint = load_trl_checkpoint(checkpoint_path, map_location="cpu")
    if _checkpoint_global_step(checkpoint) != args.global_step:
        raise ValueError("independent checkpoint global step mismatch")
    policy_state = checkpoint[metadata["source_checkpoint"]["policy_state_key"]]
    residual_shapes, _ = expected_residual_shapes(args.num_sites)
    names = tuple(residual_shapes)
    if names != tuple(metadata["source_checkpoint"]["residual_tensor_names"]):
        raise ValueError("metadata residual tensor ordering differs from training schema")
    tensors = {name: policy_state[name] for name in names}
    module = ExportableActionResidual(
        994,
        3,
        len(action_layout),
        hidden_dims=(256, 256),
        max_abs_delta=0.25,
    )
    module.load_linear_state(tensors, source_names=names)
    raw_session = load_onnxruntime_session(artifact_directory, metadata)

    max_abs_error = 0.0
    cases = 0
    for batch, sequence in ((2, 3), (1, 5)):
        generator = np.random.default_rng(99991 + batch * 100 + sequence)
        base_context = generator.normal(size=(batch, sequence, 994)).astype(np.float32)
        for mode in ("off", "on", "mixed"):
            gate = np.zeros((batch, sequence), dtype=np.bool_)
            if mode == "on":
                gate[...] = True
            elif mode == "mixed":
                gate.flat[::2] = True
            condition = np.zeros((batch, sequence, 3), dtype=np.float32)
            condition[..., 0] = gate
            condition[..., 1] = gate * 10.0
            condition[..., 2] = gate * 200.0
            context = base_context.copy()
            if mode == "mixed":
                context[~gate] = np.nan
                condition[~gate] = np.nan
            with torch.no_grad():
                expected = module(
                    torch.from_numpy(context), torch.from_numpy(condition)
                ).numpy()
            actual = raw_session.run(
                ["action_delta"],
                {
                    "release_action_context": context,
                    "motion_compliance_condition": condition,
                },
            )[0]
            if not np.isfinite(actual).all():
                raise AssertionError("ORT output contains NaN/Inf")
            if not np.array_equal(actual[~gate], np.zeros_like(actual[~gate])):
                raise AssertionError("ORT rejected rows are not exact zero")
            if gate.any():
                error = float(np.max(np.abs(actual[gate] - expected[gate])))
                max_abs_error = max(max_abs_error, error)
                if error > 1.0e-5:
                    raise AssertionError(f"PT/ORT residual mismatch: {error}")
            cases += 1

    counting = CountingSession(raw_session)
    shape = (3, 4)
    generator = np.random.default_rng(32452843)
    release = generator.normal(size=(*shape, len(action_layout))).astype(np.float32)
    release[0, 1, 0] = -0.0
    context = generator.normal(size=(*shape, 994)).astype(np.float32)
    gate = np.zeros(shape, dtype=np.bool_)
    gate.flat[::2] = True
    condition = np.zeros((*shape, 3), dtype=np.float32)
    condition[..., 0] = gate
    condition[..., 1] = gate * 10.0
    condition[..., 2] = gate * 200.0
    context[~gate] = np.nan
    condition[~gate] = np.nan
    disabled_runtime = ActionResidualRuntime(metadata, enabled=False, session=counting)
    hard_off = disabled_runtime.compose(release, context, condition, gate)
    if counting.calls != 0 or not np.array_equal(
        hard_off.view(np.uint32), release.view(np.uint32)
    ):
        raise AssertionError("host hard-off path invoked session or changed release bytes")
    enabled_runtime = ActionResidualRuntime(metadata, enabled=True, session=counting)
    all_off = enabled_runtime.compose(
        release, context, condition, np.zeros(shape, dtype=np.bool_)
    )
    if counting.calls != 0 or not np.array_equal(
        all_off.view(np.uint32), release.view(np.uint32)
    ):
        raise AssertionError("all-off batch invoked session or changed release bytes")
    composed = enabled_runtime.compose(release, context, condition, gate)
    if counting.calls != 1:
        raise AssertionError("mixed runtime must invoke the residual session exactly once")
    if not np.array_equal(composed[~gate].view(np.uint32), release[~gate].view(np.uint32)):
        raise AssertionError("mixed off rows changed release action bytes")
    max_runtime_delta = float(np.max(np.abs(composed[gate] - release[gate])))
    if max_runtime_delta > 0.25:
        raise AssertionError("runtime action residual exceeds 0.25")

    report = {
        "status": "MOTION_COMPLIANCE_PHASE5_ACCEPTANCE_PASS",
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_global_step": args.global_step,
        "metadata_sha256": args.metadata_sha256,
        "model_sha256": metadata["model"]["sha256"],
        "dynamic_shapes": [[2, 3], [1, 5]],
        "parity_cases": cases,
        "max_pt_ort_abs_error": max_abs_error,
        "runtime_session_calls": counting.calls,
        "hard_off_session_calls": 0,
        "max_runtime_abs_delta": max_runtime_delta,
        "site_count": len(site_layout),
        "action_width": len(action_layout),
        "onnx_initializer_count": len(graph.initializer),
        "release_artifact_sha256": {
            pin.name: pin.sha256 for pin in release_artifacts
        },
    }
    _atomic_json(report_path, report, overwrite=args.overwrite_report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
