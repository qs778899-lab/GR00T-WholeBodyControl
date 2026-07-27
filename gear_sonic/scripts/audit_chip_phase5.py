#!/usr/bin/env python3
"""Independently re-audit one completed CHIP Phase-5 workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from gear_sonic.compliance_control.adapters.sonic.contracts import (
    SONIC_RELEASE_TRACKING_BODY_NAMES,
)
from gear_sonic.compliance_control.core import (
    PairedEvaluationThresholds,
    compare_aligned_tracking_traces,
)
from gear_sonic.compliance_control.postprocess import (
    load_tracking_trace,
    paired_result_to_dict,
)


_DEFAULT_RUNS_ROOT = (_REPOSITORY_ROOT / "compliance_control/runs/chip").resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=_DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_artifact_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AssertionError(f"Phase-5 workflow contains a symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if path.suffix == ".log" and size > 64_000_000:
            raise AssertionError(f"oversized log: {path}")
        total += size
    if total > 500_000_000:
        raise AssertionError("Phase-5 workflow exceeds 500 MB")
    return total


def _require_canonical_payload_path(
    payload: dict,
    *,
    key: str,
    expected: Path,
    label: str,
) -> None:
    recorded = payload.get(key)
    if not isinstance(recorded, str) or not recorded:
        raise AssertionError(f"{label} is missing")
    recorded_path = Path(recorded)
    if recorded_path != expected or recorded_path.resolve() != expected:
        raise AssertionError(f"{label} mismatch")


def _require_workflow_provenance(
    workflow: dict,
    *,
    run_root: Path,
    runs_root: Path,
    checkpoint: Path,
) -> None:
    if workflow.get("complete") is not True:
        raise AssertionError("workflow is not complete")
    if workflow.get("marker") != "CHIP_PHASE5_EVAL_EXPORT_PASS":
        raise AssertionError("workflow marker mismatch")
    _require_canonical_payload_path(
        workflow,
        key="run_root",
        expected=run_root,
        label="workflow run-root",
    )
    _require_canonical_payload_path(
        workflow,
        key="runs_root",
        expected=runs_root,
        label="workflow runs-root",
    )
    _require_canonical_payload_path(
        workflow,
        key="checkpoint",
        expected=checkpoint,
        label="workflow checkpoint",
    )
    if workflow.get("evaluation_claim") != "chain_validation_not_performance_proof":
        raise AssertionError("workflow overstates the Phase-5 evidence claim")


def _require_rollout_checkpoint_provenance(
    summary: dict,
    *,
    mode: str,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> None:
    _require_canonical_payload_path(
        summary,
        key="checkpoint",
        expected=checkpoint,
        label=f"{mode} rollout checkpoint",
    )
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise AssertionError(f"{mode} rollout checkpoint SHA-256 mismatch")


def main() -> int:
    args = _parse_args()
    if args.run_root.is_symlink():
        raise AssertionError("Phase-5 workflow root must not be a symlink")
    allowed_root = args.runs_root.resolve()
    run_root = args.run_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if run_root == allowed_root or allowed_root not in run_root.parents:
        raise ValueError("run root is outside the bounded CHIP artifact root")
    if not run_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("run root or checkpoint is missing")
    workflow_bytes = _bounded_artifact_bytes(run_root)

    workflow = json.loads((run_root / "workflow.json").read_text(encoding="utf-8"))
    _require_workflow_provenance(
        workflow,
        run_root=run_root,
        runs_root=allowed_root,
        checkpoint=checkpoint,
    )
    checkpoint_sha256 = _sha256(checkpoint)

    thresholds = PairedEvaluationThresholds()
    stiff = load_tracking_trace(run_root / "stiff/trace.npz")
    compliant = load_tracking_trace(run_root / "compliant/trace.npz")
    result = compare_aligned_tracking_traces(
        stiff,
        compliant,
        thresholds=thresholds,
        alignment_atol=1.0e-5,
    )
    if not result.passed:
        raise AssertionError("independent paired metric recomputation failed")
    recorded = json.loads((run_root / "paired_metrics.json").read_text(encoding="utf-8"))
    recomputed = json.loads(json.dumps(paired_result_to_dict(result)))
    expected_thresholds = {
        name: getattr(thresholds, name) for name in thresholds.__dataclass_fields__
    }
    if recorded.get("thresholds") != expected_thresholds:
        raise AssertionError("recorded paired thresholds mismatch")
    for key in ("passed", "aligned_frames", "checks", "stiff", "compliant", "compliance_response"):
        if recorded.get(key) != recomputed[key]:
            raise AssertionError(f"recorded paired metric mismatch: {key}")

    onnx_path = run_root / "export/compliance_residual.onnx"
    manifest = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise AssertionError("export checkpoint SHA-256 mismatch")
    if manifest.get("onnx_sha256") != _sha256(onnx_path):
        raise AssertionError("export ONNX SHA-256 mismatch")
    if manifest.get("release_models_modified") is not False:
        raise AssertionError("export manifest does not preserve release models")
    spec = SonicResidualExportSpec(
        site_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
        num_future_frames=10,
        cartesian_dim=3,
        context_dim=930,
        output_dim=64,
    )
    expected_spec = json.loads(json.dumps({
        name: getattr(spec, name) for name in spec.__dataclass_fields__
    }))
    if manifest.get("spec") != expected_spec:
        raise AssertionError("export manifest spec mismatch")
    parity = verify_sonic_actor_residual_onnx(
        checkpoint_path=checkpoint,
        onnx_path=onnx_path,
        spec=spec,
        runtime="onnxruntime",
    )
    if parity.get("runtime") != "onnxruntime.InferenceSession":
        raise AssertionError("independent ONNX parity did not use ONNX Runtime")
    if parity.get("runtime_version") != PHASE5_ACCEPTED_ONNXRUNTIME_VERSION:
        raise AssertionError("independent ONNX Runtime version mismatch")
    if parity.get("providers") != ["CPUExecutionProvider"]:
        raise AssertionError("independent ONNX Runtime provider mismatch")
    recorded_parity = json.loads(
        (run_root / "export/parity.json").read_text(encoding="utf-8")
    )
    for key, expected in (
        ("runtime", "onnxruntime.InferenceSession"),
        ("runtime_version", PHASE5_ACCEPTED_ONNXRUNTIME_VERSION),
        ("expected_runtime_version", PHASE5_ACCEPTED_ONNXRUNTIME_VERSION),
        ("providers", ["CPUExecutionProvider"]),
        ("onnx_checker", True),
        ("hard_off_exact", True),
        ("zero_compliance_exact", True),
        ("mixed_row_exact", True),
    ):
        if recorded_parity.get(key) != expected:
            raise AssertionError(f"recorded ONNX Runtime parity mismatch: {key}")
    if not all(
        parity[name]
        for name in (
            "onnx_checker",
            "hard_off_exact",
            "zero_compliance_exact",
            "mixed_row_exact",
        )
    ):
        raise AssertionError("independent ONNX parity failed")

    summaries = {
        mode: json.loads((run_root / mode / "rollout.json").read_text(encoding="utf-8"))
        for mode in ("stiff", "compliant")
    }
    traces = {"stiff": stiff, "compliant": compliant}
    for mode, summary in summaries.items():
        _require_rollout_checkpoint_provenance(
            summary,
            mode=mode,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
        trace = traces[mode]
        expected_frame = {
            "kind": trace.local_frame.kind.value,
            "anchor": trace.local_frame.anchor,
            "rotation": trace.local_frame.rotation.value,
        }
        if summary.get("common_frame") != expected_frame:
            raise AssertionError(f"{mode} structured frame provenance mismatch")
        body_contract = summary.get("tracking_body_contract", {})
        if (
            body_contract.get("source") != "sonic_release_motion_body_names"
            or body_contract.get("count") != len(SONIC_RELEASE_TRACKING_BODY_NAMES)
            or body_contract.get("ordered_names")
            != list(SONIC_RELEASE_TRACKING_BODY_NAMES)
            or trace.body_names != SONIC_RELEASE_TRACKING_BODY_NAMES
        ):
            raise AssertionError(f"{mode} release 14-body provenance mismatch")
        index_contract = summary.get("site_index_contract", {})
        if index_contract.get("ordered_names") != list(trace.site_names):
            raise AssertionError(f"{mode} ordered site provenance mismatch")
        for index_space in ("reference_indices", "articulation_indices"):
            indices = index_contract.get(index_space)
            if (
                not isinstance(indices, list)
                or len(indices) != len(trace.site_names)
                or any(type(index) is not int or index < 0 for index in indices)
                or len(indices) != len(set(indices))
            ):
                raise AssertionError(f"{mode} {index_space} provenance is invalid")
    if summaries["stiff"]["common_frame"] != summaries["compliant"]["common_frame"]:
        raise AssertionError("paired common-frame provenance mismatch")
    if (
        summaries["stiff"]["site_index_contract"]
        != summaries["compliant"]["site_index_contract"]
    ):
        raise AssertionError("paired site-index provenance mismatch")
    if summaries["stiff"].get("policy_semantics") != "matched_force_release_zero_residual":
        raise AssertionError("stiff policy comparison semantics mismatch")
    if summaries["stiff"].get("peak_latent_residual") != 0.0:
        raise AssertionError("stiff residual was not exact zero")
    if summaries["compliant"].get("policy_semantics") != "matched_force_trained_residual":
        raise AssertionError("compliant policy comparison semantics mismatch")
    if summaries["compliant"].get("peak_latent_residual", 0.0) <= 0.0:
        raise AssertionError("compliant residual did not activate")
    for mode in summaries:
        log_path = run_root / f"{mode}.log"
        if "CHIP_PHASE5_ROLLOUT_PASS" not in log_path.read_text(encoding="utf-8"):
            raise AssertionError(f"rollout marker missing: {mode}")
    workflow_bytes = _bounded_artifact_bytes(run_root)
    print(
        "CHIP_PHASE5_INDEPENDENT_AUDIT_PASS",
        f"aligned_frames={result.aligned_frames}",
        f"yield_mean_m={result.compliance_response.displacement_mean_m:.9g}",
        f"onnx_max_error={parity['maximum_absolute_error']:.9g}",
        f"workflow_bytes={workflow_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
