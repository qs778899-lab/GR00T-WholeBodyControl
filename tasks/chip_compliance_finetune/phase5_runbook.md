# CHIP Phase-5 evaluation and export runbook

## Scope and architecture

Phase 5 evaluates the accepted two-wrist residual checkpoint and exports only
that optional residual. It does not finetune, rewrite release ONNX files, or
change SONIC's release policy. The layers are intentionally portable:

- `compliance_control/core/evaluation.py`: tracker-neutral trace, metrics,
  alignment, and gates; Torch only.
- `compliance_control/postprocess/evaluation_io.py`: bounded NPZ/JSON storage;
  no pickle.
- `compliance_control/adapters/sonic/export.py`: SONIC checkpoint extraction and
  residual ONNX contract.
- `compliance_control/adapters/sonic/contracts.py`: SONIC-only ordered release
  body contract; never imported by the tracker-neutral core.
- `run_chip_phase5_rollout.py`: thin Isaac Lab trace collector.
- `run_chip_phase5_eval_export.py`: serial workflow orchestration.
- `verify_chip_phase5_onnx.py`: configurable-interpreter ONNX Runtime check.
- `audit_chip_phase5.py`: fresh-process independent recomputation.

Another universal tracker should implement only a collector that emits
`AlignedTrackingTrace`; it can reuse all evaluation and storage code unchanged.
The trace supports arbitrary ordered sites. The accepted checkpoint and ONNX
spec are still two-wrist because that is the layout trained in Phase 4; using
all 14 SONIC key bodies requires a matching 14-site residual checkpoint/spec,
not changes to the metric core.

## Comparator contract

`stiff` and `compliant` are matched-force rollouts from the same trained
checkpoint. They use the same motion/reference frame, seed, episode clock,
external force, site mask, and compliance schedule. In `stiff`, only the actor
`compliance_command` is zeroed, so the post-FSQ residual is bit-exact zero and
the actor is release-equivalent. External force remains active. Therefore the
stiff trace is not the unforced Phase-4 stiff log.

Integer sample/episode/motion/reference-frame keys and boolean gates must match
exactly. Time, reference positions, normalized reference quaternions, force, and
compliance must match within the recorded numerical tolerance. No interpolation,
time shifting, nearest-frame lookup, or force substitution is allowed. A fall
keeps a fixed-size trace, marks the suffix invalid, and restricts paired metrics
to the common valid prefix.

Sample `k` records the state immediately before `env.step` transition `k`. If
that transition is a non-timeout termination, sample `k` remains valid,
`termination_sample=k`, and samples `k+1..299` remain permanently invalid even
if the simulator auto-resets. Such a trace has success/fall `0/1`. Success
requires all 300 samples valid and no early termination. A later auto-reset
timeout after a fall is treated fail-closed by the collector rather than being
silently paired as a new episode.

Trace metadata persists the structured local-frame kind, semantic anchor, and
rotation (`heading_local`, `pelvis`, `yaw_only` for this smoke). Each SONIC
rollout summary also persists ordered site names and the separately resolved
reference and articulation index lists. These are audit provenance only: the
portable metric layer remains name-ordered and never consumes robot indices.
All trace tensors must share one device, and all Cartesian/quaternion tensors
must share one floating dtype, so a new tracker adapter fails at construction
rather than later during indexing. The core allows arbitrary body counts; the
SONIC acceptance boundary separately proves its ordered names are identical to
the complete 14-body `sonic_release` contract.

## Metric definitions

- Global/local MPJPE: mean Euclidean error over valid frames and all configured
  tracking bodies, in world and declared anchor-local coordinates.
- Endpoint position: per-site Euclidean actual-to-reference error, reported as
  RMSE and P95 for all, exposed, and unexposed valid frames; aggregate mean is
  also recorded.
- Endpoint orientation: finite unit `wxyz` reference and actual quaternions,
  selected in separate reference/articulation index spaces. Error is the
  sign-invariant geodesic angle `2*acos(abs(dot(q_ref, q_actual)))`, in radians.
  RMSE/P95 are reported per site for all/exposed/unexposed frames, plus overall
  and exposed RMSE. Overall and per-site all-frame RMSE/P95 regressions are
  gated; exposed/unexposed splits are diagnostic in this phase.
- True compliant yielding: `compliant_actual_site - stiff_actual_site` on
  matched exposed frames. Reports displacement norm and signed projection along
  force, aggregate and per site. Both `actual_site_positions_w` and
  `force_on_robot_w` are world-frame quantities. This is not
  actual-to-reference tracking error, object/contact/task-frame deflection, or
  a measurement of contact-force regulation.
- Exposure: valid + command enabled + site selected + nonzero compliance +
  nonzero applied force.
- Steady force: mean force magnitude over the final 20 percent in time of every
  contiguous per-site exposure pulse. Peak force is reported separately.
- Outcome: valid frames, horizon success, and fall rate.

Acceptance requires at least 200 common frames and 20 exposed frames per site;
position/orientation upper-endpoint regressions no greater than 0.05 m/0.25 rad;
global/local MPJPE regressions no greater than 0.03 m; compliant success/fall of
1/0; and peak applied force in `[1, 30]` N. Mean paired displacement must be at
least `1e-6` m solely to prove that the residual-to-simulator chain activates.
It proves only that the two policies' world-frame site trajectories differ; it
does not prove task-space impedance/admittance, contact-force regulation, or
along-force compliant performance. The signed along-force projection is
reported without a pass threshold. The six-batch checkpoint is a
chain-validation smoke and must not be presented as evidence of
compliant-control performance.

## ONNX contract

The separate model has explicit float32 inputs:

- `compliance_target [batch, sequence, 60]`: 10 future frames x 2 wrists x xyz,
  in `heading_local:pelvis`.
- `compliance_command [batch, sequence, 9]`: enable, two ordered site-mask
  values, then two xyz compliance vectors in m/N.
- `actor_context [batch, sequence, 930]`: release actor context.

It returns `latent_residual [batch, sequence, 64]`. Deployment composes
`release_decoder_latent = release_encoder_latent + latent_residual`; the release
encoder and decoder remain unchanged. Verification checks ONNX validity,
PyTorch parity at `(1,1)` and `(2,4)`, exact zero output under hard-off even when
unused operands contain NaNs, exact zero under enabled zero-compliance input,
and mixed active/off/no-site/zero-selected-compliance rows in one `B x S` tensor
without batch or timestep broadcasting.

Portable tests may use `onnx.reference.ReferenceEvaluator` when ONNX Runtime is
absent, and label that runtime truthfully. Acceptance is stricter: the workflow
and fresh audit both require `onnxruntime.InferenceSession` 1.25.0 with exactly
`CPUExecutionProvider`, using the configurable `--onnxruntime-python` boundary.
The pinned interpreter is `/home/lab/miniconda3/envs/sonic/bin/python`; the GPU
rollouts remain in `sonic_backup`. The accepted `parity.json` records runtime,
version, providers, checker status, exact hard-off checks, and numeric error.

## Run and outputs

Use the exact commands and compatibility environment in `test_matrix.md`. The
acceptance path must not exist before launch. The workflow runs one-environment
stiff and compliant jobs serially, then writes:

`--runs-root` is an explicit portable safety boundary; `--run-root` must be a
new strict child. The default is `compliance_control/runs/chip` under the active
repository, while the worktree acceptance command supplies the shared artifact
root explicitly. Simulator runtime output is contained under each mode's
`runtime/` directory and is included in capacity and symlink checks.

Before resolving paths, the standalone export and rollout entrypoints reject an
existing or broken final-component symlink for ONNX, manifest, trace, trace
metadata, rollout summary, or runtime output. This prevents `Path.resolve()`
from turning a requested symlink into a different publication target. The
independent audit separately requires canonical workflow `run_root`,
`runs_root`, and checkpoint paths to match its arguments, and requires both
rollout summaries to contain that exact checkpoint path and SHA-256.

```text
phase5_acceptance/
  workflow.json
  paired_metrics.json
  stiff.log
  compliant.log
  stiff/{trace.npz,trace.json,rollout.json,runtime/}
  compliant/{trace.npz,trace.json,rollout.json,runtime/}
  export/{compliance_residual.onnx,compliance_residual.json,parity.json,parity.log}
```

Run the independent audit only after the workflow marker passes. Keep the whole
directory below 500 MB and logs below 64 MB. Do not edit or delete the accepted
Phase-4 checkpoint, official assets, or release ONNX files. Trace loading also
checks NPZ ZIP headers before allocation and rejects more than 64 MB of
uncompressed tensor data; metadata is capped at 1 MB and pickle is disabled.
