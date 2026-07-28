# CHIP compliance finetune handoff

## Resume here

- Task state: **COMPLETE; Phase 6 PASSED on 2026-07-28**. All six phases pass
  their documented engineering gates. The short rollout remains chain evidence,
  not a converged compliance-performance claim.
- Implemented: tracker-neutral Cartesian contracts and metrics, the SONIC/
  IsaacLab force adapter, a zero-initialized hard-gated post-FSQ actor residual,
  an independent privileged critic residual, strict official-checkpoint
  migration, residual-only 5+1-step PPO smoke, paired 300-frame evaluation,
  and a separate dynamic residual ONNX.
- Not implemented or not demonstrated: converged compliance, multiple-motion/
  multiple-seed tracking quality, arbitrary-object robustness, rotational
  compliance, a 14-site trained checkpoint, or hardware-certified force
  safety. The accepted checkpoint is dual-wrist even though the portable
  tensors and evaluator accept caller-defined sites.
- Remaining Phase-6 gates: none. The exact cache was removed, the complete
  current structural/hygiene audit passed, and the compatibility-NVML command
  emitted `CHIP_PHASE6_FINAL_AUDIT_PASS`. Rerun the full matrix after any code,
  artifact, ref, driver, or environment change.
- Preserve both accepted evidence roots exactly. Use a new nonexistent child
  for every rerun; never overwrite `phase4_acceptance_resume_fix` or
  `phase5_acceptance`.

The remote branch contains the code and this task record. Official assets,
accepted training/evaluation runs, and binaries are intentionally outside Git;
their paths and hashes below are the provenance boundary. Read `status.md`,
then execute the complete Phase 6 from `test_matrix.md` when revalidating.

The final repository audit uses two independent boundaries: the official
baseline proves the feature stayed additive, while accepted Phase-5 commit
`c925a0da115d1d6e0cc296c4a94b00a57c6461b8` pins every already-validated
core/adapter/training/export/evaluation file.  Only the ten explicit Phase-6
entrypoint/help/audit/task paths may differ after that commit.

## Delivered boundary

This branch is an additive, opt-in CHIP-style compliance extension of official
SONIC commit `4141c34280ab`.  It leaves the released experiment, generic PPO
trainer, universal-token module, observations/rewards, and deployment models
unchanged.  The implementation has three explicit boundaries:

1. `gear_sonic/compliance_control/core` is tracker-neutral Torch code for
   structured Cartesian frames, hindsight targets, schedules/damping, residual
   conditioning, aligned traces, and metrics.  It accepts arbitrary ordered
   site/body counts and contains no G1/Isaac Lab index table.
2. `gear_sonic/compliance_control/postprocess` owns bounded NPZ/JSON I/O with
   pickle disabled.  Any tracker that emits `AlignedTrackingTrace` can reuse the
   evaluator and report format.
3. `gear_sonic/compliance_control/adapters/sonic` and the derived Hydra configs
   own SONIC/G1 name resolution, Isaac Lab force application, checkpoint
   migration, actor/critic residuals, and the separate residual ONNX export.

To move this feature to another universal tracker, keep layers 1-2 unchanged,
implement a thin adapter that resolves separate reference/articulation indices,
emits the same structured-frame tensors and trace, and composes the optional
residual at the tracker's latent/action boundary.  Do not copy SONIC's body or
DOF integer indices.  Site names are semantic and ordered; the accepted SONIC
checkpoint uses two wrists, while the evaluator itself supports all 14 release
bodies or another tracker-defined set.

## Pinned inputs and environment

- Official checkpoint: `last.pt`, Hugging Face revision `7c90a56c`, step 41550,
  SHA-256
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
- Official config SHA-256:
  `f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7`.
- Robot-motion sample SHA-256:
  `005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b`.
- Training/simulator interpreter:
  `/home/lab/miniconda3/envs/sonic_backup/bin/python`.
- Accepted ONNX Runtime interpreter:
  `/home/lab/miniconda3/envs/sonic/bin/python`, ONNX Runtime 1.25.0,
  `CPUExecutionProvider` only.
- GPU commands use the extracted `580.159.03` compatibility libraries recorded
  verbatim in `test_matrix.md`; this is required while host kernel/userspace
  NVIDIA packages remain mismatched.

The exact asset and run locations are CLI arguments to the workflows/audits,
not package globals.  The binaries stay under the central
`compliance_control` artifact root and are never committed.

## Accepted evidence

Phase 4's immutable workflow is
`compliance_control/runs/chip/phase4_acceptance_resume_fix`:

- release warm-start: five iterations, 23.272 s;
- residual warm-start: five PPO batches, 24.701 s;
- independent strict resume: one new batch to step 6, 17.529 s;
- both trained checkpoints preserve all 55 policy and 17 value release tensors
  byte-for-byte and add only 6 actor plus 6 critic residual tensors;
- the optimizer owns exactly 12 tensors/770753 scalars; every residual tensor
  has finite nonzero gradient history and both sites have true force exposure;
- step-6 checkpoint SHA-256:
  `71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02`.

Phase 5's immutable workflow is
`compliance_control/runs/chip/phase5_acceptance`:

- 300 exactly aligned matched-force frames, 289 exposed frames per wrist,
  success/fall `1/0`, and 5 N peak applied force;
- stiff versus compliant upper-endpoint MPJPE:
  `0.03238153 m` versus `0.03208341 m`;
- compliant global/local MPJPE: `0.01717841/0.01112965 m`;
- compliant upper-endpoint orientation RMSE: `0.19856526 rad`;
- paired world-frame displacement mean/max:
  `0.00131442/0.00410081 m`;
- aggregate signed displacement along force: `0.00001021 m`; it is diagnostic
  and has no pass threshold;
- residual ONNX SHA-256:
  `a4ccbc9e216dd97fe5181a12f5ded7a9e544c1a477fd114c909b8564bc83e2f3`,
  with dynamic `(B,S)`, real-ORT parity, exact hard-off/zero-compliance output,
  and maximum absolute parity error `5.82076609e-10`.

These values pass the predefined tracking/regression gates and prove that the
residual-to-simulator/export chain activates.  The model saw only six PPO
batches and the evaluation is one 300-frame deterministic motion; the evidence
does not establish converged compliance, task-space impedance/admittance,
contact-force regulation, robustness across motions/objects, or 14-site
training quality.

## Output and deployment contract

The Phase-5 residual ONNX has explicit float32 inputs
`compliance_target [B,S,60]`, `compliance_command [B,S,9]`, and
`actor_context [B,S,930]`, and output `latent_residual [B,S,64]`.  Deployment
adds it to the release encoder latent before the unchanged release decoder.
Hard-off must bypass/zero the residual exactly; privileged applied force is a
critic-only training signal and is never an actor input.

The accepted Phase-5 directory has exactly 14 files, 3 directories, no
symlinks, and 1655744 bytes.  Phase 4 has exactly one intentional internal
resume symlink to its own step-5 checkpoint; no other artifact/source symlink is
accepted.  Complete location-independent tree digests, capacities, checkpoint
semantics, release invariants, refs, and idle-process gates are enforced by
`artifacts/phase6_final_audit.py`.

## Reproduction and cleanup

Use the exact commands in `test_matrix.md`.  Every training/evaluation rerun
must choose a new, nonexistent strict child of
`compliance_control/runs/chip`; the runners refuse collisions and out-of-root
paths.  Never reuse, edit, or delete `phase4_acceptance_resume_fix` or
`phase5_acceptance`, because their content hashes are final evidence.

Failed or exploratory runs may be removed only by naming their exact child
directory after confirming it is not one of the two accepted roots.  No broad
glob, repository-root deletion, or automatic cleanup is part of this handoff.
Run with `PYTHONDONTWRITEBYTECODE=1`; the final audit rejects cache and temporary
files in source/task scope and bounds every accepted log/tree.

## Phase-6 validation state

The final expanded regression passed 136 tests in both interpreters (39 expected
portable dependency skips; four expected resolved CUDA skips), all eight help
gates, both no-write dry runs, real ONNX Runtime parity, and independent Phase-5
metric recomputation.  The three AppLauncher thin entrypoints received only a
parser/exit-code repair; focused tests pin their accepted runtime `main()` ASTs
unchanged and preserve missing-required-argument exit code 2.

The accepted-Phase-5-head boundary still permits exactly the ten declared
Phase-6 paths and rejects any Phase-6 core/adapter/training/export/evaluation
edit. The unintended audit bytecode cache was removed and the current complete
source/task hygiene pass found no cache or temporary file.

During the paused handoff, the protected main refs advanced from `345c3f4` to
`6d6d8ae` through one externally published central-documentation commit. The
immutable original snapshot was not rewritten. The final audit accepts only
that exact three-ref, one-direct-commit fast-forward and exactly twelve `A`
paths under the central `compliance_control` documentation/test manifest;
partial, future, multi-commit, modified, extra-path, or any other ref movement
fails. The compatibility-NVML final audit then resolved the RTX 4090 through
NVIDIA 580.159.03, found no compute application or CHIP/Isaac process, reloaded
both immutable evidence roots, and emitted `CHIP_PHASE6_FINAL_AUDIT_PASS`.
