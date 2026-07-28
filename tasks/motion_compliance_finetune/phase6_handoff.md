# Motion-compliance finetune handoff

## Resume here

Task state is **Phase 6 / IN_PROGRESS**. Phases 1 through 5 are accepted.
Execution was paused by the user on 2026-07-28. Phase 6 has the portable CPU
evaluation contract plus a thin SONIC collector/final validator, but it does
not yet have the required fresh GPU/simulator performance evidence. Read
`status.md`, this checkpoint, and then run only Phase 6 of `test_matrix.md`;
do not infer completion from Phase-4 training smoke, C++ deployment tests, or
synthetic CPU traces.

## Paused checkpoint (2026-07-28)

Stop boundary: no GPU training, 4096-environment benchmark, or six-protocol
IsaacLab collection was launched after the user requested the pause. Phase 6
remains `IN_PROGRESS`; it is not `PASSED`, and the task is not `COMPLETE`.

New code present at this checkpoint:

- `adapters/sonic/evaluation.py` owns the concrete SONIC snapshot mapping,
  shared reference-torso coordinate frame, natural-timeout observer, protocol
  roles, checkpoint/action-byte evidence, actual composer-row evidence, and
  bounded lifecycle collector.
- `adapters/sonic/evaluation_recorder.py` is the thin IsaacLab recorder bridge.
- `phase6_collect_sonic_trace.py` runs one full audited motion clip and refuses
  publication unless the natural timeout occurs exactly on the final expected
  50 Hz step. It pins G1-only encoder selection, release 14-point order, plane
  terrain, eval terminations, reset-only events, 10 N / 0.05 m / 200 N/m
  stimulus parameters, official-versus-step-6 checkpoint roles, protocol
  gates, and exact post-timeout owned-wrench cleanup.
- `phase6_evaluate_aligned_traces.py` remains tracker-neutral. It records each
  NPZ full-file SHA from the same `O_NOFOLLOW` descriptor used for schema/ZIP/
  NumPy decoding and applies fixed tracking, force, yield, measured-yield, and
  inactive-hand cross-coupling criteria.
- `phase6_validate_sonic_collection_reports.py` is a separate SONIC-specific
  final gate. It loads the six trace files safely, binds collection/observed/
  paired hashes, recomputes the complete portable report from those traces,
  and pins the six protocols, wrist roles, motion/checkpoint hashes, 50 Hz
  timing, environment, actual composer/reset evidence, and baseline/off exact
  action parity. JSON comparisons are type-aware, so `false` cannot substitute
  for integer zero.

Latest final-scoped checks before documentation:

- evaluation + SONIC collector/validator CPU tests: `38 passed in 1.46s`;
  independent review reran `38 passed in 1.41s`;
- related CLI help, AST parsing, and `git diff --check`: passed;
- repository-local Python/pytest caches created by compile checks were removed.

Earlier in the same resume, before the final collector patches, the Phase-5
deployment suite (`33 passed`), official residual contract, trainer help/config,
generic/SONIC C++ ORT smoke, production target build/help/CLI acceptance,
accepted artifact revalidation, pinned hashes, and immutable release diff all
passed. Treat those as useful continuity checks, not a final matrix-item-1 run
on the paused code.

Two explicit P1 items must be closed before starting formal real collection:

1. Pin the termination/event config provenance beyond term names: exact
   function targets, reset mode, thresholds, body names, command names, and
   other parameters. Current source config is correct, but a future same-name
   function/parameter change is not yet fail-closed in the final validator.
2. In at least one nonzero-force Phase-6 interaction, invoke the configured
   `motion_compliance_reset` event and prove it clears command and composer
   rows. Current Phase-6 evidence uses explicit post-timeout cleanup; the real
   reset path was proven earlier by Phase 2, not yet by the Phase-6 collector.

After these two small hardening changes, rerun the final focused suite and then
resume at Phase-6 matrix item 1. Do not reuse or overwrite any partial output.

## What is implemented

The branch is an additive, opt-in extension of official SONIC commit
`4141c34280abb67c82e115342a8720f4a83d750d`. The standard release experiment,
generic PPO trainer, released encoder/decoder/config, and default deployment
path remain unchanged.

### Portable control and training boundary

- `gear_sonic/compliance_control/core` owns the structured enable/threshold/Kp
  condition, fixed-shape force schedule, reset-safe state, force/target math,
  and residual composition. It has no fixed G1 body or DOF indices.
- The operational disabled path consumes neither global CPU/CUDA RNG nor a
  dynamic CUDA due-ID tensor. The 4096-environment profiler rejects
  `aten::nonzero` and `aten::_local_scalar_dense` in the fixed-shape scheduler.
- `gear_sonic/compliance_control/adapters/sonic` owns SONIC reference versus
  articulation index resolution, frames, IsaacLab wrench writing, the concrete
  two-wrist sites, the 994-D release action context, the 3-D condition, and the
  29-D IsaacLab/BFS action layout.
- Actor and value residuals are independent and zero initialized. The released
  policy/value tensors retain their original shapes and bytes; actor-visible
  inputs exclude applied force, while the critic can use privileged force.
- The official release checkpoint was initialized into the same-shape residual
  schema, trained for five PPO iterations, and independently resumed for one
  more step. Exactly the two residual heads are trainable.

### Portable deployment boundary

- `gear_sonic/compliance_control/deployment` provides versioned schema,
  atomic ONNX export, artifact validation, and a Python ORT overlay. It accepts
  arbitrary ordered context segments and an arbitrary non-empty set of
  caller-owned release artifact pins.
- `gear_sonic_deploy/src/motion_compliance` provides the equivalent generic C++
  ORT runtime. Portable Python/C++ source contains no SONIC, G1, IsaacLab,
  MuJoCo, wrist, 14-keypoint, 29-action, or 994-context constant.
- `adapters/sonic/deployment.py`, the opt-in deployment YAML, and the small
  `g1_deploy_onnx_ref` hook own concrete SONIC assembly. The hook composes
  `release_action + bounded_delta` before the existing IsaacLab/BFS to
  MuJoCo/DFS remap.
- Host disabled mode does not read the residual artifact or create an ORT
  session. All-off and mixed rows preserve exact release action; disabled rows
  bypass NaN-bearing unused context/condition values.
- Runtime operator values are validated before middleware initialization. The
  first two wrist controls currently OR into one global binary overlay gate;
  either positive enables the residual for the deployment batch.

### Portable Phase-6 evaluation boundary

- `gear_sonic/compliance_control/evaluation` defines a tracker-neutral aligned
  trace and metrics. Pairing is exact on ordered motion ID, sequence ID, seed,
  frame, timestamp bytes, site layout, point layout, dtypes, reference point
  bytes, and original endpoint pose bytes.
- The trace separates original/selected/measured endpoints, original/measured
  orientations, global/local tracking points, force, enable/site masks,
  terminal/success/fall, and reset snapshots.
- Metrics report each site's endpoint RMSE/P95, quaternion orientation error,
  force, reference yield, measured yield along actual force, inactive-hand
  cross-coupling, local/global MPJPE, success/fall/reset/finiteness, including
  active-contact windows. Trials require zero falls and full success.
- Bounded atomic NPZ/JSON I/O refuses overwrite, disables pickle, validates ZIP
  members/sizes, and decodes through one `O_NOFOLLOW` file descriptor.
- The thin evaluator takes caller-owned endpoint roles and remains free of
  SONIC/G1/IsaacLab vocabulary. The implemented collector and final validator
  are separate thin SONIC layers and are the only evaluation components that
  know concrete body names, checkpoint roles, or simulator state.

## Accepted evidence and provenance

Pinned official input:

- checkpoint: central `official_assets/sonic_release/last.pt`
- Hugging Face revision:
  `7c90a56cfe04788c4f041daeef5b1e12930675ad`, step 41550
- checkpoint SHA-256:
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`

Accepted residual artifacts:

- step-6 checkpoint SHA-256:
  `42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba`
- action-residual ONNX SHA-256:
  `9e7a30ae8485eb153b63db81575c9b0fd24522523510560ed5d6292652568a81`
- metadata payload SHA-256:
  `e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5`
- artifact schema: `universal-tracker.action-residual.onnx.v1`, ONNX opset 17

Accepted test evidence:

- Phase 1–4 plus evaluation: `101 passed, 1 skipped in 25.00s` in
  `sonic_backup`.
- Deployment/export: `33 passed, 96 warnings in 2.75s` in the ORT-equipped
  `sonic` environment.
- PyTorch/ORT dynamic-shape maximum absolute error:
  `7.450580596923828e-09`; hard-off is exact.
- System-ORT C++ generic and SONIC smoke passed, the complete production target
  configured/linked, and eight invalid CLI values were rejected before DDS.
- Paused-code focused evaluation/collector/final-validator suite: `38 passed`
  with strict lifecycle/alignment/IO/provenance and adversarial evidence
  negatives. The older complete split regression above predates the final
  collector additions and must be rerun as Phase-6 matrix item 1.

Git phase history before this handoff:

- `fa39575` Phase 1 portable core
- `6191ed7` Phase 2 IsaacLab force adapter
- `586d6e5`, `26638c2`, `a6b70e2` Phase 3 and synchronization/RNG fixes
- `4599fc9`, `108d228` accepted Phase-4 isolated same-shape residual finetune
- older Phase-4 attempts remain in history but are not accepted evidence

Official checkpoints, motion data, run directories, ONNX binaries, and build
outputs remain in Git-ignored central artifact storage. Their hashes—not their
location alone—are the provenance contract.

## What is not implemented or not proven

- No fresh post-restart Phase-6 16-environment/5-iteration smoke or recorded
  FPS/GPU-memory result has been accepted.
- No accepted 4096-environment host-off versus enabled scheduler measurement
  exists yet. The benchmark CLI is implemented and only its CPU help gate ran.
- No real paired Phase-6 baseline, overlay-off, enabled/no-contact, single-left,
  single-right, or simultaneous two-wrist trace exists yet. The collector and
  validator exist, but their formal GPU commands have not been run.
- Therefore the branch cannot yet claim the required off-mode hand endpoint
  regression, no-contact parity, compliant yield, cross-coupling, fall/reset,
  or final upper-limb tracking quality.
- Active trials will report endpoint/orientation/local-global MPJPE, but the
  final acceptance still needs an explicit human review (and, if agreed before
  collection, fixed upper limits) for active left/right wrist endpoint,
  orientation, and whole-body tracking accuracy. This is essential to the
  user requirement that compliance must not displace tracking accuracy as the
  first priority.
- The final validator does not yet pin termination/event function targets and
  detailed parameters, and the Phase-6 collector has not yet demonstrated the
  configured reset event immediately after a nonzero force. These are the two
  P1 hardening items listed in the paused checkpoint.
- The trained artifact uses two wrists. Arbitrary-site tensor/runtime support
  does not prove that this checkpoint supports another site count or all 14
  SONIC tracking bodies.
- Compliance is translational only. There is no rotational compliance model.
- The production UI has one global binary residual gate, not independent
  left/right enable controls. Per-site masks remain available in training and
  evaluation contracts.
- Thresholds, wrench clamps, and the residual bound are engineering guards;
  they are not a certified hardware force limit or a stability proof.

## Exact Phase-6 continuation order

1. Confirm `status.md` still says Phase 6 `IN_PROGRESS`; inspect `git status`
   and do not mix a new algorithm change into the evidence run.
2. Close only the two paused-checkpoint P1 evidence items, rerun the 38-test
   focused suite, and record the new result. Do not tune the policy or force
   algorithm in this hardening step.
3. After any machine restart, revalidate the exact NVIDIA 580.159 compatibility
   userspace, `sonic_backup`, official asset hashes, and idle trainer/simulator
   state. Run the full Phase 1–5 regression plus all Phase-6 CPU tests and the
   Phase-2/3 real smoke commands required by matrix item 1.
4. Run the prescribed 16-environment, one-audited-robot-PKL, five-iteration,
   `use_wandb=false` smoke and record FPS and GPU memory.
5. From a new output path, run the exact matrix item-3 command for
   `phase6_scheduler_benchmark.py --num-envs 4096 --num-sites 2`. Preserve its
   scheduler-only label; it is not end-to-end policy latency.
6. Use the existing `phase6_collect_sonic_trace.py` in six fresh directories:
   baseline with the official checkpoint; overlay-off, enabled/no-contact,
   single-left, single-right, and simultaneous with the accepted step-6
   checkpoint. Use seed 0, the audited PKL/key, the same SMPL directory,
   `--max-steps 2500`, 10 N, and 0.05 m. Do not use a shortened horizon.
7. Run `phase6_evaluate_aligned_traces.py` with both wrist endpoint roles.
   Require each interaction site to exceed configured force/yield minima, each
   inactive site to stay below tolerance, one first-row reset snapshot and one
   final terminal per sequence, and no stale wrench or non-finite value.
8. Run `phase6_validate_sonic_collection_reports.py` over the paired report and
   all six summaries. It must recompute and match the portable report, prove
   exact baseline/off action parity, and emit a passing final SONIC report.
9. Enforce the matrix thresholds: success-rate drop at most one percentage
   point; local MPJPE regression at most 3 mm or 10%, whichever is larger;
   off-mode left/right hand RMSE regression at most 5 mm; no-contact remains in
   the same range as off. Review active left/right endpoint, wrist orientation,
   and whole-body errors explicitly before accepting tracking quality.
10. Run final output-size/duplicate/cache, portable-vocabulary,
   `git diff --check`, and complete matrix gates. Only then mark Phase 6
   `PASSED` and the
   task `COMPLETE`.

Never overwrite an existing evidence path. Use a new strict run directory for
every GPU/simulator attempt, keep logs bounded, and retain any failure report
needed to explain a metric difference.

## Known paths and collection command contract

Verified local inputs at the pause boundary:

- robot PKL:
  `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl`
  (`005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b`)
- SMPL directory:
  `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered`
- baseline checkpoint:
  `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt`
- overlay checkpoint:
  `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix/last.pt`
- NVIDIA compatibility directory:
  `/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu`

Use a new `PHASE6_RUN_ROOT` under
`compliance_control/runs/motion/`; never point it at an existing directory.
Each collector call uses the NVIDIA compatibility `LD_LIBRARY_PATH`,
`LD_PRELOAD`, and `VK_ICD_FILENAMES` values written verbatim in Phase-6 matrix
items 2 and 3, plus:

```text
/home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  tasks/motion_compliance_finetune/artifacts/phase6_collect_sonic_trace.py \
  --trial-name <name> --protocol <mode> [--active-site <site> ...] \
  --seed 0 --max-steps 2500 --force-threshold-n 10 \
  --reference-offset-common-m 0.05 0 0 \
  --motion-file <robot-PKL-above> --smpl-motion-dir <SMPL-dir-above> \
  --checkpoint <role-specific-checkpoint> \
  --trace <new-trial-dir>/trace.npz --summary <new-trial-dir>/summary.json \
  --headless
```

Required rows are:

| name | mode | checkpoint | active sites |
|---|---|---|---|
| `released_baseline` | `baseline` | official | none |
| `overlay_off` | `off` | step 6 | none |
| `enabled_no_contact` | `no_contact` | step 6 | none |
| `single_left` | `single_site` | step 6 | `left_wrist_yaw_link` |
| `single_right` | `single_site` | step 6 | `right_wrist_yaw_link` |
| `simultaneous` | `multi_site` | step 6 | left then right wrist |

Then run the portable evaluator with six `--trial NAME MODE TRACE ACTIVE`
groups in that order, `--baseline released_baseline`, both repeated
`--endpoint-site` wrist names, and a new paired-report path. Finally run the
SONIC validator with that paired report and six repeated
`--collection-report NAME SUMMARY` groups. Defaults are the fixed formal
criteria; do not relax them during evidence collection.

## Porting to another universal tracker

Keep `core`, `deployment`, and `evaluation` unchanged. Implement one adapter
that owns the target tracker's reference/articulation name mapping, coordinate
transform, actor context segments, action layout, residual insertion point,
operator control mapping, and trace collector. Explicitly verify coordinate
axes/handedness plus joint order at this boundary; do not carry SONIC's BFS
indices into a MuJoCo/URDF DFS action vector. Preserve independent release-file
pins, exact hard-off, and the standard aligned-trace identities as migration
acceptance gates.
