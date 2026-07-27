# Motion-compliance finetune handoff

## Resume here

Task state is **Phase 6 / IN_PROGRESS**. Phases 1 through 5 are accepted.
Phase 6 has a complete portable CPU evaluation contract but does not yet have
the required fresh GPU/simulator performance evidence. Read `status.md`, then
run only Phase 6 of `test_matrix.md`; do not infer completion from Phase-4
training smoke, C++ deployment tests, or synthetic CPU traces.

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
  frame, timestamp bytes, site layout, point layout, and dtypes.
- The trace separates original/selected/measured endpoints, original/measured
  orientations, global/local tracking points, force, enable/site masks,
  terminal/success/fall, and reset snapshots.
- Metrics report each site's endpoint RMSE/P95, quaternion orientation error,
  force, yield, inactive-site cross-coupling, local/global MPJPE, success/fall/
  reset/finiteness, including active-contact windows.
- Bounded atomic NPZ/JSON I/O refuses overwrite, disables pickle, validates ZIP
  members/sizes, and decodes through one `O_NOFOLLOW` file descriptor.
- The thin evaluator takes caller-owned endpoint roles. A future simulator
  collector is the only place that should know concrete SONIC body/log names.

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
- CPU aligned-evaluation suite: 14 focused tests; complete split regression
  above includes its strict lifecycle/alignment/IO negatives.

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
  single-right, or simultaneous two-wrist trace exists yet.
- Therefore the branch cannot yet claim the required off-mode hand endpoint
  regression, no-contact parity, compliant yield, cross-coupling, fall/reset,
  or final upper-limb tracking quality.
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
2. After any machine restart, revalidate the exact NVIDIA 580.159 compatibility
   userspace, `sonic_backup`, official asset hashes, and idle trainer/simulator
   state. Run the full Phase 1–5 regression required by matrix item 1.
3. Run the prescribed 16-environment, one-audited-robot-PKL, five-iteration,
   `use_wandb=false` smoke and record FPS and GPU memory.
4. From a new output path, run the exact matrix item-3 command for
   `phase6_scheduler_benchmark.py --num-envs 4096 --num-sites 2`. Preserve its
   scheduler-only label; it is not end-to-end policy latency.
5. Add/use a thin SONIC simulator collector to emit the standard trace without
   adding concrete body names to `evaluation/`. Collect baseline, overlay-off,
   enabled/no-contact, single-left, single-right, and simultaneous trials with
   identical motion/sequence/seed/frame/timestamp identities.
6. Run `phase6_evaluate_aligned_traces.py` with both wrist endpoint roles.
   Require each interaction site to exceed configured force/yield minima, each
   inactive site to stay below tolerance, one first-row reset snapshot and one
   final terminal per sequence, and no stale wrench or non-finite value.
7. Enforce the matrix thresholds: success-rate drop at most one percentage
   point; local MPJPE regression at most 3 mm or 10%, whichever is larger;
   off-mode left/right hand RMSE regression at most 5 mm; no-contact remains in
   the same range as off.
8. Run final output-size/duplicate/cache, portable-vocabulary,
   `git diff --check`, and complete matrix gates. Only then mark Phase 6
   `PASSED` and the
   task `COMPLETE`.

Never overwrite an existing evidence path. Use a new strict run directory for
every GPU/simulator attempt, keep logs bounded, and retain any failure report
needed to explain a metric difference.

## Porting to another universal tracker

Keep `core`, `deployment`, and `evaluation` unchanged. Implement one adapter
that owns the target tracker's reference/articulation name mapping, coordinate
transform, actor context segments, action layout, residual insertion point,
operator control mapping, and trace collector. Explicitly verify coordinate
axes/handedness plus joint order at this boundary; do not carry SONIC's BFS
indices into a MuJoCo/URDF DFS action vector. Preserve independent release-file
pins, exact hard-off, and the standard aligned-trace identities as migration
acceptance gates.
