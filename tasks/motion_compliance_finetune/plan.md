# Motion-compliance finetune plan

## Objective

Add an optional, policy-level compliant tracking mode to the SONIC robot-motion
encoder path while preserving the released stiff-mode tracking contract.  The
implementation follows the useful interface from `motion_tracking/compliance`
(`enable`, force threshold, and derived stiffness), but remains native to the
NVLabs IsaacLab/SONIC encoder-decoder architecture.

Baseline: NVLabs upstream `main` at
`4141c34280abb67c82e115342a8720f4a83d750d`.

## Non-negotiable contracts

- The existing robot-motion encoder input stays unchanged: 10 future frames of
  29-DoF joint position/velocity command data plus 6D anchor orientation.
- The tracking/reward skeleton and physical force sites are separate concepts.
- Original, compliant, and current site positions must be supplied in one
  adapter-selected common Cartesian frame.  Core outputs use the
  `force_on_robot` sign convention in that frame; frame transforms remain in
  adapters.
- `enable=0` produces an all-zero compliance condition, zero virtual wrench,
  and the unmodified reference exactly.
- The actor receives no privileged contact force.  Applied virtual wrench and
  active-site masks may be critic-only observations and metrics.
- Force threshold is a learned conditioning value, not a certified hardware
  force limit.  It caps the nominal term only; the separately capped tracking
  term can make the summed synthetic site force larger.
- Existing release configs and checkpoints remain usable without compliance.
- New functionality is additive; do not rewrite the large tracking command or
  the existing release experiment config in place.

## Phases

### Phase 1 — Baseline contract and tracker-agnostic core skeleton

- Record the upstream interfaces and the existing orphaned compliance code
  boundary.
- Add a simulator-independent, universal-tracker-agnostic package at
  `gear_sonic/compliance_control/core/`, split by responsibility into
  `schema.py`, `math.py`, `schedule.py`, `reference_modifier.py`, and
  `metrics.py`.
- The core must not import IsaacLab, know G1 body names, or assume a 29-DoF
  action space / 14-point skeleton.  Site counts and tensor layouts are caller
  supplied.
- The core provides:
  - explicit `[enable, enable * threshold, enable * Kp]` encoding;
  - `Kp = threshold / reference_displacement`;
  - per-site reference selection that preserves the original target exactly
    for disabled/inactive sites;
  - the audited `motion_tracking` synthetic-force formula: independently
    clamped nominal `(compliant-original)*Kp` plus tracking correction
    `(compliant-current)*tracking_gain`, with an explicit nominal-only mode;
  - site-count-agnostic scheduling and preservation metrics.
- Support optional future axes such as `[batch, future, sites, 3]`, including
  `[batch, sites]` masks/thresholds broadcast across futures.
- Add pure unit tests, including import-boundary, robot-independence, frame/sign,
  hard-gate, formula, type/device, non-finite, and differentiability tests.
  Do not compose it into an environment yet.

### Phase 2 — Thin SONIC/IsaacLab virtual-force adapter

- Add a thin adapter under `gear_sonic/compliance_control/adapters/sonic/` and
  a small MDP registration module rather than extending the already-large
  `commands.py`.  Only this layer may resolve SONIC/G1 body names, use IsaacLab,
  or translate Hydra config into core schemas.
- Implement persistent per-environment state, independent/simultaneous site
  masks, event timing, threshold sampling, force application/reset, and net
  wrench limiting.
- Resolve the tracking-reference and articulation body indices independently;
  neither index space may be inferred from the other.  Convert reference and
  measured points into an explicit current-anchor common frame, and convert the
  resulting `force_on_robot` vectors into world coordinates for offset-torque
  reconstruction and residual limiting.  Convert the final wrench using each
  body's current quaternion and write it in link-local coordinates at the
  simulator boundary, avoiding stale global-pose caching in the composer.
- Preserve each requested site wrench while a replaceable residual-wrench
  limiter adds anchor compensation so the resulting whole-robot residual is no
  larger than 20 N / 10 Nm.  Site-force synthesis, whole-body limiting, and
  PhysX writing remain separate responsibilities.
- Apply the persistent wrench directly after compliance-command computation at
  every policy step, without adding an RNG-consuming interval event.  Prefer
  IsaacLab's modern `permanent_wrench_composer`; keep the deprecated
  articulation setter only as an isolated feature-detected fallback.  Reset
  events clear both command-owned tensors and the composer before reuse.
- Keep the host-side operational switch `false` by default.  Disabled mode
  skips per-step compliance math and does not touch an already-clean composer;
  disabling after application clears owned composer rows immediately, before
  the next physics step.  Validate static tensor
  contracts outside the CUDA hot path and use adapter-private no-sync kernels
  only after that validation boundary.  Disabled reset/compute must not consume
  global CPU/CUDA RNG; enabled durations and state use a command-owned generator.
  Override the inherited command `compute(dt)` so this added command does not
  compact a dynamic due-ID tensor.  Enabled updates sample fixed-shape
  all-environment candidates and select due rows with a boolean mask; public
  index APIs retain checked validation.
- Default physical sites are resolved by body name in the SONIC adapter; the
  core remains site-count agnostic and the configuration supports more sites
  without changing the policy contract.
- Validate SONIC endpoint body-name/offset metadata in this adapter phase, not
  in the reusable core.
- Add an opt-in command composition under
  `config/manager_env/commands/`; leave the standard command composition
  unchanged.
- Validate the real lifecycle in one headless CUDA environment: 100 disabled
  policy steps with an inactive composer, 100 forced-on/all-site policy steps,
  finite state/composer checks, and an explicit stale-wrench reset assertion.
  Trace the actual bound `command.compute(dt)` with both Torch dispatch and the
  CPU/CUDA profiler and reject `aten::_local_scalar_dense` and `aten::nonzero`.

### Phase 3 — Observation, reward, and experiment composition

- Keep the released policy and critic observation groups exactly 930 and 1645
  columns.  Expose the 3D public compliance condition as a separate actor
  group, and raw threshold/current applied site force/site mask as a separate
  critic-only group of width `1 + 4*S`; `S` follows configured site count.
- Wrap the released actor and critic with independent zero-initialized residual
  heads.  The frozen `g1_dyn` input remains `64+930=994`; the frozen critic and
  its running statistics remain 1645.  Residual heads may read separate
  condition/context but must be selected through a hard per-row gate so an off
  row is byte-exact to release even in a mixed batch.  Bound the action delta,
  isolate residual initialization RNG, freeze release noise, and allowlist
  actor-visible groups before direct or temporal rollout.
- Keep the complete tokenizer subtree and robot-motion encoder term names,
  shapes, order, functions, parameters, and noise unchanged.
- Add a position reward that uses future frame zero and yielded targets only at
  active sites, with current endpoint/reference tensors recomputed locally at
  reward time to respect IsaacLab's reward-before-command-update lifecycle.
  Inactive sites remain bitwise original.  Gate new rewards by the sampled
  enable bit so every off environment adds exactly zero reward.
- Keep orientation on the original reference because Phase 2 has no rotational
  compliance.  Retain per-site selected-position, original-position, and
  orientation errors so one-hand degradation is not hidden by a mean.
- Preserve every released dense reward and add conservative endpoint terms at
  the same scale: position `weight=2.0/std=0.1`, orientation
  `weight=0.5/std=0.4`.
- Add an opt-in `sonic_release_motion_compliance.yaml` experiment that defaults
  physically off.  Verify resolved off-mode behavior against `sonic_release`
  and in one real manager environment, including official same-shape checkpoint
  loading, mixed off/on gates, residual-only gradients, identical interval-event
  structure, and global-RNG contracts.

### Phase 4 — Same-shape residual initialization and finetune workflow

- Compose a schema-v2 initialization checkpoint from the pinned release and
  the instantiated Phase-3 target.  Keep all 55 official policy tensors and
  all 17 official value tensors byte-exact at their released shapes: `g1_dyn`
  remains 994, critic/RMS remain 1645, and `std` remains unchanged.  Add only
  the six action-residual and six value-residual tensors initialized by the
  target model.  The separate residual contexts are 997 and 1657 for two
  sites; these are not release-network input expansions.
- Treat every schema-v1 expanded artifact and every old Phase-4 output as
  invalid evidence.  A non-resume load accepts only the schema-v2 residual
  initialization artifact; a resume load accepts only a complete branch
  checkpoint and remains strict for model, optimizer, scheduler, environment,
  trainer state, and root keys.
- Add documented Hydra CLI examples for sample/full motion data.  Compliance
  forces are synthesized online; no duplicate modified motion dataset is
  required.
- Use the audited official release checkpoint at
  `compliance_control/official_assets/sonic_release/last.pt` (HF revision
  `7c90a56cfe04788c4f041daeef5b1e12930675ad`, recorded SHA-256 ending
  `d8909`, training step 41550) for the real
  residual-initialization smoke; never mutate the source asset.  Independently
  reload this hash-verified file when auditing initialization, step 5, and step
  6 rather than trusting artifact-authored provenance or digests.
- Freeze every released policy/value parameter, including robot-motion
  encoder, quantizer, `g1_dyn`, `g1_kin`, critic, critic RMS, and action noise.
  Train exactly the two residual heads.  The HF decay/no-decay optimizer order
  is six weights followed by six biases, with one finite slot per tensor.
- Keep frozen action-noise checkpoint tensors byte-exact through a dedicated
  compliance actor that computes the release clamp out of place; retain the
  generic actor unchanged and preserve its state-dict schema.
- Lock the real PPO smoke to micro-batches `[4,24,*]` (16 environments, four
  mini-batches), two-site privileged width 9, tokenizer leading shape
  `[4,24,...]`, action `[4,24,29]`, and value `[4,24,1]`.  Require every
  residual tensor to receive a finite gradient and to differ from its separate
  initialization checkpoint after training.
- Before any strict-resume mutation, preflight model keys/shapes/dtypes/
  finiteness plus exact non-model payload structure.  Preserve the independent
  learning-rate boundaries: restore `args.learning_rate`, optimizer group LRs,
  and scheduler state exactly as saved; never overwrite loaded optimizer LRs
  from `args.learning_rate`.
- During the prescribed 16-environment/5-iteration smoke, separately record the
  fixed-shape all-environment candidate scheduler cost; keep the
  synchronization-safe algorithm unchanged until it has measured evidence.

### Phase 5 — Export and deployment switch

- Leave the released robot-motion encoder and decoder ONNX files byte-for-byte
  untouched.  Export only the trained action-residual head as a separate,
  versioned ONNX artifact with explicit `release_action_context [B,S,994]`
  and `motion_compliance_condition [B,S,3]` inputs and
  `action_delta [B,S,29]` output.  The graph concatenates these once into the
  exact trained 997-D residual input; it must never append condition to an
  already-997-D tensor.  Deployment composition remains
  `release_action + bounded_delta`.
- Keep export/runtime tensor contracts in a tracker-independent deployment
  module.  Provide both a Python validation/runtime layer and a C++ production
  layer whose caller-owned contract accepts arbitrary ordered context segments
  and an arbitrary non-empty set of named release-artifact pins.  A thin SONIC
  adapter owns observation names and assembly of the
  existing 64-D encoder token followed by the released 930-D actor observation;
  the 3-D condition remains the graph's second input.  No robot name,
  14-keypoint assumption, or IsaacLab import may enter the portable
  export/runtime layer.
- Add an opt-in deployment configuration and explicit runtime switch without
  changing released configs.  Integrate only a thin load/compose hook into the
  existing deploy executable, before its IsaacLab/BFS-to-MuJoCo/DFS remap.
  Disabled rows must bypass residual inference and return the released action
  bit-for-bit; mixed `[B,S]` gates must isolate off rows even when rejected
  condition/context rows contain NaN.  The host-disabled path must not read the
  artifact or create an ORT session.
- Treat operator mode selection as a tested input contract.  An enabled overlay
  uses the first two wrist controls as one global hard gate: both zero is off,
  either positive is on.  Synchronize startup and keyboard adjustments through
  every composite input manager and reject non-finite/out-of-range CLI values
  before robot middleware initialization.
- Compare PyTorch and ONNX Runtime output for dynamic batch/sequence shapes,
  all-off, all-on, and mixed gates.  Export atomically, record source checkpoint
  SHA/provenance beside the ONNX, and reject incompatible site/layout/version
  metadata before inference.  Compile the real C++ runtime against the accepted
  artifact and link the complete production target as a Phase-5 gate.

### Phase 6 — Integration, regression, and low-resource validation

- Current execution state (2026-07-28): paused by user after the collector,
  portable metrics, and final SONIC evidence validator reached `38 passed` in
  focused CPU tests. No formal Phase-6 GPU/six-protocol evidence has run, so
  this phase remains `IN_PROGRESS`.
- Run config/help/compile/unit tests plus IsaacLab smoke tests.
- Run paired stiff-mode baseline regression on fixed motion IDs/timestamps.
- Run single- and simultaneous-site compliant-force evaluation.
- Standardize simulator output behind a tracker-neutral aligned-trace schema.
  Pair rows exactly by caller-owned motion, sequence, seed, frame, and timestamp
  identities; keep all concrete body/endpoint mappings in thin adapters.
- Record endpoint RMSE/P95, orientation error, MPJPE, success rate, force peak,
  reference and measured yielded displacement (including projection along the
  actual force), inactive-hand cross-coupling, fall/reset/finiteness,
  throughput, and memory. Publish only bounded atomic NPZ/JSON artifacts.
- Bind each collection summary, full NPZ, and portable report by SHA-256. The
  SONIC final gate must reload the six bounded traces and recompute the complete
  tracker-neutral report under fixed criteria; it may not trust a report's
  self-declared `passed` field or an incomplete check list.
- Before real collection, extend the provenance gate to pin exact termination/
  event function targets and parameters, and add one Phase-6 nonzero-force
  configured-reset-event check. These are evidence hardening only; do not mix
  policy/optimizer/force-algorithm tuning into that change.
- Treat active-mode left/right endpoint, wrist orientation, and whole-body
  tracking accuracy as first-class outputs. Do not accept force/yield success
  alone when those tracking errors regress without explicit explanation.
- Add a 4096-environment performance characterization for the fixed-shape
  compliance candidate scheduler, including policy-step time and GPU memory
  against host-off/baseline.  This is a measured Phase-4/6 performance item,
  not a reason to reintroduce dynamic CUDA due-ID compaction.
- Mark complete only if all phase-6 acceptance criteria in `test_matrix.md`
  pass or any difference has a documented explanation.
- The prescribed low-resource smoke uses the `sonic_backup` environment, one
  audited robot PKL plus `sample_data/smpl_filtered`, 16 environments, 5
  iterations, and `use_wandb=false`.  CUDA is currently usable through the
  temporary NVIDIA 580.159 user-space compatibility libraries used by the real
  Phase-2/3 smokes, so the old driver mismatch is not a current blocker.  Pin
  that environment for every real test and revalidate it after a machine restart.
