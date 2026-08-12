# Execution log

## 2026-07-27 — Phase 1 started

- Fixed baseline to NVLabs upstream commit
  `4141c34280abb67c82e115342a8720f4a83d750d` on branch
  `experiment/motion-compliance`.
- The worktree contains no `AGENTS.md`; the repository engineering and mandatory
  execution-loop instructions supplied by the user are being followed.
- Confirmed current SONIC robot-motion encoder inputs are configured separately
  from the 14-body tracking/reward skeleton.
- Confirmed upstream contains an uncomposed legacy `ForceTrackingCommand` and
  compliance observation/reward helpers, while the referenced event configs are
  absent.  The new implementation will be additive and will not silently revive
  that incomplete path.
- Phase 1 intentionally makes no IsaacLab/runtime/config integration changes.
- Added user constraint: the implementation is split into a reusable universal-
  tracker core (`schema/math/schedule/reference_modifier/metrics`) and a thin
  SONIC adapter.  Core tests will reject IsaacLab imports and fixed robot/skeleton
  index assumptions.
- Audited external assets for later phases: official release checkpoint at
  `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt`
  (HF revision `7c90a56cfe04788c4f041daeef5b1e12930675ad`, recorded SHA-256
  ending `d8909`, step 41550) and six
  official sample PKLs in the same asset area.  The prescribed later smoke uses
  `sonic_backup`, one robot PKL, `sample_data/smpl_filtered`, 16 environments,
  5 iterations, and W&B disabled.
- GPU execution is currently unavailable because the NVIDIA 580.159 kernel
  module and 580.173 user-space driver do not match.  This does not block the
  pure-core Phase-1 tests, but must remain explicit for later simulator phases.
- First Phase-1 test invocation with the worktree-default Python failed because
  it resolves to `.venv_sim`, which has no `pytest`.  `conda run` inherited the
  same active `VIRTUAL_ENV`.  The matrix now invokes the audited
  `sonic_backup/bin/python` explicitly; no package installation was performed.
- The first successful pytest run generated three `__pycache__` directories and
  a root `.pytest_cache`; these were removed exactly.  The matrix now uses `-B`,
  `PYTHONDONTWRITEBYTECODE=1`, and pytest's disabled cache provider.

## 2026-07-27 — Phase 1 passed

- Recorded the baseline interface and future integration boundary in
  `artifacts/phase1_contract.md`, including immutable checkpoint/sample paths,
  digest, the later five-iteration smoke command, and the current GPU-driver
  blocker.
- Added the tracker-independent core in five responsibility-specific modules.
  Global `enabled=false` overrides even an all-active stale site mask: target
  selection is bitwise identical to the original reference and virtual force is
  exactly zero.
- Covered caller-provided 2-, 7-, and 17-site layouts.  Backpropagation through
  active selected-reference and virtual-force paths produced finite gradients;
  inactive compliant-reference gradients stayed exactly zero.
- Final Phase-1 pytest command passed: `10 passed in 0.73s` using the explicit
  `sonic_backup` interpreter with bytecode and pytest cache generation disabled.
- The standalone public-core import/shape assertion passed, and
  `git diff --check` passed.
- AST boundary checks passed: production core has no IsaacLab import,
  robot-specific body/order token, or fixed 29-DoF / 14-site integer.
- The final status audit found only the intended new package, test, and task
  files.  No existing SONIC runtime/config/training file was edited, and no
  `__pycache__`, `.pytest_cache`, `.pyc`, or `.pyo` artifact remains.
- Phase 1 is `PASSED`; `current_phase` advances to Phase 2, which remains
  `PENDING` and was not executed in this handoff.

## 2026-07-27 — Phase 1 reopened after independent review

- Per the review, reset `current_phase` to 1 and Phase 1 to `IN_PROGRESS`; no
  Phase-2 work was started.
- Rechecked Axellwppr `motion_tracking` compliance branch at commit `0526770`.
  Source lines 1254-1256 independently cap the nominal reference-displacement
  term and a `100 N/m`, `5 N` tracking-correction term, then add them with the
  `force_on_robot` sign.  This corrected the earlier nominal-only core API.
- Reworked tensor broadcasting for `[batch, future, sites, 3]` references with
  `[batch, sites]` masks/thresholds, while retaining arbitrary site counts.
- Added strict shape/dtype/device validation, binary finite enabled validation,
  non-finite displacement/input rejection, and explicit common-frame/sign
  contracts.
- Reworked metrics so the global gate is authoritative and inactive drift
  measures raw compliant-candidate pollution before target selection.
- Expanded the preliminary unit suite; all 23 tests passed before staging.  The
  complete final matrix and cached-diff check remain pending at this log point.

## 2026-07-27 — Phase 1 review fixes passed

- Staged only the Phase-1 compliance core, pure unit test, and task documents;
  no existing SONIC runtime, environment, training, or release file is in the
  staged diff.
- Final staged-state pytest passed: `23 passed in 0.91s`.  It covers multi-future
  broadcasting, arbitrary site counts, dtype/device/shape contracts, bitwise
  hard-off behavior, binary gates, NaN/Inf rejection, exact upstream two-term
  force formula/sign, nominal-only mode, finite gradients, and non-tautological
  inactive-reference metrics.
- The standalone import/condition-shape command, `git diff --check`, and
  `git diff --cached --check` all passed.  The latter confirms every reported
  EOF blank-line defect was removed.
- No unstaged source change, `__pycache__`, `.pytest_cache`, `.pyc`, or `.pyo`
  remains.  Phase 1 returns to `PASSED`; `current_phase` advances to 2, which is
  still `PENDING` and was not executed.

## 2026-07-27 — Phase 2 started

- Marked only Phase 2 `IN_PROGRESS`; Phase 1 remains staged and passed.
- Added an additive thin adapter without editing the large tracking command or
  any release configuration.  Reference-motion and articulation index spaces
  resolve independently from names.
- Added explicit current-anchor common/world transforms, persistent seeded
  per-environment sampling, full multi-future site-force state, replaceable
  20 N / 10 Nm residual-wrench limiting with anchor compensation, and a narrow
  PhysX event boundary with modern-composer feature detection.
- Added opt-in Hydra command and event compositions.  The interval writer runs
  after command computation every policy step; reset clears command buffers and
  the permanent composer.
- The combined pure suite passed preliminarily: `36 passed in 1.13s`.  It also
  composes the one-environment opt-in Hydra configuration and exercises modern
  and fallback writer paths without starting IsaacSim.
- Added a real one-environment headless acceptance script using the audited
  official robot/SMPL sample.  It performs 100 disabled and 100 forced-on
  policy steps and checks finite state, real permanent-composer application,
  and stale-wrench reset.  Phase 2 remains `IN_PROGRESS` until this CUDA smoke
  and the final matrix pass.

## 2026-07-27 — Phase 2 passed

- Fixed the standalone smoke import root explicitly so it loads the experiment
  worktree rather than the main workspace's editable package.  Its failure path
  emits a traceback and exits nonzero instead of allowing IsaacSim shutdown to
  mask an import error.
- The final combined pure matrix passed: `36 passed in 1.11s`.  This includes
  all 23 Phase-1 tests and 13 adapter/config/writer tests.
- The final real CUDA smoke passed in 22.4 seconds on one RTX 4090 environment
  using IsaacSim's modern `permanent_wrench_composer`.  It completed 100/100
  disabled steps with exactly `0.0 N` peak, then 100/100 forced-on steps with
  finite state and `8.329804 N` site / `8.329803 N` composer peak force.
- The registered reset event cleared command and composer force and torque;
  the smoke emitted `"reset_zero": true`.  It used the NVIDIA 580.159
  compatibility libraries against the audited official sample and exited 0.
- IsaacSim emitted non-fatal platform-info/NVML warnings while nevertheless
  reporting driver `580.159.03`, RTX 4090 active, constructing the real manager
  environment, and completing every acceptance assertion.
- `git diff --check` passed and repository-local Python/pytest caches were
  removed.  Phase 2 is `PASSED`; `current_phase` advances to 3.  No Phase-3
  implementation or test was executed.

## 2026-07-27 — Phase 2 reopened after boundary review

- Independent review found two construction-boundary hardening gaps, so Phase
  2 returned to `IN_PROGRESS` without beginning Phase 3.
- Body-name resolvers now reject scalar strings/bytes, empty sequences/names,
  non-string elements, duplicate available names, and an empty anchor instead
  of accidentally iterating characters or deferring an ambiguous mapping.
- Site body offsets now pass through a pure, unit-testable construction helper
  that requires exact `[num_sites, 3]` shape, real numeric data, and finite
  values before the command allocates runtime state.

## 2026-07-27 — Phase 2 review fixes passed

- Kept the portable checked APIs intact and added private, adapter-only
  unchecked tensor kernels after construction/resampling validation.  The
  per-step command path and cached condition avoid CUDA scalar extraction;
  static tests reject value-validation and scalar-sync operations there.
- Corrected the physical writer boundary for a moving robot.  Endpoint torque,
  net residual limiting, and anchor compensation remain in world coordinates;
  the final site/anchor force and torque use each body's current quaternion and
  are written in link-local coordinates with `is_global=false`.  This avoids
  reuse of the modern composer's first-write link pose.  A 90-degree changing-
  body test verifies both force and torque coordinates.
- Made the opt-in command operationally off by default.  It skips compliance
  math and leaves a clean composer untouched for all disabled steps; switching
  off after application writes zero once.  The public three-value condition is
  cached and exactly zero while disabled.
- The final combined pure matrix passed: `48 passed in 1.20s`.  It includes the
  23 Phase-1 tests plus strict mapping/offset validation, full future/site force
  math, no-sync boundary checks, moving-frame wrench conversion, 1/2/5-site net
  limiting, both writer API paths, opt-out behavior, and Hydra composition.
- The final real CUDA smoke exited 0 after exactly 100 disabled and 100 forced
  policy steps in the real one-environment RTX 4090 manager environment.  The
  disabled composer stayed inactive with `0.0 N` peak.  Forced mode reached
  `8.320412 N` site and `8.320410 N` composer peak force; reset cleared command
  and composer force/torque (`"reset_zero": true`).
- IsaacSim's platform/NVML diagnostics remained non-fatal.  The smoke used the
  specified NVIDIA 580.159 compatibility libraries and the audited official
  robot/SMPL sample; all finite-state and lifecycle assertions passed.
- Final source/cached diff checks and repository cache hygiene passed.  Phase 2
  is `PASSED`; `current_phase` advances to 3.  No Phase-3 implementation or
  test was executed.

## 2026-07-27 — Phase 3 started

- Confirmed `current_phase=3` and kept Phase 4 checkpoint/training work and
  Phase 5 deployment outside this phase.
- Audited the released encoder contract: G1 still consumes only `[10,58]`
  joint-position/velocity data and `[10,6]` anchor orientation.  The complete
  resolved tokenizer subtree is identical in the opt-in composition.
- Added thin observation/reward manager adapters plus IsaacLab-free tensor
  contracts.  Actor state is public condition only; configurable-site force,
  mask, and raw threshold remain critic-only.
- Factored the Phase-2 current-anchor reference calculation into one reusable
  state reader.  Control caches it during command update; rewards invoke it
  locally at reward time and never mutate command-owned cache or force state.
  A full Phase-2 GPU smoke rerun passed after this compatibility refactor.
- Added future-zero selected-target position reward, original orientation
  reward, and independent per-site selected/original/orientation errors.  New
  rewards are sampled-enable gated, so all off environments add exact zero.
- Preserved all release dense rewards and the inline `feet_acc=-2.5e-6` weight;
  selected endpoint position uses `weight=2.0,std=0.1` and orientation uses
  `weight=0.5,std=0.4` pending later metric-driven tuning.
- Preliminary combined pure tests passed: `54 passed in 3.52s`.  The real
  Phase-3 one-environment smoke resolved policy/critic widths `933/1657`, G1
  tokenizer shapes `[10,58]/[10,6]`, both hard-off rewards `0.0`, and bitwise
  original target selection.  Final matrix/staged-diff hygiene remains pending.

## 2026-07-27 — Phase 3 review fixes and final validation

- Independent review caught that IsaacLab evaluates rewards before the next
  command update.  The first enabled implementation therefore consumed a
  one-physics-step-old endpoint cache.  The production reward now recomputes
  current articulation/reference state locally, uses future frame zero, and
  leaves the command cache and wrench state untouched.  A stale-cache unit
  regression and a real-command enabled smoke both cover this lifecycle.
- Replaced arithmetic enable gating with a boolean `torch.where` hard gate and
  mask disabled errors before the Gaussian.  Disabled environments now return
  exact zero even when their unused error tensor contains NaN.
- Final combined pure matrix exited 0 with `54 passed in 3.46s`.  It also proves
  complete tokenizer and termination subtree equality, release dense-reward
  config equality, dynamic 1/2/5-site critic widths, actor privilege isolation,
  inactive-site bitwise reference preservation, and original-only orientation.
- The final Phase-2 real regression exited 0 after 100 disabled plus 100 forced
  steps: disabled peak `0.0 N`, forced site/composer peaks
  `8.3204117/8.3204098 N`, and `reset_zero=true`.
- The final Phase-3 real smoke exited 0 in 13.93 seconds.  It resolved actual
  policy/critic shapes `[1,933]/[1,1657]`, G1 tensors
  `[1,10,58]/[1,10,6]`, independently reconstructed the tracking future-zero
  reference, returned both hard-off rewards as `0.0`, and proved
  `shared_total_reward_exact=true`.  With the real command's prior cache
  poisoned, its enabled production position reward was exactly `1.0`; reward
  evaluation changed neither the poisoned cache nor any wrench buffer.
- IsaacSim's platform/NVML messages remained non-fatal; both GPU smokes used
  the required NVIDIA 580.159 compatibility libraries and audited official
  sample.  Phase 3 is `PASSED`; `current_phase` advances to 4, but no Phase-4
  implementation or test was executed.
- Final source and cached `diff --check` passed.  Exactly 23 Phase-3 files are
  staged with no unstaged changes or repository-local Python/pytest caches;
  the released `sonic_release.yaml` remains absent from the diff.  No commit or
  push was performed in this phase.

## 2026-07-27 — Phase 3 reopened for strict off-mode RNG parity

- A late lifecycle review found that IsaacLab's base `CommandTerm._resample`
  samples `time_left` from the global generator before invoking the subclass.
  It also found that a zero-period interval event still samples its scheduling
  state globally.  Both behaviors could shift baseline observation or motion
  randomness even when compliance was operationally disabled.
- The command now owns duration sampling through its seeded generator and uses
  a finite stable timer while host-disabled.  Reset, repeated compute, and the
  reset event leave the next CPU and CUDA global random samples bitwise equal
  to their saved baseline states.
- The command writes its link-frame wrench immediately after computing it, so
  the added interval event is no longer needed.  Resolved baseline and opt-in
  interval event names and ranges are exactly equal, and the actual simulator
  manager contains only the added compliance reset event.
- Dirty disable performs one narrow zero write to the configured compliance
  body rows immediately in the setter, before another physics step, and
  preserves every unowned composer row.  Full composer reset is reserved for
  the environment-reset lifecycle, where stale external forces must be cleared;
  only a full host-disabled reset releases global dirty ownership.
- The final pure matrix passed: `56 passed, 1 skipped in 3.79s`; the sole skip
  is the ordinary interpreter's unavailable CUDA device parametrization.
- The final real Phase-2 RTX 4090 regression exited 0 in 22.54 seconds with
  `disabled_rng_neutral=true`, `disabled_peak_force_n=0.0`, forced site and
  composer peaks `8.3204117/8.3204098 N`, and `reset_zero=true`.
- The final real Phase-3 smoke exited 0 in 13.59 seconds with policy/critic widths
  `933/1657`, tokenizer shapes `[1,10,58]/[1,10,6]`, both disabled rewards
  exactly zero, enabled freshness reward `1.0`, and exact shared total reward.
- Phase 3 is again `PASSED`; `current_phase` advances to 4, which remains
  `PENDING`.  No Phase-4 implementation or test was executed.

## 2026-07-27 — Phase 3 reopened for CUDA synchronization regression

- A Phase-4 readiness review found that the inherited IsaacLab command loop
  compacted `(time_left <= 0).nonzero()` every policy step and that the due-ID
  resample path re-entered public range checks.  On CUDA, these dynamic-index
  and Python truth operations force device-to-host synchronization.
- Public state APIs retain full shape/range validation and now resolve IDs
  exactly once.  The command overrides `compute(dt)` and, while enabled,
  samples fixed-shape all-environment candidates before selecting due state,
  timer, wrench-buffer, and counter rows with a boolean mask.  It never creates
  a dynamic due-ID tensor; host-off remains stable and consumes no RNG.
- A follow-up trace review also found data-dependent `nonzero()` compaction in
  the core site's at-least-one fallback.  Site sampling now uses fixed-shape
  random scores, an `argmax` fallback mask, and `torch.where`; it consumes a
  fixed number of owned-generator draws and dispatches no internal `nonzero`.
- The independent source review found no remaining P0/P1 issue.  A final pure
  command-owned regression directly invokes `_resample_masked_prevalidated`
  through a minimal non-Isaac command instance and proves timer, all four
  application buffers, counter, and state preserve every non-due row bitwise,
  reset/increment due rows, and consume the same owned RNG for different masks.
- The final pure matrix passed with `64 passed, 1 skipped in 3.76s`; it also
  covers fixed-shape fallback sampling, public validation count, and source
  guards against inherited/dynamic-ID compute.
- The parent thread's frozen-code Phase-2 rerun exited 0 in 23.02 seconds:
  bound `command.compute(dt)` passed both dispatch and CPU/CUDA profiler guards,
  disabled peak was `0.0 N`, forced site/composer peaks were
  `8.3204117/8.3204098 N`, and RNG neutrality/reset clearing passed.
- The parent thread's frozen-code Phase-3 rerun exited 0 in 13.73 seconds with
  policy/critic widths `933/1657`, tokenizer shapes `[1,10,58]/[1,10,6]`, exact
  zero off rewards, freshness reward `1.0`, and exact shared total reward.
- The cost of fixed-shape candidate sampling at 4096 environments remains an
  explicit Phase-4/6 performance measurement; no algorithm change is justified
  before those timing and memory measurements.  The temporary NVIDIA 580.159
  compatibility environment is working, not a current blocker, but must be
  revalidated after a machine restart.
- Phase 3 returns to `PASSED` and `current_phase` advances to 4.  The isolated
  Phase-4 work was not mixed into this regression fix.

## 2026-07-27 — Phase 4 CPU gate passed; GPU acceptance pending

- Added a strict, one-way official checkpoint adapter.  Its public official
  wrapper pins SHA-256
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`,
  full revision `7c90a56cfe04788c4f041daeef5b1e12930675ad`, and source
  step 41550.  The actor first layer expands `994→997`; critic weight and RMS
  expand `1645→1657` for two sites, with zero weight/mean tails, one variance
  tail, and unchanged RMS count.
- The generated init artifact contains complete strict policy/value state,
  fresh `OnlineTrainerState(global_step=0)`, provenance, and explicit
  `optimizer/scheduler/env=None`.  It cannot be resumed.  The normal branch
  uses an isolated `MotionCompliancePPOTrainer` and requires complete non-empty
  optimizer, scheduler, environment, and trainer state for strict resume; the
  generic PPO trainer has no diff.
- `decoder_critic` freezes every policy parameter except `g1_dyn`, including
  all encoders, `g1_kin`, quantizer parameters, and `std/log_std`; every critic
  parameter is trainable.  Trainer construction validates optimizer ownership
  exactly after its optimizer is created.
- The bounded exposure callback atomically updates one JSON per PPO step,
  requires one finite `loss/*` and timing log per iteration, independently
  counts enabled+active+nonzero force for each site, rejects stale force, and
  records process peak CUDA memory.  The post-train audit requires every new
  actor/critic column nonzero, frozen tensors bitwise official, finite model and
  optimizer tensors, fresh optimizer steps, and exact expected global step.
- Hydra is guarded against smoke-contract overrides and output collisions.  It
  resolves the official single robot PKL/SMPL directory, 16 environments, five
  iterations, W&B off, `save_last_frequency=5`, forced two-site exposure, and
  the dedicated strict trainer.  Every generated artifact is constrained to
  the central `compliance_control/runs/motion` root.  Strict step-5→6 resume
  uses a separate output directory so the step-5 evidence is preserved.
- The standalone official CPU migration exited 0 and wrote only central
  artifacts.  Its audit reported actor `[994,997]`, critic `[1645,1657]`, RMS
  count `69574656000.0`, source step 41550, and fresh/no-old-state semantics.
- The final pre-review combined pure command passed with `76 passed, 1 skipped
  in 8.34s`; after review hardening, the focused Phase-4 suite passed with
  `13 passed in 5.76s`.  Both training `--help` and the finetune `--cfg job`
  command exited 0.  A final combined rerun remains required after the last
  callback/path changes.
- Independent Phase-4 review found no P0.  Its P1/P2 findings (finite losses,
  exact smoke parameters, collision-safe paths, separate resume output,
  discoverable scheduler timing/memory command, exact test matrix, and cache
  hygiene) were incorporated.  No Phase-4 GPU command has run yet; Phase 4
  remains `IN_PROGRESS` pending the exact CUDA matrix, five-step training,
  step-5 audit, strict one-step resume, and step-6 audit.
- The final frozen Phase-4 CPU gate passed after those review fixes:
  `77 passed, 1 skipped in 8.51s`; the skip is the existing ordinary-process
  CUDA parametrization.  Official migration reran with `--overwrite` against
  only its owned generated artifact and passed in 5.4 seconds.  Training help,
  resolved Hydra config, scheduler benchmark help, and checkpoint-audit help
  all exited 0.  `git diff --check` and repository cache hygiene passed before
  staging.  GPU execution is intentionally waiting for parent approval.

## 2026-07-27 — Phase 4 GPU acceptance, frozen-noise diagnosis, and fix

- The approved serial CUDA matrix first passed the inherited Phase-2 and
  Phase-3 real smokes and the 16-environment scheduler benchmark.  The initial
  five-step PPO process also exited 0, but the required hard audit correctly
  stopped the matrix because frozen policy tensor `std` differed from the
  official checkpoint.
- This was not optimizer leakage.  The official direct-`std` values span
  `0.2968585193..0.5000106096`; only the four values above `0.5` changed, all to
  exactly `0.5`, with maximum absolute delta `1.06096e-5`.  The generic actor's
  `get_std` applies `self.std.clamp_` under `no_grad` even when
  `requires_grad=false`.  The full failed run was preserved, not deleted, by
  atomically moving it to `phase4_gpu_smoke_failed_std_clamp`.
- Added the isolated `MotionComplianceFrozenNoiseActor`.  It retains the exact
  release effective clamp and state-dict schema but computes the direct-`std`
  clamp out of place.  Only the opt-in finetune Hydra config uses it; workflow
  validation rejects another actor target, and the generic actor remains
  unmodified.  A real-release regression loads the over-bound values, builds
  the action distribution repeatedly, requires byte-exact stored `std`, and
  proves the optimizer still owns exactly `g1_dyn` plus critic parameters.
- The focused checkpoint/training suite passed with `14 passed in 5.71s`; the
  complete pure Phase-1/2/3/4 suite passed with
  `78 passed, 1 skipped in 8.44s`.  The official CPU migration rerun reported
  actor `994→997`, critic/RMS `1645→1657`, unchanged RMS count
  `69574656000.0`, source step 41550, and fresh state.  Training `--help` and
  the resolved finetune config both exited 0; the latter resolves the dedicated
  actor and strict trainer.

## 2026-07-27 — Phase 4 passed

- On the corrected frozen tree, the inherited real Phase-2 smoke passed in
  22.92 seconds: 100 disabled and 100 forced steps, exact disabled force/RNG
  neutrality, forced site/composer peaks `8.3204117/8.3204098 N`, no scalar or
  dynamic-nonzero command dispatch, and `reset_zero=true`.  The Phase-3 smoke
  passed in 13.62 seconds with policy/critic widths `933/1657`, unchanged G1
  tokenizer shapes, exact off rewards, enabled freshness `1.0`, and exact
  shared reward total.
- The fresh 16-environment scheduler benchmark used 100 warmups and 1000
  measured iterations on RTX 4090.  Enabled fixed-shape sampling cost
  `265.651 us` with `8704` bytes incremental peak allocation; the host-off path
  cost `2.358 us` with zero incremental allocation.
- Fresh training completed five PPO iterations / 1920 timesteps and exited 0
  in 24.87 seconds.  The training-loop total was 9.05 seconds; mean collection
  and learning times were `1.5149/0.2950 s`, FPS range was `159..233`, and
  process peak CUDA allocation was `543227392` bytes.  All five batches exposed
  both sites (`80/80` active and nonzero samples), peak force was
  `14.977984 N`, and all 55 recorded loss values were finite.  Final policy,
  value, entropy, and auxiliary loss values were `-0.071478`, `0.015463`,
  `13.122421`, and `0.008347`.
- The step-5 audit passed: global step 5, actor additions `3/3` nonzero, critic
  additions `12/12` nonzero, 41 frozen policy tensors byte-exact to the pinned
  official checkpoint, 28 fresh optimizer slots at PPO update step 100, and
  finite model/optimizer state.  The generated init and step-5 checkpoints are
  `149658515` and `323439369` bytes.
- Strict resume loaded complete step-5 model/optimizer/scheduler/environment/
  trainer state, ran exactly one PPO batch, and saved a separate step-6
  checkpoint without overwriting step 5.  The process exited 0 in 17.64 seconds
  at 176 FPS (`1.7455 s` collection, `0.4351 s` learning).  Both sites had
  `16/16` active and nonzero samples, peak force was `14.911679 N`, all 11 loss
  values were finite, and peak CUDA allocation was `693784064` bytes.
- The independent step-6 audit passed: global step 6, actor/critic additions
  `3/3` and `12/12` nonzero, the same 41 frozen tensors byte-exact, 28 optimizer
  slots advanced to PPO update step 120, and all state finite.  The step-6
  checkpoint is `323439689` bytes.  Both hard-audit JSON reports and bounded
  exposure reports remain under the central runs root.
- IsaacSim emitted the same non-fatal platform-info, NVML-initialization, and
  CPU-governor diagnostics while reporting driver `580.159.03` and an active
  RTX 4090.  Every prescribed CPU and CUDA acceptance command exited 0 after
  the isolated fix.  Phase 4 is `PASSED`; `current_phase` advances to Phase 5,
  which is `IN_PROGRESS` but was not executed in this handoff.
- Final source and cached `diff --check` passed with exactly the intended 20
  Phase-4 files staged and no unstaged changes.  The generic PPO trainer has no
  diff; no repository-local Python/pytest cache remains.  Post-run process
  checks found no IsaacLab/training process and no NVIDIA compute application.

## 2026-07-27 — Phase 4 reopened for exact release-noise configuration

- A late independent review found one P1 after the Phase-4 commit: the workflow
  required the dedicated actor class but did not reject CLI overrides of the
  release noise representation and clamp values.  Phase 5 was stopped before
  edits; status returned to Phase 4 while this boundary was repaired.
- The exact smoke guard now requires `use_log_std=false`,
  `use_clampped_std=true`, `std_clamp_min=0.001`, `std_clamp_max=0.5`, and
  `clamp_noise_std=false`.  Unit tests independently override every field and
  require rejection while retaining a passing valid composition.
- The same review's P2 observations were closed without modifying the generic
  PPO trainer.  Frozen-state comparison now checks tensor element bytes, so a
  `+0.0/-0.0` sign-bit difference is rejected.  The isolated compliance trainer
  replaces the inherited raw-parameter noise metric with the actual effective
  clamped mean; a release-`std` test checks that reporting does not mutate the
  parameter.
- The final focused suite passed with `14 passed in 5.67s`; the complete pure
  Phase-1/2/3/4 suite passed with `78 passed, 1 skipped in 8.41s`.  The official
  CPU migration, training help, resolved config, and strengthened step-5 and
  step-6 hard audits all exited 0.  Those audits again reported global steps
  5/6, additions `3/3` and `12/12`, 41 byte-exact frozen tensors, and optimizer
  update steps 100/120.
- No five-step retraining was needed: the accepted Hydra defaults, effective
  distribution, optimizer ownership, and checkpoint weights did not change;
  only invalid-override rejection, audit strictness, and metric reporting did.
  The existing real step-5/6 artifacts were re-audited under the stronger byte
  comparison.  Phase 4 returns to `PASSED`; `current_phase` advances to Phase 5,
  which remains unexecuted in this repair.

## 2026-07-27 — Phase 3 reopened: expanded-input design invalidated

- Phase 5 was stopped before execution after a P0 architecture review.  The
  prior 933/1657 design trained the released `g1_dyn` and whole critic, so
  `enable=0` was only a zero-tail initialization property and could not remain
  structurally identical to the official tracker after finetuning.  Phase 3 is
  reopened; Phases 4 and 5 are pending redesign.  The prior GPU training and
  audit outputs remain preserved as invalidated-design evidence and will not be
  used as an initialization/resume source.
- The replacement Phase-3 contract keeps released policy/critic groups at
  930/1645, `g1_dyn` at 994, and critic/RMS at 1645.  Public condition width 3
  and privileged width `1+4*S` are separate observation groups.  Frozen
  release actor/value paths are composed with independent, zero-initialized,
  bounded/hard-gated residual heads; only those heads may own gradients.
- Disabled rows are sanitized before the residual MLP and selected with
  `torch.where`, preventing rejected NaN values from contaminating shared
  mixed-batch gradients.  The actor uses an explicit observation allowlist for
  direct and temporal paths, residual initialization preserves caller RNG, and
  external-token rollout reuses the out-of-place release noise clamp.

## 2026-07-27 — Reopened Phase 3 passed

- The final combined Phase-1/2/3 pure suite passed with `68 passed, 1 skipped
  in 5.21s`; the skip is the existing ordinary-process CUDA parametrization.
  It covers separate observations for 1/2/5 sites, same-shape release model
  construction, zero-init byte parity, mixed `[off,on,off]` gating with rejected
  NaN rows, residual-only finite gradients/optimizer ownership, actor privacy,
  non-g1/aux paths, external tokens, RNG preservation, and bounded deltas.
- The pinned official CPU audit passed with SHA-256
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
  It confirmed actor input `2048x994`, critic input `2048x1645`, critic RMS width
  1645, unchanged release policy/critic observation subtrees, and separate
  condition/privileged widths 3/9.  Official state contains no residual keys.
- The mandatory Phase-2 real regression passed serially on RTX 4090 with 100
  disabled and 100 forced steps, zero disabled force, command/composer peaks
  `8.3204117/8.3204098 N`, reset clearing, RNG neutrality, and scalar/nonzero-
  free traced command compute.
- The final Phase-3 real smoke passed with actual observation shapes
  policy/critic `930/1645`, condition/privileged `3/9`, and unchanged G1
  tokenizer shapes `[10,58]`/`[10,6]`.  The resolved actor/value retained
  `g1_dyn=994` and critic/RMS=1645, loaded every official tensor byte-exact with
  only six action-residual and six value-residual tensors missing, and passed
  off-byte parity, mixed gates, privileged poisoning, aux/external-token paths,
  frozen-noise clamping, the 0.25 action-delta bound, and four nonzero finite
  gradient tensors.  Reward freshness remained `1.0`; both disabled new rewards
  were zero and the released shared reward total was bitwise exact.
- IsaacSim repeated its known non-fatal platform-info/NVML/CPU-governor warnings
  while reporting driver `580.159.03`; both real commands exited 0.  `git diff
  --check` and cache hygiene passed.  Phase 3 is `PASSED`; Phase 4 is now the
  current phase but no Phase-4 migration, training, or export command was run.

## 2026-07-27 — Reopened Phase 4 CPU gates passed for same-shape residuals

- Phase 4 now initializes a schema-v2 same-shape branch from the independently
  pinned official checkpoint.  The 55 policy and 17 value base tensors remain
  byte-exact; only six action-residual and six value-residual tensors are
  added.  Release widths remain actor/g1_dyn `930/994`, critic `1645`, and RMS
  `1645`; the separate two-site residual contexts are `997/1657`.
- The final combined Phase-1/2/3/4 pure suite passed with `87 passed, 1 skipped
  in 31.76s`.  It covers schema-v1 rejection, strict initialization and resume
  preflight before mutation, exact Hugging Face optimizer ordering, and two
  synthetic `[4,24,*]` PPO updates with finite gradients and byte changes for
  all twelve residual tensors.  The second forward constructively proves that
  distinct timestep contexts do not broadcast timestep zero.
- Strict resume now preflights the complete model, optimizer, scheduler,
  environment, and saved trainer-state boundary before any live mutation.
  Optimizer groups require exact saved/live parameter ID order, fixed AdamW
  flags, finite/domain-valid dynamic learning rates, exact three-key slots,
  scalar positive float32 steps, and matching finite moments.  Environment and
  trainer state receive recursive key/shape/dtype/finiteness checks and
  recursive exact post-load comparison.  Constructive negative tests corrupt
  nested environment state, same-shape optimizer order, optimizer flags,
  trainer fields, and trainer tensors and prove every live boundary unchanged.
  The positive resume test preserves distinct saved argument and optimizer
  learning rates; audit tests require both Adam moments nonzero per residual.
- The pinned official CPU residual-initialization smoke exited 0.  Its report
  independently verified SHA-256
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`,
  revision `7c90a56cfe04788c4f041daeef5b1e12930675ad`, source step 41550,
  `policy_state_dict`, official tensor counts `55/17`, RMS count
  `69574656000.0`, byte-exact base state, and a fresh step-0 checkpoint with no
  optimizer, scheduler, or environment state.
- Both `train_agent_trl.py --help` and the resolved finetune Hydra config exited
  0.  The latter resolves residual-only training, `[256,256]` action/value
  heads, action delta limit `0.25`, 24 rollout steps, five PPO epochs, four
  mini-batches, no symmetry, frozen release noise, 16 environments, five
  iterations, W&B disabled, and save-last frequency 5.
- All known CPU-review P0/P1 findings are closed; final independent read-only
  review is pending.  The generic PPO trainer has no diff.  Phase 4 remains
  `IN_PROGRESS`: no new CUDA command has run, and the real Phase-2/3
  regressions, five-step training, independent step-5 audit, strict step-6
  resume, and step-6 audit require explicit root approval first.  All prior
  expanded-input Phase-4 GPU outputs remain invalid evidence and are not
  accepted by the new schema.

## 2026-07-27 — Phase 4 independent-audit closure before GPU

- Independent review found that runtime resume already rejects a saved/live
  optimizer-ID mismatch, but the separate post-train audit could still pair
  two same-shape residual moments after swapping their serialized IDs.  The
  offline audit now pins the two groups to IDs `0..5` and `6..11`, requires the
  complete pinned PyTorch 2.7 AdamW group schema, fixed flags, and the actual
  PPOConfig default `weight_decay=0.0` in both serialized groups,
  and still permits only the dynamic learning-rate fields to vary.
- New negative cases swap the two same-shape `256x256` slots and set
  `maximize=true`; both are rejected by the independent checkpoint audit.
  The focused negative/config/resume tests passed, and the complete CPU suite
  was rerun with `87 passed, 1 skipped in 36.50s`.  GPU remains unstarted until the final
  read-only re-audit confirms this last evidence boundary.

## 2026-07-27 — First same-shape GPU launch exposed TensorDict boundary

- The pre-training Phase-2 writer/RNG/profiler regression passed on the RTX
  4090: 100 disabled steps remained at `0 N`, 100 forced steps reached
  `8.3204117 N` at the sites and `8.3204098 N` in the composer, and reset plus
  host-scalar/nonzero-free checks passed.  The Phase-3 real model regression
  also passed at policy/critic `930/1645`, condition/privileged `3/9`, with the
  frozen official off path byte-exact.
- The first five-step launch created the environment and schema-v2 official
  initialization, then failed before the first policy rollout/PPO step.  The
  isolated actor used direct mapping iteration to form its allowlist, while
  the real manager wrapper supplies a `TensorDict` that intentionally rejects
  direct iteration.  No `last.pt` or exposure artifact was produced.  The
  partial root `phase4_residual_gpu_smoke` is retained as failed evidence and
  is not reused.
- The actor boundary now uses the explicit top-level `keys()` API shared by
  dict and TensorDict.  A real `TensorDict` allowlist test was added; targeted
  testing passed and, after updating the fresh-run routing, the full CPU suite
  passed again with `87 passed, 1 skipped in 44.43s`.  The matrix/runbook route the fresh retry and resume
  to `phase4_residual_gpu_smoke_tensordict_fix` and
  `phase4_residual_gpu_resume_tensordict_fix`.

## 2026-07-27 — Reopened Phase 4 passed with real same-shape finetuning

- After the TensorDict fix, the inherited real Phase-2 and Phase-3 CUDA gates
  passed again at the accepted same-shape contracts: release policy/critic
  `930/1645`, public/privileged condition `3/9`, and unchanged G1 tokenizer
  `[10,58]` / `[10,6]`.  The Phase-2 100-disabled/100-forced run observed
  `0 N` disabled force and `8.3204117/8.3204098 N` site/composer peaks.
- The fresh 16-environment scheduler benchmark used 100 warmups and 1000
  measurements on RTX 4090.  Host-off cost `2.193408 us` and zero incremental
  allocation; enabled fixed-shape sampling cost `269.245758 us` and `8704`
  incremental bytes.  This is scheduler-only characterization, not end-to-end
  policy latency.
- The fresh official run completed five PPO iterations / 1920 simulator
  timesteps and exited 0.  Mean collection/learning times were
  `1.427531/0.257943 s`; FPS ranged `176..250`; process peak CUDA allocation was
  `353315840` bytes.  Both sites received `80/80` active and nonzero-force
  samples, peak site force was `14.996154 N`, and all 55 recorded loss metrics
  were finite.
- The independent step-5 audit passed against both the hash-pinned official
  checkpoint and separate step-0 initialization: all 55 policy plus 17 value
  release tensors remained byte-exact, all six action plus six value residual
  tensors changed, all 12 optimizer slots had finite nonzero first/second
  moments, and their update step was 100.
- One manually typed resume command used an erroneously path-prefixed Hydra
  group and failed during config lookup before application or simulator startup.
  The documented exact command was then used against a still-absent fresh output
  root; it restored step 5 strictly, ran one PPO iteration, and saved step 6.
- The independent step-6 audit passed: official 55/17 base tensors were still
  byte-exact, all 12 residual tensors remained changed with nonzero moments,
  optimizer steps advanced to 120, and the resumed batch added `16/16`
  active/nonzero-force samples with `14.913702 N` peak.  Saved args retained
  learning rate `1e-5`, optimizer groups retained `2e-5/2e-5`, canonical IDs
  remained `0..5` and `6..11`, and scheduler state advanced from epoch 5 to 6.
- The accepted directories are
  `phase4_residual_gpu_smoke_tensordict_fix` and
  `phase4_residual_gpu_resume_tensordict_fix`; old expanded-input and failed
  TensorDict-launch directories remain invalid evidence.  These short runs prove
  initialization/training/resume integrity and physical-path exposure only;
  they are not evidence of learned compliance or tracking performance.
- The generic PPO trainer has no diff, source diff checks pass, and no
  repository-local Python/pytest cache remains.  Phase 4 is `PASSED`; Phase 5
  starts with a separate residual-only ONNX/deployment switch design so the
  released encoder and decoder remain untouched.

## 2026-07-27 — Phase 5 passed with standalone residual export

- The release interface audit found that the trained action MLP does not accept
  a 997-D tensor plus condition.  Its exact input is
  `token64 + actor_obs930 + condition3 = 997`.  The standalone ONNX therefore
  accepts `release_action_context [B,S,994]` and condition `[B,S,3]`, concatenates
  exactly once, and emits only `action_delta [B,S,29]`.  This closes a potential
  1000-vs-997 first-layer mismatch without touching the frozen encoder/decoder.
- A tracker-independent deployment package now owns versioned metadata,
  six-tensor graph reconstruction, atomic directory publication, digest/layout
  verification, lazy ORT loading, hard-off inference bypass, mixed-row NaN
  isolation, and byte-preserving action composition.  It contains no IsaacLab
  import, robot/body name, 14-keypoint order, or hard-coded repository path.
  The thin SONIC adapter alone assembles token then actor observation, encodes
  the public threshold/Kp condition, and owns concrete site/action layouts.
- A first exported metadata bundle incorrectly declared MuJoCo/DFS action order.
  Code audit at `g1_deploy_onnx_ref.cpp:3123` showed that decoder output is still
  IsaacLab/BFS and the deploy remaps it afterward.  That directory remains
  unchanged as invalid evidence.  The accepted fresh bundle pins
  `joint_utils.G1_ISAACLab_ORDER` item-for-item and requires composition before
  the existing `isaaclab_to_mujoco` remap.
- The accepted bundle is
  `phase5_action_residual_export_isaaclab_order/bundle`.  Its checkpoint SHA is
  `42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba`,
  checkpoint step is 6, metadata SHA is
  `e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5`,
  and ONNX SHA is
  `9e7a30ae8485eb153b63db81575c9b0fd24522523510560ed5d6292652568a81`.
  The 1,317,966-byte graph contains exactly six initializers; the 2,679-byte
  manifest records torch 2.7.0, ONNX 1.21.0, and ONNX Runtime 1.25.0.
- Independent PT/ORT acceptance passed six all-off/all-on/mixed cases at
  dynamic `[2,3]` and `[1,5]` leading shapes.  Maximum absolute difference was
  `7.450580596923828e-09`; rejected NaN rows were finite exact zero.  The
  disabled and all-off paths made zero session calls, mixed `[3,4]` made one,
  off action rows stayed byte-exact, and maximum enabled delta was
  `0.021564483642578125`, below the declared 0.25 bound.  This is interface
  evidence from the six-step smoke checkpoint, not tracking/performance proof.
- Final tests passed: Phase-1/2/3/4 pure regression `87 passed, 1 skipped in
  35.83s`; Phase-5 deployment/export `26 passed in 1.26s`; Hydra help/config,
  ORT-free `sonic_backup` import, accepted-artifact audit, release-source diff,
  generic-trainer/prior-phase diff, and `git diff --check` all exited 0.  The
  ORT CPU run emitted only a non-fatal GPU-discovery warning.  No repository
  cache remains.  Phase 5 is `PASSED`; Phase 6 is now `IN_PROGRESS` and was not
  executed in this handoff.

## 2026-07-27 — Phase 5 reopened for production C++ closure

- The earlier Python-only export result remains valid, but its log entry marked
  Phase 5 passed before the production deploy executable consumed the artifact.
  Phase 5 was therefore reopened; task status remains `IN_PROGRESS` until the
  complete updated matrix and independent review pass.
- Added a tracker-neutral C++ action-residual runtime with a PImpl ORT boundary,
  arbitrary ordered context fields, arbitrary named/hash-pinned release
  artifacts, schema/layout/digest checks, lazy disabled behavior, hard-off
  zero-session behavior, mixed-row sanitization, a 0.25 delta bound, and
  release-action fallback.  Concrete 994-D SONIC context, release hashes,
  wrist sites, BFS action order, and operator-gate semantics remain in a thin
  adapter.
- The accepted ONNX bundle compiled and ran through system ORT 1.16 with marker
  `MOTION_COMPLIANCE_PHASE5_CPP_ORT_PASS`: dynamic `[2,3]`, six mixed rows,
  zero hard-off calls, action width 29, context width 994, and additional
  generic one-/three-field host contexts passed.  The complete
  `g1_deploy_onnx_ref` target then configured and linked successfully without a
  test dependency download.
- A read-only review found that composite input getters could read a child
  value different from the startup/keyboard value stored on the manager.  The
  startup value and adjustments now propagate through `InterfaceManager`,
  `GamepadManager`, and `ZMQManager`, including ZMQ/pose delegates.  The enabled
  overlay uses either positive wrist value as one global gate; both zero is an
  exact bypass.  Help and runtime diagnostics state this limitation explicitly.
- CLI compliance values now require one or three finite values in `[0.0,0.5]`.
  Because the production target uses `-ffast-math`, finite validation inspects
  IEEE-754 exponent bits rather than relying on an optimizable
  `std::isfinite`.  The compiled CLI gate rejected NaN, Inf, negative,
  over-range, trailing-character, wrong-width, and missing inputs before DDS or
  model initialization, with marker
  `MOTION_COMPLIANCE_PHASE5_DEPLOY_CLI_PASS invalid_values=8 dds_initializations=0 portable_fast_math=off`.
- The reusable C++ runtime itself also performs finite checks.  Its small target
  now appends `-fno-fast-math` after the repository-wide flag; the compiled
  command is an acceptance input.  This leaves the release inference target's
  arithmetic flags unchanged.
- Final read-only review reported no P0/P1 issue and three P2 contract/document
  gaps, all addressed before phase closure: trailing comma fields are now
  rejected and covered by the production CLI acceptance; keyboard shortcut
  claims are limited to the three composite managers and ZMQ comments match
  actual optional external updates; Python export/schema validation and C++
  loading now all pin artifact-v1 ONNX opset exactly 17.
- A second final audit found that only the C++ generic loader, not the Python
  generic loader, pinned the actual unchanged release files.  Python now uses
  the same arbitrary non-empty `(name, path, sha256)` host contract; the SONIC
  loader compares YAML declarations to those caller-owned pins before hashing
  each file or creating ORT.  The independent acceptance CLI takes the three
  released files and hashes as explicit inputs and records them in its report.

## 2026-07-27 — Phase 5 production closure passed

- The final combined Phase-1/2/3/4 suite passed with `87 passed, 1 skipped in
  25.52s`; the expanded deployment/export suite passed with `33 passed in
  2.54s`.  Both Hydra entrypoint/config gates exited 0.
- Independent accepted-artifact validation passed six dynamic-shape parity
  cases with maximum PT/ORT error `7.450580596923828e-09`, exact hard-off
  bypass, model SHA
  `9e7a30ae8485eb153b63db81575c9b0fd24522523510560ed5d6292652568a81`,
  metadata payload SHA
  `e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5`,
  and explicit hashes for the unchanged decoder, encoder, and observation
  config.  The regenerated acceptance report SHA is
  `86481c92d9d395d579c0ca30770a821698c7c23e0ba7ba42ae5a3571cc770ac1`.
- The system-ORT portable/SONIC C++ smoke passed, the complete production
  target configured and linked, its help exposed the opt-in overlay, and the
  compiled CLI acceptance rejected all eight invalid compliance inputs before
  DDS initialization.  The portable target retained `-fno-fast-math` after the
  repository-wide flag.
- Final diff gates passed: release artifacts and loader interfaces are
  unchanged from upstream; the generic PPO trainer and Phase-1/2/3/4
  training/environment sources have no Phase-5 diff; the portable Python/C++
  packages contain no SONIC/G1/IsaacLab/wrist vocabulary; `git diff --check`
  passed and no repository-local Python/pytest cache was present.
- Two independent reviews found no remaining P0/P1 issue.  Phase 5 is now
  `PASSED`; Phase 6 is `IN_PROGRESS`.  This phase proves a portable deployment
  boundary and operational switch, not learned tracking/compliance quality.

## 2026-07-27 — Phase 6 portable aligned-evaluation layer

- Read the Phase-6 status/matrix before changing code and kept the task in
  `IN_PROGRESS`.  No GPU, simulator, 4096-environment benchmark, paired real
  rollout, git staging, or commit was run in this work item.
- Added a tracker-neutral standard trace with exact motion/sequence/seed/frame/
  timestamp pairing, caller-owned site and tracking-point layouts, explicit
  enabled/active masks, endpoint/reference/pose/force tensors, and terminal,
  success, fall, and post-reset snapshot events.  Portable source contains no
  concrete robot/body mapping or fixed action/keypoint count.
- Added reports for per-site original/selected endpoint RMSE/P95, quaternion
  orientation, local/global MPJPE, force peak/RMS, reference yield and yield
  along force, inactive-site force/yield, paired cross-coupling shift,
  success/fall/reset behavior, stale post-reset force, and input/derived
  finiteness.  Baseline, overlay-off, enabled/no-contact, arbitrary single-site,
  and simultaneous multi-site protocols are validated separately.
- NPZ traces and JSON reports publish atomically with hard compressed and
  uncompressed size limits, pickle-disabled loading, schema-exact fields, and
  no-overwrite defaults.  The thin CPU runner exposes only caller-provided
  endpoint roles; a future simulator collector remains the sole owner of
  concrete endpoint/body names.
- Focused evaluation tests initially passed with `13 passed in 0.84s`; the runner
  `--help` gate and scoped `git diff --check` also exited 0.  The corrected
  environment-split regression passed with `100 passed, 1 skipped in 22.96s`
  for Phase-1/2/3/4 plus evaluation in `sonic_backup`, and `33 passed, 96
  warnings in 2.61s` for deployment in the ORT-equipped `sonic` environment.
- An earlier combined invocation incorrectly ran deployment tests in
  `sonic_backup`; it produced nine `ModuleNotFoundError: onnxruntime` failures
  while `124 passed, 1 skipped`.  This was an environment-selection error, not
  a product-code failure, and was replaced by the passing split runs above.
- Independent review then found five strictness paths that could accept
  incomplete evidence.  The suite now requires baseline, off, no-contact,
  exactly one single-site trial per caller-selected endpoint, and a simultaneous
  multi-site trial; enabled protocols remain enabled on every row; every trial
  constrains force/yield at inactive sites; every trial requires post-reset
  evidence; and alignment compares dtype plus exact bytes so `+0.0/-0.0`
  timestamps cannot match.  Interaction reports now include active-window
  endpoint/orientation/force/yield statistics and cross-coupling paired to the
  overlay-off trace rather than only the released baseline.  New negative tests
  cover each case; the final focused suite passed with `14 passed in 0.91s`,
  CLI help, scoped diff, portable-vocabulary, and cache-hygiene gates all passed.
- Final IO/lifecycle review closed four additional false-accept paths.  NPZ
  loading now uses one `O_NOFOLLOW` descriptor and `fstat` through ZIP and NumPy
  decoding, rejects duplicate members and non-Unicode/non-vector name fields,
  and therefore has no path-reopen TOCTOU.  Each sequence now requires exactly
  one first-row reset snapshot and one final-row terminal, with falls limited to
  terminal rows.  Single/multi interaction sites must exceed configurable
  active-force and active-yield minima, so a mask-only zero-exposure trial
  cannot pass.  The focused suite remained `14 passed in 0.91s`; final scoped
  diff, portable-vocabulary, and cache scans were clean.

## 2026-07-27 — Phase 6 CPU contract final audit

- A final code read found the inactive-yield acceptance check indented after
  the expected-active-site loop, so only its last site was checked.  It now
  applies to every caller-owned site in every trial; a single-site regression
  injects yield only at the other inactive endpoint and requires the named
  failure.
- The complete Phase-1/2/3/4 plus evaluation suite passed with `101 passed, 1
  skipped in 25.00s`; the correctly separated ORT deployment suite passed with
  `33 passed, 96 warnings in 2.75s`.  Both Phase-6 CLI help gates and
  `git diff --check` exited 0, and repository cache scanning was clean.
- Added a separate 4096-environment fixed-shape scheduler benchmark entrypoint.
  It records CUDA-event time per policy update plus allocated/reserved peak
  increments for host-off and enabled candidates, labels the result
  scheduler-only rather than end-to-end policy latency, and atomically refuses
  an existing evidence path.  Only its CPU `--help` gate ran in this handoff;
  the required compatibility-CUDA measurement remains pending.
- Phase 6 remains `IN_PROGRESS`.  No synthetic CPU trace or prior Phase-4 GPU
  smoke is accepted as the missing real paired baseline/off/no-contact,
  single-left, single-right, simultaneous, or 4096-environment evidence.

## 2026-07-27 — Phase 6 resumable handoff prepared

- Added `phase6_handoff.md` as the branch-local source of truth for implemented
  architecture, accepted model/artifact hashes, validation results, known
  limitations, portable migration boundaries, and the exact continuation
  order after a machine restart.
- Expanded `status.md` so the current CPU/evaluation work and every missing
  GPU/simulator gate are visible without reconstructing the full log.
- Phase 6 remains `IN_PROGRESS`; no prior smoke or CPU-generated trace was
  promoted to final tracking/compliance evidence.
- Prepared all Phase-5 deployment and Phase-6 CPU evaluation work for
  publication on the isolated `experiment/motion-compliance` branch only. The
  protected main branch and all pre-existing remote refs remain out of scope.

## 2026-07-28 — Phase 6 SONIC collector/evidence implementation, paused before GPU

- Re-read Phase-6 status and matrix and stayed within the current phase. No GPU
  training, real IsaacLab protocol collection, or 4096-environment benchmark
  was launched in this work item. The user then requested an explicit pause;
  Phase 6 remains `IN_PROGRESS` and the task remains `NOT_COMPLETE`.
- Added a thin SONIC evaluation adapter and IsaacLab recorder bridge. They map
  simulator state into the tracker-neutral trace using the reference
  `torso_link` full-pose frame for endpoints/force and one shared reference-
  pelvis orientation basis for root-relative 14-point errors. Original
  reference arrays are read directly, so candidate robot-anchor pose cannot
  contaminate strict alignment.
- Added a one-trial real collector with five formal roles: official baseline is
  host-off; overlay-off uses the accepted step-6 actor with host plumbing on but
  logical condition/sites off; no-contact/single/multi use host and logical
  compliance on. It pins G1-only robot-motion encoder selection, release
  14-point body order, plane terrain, 50 Hz policy timing, first-frame audited
  motion selection, relaxed eval termination names/order, and reset-only event
  names. A 2500-step limit is only a fail-safe: publication requires the
  observed natural motion timeout exactly once on the final expected full-clip
  step.
- The collector distinguishes official versus step-6 checkpoint schemas and
  hashes, records exact per-step action bytes, requires baseline/overlay-off
  parity, reads actual body-local rows from the permanent wrench composer,
  transforms site force into the shared reference frame, compares composer
  rows against command buffers, and explicitly clears every owned force/torque
  row after timeout. It records the exact 10 N, 0.05 m, derived 200 N/m
  stimulus, actual initial condition/gates, motion provenance, step timing,
  coordinate semantics, and a full trace SHA-256.
- Extended portable alignment to pin exact reference global/local points and
  original endpoint poses in addition to identity/layout fields. Added measured
  endpoint yield versus overlay-off, yield projection along actual force,
  inactive-hand RMSE/P95 cross-coupling, and zero-fall/full-success requirements.
  Fixed a real false rejection: an always-active wrist has no inactive samples,
  so inactive checks are now emitted only when an inactive sample exists;
  genuinely inactive sites/rows remain checked.
- Added same-descriptor `load_trace_npz_with_sha256`. The portable runner writes
  all six full-file hashes into its report. A separate SONIC final validator
  safely loads summaries/traces, binds collection/observed/paired hashes,
  reloads and recomputes the complete portable report under exact criteria,
  pins six-protocol wrist semantics, audited motion/checkpoint hashes,
  environment/gates/stimulus/timing/termination/composer evidence, and exact
  baseline/off action parity. Its JSON equality is recursive and type-aware so
  `false` cannot impersonate integer zero.
- Adversarial CPU tests cover changed reference bytes, lax criteria, deleted
  portable checks, rebound/replaced NPZ hashes, altered motion hash, changed
  stimulus or frame time, duplicate/wrong wrist protocols, boolean IDs, stale
  force, missing measured yield, cross-coupling, fall/success, and checkpoint/
  event role errors. Final focused result was `38 passed in 1.46s`; independent
  review reran `38 passed in 1.41s`. Related CLI help, AST parsing, and
  `git diff --check` passed. Compile-generated repository caches were removed.
- Before the final collector patches, continuity checks also passed for the
  Phase-5 deployment suite (`33 passed`), official residual contract,
  trainer help/config, generic/SONIC C++ ORT smoke, production target build and
  CLI acceptance, accepted artifact revalidation, pinned file hashes, and the
  immutable release boundary. Because the collector changed afterward, this is
  not recorded as the final full Phase-6 matrix-item-1 regression.
- Two P1 evidence gaps remain deliberately open at the pause boundary:
  termination/event validation pins names but not exact function targets and
  all config parameters; and Phase 6 uses explicit post-timeout cleanup rather
  than exercising the configured reset event after nonzero force. The source
  config is currently correct and Phase 2 covered real reset, but both items
  must be made fail-closed before formal six-protocol collection.
- Formal pending work remains unchanged: final full regression and Phase-2/3
  smokes; fresh 16-environment/five-iteration FPS-memory smoke; real 4096-env
  scheduler measurement; six full paired traces; portable plus SONIC final
  acceptance; active-mode wrist endpoint/orientation/whole-body tracking review;
  and final output/cache/diff hygiene. See `phase6_handoff.md` for exact paths
  and continuation order.

## 2026-08-12 — Phase 6 paused again before execution

- Restored the isolated CHIP and motion worktrees after temporary worktree
  directories had been cleaned, then re-read the current status, handoff, and
  Phase-6 matrix before acting. The motion branch and its remote were both at
  `7340d2b0b0571bd225512196c06e60aa527b745a`, with a clean worktree.
- The user requested another pause before implementation or execution. The
  in-progress P1 helper was interrupted before it wrote any code. No CPU test,
  IsaacLab launch, training, scheduler benchmark, or six-protocol collection
  was run, and no formal Phase-6 result was promoted.
- Confirmed that both P1 gaps in the committed collector remain open. The
  reserved outputs `phase6_residual_gpu_smoke_post_restart`,
  `phase6_scheduler_4096.json`, and `phase6_real_paired` are absent.
- The earlier `/tmp/nvidia_580_159_compat` environment no longer exists. The
  host has a natively matched NVIDIA kernel/CUDA/NVML stack at `580.173.02`.
  Phase-6 documentation now supersedes the historical temporary-loader command
  while preserving old Phase-2/3/4 commands as provenance.
- The RTX 4090 was occupied by unrelated GRAIL replay processes during the
  inspection. They were not task-owned and were not terminated; future formal
  timing or memory evidence must wait for a separately verified idle window.
- Phase 6 remains `IN_PROGRESS` and the task remains `NOT_COMPLETE`. Resume at
  the two P1 evidence-hardening items; then run the focused suite, matrix item
  1, and items 2–9 in order. Do not infer tracking/compliance performance from
  the present CPU-only collector contract.

## 2026-08-12 — Phase 6 P1 evidence hardening completed

- Resumed only current Phase 6 and left CHIP/main untouched. Revalidated the
  motion branch against official baseline `4141c34280ab`, all four pinned
  model/data hashes, the absent formal Phase-6 output paths, and the native
  NVIDIA `580.173.02` stack. The RTX 4090 remained occupied by unrelated
  processes, so no GPU evidence command was launched.
- Added a type-strict manager provenance contract that binds both composed
  Hydra strings and actual IsaacLab term configs: four resolved termination
  functions, their timeout flags and effective defaults, the sole reset event,
  reset mode/minimum interval, command names, thresholds, asset/body names,
  and all declared parameters. A real Hydra composition matched the pinned
  contract.
- Interaction collection schema v3 now waits for observed nonzero owned force,
  invokes the actual configured reset event via `event_manager.apply`, and
  verifies command and permanent-composer force/torque rows are exactly zero.
  Baseline/off/no-contact must not fabricate this evidence. The final SONIC
  validator binds the manager contract and rejects zero-before, stale-after,
  wrong-mode/function, or inactive-protocol reset evidence.
- Focused Phase-6 evaluator/collector validation passed `40 tests in 1.40s`;
  all four CLI help gates, AST parsing, and `git diff --check` passed. This
  closes the two P1 code/test gaps but does not satisfy matrix item 1 or any
  pending performance/tracking gate.

## 2026-08-12 — Paused after post-P1 CPU checkpoint

- The user requested a pause before further execution. No new training,
  Phase-2/3 real IsaacLab smoke, 4096-environment scheduler benchmark, or
  six-protocol simulator collection was started.
- Post-P1 CPU evidence obtained before the pause: combined core/adapter/
  training/checkpoint/evaluation/collector suite `127 passed, 1 skipped in
  25.63s`; deployment suite `33 passed, 96 warnings in 3.14s`; official
  residual contract, trainer help/config, official Phase-4 residual init, four
  Phase-6 help gates, and real Hydra manager-provenance composition passed.
- Step-5 and step-6 checkpoint audit JSONs were atomically regenerated and
  read back with the expected steps, twelve changed residual tensors, frozen
  55/17 base tensors, and twelve optimizer slots. Their combined orchestration
  output was truncated before exit statuses were retained, so the exact audit
  commands remain mandatory on resume and are not promoted as final evidence.
- The three reserved formal Phase-6 outputs remain absent:
  `phase6_residual_gpu_smoke_post_restart`, `phase6_scheduler_4096.json`, and
  `phase6_real_paired`.
- The environment changed during the checkpoint. An earlier native query saw
  matched NVIDIA `580.173.02` with unrelated GPU jobs; the final two
  `nvidia-smi` queries failed to communicate with the driver. No unrelated
  process was touched. GPU availability is now an explicit resume precondition.
- Branch checkpoint before documentation: local and tracked remote
  `experiment/motion-compliance` both at
  `30f2190d1b70321705e92dde2b5c004fc8bee6d4`. Main and CHIP worktrees were
  clean and untouched. Phase 6 remains `IN_PROGRESS`; task remains
  `NOT_COMPLETE`.

## 2026-08-12 — Phase 6 resumed; matrix item 1 and tracking gates

- Revalidated the clean motion branch at `5c0b20e`, official merge base, four
  pinned asset hashes, SMPL directory, untouched main/CHIP worktrees, and
  absent formal Phase-6 output paths.
- Determined that failed sandboxed NVML probes were caused by device namespace
  isolation: the host retains matched NVIDIA `580.173.02`, RTX 4090, and
  PyTorch 2.7/CUDA 12.8 availability. No device node, driver module, or
  unrelated process was changed.
- Completed matrix item 1 with retained zero exits: combined suite `127 passed,
  1 skipped`; deployment `33 passed`; step-5/step-6 checkpoint audits;
  official residual/init and trainer config gates; Phase-5 artifact acceptance,
  C++ ORT, production configure/build/help/CLI, and release-boundary gate; and
  native Phase-2/3 real IsaacLab smokes.
- Fixed active tracking limits before any formal trace. Caller-owned wrist site
  to tracking-point mapping excludes only an intentionally yielded point per
  active non-reset row. Selected endpoint RMSE/P95 regression is bounded at
  5/10 mm, original-target orientation at 0.05/0.10 rad, and remaining-point
  local/global MPJPE at max(3 mm, 10%)/max(5 mm, 10%). The portable schema is
  now `compliance_evaluation_v2`, and the SONIC validator recomputes and pins
  every mapping, value, and check. Focused validation passed `43 tests in
  1.58s`.
