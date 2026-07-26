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
