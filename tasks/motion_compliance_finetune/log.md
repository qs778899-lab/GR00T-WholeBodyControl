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
  (HF revision `7c90a56c`, recorded SHA-256 ending `d8909`, step 41550) and six
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
