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
