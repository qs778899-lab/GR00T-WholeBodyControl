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

### Phase 3 — Observation, reward, and experiment composition

- Append the 3D public compliance condition to policy proprioception and expose
  the same public condition plus raw threshold/current applied site force/site
  mask to the critic.  Privileged widths follow the configured site count.
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
  and in one real manager environment, including identical interval-event and
  global-RNG contracts.

### Phase 4 — Released-checkpoint migration and finetune workflow

- Add a shape-aware checkpoint adapter for the dynamic decoder/critic input
  expansion.  Copy all old columns exactly and zero-initialize new compliance
  columns so the initial `enable=0` policy is functionally unchanged.
- Keep normal strict resume semantics for already-migrated checkpoints.
- Add documented Hydra CLI examples for sample/full motion data.  Compliance
  forces are synthesized online; no duplicate modified motion dataset is
  required.
- Use the audited official release checkpoint at
  `compliance_control/official_assets/sonic_release/last.pt` (HF revision
  `7c90a56c`, recorded SHA-256 ending `d8909`, training step 41550) for the real
  migration smoke; never mutate the source asset.
- Add staged finetuning controls: initially freeze the robot-motion encoder and
  quantizer, train the dynamic decoder/critic, then optionally unfreeze.

### Phase 5 — Export and deployment switch

- Export the migrated encoder/decoder without changing encoder inputs.
- Add a deployment observation named `motion_compliance_condition` and explicit
  runtime `enable`/threshold options in a new observation config; do not modify
  the released config in place.
- Confirm disabled deployment supplies exactly three zeros and reproduces the
  baseline output within tolerance.

### Phase 6 — Integration, regression, and low-resource validation

- Run config/help/compile/unit tests plus IsaacLab smoke tests.
- Run paired stiff-mode baseline regression on fixed motion IDs/timestamps.
- Run single- and simultaneous-site compliant-force evaluation.
- Record endpoint RMSE/P95, orientation error, MPJPE, success rate, force peak,
  yielded displacement, throughput, and memory.
- Mark complete only if all phase-6 acceptance criteria in `test_matrix.md`
  pass or any difference has a documented explanation.
- The prescribed low-resource smoke uses the `sonic_backup` environment, one
  audited robot PKL plus `sample_data/smpl_filtered`, 16 environments, 5
  iterations, and `use_wandb=false`.  Record the current NVIDIA kernel/user-space
  driver mismatch as an external blocker until GPU execution is available.
