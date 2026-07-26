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
- Default physical sites are resolved by body name in the SONIC adapter; the
  core remains site-count agnostic and the configuration supports more sites
  without changing the policy contract.
- Validate SONIC endpoint body-name/offset metadata in this adapter phase, not
  in the reusable core.
- Add an opt-in command composition under
  `config/manager_env/commands/`; leave the standard command composition
  unchanged.

### Phase 3 — Observation, reward, and experiment composition

- Add the 3D public compliance condition to policy proprioception and
  threshold/force/site-mask state to the critic only.
- Keep the robot-motion tokenizer inputs unchanged.
- Add compliance-specific rewards that use yielded targets only at active
  interaction sites; retain original targets for every inactive site.
- Add explicit upper-endpoint tracking metrics/rewards and an opt-in
  `sonic_release_motion_compliance.yaml` experiment.
- Verify off-mode config behavior against `sonic_release`.

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
