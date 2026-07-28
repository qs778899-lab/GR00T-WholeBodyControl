# CHIP compliance finetune log

## 2026-07-27 — Phase 1 started

- Confirmed clean experiment worktree on `experiment/chip-compliance` at `4141c34280ab`.
- The worktree has no `AGENTS.md`; the repository instructions supplied by the user are being applied directly.
- Audited the released `sonic_release` config, 14-body tracking order, training entrypoint/dependencies, and dormant CHIP-related source.
- Confirmed the dormant compliant observation is not composed into the release config, mutates the reference tensor in place, and expands a `[env, force_body]` magnitude buffer with too many singleton dimensions.
- Confirmed the current CPU environment has Torch but no Isaac Lab or pytest; Phase 1 uses standard-library `unittest`, and simulator/training tests remain gated to later phases.
- Added the portability constraint: tracker-agnostic `schema/math/schedule/metrics` core, with SONIC-specific body-name/MDP/Hydra/checkpoint logic kept in a thin outer adapter and no fixed 29-DoF/14-body index tables.
- Standardized the reusable API paths as `gear_sonic/compliance_control/core` and `gear_sonic/compliance_control/adapters/sonic` so parallel experiments can share one contract.
- Recorded the official checkpoint (HF revision `7c90a56c`, SHA-256 ending `d8909`, step `41550`), six sample PKLs, and exact later `sonic_backup` 16-env/5-iteration smoke inputs. These assets remain outside the worktree and must not be committed.
- Recorded the current Phase 4 GPU blocker: NVIDIA kernel `580.159` and userspace `580.173` do not match.

## 2026-07-27 — Phase 1 passed

- Added tracker-agnostic `schema`, hindsight-target `math`, force `schedule`, and response `metrics` under `gear_sonic/compliance_control/core`.
- Added the SONIC boundary adapter under `gear_sonic/compliance_control/adapters/sonic`; it resolves runtime body names supplied by configuration and contains no robot-specific index table.
- Kept all existing SONIC MDP, Hydra, model, checkpoint, training, and deployment files unchanged.
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v`: PASSED, 13 tests.
- Source compile smoke and no-IsaacLab import smoke: PASSED.
- AST portability checks for forbidden core imports and fixed `14`/`29` numeric contracts: PASSED.
- `git diff --check`: PASSED.
- Advanced `current_phase` to 2 but paused before executing it, as required by the scoped handoff.

## 2026-07-27 — Phase 1 reopened after review

- Review found that a global-only enable flag cannot train stiff and compliant environments in the same batch.
- Review also found that scalar-only inverse stiffness prevents Cartesian-axis anisotropy needed to preserve wrist tangential tracking while yielding along contact force.
- Reopened Phase 1 to add boolean hard gates shaped `[batch]` or `[batch, future, site]`, plus isotropic and anisotropic compliance layouts and regression tests.

## 2026-07-27 — Phase 1 review fixes passed

- Added mixed-batch hard gates as either a global bool, `[batch]`, or `[batch, future, site]` boolean tensor. Disabled tensor elements select `reference_positions` through `torch.where` for exact stiff identity.
- Added isotropic static/future and Cartesian-axis anisotropic static/future compliance layouts. Ambiguous three-dimensional layouts are rejected with an explicit disambiguation instruction.
- Verified an anisotropic wrist-style example yields only along selected axes and retains the other Cartesian target components.
- Reran every Phase 1 test: 15 unit tests, source compile/import smoke, no-IsaacLab/portability AST audit, final-newline/trailing-whitespace audit, and `git diff --check`; all passed.
- Advanced `current_phase` to 2 and paused without executing any Phase 2 simulator or Hydra work.

## 2026-07-27 — Phase 1 reopened after independent review

- Reopened Phase 1 before any simulator work to harden dual index-space resolution, structured Cartesian-frame metadata, finite-value and hard-off behavior, exposure metrics, reusable target damping, and string-sequence validation.

## 2026-07-27 — Phase 1 independent-review fixes passed

- Added independently resolved, typed reference-motion and articulation index spaces. A two-order regression proves the same semantic sites map to different correct integer tuples, and `SonicComplianceSites` rejects any order mismatch with `spec.site_names`.
- Replaced free-form frame strings with restricted `CartesianFrameKind`, `CartesianRotation`, and `CartesianFrameSpec` contracts containing an explicit semantic anchor. Target and force frames must be structurally equal.
- Added strict finite reference/force validation and finite non-negative compliance validation. Global hard-off bypasses unused NaN operands safely; mixed hard gates require valid operands and produce zero force/compliance gradients in disabled rows.
- Corrected metrics so `active_fraction` and displacement summaries use actual exposure: enable gate, requested site mask, and positive compliance.
- Added portable `TargetDamper` state implementing `g_t = alpha * x_eef + (1 - alpha) * g_prev`, detached stored state, full reset, and partial per-environment reset.
- Rejected `str` and `bytes` anywhere a name sequence is expected, while retaining arbitrary caller-supplied site counts and no embodiment index tables in the core.
- Expanded Phase 2 acceptance to require a known non-zero rotation/frame/sign test and typed dual-index consumers; Phase 3 must connect the damped/hindsight target only to the compliance path.
- The first expanded hygiene run found extra EOF blank lines in two marker `__init__.py` files; both were fixed and the same matrix was rerun.
- Final Phase 1 matrix: 23 unit tests, compile/import smoke without Isaac Lab, portability AST audit, exact-one-newline/trailing-whitespace/cache audit, `git diff --check`, staging of only Phase 1 scope, and `git diff --cached --check`; all passed.
- Phase 1 is `PASSED`; `current_phase` is 2 and remains paused. No Phase 2 implementation, commit, or push was performed.

## 2026-07-27 — Phase 2 started

- Confirmed `current_phase: 2` and Phase 1 `PASSED` before beginning.
- Scope is limited to a portable SONIC/IsaacLab boundary adapter, force event/reset, non-mutating observation, structured-frame transforms, target-damper lifecycle, and a separate opt-in Hydra composition. Released configuration and Phase 3 policy/checkpoint work remain out of scope.

## 2026-07-27 — Phase 2 passed

- Added independent name-resolved reference/articulation site selection; common-frame position/vector transforms; non-mutating current-force hindsight targets; per-environment pulse/damper state; opt-in command/events/observation Hydra groups; and a bounded one-environment simulator smoke. The released `sonic_release.yaml` remains unchanged.
- Matched CHIP's training sampling envelope: uniformly sampled `0–40 N`, `1–3 s` pulses and discrete compliance `{0, 0.02, 0.05} m/N`. Retained SONIC-specific `30 N` resultant-force and `20 N·m` resultant-torque caps, applied again from current lever arms every step; state and hindsight use the final applied force.
- Converted every world force with the current link quaternion and wrote link-local force plus link-local site offset using `is_global=False`. Numeric tests and the real smoke reconstruct composer force/offset torque back to the world contract.
- Added strict writer ownership: disabled-from-start never touches the composer; active-to-off clears once; a disabled reset consumes ownership before the next update; steady disabled steps do not rewrite. Cached the full-environment ID tensor to avoid a per-policy-step allocation at large environment counts.
- Seeded only newly active damper sites from the current end effector, retained exact inactive/reference identity, and documented that current force is repeated across future target frames for alignment rather than predicted.
- Split public checked functions from explicit prevalidated simulator boundaries. CPU/CUDA `TorchDispatchMode` plus profiler tests found no `aten::_local_scalar_dense` in the production hot path.
- Final local matrix: portable suite PASSED (45 tests, four expected CUDA/Hydra skips); `sonic_backup` compatibility-driver suite PASSED (45/45, no skips); dedicated CUDA profiler PASSED (1/1); import/syntax/hygiene and release-config diff checks PASSED.
- Final real smokes PASSED: disabled 100 steps reported `0 N / 0 N·m`; enabled 100 steps reported `8.457119 N / 2.564395 N·m`, below `30 N / 20 N·m`, then passed two same-process disabled steps and reset checks. The smoke failure path was separately verified to return exit status 1.
- Independent rerun also PASSED: CPU 45 tests with only two CUDA skips, full compatibility-driver CUDA/Hydra 45/45, disabled 100-step smoke, and enabled 100-step plus active-to-off two-step smoke with the same peak metrics.
- Phase 2 is marked `PASSED`; `current_phase` advances to 3, but no Phase 3 policy/critic/checkpoint work was executed in this handoff.

## 2026-07-27 — Phase 2 reopened for RNG/event parity regression

- Review found two release-parity violations while compliance was disabled: inherited `CommandTerm._resample` sampled `time_left` from the process-global CUDA RNG, and the compliance interval event made `EventManager` consume global RNG during reset/scheduling.
- Replaced both scheduling paths with a command-owned per-environment pulse countdown and private `torch.Generator`. The command overrides `_resample`, keeps inherited `time_left` infinite, schedules only selected environments, and never samples either global RNG while disabled. The original `3.5–6.0 s` pulse interval is now an explicit command setting.
- Removed only the compliance interval event while retaining the reset event. Hydra regression proves the compliance experiment has exactly the release interval-event set and ranges (`push_robot: 4–6 s`). The smoke-only fast interval is now a command override.
- Added CPU and CUDA tests for exact global RNG-state preservation, private-generator determinism, due-environment scheduling, and unaffected partial-reset rows. Source inspection also proves the `_resample` override has no global uniform/random call.
- Added `set_operational_enabled(bool)` as the only runtime switch. It leaves static Hydra config untouched; disable synchronously cancels pulse/state/countdowns and consumes writer ownership through `ArticulationWrenchAdapter.clear()`, which addresses only the configured application-body rows. Enable schedules all environment countdowns only through the command's private generator.
- A portable execution test and the real enabled smoke preseed owned non-zero rows plus an unrelated sentinel. Before any environment step, disable clears only the owned rows, preserves the sentinel, consumes the write gate, and leaves `cfg.enabled` unchanged. Re-enable changes only private RNG state and finite countdowns; a second immediate disable cancels them.
- Final portable discovery PASSED (62 tests, 11 expected CUDA/Hydra skips); the focused Phase 2 subset PASSED (49 tests, six expected skips). Full `sonic_backup` CUDA/Hydra discovery PASSED (62/62, no skips), and the dedicated CUDA profiler again found no `aten::_local_scalar_dense`.
- Real one-environment smokes PASSED for 100 steps each. Disabled mode reported `0 N / 0 N·m` and preserved exact global CPU/CUDA RNG state across direct reset/update. Enabled mode reported `7.825543 N / 3.313526 N·m`, below the configured `30 N / 20 N·m` limits, then passed active-to-off and reset checks.
- Independent review reran the final `sonic_backup` discovery (62/62) and both real 100-step smokes with the same `0 N / 0 N·m` disabled and `7.825543 N / 3.313526 N·m` enabled metrics, including the immediate setter/sentinel assertions; all passed.
- Phase 2 is again `PASSED`; `current_phase` advances to 3. Preserved Phase 3 work was not included in this regression-fix scope.

## 2026-07-27 — Phase 2 reopened for 4096-environment CUDA host-sync regression

- Review found that the production due-pulse route re-entered strict public boundaries on every due batch. CUDA tensor-value checks for ID range/uniqueness, finite samples, non-negative compliance, and positive duration could materialize `aten::_local_scalar_dense`; both `(countdown <= 0).nonzero()` and inherited `CommandTerm.compute()` also created dynamic-length CUDA indices every step.
- The command mixin now overrides `compute(dt)` and never enters the inherited resampling implementation. Enabled compute samples fixed `[num_envs, ...]` candidates through the private generator, derives fixed-size `due_mask`/`start_mask`, and updates countdown, pulse state, damper seeds, and completion cleanup through `torch.where`/`copy_`; no dynamic due IDs are constructed. Disabled compute cancels state and keeps countdowns infinite without consuming either private or process-global RNG.
- Added explicit prevalidated sampling, force-mask, resultant-wrench limiter, masked pulse-start, masked damper-seed, and damper-reset interfaces. Strict public APIs retain all validation and have exact seeded parity coverage against the prevalidated route. The profiler initially exposed eight `aten::_local_scalar_dense` calls from boolean pulse-finish assignments; replacing those writes with fixed-shape masks removed them.
- Moved the complete fixed-shape production step into the portable mixin inherited first by `SonicComplianceCommand`: countdown, candidate sampling, state start/advance, net-wrench limit, body-local writer, and target-damper update. Cached articulation site indices, expanded local offsets, and discrete compliance tensors avoid repeated pose/index setup, and current site positions/common-frame targets are computed once per step.
- Portable discovery PASSED (65 tests, 12 expected CUDA/Hydra skips); the Phase 2-only subset PASSED (52 tests, seven expected skips). A bound `compute(0.02)` CPU profiler executed 4096 environments × 14 sites through the complete production-equivalent writer path with both `TorchDispatchMode` and `torch.profiler`, found neither `aten::nonzero` nor `aten::_local_scalar_dense`, and preserved global RNG state.
- Full `sonic_backup` CUDA/Hydra discovery PASSED: 65/65, zero skips, 2.843 s unittest time and 3.874 s command wall time. The dedicated 4096-environment × 14-site CUDA bound-compute test PASSED 1/1 in 0.346 s unittest time / 1.308 s wall time, with both dispatch and profiler rejecting `aten::nonzero` and `aten::_local_scalar_dense`.
- Real disabled and enabled one-environment smokes PASSED for 100 steps each (exit 0; 18.030 s and 18.424 s wall time). Disabled reported `0 N / 0 N·m` and passed exact global CPU/CUDA RNG, infinite countdown, no-writer, and reset assertions. Enabled reported `6.785363 N / 2.197217 N·m`, below the `30 N / 20 N·m` limits, and passed body-local force reconstruction, local-offset torque, immediate-off, unrelated-row preservation, re-enable/private-RNG, second-off, two disabled-step, and reset assertions.
- Phase 2 remains `IN_PROGRESS` for parent review; no Phase 3 work was staged or executed as part of this regression fix.

## 2026-07-27 — independent real Isaac-bound profiler correction

- Independent review found that the 4096-environment × 14-site profiler bound only `ComplianceOperationalControl` to deterministic tensors and a fake composer. It remains useful as a portable scale/arbitrary-site audit, but is no longer described as an actual Isaac-bound instance.
- Added a second audit to the real one-environment AppLauncher smoke. It runs only after the complete 100-step disabled baseline passes, temporarily enables the retrieved `SonicComplianceCommand`, forces its private countdown due, and directly profiles `command.compute(env.step_dt)` with `TorchDispatchMode` plus CPU/CUDA profiler activities. The trace covers articulation `index_select`, link-offset forward/inverse frame helpers, `ArticulationWrenchAdapter`, and the real Isaac Lab `WrenchComposer`; the portable scale fixture retains optional target-damper coverage.
- The real trace rejects both `aten::nonzero` and `aten::_local_scalar_dense`, preserves process-global CPU/CUDA RNG, requires private-RNG consumption and a non-zero real composer write, then disables before any environment step or profiler-result inspection. Immediate-off checks require command state/countdown/ownership and selected real-composer force/torque rows to be zero.
- Portable full discovery PASSED: 65 tests, 12 expected CUDA/Hydra skips. The new static contract proves disabled-loop → real trace ordering and operational enable → bound compute → immediate disable ordering. Real AppLauncher CUDA trace and the subsequent normal forced smoke remain pending; Phase 2 stays `IN_PROGRESS`.
- The first real disabled-smoke attempt stopped during environment construction because a smoke-only `target_damper_enabled=true` override made ObservationManager read the damper before its first reset. The override was removed: it is neither needed for the requested real writer/synchronization audit nor part of the derived experiment's default path. The failed attempt did not reach the 100-step baseline or trace and did not authorize the enabled smoke; the same disabled group must pass before proceeding.
- Final CUDA sequence ran strictly in order. Full `sonic_backup` discovery PASSED 65/65 with zero skips (2.876 s unittest / 3.896 s command wall time). The corrected disabled AppLauncher smoke then PASSED in 18.584 s: 100 steps remained at `0 N / 0 N·m`, exact CPU/CUDA RNG and no-writer assertions passed, `CHIP_PHASE2_REAL_BOUND_PROFILE_PASS` reported forced private due with clean dispatch and CPU/CUDA profiler traces, and immediate off left selected real-composer force/torque rows at zero before another environment step. `CHIP_PHASE2_SMOKE_PASS` also appeared.
- Only after the real trace passed, the separate enabled AppLauncher smoke ran and PASSED in 18.291 s. It observed non-zero force with peak net force `6.785363 N` and peak net torque `2.197217 N·m`, below `30 N / 20 N·m`; body-local reconstruction, local-offset torque, unrelated-row preservation, private/global RNG separation, immediate double-disable cleanup, two disabled steps, and reset checks all passed. Phase 2 remains `IN_PROGRESS` solely for parent review; no Phase 3 file was modified or staged by this correction.
- The final post-smoke portable discovery rerun PASSED 65 tests with the same 12 expected CUDA/Hydra skips (2.474 s unittest / 3.486 s command wall time), confirming the corrected smoke source and its AST ordering contract. Compile/import, EOF/cache/trailing-whitespace hygiene, and staged/unstaged diff checks also passed.
- Parent independent rerun of the final disabled AppLauncher group also PASSED (exit 0, 18.174 s wall time). It observed both `CHIP_PHASE2_REAL_BOUND_PROFILE_PASS` and `CHIP_PHASE2_SMOKE_PASS`, retained `0 N / 0 N·m` throughout the disabled baseline, and independently confirmed forced-due dispatch/profiler cleanliness, exact global RNG preservation, and immediate real-composer force/torque row clearing. With the full 65/65 suite, this independent rerun, and the separate enabled 100-step `6.785363 N / 2.197217 N·m` result all passing, Phase 2 is marked `PASSED`; only the current phase advances to Phase 3, and the preserved Phase 3 WIP remains unstaged.

## 2026-07-27 — Phase 3 implementation and validation pending review

- Added an opt-in, configurable-site post-FSQ 64D actor residual and a separate privileged critic value residual. The actor filters direct forwards and rollout history to `actor_obs`, the unchanged release tokenizer, `compliance_target`, and `compliance_command`; applied force is available only to the critic.
- Zero-initialized, hard-gated output heads preserve the default-off release actor output byte-for-byte. First enabled backward gives nonzero actor and critic output-head gradients; zero head initialization intentionally makes every trunk gradient zero on that first backward.
- Froze the release G1/teleop/SMPL encoders, FSQ, G1 dynamic/kinematic decoders, action noise, base critic, and critic running statistics. The only trainable names are the six actor residual and six critic residual parameters.
- The pinned SHA-256 official checkpoint loaded complete 55-policy/17-value legacy schemas. Every legacy tensor remained byte-exact, exactly six initialized residual keys were added to each model, and migrated branch checkpoints resumed strictly even when called with `strict=false`.
- Portable discovery PASSED 72 tests with 19 expected Hydra/CUDA/model skips. The `sonic_backup` CPU/Hydra/official suite PASSED 72 tests with only four expected CUDA skips. The focused official-model integration suite PASSED 6/6.
- After explicit approval, the compatibility-driver CUDA/Hydra suite PASSED 72/72 with zero skips (11.745 s command wall / 10.020 s unittest). The dedicated 4096-environment × 14-site profiler PASSED 1/1 (1.238 s wall / 0.336 s unittest).
- Inherited real Phase-2 smokes PASSED: disabled 100 steps plus real-bound marker in 18.309 s at `0 N / 0 N·m`; enabled 100 steps in 18.315 s at `6.785363 N / 2.197217 N·m`, below `30 N / 20 N·m`.
- The new real Phase-3 one-environment smoke PASSED in 16.400 s. Observation widths resolved to actor/critic/tokenizer/target/command/force `930/1645/1761/60/9/6`; official models produced action/value shapes `(1,1,29)/(1,1,1)` and default-off actor output was byte-exact against the release path.
- One initial CUDA-suite shell invocation never entered Python because an unquoted zsh glob failed immediately; the corrected same command produced the passing 72/72 result above. No Phase-4 training was started.
- Compile/import, release-config diff, final-newline/trailing-whitespace/cache audits, and staged/unstaged diff checks passed. No Isaac/Python/training process remains. Phase 3 stays `IN_PROGRESS` pending independent review.

## 2026-07-27 — Phase 3 independent-review corrections

- Closed the actor privacy override: `compliance_force` is now a hard-coded forbidden key rather than a configurable denylist. Both direct forward and real rollout reject it even after internal allowlist mutation; normal rollout history contains only the four public groups.
- Removed the implicit two-site offset assumption by defaulting local offsets to `null`, and separated the configured compliance future horizon from the release tokenizer horizon. Real actor/critic constructors pass for `(sites, future) = (1,1), (2,10), (5,3), (14,4), (17,7)`.
- Strengthened whole-actor hard gating with artificially nonzero trunk/head parameters: global off and zero compliance remain byte-exact to release, inactive rows remain byte-exact in a mixed batch, and an active row changes.
- The critic now normalizes `critic_obs` exactly once and shares that exact tensor with the unchanged base value path and the residual. The zero-initialized residual preserves the base value bytes.
- Residual construction now uses a forked RNG scope. Release versus compliance actor and critic construction leaves identical CPU/CUDA RNG states and identical following random sequences.
- Review exposed that the inherited direct-std getter clamps the official tensor in place. `SonicComplianceActor` now uses an out-of-place equivalent clamp only for a frozen direct std; the generic Actor, log-std, and trainable direct-std paths are unchanged. Three distribution updates plus an AdamW step preserve the official std bytes and frozen ownership.
- Expanded the baseline byte audit across the released experiment, actor/critic/FSQ definitions, all encoders/decoders, auxiliary loss, dense policy/critic/tokenizer groups, their referenced observation terms, and every released reward term. The recorded compatibility-driver workaround resolves the old Phase-4 launch blocker without weakening the real-GPU requirement.
- Post-correction portable discovery PASSED 79/79 with 26 expected dependency skips. The resolved `sonic_backup` CPU/Hydra/official suite PASSED 79/79 with only four expected CUDA skips; focused resolved integration PASSED 12/12.
- The first authorized Phase-3 real shape-smoke invocation ran inside the managed sandbox, where the GPU device was unavailable. It stopped before environment/model construction (5.42 s, no pass marker) with CUDA error 100 and `RuntimeError: No CUDA GPUs are available`; no loop retry or other GPU test followed that failure.
- After a root-side device/idle check, an explicitly approved compatibility-environment preflight reported `CUDA_AVAILABLE=True` and created a real `cuda:0` tensor. The one authorized smoke retry then PASSED in 16.48 s with `CHIP_PHASE3_SHAPE_SMOKE_PASS`: real observation widths were `930/1645/1761/60/9/6`, action/value shapes were `(1,1,29)/(1,1,1)`, default-off action was byte-exact, and three distribution constructions left the official frozen std byte-exact while returning the release-equivalent clamp.
- Compile/import, release-path, final-newline, trailing-whitespace, cache, and staged/unstaged diff hygiene passed after the review corrections. Phase 3 is marked `PASSED`; Phase 4 becomes the sole `IN_PROGRESS` phase, but no Phase-4 training was started.

## 2026-07-27 — Phase 4 implementation and resume-boundary correction

- Added an isolated residual-only PPO trainer, strict checkpoint/training audits,
  a derived 16-environment five-batch smoke config, and a collision-safe runner
  for stiff step 5, compliant step 5, and an independent one-batch step-6
  resume. The released SONIC config and generic PPO trainer remain unchanged.
- Corrected the actor residual integration for PPO's real `[B=4, S=24]`
  microbatch shape. A resolved model test proves per-timestep target/gate
  alignment after FSQ, unchanged leading action/decoder/aux shapes, exact stiff
  rows, and finite nonzero gradients for all six actor residual tensors.
- Before the final CPU-only repair, the expanded compatibility CUDA/Hydra suite
  passed 93/93 with no skips; the 4096-environment/14-site profiler, both
  100-step simulator smokes, and the real Phase-3 shape smoke also passed. The
  disabled smoke stayed at `0 N / 0 N*m`; the enabled smoke peaked at
  `6.785363 N / 2.197217 N*m`.
- The first formal directory completed stiff and compliant step 5, then stopped
  before resume because file-path execution lacked the repository import root.
  A bootstrap-only fix and subprocess regression closed that issue. Independent
  review of its step-5 checkpoint passed all strict schema, optimizer,
  gradient, exposure, loss, and memory audits.
- The second formal directory again completed both step-5 jobs and independent
  audit, then stopped before batch 6 because the generic loader changed the
  optimizer LR from serialized `2e-5` to checkpoint-argument `1e-5`. Both
  failed-run directories remain untouched; no third formal run was started.
- Root cause is a legitimate split boundary: adaptive KL saved argument LR
  `1e-5`, while the scheduler left the optimizer's serialized LR at `2e-5`.
  `SonicComplianceResidualPPOTrainer.load_checkpoint` now calls the generic
  restoration and then reloads only that same optimizer payload. It leaves the
  restored argument, scheduler, environment, model, and counters untouched.
- A constructive AdamW/LambdaLR unit test reproduces the LR disagreement and
  proves recursive exact optimizer and scheduler restoration while retaining
  argument LR `1e-5`. Phase-4 focused tests passed 15/15 in `sonic_backup` and
  9 runnable plus six expected dependency skips portably. Full CPU discovery
  passed 95 tests with 33 expected dependency skips portably and 95 tests with
  four expected CUDA skips in `sonic_backup`.
- Phase 4 remains `IN_PROGRESS`. No GPU process or new formal acceptance
  directory was launched for this correction; step-6 validation still requires
  explicit authorization.
- The pinned training `--help` and runner `--dry-run` gates exited zero; the dry
  run printed the three exact commands and left its requested path absent.
  Syntax/core import, 46-file EOF/trailing-whitespace/cache, and both unstaged
  and staged diff checks passed. `/proc` contained no training/Isaac process,
  and compatibility-NVML `nvidia-smi` reported no compute application.

## 2026-07-27 — Phase 4 passed

- After explicit one-run authorization, the fresh canonical root
  `phase4_acceptance_resume_fix` ran serially with the pinned `580.159.03`
  compatibility environment. Stiff step 5 passed in 23.272135 s, compliant
  step 5 passed in 24.701326 s, and the independent one-batch resume reached
  step 6 in 17.529439 s. The runner emitted `CHIP_PHASE4_FINETUNE_PASS`; no
  concurrent or retry GPU job was started.
- Fresh-process independent audits passed for both checkpoints. Step 5 and
  step 6 each preserve all 55 policy and 17 value legacy tensors byte-exactly,
  contain exactly six actor plus six critic residual tensors, and retain
  optimizer ownership of 12 tensors/770753 scalars. Finite loss steps are
  exactly `[1,2,3,4,5]` and `[6]`; site exposures are `[79,79]` and `[16,16]`;
  peak CUDA allocations are 727262208 B and 407661056 B. Every residual tensor
  has nonzero gradient history, with minimum counts 99 and 20 respectively.
- The completed workflow occupies 318016496 B across 32 files; its largest log
  is 55249 B. The resume source is a symlink to the preserved step-5 checkpoint,
  and the official checkpoint still hashes to
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
- A first ad-hoc read-only artifact assertion expected `workflow.complete`, but
  the documented workflow schema uses `status: PASSED`; correcting only that
  inspection produced `PHASE4_ARTIFACT_GATE_PASS` without changing artifacts.
- Host `/proc` and compatibility-NVML checks found no residual training/Isaac
  process or GPU compute application. Final compile/core-import, cache/EOF/
  whitespace, release/generic-trainer unchanged, and staged/unstaged diff
  checks all passed.
- Phase 4 is `PASSED`; `current_phase` advances to 5. No Phase-5 evaluation or
  export work was started, and Phase-4 changes remain unstaged and uncommitted.

## 2026-07-27 — Phase 4 final audit correction

- A final read-only audit found no P0 issue. It found one P1 documentation
  issue: the purported canonical rerun path named an already-retained failed
  directory, which the collision-safe runner correctly refuses to reuse. The
  plan and test matrix now identify `phase4_acceptance_resume_fix` solely as
  immutable accepted evidence and use `<fresh-run-root>` for reproduction; the
  replacement must be a unique nonexistent child of the centralized run root.
- The audit also found one P2 evidence-label issue. The runner measured bytes
  before replacing its small RUNNING manifest with the larger PASSED manifest,
  but called the field `workflow_bytes`. Future workflows call it
  `workflow_bytes_before_final_manifest` and recheck both the final workflow
  and final largest-log limits after the PASSED manifest is written. This avoids
  a self-referential exact-size field while retaining a hard final capacity
  gate.
- The accepted training directory and every training artifact remain byte-for-
  byte untouched. Its legacy manifest field retains the earlier pre-final
  measurement `318014905`; the independent symlink-safe final measurement
  recorded in status remains `318016496` bytes.
- New CPU contract tests lock both distinctions. Focused Phase-4 tests passed
  17 discovered tests with six expected dependency skips portably and 17/17 in
  `sonic_backup`. Full discovery passed 97 tests with 33 expected dependency
  skips portably and 97 tests with four expected CUDA skips in `sonic_backup`.
  The training `--help` gate passed; runner dry-run printed all three commands
  and left the requested directory absent before and after execution.
- Phase-4 syntax/import, EOF/trailing-whitespace/cache, staged/unstaged diff,
  and release/generic source gates passed after these corrections. Phase 4 is
  again `PASSED`; `current_phase` returns to 5 without starting Phase-5 work.

## 2026-07-27 — Phase 5 CPU implementation ready; GPU acceptance pending

- Added a tracker-neutral fixed-horizon trace/evaluator and bounded NPZ/JSON
  postprocess layer. Exact pairing rejects key/time/reference/force/gate/frame
  substitutions and evaluates only the common valid prefix without interpolation.
- Tracking output includes global/local MPJPE and per ordered site position and
  sign-invariant `wxyz` orientation RMSE/P95 for all/exposed/unexposed frames.
  Acceptance gates aggregate endpoint errors plus every site's all-frame
  position/orientation RMSE and P95, preventing one wrist from hiding another.
- Compliance response is true paired yielding
  `compliant_actual_site - stiff_actual_site`, including signed force projection.
  Steady force uses the last 20 percent of each contiguous exposure pulse. A
  `1e-6` m mean displacement gate proves only chain activation; along-force sign
  remains diagnostic, and the six-batch checkpoint is not a performance claim.
- Added a separate residual ONNX contract (`60/9/930 -> 64`) without rewriting
  release models. The accepted path runs `onnxruntime.InferenceSession` 1.25.0
  with only `CPUExecutionProvider`; the portable reference evaluator is labelled
  as fallback and cannot satisfy acceptance. Dynamic shapes, exact hard-off/
  zero-compliance behavior, and mixed active/off/no-site/zero-selected-
  compliance rows pass.
- Runtime outputs are contained under the collision-safe run root. Trace and
  ONNX publication refuse existing files and symlinks; workflow completion
  rejects symlinks, logs above 64 MB, total output above 500 MB, oversized NPZ
  expansion, and out-of-root paths. Artifact roots are explicit/configurable for
  worktree and universal-tracker migration.
- Failure injection after NPZ/ONNX publication proves trace-metadata and
  export-manifest failures roll back both final paths and all hidden temporary
  files. A single fixed-horizon state-machine test locks pre-transition sample
  `k`, inclusive termination, permanently invalid auto-reset suffixes, and the
  all-300-frame success requirement.
- Final portable discovery passed 124 tests with 38 expected dependency skips;
  final `sonic_backup` discovery passed 124 tests with four expected CUDA skips.
  Both help gates and the exact 300-frame dry-run passed, and the dry-run target
  remained absent. No real rollout or GPU process was started.
- The portable evaluator remains arbitrary-body, while a SONIC-only Hydra test
  compares the eval and release ordered tracking names element-for-element and
  requires all 14. Runtime summaries persist that contract and independent audit
  rejects a missing, shortened, or reordered body list.
- Paired displacement and force are explicitly world-frame trace quantities.
  The `1e-6` m gate proves only different world trajectories, not task-space
  impedance/admittance, contact regulation, or along-force performance; the
  along-force value has no threshold.
- Phase 5 remains `IN_PROGRESS`. The one serial stiff/compliant GPU workflow and
  fresh-process artifact audit still require explicit authorization and must
  pass before status can advance to Phase 6.

## 2026-07-27 — Phase 5 GPU artifact and post-run provenance corrections

- The authorized immutable `phase5_acceptance` workflow completed in 63.656 s
  with `CHIP_PHASE5_EVAL_EXPORT_PASS`, 300 aligned frames, all named checks
  passing, and ONNX Runtime 1.25.0 using only `CPUExecutionProvider`. The
  acceptance directory is retained read-only during the corrections below.
- Independent review identified two P2 fail-closed gaps. Standalone export and
  rollout paths now reject existing/broken leaf symlinks before `resolve`, with
  negatives proving the escaped targets remain absent. The artifact audit now
  requires canonical workflow run/runs roots and checkpoint to match its CLI,
  and checks both rollout summaries against that checkpoint path and SHA-256.
- Portable/resolved focused regressions passed 11 tests with one expected ONNX
  dependency skip portably and no skips in `sonic_backup`. Full discovery then
  passed 127 tests with 38 expected dependency skips portably and 127 tests with
  four expected CUDA skips in `sonic_backup`; real ORT focused parity passed
  1/1. The corrected fresh-process artifact audit then passed with 300 aligned
  frames, mean paired displacement `0.00131441758` m, and ONNX Runtime maximum
  absolute error `5.82076609e-10`. The immutable workflow contains 14 files,
  totals 1,655,744 bytes, and has digest
  `31b836609702fd12284aad63343096e5254108ec0651847abee893d37571010f`.
  No GPU rerun or acceptance-artifact rewrite occurred. Phase 5 is `PASSED`;
  `current_phase` advances to 6. These results prove the finetune/evaluation/
  export chain and short-checkpoint regression gates, not converged compliant
  control performance.

## 2026-07-27 — Phase 6 final regression; compatibility-NVML gate pending

- Refined Phase 6 into a repeatable final matrix: complete Phase-1-through-5
  portable/resolved regression, every entrypoint help and both no-write dry
  runs, real ONNX Runtime parity, independent 300-frame metric audit, complete
  accepted-tree/checkpoint golden checks, release/ref invariants, bounded output
  layout, source hygiene, and idle process/NVML gates. The accepted Phase-4 and
  Phase-5 GPU directories remain read-only and were not rerun or rewritten.
- The first expanded help group exposed three thin AppLauncher entrypoints that
  returned 1 for bare `--help`: AppLauncher preliminarily parsed required
  application arguments, and the module guard converted argparse's normal
  `SystemExit(0)` into failure. Required flags are now enabled after
  AppLauncher registration and `SystemExit` codes are preserved. No runtime
  `main()` body changed.
- Added focused help tests. Their pinned AST digests exactly match the three
  accepted pre-Phase-6 runtime `main()` bodies; bare help is zero-exit and
  non-writing, while missing required launch arguments remain exit 2 without a
  traceback. Portable focused execution passed two tests with one expected
  Isaac Lab skip; `sonic_backup` passed 2/2.
- After the CLI-only repair, complete discovery passed 129 tests with 39
  expected dependency skips portably and 129 tests with only four CUDA skips in
  `sonic_backup`. All eight help gates then exited zero. Phase-4 and Phase-5 dry
  runs printed their exact three-job/two-rollout contracts and left both target
  directories absent before and after.
- Real ONNX Runtime 1.25.0 CPU parity passed 1/1. The independent accepted
  Phase-5 audit passed again at 300 aligned frames, mean paired displacement
  `0.00131441758 m`, maximum ORT error `5.82076609e-10`, and 1655744 bytes.
- Added a read-only final audit with configurable repository/asset/run paths.
  It reloads both trained checkpoints; pins official config/checkpoint/sample
  data, step-5/step-6 checkpoints, ONNX, refs, and release paths; and hashes
  complete evidence layouts without depending on the root location. Structural
  execution passed: Phase 4 digest
  `34cba4405dee146c7dd5f29d4731001737e8ae85f6f4d79e3928317b5bb02503`
  over 31 files/9 directories/the one internal resume link/318016496 bytes;
  Phase 5 digest
  `9efef42178353072faa457f49934c6fa67ffbf852628470e1f9bbc384046c81e`
  over 14 files/3 directories/zero links/1655744 bytes. The `/proc` CHIP
  workflow-process gate also passed.
- The audit verifies that `nvidia-smi` resolves a real GPU before accepting an
  empty compute-process list. Its explicitly labelled
  `--skip-gpu-process-check` path emits only
  `CHIP_PHASE6_STRUCTURAL_AUDIT_PASS` with
  `gpu_process_gate=SKIPPED_NOT_ACCEPTED`; a fake executable cannot produce the
  final acceptance marker. A focused `/usr/bin/true` negative exited nonzero at
  the real-GPU discovery gate after all structural checks passed.
- The exact compatibility-NVML command in Phase-6 matrix item 4 was requested
  as a read-only escalated execution. Auto-review rejected it because the
  account usage limit was reached; a main-thread retry received the same
  external rejection, and sandboxed NVML cannot access the driver. No bypass
  was attempted. The Phase-5 final gate had already found no residual
  CHIP/Isaac process and no GPU compute application, but Phase 6 remains
  `IN_PROGRESS` until the fresh item-4 command emits
  `CHIP_PHASE6_FINAL_AUDIT_PASS` and final hygiene is repeated.

## 2026-07-27 — Phase 6 Phase-5-head immutability gate

- Independent review found that the original final audit only required all
  official-baseline changes to be additive and under CHIP-owned directories.
  Because every accepted Phase-1..5 implementation file is itself an addition,
  that check could not detect a Phase-6 edit to an already-accepted core,
  adapter, training, export, or evaluation file.
- The repository audit now pins accepted Phase-5 commit
  `c925a0da115d1d6e0cc296c4a94b00a57c6461b8` and requires the exact ten-path
  Phase-6 diff with fixed `A`/`M` status.  The original official-baseline and
  release-boundary checks remain independent and unchanged.
- Phase 6 remains `IN_PROGRESS`: a fresh compatibility-NVML process audit is
  still mandatory, and repository hygiene currently detects an unintended
  Phase-6 audit bytecode cache whose precise cleanup was rejected by the
  platform approval layer after the account usage limit was reached.

## 2026-07-27 — Phase 6 resumable handoff prepared

- Consolidated the delivered architecture, accepted assets/artifact hashes,
  engineering evidence, evidence limitations, portability boundary, exact
  remaining gates, and safe cleanup/rerun rules in `phase6_handoff.md`.
- The task remains `IN_PROGRESS`. This documentation does not convert the
  previously blocked cache/NVML checks into passes and does not modify either
  immutable accepted evidence root.
- Prepared the existing Phase-6 entrypoint/help/audit/task diff for publication
  on the isolated `experiment/chip-compliance` branch only. The protected main
  branch and all pre-existing remote refs remain outside this change boundary.

## 2026-07-28 — Phase 6 final audit passed; task complete

- Re-read the Phase-6 status/matrix after restart and removed only the recorded
  audit bytecode cache plus its now-empty `__pycache__` directory. Both accepted
  Phase-4/5 evidence roots remained byte-for-byte unchanged.
- Repeated matrix items 1–3: portable and `sonic_backup` discoveries passed
  129 tests with 39/four expected skips, all eight help paths and both focused
  help tests passed, both dry-run roots remained absent, real ORT 1.25 parity
  passed, and the independent 300-frame Phase-5 audit passed again.
- The first current structural run correctly rejected a protected-ref change.
  Reflog and commit inspection proved one external fast-forward of local/main,
  origin/main, and origin/HEAD from `345c3f4` to `6d6d8ae`; it adds exactly the
  twelve central `compliance_control` documentation/test files and nothing
  else. The original snapshot was retained unchanged.
- Added a fail-closed pinned exception that requires the exact three refs,
  old/new commits, one direct non-merge commit, ancestry/count, and exact twelve
  `A` paths. Seven new tests reject partial/future/unrelated moves, multiple
  commits, modified paths, and extra paths. Final discoveries passed 136 tests
  with the same 39/four expected skips; focused suites passed 9 tests with
  one/zero skips.
- The complete structural audit passed with protected-ref marker
  `PINNED_DOCS_ONLY_FAST_FORWARD`. The exact compatibility-NVML item-4 command
  resolved the RTX 4090 through 580.159.03, found no GPU compute application or
  CHIP/Isaac workflow, revalidated both tree digests and all pinned assets, and
  emitted `CHIP_PHASE6_FINAL_AUDIT_PASS`.
- Final compile covered 56 Python files; portable imports, staged/unstaged diff,
  cache/temporary/final-newline hygiene all passed. Phase 6 is `PASSED` and the
  task is `COMPLETE`. This remains an engineering/short-rollout handoff, not a
  claim of converged multi-motion compliance performance.
