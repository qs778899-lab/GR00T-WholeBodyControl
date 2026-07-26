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
