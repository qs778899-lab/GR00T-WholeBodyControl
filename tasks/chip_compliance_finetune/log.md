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
