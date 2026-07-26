# CHIP compliance finetune status

task: chip_compliance_finetune
baseline_commit: 4141c34280abb67c82e115342a8720f4a83d750d
branch: experiment/chip-compliance
status: IN_PROGRESS
current_phase: 3

phases:
  - id: 1
    name: Baseline, contracts, and architecture skeleton
    status: PASSED
  - id: 2
    name: Simulator command/event and observation integration
    status: PASSED
  - id: 3
    name: Checkpoint-compatible policy/critic integration
    status: IN_PROGRESS
  - id: 4
    name: Low-resource finetune smoke and parity regression
    status: PENDING
  - id: 5
    name: Tracking/compliance evaluation and export
    status: PENDING
  - id: 6
    name: Final regression and handoff
    status: PENDING

phase_1_scope:
  - Record the released SONIC robot-motion-encoder and 14-body reference contracts.
  - Add tracker-agnostic pure Torch schema, math, schedule, and metrics modules under `gear_sonic/compliance_control/core`.
  - Add portable target-damper state/update/reset under `gear_sonic/compliance_control/core`.
  - Add only dual name-based reference/articulation resolvers under `gear_sonic/compliance_control/adapters/sonic`.
  - Prove exact disabled/zero-compliance parity and arbitrary-site support using CPU tests.
  - Prove structured common-frame, finite-value, exposure-metric, and index-space contracts.
  - Prove the core imports without Isaac Lab and contains no fixed G1/29-DoF/14-index contract.

phase_1_exclusions:
  - No Isaac Lab command, event, reward, or observation-manager wiring.
  - No policy/checkpoint/config changes.
  - No training, simulator, ONNX, or deployment changes.

phase_1_result:
  completed_on: 2026-07-27
  status: PASSED
  unit_tests: 23
  dual_index_spaces: PASSED
  structured_common_frame: PASSED
  finite_and_nonnegative_contract: PASSED
  hard_off_nan_and_backward: PASSED
  exposure_metrics: PASSED
  target_damper_update_and_reset: PASSED
  sequence_string_rejection: PASSED
  core_import_without_isaaclab: PASSED
  fixed_g1_index_audit: PASSED
  cached_diff_check: PASSED

phase_2_scope:
  - Add an opt-in SONIC/Isaac Lab compliance command, events, observation, and derived Hydra experiment.
  - Preserve separate name-resolved reference and articulation index spaces and one explicit common Cartesian frame.
  - Apply current body-local wrenches with local site offsets, per-step resultant-wrench limits, asynchronous pulses, and reset-safe ownership.
  - Match CHIP sampling defaults: 0-40 N, 1-3 s, and discrete compliance values 0/0.02/0.05 m/N.
  - Keep current force repetition over future frames explicit as alignment, not force prediction.
  - Keep all release policy, critic, checkpoint, reward, deployment, and `sonic_release.yaml` paths unchanged.

phase_2_result:
  completed_on: 2026-07-27
  status: PASSED
  regression_reason: Disabled compliance inherited global RNG consumption from CommandTerm/EventManager; follow-up review found dynamic CUDA due indices and boolean pulse-finish writes in the per-step path.
  private_countdown_scheduler: PASSED
  disabled_global_cpu_cuda_rng_parity: PASSED
  release_interval_event_parity: PASSED
  deterministic_partial_reset: PASSED
  explicit_operational_enable_api: PASSED
  immediate_owned_row_clear_without_env_step: PASSED
  unrelated_composer_row_preservation: PASSED
  static_config_unchanged_by_runtime_switch: PASSED
  portable_full_discovery: 65 tests, 12 expected CUDA/Hydra skips
  portable_phase_2_subset: 52 tests, 7 expected CUDA/Hydra skips
  strict_prevalidated_parity: PASSED
  portable_fixed_shape_cpu_profiler: PASSED, mixin bound compute / fake composer / 4096 envs / 14 sites / no aten::nonzero or aten::_local_scalar_dense
  sonic_backup_suite: 65 tests, 0 skips, PASSED
  independent_sonic_backup_suite: 65 tests, 0 skips, PASSED_AFTER_PREVALIDATED_DUE_FIX
  portable_cuda_scale_profiler: PASSED, mixin bound compute / fake composer / 4096 envs / 14 sites / no aten::nonzero or aten::_local_scalar_dense
  real_isaac_bound_cpu_cuda_profiler: PASSED, AppLauncher SonicComplianceCommand / forced private due / actual WrenchComposer / no aten::nonzero or aten::_local_scalar_dense
  real_trace_global_rng_parity: PASSED, exact CPU and CUDA states
  real_trace_immediate_owned_row_clear: PASSED, force and torque rows zero before env.step
  parent_independent_real_bound_disabled_smoke: PASSED, wall 18.174 s / both markers / peak 0 N and 0 N*m / trace RNG and immediate-off clear
  discrete_chip_sampling: PASSED
  disabled_smoke_100_steps: PASSED, peak 0 N / 0 N*m, then real-bound trace passed
  enabled_smoke_100_steps: PASSED, peak 6.785363 N / 2.197217 N*m
  independent_disabled_enabled_smokes: PASSED_AFTER_PREVALIDATED_DUE_FIX
  active_to_off_two_step_smoke: PASSED
  reset_before_disabled_update_clear_once: PASSED
  body_local_force_offset_reconstruction: PASSED
  release_config_unchanged: PASSED
  syntax_import_hygiene: PASSED
  cuda_rerun_blocker: RESOLVED

next_action: EXECUTE_PHASE_3_ONLY
