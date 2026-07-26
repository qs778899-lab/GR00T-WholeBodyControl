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
    status: PENDING
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
  portable_suite: 45 tests, 4 expected CUDA/Hydra skips
  sonic_backup_suite: 45 tests, 0 skips
  dedicated_cuda_profiler: PASSED, no aten::_local_scalar_dense
  discrete_chip_sampling: PASSED
  disabled_smoke_100_steps: PASSED, peak 0 N / 0 N*m
  enabled_smoke_100_steps: PASSED, peak 8.457119 N / 2.564395 N*m
  active_to_off_two_step_smoke: PASSED
  reset_before_disabled_update_clear_once: PASSED
  body_local_force_offset_reconstruction: PASSED
  independent_cpu_suite: 45 tests, 2 expected CUDA skips
  independent_sonic_backup_suite: 45 tests, 0 skips
  independent_disabled_enabled_smokes: PASSED
  release_config_unchanged: PASSED
  syntax_import_hygiene: PASSED

next_action: EXECUTE_PHASE_3_ONLY
