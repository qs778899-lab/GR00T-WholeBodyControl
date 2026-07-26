# CHIP compliance finetune status

task: chip_compliance_finetune
baseline_commit: 4141c34280abb67c82e115342a8720f4a83d750d
branch: experiment/chip-compliance
status: IN_PROGRESS
current_phase: 2

phases:
  - id: 1
    name: Baseline, contracts, and architecture skeleton
    status: PASSED
  - id: 2
    name: Simulator command/event and observation integration
    status: PENDING
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

next_action: PAUSED_BEFORE_PHASE_2
