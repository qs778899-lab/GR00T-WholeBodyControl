# CHIP compliance finetune status

task: chip_compliance_finetune
baseline_commit: 4141c34280abb67c82e115342a8720f4a83d750d
branch: experiment/chip-compliance
status: IN_PROGRESS
current_phase: 6

phases:
  - id: 1
    name: Baseline, contracts, and architecture skeleton
    status: PASSED
  - id: 2
    name: Simulator command/event and observation integration
    status: PASSED
  - id: 3
    name: Checkpoint-compatible policy/critic integration
    status: PASSED
  - id: 4
    name: Low-resource finetune smoke and parity regression
    status: PASSED
  - id: 5
    name: Tracking/compliance evaluation and export
    status: PASSED
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

phase_3_result:
  completed_on: 2026-07-27
  status: PASSED
  portable_suite: 79 tests, 26 expected Hydra/CUDA/official-model skips
  resolved_cpu_hydra_official_suite: 79 tests, 4 expected CUDA skips
  focused_resolved_integration: 12 tests, PASSED
  compatibility_cuda_hydra_suite: 72 tests, 0 skips, PASSED_BEFORE_REVIEW_ONLY_PHASE3_MODEL_CHANGED
  inherited_cuda_scale_profiler: PASSED_BEFORE_REVIEW_PHASE2_UNCHANGED
  inherited_real_disabled_enabled_smokes: PASSED_BEFORE_REVIEW_PHASE2_UNCHANGED
  official_checkpoint_sha256: e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909
  official_legacy_schema: 55 policy keys, 17 value keys
  initialized_residual_keys: 6 actor, 6 critic
  legacy_tensor_bitwise_exact: PASSED
  strict_branch_resume: PASSED
  hard_privileged_force_rejection: PASSED
  variable_site_future_construction: PASSED_FOR_1_2_5_14_17_SITES
  post_fsq_hard_gate: PASSED_GLOBAL_ZERO_COMPLIANCE_AND_MIXED_ROWS
  residual_construction_rng_parity: PASSED_CPU
  critic_single_shared_normalization: PASSED
  frozen_official_std_distribution_optimizer_parity: PASSED
  first_backward_head_gradient: PASSED
  release_shared_file_byte_audit: PASSED
  real_phase_3_shape_smoke: PASSED, wall 16.48 s
  real_observation_widths: 930/1645/1761/60/9/6
  real_action_value_shapes: (1,1,29)/(1,1,1)
  real_default_off_action_bitwise_exact: PASSED
  real_frozen_std_bitwise_exact: PASSED
  syntax_import_hygiene: PASSED

phase_4_result:
  completed_on: 2026-07-27
  status: PASSED
  inherited_compatibility_cuda_hydra_suite: PASSED, 93 tests, 0 skips, before the CPU-only resume repair
  inherited_cuda_scale_profiler: PASSED, 4096 environments / 14 sites
  inherited_disabled_smoke: PASSED, 100 steps / 0 N / 0 N*m / real-bound profiler marker
  inherited_enabled_smoke: PASSED, 100 steps / 6.785363 N / 2.197217 N*m
  real_phase_3_shape_smoke: PASSED
  first_acceptance_step_5: PASSED, then runner lazy-import bootstrap failed before resume
  second_acceptance_step_5: PASSED, then resume start audit rejected optimizer LR 1e-5 versus serialized 2e-5 before batch 6
  failed_run_artifacts: RETAINED_UNCHANGED
  resume_root_cause: generic load overwrote the serialized post-scheduler optimizer LR with the checkpoint adaptive-KL args LR
  isolated_resume_fix: PASSED_CPU, dedicated trainer reloads only the same serialized optimizer payload after generic restoration
  resume_boundary_unit_test: PASSED, optimizer and scheduler recursively exact while args LR remains unchanged
  portable_cpu_suite: PASSED, 95 tests / 33 expected dependency skips
  resolved_cpu_suite: PASSED, 95 tests / 4 expected CUDA skips
  training_help_gate: PASSED
  dry_run_gate: PASSED, exact three command vectors and no filesystem write
  syntax_import_hygiene: PASSED, 38 compiled files / 46 text files
  process_gate: PASSED, no training/Isaac process and no GPU compute application
  canonical_run_root: /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix
  canonical_marker: CHIP_PHASE4_FINETUNE_PASS
  canonical_durations_s: stiff 23.272135 / initial 24.701326 / resume 17.529439
  canonical_step_5_audit: PASSED, loss steps 1-5 / exposure 79,79 / peak CUDA 727262208 bytes
  canonical_step_6_audit: PASSED, loss step 6 / exposure 16,16 / peak CUDA 407661056 bytes
  independent_step_5_step_6_audits: PASSED
  canonical_checkpoint_schema: 55 policy legacy / 17 value legacy / 6 actor residual / 6 critic residual
  canonical_optimizer_ownership: 12 tensors / 770753 scalars
  canonical_gradient_nonzero_min: step5 99 / step6 20
  canonical_workflow_bytes: 318016496
  canonical_largest_log_bytes: 55249
  official_checkpoint_sha256: e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909
  released_config_and_generic_trainer: UNCHANGED
  final_read_only_audit: PASSED, no P0; one documentation P1 and one manifest-label P2 corrected
  accepted_evidence_root: /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix
  rerun_contract: PASSED, documented <fresh-run-root> must not exist
  accepted_training_artifacts: UNCHANGED
  accepted_legacy_manifest_pre_final_bytes: 318014905
  future_manifest_size_semantics: PASSED, explicit pre-final field plus post-write final cap check
  final_audit_focused_portable: PASSED, 17 tests / 6 expected dependency skips
  final_audit_focused_resolved: PASSED, 17 tests / 0 skips
  final_audit_portable_cpu_suite: PASSED, 97 tests / 33 expected dependency skips
  final_audit_resolved_cpu_suite: PASSED, 97 tests / 4 expected CUDA skips
  final_audit_training_help: PASSED
  final_audit_dry_run: PASSED, target absent before and after
  final_audit_hygiene: PASSED

phase_5_result:
  completed_on: 2026-07-27
  status: PASSED
  portable_cpu_suite: PASSED, 127 tests / 38 expected dependency skips
  resolved_cpu_suite: PASSED, 127 tests / 4 expected CUDA skips
  focused_onnx_parity: PASSED, ORT 1.25.0 CPU dynamic shapes plus mixed BxS rows
  portable_onnx_fallback: labelled onnx.reference.ReferenceEvaluator only
  trace_schema: v2, structured frame / normalized wxyz / bounded non-pickle NPZ
  paired_semantics: matched force / release-equivalent zero residual versus trained residual
  per_site_tracking_gates: position RMSE/P95 and orientation RMSE/P95
  sonic_release_body_contract: PASSED, ordered 14-body Hydra/runtime/audit gate
  transactional_pair_rollback: PASSED, trace metadata and ONNX manifest injection
  fixed_horizon_transition_semantics: PASSED, pre-step k valid / suffix permanently invalid
  paired_activation_gate_m: 1.0e-6, chain validation only
  help_gates: PASSED
  dry_run_gate: PASSED, ORT subprocess pinned and target absent before and after
  gpu_workflow: PASSED, immutable phase5_acceptance / 300 aligned frames / 63.656 s
  gpu_workflow_marker: CHIP_PHASE5_EVAL_EXPORT_PASS
  gpu_workflow_checks: PASSED, no failed named threshold
  post_gpu_p2_leaf_symlink_gate: PASSED, focused and full CPU regressions
  post_gpu_p2_provenance_gate: PASSED, focused and full CPU regressions
  independent_artifact_audit: PASSED_CORRECTED, 300 aligned frames / mean paired displacement 0.00131441758 m / ORT max abs error 5.82076609e-10
  accepted_artifact_digest: 31b836609702fd12284aad63343096e5254108ec0651847abee893d37571010f
  accepted_workflow_bytes: 1655744
  performance_claim: CHAIN_ACTIVATION_AND_REGRESSION_ONLY

next_action: EXECUTE_PHASE_6_ONLY
