# Task status

task: `motion_compliance_finetune`

baseline: `NVlabs/GR00T-WholeBodyControl@4141c34280abb67c82e115342a8720f4a83d750d`

overall_status: `IN_PROGRESS`

current_phase: `6`

| Phase | Name | Status |
|---|---|---|
| 1 | Baseline contract and tracker-agnostic core skeleton | PASSED |
| 2 | IsaacLab virtual-force command adapter | PASSED |
| 3 | Observation, reward, and experiment composition | PASSED |
| 4 | Same-shape residual initialization and finetune workflow | PASSED |
| 5 | Export and deployment switch | PASSED |
| 6 | Integration and regression validation | IN_PROGRESS |

completion: `NOT_COMPLETE`

execution_state: `PAUSED_BY_USER_2026-08-12_AFTER_P1_CPU_CHECKPOINT`

paused_implementation_head: `30f2190d1b70321705e92dde2b5c004fc8bee6d4`

remote_head_before_documentation_checkpoint: `origin/experiment/motion-compliance@30f2190d1b70321705e92dde2b5c004fc8bee6d4`

## Phase 6 current result

Completed in the current phase:

- Tracker-neutral aligned traces, strict reference/layout pairing, per-site and
  whole-body metrics, lifecycle/exposure validation, measured yield along the
  applied force, inactive-hand cross-coupling, and bounded atomic NPZ/JSON
  persistence are implemented.
- A thin SONIC real-simulator collector/recorder and a separate SONIC final
  validator are implemented.  The collector pins the G1 robot-motion encoder,
  release 14-point order, 50 Hz full-clip timing, flat terrain, relaxed eval
  terminations, reset-only event set, checkpoint roles, protocol gates,
  stimulus parameters, actual composer rows, and exact baseline/off policy
  action bytes.
- The portable evaluator records each full NPZ SHA from the same verified
  `O_NOFOLLOW` descriptor.  The SONIC validator binds collection summary,
  observed NPZ, and paired report hashes and recomputes the complete portable
  report from the six bound traces under fixed Phase-6 criteria.
- Both paused P1 evidence gaps are now closed in code.  The collector binds the
  exact composed and runtime termination/event function targets, timeout/mode,
  declared and effective parameters, thresholds, body names, and command
  names.  Every single-site/multi-site trial must invoke the configured reset
  event after observed nonzero command/composer force and prove exact clearing
  of both buffers.  The final validator rejects missing, stale, or altered
  evidence.
- Post-P1 focused CPU validation passed `40 tests in 1.40s`; all four Phase-6
  help gates, AST parsing, `git diff --check`, and a real Hydra-compose
  provenance probe also passed.
- The post-P1 combined pure Python suite passed `127 passed, 1 skipped in
  25.63s`, and the deployment suite passed `33 passed, 96 warnings in 3.14s`.
  The official residual-contract smoke, trainer help/config gate, and official
  Phase-4 residual-initialization smoke also passed.
- The step-5 and step-6 checkpoint audit JSON files were atomically regenerated
  at `2026-08-12T17:17:40+08:00` after their validators reached the write step;
  read-back shows steps 5/6, all twelve changed residual tensors, 55/17 frozen
  tensors, and twelve optimizer slots.  The orchestration output containing
  their process exit statuses was truncated, so these two commands must be
  rerun rather than promoted as final matrix evidence.
- Earlier accepted Phase-5 Python/C++ runtime, artifact-hash, production-hook,
  CLI, and release-boundary evidence remains available, but the exact Phase-5
  acceptance/build commands were not rerun after P1 in this pause window.

Still required before completion:

- Finish Phase-6 matrix item 1 on the final paused tree.  The CPU suites above
  passed, but the exact Phase-2/3 real IsaacLab smokes, the two checkpoint-audit
  commands with retained exit status, and the remaining exact Phase-5
  acceptance/C++ build gates were not completed in this continuation.
- Fresh post-restart 16-environment/5-iteration native matched-driver CUDA
  smoke with FPS and GPU-memory recording.  Earlier on 2026-08-12 the host was
  observed natively matched at NVIDIA `580.173.02`, but at the final pause
  inspection `nvidia-smi` could no longer communicate with the driver.  Treat
  GPU/driver availability as unverified and repair/revalidate it before any
  CUDA evidence command.  The removed temporary `580.159` compatibility
  directory must not be referenced.
- Real 4096-environment host-off/enabled scheduler-only measurement.
- Strictly paired real simulator traces for baseline, overlay-off,
  enabled/no-contact, single-left, single-right, and simultaneous trials.
- Final endpoint/orientation/MPJPE/force/yield/cross-coupling/success/fall/reset
  thresholds and final output/cache/diff hygiene.  In particular, active-mode
  left/right endpoint, wrist orientation, and whole-body tracking upper bounds
  must be reviewed explicitly because tracking accuracy remains the priority.
- No formal Phase-6 GPU/simulator output was created during this continuation:
  `phase6_residual_gpu_smoke_post_restart`, `phase6_scheduler_4096.json`, and
  `phase6_real_paired` remain absent and are fresh targets.  No training,
  4096-environment benchmark, or six-protocol simulator collection was
  started.

next_action: On explicit resume, start from `phase6_handoff.md`; verify the
branch/worktree and restore a matched, visible, idle GPU stack.  Then finish
Phase-6 matrix item 1 from the exact commands, freeze active-mode tracking
acceptance bounds before collecting data, and execute items 2–9 in order.  Do
not mark Phase 6 `PASSED` or the task `COMPLETE` from this CPU-only checkpoint.
