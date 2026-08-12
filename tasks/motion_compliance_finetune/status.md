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

execution_state: `PAUSED_BY_USER_AFTER_PHASE6_GPU_FUNCTIONAL_SMOKE_BUSY_GPU`

paused_implementation_head: `2dddfdaee9ce8d4c5debed1c51fbc224ec942606`

remote_head_before_documentation_checkpoint: `origin/experiment/motion-compliance@2dddfdaee9ce8d4c5debed1c51fbc224ec942606`

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
- On resume, Phase-6 matrix item 1 passed on the current tree: `127 passed, 1
  skipped`; deployment `33 passed`; exact step-5/step-6 audits; Phase-5
  acceptance, C++ ORT, production configure/build/help/CLI, immutable release
  boundary; official residual/init and trainer config gates; and native-driver
  Phase-2/3 real IsaacLab smokes all returned zero.
- Before formal collection, active tracking gates were fixed in portable code
  and the SONIC recomputation contract. Each active wrist is checked against
  its yielded position target (5/10 mm RMSE/P95 regression limits), retains the
  original orientation target (0.05/0.10 rad), and excludes only its explicitly
  mapped point from remaining-body local/global MPJPE (max(3 mm, 10%)/max(5
  mm, 10%)). Focused evaluator/final-validator tests pass `43 tests in 1.58s`.
- The first Phase-6 item-2 attempt reached the prescribed 16-environment,
  five-iteration native CUDA training boundary in the previously fresh
  `phase6_residual_gpu_smoke_post_restart` directory. The run used the audited
  robot PKL and SMPL directory with W&B disabled, reached global step 5, and
  left no task-owned process. The original launcher exit code was lost when its
  verbose output exceeded the orchestration context, so it is not claimed as a
  retained zero exit. Its exposure record reports five
  observed/nonzero-force batches, 160 active-site samples, finite loss metrics,
  FPS 179--262, peak site force 14.9962 N, and process peak allocated CUDA
  memory 353,315,840 bytes.
- An independent step-5 audit returned zero for that run: all six policy and six
  value residual tensors changed, 55 policy and 17 value base tensors remained
  frozen, and all 12 optimizer slots were present. The retained output is a
  valid functional-smoke diagnostic, with checkpoint SHA-256
  `8f44cd1bc0bcc5f7264a1bc41851ac2f57832570098ee27518e302a0503ec354`.
- The same attempt is **not accepted as item-2 performance evidence** because
  unrelated GRAIL compute processes were resident on the RTX 4090 before and
  during launch, violating the matrix idle-GPU precondition. The measured FPS
  is therefore contention-affected. Preserve this 298 MiB directory and never
  overwrite it.

Still required before completion:

- An accepted idle-GPU repeat of the 16-environment/5-iteration native CUDA
  smoke, using the still-missing fresh directory
  `phase6_residual_gpu_smoke_idle_retry1`. Host-side validation shows the native
  `580.173.02` stack and RTX 4090 are available; default sandbox device
  isolation, not a driver failure, caused the earlier `nvidia-smi` error.
  Re-run only after no unrelated compute process is present, with host device
  access and no removed temporary `580.159` compatibility override.
- Real 4096-environment host-off/enabled scheduler-only measurement.
- Strictly paired real simulator traces for baseline, overlay-off,
  enabled/no-contact, single-left, single-right, and simultaneous trials.
- Final endpoint/orientation/MPJPE/force/yield/cross-coupling/success/fall/reset
  thresholds and final output/cache/diff hygiene.  In particular, active-mode
  left/right endpoint, wrist orientation, and whole-body tracking upper bounds
  must be reviewed explicitly because tracking accuracy remains the priority.
- `phase6_residual_gpu_smoke_post_restart` now exists and is reserved as the
  contention-affected functional attempt described above. The accepted retry
  path `phase6_residual_gpu_smoke_idle_retry1`, `phase6_scheduler_4096.json`,
  and `phase6_real_paired` remain absent and fresh. No 4096-environment
  benchmark or six-protocol simulator collection was started.

next_action: After confirming the host GPU has no unrelated compute process,
repeat Phase-6 matrix item 2 in the fresh
`phase6_residual_gpu_smoke_idle_retry1` path and independently audit step 5.
Do not overwrite the first attempt and do not start item 3 until the idle-GPU
repeat passes. Then execute items 3–9 in order without changing the pre-data
acceptance limits. Do not mark Phase 6 `PASSED` or the task `COMPLETE` before
all formal evidence passes.
