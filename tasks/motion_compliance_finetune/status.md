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

execution_state: `PAUSED_BY_USER_2026-07-28`

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
- Latest focused CPU validation after the final evidence-integrity fixes:
  `38 passed in 1.46s`; independent review reran it with `38 passed in 1.41s`.
  Related CLI help, AST parsing, and `git diff --check` passed.
- Earlier in this Phase-6 resume, the deployment suite (`33 passed`), official
  residual contract, trainer help/config, Phase-5 Python/C++ ORT smoke,
  production CMake target/CLI acceptance, accepted artifact revalidation,
  pinned hashes, and release-boundary diff all passed.  These were not a final
  rerun of the complete suite after the latest collector changes.
- Phase-5 Python/C++ runtime, accepted artifact hashes, production hook, CLI
  validation, and release-boundary gates remain passed.

Still required before completion:

- Rerun all Phase 1–5 tests and the Phase-2/3 real IsaacLab smoke commands on
  the final paused commit; the current focused result is not matrix item 1.
- Fresh post-restart 16-environment/5-iteration compatibility-CUDA smoke with
  FPS and GPU-memory recording.
- Real 4096-environment host-off/enabled scheduler-only measurement.
- Strictly paired real simulator traces for baseline, overlay-off,
  enabled/no-contact, single-left, single-right, and simultaneous trials.
- Final endpoint/orientation/MPJPE/force/yield/cross-coupling/success/fall/reset
  thresholds and final output/cache/diff hygiene.  In particular, active-mode
  left/right endpoint, wrist orientation, and whole-body tracking upper bounds
  must be reviewed explicitly because tracking accuracy remains the priority.
- P1 evidence hardening: pin termination/event function targets and all
  threshold/body/command/mode parameters, and exercise the configured Phase-6
  reset event after a nonzero-wrench interaction rather than relying only on
  the explicit post-timeout cleanup plus the earlier Phase-2 reset smoke.

next_action: Resume from `phase6_handoff.md` section "Paused checkpoint
(2026-07-28)".  First close its two P1 evidence items and rerun matrix item 1;
then execute Phase-6 items 2–9 in order.  Do not mark Phase 6 `PASSED` or the
task `COMPLETE` from the current CPU-only evidence.
