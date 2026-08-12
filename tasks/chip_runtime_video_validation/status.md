# Task status

task: `chip_runtime_video_validation`

source_branch: `experiment/chip-compliance`

source_commit: `3dbfb6f211511bb04fedcd326f3265cdafcfa68c`

branch: `experiment/chip-runtime-video-validation`

overall_status: `IN_PROGRESS`

current_phase: `4`

completion: `NOT_COMPLETE`

execution_state: `WAITING_FOR_IDLE_GPU_PHASE4_SMOKES`

last_completed_phase: `3`

last_implementation_commit: `cf859530565fa01865fb4abd2a1c61101ccf289e`

| Phase | Name | Status |
|---|---|---|
| 1 | Source and acceptance contract | PASSED |
| 2 | Portable review core | PASSED |
| 3 | Thin SONIC collection and rendering | PASSED |
| 4 | Current-environment regression | IN_PROGRESS |
| 5 | Formal full clips and review videos | PENDING |
| 6 | Final audit and handoff | PENDING |

## Current facts

- The isolated worktree was created from the clean, remotely synchronized CHIP
  acceptance commit. `main`, `experiment/chip-compliance`, and
  `experiment/motion-compliance` were not moved.
- Existing CHIP evidence proves residual-only training/resume and one 300-frame
  matched-force two-wrist chain, but explicitly does not prove full clips,
  multiple motion variants, converged compliance performance, or review video.
- The original and mirrored audited motion pairs, official checkpoint, CHIP
  step-6 checkpoint, accepted ONNX, ffmpeg, and ffprobe are present.
- Formal output root `compliance_control/runs/chip/runtime_video_validation_v1`
  was absent before task creation.
- Unrelated GRAIL GPU compute was last observed before the pause. It must never
  be terminated; Phase 5 requires a fresh host-visible idle-window check.
- The task resumed from the pushed pause checkpoint `f1f8db4`. Phase 3 passed
  its complete CPU/fake-manager/Hydra/rendering matrix. No Phase-4 simulator
  run or Phase-5 formal output has started.
- Phase 4 CPU/entrypoint/dry-run/real-ORT/accepted-artifact structural gates
  have passed. Native GPU smokes and the separate 32-frame rendered diagnostic
  remain pending because unrelated compute currently occupies the RTX 4090.
  Those processes have not been touched.
- The diagnostic evidence contract now has an end-to-end CPU golden test. It
  writes 32 trace-aligned frames through the real atomic H.264/yuv420p writer,
  persists bounded trace/summary artifacts, and requires the independent
  auditor to recheck checkpoint/motion/trace/video hashes and exact ffprobe
  properties. This validates the publication/audit path without claiming a
  simulator result or creating either runtime output root.
- The latest host-visible idle check still showed the native RTX 4090 at 26%
  utilization with 14036 MiB used and the same three unrelated Python compute
  jobs. Phase 4 therefore remains safely at the GPU wait boundary.

## Phase 1 result

- Pinned contract audit and help gate passed without creating the formal root.
- All seven asset/checkpoint/ONNX hashes and ffmpeg/ffprobe availability passed.
- Source ancestry and exact protected local/remote refs passed.
- Diff scope contained only the five new task files; controller/model/trainer/
  config and accepted evidence remained untouched.
- Unstaged/staged diff, cache/temporary, LFS-aware status, and fresh-output
  hygiene passed. The initial sandboxed diff was blocked only by Git LFS's
  read-only temporary directory; the identical host-permission rerun passed.

## Phase 2 result

- Added the tracker-neutral `compliance_control/review` layer: caller-owned
  trace layouts, strict identity/reference/time matching, nine-role matched
  interaction contracts, tracking-first gates, bounded atomic NPZ/JSON I/O,
  and trace-bound ffprobe/video manifests.
- Added exact hard-off/no-contact action checks, exact zero force/yield and
  disabled-mode compliance checks, selected-target endpoint and original-target
  orientation errors, invariant-point local/global MPJPE, measured yield along
  force, inactive-site cross-coupling, lifecycle, reset, and finiteness gates.
- Parent-package exports are now lazy. Existing public core names are unchanged,
  while importing the portable review package no longer imports Torch or any
  simulator package.
- The isolated `/usr/bin/python3`-derived Python 3.10 environment and
  `sonic_backup` each passed all 32 portable tests. Generated tiny videos proved
  H.264/yuv420p/50-fps/frame-count/duration validation and artifact rebound
  rejection. Existing core compatibility passed 23 tests and 10 subtests.
- Ruff, source compilation, CLI help, diff, formal-root absence, and cache/
  temporary hygiene gates passed. No IsaacLab application or GPU task ran.

## Phase 3 result

- Added an additive SONIC review boundary with exact nine-role definitions,
  deterministic 5 N world-frame wrist schedules, signed undamped CHIP targets,
  strict official-migration versus trained-resume semantics, name-resolved
  14-point snapshots, frame-exact traces, and reset-owned wrench clearing.
- Added deterministic Hydra compositions for all nine roles, one environment,
  seed/frame zero, original or mirrored motion, plane terrain, 50 Hz, disabled
  stochastic augmentation, fixed wrist offsets, and a parked random sampler.
- Added a fixed 960x720 front-oblique RGB/RGBA camera path, visible provenance
  overlay, bounded atomic H.264/yuv420p panel writer, descriptor-pinned ffmpeg
  compositor, and collector/evaluator/final-validator thin CLIs. Help and
  no-write dry-run paths do not import or launch IsaacLab.
- The new Phase-3 suite passed `31` tests, including 18 Hydra compositions,
  exact fake-manager action/force/mask/reset checks, real tiny panel/composite
  encoding and ffprobe checks, collisions/cleanup, and all three CLI dry-runs.
  The portable core passed `33` tests in both Python 3.11 and isolated Python
  3.10. The complete pre-existing CHIP CPU suite passed `136` tests with `4`
  skips.
- Ruff E/F/I, read-only compilation of 24 sources, no-Isaac import, protected
  release-tree/ref checks, staged/unstaged diff checks, formal-root absence,
  and cache/temporary hygiene passed. No simulator or GPU process ran.

next_action: Execute only Phase 4. Re-run the accepted Phase-6 matrix at the
documented descendant boundary, native disabled/enabled/profiler and real-shape
smokes, checkpoint/ONNX Runtime audits, and both dry-run workflows. Only after a
fresh host-visible idle-GPU check, collect one separate 32-frame rendered smoke;
never terminate unrelated compute. Keep accepted evidence byte-identical and do
not create the Phase-5 formal output root.
