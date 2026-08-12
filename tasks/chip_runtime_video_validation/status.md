# Task status

task: `chip_runtime_video_validation`

source_branch: `experiment/chip-compliance`

source_commit: `3dbfb6f211511bb04fedcd326f3265cdafcfa68c`

branch: `experiment/chip-runtime-video-validation`

overall_status: `IN_PROGRESS`

current_phase: `3`

completion: `NOT_COMPLETE`

execution_state: `RUNNING_PHASE3_THIN_COLLECTION_RENDERING`

| Phase | Name | Status |
|---|---|---|
| 1 | Source and acceptance contract | PASSED |
| 2 | Portable review core | PASSED |
| 3 | Thin SONIC collection and rendering | IN_PROGRESS |
| 4 | Current-environment regression | PENDING |
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
- The GPU currently has unrelated GRAIL compute. It must not be terminated;
  Phase 5 waits for a verified idle window while Phases 1–3 proceed on CPU.

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

next_action: Implement only the Phase-3 thin SONIC review adapter, deterministic
nine-role collector, frame-exact renderer, portable evaluator/final-validator
CLIs, and run every Phase-3 fake-manager/Hydra/help/regression gate. Do not start
the Phase-4 real simulator or create the formal Phase-5 output root.
