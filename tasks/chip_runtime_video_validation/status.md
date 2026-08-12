# Task status

task: `chip_runtime_video_validation`

source_branch: `experiment/chip-compliance`

source_commit: `3dbfb6f211511bb04fedcd326f3265cdafcfa68c`

branch: `experiment/chip-runtime-video-validation`

overall_status: `IN_PROGRESS`

current_phase: `2`

completion: `NOT_COMPLETE`

execution_state: `RUNNING_PHASE2_PORTABLE_REVIEW_CORE`

| Phase | Name | Status |
|---|---|---|
| 1 | Source and acceptance contract | PASSED |
| 2 | Portable review core | IN_PROGRESS |
| 3 | Thin SONIC collection and rendering | PENDING |
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

next_action: Implement only the Phase-2 tracker-neutral review core and run all
Phase-2 tests. Do not add IsaacLab/SONIC names or begin collector/GPU work.
