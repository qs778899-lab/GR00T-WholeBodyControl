# Execution log

## 2026-08-12 — Isolated CHIP runtime/video task started

- Re-read the repository engineering rules and the completed CHIP task state.
- Verified clean synchronized protected worktrees before branching:
  `main@6d6d8ae`, `experiment/chip-compliance@3dbfb6f`, and
  `experiment/motion-compliance@9c290f2`.
- Created `experiment/chip-runtime-video-validation` in the independent
  `/tmp/gr00t_chip_runtime_video` worktree from exact CHIP source `3dbfb6f`.
- Read-only inspection found that accepted CHIP evidence is one 300-frame,
  dual-wrist, matched-force stiff/compliant run with no video. It is valid chain
  evidence but not the requested full-runtime or visual-effect evidence.
- Confirmed the existing environment exposes a reusable `eval_camera` and an
  H.264/yuv420p imageio writer, while the existing CHIP collector supports only
  `stiff` and `compliant`. Planned a new thin review/collector layer rather than
  modifying the compliance controller, residual network, or training code.
- Verified original and mirrored robot/SMPL assets, official/step-6 checkpoints,
  accepted ONNX, system ffmpeg/ffprobe, and the absent formal output root.
- Native host PyTorch sees CUDA, but unrelated GRAIL compute currently occupies
  the RTX 4090. It was not touched; formal GPU evidence waits for an idle window.

## 2026-08-12 — Phase 1 passed

- The pinned read-only audit verified source ancestry, seven immutable hashes,
  both video tools, and the absent formal output root.
- Help, protected-ref, task-only scope, unstaged/cached diff, and repository
  cache/temporary checks passed. Default-sandbox Git diff initially failed
  because Git LFS could not create its clean-filter temporary under the shared
  read-only Git directory; the same LFS-aware check passed with host access and
  reported no non-task change.
- Phase 1 is `PASSED`; Phase 2 is now the only in-progress phase. No simulator,
  training, video, or formal output was launched.

## 2026-08-12 — Phase 2 passed

- Implemented the portable `gear_sonic/compliance_control/review` package with
  immutable caller-owned trace schemas, strict alignment digests, generic
  metrics, and a formal nine-role matched-force suite. No body count, joint
  order, endpoint name, tracker name, or simulator import is embedded there.
- The formal suite rejects missing/extra roles; mismatched target/force/
  compliance/mask schedules; non-exact hard-off and no-contact actions; nonzero
  disabled/no-contact force or yield; nonzero disabled compliance; tracking,
  orientation, invariant-point, and inactive-site regressions; absent action
  activation; yield not aligned with force; falls, incomplete success, stale
  reset wrench, and non-finite data.
- Added bounded same-descriptor trace/JSON/video reads with `O_NOFOLLOW`, no
  pickle loading, duplicate-ZIP rejection, size caps, atomic no-overwrite
  writes, portable Python-3.10 stream hashing, exact ffprobe checks, and live
  hash rebinding for video manifests.
- Added a simulator-free `python -m gear_sonic.compliance_control.review
  --help`/`probe` entrypoint. Changed only package export mechanics to lazy-load
  the existing core API; an explicit compatibility import and the existing core
  test suite verified the public names remain available.
- Built an ephemeral dependency-only environment at
  `/tmp/chip_review_system_venv` from `/usr/bin/python3` (Python 3.10.12,
  NumPy 2.2.6, pytest 9.0.2). It passed the portable suite: `32 passed`.
- `sonic_backup` (NumPy 2.4.4, pytest 9.0.3) independently passed the same
  portable suite: `32 passed`. The pre-existing core suite passed `23 passed,
  10 subtests passed`.
- Generated test-only six-frame videos under pytest temporary directories and
  positively verified H.264, yuv420p, 50/1 fps, six frames, 0.12 s duration,
  panel order, SHA-256 bindings, atomic publication, and revalidation. Negative
  videos proved rejection of wrong codec, pixel format, rate, dimensions,
  frame count, duration, symlinks, missing/extra fields, and rebound artifacts.
- Ruff E/F/I checks, read-only compilation of 11 sources, both CLI help gates,
  `git diff --check`, absent formal output root, and cache/temporary hygiene all
  passed. The two compilation-created cache directories were removed; no user
  data or accepted evidence was deleted.
- Phase 2 is `PASSED`; Phase 3 is now the only in-progress phase. No simulator,
  GPU, accepted evidence, or formal output was touched.
