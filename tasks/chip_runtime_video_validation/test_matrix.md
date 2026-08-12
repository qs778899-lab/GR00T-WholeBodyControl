# Test matrix

Run from the `experiment/chip-runtime-video-validation` worktree root. Use
`PYTHONDONTWRITEBYTECODE=1` and disable pytest's cache provider. A phase advances
only after every command/check in that phase passes.

## Phase 1 — Source and acceptance contract

1. Run the read-only pinned contract audit:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/chip_runtime_video_validation/artifacts/phase1_contract_audit.py --repo-root . --runs-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip --output-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/runtime_video_validation_v1`.
2. Run its help gate and require zero exit without filesystem writes.
3. Verify the current branch descends directly from `3dbfb6f`, while local and
   tracked remote `main`, CHIP, and motion refs equal their recorded commits.
4. Confirm Phase-1 diff adds only `tasks/chip_runtime_video_validation/*`; no
   accepted controller/model/trainer/config/evidence file changes.
5. `git diff --check`; after staging only Phase-1 files,
   `git diff --cached --check`.
6. Confirm no repository `__pycache__`, `.pytest_cache`, `.pyc`, `.pyo`, hidden
   temporary, or formal-output path exists.

## Phase 2 — Portable review core

1. Run `gear_sonic/tests/compliance/test_chip_review_core.py` in the system
   Python and `sonic_backup`; no IsaacLab import is allowed.
2. Cover arbitrary body/site counts, exact identity/layout/time/reference/
   force pairing, active/inactive site masks, original versus selected endpoint
   error, original-target quaternion error, invariant-point local/global MPJPE,
   measured yield/cross-coupling, success/fall/reset/finiteness, and every fixed
   numeric threshold with fail-closed adversarial negatives.
3. Test bounded same-descriptor `O_NOFOLLOW` trace/JSON loading, duplicate ZIP
   rejection, atomic no-overwrite publication, symlink rejection, and no pickle.
4. Test ffprobe JSON parsing and video manifest validation for exact H.264,
   yuv420p, 50/1 fps, trace frame count, duration, SHA bindings, panel order,
   and refusal of missing/extra/rebound artifacts. Use generated tiny videos;
   no GPU is required.
5. `--help`, AST/import portability, `git diff --check`, staged diff, and cache/
   temporary hygiene must pass.

## Phase 3 — Thin SONIC collection and rendering

1. Run the focused SONIC review adapter/collector tests in `sonic_backup`.
2. Hydra-compose all nine roles at one environment for original and mirrored
   inputs. Pin release 14-body order, two wrist names/offsets, 50 Hz, plane,
   deterministic first frame/seed, complete natural timeout, no stochastic
   augmentation, and role-specific official/trained checkpoint semantics.
3. Fake-manager lifecycle tests must prove exact release/hard-off action bytes,
   no-contact zero force/yield, matched stiff/compliant force bytes, single-left/
   single-right/both masks, selected hindsight targets, and reset-event clearing
   of command plus composer force/torque after observed nonzero force.
4. Camera tests must pin front-oblique pose, dimensions, frame/sample ordering,
   RGBA-to-RGB conversion, metadata overlay, H.264/yuv420p writer settings,
   collision/partial-file cleanup, and bounded output.
5. Collector/evaluator/final-validator `--help` and a no-write `--dry-run`
   workflow must pass. No AppLauncher/environment may start for help/dry-run.
6. Full existing CHIP CPU tests plus new tests, compile/import, release-tree,
   `git diff --check`, staged diff, and cache hygiene must pass.

## Phase 4 — Current-environment regression

1. Re-run the complete prior CHIP Phase-6 matrix on the new tree, adapting only
   the old exact-head audit to the documented descendant boundary. Preserve all
   accepted Phase-4/5 directories byte-for-byte.
2. Run native-driver Phase-2 disabled/enabled/profiler smoke and Phase-3 real
   shape/off-parity smoke; reject CUDA scalar-sync operators as before.
3. Reload/audit official, CHIP step-5/step-6, and residual ONNX hashes/schemas;
   run real ONNX Runtime 1.25.0 parity and both dry-run workflows.
4. On an idle GPU only, collect a separate short 32-frame rendered smoke in a
   new diagnostic directory. Require trace/video frame equality, ffprobe gate,
   finite observations/actions, no fall, and clean process exit. This diagnostic
   is not formal performance evidence.
5. Rerun all Phase-1–3 tests and final Phase-4 cache/diff/output/process checks.

## Phase 5 — Formal full clips and review videos

1. Refuse launch unless host NVML shows the native 580.173.02 RTX 4090 and no
   unrelated compute process. Never terminate another job.
2. From the missing `runtime_video_validation_v1` root, collect all 18
   original/mirrored role traces. Require natural timeout at the exact expected
   full 50 Hz frame and no shortened horizon or auto-reset suffix.
3. Independently evaluate exact action/trace pairing and all fixed tracking,
   compliance, force, yield, cross-coupling, fall, success, reset, and finiteness
   gates. A failed gate stops the phase; thresholds are not changed.
4. Generate ten primary side-by-side H.264/yuv420p 50-fps videos. Each must have
   exactly its trace frame count and overlay the required provenance/state.
5. Independently ffprobe and hash every MP4, trace, summary, and metrics report;
   validate one-to-one manifest coverage and produce the manual-review checklist.
6. Record per-role FPS, process peak CUDA memory, total size, and largest log;
   no debug stream may exceed 64 MiB and the complete root must stay below 2 GiB.

## Phase 6 — Final audit and handoff

1. Rerun all portable/resolved tests, real smokes, checkpoint/ONNX audits, all
   CLI help/dry-run gates, and independent formal evidence recomputation.
2. Verify source ancestry; protected main/accepted-CHIP/motion local and remote
   refs unchanged; accepted old evidence tree hashes unchanged; only the new
   experiment branch is publishable.
3. Verify every requested comparison/motion/video, numeric check, manifest hash,
   ffprobe property, manual checklist, output capacity, no symlink/duplicate/
   cache/temp, no task process, and `git diff --check`.
4. Update `status.md` to COMPLETE only after all checks pass, document clickable
   video paths and limitations, commit, and push only
   `experiment/chip-runtime-video-validation`.
