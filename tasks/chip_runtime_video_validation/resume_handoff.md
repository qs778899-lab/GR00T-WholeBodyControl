# Current continuation handoff

## Exact state on 2026-08-13

- Task: `chip_runtime_video_validation`.
- Worktree: `/tmp/gr00t_chip_runtime_video`.
- Branch: `experiment/chip-runtime-video-validation`.
- Immutable accepted source:
  `experiment/chip-compliance@3dbfb6f211511bb04fedcd326f3265cdafcfa68c`.
- Last implementation commit:
  `10f64e50e2d3a764eef8cc21e13f4afc36bfad6d`.
- Phase 1, Phase 2, and Phase 3: `PASSED`.
- Phase 4: `IN_PROGRESS` at `WAITING_FOR_IDLE_GPU_PHASE4_SMOKES`.
- Phase 5 and Phase 6: `PENDING`; overall task: `NOT_COMPLETE`.
- Protected `main`, accepted CHIP, and motion-compliance branches were not
  moved by this task.

The formal root
`/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/runtime_video_validation_v1`
and the separate Phase-4 diagnostic root
`/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/runtime_video_phase4_smoke_v1`
remain absent. Do not create the formal root before Phase 5. Do not describe a
short Phase-4 diagnostic as formal performance evidence.

## Implemented and verified

### Inherited CHIP controller remains unchanged

The accepted optional CHIP path still owns the controller, force command,
residual actor/critic, checkpoint migration/resume, finetune workflow, ONNX
export, and accepted 300-frame numeric evidence. The runtime-video branch does
not modify those accepted implementation paths. The selected target contract is
explicitly `selected = nominal - C * force_on_robot`; the separate measured
physical-yield metric remains compliant-minus-stiff displacement projected
along applied force.

### Phase 1: source/evidence contract

The pinned audit verifies source ancestry, protected refs, official and trained
checkpoint hashes, residual ONNX, original/mirrored robot and SMPL assets,
ffmpeg/ffprobe, output-root freshness, diff scope, and cache/temp hygiene.

### Phase 2: portable tracker-neutral review core

`gear_sonic/compliance_control/review` provides arbitrary-size trace schemas,
strict identity/time/reference/force alignment, the fixed tracking-first gates,
bounded non-pickle NPZ/JSON persistence, descriptor-safe SHA-256 binding,
ffprobe validation, and atomic panel composition. It contains no SONIC/G1 body
order or IsaacLab dependency. The top-level compliance package uses lazy exports
so this review core can be imported without Torch or a simulator.

### Phase 3: thin SONIC collection/rendering boundary

The additive `adapters/sonic/review` layer now contains:

- exact nine-role definitions and official-versus-trained load semantics;
- deterministic single-left, single-right, and simultaneous 5 N wrist forcing;
- name-resolved 14-keypoint snapshots with explicit world/common force;
- natural-timeout formal traces and reset-owned composer-row clearing;
- fixed front-oblique 960x720 capture, visible provenance overlays, and atomic
  H.264/yuv420p 50-fps panel output; and
- Hydra configs for all nine roles plus thin collect/evaluate/validate CLIs.

The signed selected-target relation and the physical-yield direction are tested
independently. Phase 3 passed its Hydra, fake-manager, lifecycle, camera, real
tiny-video, CLI help/dry-run, portable-core, full existing CHIP, compile, Ruff,
protected-tree, and hygiene gates.

### Phase 4 work already complete

The following current-environment regression work has passed:

- accepted Phase-4/5 evidence-tree digests remain exactly
  `34cba4405dee146c7dd5f29d4731001737e8ae85f6f4d79e3928317b5bb02503`
  and
  `9efef42178353072faa457f49934c6fa67ffbf852628470e1f9bbc384046c81e`;
- descendant source, release-path, protected-ref, workflow-process, diff, and
  formal-root structural gates pass;
- all eight accepted help paths and both accepted no-write dry-run workflows;
- real ONNX Runtime 1.25.0 dynamic/mixed/hard-off parity;
- independent accepted Phase-5 recomputation: 300 aligned frames, mean paired
  displacement `0.00131441758 m`, maximum ONNX error `5.82076609e-10`;
- resolved existing suite: `136` tests, `4` expected CUDA skips;
- Torch-bearing portable suite: `139` tests, `42` expected dependency/CUDA
  skips;
- all new review tests: `71 passed`; tracker-neutral Python-3.10 tests:
  `33 passed`; and
- Ruff E/F/I, built-in compilation, diff, cache, temp, and absent-output gates.

The separate diagnostic path is implemented but has not yet run in IsaacLab.
It has an explicit fixed-cutoff schema, never calls a cutoff a timeout, verifies
finite observations/actions, checks the two real composer-owned force/torque
rows are exactly zero after both command resets, and records immutable motion
and checkpoint provenance. Its independent auditor requires exactly 32 trace
and video frames, H.264/yuv420p/960x720/50 fps, 0.64 s duration, exact frame and
timestamp indices, no terminal/timeout/fall, both-wrist 5 N activation, the
signed target relation, bounded output, and live SHA-256 rebinding.

An end-to-end CPU golden test exercises that publication path using the real
atomic video writer and the independent audit subprocess. It passed all six
focused diagnostic tests. This proves the evidence machinery, not the robot
behavior.

A seventh CPU integration test drives the production collector orchestration
through fake Isaac-bound seams while retaining the real protocol, accumulator,
atomic trace/video/summary publication, ffprobe, hash binding, checkpoint-load
semantics, reset count, and environment-close path. It proves the collector's
assembly and lifecycle before spending an idle GPU window; it still does not
substitute for the pending real IsaacLab diagnostic.

## Current blocker and safety boundary

The latest host-visible read-only query reported:

- NVIDIA GeForce RTX 4090, driver `580.173.02`;
- 14026–14039 MiB of 24564 MiB used and 26–27% GPU utilization;
- unrelated compute PIDs `2472251`, `2660761`, and `2661453`, each using about
  4.1 GiB, plus the terminal GPU process `195750`.

The Python jobs are unrelated GRAIL stair replay/video work. Never signal,
terminate, renice, or otherwise alter them. Do not launch any CHIP GPU test
until a fresh host query shows no unrelated compute application. The current
native driver is the required `580.173.02`; the old temporary 580.159
compatibility directory is absent and should not be reconstructed unless a real
native failure demonstrates that it is necessary.

One discarded test invocation used the dependency-only Python-3.10 review
environment for the complete old suite and failed because that environment
intentionally has no Torch. It is not an acceptance result. Use
`/home/lab/Desktop/GR00T-WholeBodyControl/.venv_sim/bin/python` for the accepted
portable full suite and `/tmp/chip_review_system_venv/bin/python` only for
`test_chip_review_core.py`.

## Work not yet implemented or proven

### Remaining Phase 4 GPU gates

No current-branch native CUDA or IsaacLab process has run. Still mandatory:

1. resolved full discovery with CUDA tests enabled and no skips;
2. the 4096-environment/14-site profiler gate rejecting `aten::nonzero` and
   `aten::_local_scalar_dense`;
3. one-environment disabled and enabled 100-step Phase-2 simulator smokes;
4. the resolved Phase-3 observation/action/value shape and hard-off parity
   smoke; and
5. one real 32-frame `simultaneous_compliant` rendered diagnostic followed by
   `phase4_rendered_smoke_audit.py` and the non-skipped descendant GPU/process
   audit.

After those pass, rerun every Phase-1–3 test and all final Phase-4 hygiene
checks. Only then mark Phase 4 `PASSED` and advance to Phase 5.

### Phase 5 formal full-clip evidence

None of the requested formal videos exists yet. Phase 5 must collect all 18
full natural-timeout traces: nine roles for both original and mirrored motions.
It must evaluate the fixed tracking-first thresholds without relaxing them,
then generate ten trace-bound side-by-side MP4s (five comparisons per motion),
independently hash/probe every artifact, produce the manual-review checklist,
and record FPS, peak CUDA memory, total size, and largest log. These MP4s—not
the short diagnostic—are the videos to provide for human effect review.

### Phase 6 final handoff

The final all-tests/audits rerun, formal evidence recomputation, protected-ref
verification, capacity/cache/process checks, COMPLETE status, clickable video
index, final commit, and push remain pending.

## Exact safe continuation order

1. Read repository `AGENTS.md`, then `status.md`, `plan.md`, this handoff, and
   the Phase-4 section of `test_matrix.md`.
2. Work only in `/tmp/gr00t_chip_runtime_video` on
   `experiment/chip-runtime-video-validation`. Require a clean worktree and a
   synchronized remote branch before new implementation.
3. Reconfirm both output roots are absent and rerun a host-visible NVML process
   query. If any unrelated compute process remains, stop before GPU launch and
   keep Phase 4 `IN_PROGRESS`.
4. Once idle, execute the five remaining Phase-4 GPU gates serially. Use the
   current native driver environment first; do not preload the obsolete
   compatibility path.
5. Write the real diagnostic only below a new dedicated Phase-4 diagnostic
   root. On failure preserve the failed attempt for diagnosis and use a new
   versioned root; never overwrite or silently relabel it.
6. Run the independent rendered-smoke and descendant audits after all CHIP
   processes exit. Then rerun all Phase-1–3 tests and Phase-4 hygiene gates.
7. Update `status.md` only from the actual results. Do not enter Phase 5 unless
   every Phase-4 item passes.
8. Run Phase 5 only in a fresh idle window, from the still-missing strict formal
   root. A failed fixed acceptance gate stops the phase; thresholds and trace
   horizons are immutable.

`log.md` contains the chronological command/result history. `status.md` is the
machine-readable current state. This handoff is the authoritative restart map;
the older Phase-2 pause claims have been superseded.
