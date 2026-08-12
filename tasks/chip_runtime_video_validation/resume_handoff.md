# Pause and resume handoff

## Exact checkpoint

- Task: `chip_runtime_video_validation`.
- Worktree: `/tmp/gr00t_chip_runtime_video`.
- Branch: `experiment/chip-runtime-video-validation`.
- Accepted source: `experiment/chip-compliance@3dbfb6f211511bb04fedcd326f3265cdafcfa68c`.
- Last implementation commit: `625b3299bb302a78a3b8cb7fe50a60c8c561730f`.
- State at pause: Phase 1 and Phase 2 `PASSED`; Phase 3 `IN_PROGRESS` but no
  Phase-3 implementation or test execution has begun.
- Completion: `NOT_COMPLETE`. Phases 3, 4, 5, and 6 remain mandatory.
- Formal result root
  `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/runtime_video_validation_v1`
  was still absent at pause. It must remain absent until Phase 5.

The user requested a pause, so this document records an intentionally incomplete
checkpoint. Do not mark Phase 3 passed and do not jump to a later phase.

## What is implemented and verified

### Accepted CHIP source inherited unchanged

The branch starts from the completed optional CHIP-style SONIC implementation.
Its tracker-neutral compliance math uses arbitrary named sites and the relation
`g_hind = g_nominal - C * f_on_robot`; the SONIC adapter owns name resolution,
coordinate conversion, force application, checkpoint migration, and the gated
residual. Hard-off remains an exact identity path. The accepted controller,
residual model, trainer, deployment export, and previous evidence were not
modified by this runtime/video task.

### Phase 1: source and evidence contract

Phase 1 added the task plan/status/test matrix/log and the read-only pinned audit
at `artifacts/phase1_contract_audit.py`. It verified:

- the accepted source ancestry and protected local/remote refs;
- official checkpoint, CHIP step-6 checkpoint, residual ONNX, and original plus
  mirrored robot/SMPL SHA-256 values;
- system `ffmpeg` and `ffprobe` availability;
- the fresh formal-output invariant; and
- task-only diff, cache, temporary-file, staged-diff, and LFS-aware hygiene.

### Phase 2: portable review core

The new `gear_sonic/compliance_control/review` package is independent of SONIC,
G1, IsaacLab, and Torch imports. It contains:

- `schema.py`: immutable generic traces, protocol roles, pair definitions, and
  fixed tracking-first acceptance criteria;
- `alignment.py`: exact identity, ordering, reference, time, force, compliance,
  mask, and action alignment checks with no interpolation;
- `metrics.py` and `suite.py`: endpoint/orientation, invariant whole-body,
  force/yield/cross-coupling, lifecycle, reset, finiteness, and nine-role gates;
- `io.py` and `_hashing.py`: bounded `O_NOFOLLOW`, no-pickle NPZ/JSON reads,
  duplicate-member rejection, SHA-256 binding, atomic no-overwrite writes;
- `video.py`: strict ffprobe validation for H.264, yuv420p, 50 fps, exact frame
  count/duration/dimensions, ordered panels, and live artifact rebinding; and
- `__main__.py`: simulator-free review CLI help/probe entrypoint.

The top-level compliance package now lazily exports its existing public core API
so importing the portable review layer does not import Torch or a simulator.
Existing public names were compatibility-tested.

Phase-2 evidence at commit `625b329`:

- portable review suite: `32 passed` in an isolated `/usr/bin/python3`-derived
  Python 3.10 environment;
- portable review suite: `32 passed` independently in `sonic_backup`;
- pre-existing core suite: `23 passed, 10 subtests passed`;
- tiny generated-video positive and adversarial codec/fps/frame/hash tests;
- Ruff E/F/I, source compilation, CLI help, diff, cache, temporary, and absent
  formal-root checks all passed; and
- no AppLauncher, IsaacLab environment, training, GPU rollout, or formal video
  workflow ran.

## What is not implemented or proven

### Phase 3: thin SONIC collection and rendering

None of the following exists yet for this task:

- a SONIC-specific review adapter under
  `gear_sonic/compliance_control/adapters/sonic/review`;
- deterministic configs for all nine roles and a review-only event config that
  removes startup/reset randomization and forces a plane terrain;
- a complete-natural-timeout collector with exact pre-transition trace samples,
  action bytes, termination/reset evidence, and one video frame per sample;
- a bounded fixed front-oblique H.264/yuv420p panel writer with provenance and
  force/compliance overlays;
- thin collector, portable evaluator/compositor, and final-validator CLIs; or
- the Phase-3 fake-manager, Hydra composition, camera, help/dry-run, and full CPU
  regression tests.

The existing `run_chip_phase5_rollout.py` is only a two-role, fixed-step numeric
collector. It is not the formal nine-role/full-horizon/video workflow and must
not be presented as such.

### Phase 4: current-environment regression

No new-tree real simulator smoke, native-driver profiler/parity regression,
checkpoint/ONNX re-audit, 32-frame rendered diagnostic, or complete accepted
CHIP regression matrix has run for this branch.

### Phase 5: formal evidence and videos

No formal rollout has run. The required 18 full traces (nine roles for original
and mirrored motion), ten paired review MP4s, metric reports, independent
manifests, manual checklist, runtime/FPS/memory data, and capacity audit do not
exist. Consequently there are currently no videos suitable for human review.

### Phase 6: final audit and handoff

No final recomputation, protected-ref/evidence-tree audit, process/output/cache
audit, completion mark, or final video handoff has run.

## First issue to resolve on resume

The accepted CHIP contract and implementation define the selected target as
`nominal - C * force_on_robot`. The Phase-2 synthetic review fixture currently
constructs its selected target with a positive `+0.10 m` shift for a positive
`5 N` force. The generic review suite uses the norm of selected-target shift, so
this unrealistic fixture still passed. This is a test-coverage issue, not
evidence that the accepted controller math is wrong.

At the same time, the accepted Phase-5 evaluation intentionally defines physical
policy yield as
`compliant_actual_site - stiff_actual_site` projected **along** the matched
force. That positive measured-yield check should not be flipped merely because
the hindsight observation has a minus sign: the selected observation target and
the robot's physical displacement under applied force are different quantities.

Before writing the real collector:

1. change/add a deterministic fixture that asserts the selected target exactly
   follows `nominal - C * force_on_robot` for active sites and is exact nominal
   for inactive/hard-off sites;
2. preserve a separate positive projection gate for measured compliant-versus-
   stiff physical yield along force;
3. run the complete Phase-2 portable tests again after any Phase-2 test/core
   correction; and
4. make the Phase-3 fake-manager test compare the actual SONIC target builder
   against that signed contract.

Do not infer force direction from wrist names or body index order. Record the
world vector, common-frame vector, target delta, and measured paired displacement
explicitly so the numeric trace and video overlay can be audited together.

## Phase-3 implementation boundary already audited

Use three narrow layers and do not modify accepted controller/model/trainer code:

1. Generic review code remains under `compliance_control/review`.
2. SONIC-only name/index/frame/protocol/camera logic goes under
   `adapters/sonic/review`.
3. Scripts under `gear_sonic/scripts` stay thin and import AppLauncher only
   after ordinary argument parsing, so `--help` and `--dry-run` cannot launch a
   simulator or create an output directory.

Compose every one of the nine roles for both motion inputs. Pin one environment,
the release 14-body ordered contract, named left/right wrist offsets, 50 Hz,
seed 0, deterministic first frame, complete natural timeout, plane terrain, no
observation corruption, and no stochastic startup/reset augmentation. The
review event config should retain only lifecycle-safe compliance reset behavior.

For stiff/compliant A/B roles, drive the same deterministic 5 N force schedule
and site mask into the simulator in both runs. The stiff actor receives exact
hard-off/zero compliance conditioning; the compliant actor receives the active
command. Record the final safety-limited force actually applied, not the
requested value. Stop immediately on the one natural timeout so no auto-reset
suffix enters the trace.

## Safe resume procedure

1. Read `/home/lab/Desktop/GR00T-WholeBodyControl/AGENTS.md`, this file,
   `status.md`, `plan.md`, and the Phase-3 section of `test_matrix.md`.
2. Enter `/tmp/gr00t_chip_runtime_video`. If `/tmp` was cleared, recreate a
   worktree for the already-pushed
   `experiment/chip-runtime-video-validation` branch from the main repository;
   do not branch from current `main` or cherry-pick onto the accepted CHIP branch.
3. Require a clean worktree and verify `HEAD`/remote ancestry. At this pause the
   protected refs were exactly:
   - `main` and `origin/main`: `6d6d8ae9a04b67a977b027acecfe20c65aca0647`;
   - CHIP local/remote: `3dbfb6f211511bb04fedcd326f3265cdafcfa68c`;
   - motion local/remote: `9c290f29b31017be1ff54c23bbe497d9278249ae`.
4. Re-run the Phase-1 read-only contract audit and require the formal root to be
   absent. Revalidate hashes instead of trusting this snapshot.
5. Execute only Phase 3, beginning with the signed-target test above. Run every
   Phase-3 test in `test_matrix.md`; fix and rerun the same tests on failure.
6. Only after all Phase-3 tests pass, mark Phase 3 `PASSED`, advance status to
   Phase 4, and commit/push a clean checkpoint.
7. Phase 5 must wait for host-visible NVML to show the native RTX 4090 idle. An
   unrelated GRAIL process was previously observed; never terminate it. The
   sandboxed `nvidia-smi` check at pause could not access the driver, so this
   must be checked with the approved host execution path immediately before a
   GPU run.

## Pause integrity

Before this pause documentation was written, the runtime-video worktree had no
uncommitted files, its local branch matched
`origin/experiment/chip-runtime-video-validation@625b329`, the protected refs
matched the hashes above, and the formal output root was absent. No task
simulator or training process was started during the pause audit. The
documentation commit created for this pause is the only intended change after
`625b329`.
