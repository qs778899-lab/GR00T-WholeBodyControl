# CHIP runtime and review-video validation plan

## Objective

Starting from the immutable accepted CHIP implementation
`experiment/chip-compliance@3dbfb6f211511bb04fedcd326f3265cdafcfa68c`,
prove on the current native NVIDIA/IsaacLab environment that the optional
CHIP-style SONIC path runs end to end over complete motion clips, preserves
tracking as the primary objective, and produces trace-bound videos suitable for
human review. This work is isolated on
`experiment/chip-runtime-video-validation`; it must not move or edit `main`,
`experiment/chip-compliance`, or `experiment/motion-compliance`.

The accepted Phase-4/5 CHIP checkpoints and evidence directories are immutable
inputs. New evidence is written only below the fresh strict root
`compliance_control/runs/chip/runtime_video_validation_v1`, which must not exist
before the first formal workflow.

## Fixed inputs

- Official SONIC checkpoint SHA-256:
  `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
- CHIP step-6 checkpoint SHA-256:
  `71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02`.
- Accepted residual ONNX SHA-256:
  `a4ccbc9e216dd97fe5181a12f5ded7a9e544c1a477fd114c909b8564bc83e2f3`.
- Original robot/SMPL motion hashes:
  `005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b` /
  `f31a00cd23cedb9b6cc50805d912276234a35a40678529d726df3b1dec3682d8`.
- Mirrored robot/SMPL motion hashes:
  `7d9ec8a24acbb952cfce2048e2d3b5c156e8ae0c43e32443eb5ea42cbb22038e` /
  `49cbf3c604f78952474d3bcecb6bbc0b4a136eab78dc3ab8580869594383bb4f`.
- Native current driver contract: NVIDIA `580.173.02`, PyTorch 2.7/CUDA 12.8;
  host device access is required because the default sandbox hides GPU nodes.
- Video contract: system `ffmpeg`/`ffprobe`, H.264, `yuv420p`, 50 fps.

## Architecture boundary

Keep the accepted CHIP controller, residual model, checkpoint migration, and
training code unchanged. Add three narrow layers:

1. `compliance_control/review`: tracker-neutral protocol schema, aligned metrics,
   bounded trace/report/video-manifest I/O, ffprobe validation, and pair layout.
   It owns no SONIC/G1/IsaacLab/body-name constant.
2. `adapters/sonic/review`: name-based SONIC snapshot mapping, exact reference
   versus articulation index spaces, heading-frame conversion, deterministic
   protocol forcing, camera state, and reset evidence.
3. Thin scripts under `gear_sonic/scripts`: one real collector, one portable
   evaluator/video compositor, and one final SONIC provenance validator.

Rendering is evidence only. Numerical traces are sampled on the policy clock
before each corresponding transition; video frame `k` must depict and be
labelled as the same trace sample `k`. No interpolation, nearest-frame pairing,
or BFS/DFS action remap is introduced in this simulator-only path.

## Formal protocols

Run every role for both audited motions (`original`, `mirrored`) with seed 0 and
the complete natural 50 Hz horizon. A fail-safe maximum may be configured, but
publication requires exactly one natural timeout at the expected final frame.

| role | external force | actor residual | active site(s) |
|---|---:|---:|---|
| `release_baseline` | off | official/migrated hard-off | none |
| `chip_hard_off` | off | trained hard-off | none |
| `enabled_no_contact` | off | trained enabled | none |
| `single_left_stiff` | matched | hard-off | left wrist |
| `single_left_compliant` | matched | enabled | left wrist |
| `single_right_stiff` | matched | hard-off | right wrist |
| `single_right_compliant` | matched | enabled | right wrist |
| `simultaneous_stiff` | matched | hard-off | both wrists |
| `simultaneous_compliant` | matched | enabled | both wrists |

The three active A/B pairs use identical 5 N force vectors, phase schedules,
motion/seed/frame/time, and site masks. Compliance is fixed at 0.02 m/N. Force
directions must be deterministic and visually distinguishable; applied
resultant wrench remains under the accepted 30 N / 20 N·m guards.

## Tracking-first acceptance fixed before data

- `release_baseline` versus `chip_hard_off`: exact policy action bytes and all
  force/compliance buffers zero; endpoint RMSE regression no more than 5 mm,
  local MPJPE no more than max(3 mm, 10%), global MPJPE no more than
  max(5 mm, 10%).
- `enabled_no_contact` remains in the same numeric range as hard-off and emits
  no force/yield.
- For each active wrist and each single/simultaneous compliant trial, selected
  hindsight-target endpoint RMSE/P95 regression versus its matched stiff trial
  is no more than 5/10 mm. Orientation remains referenced to the original
  target and may regress no more than 0.05/0.10 rad.
- Whole-body preservation excludes only the explicitly active tracking point on
  that row. Remaining-point local/global MPJPE regression is no more than
  max(3 mm, 10%)/max(5 mm, 10%). The inactive wrist remains below the same
  5/10 mm cross-coupling limits.
- Every full clip has success 1, falls 0, finite inputs/derived metrics, active
  force/yield above configured minima, inactive sites below tolerance, and an
  exercised reset path that clears command-owned and actual composer wrench
  rows exactly.

These limits cannot be relaxed after viewing formal traces or videos.

## Review-video acceptance

Produce ten primary side-by-side MP4s: five comparisons for each motion:

1. release baseline / CHIP hard-off;
2. CHIP hard-off / enabled no-contact;
3. single-left stiff / compliant;
4. single-right stiff / compliant;
5. simultaneous stiff / compliant.

Use one fixed front-oblique full-body camera with both wrists and feet visible.
Each panel shows role; the video shows commit/checkpoint prefix, motion, seed,
frame/time, active sites, force, and compliance. Encode exactly one video frame
per 50 Hz trace sample as H.264/yuv420p. A bounded manifest records video,
source trace, summary, and metrics SHA-256 plus ffprobe codec/resolution/fps/
frame-count/duration. Human review checks falls/foot slide, hand tracking,
intended yield direction, jitter, inactive-hand motion, and reset artifacts.

## Phases

1. **Source and acceptance contract**: create the isolated task state, pin
   sources/assets/tools/output paths, inventory the accepted collector/render
   boundaries, and prove protected refs/worktrees are unchanged.
2. **Portable review core**: implement protocol/trace schemas, exact pairing,
   fixed numeric gates, bounded atomic I/O, ffprobe/video manifests, and pure
   unit/golden/adversarial tests without IsaacLab.
3. **Thin SONIC collection and rendering**: implement named-site protocol
   control, full-natural-timeout collection, exact action/reset/composer
   evidence, one-frame-per-sample camera capture, CLI help/dry-run, and fake
   IsaacLab lifecycle tests.
4. **Current-environment regression**: rerun the complete accepted CHIP matrix
   plus new suites, real Phase-2/3 smokes, checkpoint/ONNX audits, and one short
   rendered smoke under native 580.173.02. Do not proceed on any failure.
5. **Formal full clips and videos**: on an idle GPU, collect the 18 fresh role/
   motion traces, evaluate all gates, create ten MP4s, independently recompute
   reports/manifests, and record runtime/FPS/GPU memory.
6. **Final audit and handoff**: verify every requested artifact, protected refs,
   immutable old evidence, output capacities, hashes, caches/diffs/processes,
   commit/push only this new branch, and provide clickable review videos.
