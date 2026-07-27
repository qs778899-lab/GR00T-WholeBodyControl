# CHIP compliance finetune plan

## Objective

Add an opt-in CHIP-style compliant tracking mode to the released SONIC robot-motion-encoder path while preserving the stiff-mode tracking baseline, especially upper-limb end-effector accuracy.

The implementation follows these boundaries:

- Start from `nvlabs/main@4141c342` and the released `sonic_release` checkpoint/config contract.
- Keep the existing G1 kinematic encoder and dense whole-body rewards as the baseline path.
- Treat compliance as inverse Cartesian stiffness in metres per newton and apply the CHIP hindsight relation `g_hind = g_ref - C * f_robot` in one explicit coordinate frame.
- Support an arbitrary ordered site set, including SONIC's 14 reference bodies; never hard-code a three-point tensor shape.
- Make disabled mode and zero compliance exact value identities before attempting finetuning.
- Add new modules/configs where possible instead of changing the release path in place.
- Keep the reusable core tracker-agnostic under `gear_sonic/compliance_control/core`: `schema`, `math`, `schedule`, and `metrics` must import neither Isaac Lab nor SONIC/G1 modules and must not contain robot-specific index tables.
- Keep SONIC integration thin and name-based under `gear_sonic/compliance_control/adapters/sonic`: body-name resolution, MDP/Hydra composition, and checkpoint adaptation live outside the core so another universal tracker can reuse the core unchanged.

## Upstream baseline contract

- Release experiment: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`.
- Robot-motion encoder inputs: multi-future G1 joint position/velocity command plus anchor orientation.
- Tracking reference bodies: 14 ordered bodies in `gear_sonic/config/manager_env/commands/terms/motion.yaml`.
- Existing dormant compliance code is not a usable baseline: it is absent from released config composition and its compliant target observation has incompatible force/future/site broadcasting and an in-place reference mutation.
- Full training requires Python 3.11, Isaac Lab 2.3+, CUDA 12.x and the SONIC training assets; those dependencies are not required for Phase 1's CPU contract tests.

## Phases

### Phase 1 — Baseline, contracts, and architecture skeleton

Create a dependency-light, tracker-agnostic compliance package split into `schema`, `math`, `schedule`, `damper`, and `metrics`. It provides explicit site ordering, structured common Cartesian frame/anchor/rotation metadata, strict tensor-shape and finite-value checks, global or per-sample/per-site hard gating, isotropic or Cartesian-axis anisotropic non-negative compliance, site masks, optional displacement limiting, a pure future-force phase schedule, CHIP target-damper state/reset/update, response metrics based on true exposure, and no in-place mutation. Disabled gate elements must select the original reference with `torch.where` so mixed stiff/compliant batches retain exact identity; global disabled mode may bypass unused force/compliance operands entirely. Add a SONIC-only dual name resolver as a thin boundary adapter: reference-motion and articulation indices are distinct typed spaces resolved independently against the same ordered `site_names`, never a shared integer table. Establish CPU unit tests for stiff parity, mixed-batch gates, finite/NaN contracts, CHIP sign/broadcast semantics, anisotropic wrist-style response, target damping, arbitrary site counts, gradients, scheduling/exposure metrics, dual index spaces, and rejection of the known five-dimensional force-shape failure.

The core must import and run in an environment without Isaac Lab. No Isaac Lab command/event/config, policy observation, reward, checkpoint, or deployment wiring is allowed in this phase.

### Phase 2 — Simulator command/event and observation integration

Add a separate compliance experiment config, force/compliance sampling events, same-frame force projection, per-environment target-damper state/reset, and a non-mutating SONIC MDP observation adapter that consumes the Phase 1 core. Resolve reference-motion and articulation sites independently from Hydra/runtime body-name fields. Apply the declared full/yaw rotation into the structured common frame before hindsight math; never pass indices between spaces. Match CHIP's training disturbance envelope with `0–40 N`, `1–3 s` pulses and discrete inverse-stiffness values `{0, 0.02, 0.05} m/N`. Retain SONIC-specific `30 N` resultant-force and `20 N·m` resultant-torque caps; therefore multi-site or single-site samples above the safe resultant limit may be uniformly scaled before application.

### Phase 3 — Checkpoint-compatible policy/critic integration

Add the smallest compliance-conditioned SONIC adapter/residual branch to the G1 encoder path, feed the Phase 2 damped/hindsight target through the compliance-only path, expose privileged force only to the critic during training, and add a thin checkpoint adapter with zero-initialized/gated parameters so the released checkpoint retains exact stiff behavior on load. Keep the dense tracking rewards and upper-limb terms enabled; do not move policy architecture into the tracker-agnostic core.

### Phase 4 — Low-resource finetune smoke and parity regression

Run the documented low-resource training smoke, first with compliance disabled and then with force/compliance sampling enabled. Confirm finite losses, nonzero adapter gradients, deterministic zero-compliance parity, bounded artifacts, and checkpoint save/resume.

Pinned local smoke assets (never stage these binaries):

- Checkpoint: `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt`, Hugging Face revision `7c90a56c`, SHA-256 `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`, recorded step `41550`.
- Robot motion: `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl`.
- SMPL directory: `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered`.

The immutable accepted Phase-4 evidence is under
`/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix`.
It is an evidence directory, not a rerun target. For reproduction, replace the
literal `<fresh-run-root>` below with a unique absolute child of
`compliance_control/runs/chip`; that path must not exist before launch.

Pinned stiff-baseline smoke command template:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt \
  +resume=false \
  num_envs=16 headless=True use_wandb=false \
  ++algo.config.num_learning_iterations=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered \
  experiment_dir=<fresh-run-root>/stiff_release_step5 \
  save_dir=<fresh-run-root>/stiff_release_step5/.hydra \
  output_dir=<fresh-run-root>/stiff_release_step5/output
```

The official checkpoint is a warm start only: `resume=false` deliberately
discards its 69-parameter optimizer and recorded step 41550. The opt-in CHIP
branch is trained separately, with every released policy/value/std/RMS tensor
frozen byte-for-byte and exactly 12 residual parameter tensors (770753 scalars
for the two-wrist smoke layout) owned by the optimizer.

The bounded workflow reproduction template is:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  gear_sonic/scripts/run_chip_phase4_finetune.py \
  --run-root <fresh-run-root>
```

It executes the exact stiff command above, then an official-to-residual
five-batch run and an independent one-batch step-5-to-step-6 resume. The
compliance smoke forces both configured wrist sites to be enabled, uses the
short smoke-only pulse interval `[0.02, 0.04]` seconds and nonzero compliance
values `{0.02, 0.05}` m/N, saves `last.pt` at step 5, and records finite losses,
per-site true exposure, all 12 gradient histories, and peak CUDA allocation.
Normal CHIP training retains the physical 3.5-6 second pulse interval; the
accelerated interval is only an acceptance-smoke mechanism.

The released PPO update keeps a temporal rollout tensor: 16 environments and
4 minibatches produce actor microbatches with leading shape ``[4, 24]``. The
SONIC adapter therefore adds its ``[B, S, 64]`` residual only after encoder
outputs have been reassembled from flattened ``B*S`` rows into
``[B, S, 2, 32]`` FSQ tokens. It neither selects timestep zero nor broadcasts
one timestep across the rollout; disabled timesteps remain release-bit-exact,
and the generic ``UniversalTokenModule`` remains unchanged.

The resume job creates a symlink to the preserved step-5 checkpoint inside a
new output directory, passes `resume=true`, requests exactly one new PPO batch
with `num_learning_iterations=1`, and saves on frequency 1 when global step
reaches 6. Its start callback requires byte-exact model, optimizer, and
scheduler state plus global step 5 before that batch. The generic trainer also
loads its serialized environment payload. It does not serialize process
CPU/CUDA/Python/NumPy RNG state or the compliance command's private generator,
countdown, damper, and wrench buffers, so this is a strict training-state
resume, not a claim of trajectory-bitwise replay.

All generated files stay under `compliance_control/runs/chip`; the runner
rejects a pre-existing or out-of-root workflow, caps each log at 64 MB, caps
each checkpoint run at 1.2 GB and the complete workflow at 2.5 GB, and never
overwrites the official checkpoint or the accepted step-5 evidence. Use
`--dry-run` to inspect all three argument vectors without creating files.

The host still has a kernel/userspace NVIDIA mismatch (`580.159` versus
`580.173`), but the pinned `580.159.03` compatibility-driver workaround is
validated: the complete Phase-3 CUDA/Hydra suite and all real Isaac Lab smokes
pass through the extracted local libraries. Phase 4 must use that recorded
environment until the host packages are aligned; it still requires the real
GPU training command and cannot substitute a CPU result.

### Phase 5 — Tracking/compliance evaluation and export

Add evaluation as a separate, portable layer rather than embedding metrics into
SONIC's policy or simulator code. The tracker-neutral core owns a fixed-horizon
`AlignedTrackingTrace`, exact pair alignment, finite metric calculations, and
acceptance budgets. A small postprocess layer owns bounded NPZ/JSON I/O. The
SONIC boundary owns only Isaac Lab trace collection and the separate residual
ONNX adapter. This division allows another universal tracker to provide the same
trace contract without importing SONIC, G1, Hydra, or Isaac Lab.

The comparator is a matched-force A/B test. Both jobs use the same checkpoint,
initial reference frame, motion/frame clock, force sample and schedule, ordered
site mask, and compliance schedule. The compliant job passes its actor-safe
compliance command to the trained residual. The stiff comparator zeros only
that command, making the latent residual exactly zero and recovering the
release-equivalent actor path while the same external force remains applied.
It is not the unforced Phase-4 stiff training log. Pairing requires exact integer
keys and boolean gates plus tolerance-bounded equality of times, references,
forces, compliance, and normalized reference quaternions; no interpolation or
nearest-frame substitution is permitted. If either rollout falls, metrics use
only their common valid prefix while retaining the complete fixed horizon.
Each trace persists a structured local-frame kind, semantic anchor, and rotation
mode. The SONIC summary additionally records the ordered site names and both
resolved reference/articulation index lists for provenance; metrics never treat
those index spaces as interchangeable.
The portable trace accepts arbitrary body/site counts, but the SONIC acceptance
adapter additionally compares the derived experiment's ordered tracking bodies
against `sonic_release` and requires all 14 names unchanged at Hydra, runtime,
and independent-audit boundaries.

Tracking preservation is measured with global and anchor-local MPJPE plus wrist
endpoint position RMSE/P95 and sign-invariant quaternion geodesic-angle
RMSE/P95. Endpoint summaries are reported per ordered left/right site for all,
force-exposed, and unexposed frames. Reference quaternions are selected only in
the motion-reference index space and actual quaternions only in the
articulation index space, normalized as finite `wxyz`, and compared as
`2*acos(abs(dot(q_ref, q_actual)))`. The overall upper-endpoint regressions and
each site's all-frame position RMSE/P95 and orientation RMSE/P95 regressions are
acceptance gates, so one wrist cannot hide behind the other. Exposed/unexposed
per-site values remain separate diagnostics. Force response is true paired
yielding, `compliant_actual_site - stiff_actual_site`, not actual-to-reference
tracking error; it includes norm and signed projection along the matched force,
overall and per site. Steady force is the last 20 percent of each contiguous
exposure pulse in time, never a magnitude percentile or peak threshold.

The pinned acceptance budgets are at least 200 aligned frames and 20 exposed
frames per configured site; at most 0.05 m upper-endpoint position regression,
0.25 rad upper-endpoint orientation regression, and 0.03 m global/local MPJPE
regression; compliant success/fall rates of 1/0; and a measured peak force in
`[1, 30]` N. Mean paired displacement must reach `1e-6` m only as a chain
activation gate. Its signed along-force projection is reported but not
thresholded; this six-batch smoke is not evidence of compliant-control
performance quality. The deterministic smoke uses both wrist sites, 5 N sampled force,
0.02 m/N compliance, one-second pulses, and 300 frames. The accepted Phase-4
step-6 checkpoint is the candidate under test.

Export only the trained actor residual to a new ONNX file. Its explicit inputs
are `compliance_target [B,S,60]`, `compliance_command [B,S,9]`, and
`actor_context [B,S,930]`; its output is `latent_residual [B,S,64]`, composed as
`release_encoder_latent + latent_residual` before the unchanged release decoder.
The adjacent manifest records ordered sites, command layout, frame, hashes,
dynamic axes, and composition. ONNX checker plus pinned ONNX Runtime 1.25.0 CPU
inference must match PyTorch at dynamic `(B,S)=(1,1)` and `(2,4)` and must return
bit-exact zeros for hard-off inputs, including unused NaNs, and zero compliance.
The ONNX reference evaluator remains only a truthfully labelled portable
fallback and cannot satisfy accepted-artifact parity.
One mixed `B x S` case must combine active, global-off, no-site, and
zero-selected-compliance rows without batch/timestep broadcasting.
Released encoder/decoder ONNX files are never rewritten.

Standalone export and rollout boundaries inspect the requested final path with
`lexists`/`is_symlink` before canonicalization, so a broken leaf symlink cannot
redirect publication outside the requested path. The independent audit requires
the workflow's canonical run root, runs root, and checkpoint to equal its CLI
arguments. It also checks each stiff/compliant rollout summary against the same
canonical checkpoint path and SHA-256; workflow-level provenance cannot mask a
sub-rollout checkpoint substitution.

The evaluation core and trace format support any positive ordered site count;
the Phase-5 trained checkpoint/export remains the Phase-4 two-wrist layout. A
14-site deployment therefore reuses the evaluator unchanged but requires a
14-site residual checkpoint and matching export spec/config. Exact commands,
output layout, comparator semantics, and metric definitions are recorded in
`phase5_runbook.md` and `test_matrix.md`.

### Phase 6 — Final regression and handoff

Run the complete Phase-1-through-Phase-5 CPU/Hydra/checkpoint regression from
one clean worktree, then repeat the real ONNX Runtime parity and independent
Phase-5 metric audit.  Phase 6 adds no policy/model, simulator behavior,
trainer, or export-graph change, so it must not rerun or rewrite the already
accepted GPU training and 300-frame evaluation directories.  A CLI-only repair
keeps bare `--help` non-writing and zero-exit while pinning each affected
entrypoint's runtime `main()` AST.  A fresh read-only audit must load
both accepted checkpoints, repeat the strict legacy/residual/optimizer checks,
recompute Phase-5 metrics and ONNX parity, and pin the complete artifact trees
with location-independent layout/content digests.

The final audit also requires the official checkpoint/config and sample motion
hashes, the original git-ref snapshot, an additive-only diff from
`4141c34280ab`, byte-exact release paths, bounded output inventories, no
Phase-5 symlink, and no cache/temporary file.  Phase 4 retains exactly one
documented internal resume symlink; it is accepted only when it resolves to the
step-5 checkpoint inside the same immutable workflow.  A compatibility-NVML
query must find no compute application, and `/proc` must contain no CHIP
training/rollout process.

Record the environment, commands, output layout, migration boundary, hashes,
and measured tracking/activation values in `phase6_handoff.md`.  The short
six-batch checkpoint and 300-frame matched-force rollout are evidence that the
finetune/evaluation/export chain works and preserves the gated tracking budgets;
they are not a converged compliant-control, impedance, or contact-regulation
performance claim.  Mark the task complete only after every Phase-6 command in
`test_matrix.md` passes.
