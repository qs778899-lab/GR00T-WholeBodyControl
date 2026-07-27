# CHIP compliance finetune test matrix

Every test listed for the active phase is mandatory. A later phase must not start until all tests for the current phase pass and `status.md` is updated.

## Phase 1 — Baseline, contracts, and architecture skeleton

Run from the repository root:

1. Contract unit tests

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Required coverage: structured frame/anchor/rotation and mismatch validation; global and mixed-batch/per-site hard-off forward/backward identity; global-off NaN operand isolation; strict finite reference/force and finite non-negative compliance validation when active; zero-compliance identity; CHIP sign; static/future force broadcasting; isotropic and Cartesian-axis anisotropic compliance; site masking; arbitrary 17-site support; displacement limiting; differentiability; no input mutation; target-damper update/full reset/partial reset; exposure metrics using enable, compliance, and site mask; dual reference/articulation name resolution with different runtime orders; `str`/`bytes` sequence rejection; and rejection of invalid/ambiguous/five-dimensional shapes.

2. Import and syntax smoke

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
   from pathlib import Path

   for root in (Path("gear_sonic/compliance_control"), Path("gear_sonic/tests/compliance")):
       for path in root.rglob("*.py"):
           compile(path.read_bytes(), str(path), "exec")
   PY
   PYTHONDONTWRITEBYTECODE=1 python -c "from gear_sonic.compliance_control import CartesianFrameSpec, ComplianceTargetSpec, TargetDamper, apply_hindsight_target; print(CartesianFrameSpec, ComplianceTargetSpec, TargetDamper, apply_hindsight_target)"
   ```

   The unit suite must also verify by AST inspection that tracker-agnostic core modules have no Isaac Lab/SONIC environment imports, and that the SONIC name resolver contains no literal `14`/`29` index contract. Successful execution in this environment is the no-IsaacLab runtime proof.

3. Patch hygiene

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
   from pathlib import Path

   roots = (
       Path("gear_sonic/compliance_control"),
       Path("gear_sonic/tests/compliance"),
       Path("tasks/chip_compliance_finetune"),
   )
   for root in roots:
       for path in root.rglob("*"):
           assert path.name not in {"__pycache__", ".pytest_cache"}, \
               f"cache directory remains: {path}"
           assert path.suffix not in {".pyc", ".pyo"}, f"compiled Python remains: {path}"
           if not path.is_file():
               continue
           text = path.read_text(encoding="utf-8")
           assert text.endswith("\n"), f"missing final newline: {path}"
           assert not text.endswith("\n\n"), f"extra EOF blank line: {path}"
           assert all(line == line.rstrip() for line in text.splitlines()), \
               f"trailing whitespace: {path}"
   PY
   git diff --check
   git diff --cached --check
   ```

The training entrypoint `--help` smoke requires Isaac Lab, which is not installed in the Phase 1 CPU environment and is mandatory in Phase 4 before any finetune result can pass.

## Phase 2 — Simulator command/event and observation integration

Run from the repository root. These commands include the inherited Phase 1 checks.

1. Portable CPU suite

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Required result: every discovered test passes. The current regression run discovers 65 tests with 12 expected CUDA/Hydra skips because preserved Phase 3 work is also present; the Phase 2-only subset contains 52 tests with seven expected skips. The suite covers CPU shape alignment, separate typed index spaces, arbitrary site sets, deterministic CHIP-faithful discrete compliance sampling, non-zero yaw/full-rotation frame and CHIP-sign checks, non-mutation, command-owned private-RNG pulse countdowns, exact global CPU/CUDA RNG preservation while disabled, deterministic partial reset, explicit runtime enable/disable without mutating static config, immediate state/countdown cancellation, selected-body-only writer clearing with unrelated composer-row preservation, activation-edge damper seeding, inactive-site masking, fixed-shape pulse completion, per-step net-wrench limiting, body-local wrench conversion, release interval-event parity, strict/public versus prevalidated parity, and production-source fast-path routing without dynamic due indices.

2. Full `sonic_backup` CPU/CUDA/Hydra suite

   ```bash
   _CHIP_PHASE2_CUDA_LIB=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu
   _CHIP_PHASE2_CUDA_PRELOAD="${_CHIP_PHASE2_CUDA_LIB}/libnvidia-ml.so.580.159.03:${_CHIP_PHASE2_CUDA_LIB}/libcuda.so.580.159.03"
   _CHIP_PHASE2_VK_ICD=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json
   env LD_LIBRARY_PATH="${_CHIP_PHASE2_CUDA_LIB}" \
       LD_PRELOAD="${_CHIP_PHASE2_CUDA_PRELOAD}" \
       VK_ICD_FILENAMES="${_CHIP_PHASE2_VK_ICD}" \
       PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Required result: every discovered test passes with no skip (65/65 in the current regression run). The Hydra tests must compose only the derived compliance experiment, prove its interval-event set and ranges exactly match the release experiment, and resolve both full and partial runtime body-name sets without fixed indices.

3. Dedicated portable CUDA scale-profiler assertion

   ```bash
   env LD_LIBRARY_PATH="${_CHIP_PHASE2_CUDA_LIB}" \
       LD_PRELOAD="${_CHIP_PHASE2_CUDA_PRELOAD}" \
       VK_ICD_FILENAMES="${_CHIP_PHASE2_VK_ICD}" \
       PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       -m unittest \
       gear_sonic.tests.compliance.test_sonic_phase2_adapter.HotPathHostSyncTest.test_cuda_bound_compute_has_no_dynamic_indices_or_scalar_extraction \
       -v
   ```

   Required result: one test passes. At 4096 environments and 14 configurable sites, both `TorchDispatchMode` and `torch.profiler` call the portable mixin's bound `compute(dt)` with deterministic tensor fixtures and a fake articulation/composer. The scale audit executes countdown → fixed-size candidate sampling → due/start masks → state update → pulse completion → resultant-wrench limit → body-local adapter → damper update. The trace must contain neither `aten::nonzero` nor `aten::_local_scalar_dense`; disabled compute must consume no private/global RNG, and enabled compute must preserve the process-global CPU/CUDA RNG states exactly. This test complements, but does not substitute for, the real Isaac-bound smoke below.

4. One-environment disabled and enabled simulator smokes

   ```bash
   _CHIP_PHASE2_MOTION=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl
   _CHIP_PHASE2_SMPL=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered
   env LD_LIBRARY_PATH="${_CHIP_PHASE2_CUDA_LIB}" \
       LD_PRELOAD="${_CHIP_PHASE2_CUDA_PRELOAD}" \
       VK_ICD_FILENAMES="${_CHIP_PHASE2_VK_ICD}" \
       PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       gear_sonic/scripts/run_chip_compliance_smoke.py \
       --headless --device cuda:0 --no-enabled --steps 100 \
       --motion-file "${_CHIP_PHASE2_MOTION}" \
       --smpl-motion-dir "${_CHIP_PHASE2_SMPL}"
   env LD_LIBRARY_PATH="${_CHIP_PHASE2_CUDA_LIB}" \
       LD_PRELOAD="${_CHIP_PHASE2_CUDA_PRELOAD}" \
       VK_ICD_FILENAMES="${_CHIP_PHASE2_VK_ICD}" \
       PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       gear_sonic/scripts/run_chip_compliance_smoke.py \
       --headless --device cuda:0 --enabled --steps 100 \
       --motion-file "${_CHIP_PHASE2_MOTION}" \
       --smpl-motion-dir "${_CHIP_PHASE2_SMPL}"
   ```

   Both runs must print `CHIP_PHASE2_SMOKE_PASS`. Disabled mode must first complete all 100 steps with the writer inactive, zero applied wrench, infinite command-owned pulse countdowns, and exact process-global CPU/CUDA RNG preservation. Only after those checks pass, the same real AppLauncher environment must print `CHIP_PHASE2_REAL_BOUND_PROFILE_PASS`: it temporarily enables the actual `SonicComplianceCommand`, forces its private countdown due, and calls the bound `command.compute(dt)` under `TorchDispatchMode` plus CPU/CUDA profiler activities. That trace must reject `aten::nonzero` and `aten::_local_scalar_dense` while covering actual articulation `index_select`/link-frame helpers, `ArticulationWrenchAdapter`, and Isaac Lab `WrenchComposer`; the portable scale audit separately covers the optional target-damper path. Before any subsequent `env.step`, it must switch off and prove command state/countdown/ownership plus the real composer's selected force and torque rows are cleared. The separate enabled invocation then runs the normal forced test: it must observe a non-zero wrench, reconstruct world force from current body-local rows, verify local-offset torque, and report peak net force/torque below configured limits. After 100 enabled steps it must also preserve an unrelated sentinel through immediate off, re-enable using only private RNG, cancel again, remain clear for two disabled steps, and reset all command/damper state.

5. Import, syntax, and patch hygiene

   Run the Phase 1 import/syntax and patch-hygiene commands unchanged. The released `sonic_release.yaml` must have no diff, and no generated cache or temporary result may remain in the task scope.

## Phase 3 — Checkpoint-compatible policy/critic integration

Run from the repository root. Phase 3 inherits the complete Phase 2 matrix.

1. Portable CPU suite

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Required result: all 79 discovered tests pass; 26 Hydra/CUDA/official-model
   tests are expected to skip in the portable environment.

2. Resolved CPU/Hydra and official-checkpoint suite

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Required result: all 79 discovered tests pass, with only four expected CUDA
   skips when the compatibility driver is not active. The twelve Phase-3 resolved
   integration tests must verify the pinned official checkpoint hash and
   55-policy/17-value legacy schemas; exactly six initialized residual keys per
   model; byte-exact legacy tensors; strict branch-checkpoint resume; 64D
   post-FSQ disabled and zero-compliance parity even after forcing nonzero branch
   weights; byte-exact inactive rows under mixed gating; a non-overridable
   privileged-force rejection in direct and rollout paths; public-only rollout
   history; real actor/critic construction for 1/2/5/14/17 sites with independent
   future counts; no extra global CPU/CUDA RNG advance during residual
   construction; one shared, once-normalized critic context; frozen release
   encoder/quantizer/G1 decoder/noise/value ownership; repeated distribution and
   optimizer steps that leave the official frozen std byte-exact; finite
   G1/teleop/SMPL auxiliary losses; and nonzero first-backward output-head
   gradients. Zero-initialized residual trunks are expected to have zero first
   backward gradients and must not be asserted to update on that step.

3. Inherited Phase-2 CUDA/Hydra suite and scale profiler

   Run Phase-2 items 2 and 3 unchanged after explicit GPU approval. The expanded
   discovery must pass 79/79 with no skips, and the dedicated 4096-environment,
   14-site profiler must still pass.

   For the final 2026-07-27 independent-review correction, the parent accepted
   the already-passing inherited CUDA/Hydra suite and scale profiler without a
   repeat because no Phase-2 command/writer path changed. The changed Phase-3
   model paths instead require the complete 79-test CPU/Hydra/official suite and
   a fresh real Phase-3 model smoke from item 5.

4. Inherited real disabled/enabled simulator smokes

   Run Phase-2 item 4 unchanged after the CUDA/Hydra suite passes. Both
   `CHIP_PHASE2_SMOKE_PASS` invocations and the disabled run's
   `CHIP_PHASE2_REAL_BOUND_PROFILE_PASS` remain mandatory.

5. One-environment resolved Phase-3 shape/model smoke

   ```bash
   env LD_LIBRARY_PATH="${_CHIP_PHASE2_CUDA_LIB}" \
       LD_PRELOAD="${_CHIP_PHASE2_CUDA_PRELOAD}" \
       VK_ICD_FILENAMES="${_CHIP_PHASE2_VK_ICD}" \
       PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       gear_sonic/scripts/run_chip_phase3_shape_smoke.py \
       --headless --device cuda:0 \
       --motion-file "${_CHIP_PHASE2_MOTION}" \
       --smpl-motion-dir "${_CHIP_PHASE2_SMPL}" \
       --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt
   ```

   Required marker: `CHIP_PHASE3_SHAPE_SMOKE_PASS`. The real observation manager
   must resolve actor/critic/tokenizer/target/command/force widths to
   `930/1645/1761/60/9/6`, instantiate and load both derived models, produce
   `(1,1,29)` actions and `(1,1,1)` values, keep command/force exactly zero by
   default, match the release actor output byte-for-byte, and leave the official
   frozen direct-std tensor byte-exact after repeated distribution construction
   while returning the same effective clamped std as the release policy.

6. Import, syntax, release-path, and patch hygiene

   Run Phase-1 item 2 and item 3 unchanged, including the new Phase-3 sources.
   Additionally compile and audit
   `gear_sonic/scripts/run_chip_phase3_shape_smoke.py` for one final newline,
   no trailing whitespace, and no generated cache beside it.
   The released `sonic_release.yaml`, released tokenizer/dense observations,
   rewards, encoder/decoder definitions, and existing checkpoint files must
   remain unchanged.

## Phase 4 — Low-resource finetune smoke and parity regression

Run from the repository root. Phase 4 inherits the complete Phase 3 matrix.

1. Portable and resolved CPU suites

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python \
     -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Every discovered test must pass. Expected skips are limited to explicitly
   unavailable Hydra/official-model/CUDA tests in the portable interpreter and
   CUDA-only tests in the pinned interpreter before compatibility libraries are
   activated. Phase-4 coverage must prove byte-exact tensor comparison,
   recursive optimizer/scheduler comparison, finite-loss rejection, bounded
   symlink-safe artifact accounting, atomic JSON cleanup, five initial versus
   one resumed batch semantics, centralized collision-safe paths, exact command
   vectors, forced per-site exposure, 12 tensors/770753 scalars, and final audit
   execution in `on_step_end` before the trainer's early return. A constructive
   resume test must cover a checkpoint whose adaptive-KL argument LR differs
   from its serialized post-scheduler optimizer LR, and prove exact optimizer
   and scheduler restoration while retaining the argument LR. The resolved model
   test must also exercise the real PPO leading shape ``[B=4, S=24]``
   with different per-timestep gates/targets, prove post-FSQ residual alignment
   without timestep-zero broadcast, retain ``[B,S]`` on action/decoder/aux
   paths, and produce finite nonzero gradients for all six actor residual
   tensors.

2. Pinned training entrypoint and resolved command gate

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/train_agent_trl.py --help
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/run_chip_phase4_finetune.py \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_dry_run \
     --dry-run
   ```

   Both commands must exit zero. Dry-run must create no directory and must show
   release `resume=false`, residual `resume=false`, and independent branch
   `resume=true` with exactly one new batch (not six) and final-step audit 6.

3. Inherited real CUDA/Hydra and simulator matrix

   After explicit GPU approval, run Phase-3 items 3, 4, and 5 serially using
   the `580.159.03` compatibility environment. The complete expanded suite,
   4096-environment profiler, disabled/enabled Phase-2 simulator smokes, and
   Phase-3 real shape/model smoke must retain their prior pass markers. No
   compliance training process may run concurrently with another GPU job.

4. Exact stiff and compliant training workflow

   The accepted evidence root is
   `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix`.
   Preserve it unchanged. To reproduce the workflow, replace
   `<fresh-run-root>` in `plan.md` with a unique absolute child of
   `compliance_control/runs/chip`; that path must not exist before launch. The
   workflow must serially execute the exact
   16-environment, five-iteration released-config command from `plan.md`, the
   16-environment five-batch residual warm-start, and the independent one-batch
   strict resume. Required final marker: `CHIP_PHASE4_FINETUNE_PASS`.

   The stiff log must reach learning iteration 5 and exit successfully. The
   compliance step 5 must save `last.pt` and a complete audit proving: official
   SHA/source step, 55 policy and 17 value legacy tensors byte-exact, exactly
   six actor plus six critic residual tensors, optimizer ownership of exactly
   those 12 tensors/770753 scalars, nonzero change and finite nonzero gradient
   history for every tensor, finite loss at each step 1-5, true nonzero-force
   exposure at every configured site, and positive peak CUDA allocation.

   The resume job must preserve step 5, restore branch model/optimizer/scheduler
   byte-exactly at callback start and global step 5, run one new batch, save an
   independent step-6 `last.pt`, and pass the same audit for loss step 6 plus
   residual change relative to step 5. Do not describe this as trajectory-level
   bitwise replay because process RNG and private command state are not saved.

5. Artifact, release-path, process, and patch hygiene

   Require each run to stay below 1.2 GB, every log below 64 MB, and the complete
   workflow below 2.5 GB. All generated artifacts must remain below
   `compliance_control/runs/chip`; official assets must retain their hash and
   must never be staged. Verify no Isaac/training process remains and no GPU
   compute application remains after the serial jobs. Then run Phase-1 syntax,
   import, cache, `git diff --check`, and `git diff --cached --check` gates over
   the new Phase-4 modules, runner, tests, and task documents. The released
   `sonic_release.yaml`, generic PPO trainer, and generic
   `UniversalTokenModule` must have no diff. Changes to previously added
   compliance adapters must remain Phase-4-scoped and explicitly tested.

Until the host NVIDIA packages are aligned, Phase 4 must use the validated
`580.159.03` compatibility environment recorded in `plan.md`. This workaround
does not waive any real-GPU training or smoke requirement.

## Phase 5 — Tracking/compliance evaluation and export

Run from the repository root. Phase 5 retains the immutable accepted Phase-4
checkpoint and reruns all portable/resolved regression tests because the
evaluation code extends the reusable compliance package. It does not retrain or
rewrite the accepted Phase-4 artifacts.

1. Portable and resolved CPU suites

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
     -s gear_sonic/tests/compliance -p 'test_*.py' -v
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python \
     -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Every discovered test must pass. The current final CPU regression discovers
   127 tests: portable Python passes with 38 expected dependency skips, and
   `sonic_backup` passes with four expected CUDA skips. Dependency-based skips
   are allowed only in the portable interpreter, and CUDA-only tests may skip
   in `sonic_backup` before compatibility libraries are activated. Phase-5
   coverage must prove:
   strict key/time/reference/force/gate alignment without interpolation; fixed
   horizons and common valid prefixes; persisted structured local-frame metadata
   and distinct reference/articulation index provenance; true paired
   compliant-minus-stiff site
   yielding with force projection; the temporal tail of each contiguous force
   pulse; left/right position and orientation RMSE/P95 split into all/exposed/
   unexposed frames; finite normalized `wxyz` and sign-invariant geodesic angle;
   named per-site all-frame position/orientation RMSE and P95 acceptance checks;
   arbitrary 17-site evaluation; bounded non-pickle NPZ/JSON round trips;
   explicit residual ONNX I/O and dynamic axes; active PyTorch/ONNX parity; and
   exact hard-off/zero-compliance output.

   Trace/ONNX pair publication must also inject a failure after the binary has
   been published but before metadata/manifest publication, then prove both
   final paths and every same-directory hidden temporary are removed.
   Standalone export/rollout negatives must place broken symlinks at requested
   leaf outputs and prove no resolved target is created. Audit negatives must
   reject substituted workflow run/runs roots, a checkpoint symlink alias, and
   an incorrect per-rollout checkpoint SHA-256.

   The tracker-neutral trace remains arbitrary-size, but the SONIC Hydra test
   must compare eval and release `motion.body_names` element-for-element,
   require exactly 14 bodies, and lock matching runtime/audit provenance.

2. Entrypoint, dry-run, syntax, and import gates

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/run_chip_phase5_eval_export.py --help
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/audit_chip_phase5.py --help
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/run_chip_phase5_eval_export.py \
     --runs-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase5_dry_run \
     --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix/compliance_residual_step6_resume/last.pt \
     --motion-file /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
     --smpl-motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered \
     --onnxruntime-python /home/lab/miniconda3/envs/sonic/bin/python \
     --onnxruntime-version 1.25.0 \
     --steps 300 --seed 0 --device cuda:0 --dry-run
   ```

   Both help commands and the dry-run must exit zero. The dry-run path must be
   absent before and after execution; output must contain exactly one stiff and
   one compliant serial rollout, dimensions `60/9/930 -> 64`, the complete
   acceptance thresholds, and the matched-force zero-residual stiff semantics.
   Compile all Phase-5 Python files with built-in `compile`, then import the
   tracker-neutral evaluation and postprocess modules without Isaac Lab.

   Run the focused deployment-runtime parity in the pinned CPU interpreter:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic/bin/python -B -m unittest \
     gear_sonic.tests.compliance.test_sonic_phase5_export.SonicPhase5ExportTest.test_separate_onnx_dynamic_and_hard_off_parity \
     -v
   ```

   It must report a real `onnxruntime.InferenceSession` version `1.25.0` with
   only `CPUExecutionProvider`; a reference-evaluator result is not accepted.

3. One bounded real-GPU evaluation and export workflow

   The destination below must not exist before launch. Run only after explicit
   GPU authorization and with no concurrent training/Isaac process:

   ```bash
   env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
     LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
     VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
     PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/run_chip_phase5_eval_export.py \
     --runs-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase5_acceptance \
     --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix/compliance_residual_step6_resume/last.pt \
     --motion-file /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
     --smpl-motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered \
     --onnxruntime-python /home/lab/miniconda3/envs/sonic/bin/python \
     --onnxruntime-version 1.25.0 \
     --steps 300 --seed 0 --device cuda:0
   ```

   Required final marker: `CHIP_PHASE5_EVAL_EXPORT_PASS`. The stiff and
   compliant logs must each contain `CHIP_PHASE5_ROLLOUT_PASS`; stiff peak
   latent residual must be exactly zero, compliant peak residual positive, and
   their reference/force schedules exactly aligned. Every named threshold from
   the Phase-5 plan must pass, including at least `1e-6` m paired displacement
   as a chain-activation check. The signed along-force result remains a reported
   diagnostic and must not be described as a performance claim. The workflow
   must also emit a checked residual
   ONNX and manifest with dynamic shape parity, bit-exact hard-off and
   zero-compliance results, and unchanged release-model semantics.

4. Independent artifact and metric audit

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic/bin/python -B \
     gear_sonic/scripts/audit_chip_phase5.py \
     --runs-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase5_acceptance \
     --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix/compliance_residual_step6_resume/last.pt
   ```

   Required marker: `CHIP_PHASE5_INDEPENDENT_AUDIT_PASS`. A fresh process must
   reload both traces with pickle disabled, recompute every metric/check,
   compare it to the recorded JSON, re-run ONNX checker/parity, verify checkpoint
   and ONNX hashes, and confirm the stiff/compliant policy semantics. It must
   compare canonical workflow `run_root`, `runs_root`, and checkpoint paths to
   the CLI arguments, then independently compare each rollout summary's
   canonical checkpoint path and SHA-256 to that same checkpoint.

5. Artifact, release-path, process, and patch hygiene

   The workflow must stay below 500 MB and each log below 64 MB; all generated
   files must stay inside `compliance_control/runs/chip`, contain no symlink, and
   never overwrite the accepted checkpoint or official assets. Confirm no
   training/Isaac process or GPU compute application remains. Run the Phase-1
   compile/import/cache/EOF/trailing-whitespace and staged/unstaged diff gates.
   The released `sonic_release.yaml`, release encoder/decoder ONNX and observation
   contracts, generic PPO trainer, and generic `UniversalTokenModule` must have
   no diff. Existing git refs must still match
   `compliance_control/existing_refs_before.txt`.

## Phase 6 — Final regression and handoff

Run from the repository root.  Phase 6 is a final regression and evidence
handoff; it changes no policy/model, simulator behavior, trainer, or export
graph.  The
complete earlier CPU matrices are rerun below.  The accepted Phase-4 GPU
training and Phase-5 300-frame GPU workflow are immutable inputs: do not launch
them again and do not modify their files.  Their GPU-only claims are inherited
only after the fresh semantic/hash/metric/ORT audits below pass.

1. Complete portable and resolved Phase-1-through-Phase-5 regression

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
     -s gear_sonic/tests/compliance -p 'test_*.py' -v
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     -m unittest discover -s gear_sonic/tests/compliance -p 'test_*.py' -v
   ```

   Both commands must discover and pass all 129 tests.  The portable interpreter
   has 39 explicitly dependency-gated skips; `sonic_backup` has only four
   CUDA-only skips when the compatibility libraries are not preloaded.  This is
   the complete CPU/Hydra/official-checkpoint regression for Phases 1-5, not a
   selected subset.

2. Entrypoint help, derived-config, and collision-free dry-run gates

   Run every shipped CHIP entrypoint help path in `sonic_backup`:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/scripts/run_chip_compliance_smoke.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/scripts/run_chip_phase3_shape_smoke.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/scripts/run_chip_phase4_finetune.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/scripts/run_chip_phase5_rollout.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/scripts/run_chip_phase5_eval_export.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/scripts/audit_chip_phase5.py --help
   PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/chip_compliance_finetune/artifacts/phase6_final_audit.py --help
   ```

   Every command must exit zero without creating a simulator/runtime output.
   The three AppLauncher entrypoints have a focused regression:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
     gear_sonic.tests.compliance.test_phase6_entrypoint_help -v
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m unittest \
     gear_sonic.tests.compliance.test_phase6_entrypoint_help -v
   ```

   Portable execution must pass the source/AST test with one expected Isaac Lab
   skip; `sonic_backup` must pass both tests.  The tests pin each accepted
   runtime `main()` AST, require bare help to exit zero without warning or
   traceback, and require missing launch arguments to remain an argparse exit 2.
   The full resolved suite in item 1 remains the derived Hydra configuration
   composition gate.  Then run both exact orchestration dry runs, with each
   destination absent before and after:

   ```bash
   test ! -e /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase6_phase4_dry_run
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/run_chip_phase4_finetune.py \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase6_phase4_dry_run \
     --dry-run
   test ! -e /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase6_phase4_dry_run
   test ! -e /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase6_phase5_dry_run
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     gear_sonic/scripts/run_chip_phase5_eval_export.py \
     --runs-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase6_phase5_dry_run \
     --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix/compliance_residual_step6_resume/last.pt \
     --motion-file /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
     --smpl-motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered \
     --onnxruntime-python /home/lab/miniconda3/envs/sonic/bin/python \
     --onnxruntime-version 1.25.0 \
     --steps 300 --seed 0 --device cuda:0 --dry-run
   test ! -e /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase6_phase5_dry_run
   ```

   Phase 4 must still print one release warm start, one residual warm start, and
   one one-batch resume.  Phase 5 must print exactly one serial stiff rollout,
   one serial compliant rollout, the `60/9/930 -> 64` residual export, all
   thresholds, and the matched-force release-equivalent comparator semantics.

3. Real ONNX Runtime and independent Phase-5 accepted-artifact audit

   ```bash
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic/bin/python -B -m unittest \
     gear_sonic.tests.compliance.test_sonic_phase5_export.SonicPhase5ExportTest.test_separate_onnx_dynamic_and_hard_off_parity \
     -v
   PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic/bin/python -B \
     gear_sonic/scripts/audit_chip_phase5.py \
     --runs-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip \
     --run-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase5_acceptance \
     --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix/compliance_residual_step6_resume/last.pt
   ```

   The focused test must use `onnxruntime.InferenceSession` 1.25.0 with only
   `CPUExecutionProvider`.  The second command must emit
   `CHIP_PHASE5_INDEPENDENT_AUDIT_PASS`, reload NPZ with pickle disabled,
   recompute every metric/check, repeat real-ORT dynamic/mixed/hard-off parity,
   and revalidate workflow plus per-rollout checkpoint provenance.

4. Read-only golden, checkpoint, layout, refs, and process audit

   Run the audit with the validated compatibility libraries so its NVML process
   query is meaningful.  This command reads the accepted artifacts and must not
   rewrite them:

   ```bash
   env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
     LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
     VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
     PYTHONDONTWRITEBYTECODE=1 \
     /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
     tasks/chip_compliance_finetune/artifacts/phase6_final_audit.py \
     --refs-snapshot /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/existing_refs_before.txt \
     --official-checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt \
     --official-config /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/config.yaml \
     --robot-motion /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
     --smpl-motion-dir /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered \
     --phase4-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase4_acceptance_resume_fix \
     --phase5-root /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/chip/phase5_acceptance
   ```

   Required marker: `CHIP_PHASE6_FINAL_AUDIT_PASS`.  The audit must reload both
   Phase-4 checkpoints and re-prove 55/17 legacy tensors, 6+6 residual tensors,
   12 optimizer slots/770753 scalars, finite loss/exposure/gradient evidence,
   step-5-to-step-6 advancement, and official frozen-tensor identity.  It must
   pin the official/sample assets, checkpoint/ONNX hashes, and complete evidence
   trees.  Expected layout digests and inventories are:

   - Phase 4: digest
     `34cba4405dee146c7dd5f29d4731001737e8ae85f6f4d79e3928317b5bb02503`,
     31 files, 9 directories, exactly one internal resume symlink, and
     318016496 bytes including the link entry.
   - Phase 5: digest
     `9efef42178353072faa457f49934c6fa67ffbf852628470e1f9bbc384046c81e`,
     14 files, 3 directories, zero symlinks, and 1655744 bytes.

   It must also prove that every difference from the official base commit is an
   addition and that the exact diff after accepted Phase-5 commit
   `c925a0da115d1d6e0cc296c4a94b00a57c6461b8` contains only the ten declared
   Phase-6 entrypoint/help/audit/task paths.  Any Phase-6 edit to an accepted
   core/adapter/training/export/evaluation path must fail.  All release
   config/model/trainer/reward/deployment paths are
   byte-exact, every ref in `existing_refs_before.txt` is unchanged, source/task
   trees contain no symlink/cache/temp/trailing whitespace, no CHIP training or
   rollout process remains, and NVML reports no GPU compute application.
   `--skip-gpu-process-check` is available only to diagnose all structural and
   `/proc` gates when NVML cannot be accessed; it emits
   `CHIP_PHASE6_STRUCTURAL_AUDIT_PASS` plus
   `gpu_process_gate=SKIPPED_NOT_ACCEPTED` and cannot satisfy this item.

5. Final syntax, portable import, and patch hygiene

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -B - <<'PY'
   from pathlib import Path

   roots = (
       Path("gear_sonic/compliance_control"),
       Path("gear_sonic/scripts"),
       Path("gear_sonic/tests/compliance"),
       Path("tasks/chip_compliance_finetune"),
   )
   for root in roots:
       for path in root.rglob("*.py"):
           if root.name == "scripts" and "chip" not in path.name:
               continue
           compile(path.read_bytes(), str(path), "exec")
   PY
   PYTHONDONTWRITEBYTECODE=1 python -B -c "from gear_sonic.compliance_control import AlignedTrackingTrace, CartesianFrameSpec, ComplianceTargetSpec, TargetDamper, apply_hindsight_target; from gear_sonic.compliance_control.postprocess import load_tracking_trace; print(AlignedTrackingTrace, CartesianFrameSpec, ComplianceTargetSpec, TargetDamper, apply_hindsight_target, load_tracking_trace)"
   git diff --check
   git diff --cached --check
   ```

   The compile path must create no bytecode.  Repeat the cache/temporary/final-
   newline audit after every preceding command.  `git status --short` may list
   only the Phase-6 task/audit/handoff changes before the final commit is made.
