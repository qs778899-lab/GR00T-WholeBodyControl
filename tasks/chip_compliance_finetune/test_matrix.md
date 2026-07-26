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

- Phase 3 matrix.
- `train_agent_trl.py --help` in the pinned Isaac Lab training environment.
- The exact `sonic_backup` 16-environment, 5-iteration release-config command in `plan.md`, using the pinned single robot PKL, SMPL sample directory, official checkpoint, and `use_wandb=false`.
- Compliance-enabled 16-environment smoke with finite losses and checkpoint resume.
- Bounded log/artifact size and cleanup verification.

Until the host NVIDIA packages are aligned, Phase 4 must use the validated
`580.159.03` compatibility environment recorded in `plan.md`. This workaround
does not waive any real-GPU training or smoke requirement.

## Phase 5 — Tracking/compliance evaluation and export

- Phase 4 matrix.
- Frame/time-aligned stiff baseline versus compliant checkpoint evaluation.
- Upper-limb endpoint, local/global MPJPE, success/fall, displacement, and force metrics.
- ONNX export plus disabled-mode inference parity.

## Phase 6 — Final regression and handoff

- All earlier matrices.
- Full documented regression/golden comparison and output-layout check.
- Final artifact/temporary-file audit.
