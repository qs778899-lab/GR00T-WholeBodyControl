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

   Required result: 45 tests pass. The repository's portable `.venv_sim` run has four expected CUDA/Hydra skips; a CPU environment with Hydra installed has only the two CUDA skips. The suite covers CPU shape alignment, separate typed index spaces, arbitrary site sets, deterministic CHIP-faithful discrete compliance sampling, non-zero yaw/full-rotation frame and CHIP-sign checks, non-mutation, pulse scheduling, activation-edge damper seeding, full/partial reset, inactive-site masking, per-step net-wrench limiting, body-local wrench conversion, disabled writer ownership, and production-source fast-path routing.

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

   Required result: all 45 tests pass with no skip. The Hydra tests must compose only the derived compliance experiment and resolve both full and partial runtime body-name sets without fixed indices.

3. Dedicated CUDA host-sync profiler assertion

   ```bash
   env LD_LIBRARY_PATH="${_CHIP_PHASE2_CUDA_LIB}" \
       LD_PRELOAD="${_CHIP_PHASE2_CUDA_PRELOAD}" \
       VK_ICD_FILENAMES="${_CHIP_PHASE2_VK_ICD}" \
       PYTHONDONTWRITEBYTECODE=1 \
       /home/lab/miniconda3/envs/sonic_backup/bin/python \
       -m unittest \
       gear_sonic.tests.compliance.test_sonic_phase2_adapter.HotPathHostSyncTest.test_cuda_prevalidated_hot_path_does_not_extract_scalars \
       -v
   ```

   Required result: one test passes. Both `TorchDispatchMode` and `torch.profiler` must observe no `aten::_local_scalar_dense` in the prevalidated production path.

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

   Both runs must print `CHIP_PHASE2_SMOKE_PASS`. Disabled mode must keep the writer inactive and report zero applied wrench. Enabled mode must observe a non-zero wrench, reconstruct the world force from the composer's current body-local rows, verify local offset torque, and report actual peak net force/torque below configured limits. After 100 enabled steps it must switch the same command to disabled for two steps: the first clears ownership/selected composer rows, the second keeps ownership false and those rows zero. Reset must then clear all command and damper state.

5. Import, syntax, and patch hygiene

   Run the Phase 1 import/syntax and patch-hygiene commands unchanged. The released `sonic_release.yaml` must have no diff, and no generated cache or temporary result may remain in the task scope.

## Phase 3 — Checkpoint-compatible policy/critic integration

- Phase 2 matrix.
- Released checkpoint load with documented missing/new keys only.
- Disabled-mode policy output parity against the release path.
- Nonzero compliance-adapter gradient and no actor access to privileged force.
- Encoder selection and auxiliary-loss tests for G1/teleop/SMPL samples.
- Assert the compliance-only actor path consumes the Phase 2 damped/hindsight target while the released dense reference/reward path remains unchanged.

## Phase 4 — Low-resource finetune smoke and parity regression

- Phase 3 matrix.
- `train_agent_trl.py --help` in the pinned Isaac Lab training environment.
- The exact `sonic_backup` 16-environment, 5-iteration release-config command in `plan.md`, using the pinned single robot PKL, SMPL sample directory, official checkpoint, and `use_wandb=false`.
- Compliance-enabled 16-environment smoke with finite losses and checkpoint resume.
- Bounded log/artifact size and cleanup verification.

The 2026-07-27 `580.159` kernel / `580.173` userspace NVIDIA mismatch is an explicit blocker, not a waiver for these tests.

## Phase 5 — Tracking/compliance evaluation and export

- Phase 4 matrix.
- Frame/time-aligned stiff baseline versus compliant checkpoint evaluation.
- Upper-limb endpoint, local/global MPJPE, success/fall, displacement, and force metrics.
- ONNX export plus disabled-mode inference parity.

## Phase 6 — Final regression and handoff

- All earlier matrices.
- Full documented regression/golden comparison and output-layout check.
- Final artifact/temporary-file audit.
