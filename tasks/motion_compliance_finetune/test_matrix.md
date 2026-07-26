# Test matrix

Every command is run from the repository root.  A phase may advance only after
all tests listed for that phase pass.  Python tests use
`PYTHONDONTWRITEBYTECODE=1`; pytest runs disable the cache provider so validation
does not leave repository caches.

## Phase 1 — Baseline contract and tracker-agnostic core skeleton

1. `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_core.py`
2. `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -c 'from gear_sonic.compliance_control.core import ComplianceSpec, encode_compliance_condition; import torch; s=ComplianceSpec(); assert encode_compliance_condition(torch.tensor([False]), torch.tensor([s.force_threshold_range_n[0]])).shape == (1, 3)'`
3. `git diff --check`
4. After staging all and only Phase-1 files, `git diff --cached --check`.
5. The unit test must import every core module without IsaacLab and inspect the
   core AST to reject IsaacLab imports, robot-specific names, and integer index
   assumptions for 29-DoF / 14-point layouts or a concrete torso frame.
6. Run shape tests with caller-provided 2-, 7-, and 17-site layouts plus
   `[batch, future, sites, 3]` references using `[batch, sites]` masks and
   thresholds.  No site/future count may be encoded in production core.
   Backpropagate through selected references and both force terms and require
   finite gradients.
7. With every site mask bit active, assert `enabled=false` still returns the
   original reference bit-for-bit and exactly zero virtual force.
8. Verify the upstream formula and sign: nominal and tracking terms are clamped
   separately, summed without a second clamp, point from original/current toward
   compliant target, and the result is `force_on_robot`.  Verify nominal-only
   compatibility explicitly.
9. Reject reference shape/dtype/device mismatches, non-boolean active masks,
   non-binary/non-finite `enabled`, and non-finite displacement/current/force
   parameters.
10. Metrics must apply the global enabled gate and detect candidate compliant-
    reference drift at inactive sites before reference selection.
11. Confirm the Phase-1 diff contains no edits to existing SONIC runtime,
   training, environment, or release configuration files.
12. Confirm no `__pycache__`, `.pytest_cache`, or compiled Python files remain.

## Phase 2 — IsaacLab virtual-force command adapter

1. Run the combined Phase-1/Phase-2 pure suite:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_core.py gear_sonic/tests/test_compliance_sonic_adapter.py`.
2. The suite must cover deterministic seeded sampling, disabled/single-site/
   simultaneous masks, threshold/Kp coupling, partial reset clearing every
   dynamic tensor, full per-site formula over multiple future frames, separate
   reference/articulation body-index spaces, non-identity frame rotation and
   sign, a changing-body quaternion for current world-to-local wrench
   conversion, and 1/2/5-site residual-wrench reconstruction/clamping.
3. The suite must assert that only the SONIC adapter imports IsaacLab or owns
   SONIC body names, exercise both modern-composer and deprecated-setter feature
   paths with body-local fakes, reject scalar extraction/value validation in the
   per-step unchecked tensor path, verify disabled-clean writer calls are zero,
   and Hydra-compose the opt-in command/event groups with one environment.
4. Run the real one-environment headless smoke with the NVIDIA 580.159
   compatibility libraries:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase2_isaaclab_smoke.py`.
   It must instantiate the real manager environment with the audited sample,
   execute exactly 100 disabled and 100 forced-on policy steps, use the modern
   composer, prove it remains inactive throughout disabled mode, keep every
   checked tensor finite, observe a nonzero forced wrench, and clear
   command/composer force and torque on reset.
5. `git diff --check` and, after staging only Phase-1/Phase-2 files,
   `git diff --cached --check`.
6. Confirm no `__pycache__`, `.pytest_cache`, `.pyc`, or `.pyo` remains.

## Phase 3 — Observation, reward, and experiment composition

1. Run all Phase-1 and Phase-2 tests.
2. Hydra compose `sonic_release_motion_compliance` and inspect resolved groups.
3. Assert robot-motion tokenizer term names and shapes equal `sonic_release`.
4. Assert policy observation grows by exactly 3 and critic-only fields do not
   enter actor observations.
5. Off-mode reward/reference golden test: active-site mask is empty and every
   original reference tensor is bitwise equal before/after adaptation.
6. Active-site test: only selected endpoint targets yield; all inactive
   reference targets remain bitwise equal.

## Phase 4 — Checkpoint migration and finetune workflow

1. Run all earlier tests.
2. Synthetic checkpoint migration test: old weights/biases copy exactly and new
   input columns are exactly zero.
3. Released-checkpoint load smoke with `resume=false`; assert no missing legacy
   weights and report only expected new parameters.  Source checkpoint:
   `compliance_control/official_assets/sonic_release/last.pt`, HF revision
   `7c90a56c`, step 41550; verify its recorded digest before use.
4. One PPO rollout/update with a small environment count; verify finite losses,
   checkpoint save, and strict reload of the migrated checkpoint.
5. Motion data smoke with one robot PKL from the six audited official samples
   and `sample_data/smpl_filtered`; no modified/duplicate dataset is generated.
6. Training entrypoint `--help`/Hydra help succeeds.

## Phase 5 — Export and deployment switch

1. Run all earlier unit/config tests.
2. Export encoder and decoder ONNX.
3. Verify encoder input names/shapes exactly match the released robot-motion
   encoder contract.
4. Compare PyTorch and ONNX decoder outputs for disabled and enabled conditions.
5. Disabled deployment golden: three condition zeros and baseline action output
   within `1e-5` absolute tolerance.
6. CLI/config validation rejects invalid threshold/displacement combinations.

## Phase 6 — Integration and regression validation

1. Run the complete test suite from Phases 1–5.
2. Low-resource IsaacLab smoke/performance run in `sonic_backup`: one audited
   robot PKL, `sample_data/smpl_filtered`, `num_envs=16`, 5 iterations, and
   `use_wandb=false`; record FPS/GPU memory.  If GPU execution remains blocked
   by the NVIDIA 580.159 kernel / 580.173 user-space mismatch, record the exact
   failure and do not claim this test passed.
3. Paired baseline regression, same seeds/motion IDs/timestamps:
   - success-rate drop no more than 1 percentage point;
   - local MPJPE regression no more than 3 mm or 10%, whichever is larger;
   - left/right hand endpoint RMSE regression no more than 5 mm in off mode.
4. Enabled/no-contact endpoint metrics remain in the same range as off mode.
5. Single-left, single-right, and simultaneous two-site force trials:
   record force peaks, yielded displacement, endpoint RMSE, falls, and reset
   behavior; no NaN/Inf or persistent wrench is permitted.
6. Confirm output directories contain no unintended caches, duplicate JSON, or
   multi-GB debug logs.
