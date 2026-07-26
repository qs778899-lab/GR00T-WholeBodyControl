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
   and Hydra-compose the opt-in command/event groups with one environment.  It
   must prove command-owned enabled sampling does not advance global CPU/CUDA
   RNG and host-off sampling does not advance even the command generator.
4. Run the real one-environment headless smoke with the NVIDIA 580.159
   compatibility libraries:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase2_isaaclab_smoke.py`.
   It must instantiate the real manager environment with the audited sample,
   execute exactly 100 disabled and 100 forced-on policy steps, use the modern
   composer, prove it remains inactive throughout disabled mode, keep every
   checked tensor finite, observe a nonzero forced wrench, and clear
   command/composer force and torque on reset.  The enabled-to-disabled setter
   must clear owned composer rows immediately before another physics step.
   Before stepping, disabled real
   command reset, repeated compute, and reset-event calls must preserve the next
   global CPU and CUDA random samples bit for bit.  Force the real command timer
   due and call the bound `command.compute(dt)` under
   both `TorchDispatchMode` and `torch.profiler.profile(activities=[CPU,CUDA])`;
   neither complete trace may contain `aten::_local_scalar_dense` or
   `aten::nonzero`.
5. `git diff --check` and, after staging only Phase-1/Phase-2 files,
   `git diff --cached --check`.
6. Confirm no `__pycache__`, `.pytest_cache`, `.pyc`, or `.pyo` remains.

## Phase 3 — Observation, reward, and experiment composition

1. Run the combined Phase-1/2/3 pure suite:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_core.py gear_sonic/tests/test_compliance_sonic_adapter.py gear_sonic/tests/test_compliance_sonic_training.py`.
2. Hydra-compose both `sonic_release` and
   `sonic_release_motion_compliance`.  The resolved tokenizer subtree must be
   fully equal, including keys, functions, parameters, and noise.  The G1 terms
   remain `command_multi_future_nonflat [10,58]` and
   `motion_anchor_ori_b_mf_nonflat [10,6]`, in that order.
   The resolved termination subtree must also remain fully equal.
3. Assert released policy/critic observation subtrees remain fully equal at
   widths 930/1645.  The actor-visible condition is a separate 3D group; the
   critic-only group contains scalar threshold, `3*S` current-frame applied
   site force, and `S` mask for width `1+4*S`, where `S` comes from command
   configuration.  Privileged fields must not enter direct actor input or its
   temporal history.
4. Off-mode golden: the active mask and condition are zero, selection preserves
   every original reference bitwise, both new rewards contribute exactly zero,
   all released dense reward configs remain equal, and the inline
   `feet_acc.weight=-2.5e-6` override survives composition.  The hard gate must
   still return exact zero when disabled errors contain NaN.  Resolved baseline
   and compliance interval-event names/ranges must be identical; there is no
   compliance interval writer event.
5. Active-site test: future frame zero is aligned to current endpoint state,
   only selected position targets yield, every inactive target remains bitwise
   original, and per-site selected/original errors remain independently
   reportable.  The reward must re-read current articulation/reference tensors
   locally rather than consume the prior command-update cache.  Orientation
   always uses original future-zero reference and is independently reportable
   per site.
6. Run the pinned official CPU residual-contract smoke:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase3_official_residual_contract_smoke.py`.
   Require SHA-256 ending `d8909`, actor input `2048x994`, critic input
   `2048x1645`, critic RMS width 1645, exact release policy/critic observation
   groups, and separate condition/privileged widths 3/9 for two sites.  The
   official checkpoint contains no residual keys; residual keys are initialized
   from the target model, never by expanding official tensors.
7. Rerun the Phase-2 real 100-disabled/100-forced smoke after the command cache
   refactor using the exact Phase-2 command above.
8. Run the real Phase-3 resolved-shape/off-golden smoke:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase3_isaaclab_smoke.py`.
   It must instantiate one real manager environment, observe policy/critic
   shapes `[1,930]`/`[1,1645]` and separate condition/privileged shapes
   `[1,3]`/`[1,9]` for two sites, and preserve both G1 tokenizer shapes.  It
   must instantiate the resolved actor/value, retain `g1_dyn=994`,
   critic/RMS=1645, load every official tensor byte-exact with only new residual
   keys missing, and cover zero-init off parity, mixed `[off,on,off]`, poisoned
   rejected rows with finite residual-only gradients, privileged actor
   isolation, aux and external-token paths, bounded deltas, and out-of-place
   frozen-noise clamping.  It also independently reconstructs the tracking
   command's future-zero endpoint reference, returns exact-zero new rewards
   while host-disabled, proves the manager total is bitwise equal to released
   shared contributions, then poisons the prior command cache and proves the
   reward re-reads current physics state without mutating cache or wrench
   buffers.
9. `git diff --check` and, after staging only task-scoped Phase-3 files,
   `git diff --cached --check`.
10. Confirm no `__pycache__`, `.pytest_cache`, `.pyc`, or `.pyo` remains.

## Phase 4 — Checkpoint migration and finetune workflow

1. Run the combined Phase-1/2/3/4 pure suite:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_core.py gear_sonic/tests/test_compliance_sonic_adapter.py gear_sonic/tests/test_compliance_sonic_training.py gear_sonic/tests/test_compliance_training_checkpoint.py`.
   It must cover exact synthetic legacy copies/zero tails, derived `4+4*S`
   critic widths, official-contract pinning, strict init/resume, complete
   non-empty resume state, exact freeze/optimizer ownership, finite loss and
   per-site exposure callbacks, path/collision rejection, post-train poison
   checks, and exact Hydra smoke guards.  Load the pinned release `std`, require
   the dedicated compliance actor target, and prove repeated distribution
   construction retains byte-exact frozen noise state while producing the same
   effective clamp and excluding `std/log_std` from the optimizer.  Reject
   overrides of `use_log_std=false`, `use_clampped_std=true`, clamp bounds
   `[0.001,0.5]`, and `clamp_noise_std=false`; require the isolated trainer's
   noise metric to report the effective clamp rather than the raw parameter.
2. Run the pinned official CPU migration smoke (reruns may atomically overwrite
   only this generated central artifact):
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_official_migration_smoke.py --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_cpu_gate/artifacts/motion_compliance_init.pt --report /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_cpu_gate/artifacts/migration_audit.json --num-sites 2 --overwrite`.
   Require SHA-256
   `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`,
   full revision `7c90a56cfe04788c4f041daeef5b1e12930675ad`, source
   step 41550, actor `994→997`, critic/RMS `1645→1657`, unchanged RMS
   count, fresh step 0, and `optimizer/scheduler/env=None`.
3. Run both entrypoint/config gates:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py --help` and
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune exp_base=phase4_test timestamp=20260727_000000 --cfg job`.
   The latter must resolve the isolated strict trainer, central runs root,
   official checkpoint, one official robot PKL plus official SMPL directory,
   16 environments, 5 iterations, W&B off, and `save_last_frequency=5`.
4. Rerun the exact Phase-2 and Phase-3 real CUDA regression commands from their
   matrix sections with the NVIDIA 580.159 compatibility environment before
   the new training smoke.
5. Record the 16-environment fixed-shape scheduler cost without changing its
   algorithm:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_candidate_scheduler_benchmark.py --num-envs 16 --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/artifacts/candidate_scheduler_benchmark.json`.
   The JSON must contain enabled/host-off CUDA time and incremental peak-memory
   measurements.
6. Using `phase4_gpu_smoke` with no prior checkpoint/training files (the
   scheduler JSON from command 5 is allowed), run the exact five-step training:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke headless=true`.
   Require `last.pt` at step 5, one finite loss/timing record per iteration,
   nonzero enabled/left/right force exposure, and process peak CUDA memory.
7. Audit step 5:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_checkpoint_audit.py --official /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt --trained /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/last.pt --exposure /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/artifacts/motion_compliance_exposure.json --expected-step 5 --num-sites 2 --output-json /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/artifacts/step5_audit.json`.
   Every added actor/critic column must be nonzero; encoder, `g1_kin`,
   quantizer state, and `std/log_std` must remain byte-exact; all tensors and
   optimizer moments must be finite; optimizer ownership/steps must be fresh.
8. Strict-resume step 5 into a separate step-6 output directory:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume resume=true motion_compliance_finetune.resume_output_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume motion_compliance_checkpoint_migration.enabled=false checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/last.pt algo.config.num_learning_iterations=1 callbacks.model_save.save_last_frequency=1 headless=true`.
   It must restore strict model/optimizer/scheduler/environment/trainer state,
   execute exactly one PPO batch, preserve step 5, and save step 6 under
   `phase4_gpu_resume`.
9. Audit the independently saved step 6:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_checkpoint_audit.py --official /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt --trained /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume/last.pt --exposure /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume/artifacts/motion_compliance_exposure.json --expected-step 6 --num-sites 2 --output-json /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume/artifacts/step6_audit.json`.
10. `git diff --check`; after staging all and only Phase-4 files,
    `git diff --cached --check`.  Confirm the generic
    `gear_sonic/trl/trainer/ppo_trainer.py` has no diff and remove all
    repository-local `__pycache__`, `.pytest_cache`, `.pyc`, and `.pyo` files.

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
   `use_wandb=false`; record FPS/GPU memory.  Use the same temporary NVIDIA
   580.159 compatibility-library environment as the passing Phase-2/3 real
   tests and revalidate it after a machine restart.
3. Characterize fixed-shape compliance-candidate overhead at 4096 environments:
   report policy-step time and GPU memory for host-off/baseline and enabled
   scheduling without changing the synchronization-safe algorithm first.
4. Paired baseline regression, same seeds/motion IDs/timestamps:
   - success-rate drop no more than 1 percentage point;
   - local MPJPE regression no more than 3 mm or 10%, whichever is larger;
   - left/right hand endpoint RMSE regression no more than 5 mm in off mode.
5. Enabled/no-contact endpoint metrics remain in the same range as off mode.
6. Single-left, single-right, and simultaneous two-site force trials:
   record force peaks, yielded displacement, endpoint RMSE, falls, and reset
   behavior; no NaN/Inf or persistent wrench is permitted.
7. Confirm output directories contain no unintended caches, duplicate JSON, or
   multi-GB debug logs.
