# Phase 4 residual-only finetune runbook

All generated checkpoints, Hydra files, logs, and JSON reports must remain
below:

```text
/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion
```

Schema-v1 expanded checkpoints and the old `phase4_cpu_gate`,
`phase4_gpu_smoke`, and `phase4_gpu_resume` outputs are invalid evidence.  Use
only the new `phase4_residual_*` directories.

The initializer accepts only the audited SONIC release:

- checkpoint: `compliance_control/official_assets/sonic_release/last.pt`
- SHA-256: `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`
- Hugging Face revision: `7c90a56cfe04788c4f041daeef5b1e12930675ad`
- trainer step: `41550`
- state schema: 55 policy tensors and 17 value tensors under
  `policy_state_dict` / `value_state_dict`

The release actor input remains 994 and the release critic/RMS remain 1645.
Only six action-residual tensors and six value-residual tensors are added; for
two sites their independent contexts are 997 and 1657.  Every release tensor,
including `g1_dyn`, critic, RMS, quantizer, and `std`, is frozen byte-exact.

## CPU initialization gate

This command is safe to rerun with `--overwrite`; it reloads the hash-verified
official file and directly compares every base tensor:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  tasks/motion_compliance_finetune/artifacts/phase4_official_residual_init_smoke.py \
  --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_cpu_gate/artifacts/motion_compliance_residual_init.pt \
  --report /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_cpu_gate/artifacts/residual_init_audit.json \
  --num-sites 2 \
  --overwrite
```

## Initial five-step sample run

Do not start a GPU run until the CPU gates pass and the root agent explicitly
confirms it.  The prescribed smoke uses one official robot PKL, the official
SMPL directory, 16 environments, five PPO iterations, four mini-batches, five
epochs, and 24 rollout frames:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune \
  experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix \
  headless=true
```

The config first writes
`artifacts/motion_compliance_residual_init.pt`, then trains only the two
residual heads.  The isolated trainer rejects any PPO micro-batch other than
actor/critic/condition/privileged/tokenizer leading shape `[4,24,...]`, with
privileged width 9, action width 29, and value width 1.  The exposure JSON is
coarse physical evidence sampled once per outer iteration; it is not evidence
that all 24 frames reached the residual path.

The compliance-only actor computes the released `[0.001,0.5]` noise clamp out
of place.  It leaves the raw official `std` bytes unchanged and excludes noise
from the optimizer.

Record scheduler overhead separately:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  tasks/motion_compliance_finetune/artifacts/phase4_candidate_scheduler_benchmark.py \
  --num-envs 16 \
  --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/candidate_scheduler_benchmark.json
```

Audit step 5 against both independent sources:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  tasks/motion_compliance_finetune/artifacts/phase4_checkpoint_audit.py \
  --official /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt \
  --initialization /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/motion_compliance_residual_init.pt \
  --trained /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/last.pt \
  --exposure /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/motion_compliance_exposure.json \
  --expected-step 5 \
  --num-sites 2 \
  --output-json /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/step5_audit.json
```

## Strict step-5 to step-6 resume

Strict resume preflights every model and non-model boundary before mutation.
It restores optimizer group LRs and scheduler state exactly; the independently
saved `args.learning_rate` never overwrites optimizer LRs.

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune \
  experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix \
  resume=true \
  motion_compliance_finetune.resume_output_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix \
  motion_compliance_checkpoint_initialization.enabled=false \
  checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/last.pt \
  algo.config.num_learning_iterations=1 \
  callbacks.model_save.save_last_frequency=1 \
  headless=true
```

Audit step 6 with the same pinned official and step-0 residual initialization,
changing only trained/exposure/output paths and `--expected-step 6`.

## Full motion-data finetune

Compliance is synthesized online, so no modified dataset is needed.  Override
only the loader-boundary paths and disable the exact low-resource smoke guard:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune \
  experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/full_data_residual_run \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/absolute/robot_filtered \
  manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/absolute/smpl_filtered \
  manager_env.commands.motion.motion_lib_cfg.multi_thread=true \
  motion_compliance_finetune.enforce_phase4_smoke_contract=false \
  headless=true
```

Do not point initialization, checkpoint, audit, exposure, or Hydra outputs
outside the owned central runs root.
