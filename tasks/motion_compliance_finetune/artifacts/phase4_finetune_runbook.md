# Phase 4 motion-compliance finetune runbook

All generated checkpoints, Hydra files, logs, and JSON reports must remain
below:

```text
/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion
```

The initialization adapter accepts only the audited SONIC release:

- checkpoint: `compliance_control/official_assets/sonic_release/last.pt`
- SHA-256: `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`
- Hugging Face revision: `7c90a56cfe04788c4f041daeef5b1e12930675ad`
- trainer step: `41550`

The default Phase-4 experiment is the prescribed low-resource run: one
official robot PKL plus the official SMPL directory, 16 environments, five PPO
iterations, W&B disabled, and `last.pt` saved at step 5. Compliance examples
are synthesized online; this workflow does not create a modified motion
dataset.

## Initial five-step sample run

Choose one new directory under the owned runs root and use it consistently:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune \
  experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke \
  headless=true
```

The config migrates the official checkpoint to
`phase4_gpu_smoke/artifacts/motion_compliance_init.pt` with `resume=false`, then
trains only `g1_dyn` and the critic. It writes per-step, per-site exposure to
`phase4_gpu_smoke/artifacts/motion_compliance_exposure.json`. That bounded JSON
also records one finite loss/timing sample per PPO iteration and process peak
CUDA memory.

The experiment selects the compliance-only
`MotionComplianceFrozenNoiseActor`. It has the same checkpoint keys and
effective `[0.001, 0.5]` action-noise clamp as the released actor, but computes
the clamp out of place. This is required because the staged workflow freezes
`std`: four values in the audited release are only about `1e-5` above `0.5`, and
the generic actor's in-place forward clamp would otherwise mutate frozen state.
Do not replace this target with the generic actor for `decoder_critic` runs.
The smoke validator also rejects overrides of `use_log_std=false`,
`use_clampped_std=true`, `std_clamp_min=0.001`, `std_clamp_max=0.5`, or
`clamp_noise_std=false`. The isolated trainer logs the effective clamped mean
noise while leaving the raw frozen checkpoint tensor untouched.

Record the fixed-shape all-environment scheduler cost separately, without
instrumenting or changing the training hot path:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  tasks/motion_compliance_finetune/artifacts/phase4_candidate_scheduler_benchmark.py \
  --num-envs 16 \
  --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/artifacts/candidate_scheduler_benchmark.json
```

Audit step 5 before resuming:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B \
  tasks/motion_compliance_finetune/artifacts/phase4_checkpoint_audit.py \
  --official /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt \
  --trained /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/last.pt \
  --exposure /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/artifacts/motion_compliance_exposure.json \
  --expected-step 5 \
  --output-json /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/artifacts/step5_audit.json
```

## Strict step-5 to step-6 resume

The resume branch disables migration explicitly and requires complete,
non-empty model, optimizer, scheduler, environment, and trainer state:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune \
  experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume \
  resume=true \
  motion_compliance_finetune.resume_output_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_resume \
  motion_compliance_checkpoint_migration.enabled=false \
  checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_gpu_smoke/last.pt \
  algo.config.num_learning_iterations=1 \
  callbacks.model_save.save_last_frequency=1 \
  headless=true
```

This writes step 6 to
`phase4_gpu_resume/last.pt` and preserves the audited step-5 checkpoint. Run the
same audit with `--expected-step 6`,
`phase4_gpu_resume/artifacts/motion_compliance_exposure.json`, and a new
`phase4_gpu_resume/artifacts/step6_audit.json` output.

## Full motion-data finetune

Keep the same experiment config and override only the two loader-boundary
paths. The core, trainer, and compliance controller are unchanged:

```bash
env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 \
  VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune \
  experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/full_data_run \
  manager_env.commands.motion.motion_lib_cfg.motion_file=/absolute/robot_filtered \
  manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/absolute/smpl_filtered \
  manager_env.commands.motion.motion_lib_cfg.multi_thread=true \
  motion_compliance_finetune.enforce_phase4_smoke_contract=false \
  headless=true
```

Do not point `experiment_dir`, migration output, checkpoint audit output, or
exposure output outside the owned central runs root; the workflow rejects such
paths before training artifacts are created.
