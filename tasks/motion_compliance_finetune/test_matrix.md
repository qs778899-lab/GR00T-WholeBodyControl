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

## Phase 4 — Same-shape residual initialization and finetune workflow

1. Run the combined Phase-1/2/3/4 pure suite:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_core.py gear_sonic/tests/test_compliance_sonic_adapter.py gear_sonic/tests/test_compliance_sonic_training.py gear_sonic/tests/test_compliance_training_checkpoint.py`.
   It must prove that the release actor/critic/RMS remain 994/1645/1645 and
   that 997/1657 are separate residual contexts, never expanded base inputs.
   Require exactly 55 official policy + six action-residual tensors and 17
   official value + six value-residual tensors, float32 residuals, zero output
   layers, byte-exact base tensors, fresh step 0, and no optimizer/scheduler/env
   state.  Reject schema-v1 artifacts, unmarked non-resume checkpoints, partial
   model schemas, and incomplete resume roots before any live-state mutation.
2. The suite must freeze every released policy/value parameter and make exactly
   twelve residual tensors trainable.  Match HF optimizer ordering precisely:
   policy W0/W2/W4, value W0/W2/W4, policy b0/b2/b4, value b0/b2/b4.  Run two
   synthetic `[4,24,*]` updates and require finite gradients plus a byte change
   for every residual tensor.  Validate action `[4,24,29]`, value `[4,24,1]`,
   condition `[4,24,3]`, privileged `[4,24,9]`, and tokenizer
   `[4,24,...]`; the physical exposure callback remains coarse exposure
   evidence and is not treated as proof of all 24 frames.
3. The suite must constructively test strict resume with different saved
   boundaries (`args.learning_rate=1e-5`, optimizer group LRs `2e-5/3e-5`).
   Before loading, preflight exact checkpoint roots, both model schemas and
   finiteness, two optimizer groups/12 slots/moment shapes, scheduler state,
   `env_state_dict.motion_lib`, trainer tensors, and global step.  After load,
   recursively compare model/optimizer/scheduler/global-step state to the saved
   payload and prove optimizer LRs were not overwritten from args.
4. Run the pinned official CPU residual-initialization smoke.  Reruns may
   atomically overwrite only this new generated artifact; old Phase-4 outputs
   are invalid evidence:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_official_residual_init_smoke.py --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_cpu_gate/artifacts/motion_compliance_residual_init.pt --report /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_cpu_gate/artifacts/residual_init_audit.json --num-sites 2 --overwrite`.
   Require SHA-256
   `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`,
   revision `7c90a56cfe04788c4f041daeef5b1e12930675ad`, source step
   41550, source policy key `policy_state_dict`, official tensor counts 55/17,
   byte-exact base state loaded independently from the pinned file, release
   widths 994/1645/1645, residual contexts 997/1657, and fresh empty training
   state.
5. Run both entrypoint/config gates:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py --help` and
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune exp_base=phase4_test timestamp=20260727_000000 --cfg job`.
   Resolve the isolated trainer/actor/backbone/critic targets, residual-only
   stage, `[256,256]` residual heads, delta limit 0.25, 24 rollout steps, five
   PPO epochs, four mini-batches, no symmetry, frozen noise, 16 environments,
   five iterations, W&B off, official sample paths, and save frequency 5.
6. Before any Phase-4 GPU command, obtain explicit root-agent confirmation.
   Then rerun the exact Phase-2 and Phase-3 real CUDA regression commands from
   their matrix sections with the NVIDIA 580.159 compatibility environment.
7. Record the 16-environment fixed-shape scheduler cost without changing its
   algorithm:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_candidate_scheduler_benchmark.py --num-envs 16 --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/candidate_scheduler_benchmark.json`.
8. With no prior checkpoint/training files in `phase4_residual_gpu_smoke_tensordict_fix` (the
   benchmark JSON is allowed), run exactly five iterations:
   `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix headless=true`.
   Require the isolated trainer's `[4,24,*]` gates on every PPO micro-batch,
   `last.pt` at step 5, finite loss/timing per iteration, nonzero two-site
   physical exposure, and process peak CUDA memory.
9. Independently audit step 5 against both pinned official and the separate init:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_checkpoint_audit.py --official /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt --initialization /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/motion_compliance_residual_init.pt --trained /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/last.pt --exposure /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/motion_compliance_exposure.json --expected-step 5 --num-sites 2 --output-json /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/step5_audit.json`.
   All 55+17 base tensors, including `g1_dyn`, critic, RMS, quantizer, and
   `std`, must remain byte-exact to official.  Every one of the twelve residual
   tensors must differ from initialization and have a matching finite nonzero
   optimizer moment; no official optimizer step may survive.
10. Strict-resume step 5 into an independent step-6 directory:
    `env LD_LIBRARY_PATH=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu LD_PRELOAD=/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.159.03:/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu/libcuda.so.580.159.03 VK_ICD_FILENAMES=/tmp/nvidia_580_159_compat/extracted/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix resume=true motion_compliance_finetune.resume_output_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix motion_compliance_checkpoint_initialization.enabled=false checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/last.pt algo.config.num_learning_iterations=1 callbacks.model_save.save_last_frequency=1 headless=true`.
    Before the next batch, require recursive-exact restored model, optimizer,
    scheduler and global-step boundaries, while preserving distinct saved args
    and optimizer LRs.  Execute one PPO batch and save step 6 without changing
    step 5.
11. Audit step 6 with the same independently saved initialization:
    `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase4_checkpoint_audit.py --official /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt --initialization /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_smoke_tensordict_fix/artifacts/motion_compliance_residual_init.pt --trained /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix/last.pt --exposure /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix/artifacts/motion_compliance_exposure.json --expected-step 6 --num-sites 2 --output-json /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix/artifacts/step6_audit.json`.
12. `git diff --check`; after staging all and only Phase-4 files,
    `git diff --cached --check`.  Confirm the generic
    `gear_sonic/trl/trainer/ppo_trainer.py` has no diff and remove all
    repository-local `__pycache__`, `.pytest_cache`, `.pyc`, and `.pyo` files.

## Phase 5 — Export and deployment switch

1. Rerun the combined Phase-1/2/3/4 pure suite in the pinned IsaacLab
   environment:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_core.py gear_sonic/tests/test_compliance_sonic_adapter.py gear_sonic/tests/test_compliance_sonic_training.py gear_sonic/tests/test_compliance_training_checkpoint.py`.
   Rerun the Phase-4 help/config gates, then run the Phase-5 deployment suite in
   the existing CPU ORT environment:
   `env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/lab/miniconda3/envs/sonic/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_deployment.py`.
   The `sonic` pytest plugin autoload is disabled because its unrelated
   typeguard entry point is incompatible with that environment's
   `typing_extensions`; torch 2.7, ONNX 1.21, and ONNX Runtime 1.25 remain the
   tested application libraries.  The portable deployment package must also
   import under `sonic_backup`, which has no Python ONNX Runtime installation.
2. From a missing final bundle directory, export the accepted step-6 action
   residual exactly once:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic/bin/python -B tasks/motion_compliance_finetune/artifacts/phase5_export_action_residual.py --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix/last.pt --expected-checkpoint-sha256 42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba --expected-global-step 6 --num-sites 2 --deployment-overlay gear_sonic_deploy/policy/motion_compliance/action_residual_overlay.yaml --output-directory /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase5_action_residual_export_isaaclab_order/bundle`.
   Export only the six trained action-residual tensors.  Require inputs
   `release_action_context [B,S,994]` (64-D token then unchanged 930-D actor
   observation) and `motion_compliance_condition [B,S,3]`; concatenate once
   inside the graph to recover the trained 997-D MLP context.  Require output
   `action_delta [B,S,29]`, dynamic `B/S`, exactly six ONNX initializers, and
   metadata pinning schema, site/action layouts, checkpoint SHA-256/global step,
   residual tensor names/shapes, model digest, and framework versions.  Do not
   rewrite or copy either released ONNX file.  The action layout must equal
   `joint_utils.G1_ISAACLab_ORDER` item-for-item; compose residual and release
   action before the existing deploy-side `isaaclab_to_mujoco` remap.  The
   earlier `phase5_action_residual_export` directory used an incorrect
   MuJoCo/DFS metadata layout and is retained only as invalid evidence.
3. Independently validate the published model and externally pin the accepted
   metadata digest:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic/bin/python -B tasks/motion_compliance_finetune/artifacts/phase5_export_acceptance.py --artifact-directory /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase5_action_residual_export_isaaclab_order/bundle --metadata-sha256 e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5 --checkpoint /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase4_residual_gpu_resume_tensordict_fix/last.pt --checkpoint-sha256 42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba --global-step 6 --num-sites 2 --deployment-overlay gear_sonic_deploy/policy/motion_compliance/action_residual_overlay.yaml --release-artifact decoder /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed --release-artifact encoder /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx 013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3 --release-artifact observation_config /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml 466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b --report /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase5_action_residual_export_isaaclab_order/acceptance.json --overwrite-report`.
   Compare PyTorch and ONNX Runtime for dynamic `[2,3]` and `[1,5]` shapes,
   all-off/all-on/mixed gates, and NaN-poisoned rejected condition/context rows.
   Off rows must be exact zero and finite on rows must agree within `1e-5`.
4. Compile and run the actual portable C++ runtime plus thin SONIC adapter
   against system ONNX Runtime 1.16 and the accepted artifact:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase5_cpp_ort_smoke.py --bundle /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase5_action_residual_export_isaaclab_order/bundle --decoder /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx --encoder /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx --observation-config /home/lab/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml`.
   Require dynamic `[2,3]` mixed-row inference, one- and three-segment arbitrary
   portable host contexts, default-disabled and all-off zero ORT calls, exact
   release-action fallback, finite bounded deltas, actual release-file hashes,
   and incompatible-base rejection.
5. Configure and link the real deployment target without downloading GoogleTest:
   `cmake -S gear_sonic_deploy -B /tmp/gr00t_motion_phase5_cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DTensorRT_ROOT=/home/lab/TensorRT -DG1_DEPLOY_RUNTIME_OUTPUT_DIRECTORY=/tmp/gr00t_motion_phase5_cmake/bin`,
   `cmake --build /tmp/gr00t_motion_phase5_cmake --target g1_deploy_onnx_ref -j2`,
   and `/tmp/gr00t_motion_phase5_cmake/bin/g1_deploy_onnx_ref`.
   The executable must link the portable residual and SONIC adapter and show
   `--motion-compliance-overlay` in help.  The dependency installer must declare
   libmd for apt, yum, and pacman targets.  Then run
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase5_deploy_cli_acceptance.py --binary /tmp/gr00t_motion_phase5_cmake/bin/g1_deploy_onnx_ref --compile-commands /tmp/gr00t_motion_phase5_cmake/compile_commands.json`.
   It must reject NaN, Inf, out-of-range, trailing-character, trailing-empty,
   wrong-width, and missing compliance values before DDS/model initialization even though the
   production target is compiled with `-ffast-math`.  The small portable
   residual target must place `-fno-fast-math` after the global flag so its
   internal finite-value checks remain effective.
6. Runtime composition must preserve supplied release-action bytes for disabled
   rows and bound enabled deltas by 0.25.  The host-disabled and all-row-off
   paths must not call the optional session.  Cover a multi-row/multi-step mixed
   gate in both unit and accepted-artifact tests.
7. The opt-in SONIC overlay remains disabled by default.  Its thin adapter must
   assemble token then actor observation as 994 columns and pass condition only
   as the second graph input.  The production entrypoint may contain only the
   additive CLI/load/compose hook and must compose in IsaacLab/BFS order before
   the existing remap while storing the composed action in `last_action`.
   `--set-compliance 0` must be an explicit run-time residual-off setting.  The
   startup value and `g/h/b/v` adjustments must reach the storage read by
   `InterfaceManager`, `GamepadManager`, and `ZMQManager`, including their
   switched ZMQ/pose delegates; switching an input manager must not silently
   restore the default-on gate.
   Confirm the immutable release boundary with
   `git diff --exit-code 4141c34280abb67c82e115342a8720f4a83d750d -- gear_sonic_deploy/policy/release gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp`.
8. The portable Python and C++ layers must contain no SONIC/G1/IsaacLab/wrist
   names and must accept an ordered caller-owned context layout and an arbitrary
   non-empty vector of named base-artifact pins.  Concrete release hashes,
   token/actor names, wrist sites, action order, and operator-gate semantics
   belong only to the SONIC adapters.
9. Unit and CLI/config validation must reject incompatible schema/site/action
   widths, checkpoint/model/metadata digests, non-binary/non-finite gates, and
   invalid enabled-condition combinations.  Artifact schema v1 must require
   ONNX opset exactly 17 in both Python export/validation and C++ loading.
   ONNX and JSON publishing must
   be atomic; a failed export leaves neither a final partial file nor a staging
   directory.
10. Run `git diff --check`.  Confirm no repository-local `__pycache__`,
   `.pytest_cache`, `.pyc`, or `.pyo` remains, and confirm the released encoder,
   decoder, release observation config, generic trainer, and Phase-1/2/3/4
   training/environment source have no Phase-5 diff.  The additive SONIC export
   symbols in its adapter `__init__` are explicitly allowed.

## Phase 6 — Integration and regression validation

Environment contract for the current Phase-6 continuation: the historical
Phase-2/3/4 commands above preserve the NVIDIA 580.159 compatibility paths that
were actually used for those accepted phase results.  They are provenance, not
the current launch contract.  At the 2026-08-12 pause that temporary directory
is absent. The native host kernel module/CUDA/NVML userspace was observed
matched at 580.173.02 earlier on 2026-08-12, but the final pause query could no
longer communicate with the NVIDIA driver. Phase-6 reruns may use the native
loader without temporary `LD_LIBRARY_PATH`, `LD_PRELOAD`, or
`VK_ICD_FILENAMES` overrides only after revalidating the exact current
versions, CUDA availability in `sonic_backup`, and an idle GPU. If the host
state differs, update this Phase-6 contract before execution.

1. Run the complete test suite from Phases 1–5.
2. Low-resource IsaacLab smoke/performance run in `sonic_backup`: one audited
   robot PKL, `sample_data/smpl_filtered`, `num_envs=16`, 5 iterations, and
   `use_wandb=false`; record FPS/GPU memory.  Use the native matched-driver
   environment above and the fresh output root
   `compliance_control/runs/motion/phase6_residual_gpu_smoke_post_restart`:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_motion_compliance_finetune experiment_dir=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase6_residual_gpu_smoke_post_restart headless=true`.
   Before launch, Hydra composition must still resolve `num_envs=16`, five
   learning iterations, the audited robot/SMPL inputs, and `use_wandb=false`;
   refuse the run rather than silently overriding a changed config.
3. Characterize fixed-shape compliance-candidate overhead at 4096 environments:
   report policy-step time and GPU memory for host-off/baseline and enabled
   scheduling without changing the synchronization-safe algorithm first.  From
   a missing output path, run:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase6_scheduler_benchmark.py --num-envs 4096 --num-sites 2 --output /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/runs/motion/phase6_scheduler_4096.json`.
   The report must label itself scheduler-only, record CUDA-event time per
   policy update plus allocated/reserved peak increments, and refuse to replace
   an existing evidence file.
4. Paired baseline regression, same seeds/motion IDs/timestamps:
   - success-rate drop no more than 1 percentage point;
   - local MPJPE regression no more than 3 mm or 10%, whichever is larger;
   - left/right hand endpoint RMSE regression no more than 5 mm in off mode.
5. Enabled/no-contact endpoint metrics remain in the same range as off mode.
6. Single-left, single-right, and simultaneous two-site force trials:
   record force peaks, yielded displacement, endpoint RMSE, falls, and reset
   behavior; no NaN/Inf or persistent wrench is permitted.
   The following active-mode tracking limits were fixed in source before any
   formal trace was collected and may not be relaxed after observing results:
   - map each caller-owned endpoint site one-to-one to its tracking point;
   - on non-reset active rows, selected-target endpoint RMSE/P95 regression
     versus the exactly aligned overlay-off original-target error is at most
     5 mm / 10 mm for each active wrist;
   - wrist orientation remains referenced to the original target and its
     RMSE/P95 regression versus overlay-off is at most 0.05 / 0.10 rad;
   - whole-body preservation excludes only a tracking point whose mapped site
     is active on that row. Remaining-point local MPJPE regression is at most
     3 mm or 10% of overlay-off, whichever is larger; global MPJPE regression
     is at most 5 mm or 10%, whichever is larger;
   - apply every bound independently to single-left, single-right, and
     simultaneous trials. The portable report and SONIC recomputation gate
     must contain the exact fixed values and fail closed on a changed mapping,
     missing check, or relaxed limit.
7. Confirm output directories contain no unintended caches, duplicate JSON, or
   multi-GB debug logs.
8. Run the tracker-neutral aligned-evaluation CPU contract:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B -m pytest -p no:cacheprovider -q gear_sonic/tests/test_compliance_evaluation.py gear_sonic/tests/test_compliance_sonic_evaluation_collector.py`
   and the thin runner help gate:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase6_evaluate_aligned_traces.py --help`.
   Also run the non-GPU scheduler CLI gate:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase6_scheduler_benchmark.py --help`.
   Run the SONIC collector/final-validator help gates:
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase6_collect_sonic_trace.py --help --headless` and
   `env PYTHONDONTWRITEBYTECODE=1 /home/lab/miniconda3/envs/sonic_backup/bin/python -B tasks/motion_compliance_finetune/artifacts/phase6_validate_sonic_collection_reports.py --help`.
   It must reject any motion/sequence/seed/frame/timestamp or ordered-layout
   mismatch, including a one-bit timestamp change.  With caller-owned site and
   point layouts and explicit endpoint-to-tracking-point mapping it reports
   per-site original/selected endpoint RMSE/P95,
   quaternion orientation error, local/global MPJPE, force/yield, paired
   inactive-site cross-coupling, success/fall/reset state, stale post-reset
   force, and input/derived finiteness for baseline, hard-off, enabled/no-contact,
   single-site, and simultaneous multi-site trials.
   Every sequence must have one first-row post-reset snapshot and one final-row
   terminal outcome; falls are terminal-only.  Every expected interaction site
   must show caller-thresholded nonzero active force and reference yield, while
   every inactive site remains below the configured force/yield tolerances.
   The portable report must include each input NPZ full-file SHA-256 obtained
   through the same verified descriptor used for decoding. The SONIC final
   validator must bind that hash to the collection summary and observed file,
   recompute the complete portable report from all six traces under the exact
   Phase-6 criteria, and reject deleted/fabricated checks, boolean-for-integer
   substitutions, changed motion/checkpoint/protocol/stimulus/timing evidence,
   and any baseline/off action-byte difference.
9. The portable evaluation package must contain no SONIC/G1/IsaacLab/MuJoCo,
   concrete body-name, fixed action-width, or fixed tracking-point-count
   dependency.  Persisted traces/reports use bounded atomic NPZ/JSON writers;
   loading disables pickle, bounds both compressed and uncompressed sizes, and
   publication refuses accidental overwrite by default.  Concrete endpoint
   roles and simulator log mapping belong only to a thin collector/adapter.
   Loading must use one `O_NOFOLLOW` file descriptor for regular-file, ZIP-size,
   duplicate-member, schema/dtype/rank validation and NumPy decoding, so a path
   replacement cannot swap the verified inode.
   The thin SONIC provenance gate must additionally pin exact termination and
   reset-event function targets, modes, thresholds, body names, command names,
   and parameters rather than accepting names alone. At least one nonzero-force
   Phase-6 trial must exercise the configured `motion_compliance_reset` event
   and prove both command-owned and actual composer force/torque rows are clear;
   explicit post-timeout cleanup alone is insufficient for this final gate.
