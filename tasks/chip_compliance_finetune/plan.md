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

Add a separate compliance experiment config, force/compliance sampling events, same-frame force projection, per-environment target-damper state/reset, and a non-mutating SONIC MDP observation adapter that consumes the Phase 1 core. Resolve reference-motion and articulation sites independently from Hydra/runtime body-name fields. Apply the declared full/yaw rotation into the structured common frame before hindsight math; never pass indices between spaces.

### Phase 3 — Checkpoint-compatible policy/critic integration

Add the smallest compliance-conditioned SONIC adapter/residual branch to the G1 encoder path, feed the Phase 2 damped/hindsight target through the compliance-only path, expose privileged force only to the critic during training, and add a thin checkpoint adapter with zero-initialized/gated parameters so the released checkpoint retains exact stiff behavior on load. Keep the dense tracking rewards and upper-limb terms enabled; do not move policy architecture into the tracker-agnostic core.

### Phase 4 — Low-resource finetune smoke and parity regression

Run the documented low-resource training smoke, first with compliance disabled and then with force/compliance sampling enabled. Confirm finite losses, nonzero adapter gradients, deterministic zero-compliance parity, bounded artifacts, and checkpoint save/resume.

Pinned local smoke assets (never stage these binaries):

- Checkpoint: `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt`, Hugging Face revision `7c90a56c`, SHA-256 `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`, recorded step `41550`.
- Robot motion: `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl`.
- SMPL directory: `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered`.

Pinned stiff-baseline smoke command:

```bash
conda run -n sonic_backup python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt \
  num_envs=16 headless=True use_wandb=false \
  ++algo.config.num_learning_iterations=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered
```

As of 2026-07-27 this command is blocked before launch by an NVIDIA driver mismatch: kernel module `580.159` versus userspace `580.173`. Phase 4 cannot pass until GPU health is restored and the command is rerun; do not substitute a claimed CPU training result.

### Phase 5 — Tracking/compliance evaluation and export

Evaluate stiff tracking and force-response trade-offs with frame-aligned logs. Gate acceptance on upper-limb end-effector tracking, global/local MPJPE, fall/success rate, contact displacement, and peak/steady force metrics. Export ONNX and verify that the optional compliance input is explicit and disabled-mode behavior matches the release path.

### Phase 6 — Final regression and handoff

Run the complete matrix, document data/environment requirements and cleanup, record metrics against the release baseline, and mark the task complete only if every phase passes.
