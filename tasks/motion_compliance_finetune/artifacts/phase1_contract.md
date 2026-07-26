# Phase-1 baseline and portability contract

## Audited upstream interfaces

- Baseline is NVLabs upstream commit
  `4141c34280abb67c82e115342a8720f4a83d750d`.
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
  lines 88-90 feed `command_multi_future_nonflat` and
  `motion_anchor_ori_b_mf_nonflat` to the G1 robot-motion encoder.
- `gear_sonic/envs/manager_env/mdp/commands.py` lines 897-903 define the first
  term as future joint positions concatenated with future joint velocities.
- `gear_sonic/envs/manager_env/mdp/observations.py` lines 1022-1043 define the
  second term as a 6D anchor-orientation difference for each future frame.
- The separate tracking/reward skeleton is the body-name list in
  `gear_sonic/config/manager_env/commands/terms/motion.yaml` lines 52-67.  It is
  not an encoder keypoint tensor and must not define compliance site count.
- Policy observations use the `local_dir_hist` group selected at line 14 of
  `sonic_release.yaml`.  The three public compliance scalars will be added there
  only by a new opt-in experiment in Phase 3; robot-motion encoder inputs remain
  unchanged.

## Legacy boundary

Upstream has an uncomposed `ForceTrackingCommand` at
`gear_sonic/envs/manager_env/mdp/commands.py:3468`.  It carries hard-coded
three-point state and reports metrics, but its command output is `None` and the
docstring states that absent event terms must apply forces.  The push-event
configuration references `compliance_force_push` and
`chip_change_compliance_discrete/unified`, neither of which exists in this
baseline.  Legacy helpers in `observations.py` and `rewards.py` are therefore
not used as the new integration point.

## Reusable core boundary

The new `gear_sonic/compliance_control/core` package is intentionally small:

- `schema.py`: validated public parameters and event-schedule values;
- `math.py`: condition encoding, threshold-to-stiffness conversion, and norm
  clamping;
- `schedule.py`: tracker-independent event envelope and site-mask sampling;
- `reference_modifier.py`: globally gated per-site target selection and virtual
  force construction;
- `metrics.py`: tracking, yield, force, and preservation metrics.

It accepts caller-supplied tensors and site counts.  References use shape
`[batch, ..., sites, 3]`; a `[batch, sites]` mask/threshold broadcasts across
optional axes such as future frames.  It has no IsaacLab import, body or joint
name, kinematic-tree order, fixed action dimension, fixed skeleton size, or
concrete torso/world frame.  `enabled=false` overrides even a stale all-active
site mask, leaving the original reference bit-for-bit unchanged and producing
exactly zero virtual force.  Tensor paths remain differentiable for finetuning.

## Audited `motion_tracking` force contract

The source was verified against Axellwppr `motion_tracking` compliance branch
commit `0526770c015cb3175074c1defa52357f76b37964`.  In
`active_adaptation/envs/mdp/commands/motion_tracking.py` lines 1218-1222 and
1254-1262 it implements:

```text
Kp = threshold / 0.05
nominal = clamp_norm((modified - original) * Kp, threshold)
tracking = clamp_norm((modified - current) * 100, 5 N)
force_on_robot = nominal + tracking
```

The two terms are independently norm-clamped and their sum is not clamped
again, so the per-site sum can reach `threshold + 5 N`.  Positive force points
from the original/current position toward the modified target and is applied to
the robot.  The source computes reference and current quantities in semantically
corresponding local torso coordinates and rotates the force to world at the
simulator boundary.  The portable core does not reproduce that implicit frame
approximation: it requires the adapter to express original, compliant, and
current positions in one explicitly chosen common Cartesian frame and returns
`force_on_robot` in that same frame.

Omitting `current_reference` or setting `include_tracking_term=false` selects
the nominal-only compatibility path.  The original reference continues to feed
the universal tracker encoder; compliant selection is for synthetic-force and
training-target/reward paths, not an encoder-input replacement.  Metrics inspect
the raw compliant candidate at inactive sites, so inactive drift can expose
adapter pollution rather than being a tautological zero after selection.

## Planned thin integration map

Phase 2 will put all simulator and embodiment knowledge under
`gear_sonic/compliance_control/adapters/sonic/`: a name/offset resolver, a
persistent IsaacLab command adapter, and a narrow MDP registration module.  A
new command composition will opt into it.  Phase 3 will add separate observation
and reward terms plus a new experiment YAML.  Phase 4 will add a shape-aware
checkpoint migrator; Phase 5 will add a compliance-specific deployment
observation config.  Existing release files remain untouched.

## Audited later-phase assets and smoke command

The immutable official checkpoint is
`/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt`
at Hugging Face revision `7c90a56c`, training step 41550, SHA-256
`e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`.
The matching single robot input and SMPL directory are:

- `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl`
- `/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered`

After the opt-in experiment exists, the prescribed five-iteration smoke is:

```bash
/home/lab/miniconda3/envs/sonic_backup/bin/python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_motion_compliance \
  +checkpoint=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sonic_release/last.pt \
  num_envs=16 headless=True use_wandb=false \
  algo.config.num_learning_iterations=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets/sample_data/smpl_filtered
```

This command is documented now but must not be claimed as passing before the
later phases exist.  GPU execution is currently externally blocked by the
NVIDIA 580.159 kernel module / 580.173 user-space driver mismatch.

Finally, the threshold is policy conditioning for synthesized virtual forces;
it is not a certified safety limit for real hardware contact.
