# 代码与论文证据

## 审计基线

| 对象 | 固定版本 |
|---|---|
| `Axellwppr/motion_tracking` | [`0526770c015cb3175074c1defa52357f76b37964`](https://github.com/Axellwppr/motion_tracking/tree/0526770c015cb3175074c1defa52357f76b37964)，`compliance` branch |
| `Axellwppr/gentle-humanoid` | [`12aee8d84b9bf998c910b717eb9e3cbedd4d0559`](https://github.com/Axellwppr/gentle-humanoid/tree/12aee8d84b9bf998c910b717eb9e3cbedd4d0559) |
| README 链接的 Gentle training repo | [`9fd915d9639fc9b219e4b12b04076f67db82c7b5`](https://github.com/Axellwppr/gentle-humanoid-training/tree/9fd915d9639fc9b219e4b12b04076f67db82c7b5) |
| `unified-force/UniFP` | [`68847a070f88d731058c3d8476929bc3b205f5bd`](https://github.com/unified-force/UniFP/tree/68847a070f88d731058c3d8476929bc3b205f5bd) |
| `Improbable-AI/softmimic` | [`823582557d5417c9382303ba749262afab625076`](https://github.com/Improbable-AI/softmimic/tree/823582557d5417c9382303ba749262afab625076) |
| CHIP | [arXiv:2512.14689 v2](https://arxiv.org/abs/2512.14689)，2026-02-09 |
| GentleHumanoid paper | [arXiv:2511.04679 v1](https://arxiv.org/abs/2511.04679v1)，2025-11-06 |
| UniFP paper | [arXiv:2505.20829 v2](https://arxiv.org/abs/2505.20829v2)，2025-10-04 |
| SoftMimic paper | [arXiv:2510.17792 v1](https://arxiv.org/abs/2510.17792v1)，2025-10-20 |

分支和论文都会变化，所以以下判断只对应这些版本。`motion_tracking/compliance`
的 audited README 没有给出论文或 benchmark，本文将它视为 repo-only 工程参考。
仓库根目录没出现 license 时，只记录“repo-level license 未发现”，不把论文公开
等同为源码授权。

截至 2026-07-27 又做了一次公开页面复核：[CHIP 官方项目页](https://nvlabs.github.io/CHIP/)仍显示
`Code (Coming Soon)`，因此本项目的 CHIP 分支是依据论文公式和 SONIC 现有接口的
clean-room 工程实现，不是官方 CHIP 代码移植；`motion_tracking/compliance` README
仍给出约 4×A100、15 小时的完整训练口径；Gentle 主仓仍定位为 inference/deploy，
训练实现由另一个仓库承载；UniFP 主仓仍把训练 pipeline、ROS2 sim2real、MuJoCo
sim2sim 和 imitation pipeline 列为 TODO；SoftMimic 主仓仍是带 pretrained models、
augmentation、训练和部署目录的 MIT release。固定 commit 的逐行结论优先于这些
会变化的 README 状态。

## 1. 当前 SONIC 链路

### 1.1 robot-motion encoder 的真实输入

Release 配置把 G1 encoder 输入定义为：

- `command_multi_future_nonflat`：`q_ref` 与 `qdot_ref`；属性实现在 [`commands.py`](../gear_sonic/envs/manager_env/mdp/commands.py) L897-L903；
- `motion_anchor_ori_b_mf_nonflat`：reference/robot anchor orientation 的 6D rotation difference；实现在 [`observations.py`](../gear_sonic/envs/manager_env/mdp/observations.py) L1022-L1043；
- 10 个 future frames，见 [`sonic_release.yaml`](../gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml) L44-L50；
- encoder MLP 与 2-token 输出见 [`g1_mf_mlp.yaml`](../gear_sonic/config/actor_critic/encoders/g1_mf_mlp.yaml)。

G1 是 29 DoF，所以每帧 `29 + 29 + 6 = 64`，原始 temporal tensor 为 `[E,10,64]`，编码后部署 token dimension 为 64。Release deployment 的 G1 mode 也只要求 10-frame joint position、velocity、anchor orientation，见 [`observation_config.yaml`](../gear_sonic_deploy/policy/release/observation_config.yaml) L57-L64。

### 1.2 14 点是什么

[`motion.yaml`](../gear_sonic/config/manager_env/commands/terms/motion.yaml) L52-L67 定义了 14 个 `body_names`：pelvis，左右 hip/knee/ankle，torso，以及左右 shoulder/elbow/wrist。它们供 motion state、reward 和 evaluation 使用，不是 G1 encoder 的直接输入。

Release 还专门提高 torso 与双 wrist 的 3-point reward 权重，见 [`sonic_release.yaml`](../gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml) L44-L47。这是保护上肢末端精度时应继续使用的现成监督信号。

### 1.3 已存在的 compliance/CHIP 残留不能直接运行

本仓已有若干接口线索：

- [`ForceTrackingCommand`](../gear_sonic/envs/manager_env/mdp/commands.py) L3468-L3613：force buffers、双腕/头 3 维 `eef_stiffness_buf`、Jacobian 和指标；
- [`vr_3point_target_compliant_multi_future`](../gear_sonic/envs/manager_env/mdp/observations.py) L1249-L1292、`compliance` 与 `ext_forces` L1836-L1860；
- [`tracking_vr_3point_error_pos_force`](../gear_sonic/envs/manager_env/mdp/rewards.py) L368-L394；
- [`token_losses.py`](../gear_sonic/trl/losses/token_losses.py) L819-L1045 中的 stiff-only G1/SMPL latent alignment；
- C++ deploy 已能构造 `vr_3point_compliance`，但 release observation config 没有把它接入任何 encoder mode。

链路仍有四个硬缺口：

1. command config 只注册 `motion`，没有 `ForceTrackingCommandCfg`；
2. [`push_robot.yaml`](../gear_sonic/config/manager_env/events/terms/push_robot.yaml) 注释引用的 `compliance_force_push` 和 `chip_change_compliance_discrete` 文件/事件函数不存在；
3. release 模型没有 compliance observation，部署端改变值不会改变策略；
4. multi-future observation 当前有可复现的广播错误：`body_force_magnitude_buf` 是 `[E,B]`，代码用 `[:,None,None,None]` 得到 `[E,1,1,1,B]`，无法与 `[E,F,1,1]`、`[E,1,B,3]` 相乘；后续还把 2 个 wrist force displacement 直接减到 3-point/head target。

因此这些代码只能作为接口草稿，不能作为已集成的 CHIP 实现。

## 2. `motion_tracking/compliance`

### 关键代码

- G1 task 采用 `MotionTrackingComplianceCommand`，policy 加 `compliance_flag_obs`，同时保留 full-body tracking rewards：[`G1_tracking.yaml` L31-L147](https://github.com/Axellwppr/motion_tracking/blob/0526770c015cb3175074c1defa52357f76b37964/cfg/task/G1/G1_tracking.yaml#L31-L147)。
- tracking points 共 13 个；virtual-force points 是左右 `hand_mimic` 与 `wrist_roll_link` 共 4 个。
- Actor 的条件是 `[flag, flag*threshold, flag*Kp]`；threshold 为 10–20 N，`Kp=threshold/0.05`：[`motion_tracking.py` L1061-L1223](https://github.com/Axellwppr/motion_tracking/blob/0526770c015cb3175074c1defa52357f76b37964/active_adaptation/envs/mdp/commands/motion_tracking.py#L1061-L1223)。
- 上肢肩/肘 joint trajectory 被 A-B-C smoothstep 修改；policy 的 joint target 仍显式选择 `source: original`，而 FK/reward target 使用 modified motion：[`joint_modifier.py` L42-L214](https://github.com/Axellwppr/motion_tracking/blob/0526770c015cb3175074c1defa52357f76b37964/active_adaptation/utils/joint_modifier.py#L42-L214)。
- 根据 original/modified endpoint displacement 生成 virtual force，各 link 截断到 threshold；总 wrench 超限部分由 torso residual 抵消：[`motion_tracking.py` L1241-L1280](https://github.com/Axellwppr/motion_tracking/blob/0526770c015cb3175074c1defa52357f76b37964/active_adaptation/envs/mdp/commands/motion_tracking.py#L1241-L1280)。
- 部署配置直接暴露 `compliance_flag_value` 与 `compliance_flag_threshold`：[`sim2real/config/tracking.yaml` L1-L6](https://github.com/Axellwppr/motion_tracking/blob/0526770c015cb3175074c1defa52357f76b37964/sim2real/config/tracking.yaml#L1-L6)。Actor 不读取真实 force/contact，而是从 motion target、root angular velocity、projected gravity、joint position/velocity 与 previous-action history 等可部署输入隐式响应，见 [`G1_tracking.yaml` L82-L98](https://github.com/Axellwppr/motion_tracking/blob/0526770c015cb3175074c1defa52357f76b37964/cfg/task/G1/G1_tracking.yaml#L82-L98)；输出仍为固定 KP/KD 下的 joint-position target。

### 适配判断

优点是模式开关干净、original target 仍在、训练/ONNX/VR runtime 齐全，最适合借鉴成 SONIC 的首版可切换接口。它也允许四个 hand/wrist links 同时受 virtual force，但四点由同一次肩/肘 trajectory modification 耦合产生，并共享一个 global flag、threshold 和 `Kp`；这不是四点独立 compliance command。

但它不是 runtime 闭环力控制器，10–20 N 只是训练条件而不是硬安全上限；没有 force-matching safety reward、公开 benchmark、单测或 CI。其 policy observation 为 1590 维，框架是 mjlab，不能 shape-compatible 地加载 SONIC checkpoint。README 给出的完整 train/adapt/finetune 流程约需 4×A100、15 小时，并不是对 SONIC 的小型 supervised finetune。

该固定 commit 的 branch tree 未包含 repo-level `LICENSE`；默认分支的许可证不能未经确认自动外推到该快照。推荐 clean-room 重写机制或先向作者确认授权。

## 3. GentleHumanoid

### 两个仓库的职责

用户给出的 [`gentle-humanoid`](https://github.com/Axellwppr/gentle-humanoid) 是部署仓：预训练 ONNX/PT、MuJoCo sim2sim、Unitree runtime。训练实现位于其 README 链接的独立 [`gentle-humanoid-training`](https://github.com/Axellwppr/gentle-humanoid-training)。只审部署仓无法评估算法。

### 关键代码

- `MotionTrackingCommand_impedance` 选择左右 shoulder-yaw、wrist-roll、hand 共 6 个 force links：[`motion_tracking.py` L541-L562](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/active_adaptation/envs/mdp/commands/motion_tracking.py#L541-L562)。Tracking skeleton 是 11 点，不是 SONIC 的 14 点。
- force pattern 包含 zero、all、left、right、partial，可在六点上形成组合接触：mask/范围见 [`motion_tracking.py` L580-L622](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/active_adaptation/envs/mdp/commands/motion_tracking.py#L580-L622)，采样见同文件 [L762-L784](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/active_adaptation/envs/mdp/commands/motion_tracking.py#L762-L784)。`kp_left`/`kp_right` 分别广播到同侧三点，所以是逐点 mask、按侧共享 stiffness，不是六点独立 stiffness。
- 显式 `AdmittanceMassChain` 以 0.1 kg mass、2.0 damping、4 个 integration substeps 生成 compliant reference；外部 spring stiffness 采 5–250 N/m，safe threshold 采 5–15 N：[`admittance.py` L13-L73](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/active_adaptation/envs/mdp/commands/admittance.py#L13-L73)、[`motion_tracking.py` L830-L922](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/active_adaptation/envs/mdp/commands/motion_tracking.py#L830-L922)。
- reward 把受力 link 的原始 target 换成 compliant target，另加 force matching、force exceed penalty、位置/速度 tracking：[`G1_gentle.yaml` L109-L135](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/cfg/task/G1/G1_gentle.yaml#L109-L135)、[`motion_tracking.py` L1109-L1150](https://github.com/Axellwppr/gentle-humanoid-training/blob/9fd915d9639fc9b219e4b12b04076f67db82c7b5/active_adaptation/envs/mdp/commands/motion_tracking.py#L1109-L1150)。
- Actor 看 threshold、motion、IMU、joint/action history；真实 force 仅进 privileged critic。部署模型输入 450 维，仍输出固定 PD 的 position target：[`observation.py` L16-L98](https://github.com/Axellwppr/gentle-humanoid/blob/12aee8d84b9bf998c910b717eb9e3cbedd4d0559/src/observation.py#L16-L98)。

### 适配判断

这是五个候选中公开训练代码里 upper-chain simultaneous contact curriculum 最完整的：不只手腕，还通过肩/腕/手链路训练协调让位，且有 force/safety reward。这里的“完整”指采样覆盖，不代表论文验证了六点独立控制。IsaacLab 2.2 也比 mjlab 更接近 SONIC。

代价是它主动改 compliant reference 和 reward。[CHIP v2 Table I](https://arxiv.org/abs/2512.14689v2) 在同一篇论文组织的无外力 local 3-point 对照中报告 Gentle stiff/compliant 为 4/5 cm、0.15/0.16 rad，而 CHIP 为 2 cm、0.09–0.11 rad；这是 CHIP 对照，不应写成 Gentle 自己报告的 cross-paper 数值。[GentleHumanoid v1](https://arxiv.org/abs/2511.04679v1) 展示了手/肘/肩更低且稳定的接触力，但没有证明 SONIC robot-motion encoder 的 14-point whole-body accuracy。

部署接口只有 5–15 N threshold，没有一个保证精确恢复原 SONIC action 的 hard gate；六个 links 还共享一项安全阈值。部署和训练仓均未发现 repo-level `LICENSE`，所以不建议直接复制实现。

## 4. UniFP

### 它是什么

论文方法是 force/position command-conditioned learned WBC：公开 B2Z1 config 的 actor 用 32 帧 history，见 [`b2z1_pos_force_config.py` L102-L124](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/b2/b2z1_pos_force_config.py#L102-L124)；adaptation encoder/decoder 的监督标签包含 gripper/base force，见 [`actor_critic.py` L7-L16](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py#L7-L16) 和 [L43-L79](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py#L43-L79)。PPO 输出 joint-position targets，奖励把 endpoint position goal 按 `(F_external + F_command)/K` 偏移，代码入口见 [`legged_robot_b2z1_pos_force.py` L1886-L1900](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py#L1886-L1900)。这是一整个 learned policy，不是一个能插在 SONIC 后面的 classical controller。

### 公开代码与论文的差距

- 唯一 asset 是 B2Z1 quadruped + Z1 single arm；唯一 task implementation 也是 `b2z1_pos_force`。
- action 是 17 维，command 是 15 维，force sites 是一个 gripper 与 base：[`b2z1_pos_force_config.py` L102-L176](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/b2/b2z1_pos_force_config.py#L102-L176)。
- README 虽写支持 G1 等配置，但仓库没有 G1 asset/env；论文里的 G1 实验是 locomotion/base push，不是双臂或全身 14-point motion tracking。
- sim2real、MuJoCo sim2sim、imitation data pipeline 都仍在 TODO；无 checkpoint 与仓库测试。

[UniFP v2 Section 4.2 / Figure 4](https://arxiv.org/abs/2505.20829v2) 报告把该 policy 接入 imitation pipeline 后，在四个 real-world contact-rich tasks、每项 50 trials 上，相对 position-only 平均成功率提高约 39.5%。论文 abstract、Figure 4 和 Section 4.2 写 four tasks，但 introduction/contribution 有一处写 three，本文采用有逐项实验定义的 four-task 口径。该结果不能外推为 SONIC tracking precision 或多上肢接触能力。

仓库有 [BSD-3-Clause root `LICENSE`](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/LICENSE)，版权头署名 Unitree；`legged_gym/LICENSE` 另有 ETH/NVIDIA 版权，复用具体文件时仍要核对第三方来源。

## 5. SoftMimic

### 关键代码

- G1 config 明确为 29 DoF，并列出 14 个 keypoint body IDs：[`g1_force_control.py` L32-L147](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/softmimic_gym/softmimic_gym/tasks/locomotion/tracking/config/g1/g1_force_control.py#L32-L147)。但其 ordered body list 使用 hip-yaw/shoulder-yaw，见同文件 [L220-L236](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/softmimic_gym/softmimic_gym/tasks/locomotion/tracking/config/g1/g1_force_control.py#L220-L236)；SONIC 使用 hip-roll/shoulder-roll。二者只有数量相同，不能按 index 直接兼容。
- 先用 Mink IK 做 compliant motion augmentation：生成入口导入 `G1_Mink_IK_Solver`，见 [`mink_generator_ff.py` L15-L31](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/compliant_motion_augmentation/mink_generator_ff.py#L15-L31)；runner 打包 link/force/torque/stiffness/force-field metadata 并写 CSV，见 [`runner.py` L390-L457](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/compliant_motion_augmentation/runner.py#L390-L457) 和 [L540-L552](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/compliant_motion_augmentation/runner.py#L540-L552)。RL 主要跟踪预先求得的 feasible compliant reference，而不是在线改 sparse goal。
- Actor 不看 applied wrench，只看 reference/proprio/action histories 与 translational/rotational log stiffness；真实 force、torque、adapted keypoint error 只给 critic：[`g1_force_control.py` L245-L326](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/softmimic_gym/softmimic_gym/tasks/locomotion/tracking/config/g1/g1_force_control.py#L245-L326)。
- 输出是 29 维 joint-position target，不是 torque/stiffness matrix。
- release command 的 `_force_link_id` 为 `[E,F,1]`，每个 future frame 只有一个 link：[`compliance_augmented_reference_command.py` L165-L192](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/softmimic_gym/softmimic_gym/tasks/locomotion/tracking/mdp/commands/compliance_augmented_reference_command.py#L165-L192)。公开 G1 force observation 又只列 wrist IDs 36/37，所以“14 tracking keypoints”不等于“14 simultaneous compliance sites”。

### 适配判断

SoftMimic 的优点是 G1/29-DoF/14-point cardinality、asymmetric actor-critic、train/deploy pipeline 与 [MIT license](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/LICENSE) 都很清晰。[SoftMimic v1 Figure 8](https://arxiv.org/abs/2510.17792v1) 显示 IK augmentation 在低 stiffness 时把 displacement error 相对 no-augmentation ablation 降低约 50%，但无扰动 tracking 并非所有动作都与 stiff baseline 等精度：同文 Table I（paper p.6，36 episodes 的 mean ± SEM）中，Box Pick 为 `5.04°/2.65 cm` 对 `2.04°/1.36 cm`，Walk 为 `6.39°/3.44 cm` 对 `6.09°/3.50 cm`，Dance 为 `11.10°/6.05 cm` 对 `5.16°/3.01 cm`（joint/keypoint）。也就是 walking 接近基线，而 manipulation 与 dynamic dance 有明显回归。

但它需要先为大量 motion/contact/stiffness 组合跑离线 IK，再训练 policy；这正是 CHIP 想避免的工程量。公开 release 每时刻只支持 single force link，[SoftMimic v1 Conclusion](https://arxiv.org/abs/2510.17792v1) 也把 simultaneous multi-link augmented data 列为 future work；论文同时说明当前 policy 已泛化到 box picking 等部分 multi-contact case，因此准确结论是“有定性泛化、没有多 link augmentation 的保证”，而不是完全没有多接触行为。本次审计还发现 config 指向不存在的 `compliant_motion_augmentation/release_examples` 和 `g1_full_reduced_handcollisions.usd`，唯一 golden test 也读取 `release_examples`，见 [`g1_force_control.py` L29-L30、L65-L68](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/softmimic_gym/softmimic_gym/tasks/locomotion/tracking/config/g1/g1_force_control.py#L29-L68) 与 [`test_mink_generator.py` L39-L84](https://github.com/Improbable-AI/softmimic/blob/823582557d5417c9382303ba749262afab625076/tests/test_mink_generator.py#L39-L84)。部署 stiffness 还有硬编码路径，因此不能把 README 命令视为完整即跑复现。

适合的使用时机是：CHIP-style online training 无法稳定学到某类强约束接触响应时，针对少量关键技能生成 SoftMimic augmented clips，而不是首版覆盖全 motion library。

## 6. CHIP

### 方法与证据

CHIP 训练时向末端施加 0–40 N、随机 3D 方向、持续 1–3 s 的梯形扰动力，并把 actor 看到的 sparse goal 改成：

```text
goal_hindsight = goal_reference - compliance * force
```

critic 看真实 external force；actor 只看 modified goal、每个 end-effector 的连续 compliance、10-step proprio/action history。reward 始终使用原始 reference 的 link pose/orientation/velocity/joint state，不做 motion augmentation 或 reward retuning。部署时 actor 不看 force，靠历史隐式估计。

论文的两个实际接口都是 3-point：

- global：head 6D + 两个 wrist 3D，带 0.2 s future；
- local：head 与两个 wrist 的 SE(3)，另带 SONIC kinematic planner 的 lower-body q/qdot。

它支持双腕同时接触和左右不同 compliance，但没有验证 elbow、torso、knee、foot、未知接触点或同一手臂多个独立接触；compliance 是 position scalar，不是 3D/6D anisotropic matrix。附录还把 ankle/wrist 之外的接触默认视为 undesired contact。

[CHIP v2 Table I](https://arxiv.org/abs/2512.14689v2) 在 100 条 TWIST trajectories、无 external force 下报告：

| 设置 | local position | local orientation |
|---|---:|---:|
| CHIP，compliance 0 / 0.02 / 0.05 | 0.02 m | 0.09 / 0.10 / 0.11 rad |
| Gentle，stiff / compliant | 0.04 / 0.05 m | 0.15 / 0.16 rad |
| no-force baseline | 0.02 m | 0.08 rad |

这支持“无接触时保持 tracking”，但不能证明 contact 下仍是 2 cm，也不是 wrist-only 指标。[同文 Table II](https://arxiv.org/abs/2512.14689v2) 的双机器人抓取每个 setting 仅 5 次，CHIP 平均成功率 80%，FALCON 5%，no-perturbation 40%；属于小样本 task success，不是 stiffness/force tracking curve。

### 代码与 SONIC 接入限制

[官方项目页](https://nvlabs.github.io/CHIP/) 截至审计日仍写 `Code (Coming Soon)`；没有权重、训练配置或 checkpoint。论文也没有披露网络宽度、PPO 超参数、compliance sampling、damper 系数和 action 语义。全部 policy 训练报告为 4 天、64×L40S、每 GPU 4096 env；论文没有证明少量 finetune 即可得到相同性能。

它与 SONIC 的“兼容”特指 3-point teleop + lower-body planner，不是 released robot-motion encoder。对 full q/qdot 做 Cartesian `goal - cF` 维度不成立；用 IK 重写 q reference 又回到 SoftMimic 路线。保留 G1 token、增加 sparse contact residual adapter 是本项目的工程假设，不是 CHIP 论文公开并验证的网络结构；SONIC-specific post-FSQ injection 也不应被表述成 CHIP 方法本身。

本地 CHIP branch 目前已进一步完成单 motion/seed、300-frame 的 Phase-5 配对 trace：
记录了上肢末端 position/orientation、14-body local/global MPJPE、force、fall 与 paired
yield，并通过独立 ONNX parity；motion branch 也完成了官方模型 5+1 步微调和独立
action-residual 部署。100-step/5-step smoke 中的 N/N·m 仍是**注入到 simulator 的测试
wrench**，不是策略产生的接触力或 measured compliance；单条 300-frame trace 也没有
覆盖多 motion/seed、single-left/right/two-site、`Delta x/F` 单调性、settling 与
cross-coupling。完整 go/no-go 仍未完成。因此“CHIP 首选”目前仍是外部论文证据支持的
试验优先级，不是本项目性能结论。

## 7. 可复用与不可复用边界

可以 clean-room 借鉴：

- CHIP：原 dense reward + hindsight sparse input；
- motion branch：hard enable gate、threshold normalization、original-target regression；
- Gentle：多 upper-link force masks、net-wrench curriculum、force exceed metrics；
- SoftMimic：log-uniform stiffness、actor/critic 信息隔离、困难技能的离线 IK fallback；
- UniFP：若以后要加入显式 desired-force command，可参考 history-based force estimator，但不放进首版。

不应直接做：

- 替换 SONIC encoder 或加载任一候选 checkpoint；
- 把 14 tracking points 当作 14 个可独立受力点；
- 把 simulation threshold 宣称为真机硬安全上限；
- 把多机器人双腕示例宣称为任意全身多接触；
- 直接复制 repo-level license 不清楚的 motion/Gentle 实现。
