# 柔顺控制实现分支登记

更新日期：2026-07-27。

## 分支隔离

两条实验分支都直接以 NVlabs 官方仓库 `main` 的固定提交为父提交：

- 官方来源：`https://github.com/NVlabs/GR00T-WholeBodyControl.git`
- 官方提交：`4141c34280abb67c82e115342a8720f4a83d750d`
- CHIP：`experiment/chip-compliance`，worktree `/tmp/gr00t_chip_compliance`
- motion_tracking/compliance：`experiment/motion-compliance`，worktree `/tmp/gr00t_motion_compliance`

创建前的本地分支为 `main`、`backup-before-reset-0506`、`s0_git`、
`s0_git_0507`。实现过程中不 checkout、reset、rebase 或提交到这些分支。
两个实验分支只允许新增自己的提交；如后续发布远程分支，只允许创建同名
新 ref，禁止 force-push 或覆盖已有远程 ref。

## 阶段回溯点

本节把证据分为四级：纯 tensor/配置合同、真实 simulator writer/lifecycle smoke、
短训练启动、tracking/compliance 性能。下列 `N`/`N·m` 峰值都是测试模块**注入到
simulator 的 external wrench**，不是机器人实测接触力、策略 force tracking 或硬件
安全上限；`100 steps` 也只验证 writer/frame/reset。除非条目明确列出
wrist/14-point RMSE、orientation、`Delta x/F`、settling、cross-coupling 和 fall rate，
否则不得把它引用成跟踪或柔顺性能证据。目前没有任何条目达到第四级。

- CHIP Phase 1：`981d838`（portable schema/math/schedule/metrics、结构化 frame、
  双 index space、target damper 与 23 项 CPU 合同测试）。
- CHIP Phase 2：`256c61b`（name-resolved SONIC/IsaacLab adapter、CHIP 在线
  force/compliance pulse、非变异 hindsight target、link-local wrench 与 opt-in
  Hydra composition）。完整 CUDA/Hydra 合同为 45/45；RTX 4090 disabled/enabled
  各 100 步通过，峰值分别为 `0 N / 0 N·m` 与
  `8.457119 N / 2.564395 N·m`，并覆盖 active→off 只清一次及 reset 清零。
- CHIP Phase 2 调度/同步回归修复：`04b15ce`。移除新增 interval event 和
  `CommandTerm` 的全局 RNG/dynamic due-index 路径，改为 command 私有 generator、
  固定全环境 candidate、布尔 mask 和预校验 writer；运行时关闭在 setter 返回前
  只清本模块拥有的 composer rows。最终 `sonic_backup` 为 65/65；4096 环境×14
  site 的 portable CUDA profiler 与真实 AppLauncher `SonicComplianceCommand`
  bound profiler 均未出现 `aten::nonzero`/`aten::_local_scalar_dense`。独立真实
  disabled 100 步为 `0 N / 0 N·m`（18.174 s），enabled 100 步峰值为
  `6.785363 N / 2.197217 N·m`，并验证全局 RNG 不变、body-local wrench/offset
  torque 重构和下一 physics step 前即时清零。
- CHIP Phase 3：`f42f1c8`。在不改 release tokenizer/decoder 的前提下，
  增加 post-FSQ 64D 零初始化 actor residual 和独立 privileged critic
  residual；actor 硬拒绝 `compliance_force`，只读入 target/command，critic 才
  可读入实际 force。分别构建的 site/future 组合
  `(1,1)/(2,10)/(5,3)/(14,4)/(17,7)` 均通过。最终 CPU/Hydra/official 套件为
  79/79；真实 Isaac +
  官方 checkpoint smoke 中 policy/critic/tokenizer/target/command/force 宽度为
  `930/1645/1761/60/9/6`，action/value shape 为 `(1,1,29)/(1,1,1)`，
  初始化/forward smoke 的 hard-off action 和冻结 official std 均为逐字节不变。
  不同 site/future 组合是分别构建模型的合同测试；同一 checkpoint 的 cardinality
  仍固定，也没有证明多点接触协调。
- motion_tracking/compliance Phase 1：`fa39575`（portable condition/reference/
  virtual-force core、完整 nominal + tracking force 语义与 23 项 CPU 合同测试）。
- motion_tracking/compliance Phase 2：`6191ed7`（可迁移的 SONIC/IsaacLab
  adapter、按配置 name 解析的交互 sites、运行时 hard-off、opt-in Hydra 配置与
  真实模拟器 smoke）。
  纯 Torch/adapter 测试共 48 项通过；RTX 4090 单环境分别运行 disabled/enabled
  100 步，关闭态 composer 从未激活且峰值为 `0 N`，开启态 site/composer 峰值均为
  `8.3204 N`，reset 后归零。
- motion_tracking/compliance Phase 3：`586d6e5`（追加 actor-safe 3D 条件、
  构建时 site 数量可配置但 checkpoint 内固定的 critic 特权量、future-0 柔顺端点
  奖励与原参考姿态奖励；保持
  release tokenizer、termination 和全部 dense reward 合同）。54 项纯测试通过；
  真实 RTX 4090 环境解析为 policy/critic `933/1657`，未训练组合 smoke 中关闭态
  新增奖励与总奖励影响均为精确零，并验证 reward-before-command-update 时序下
  不读取旧缓存；这不是训练后 policy action 与 official release 的等价证明。
- motion_tracking/compliance Phase 2/3 回归修复：`26638c2`。command 使用私有
  generator 采样 duration，关闭态 reset/compute 不消耗全局 CPU/CUDA RNG；移除
  零周期 apply event 后，opt-in 与 release 的 interval event 集合/周期完全相同。
  运行时 `enabled -> disabled` 在 setter 返回前只对 compliance-owned body rows
  定向写零，不让陈旧 PhysX wrench 多作用一个 policy step，也不覆盖其他模块的
  composer rows。最终为 56 项纯测试通过（普通解释器仅 CUDA case 跳过）；两次
  独立真实 GPU 复验分别通过 100+100 步 force smoke 与 `933/1657` 训练组合 smoke。
- motion_tracking/compliance CUDA 同步回归修复：`a6b70e2`。覆盖继承的
  `compute(dt)`，用固定 shape 全环境采样和 mask 更新代替 dynamic due IDs；
  4096×14 profiler 以及真实绑定命令的 Torch dispatch/CPU+CUDA profiler 均拒绝
  `aten::nonzero` 与 `aten::_local_scalar_dense`。最终纯测试 64 项通过；真实
  disabled 100 步为 `0 N`，forced 100 步 site/composer 峰值分别为
  `8.320412 N`/`8.320410 N`，Phase 3 真实组合仍保持 policy/critic
  `933/1657` 与关闭态新增奖励精确为零。
- motion_tracking/compliance 旧 Phase 4：`58257f0` + `7dbd0cf`。将官方
  step-41550 checkpoint 迁移为 actor `994→997`、critic/RMS `1645→1657`，
  encoder、FSQ、`g1_kin`、noise 保持字节不变。RTX 4090 上 16 环境完成
  5 步/1920
  timesteps，速度 `159..233 FPS`，峰值 CUDA allocation `543227392 B`，
  两 site 曝光各 `80/80`，力峰值 `14.977984 N`；独立目录 strict
  resume 至 step 6 也通过。step-5/6 均证明新 actor `3/3`、critic `12/12`
  列非零、41 个冻结 tensor 逐字节精确、optimizer 完整且有限。
  后续审查锁定 release noise clamp 配置，改用非原地 clamp 和 effective
  std 监控，通用 trainer 保持未改。
  **该实现已于最终独立审查中判定为不合格，不是最终训练方案。**
  原因是 optimizer 实际持有整个 `g1_dyn`，不只是新增三列；旧列和
  后续层可在训练中漂移，而旧 audit 又整体排除了该 decoder，因此不能
  证明训练后 hard-off 精确回到 release action。任务已重开 Phase 3，将改为
  完整冻结 release decoder 的独立零初始化、flag 硬门控 condition residual；
  上述 checkpoint 与训练指标只作为失败实验记录，不可用于导出或验收。
- motion_tracking/compliance 重构后 Phase 3：`4599fc9`。发布 actor/critic
  observation 继续为 `930/1645`，`g1_dyn` 与 critic/RMS 继续为
  `994/1645`；公开 condition 和 privileged site state 改为独立 `3/9`
  observation groups。release encoder/FSQ/decoder/value/noise/RMS 全部冻结，
  只增加零初始化 action/value residual，用 `torch.where` 按样本硬选择，
  使训练后的 off row 仍结构性地返回 release 路径。纯测试为
  `68 passed, 1 skipped`；官方 checkpoint CPU 审计和 RTX 4090 上的
  Phase-2 `100+100` 回归、Phase-3 真实模型 smoke 均通过。smoke
  证明 official tensor 字节不变、mixed off/on 门控、privileged 注毒隔离、
  aux/external-token 路径和 residual-only 梯度；它仍不是 tracking 性能证据。

- CHIP Phase 4：`ee66708`。官方 step-41550 checkpoint 的 55 个 policy 与
  17 个 value tensors 全部逐字节冻结，只训练 6 个 actor residual 与 6 个
  critic residual tensors。RTX 4090、16 环境的 fresh 5-step 与 strict resume
  step 6 均正常结束；step-6 checkpoint SHA 为
  `71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02`，
  12 个 optimizer slots 均存在且 step 为 120。三个训练阶段耗时约
  `23.27/24.70/17.53 s`。这只证明官方权重加载、residual-only 更新与严格恢复，
  不证明已学得目标柔顺性。
- CHIP Phase 5：`c925a0d`。增加 tracker-neutral 的严格对齐 trace/metrics 与
  有界 NPZ/JSON I/O，SONIC rollout 只做 name/frame/checkpoint 适配，并单独导出
  post-FSQ latent residual ONNX。300-frame 配对链路通过：上肢末端
  stiff/compliant MPJPE 为 `0.0323815/0.0320834 m`，compliant orientation RMSE
  为 `0.198565 rad`，平均 paired yield 为 `0.00131442 m`，注入峰值为 `5 N`；
  ONNX SHA 为
  `a4ccbc9e216dd97fe5181a12f5ded7a9e544c1a477fd114c909b8564bc83e2f3`，
  ORT 最大误差 `5.82e-10`。这是单 motion/seed 的 chain-activation evidence，
  不是 CHIP 论文性能复现，也不能把 3.2 cm 宣称为最终上肢精度。Phase 6 当前
  未提交且 `IN_PROGRESS`；还缺当前 compatibility-library 环境下的 fresh NVML gate，
  该 gate 本身不要求重启机器。
- motion_tracking/compliance 重构后 Phase 4：`108d228`。发布 actor/critic/RMS
  保持 `994/1645/1645`，condition `3` 与 privileged site state `9` 只进入两条
  独立 residual contexts `997/1657`；release encoder、FSQ、decoder、critic、RMS、
  quantizer 与 noise 全冻结。RTX 4090、16 环境 fresh 5-step 与 strict resume
  step 6 正常结束，step-6 SHA 为
  `42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba`；
  55+17 official tensors 全部逐字节一致，12 个 residual tensors 与 optimizer
  moments 均发生有限非零更新。fresh run 为 1920 simulator timesteps，FPS
  `176..250`，process peak CUDA allocation `353315840 B`，两 site 各有 `80/80`
  非零 force exposure。它证明微调链路正常启动和更新，不是跟踪性能结论。
- motion_tracking/compliance Phase 5（截至本页更新时间尚未形成 Git 提交）：
  worktree 在 `108d228` 后完成独立 action-residual ONNX、tracker-neutral
  Python/C++ runtime 与 SONIC 薄 adapter。accepted graph 输入为 release context
  `[B,S,994]` 与 condition `[B,S,3]`，输出 `[B,S,29]`，只含 6 个 residual
  initializers；ONNX SHA 为
  `9e7a30ae8485eb153b63db81575c9b0fd24522523510560ed5d6292652568a81`，
  metadata payload SHA 为
  `e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5`。
  Python PT/ORT 最大误差 `7.45e-9`，C++ system-ORT dynamic/mixed/off smoke 与完整
  `g1_deploy_onnx_ref` link/CLI gate 均通过。runtime 现阶段把左右腕 control 做 OR
  得到一个 global binary residual gate；它支持双腕同时训练施力，但**不等于**部署时
  可逐腕独立调 compliance。该状态已通过 Phase 5 测试，但 Git 写审批因本会话平台
  额度耗尽而被拒绝，不能引用一个不存在的 commit hash；Phase 6 为 `IN_PROGRESS`。
- motion_tracking/compliance Phase 6 CPU contract（未提交）：新增不依赖 tracker 的
  aligned-trace schema/metrics/bounded NPZ+JSON I/O，严格匹配
  motion/sequence/seed/frame/timestamp，要求 baseline、off、on/no-contact、每个目标
  endpoint 的 single-site 和 simultaneous multi-site trials，并逐 site 报告 endpoint
  RMSE/P95、orientation、local/global MPJPE、active force/yield、inactive
  cross-coupling、fall/reset/finiteness。Phase-1..4 + evaluation 为
  `101 passed, 1 skipped`，部署仍为 `33 passed`。这只是 CPU 合同；真实 paired
  simulator traces 与 4096-env CUDA benchmark 尚未运行。

上述已有 commit 都只存在于各自的新实验分支；明确标成“尚未提交”的 worktree
状态不在此列。两条分支的共同基线仍是上面的 NVlabs 官方固定提交。两条分支在
同一个 `gear_sonic/compliance_control/` 路径下使用不同
core API，是互斥备选，不应直接 merge 或混合引用文件；跨 tracker 迁移前先固定
分支和 commit，接口边界见 `PORTING.md`。

IsaacLab 5.1 的 `WrenchComposer` 会缓存首次 `set` 时的 link pose，不能在运动
过程中持续用 `is_global=True` 写世界作用点，否则力与力矩会基于陈旧姿态转换。
Phase 2 因此在 adapter 边界每步用当前 body quaternion 把已经完成世界系限幅的
wrench 转为 body-local，并以配置的 local site offset、`is_global=False` 写入。
portable core 不依赖这个 IsaacLab 细节，迁移到其他 tracker 时只需替换薄 adapter。

## 官方微调资产

两条 worktree 共享只读目录 `compliance_control/official_assets`。资产由官方
`download_from_hf.py` 下载，不进入 Git：

```bash
python download_from_hf.py --training --no-smpl \
  --output-dir /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets
python download_from_hf.py --sample \
  --output-dir /home/lab/Desktop/GR00T-WholeBodyControl/compliance_control/official_assets
```

当前包含默认 SONIC PyTorch checkpoint、匹配配置以及一组 walking 样例的
robot/SMPL/SOMA motion PKL。checkpoint 对应 Hugging Face revision
`7c90a56cfe04788c4f041daeef5b1e12930675ad`，已在 CPU 上通过官方 TRL
compatibility shim 解包，训练 step 为 41,550，包含 55 个 policy tensors。

最终 smoke 使用 `/home/lab/miniconda3/envs/sonic_backup/bin/python`；该环境已
安装 IsaacLab、Isaac Sim、TRL 0.28 和 `smpl_sim`。robot 输入必须指定上述官方
sample 的单个 PKL，避免把当前仓库其他历史样例混入 motion-key 配对。

## 宿主机临时会话状态（非性能复现证据）

当前 NVIDIA 内核模块为 `580.159.03`，用户态库为 `580.173.02`；CUDA 返回
error 804。为避免重启活跃的 GNOME/VS Code/ToDesk 会话，已把 NVIDIA CUDA
仓库的 `580.159.03-1ubuntu1` compute、GL/Vulkan 及其用户态依赖
**仅下载并解压**到：

```text
/tmp/nvidia_580_159_compat/extracted/usr/lib/x86_64-linux-gnu
```

系统没有安装、降级或改写任何驱动包。以该目录临时设置
`LD_LIBRARY_PATH`/`LD_PRELOAD`，并令 `VK_ICD_FILENAMES` 指向解压出的
`nvidia_icd.json` 后，`nvidia-smi`、Torch CUDA 张量和 Isaac Sim 5.1
`AppLauncher(headless=True)` smoke 均已在 RTX 4090 上通过。后续本次会话的
IsaacLab/GPU 测试使用同一临时环境；机器重启并加载磁盘上的 `580.173.02`
模块后应移除该临时兼容设置。这段记录只解释本次会话为何能启动 GPU，不提供
可持久复现环境，也不能作为 tracking/compliance 或吞吐基线。两条分支已经分别
实际运行官方 checkpoint 的 GPU 5-step + strict-resume smoke，因此“微调链路正常
开始”已有真实证据；这仍不是 tracking/compliance 性能验收。后者必须另附命令、
commit、config、motion/seed、原始日志和汇总 artifact。

## `existing_refs_before.txt` 的语义

该文件是实验开始时选定的**受保护 ref 不可移动集合**，不是 NVlabs 源码基线，也
不是完整 `git show-ref` 快照。源码基线单独固定为 `4141c342...`；文件中的
`main@345c3f...` 仅表示本地受保护主分支。筛选时故意不纳入新建 experiment refs、
`nvlabs/main` 和 tags，后续审计逐行验证已有 `<ref> <sha>` 未移动，但允许新增 ref。
不要重生成该文件或补入当前 experiment refs；现有严格 parser 也不接受文件内注释。
