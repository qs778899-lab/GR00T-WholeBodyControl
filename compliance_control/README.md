# SONIC 全身跟踪柔顺控制选型

审计日期：2026-07-27。这里分析的是 SONIC released robot-motion encoder 链路，并以“可关闭、尽量少改动、优先保持跟踪精度，尤其是上肢末端精度”为决策目标。下述选型是**论文与固定代码快照驱动的待验证推荐**；本项目当前的 contract、wrench 和短训练 smoke 不能替代 tracking/compliance benchmark。

## 结论

最合适的不是直接换成某一个候选仓库，而是：

> **以 CHIP 的 hindsight perturbation 作为训练原则，在现有 SONIC 的 FSQ 后、decoder 前增加零初始化、硬门控的 residual compliance adapter；用 `motion_tracking/compliance` 的显式开关与扰动实现做首版工程参考，用 GentleHumanoid 的多上肢 link 接触采样和安全指标做第二阶段扩展。**

如果必须只选一个名字：

- **方法首选：CHIP**。在这些候选中，它明确把原始 dense tracking reward 保持不变；论文的无外力 local 3-point 聚合误差约为 2 cm，优于同一篇论文对 GentleHumanoid 报告的 4–5 cm。这支持优先试验 CHIP 原则，但不是 SONIC released robot-motion encoder、wrist-only 或接触中精度的本地验证。
- **最快 PoC 的代码参考：`motion_tracking/compliance`**。它已经有 `[enable, force_limit, Kp]` 条件和部署开关，但不能加载 SONIC checkpoint，也没有任意多点接触或论文级量化。
- **多上肢接触的训练参考：GentleHumanoid training**。它训练肩、腕、手共 6 个 virtual-force links 的组合接触，但主要按左右侧共享 stiffness，并通过修改 compliant reference/reward 获得柔顺，不能视为 6 点完全独立控制。
- **SoftMimic 只作为第二方案**。它是 G1/29-DoF 且同为 14 个 tracking keypoints，MIT 许可清晰，但这 14 点的 body 语义和顺序与 SONIC 不同；它还依赖离线 Mink IK motion augmentation，公开实现每个时刻只有一个 force link，不符合“减小工程量”和多点同时接触的首版目标。
- **不选 UniFP 作为底座**。它不是可插拔柔顺控制器；公开代码只有 B2Z1 单臂 loco-manipulation policy，没有 G1 motion tracker、checkpoint 或部署闭环。

## 一个容易混淆但决定架构的事实

SONIC release 的 G1 encoder **并不直接输入 14 个 keypoints**。实际输入是 10 个 future frames 的 29-DoF `q/qdot` 与 6D anchor orientation，即每帧 64 维、原始输入共 640 维，再编码为 64 维 token state。14 个 `body_names` 是 motion command、dense reward 与 evaluation skeleton。

因此不能把 CHIP 的 Cartesian 公式 `goal_hindsight = goal - compliance * force` 直接套到 29-DoF joint tensor，也不应把 14 个 skeleton 点全部视为独立接触点。推荐保留原 G1 encoder，另加稀疏 contact adapter。

## 候选对比

这里把“改出接口”和“达到论文性能”拆开；后者没有统一硬件/数据集，不能直接横向换算。

| 候选 | 公开精度证据 | 与 SONIC/off-mode 的关系 | 公开多点边界 | 机制重写量 | 达到论文性能的成本/置信度 | 定位 |
|---|---|---|---|---|---|---|
| CHIP | **最强但有限**：无外力 local 3-point 聚合 2 cm；不是 wrist-only，也不是接触中指标 | 无代码可直接复用；本项目另设计 gated residual | 双腕可独立/同时；未验证任意全身点 | 中 | **高且不确定**：无代码/权重/完整超参；论文 64×L40S、4 天 | **推荐训练原则** |
| motion_tracking/compliance | 无公开 benchmark | 网络不兼容；有明确 flag，但 SONIC 重写后仍须训练后 off regression | 4 个 hand/wrist links 可同时施力，但同一 global condition、耦合修改 | 低到中 | 高：原仓完整流程约 4×A100、15 小时 | **首版代码参考** |
| GentleHumanoid | CHIP Table I 对照为 4–5 cm | 网络不兼容；只有 threshold，需补 hard gate | 6 个上肢 links 有组合 mask，stiffness 主要按左右侧共享 | 中 | 中高；训练代码公开但 license 不明 | 多 link curriculum 参考 |
| SoftMimic | walk 接近基线，box/dance 明显回归 | 需重做 augmentation/训练接口 | release 每时刻单 link；论文仅报告部分多接触泛化 | 高 | 高：每个 motion/contact/stiffness 需离线 IK | 离线 IK fallback |
| UniFP | 不报告 humanoid motion-tracking 精度 | 整体 policy/embodiment 不兼容 | 一个 gripper 与 base | 很高 | 高且不可外推：公开 G1 pipeline 缺失 | 不采用 |

“多点”还分为张量支持、同一时刻施力、逐点独立命令和训练后协调四级。当前 branch 的 1/2/5/14/17-site contract tests 只证明 shape；100-step force smoke 只证明 writer/lifecycle，不证明同一 checkpoint 支持可变 site 数，也不证明多接触下的策略性能。

“柔顺”在这些工作中主要是固定低层 PD 之上的 **learned apparent/task-space compliance**。它们都不是有形式化稳定性保证的经典阻抗控制器；force limit 也不是硬件安全认证的接触力上界。

## 文档与复现

- [当前阶段与续接手册](STATUS_HANDOFF.md)：两条实验分支已实现/未实现清单、证据边界、迁移原则和下一次严格执行顺序。
- [代码与论文证据](evidence.md)：固定 commit、关键代码入口、输入输出、定量结果、许可和公开实现缺口。
- [推荐接入设计](design.md)：adapter 接入点、14-point/interaction-site 分层、训练阶段和验收门槛。
- [迁移指南](PORTING.md)：稳定中间表示、目标 tracker adapter 职责、不可复用的 SONIC 常数与最小验收矩阵。
- [实现分支登记](implementation_branches.md)：官方基线、两条隔离 worktree、阶段提交、官方资产与 GPU smoke 环境。
- [当前仓库契约测试](tests/test_sonic_contracts.py)：验证 G1 encoder 输入、14 点 skeleton、release 部署接口、BFS/DFS permutation，以及 dormant CHIP 张量错误没有被误当成可用链路。

运行静态审计：

```bash
python -m unittest discover -s compliance_control/tests -v
```

候选外部仓库只在 `/tmp` 中做只读审计；实际实现位于两条从 NVLabs 官方固定
提交派生的独立 worktree。官方 checkpoint/样例放在 Git 忽略的
`official_assets/`，其来源与校验信息见该目录的 `MANIFEST.md`，不复制进实现分支。

## 当前实现结论（不是性能结论）

- `experiment/chip-compliance` 已完成纯 tensor core、SONIC/IsaacLab adapter、
  post-FSQ hard-gated residual、官方 checkpoint 的 5+1 步 residual-only 微调、
  配对 rollout 与独立 ONNX 导出。短 rollout 只证明链路激活且数值有限；它没有
  覆盖足够 motion/seed，不能替代上肢末端性能门槛。
- `experiment/motion-compliance` 已完成显式 `[enable, threshold, Kp]`、多 site
  virtual wrench、独立 action/value residual、官方 checkpoint 的 5+1 步微调以及
  Python/C++ 独立 residual 部署验收。portable runtime 不含 SONIC/G1/IsaacLab
  常数，release encoder/decoder/config 由宿主按名称和 SHA-256 外部绑定；具体
  994/29 维、BFS 顺序和双腕 operator gate 只在 SONIC adapter。
- 两条分支都已真实启动官方模型微调，但目前只有低资源工程 smoke。完整 Phase 6
  仍需同 motion/frame/timestamp 的 stiff/off、on/no-contact、单左、单右和双点配对
  回归，尤其要分别报告左右手 position/orientation，不能用全身均值掩盖退化。
  motion 分支已实现这套 tracker-neutral trace/metric CPU 合同，但尚未采集所需真实
  simulator traces。

`design.md` 描述推荐目标架构；实际分支产品、checkpoint、artifact 哈希和每一级
证据边界以 [实现分支登记](implementation_branches.md) 为准。外部参考仓库
`motion_tracking/compliance` 与本地 `experiment/motion-compliance` 是两个对象，
后者是面向 SONIC、按本文合同 clean-room 实现的实验分支。

当前中央 `compliance_control/` 在受保护的本地 `main` 上仍是 untracked 交付目录；
本文没有为此移动或提交 `main`。正式归档时应新建独立 docs/artifact 分支并固定整目录
manifest，而不是把当前路径误称为已有 Git provenance。两条实验实现分支及其已列出
的阶段提交不受这一中央文档状态影响。
