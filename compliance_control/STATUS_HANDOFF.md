# SONIC 柔顺控制实现状态与续接手册

更新时间：2026-07-27

这份文档是下次工作的总入口。它汇总两条隔离实验分支的实现、证据、未实现项、
迁移边界和续接顺序。任务尚未完成：两条分支都已通过 Phase 1–5，当前均为
**Phase 6 / IN_PROGRESS**。

## 一眼看清当前状态

| 分支 | 当前阶段 | 已完成 | 尚未完成 |
|---|---|---|---|
| `experiment/chip-compliance` | Phase 6 | portable core、SONIC/IsaacLab adapter、post-FSQ actor/value residual、官方 checkpoint 5+1 步微调、300 帧配对 rollout、独立 residual ONNX、Phase 1–5 验收 | 删除一个审计生成的精确 `__pycache__` 文件；重跑完整 structural/hygiene audit；在兼容 NVIDIA userspace 下完成 fresh NVML/idle-process gate |
| `experiment/motion-compliance` | Phase 6 | 显式 `[enable, threshold, Kp]`、多 site virtual wrench、独立 action/value residual、官方 checkpoint 5+1 步微调、独立 residual ONNX、Python/C++ runtime 与 SONIC production hook、tracker-neutral aligned evaluation | 重启后重新确认 GPU 环境；采集真实 paired baseline/off/no-contact/单左/单右/双腕 traces；跑 4096-env scheduler benchmark；按门槛生成最终指标并完成 Phase 6 |

结论没有变化：**CHIP 路线是优先验证的方法路线**，因为它最直接保护原始 dense
tracking 目标；`motion-compliance` 路线则是当前更完整的部署与操作接口参考。两条线
都是从官方 SONIC 基线 `4141c34280ab` 派生的 additive、opt-in 实验，并没有替换
release tracker。

## 已实现：CHIP 路线

分支工作树：`/tmp/gr00t_chip_compliance`

分支内权威续接文档：
`tasks/chip_compliance_finetune/phase6_handoff.md`

实现边界：

- `gear_sonic/compliance_control/core`：与 tracker/robot 无关的 structured Cartesian
  frame、hindsight target、schedule/damping、residual conditioning、aligned trace 和
  metrics；site/body 数量由调用方定义。
- `gear_sonic/compliance_control/postprocess`：受限、原子化、禁用 pickle 的 NPZ/JSON
  证据输出。
- `gear_sonic/compliance_control/adapters/sonic`：唯一持有 SONIC/G1 body 名称解析、
  IsaacLab wrench、checkpoint 迁移、policy/critic residual 和 ONNX 接线的层。
- 原 release encoder、decoder、994-D `g1_dyn`、1645-D critic 和 dense rewards 保持
  原形；新 residual 零初始化并按行 hard gate，off row 精确走 release 输出。
- 已用官方 `last.pt` 做 5 个 residual-only PPO batch，并严格 resume 到第 6 步；
  release policy/value tensors 保持 byte-exact，仅 12 个 residual tensors 可训练。
- Phase 5 已在一个固定 motion 上完成 300 帧 matched-force stiff/compliant 配对；
  这证明链路激活且门槛未回归，不证明收敛的接触柔顺、多物体泛化或 14-site 能力。
- residual 单独导出为动态 `(B,S)` ONNX，部署时在 release encoder latent 后、原
  decoder 前相加；hard-off 和 zero-compliance 精确为零。

已接受的关键证据：

- Phase-4 step-6 checkpoint SHA-256：
  `71bce134e7d2d5f83f5ad9a4576650c419a2d70bcc764a4e68480242dfc67c02`
- Phase-5 residual ONNX SHA-256：
  `a4ccbc9e216dd97fe5181a12f5ded7a9e544c1a477fd114c909b8564bc83e2f3`
- compliant upper-endpoint MPJPE `0.03208341 m`，orientation RMSE
  `0.19856526 rad`，paired displacement mean/max `0.00131442/0.00410081 m`
- 完整 portable/resolved regression：各 129 tests；real ORT parity 和独立 Phase-5
  metric audit 已通过。

## 已实现：motion-compliance 路线

分支工作树：`/tmp/gr00t_motion_compliance`

分支内权威续接文档：
`tasks/motion_compliance_finetune/phase6_handoff.md`

实现边界：

- `gear_sonic/compliance_control/core`：通用 enable/threshold/Kp contract、virtual
  force scheduling 与 residual 组合，不固定 G1 数值索引。
- `gear_sonic/compliance_control/adapters/sonic`：拥有 SONIC 的 994-D context、
  29-D BFS action、双腕 site 和 operator gate 语义。
- `gear_sonic/compliance_control/deployment` 与对应 C++ runtime：接受任意 ordered
  context segments 和宿主提供的 release artifact pins；portable 层不含
  SONIC/G1/IsaacLab/wrist 常数。
- 已用官方 checkpoint 做 5+1 步 residual-only 微调；release 网络保持同 shape、
  byte-exact，独立 residual 的幅值上限为 `0.25`。
- 已完成 Python/ORT 和 system-ORT C++ 动态 shape、mixed-row、NaN 隔离、hard-off、
  lazy disabled 与 release fallback 验收。
- production hook 位于 release action 产生后、IsaacLab BFS → MuJoCo DFS 重排前；
  host-off 不读取 residual artifact、不创建 ORT session。
- 已实现严格按 motion/sequence/seed/frame/timestamp 对齐的 tracker-neutral trace、
  metric 与 bounded atomic NPZ/JSON；它是 Phase-6 采集器的稳定目标接口。

已接受的关键证据：

- step-6 checkpoint SHA-256：
  `42dd92200da1e626436225414ddfa59ba2198953c304f25f217454f24fb84aba`
- residual ONNX SHA-256：
  `9e7a30ae8485eb153b63db81575c9b0fd24522523510560ed5d6292652568a81`
- metadata payload SHA-256：
  `e954d093603d910e8cde4c2a5842db4d734d1ec8fbc3180f03a9399b5c17d8c5`
- Phase 1–4 + evaluation：`101 passed, 1 skipped`；deployment：`33 passed`；
  production target、C++ ORT smoke 和 CLI invalid-input gate 已通过。

## 明确未实现或不能声称的能力

- 尚无覆盖多 motion、多 seed、不同物体/接触几何的性能结论。
- 尚无满足最终门槛的真实 Phase-6 左腕、右腕、双腕同时接触数据；不能用当前短
  smoke 声称“上肢末端精度已优于基线”。
- 当前训练的 checkpoint 固定为双腕 interaction sites。底层 tensor/evaluator 可变
  site 数不等于同一个 checkpoint 已学会 14 点或任意点接触。
- 当前实现只有平移 compliance，没有独立 rotational compliance。
- motion production overlay 的前两个 wrist controls 通过 OR 形成一个全局 binary
  gate；不是两个手腕可独立开关的部署 UI。训练/evaluation tensor 仍区分 sites。
- force threshold、resultant wrench clamp 和 residual bound 是工程限制，不是经过
  形式化证明或硬件认证的安全力控器。
- 这两条路线实现的是固定低层 PD 上的 learned apparent/task-space compliance，
  不是具有稳定性证明的经典 impedance/admittance controller。

## 可迁移架构：以后换 universal tracker 时保留什么

稳定中间表示和 portable 层应原样迁移：

1. 有序 `site_names`、frame metadata、reference/current/force/compliance tensors。
2. actor 只读可部署 condition；真实 applied force 只进 critic/evaluator。
3. hard-off 在 residual 组合边界完成，禁止让 adapter 或网络“近似 off”。
4. aligned trace 由调用方提供 motion/sequence/seed/frame/timestamp 主键。
5. residual artifact 独立版本化；release encoder/decoder 由宿主按名称和 SHA-256
   pin，不复制进 portable package。

每个新 tracker 只新增薄 adapter：解析 reference index 与 articulation index、转换到
统一 Cartesian frame、组装该 tracker 的 actor context/action layout、把 residual
插入选定边界并输出标准 trace。不要复制 SONIC 的 14-body/29-DoF/994-D 常数；尤其
要显式核对目标仓库的 BFS/DFS DOF 顺序和坐标系。

## 下次严格续接顺序

### CHIP

1. 进入 `/tmp/gr00t_chip_compliance`，先读 `status.md`、`test_matrix.md` 和分支
   `phase6_handoff.md`，确认仍是 Phase 6。
2. 只删除文档中列出的那个精确 audit bytecode cache；不要清理整个目录或改动两个
   accepted artifact roots。
3. 用 `PYTHONDONTWRITEBYTECODE=1` 重跑 Phase-6 structural/hygiene audit。
4. 按 `test_matrix.md` item 4 的原样命令，在已验证的 NVIDIA 580.159 compatibility
   userspace 下重跑 fresh NVML/idle-process gate；必须出现
   `CHIP_PHASE6_FINAL_AUDIT_PASS`。
5. 重跑该 phase 的最终 diff/cache 检查，通过后才把 Phase 6 标为 `PASSED`、任务标为
   `COMPLETE`。

### motion-compliance

1. 进入 `/tmp/gr00t_motion_compliance`，先读 `status.md`、`test_matrix.md` 和分支
   `phase6_handoff.md`。
2. 机器重启后先按矩阵验证 `sonic_backup`、compatibility CUDA、官方 asset hashes 和
   无残留 simulator/trainer process；不要直接复用旧 GPU 结论。
3. 先跑独立 4096-env scheduler-only benchmark，记录 host-off/enabled CUDA event
   time 与 allocated/reserved peak；它不能冒充 end-to-end policy latency。
4. 用同一 motion/sequence/seed/frame/timestamp 采集 baseline、overlay-off、
   enabled/no-contact、single-left、single-right、simultaneous traces；每个 sequence
   都必须有 reset snapshot 和 terminal row。
5. 用 tracker-neutral evaluator 生成逐 site endpoint position/orientation、MPJPE、
   force/yield/cross-coupling、success/fall/reset/finiteness 指标；上肢左右手必须分别
   报告，不能只报全身均值。
6. 只有 `test_matrix.md` Phase-6 全部通过后才能修改 `status.md` 为 COMPLETE。

## Git 与资产边界

- 已发布远端代码边界：
  `origin/experiment/chip-compliance@035a68cae6d5bb35319b23ccf73a65a337ae19ee`
  和
  `origin/experiment/motion-compliance@b773e0b15842924e3b0b74e2eef7f37bb0df52fe`。
  推送前后核验显示已有 `origin/main` 始终为
  `345c3f442b2d33e7eb784afd2f5d7c17066d794e`；本次仅新增上述两个远端 refs。
- 受保护的本地 `main` 不提交、不推送；中央 `compliance_control/` 目前仍是 main
  上的 untracked 工作目录，因此本文件是本地总索引。
- 两条实验分支才是远端可复现代码边界；分支内各自包含 task plan/status/matrix/log
  和 handoff。
- 官方 checkpoint、SMPL/robot sample、训练 runs、ONNX/binary evidence 位于被忽略
  的中央 artifact 目录，不进入 Git。来源、revision 和 SHA-256 见
  `official_assets/MANIFEST.md`。
- `phase4_acceptance_resume_fix` 与 `phase5_acceptance` 是已接受的不可变 CHIP
  evidence roots；不要覆盖、移动或删除。

相关背景和详细设计见 [README](README.md)、[evidence](evidence.md)、
[design](design.md)、[PORTING](PORTING.md) 和
[implementation_branches](implementation_branches.md)。
