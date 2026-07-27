# 推荐接入设计

本页是方法级推荐，不是两条实验分支当前文件树的统一规格。CHIP 分支采用了这里的
post-FSQ latent residual；motion 分支经审计后采用冻结 release decoder 的独立
action/value residual，并在 Phase 5 形成更通用的 action-residual deployment
contract。两者都满足 hard-off 原则，但 topology、checkpoint 和 artifact schema
互不兼容；实际状态以 `implementation_branches.md` 为准。

## 1. 设计目标

必须同时满足：

1. `compliance_enable=0` 时严格回退 released SONIC，不靠“大 stiffness 近似关闭”；
2. 保留 G1 robot-motion encoder、FSQ token、29-DoF action contract；
3. 无接触时优先保持双腕位置与姿态精度；
4. actor 不依赖理想仿真 force，部署首版沿用现有 proprio/action history；
5. 数据结构允许多个 interaction sites 同时激活，但不把 14-point tracking skeleton 和碰撞点混为一谈；
6. 新逻辑放进独立模块，训练、部署、评估边界清楚。
7. 柔顺核心不依赖 SONIC、IsaacLab、G1 或固定 14-point 顺序，可被其他
   universal tracker 通过薄 adapter 复用。

这里实现的是 learned apparent compliance，不承诺经典 impedance controller 的 passivity，也不把训练 force threshold 当作硬件安全保证。

### 1.1 可迁移性合同

实现必须分成两层：

- `compliance_core`：纯 Python/PyTorch 的 schema、单位和 frame 校验、扰动时序、
  hindsight/reference 数学、hard gate 与指标。禁止 import IsaacLab，禁止保存裸
  rigid-body index，禁止假定 29 DoF、14 tracking points 或固定 token dimension。
- `sonic_adapter`：通过 body name 解析 site，把 manager command/event/observation
  转成 core 的标准张量，并把 residual 接到 SONIC token/checkpoint/部署接口。

core 的批量张量合同使用具名维度语义，例如 `[..., site, xyz]`、
`[..., site]` mask；tracker adapter 负责展平。迁移到另一 universal tracker 时，
至少要重写 site/frame resolver、observation、policy injection、checkpoint 和 writer
lifecycle adapter；是否已有 quantized latent 决定 injection 位置。单元测试必须能在
没有 IsaacLab/Isaac Sim 的普通 Python 进程中运行。

当前 CHIP 与 motion 两条隔离分支在同名 `compliance_control/core/` 下实现了不同 API，
属于互斥实验基线，不应同时复制或直接 merge。这里描述的是期望的 method-neutral
边界；在真正抽出共同 schema/version 前，“两条分支都有 core”不等于它们 API 兼容。
另外，当前 MLP 会把 site/future 维展平，所以 site 有序列表和 cardinality 固定进
checkpoint；1/2/5/14/17-site 测试只是分别实例化后的 shape contract，不是一个
checkpoint 可在运行时任意增删 site，也没有 site-permutation equivariance。

## 2. 最小改动接入点

本仓已经有一条适合复用的接口：[`UniversalTokenModule.forward`](../gear_sonic/trl/modules/universal_token_modules.py) L741-L880 接收 64 维 `latent_residual`，支持在 FSQ 后把连续 residual 加到两个 32D tokens；[`manager_env_wrapper.py`](../gear_sonic/envs/wrapper/manager_env_wrapper.py) L93-L109、L698-L712 已有 residual mode 和 scale；[`token_losses.py`](../gear_sonic/trl/losses/token_losses.py) L1063-L1095 还有 residual L1/L2 regularizer。

因此首版不应改 G1 encoder input dimension，也无需另造 token pipeline：

```text
10-frame q/qdot/orientation
          │
          ▼
 frozen SONIC G1 encoder → FSQ → z_base [2, 32]
                                  │
 contact goals + compliance       │
 proprio/action history → small adapter → delta_z [64]
                                  │
             z = z_base + enable * clamp(delta_z)
                                  │
                         frozen g1_dyn decoder
                                  │
                         29-D joint position target
```

推荐 `post_quantization`，因为它是现成路径，关闭时不改变 encoder/FSQ code；`pre_quantization` 会引入 codebook 边界跳变，不利于精确回退。若 latent residual 在精度或控制带宽上不足，再评估 action residual head，不作为第一步。

### 硬回退

adapter 最后一层零初始化，并在 residual 进入 actor decoder 前做显式乘法：

```python
delta_z = compliance_enable * delta_z_raw
```

关闭模式不调用任何 IK、force estimator 或 modified reference。保留原 SONIC checkpoint 和单独的 adapter checkpoint；部署失败时可直接不加载 adapter。

## 3. 两套索引，不要合并

### 3.1 Tracking skeleton

沿用现有 14 个 `body_names`，职责是：

- dense pose/orientation/velocity reward；
- whole-body regression；
- evaluation 与可视化；
- 从 q reference 做 FK，得到 adapter 所需的 sparse site goal。

### 3.2 Interaction sites

另定义按 body name 解析的 `InteractionSiteSpec`，每项至少包含：

```text
body_name
local_offset_xyz
enabled
compliance_xyz_m_per_n
optional rotational_compliance_xyz_rad_per_nm
contact_allowed
force_limit_n
```

schema 还要带显式 `schema_version`、common-frame 语义与有序 `site_names`；每个运行时
sample 要携带 motion/frame/timestamp 身份。若首版不实现 orientation/torque 路径，
必须把它标成 translational-only v1，而不是保留一个没有数据流和测试的 optional 字段。

建议阶段：

- MVP：左右 `wrist_yaw_link`，覆盖单腕和双腕同时受力；
- V2：左右 shoulder/elbow/wrist 共 6 个上肢 sites，独立 mask/compliance；
- V3：按真实任务加入 torso/forearm collision sites；
- lower body 暂不纳入同一机制。hip/knee/ankle 与地面支撑耦合，尤其 stance foot compliance 是接触规划问题，必须另设训练和验收。

虽然 schema 可以映射到全部 14 点，第一版不应随机让 pelvis、knee、stance ankle 都柔顺。

### 3.3 坐标和关节顺序

- site position、force 与 compliance direction 统一在 robot heading-local 或 reference-root local frame；训练、可视化、部署三处必须相同；
- site 通过 body name 查找，不保存裸 body index；
- q/qdot 始终使用 SONIC 现有 IsaacLab BFS order；对 MuJoCo/外部 motion 只在 adapter 边界复用 `G1_ISAACLAB_TO_MUJOCO_DOF`，不新增第三份 permutation；
- reward/reference 必须按相同 frame、相同 motion frame index 对齐。

## 4. CHIP-style 训练数据流

设第 `i` 个 site 的 reference goal 为 `p_ref_i`，训练 external force 为 `f_i`，命令 compliance 为 `c_i=1/k_i`。只修改 adapter 观察到的 sparse goal：

```text
p_hindsight_i = p_ref_i - c_i ⊙ f_i
```

其中 `⊙` 支持逐轴 compliance；若复现论文首版，可令三个轴相同。原 14-link reference、q/qdot 和全部 dense tracking rewards **保持不变**。

训练信息隔离：

| 信息 | Actor/adapter | Critic | Reward/metrics |
|---|---:|---:|---:|
| 原 G1 motion token | 是 | 是 | — |
| hindsight sparse goals | 是 | 是 | — |
| compliance + site mask | 是 | 是 | — |
| 10-step proprio/action history | 是 | 是 | — |
| 仿真真实 applied force/contact | **否** | 是 | 是 |
| 原 14-link dense reference | 间接在 base token | 是 | 是 |

部署时没有 `f_i`，adapter 接收正常 `p_ref_i`、compliance 与 history；外力由状态/动作残差隐式推断。若实机多接触分辨率不足，再增加 motor-current/external-torque observer 或腕部 F/T，不能假设 history 能唯一分解任意全身多接触。

## 5. 分阶段训练

### Phase A：关闭模式合同

- 加载 released SONIC，冻结 G1 encoder、quantizer、g1_dyn decoder；
- adapter 零初始化；
- 对固定 motion/seed 做 base 与 `enable=0` action/token regression；
- 只有二者在 float tolerance 内一致才进入训练。

### Phase B：双腕 PoC

- 只训练 64D residual adapter 与新 critic 分支；
- mixed batch 至少包含：off/no-force、on/no-force、单腕 force、双腕 simultaneous force；
- 先采用 CHIP 的在线 random force + hindsight goal，force 从小到大 curriculum；
- 原 tracking reward 不改，保留 released config 对双腕/torso 的重点监督；
- 对 `on/no-force` 增加 frozen SONIC action anchor，对 residual 加现有 L2，再视需要加 L1；
- actor 不看 true force，critic 看 per-site force。

这个阶段回答一个问题：只训练小 residual，能否在无力时保持 wrist accuracy、受力时产生可调位移。若不能，不应立即扩到 14 点。

### Phase C：六个上肢 sites 与多接触

- 将 mask 扩到肩/肘/腕，独立采样每个 site 的 compliance；
- 每个 episode 随机激活 1–3 个 sites，并专门保留左右腕同时交互 case；
- 对未激活 sites 继续使用高权重 original tracking；
- 对激活 site 分解 normal/tangent error：只在命令 compliant 的方向允许位移，切向与 orientation 默认保持 stiff；
- 参考 Gentle 的 net force/torque curriculum，避免训练早期只学跌倒或整体平移；
- contact allow-list 由 active mask 控制，未授权身体接触继续处罚。

逐轴 compliance 对末端精度很重要。例如擦拭任务只放松表面法向，双腕的切向轨迹与姿态仍可精确跟踪；单个 isotropic scalar 会不必要地牺牲三个方向的精度。

### Phase D：扩 residual，不漂移 release decoder

只有 residual capacity 明确不足时，先增加 gated latent/action residual 的容量或控制
带宽，并保持 released encoder、FSQ、`g1_dyn`、`g1_kin` 和 action-noise tensor 冻结。
每个训练 checkpoint 都必须重跑 official-vs-off action/token rollout regression，而
不只是初始化前后比较。

解冻 `g1_dyn` 的实验即使使用 baseline action KL/MSE，也只能近似限制漂移；hard gate
无法撤回 base decoder 的权重变化，所以它不满足本设计第 1 条“严格回退 released
SONIC”。如确需研究，应另开明确标为 non-reversible 的实验分支，不能作为主方案验收。

### Phase E：SoftMimic fallback

若少量特定技能在 online hindsight 下仍不稳定，再只对这些 motion/contact 运行 Mink IK augmentation，并经 adapter/loader 转成统一中间表示。不要先为整个 universal motion library 生成 augmented CSV。

## 6. 评价矩阵与 go/no-go

所有 reference/actual 比较必须严格按同一时间戳或 motion frame 对齐。报告左右手分别统计，不能只给全身平均。

| 场景 | 必报指标 | 首轮建议门槛 |
|---|---|---|
| off / no contact | action max-abs、token max-abs、原回报与 success | action/token `atol <= 1e-6`；success 不下降 |
| on / no contact | 左右 wrist position RMSE/P95、orientation、14-point error | wrist mean 相对 baseline `+2 mm` 内，P95 `+5 mm` 内，orientation `+0.02 rad` 内 |
| one-site contact | 命令方向 `Delta x/F`、peak/steady force、settling、oscillation | effective compliance 单调；峰值与稳态单列，不宣称硬上限 |
| two/multi-site contact | 每点 compliance error、net wrench、fall rate、cross-coupling | 未受力 wrist/points 仍满足 no-contact 精度门槛 |
| dynamic motions | walk/run/squat/dance 的 tracking 与 fall rate | 与同 motion baseline 同量级；差异逐项解释 |

门槛是本项目的工程 go/no-go 建议，不是论文已经证明、也不是当前 branch 已经测得的数值。若 released baseline 本身噪声超过门槛，应先用多 seed 置信区间重新定标。

还需记录：

- site/body name 与碰撞 geometry；
- command compliance、实际 force、position/orientation error；
- 采样频率和 frame IDs；
- 只保存抽样窗口与统计，不做无界全量日志；
- 自动滚动清理 debug traces。

## 7. 推荐文件边界

正式实现时建议新增，不把实验逻辑继续堆进现有大文件：

```text
gear_sonic/compliance_control/
  core/
    schema.py            # site、mask、units、frame contract
    residual.py          # residual network 与 hard gate
    perturbation.py      # online force schedule / hindsight goal
    metrics.py           # endpoint/compliance/cross-coupling metrics

gear_sonic/compliance_control/adapters/sonic/isaaclab/
  command.py             # buffer/state/schedule；更新末尾调用窄 writer
  events.py              # wrench writer primitive 与 reset cleanup
  observations.py        # actor/critic separation
  rewards.py             # metrics and optional safety terms

gear_sonic/config/compliance/
  off.yaml
  chip_wrist.yaml
  chip_upper6.yaml

gear_sonic_deploy/policy/compliance/
  observation_config.yaml
  compliance_adapter.onnx
```

核心 `UniversalTokenModule` 已有 residual API，首版应复用而不改。MDP command
管理状态并调用唯一的 wrench writer primitive，reset event 只做生命周期清理；
observation 只做数据适配，reward/metrics 不反向修改 motion core。

这里的“event”是模块职责名，不表示必须注册 IsaacLab interval event。实测
`CommandTerm` 的基类 resample 和新增 interval term 都会推进全局 RNG；即使柔顺
关闭，也会破坏和 release 的严格随机序列对齐。因此生产链路应由 command 自有
generator 调度，并在 command update 末尾直接调用窄 writer；EventManager 只保留
reset cleanup。动态关闭必须在 host setter 返回前定向清除本模块拥有的 body rows，
不能等到下一次 physics，也不能用全 composer reset 覆盖其他外力模块。

## 8. 风险与安全边界

- implicit force inference 对多个未知接触是欠定的；多接触不稳定时优先补 sensing/observer，不靠加大网络掩盖；
- fixed-PD learned compliance 没有 passivity 保证；真机仍需 torque/velocity/power/joint limits、姿态终止、damping e-stop；
- `force_limit` 是命令与训练指标，不是 ISO safety certification；
- lower-body contact、台阶、跪地、被动支撑等需独立任务与 contact-state machine；
- motion/Gentle 的 repo-level license 不清楚，按论文重实现或先获得授权；SoftMimic MIT 可复用时仍保留第三方声明；
- 风险较高的训练应按 AGENTS.md 在独立 branch/worktree 进行，并先做低资源 smoke/performance baseline。
