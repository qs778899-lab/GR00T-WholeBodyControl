# 迁移到其他 Universal Tracker

本页只描述代码边界和迁移顺序；方法选型与论文证据分别见 `README.md` 和
`evidence.md`。目标是让柔顺逻辑迁移时不携带 SONIC、G1、IsaacLab 或固定
14-point 假设。

## 1. 先选实现基线

- 优先保跟踪精度、希望 released policy 结构性可回退：迁移
  `experiment/chip-compliance` 的 hindsight math、信息隔离和 hard-gate 原则。
  该分支的 post-FSQ 位置是 **SONIC adapter 的选择**，不是 CHIP 方法的通用定义。
- 希望最快得到显式 `enable/threshold/Kp`：参考
  `experiment/motion-compliance` 的 condition/reference/wrench 机制。其公开参考和
  当前 branch 都只有 3D global condition；多个 site 可同时施力，不代表 actor 能
  接收逐点独立 compliance。若要求 strict off，不得微调 released decoder 本体。

两条分支的 simulator-independent 代码虽然都位于
`gear_sonic/compliance_control/core/`，但它们位于互斥 branch，文件/API 不同，不能
同时复制或直接 merge。迁移时先选一条固定 commit，并以本页 IR 为目标写映射；在
共同 schema 真正抽出前，不宣称两条 core binary/API compatible。不要复制
`adapters/sonic/` 后只改几个 body name；那一层包含 SONIC observation、checkpoint
和 IsaacLab 生命周期语义，应为目标 tracker 重写。

## 2. 文档级平移中间表示

`compliance.translation.v1` 是本页定义的**迁移 IR 与验收目标**，不是两个分支已经
共同实现的 Python schema、serializer、checkpoint 或 binary API。它只表达平移柔顺；
目标 adapter 应把原始 tracker 状态转换成以下 metadata 和语义张量，再显式映射到
所选分支的实际 API：

```text
schema_version         "compliance.translation.v1"
dtype/device           one declared floating dtype and runtime device
units                  explicit unit for every dimensional field
site_names             ordered [site]
site_link_offset        [site, xyz] m, link-local
link_offset_semantics   link origin -> application point, link-local
force_semantics         force applied by environment on robot
common_frame            {kind, anchor_name, rotation_rule, handedness,
                         up_axis, forward_axis, lateral_axis,
                         quaternion_order}
future_time_offset_s    [future]
motion_id               [batch]
reference_frame_index   [batch, future]
reference_timestamp_s   [batch, future]
sample_timestamp_s      [batch]

reference_target        [batch, future, site, xyz] m
current_site            [batch, site, xyz] m
site_mask               [batch, site] bool
compliance              [batch, site, xyz] m/N
force_on_robot          [batch, site, xyz] N
enable                  [batch] bool/0-1
```

branch-specific mapper 必须另输出
`policy_target [batch, future, site, xyz] m`；它不是 common input IR 的字段，因为
两条 route 的生成规则不同。

| 选择的分支 | IR → 实际训练 API | 已实现的对齐/评价 API |
|---|---|---|
| CHIP | `ComplianceTargetSpec`、`apply_hindsight_target` | `AlignedTrackingTrace` 与严格 paired metrics |
| motion | `ComplianceSpec`、`select_reference`、`virtual_force_from_reference_delta` | Phase 6 评价层需单独映射，不与 CHIP core 互相 import |

这些映射是语义说明，不代表两套类能互载 state dict。中央
`tests/fixtures/translation_v1_two_site.json` 明确标为
`compliance.translation.mapping-fixture.v1`：其中 `ir` 才是 v1 输入，`expected`
保存 CHIP/motion 两条 route 的期望输出，不把 route-only 字段伪装成 IR 字段。目标
adapter 应复用同类 golden fixture，对具名两-site 样例验证 target、force 符号和 mask。

`policy_target` 是 mapper 送入 policy-side adapter 的目标：CHIP 路线可为 hindsight target，
motion 路线可为 selected compliant reference；关闭时必须逐位等于
`reference_target`。所有 Cartesian 张量必须在同一个具名 frame 中；`site` 顺序
来自唯一的 name 列表。future、site、action/DoF 数量由调用方在**构建模型时**提供，
core 不保存裸 body index，也不做 BFS/DFS permutation。

v1 fixture 还必须固定以下不变量：

- Cartesian/force/compliance 张量使用同一 floating dtype/device，全部 finite；mask
  为 bool，若 `enable` 用 float 表示则必须 finite 且逐元素精确为 0 或 1；
- `site_link_offset` 是从 link frame 原点指向 wrench 作用点的 link-local 向量；
- `common_frame` 明确右/左手系、up/forward/lateral 轴、`wxyz`/`xyzw` quaternion
  顺序和向量旋转
  方向，不能只写一个模糊的 `local`；
- `force_on_robot` 明确是环境施加在机器人上的力，不是机器人对环境的反力；
- `sample_timestamp_s`、`reference_timestamp_s`、motion ID 与 frame index 必须严格
  一一相等或由显式、固定的采样映射得到；评价层禁止 nearest-frame 猜测对齐；
- CHIP 的 `policy_target` 是 hindsight target，motion 的 `policy_target` 是
  original/compliant selection；host off 时两者都必须逐位等于 `reference_target`。

当前 flatten-MLP 把 ordered site list、site count 和 future horizon 固定进 checkpoint。
1/2/5/14/17-site contract test 表示五个独立构型都能实例化，不表示一个 checkpoint
可在 runtime 改 cardinality，也不表示对 site permutation 等变。改变 name/order/count
或 horizon 时必须重新构建并执行 checkpoint migration/finetune。

若目标 tracker 要支持 6D compliance，应新增 versioned extension，例如 quaternion
`reference/current_orientation`、`torque_on_robot [N·m]`、
`rotational_compliance [rad/(N·m)]`，并规定 quaternion 顺序、左右乘、最短弧和 torque
frame。不能把这些字段无版本地塞进 v1；v1 的 wrist orientation 只作为应保持 stiff
并单独评估的 tracking 指标。

Tracking skeleton 和 interaction sites 必须是两个独立集合。前者可继续是目标
tracker 的 dense reward/evaluation bodies；后者按任务选择 wrist/elbow/shoulder
等实际作用点，并携带 link-local offset。两者即使 name 相同，也要独立解析各自
的数据源 index。

## 3. 目标 tracker 只需实现的边界

1. `site resolver`：由同一有序 name 列表分别解析 reference source 和 robot
   articulation；启动时验证缺失、重复和顺序。
2. `frame adapter`：把 reference/current/force 转到声明的 common frame；写物理
   wrench 时再使用当前 link pose 转到 simulator 所需 frame。
3. `state/scheduler adapter`：为每个 environment 保存 enable、mask、pulse 和
   force；使用模块私有 RNG 和 fixed-shape candidate，关闭模式不得推进全局 RNG。
4. `observation adapter`：actor 只接 public command、target 和可部署 history；
   applied force/contact 只给 critic、reward 与 metrics。
5. `policy injection`：先探测目标模型能力，再选一个明确 hard-gated 的位置：有
   quantized latent 时可 post-quantization；有连续 latent 时可 post-encoder；没有
   可插 latent 时用独立 action residual。直接 condition concat 会改变旧输入层，且
   3D global condition 不能表达逐点独立命令。portable core 只提供数学/schema，
   不固定 residual width 或 injection topology。
6. `checkpoint adapter`：旧键和旧列逐位复制，新参数零初始化；released backbone
   在 strict-off 路线训练期间保持冻结。官方初始化与已迁移 checkpoint 的 strict
   resume 是两条不同路径；每个训练后 checkpoint 都要与 official policy 做 off-mode
   action/token rollout regression，不能只检查“未训练参数”或新增列非零。
7. `writer lifecycle`：唯一 writer 负责本模块拥有的 body rows。动态关闭必须在
   setter 返回前写零，reset 再做幂等清理，不能 reset 整个共享 composer。

### 3.1 独立部署插件合同

训练/平移 IR 与部署模型 manifest 是两个层次；部署 manifest 不直接消费
`compliance.translation.v1`，而由目标 tracker adapter 桥接。推荐的宿主合同是：

```text
ordered_context_fields  [(name, width), ...]
condition               [..., condition_width]
release_action          [..., action_width]
enabled                 [...] hard gate
release_artifact_pins   [(name, path, sha256), ...]

residual artifact  -> bounded_delta [..., action_width]
compose            -> release_action + enabled * bounded_delta
```

portable loader 必须先验证 schema/opset、context/condition/action/site layout、artifact
digest，以及调用方给出的任意非空 release pin 集合，再创建推理 session。**宿主配置
disabled** 时要直接返回 supplied release action 的原始 bytes，不读取 artifact、不
创建 session。宿主已 enabled 并完成加载后，**当前 batch all-row-off** 只保证不调用
已有 session 的 `run`；它不承诺撤销已经完成的 metadata/model load。mixed batch 要先
隔离 off rows，不能让其中的 NaN/Inf 进入 optional model。具体 token 名、维度、
body/site 名、action permutation 和 operator UI 都属于 `<tracker>` adapter。

| 本地实现 | 当前部署产物 | 可迁移边界 |
|---|---|---|
| motion Phase 5 | `universal-tracker.action-residual.onnx.v1`；ordered context/action/site layout、checkpoint/model/metadata digest、任意 host-owned release pins、Python/C++ host | generic action-residual runtime 可迁；SONIC adapter 独占 `64+930`、3D global condition、双腕/BFS 与 remap 前 compose |
| CHIP Phase 5 | `SonicResidualExportSpec`；target + per-site command + 930 context → 64D post-FSQ latent residual | trace/metrics 是 tracker-neutral；导出的 latent topology、SONIC context 与 base binding 仍是 adapter-level，不能当成 turnkey 通用部署 bundle |

若目标 tracker 没有兼容的 post-encoder/post-quantization hook，优先采用 action
residual，而不是把 CHIP 的 64D graph 强接进去。反之，若希望用 latent residual，
必须为目标 base 重新训练，并补上等价的 base-artifact pin 与 host hard-off 合同。

## 4. 有序迁移流程

1. 固定一个 branch 与 commit；两套 core 二选一，不直接 merge。
2. 用 semantic site names、link-local offsets、frame/handedness、force sign、motion ID、
   frame index 与 timestamp 把目标 tracker 归一到文档 IR。
3. 写 branch-specific mapper 与 golden fixture，固定 ordered site list、site count 和
   future horizon；这些改变都发生在 checkpoint 构建/微调前。
4. 选择 latent 或 action injection，并零初始化 residual；冻结 release base，分别
   实现 official initialization 与 strict resume。
5. 导出 residual-only artifact，固定 context/action/site layouts 和所有宿主 release
   artifact hashes；在目标 tracker 自己的 action permutation 之前 compose。
6. 实现结构性 disabled bypass，再按 contract → 真实 writer/frame → 16-env 训练启动
   → 4096-env performance → 配对 tracking/compliance 的顺序验收。

## 5. 迁移时不可复用的 SONIC 细节

- 10×`(29 q + 29 qdot + 6 anchor)` 的 robot-motion encoder 输入；
- IsaacLab BFS DoF 顺序和 MuJoCo DFS permutation；
- SONIC 的 14-body reward skeleton、29D action、64D FSQ token；
- `WrenchComposer` 的 API、ManagerBasedEnv command/event 时序；
- released checkpoint 的 actor `994`、critic `1645` 输入宽度。

这些都属于目标 tracker adapter/checkpoint config。可复用的是按 name/site 的
数学、hard gate、时序、限幅和指标；residual 网络只有在目标 tracker 暴露兼容
injection capability 时才能复用，不是无条件 portable。

## 6. 最小迁移验收

在接真实 simulator 前先运行无 simulator 的 contract tests：

- 分别实例化 1、2、5、14、17 个固定-site 构型覆盖 shape，并确认不同 cardinality
  的 checkpoint 明确拒绝互载；
- schema version、site name/order、link-local offset、frame、future offsets 和
  motion/frame/timestamp 缺失或不一致时 fail closed；
- `enable=false` 和零 compliance 对 target/residual 精确恒等；
- mixed batch 只改变 active sites，inactive sites bitwise 不变；
- 非单位旋转下验证 frame 和 `force_on_robot` 符号；
- reference/articulation index 表故意不同仍能得到正确结果；
- 初始化迁移时旧 checkpoint 的所有旧 tensor/列逐位相等，新 head/列精确为零；
- 至少一个训练后 checkpoint 的 `enable=false` action/token 与 official rollout
  逐帧一致；只验证冻结 tensor 或初始化零列不够；
- actor 图中不存在 privileged force，critic 图中存在；
- reset、active→off、重复 off 后 owned writer rows 恒为零且不影响其他 rows；
- CUDA command hot path 无 `aten::nonzero` 和 `aten::_local_scalar_dense`。

随后只用 1 environment 做真实 writer/frame smoke，再做 16-environment、5-step
训练启动；这些都只证明链路可运行。最后才做 4096-environment 性能表征，以及
`design.md` 第 6 节完整的多-motion tracking/compliance 回归。任何 target 和实际
轨迹对比都要记录相同 motion ID、frame ID 和时间戳，左右末端分开统计。

## 7. 建议目录

```text
target_tracker/compliance_control/
  core/                 # 选定方法的纯 math/schema；两实验分支 API 当前不兼容
  deployment/           # artifact schema、digest、lazy runtime、compose
  adapters/<tracker>/
    resolver.py         # name -> reference/articulation IDs
    frames.py
    command.py
    observations.py
    writer.py
    checkpoint.py
    deployment.py       # base pins、context assembly、action order、operator gate
  training/
    finetune.py
    audit.py
  configs/
    off.yaml
    wrist.yaml
  tests/
    test_core_contract.py
    test_<tracker>_adapter.py
```

如果目标 tracker 已有 command、writer 或 checkpoint registry，只写薄 wrapper，
不要在 entrypoint 复制 core 流程；部署和训练共用同一份 site/frame schema。
