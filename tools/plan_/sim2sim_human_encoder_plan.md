# 方案分析：parquet → human motion encoder → sim2sim/IsaacSim eval

**注意**：用户指定"只先做分析+预研，不实现"。本文档是方案分析，不是实现步骤。如果后续要落地，再基于本文档展开。

---

## Context（为什么需要这个功能）

当前的sim2sim error分析链路（见 `sim2sim.md`）都走的是 **robot motion encoder（mode_id=0, "g1"）**：
- Terminal C：`tools/sonic_eval/stream_motionlib_to_deploy.py` 把motionlib pkl中的G1机器人关节轨迹通过 **ZMQ Protocol v1** 发送
- Terminal B：deploy的encoder检测到v1→设定 `active_protocol_version_=1` → encoder_mode=0
- Terminal A：MuJoCo仿真，记录14个link的world position做metrics

这套链路无法评测policy在 **human motion encoder（mode_id=2, "smpl"）** 下的tracking精度。

`data_0424/data/chunk-000/episode_000000.parquet`是sonic官方数据采集pipeline（`gear_sonic/scripts/run_data_exporter.py`）落盘的数据，里面同时包含SMPL人体动作和G1机器人状态。本方案要分析的是：**如何用这份数据驱动deploy的SMPL encoder走完一条sim2sim error分析链路，并对应到IsaacSim eval的SMPL encoder评测语义**。

---

## 0. 关键澄清（2026-05-18 补充）：两条互斥链路，retarget 不是 deploy 推理的依赖

本文档原 4.5/5.0.B 节主张 "parquet → human motion encoder 必须经过 SMPL→G1 retargeting 生成 robot pkl"。经代码验证，这一主张混淆了两条**解耦**的链路。读后续章节前必须区分清楚：

### 0.1 Deploy 推理链路（ZMQ 流式，本任务的核心）

```
SMPL 数据源（pico 真机 / parquet 重放）
  → 拼 ZMQ Protocol v3 包
  → deploy SMPL encoder (mode_id=2)
  → policy → MuJoCo / 真机执行
```

**不经过 motion_lib，不需要 pkl，不需要 retargeter**。

代码证据：[gear_sonic/scripts/pico_manager_thread_server.py:1346-1403](gear_sonic/scripts/pico_manager_thread_server.py#L1346-L1403) 在 sender 端**内联做 6 个 G1 wrist DOF 的解析式 retargeting**（swing/twist 分解 + Euler，**非** `Humanoid_Batch.fk_batch`），把结果直接写入 `joint_pos[23..28]` 通过 ZMQ pose 消息发 deploy。

parquet 是这条 ZMQ 流的**镜像落盘**：[run_data_exporter.py:410-447](gear_sonic/scripts/run_data_exporter.py#L410-L447) 把 pico 发的 `joint_pos[23,25,27]` 切成 `teleop.left_wrist_joints (3,)`、`joint_pos[24,26,28]` 切成 `teleop.right_wrist_joints (3,)`。这意味着 parquet 已经**完整包含**了 deploy SMPL encoder 推理所需的全部输入字段。

### 0.2 IsaacSim eval 链路（仅当需要算 14-link MPJPE GT 时）

```
SMPL pkl + robot pkl
  → motion_lib.load() + Humanoid_Batch.fk_batch()
  → IsaacSim env: encoder 观测 + 14-link GT → MPJPE
```

**这条路径才必须 retarget + pkl**，因为 motion_lib 只读 pkl，且 14-link GT 需要 G1 完整 FK。

### 0.3 实操推论

| 任务 | 链路 | 需要 retarget？ | 需要 pkl？ |
|---|---|---|---|
| parquet 重放 → deploy SMPL encoder 推理（本任务） | 0.1 | **否** | **否** |
| 算 MPJPE 配 GT 做 sim2sim error 分析 | 0.2 | 是（GMR 或 fk_batch） | 是 |
| IsaacSim eval (`+use_encoder=smpl`) | 0.2 | 是 | 是 |

**本文档后续 4.5 节和 5.0.B 节的 GMR 集成讨论，仅对 0.2 路径有效。如果你只关心让 deploy SMPL encoder 跑起来（推理链路），直接跳到 5.1 看新 streamer 设计。**

---

## 1. parquet里有什么（已确认）

`data_0424/`是LeRobot格式的TorchRL dataset，50 Hz采样，每个episode一个parquet（约500行），目录结构：
```
data_0424/
├── data/chunk-000/episode_{000000..000003}.parquet
├── meta/{info.json, modality.json, episodes.jsonl, tasks.jsonl}
└── videos/observation.images.ego_view/episode_*.mp4
```

关键字段分三类：

**SMPL human motion（teleop输入端）**
| 字段 | 形状 | 含义 |
|---|---|---|
| `teleop.smpl_joints` | (72,) | 24个SMPL joint × 3D xyz |
| `teleop.smpl_pose` | (63,) | axis-angle，root3 + 21 body joints × 3（**注意：21不是24**） |
| `teleop.body_quat_w` | (4,) | SMPL world body quaternion (wxyz) |
| `teleop.smpl_frame_index` | (1,) | SMPL源帧索引 |
| `teleop.left_hand_joints` | (7,) | 左手关节 |
| `teleop.right_hand_joints` | (7,) | 右手关节 |
| `teleop.left_wrist_joints` | (3,) | 左腕roll/pitch/yaw |
| `teleop.right_wrist_joints` | (3,) | 右腕roll/pitch/yaw |
| `teleop.stream_mode` | int32 | 1=SMPL, 4=SMPL alt, 5=planner |

**G1机器人状态（数采时WBC输出+传感器测量）**
| 字段 | 形状 | 含义 |
|---|---|---|
| `observation.state` | (43,) | G1实际达到的关节角（29 body + 14 hand） |
| `action.wbc` | (43,) | 数采时WBC给G1的关节命令 |
| `observation.root_orientation` | (4,) | G1 base四元数 |
| `observation.init_base_quat` | (4,) | G1初始base四元数 |
| `action.motion_token` | (64,) | motion encoder token |

**metadata**: `frame_index`、`timestamp`、`episode_index`、`task`。

视频在`videos/`下，parquet里只存路径。

---

## 2. deploy侧的SMPL encoder mode已经布好线（已确认）

`gear_sonic_deploy/policy/release/observation_config.yaml` (lines 74-80) 已经定义：

```yaml
- name: "smpl"
  mode_id: 2
  required_observations:
    - encoder_mode_4
    - smpl_joints_10frame_step1            # 720D = 24×3×10
    - smpl_anchor_orientation_10frame_step1 # 60D = 6×10
    - motion_joint_positions_wrists_10frame_step1  # 60D = 6×10
```

**deploy的mode dispatch**（见 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp:919-952`）：

| Protocol版本 | 必填字段 | 设定的encoder mode |
|---|---|---|
| v1 | `joint_pos`, `joint_vel` (+ body_pos/quat/frame_index) | 0 (g1) |
| v2 | `smpl_joints`, `smpl_pose` (+ joint_pos/joint_vel可选) | 2 (smpl) |
| v3 | **全部**: `smpl_joints` + `smpl_pose` + `joint_pos` + `joint_vel` | 2 (smpl) |
| v4 | token-only (不需要) | - |

也就是说：**deploy本身根据protocol版本自动切到SMPL encoder，不需要任何额外的CLI flag**。

发现的限制：v3要求 `joint_pos`/`joint_vel`/`smpl_joints` 帧数完全一致（hpp lines 985-994）。

---

## 3. IsaacSim eval的SMPL encoder链路（已澄清）

### 3.1 SMPL pkl字段含义

`gear_sonic/utils/motion_lib/motion_lib_base.py:1882-1932` 加载SMPL pkl。这是参数化人体模型（24个anatomical joints），三个字段是同一帧人体姿态的不同表达：

| 字段 | 含义 |
|---|---|
| `pose_aa[T, 72]` | 24个SMPL joint的**关节角度**，axis-angle。前3维是root（pelvis）在**Y-up世界系**中的全局旋转；后69维是23个body joints相对于父joint的**局部旋转**。手指最后6维通常归零。 |
| `transl[T, 3]` | SMPL pelvis在**Y-up世界系**中的3D平移坐标 |
| `smpl_joints[T, 24, 3]` | 24个SMPL joint的3D世界坐标位置（Y-up），由 `pose_aa + transl` 经SMPL人体FK展开。pkl没存就运行时算 |

**没有任何机器人侧字段**——pkl只描述人体姿态。

### 3.2 SMPL encoder的三个观测：**人体语义+机器人语义混合**

deploy的 `observation_config.yaml:74-80` 声明SMPL mode需要三个observations，但它们不全是人体语义：

| 观测 | 维度 | 物理含义 | 骨架 |
|---|---|---|---|
| `smpl_joints_10frame_step1` | 720 = 10×24×3 | 24个**SMPL人体joint**的3D位置（去heading后相对坐标） | **人体SMPL** |
| `smpl_anchor_orientation_10frame_step1` | 60 = 10×6 | SMPL root的orientation | **人体SMPL** |
| `motion_joint_positions_wrists_10frame_step1` | 60 = 10×**6个DOF角度** | **G1**左右wrist的roll/pitch/yaw关节角（IsaacLab indices {23..28}） | **G1机器人** |

**关键澄清**：`motion_joint_positions_wrists` 的60维 = `10帧 × 6个scalar`（G1的6个wrist DOF角度），**不是** `10×2×3`（人体两个wrist的3D位置）。代码证据：
- 训练观测函数 `joint_pos_multi_future_select_joints_for_smpl`（observations.py:1567-1588）
- 读 `TrackingCommand.joint_pos_multi_future_for_smpl` → `motion_lib.get_dof_pos(...)`
- `motion_lib.dof_pos` 的来源：`motion_lib_base.py:1873-1881` 调 `Humanoid_Batch.fk_batch(pose_aa, transl)`，里面 `dof_pos = pose.sum(dim=-1)[..., actuated_joints_idx]`——把SMPL pose **retarget成G1 actuated DOF角度值**

### 3.3 为什么wrist必须用机器人语义补充？

SMPL人体wrist是**1个joint**（3维axis-angle），G1机器人wrist是**3个独立joint串联**（roll→pitch→yaw）。光给policy一个SMPL wrist的3D位置，G1有无穷多个IK解。所以显式喂retargeted的6个G1 wrist DOF角度给policy当目标。

**SMPL encoder不是"纯human input"，是"以人体SMPL为主 + G1 wrist DOF target补充"的混合输入**。

### 3.4 IsaacSim eval的GT是什么

**GT和encoder输入语义不一样**。GT在 `im_eval_callback.py:532-547, 600-610` 计算：
- GT = `motion_lib.get_body_pos_w()` 取14个**G1机器人link**世界坐标（pelvis, hip_roll×2, knee×2, ankle_roll×2, torso, shoulder_roll×2, elbow×2, wrist_yaw×2）
- 这14个link位置是**motion_lib加载时**就由`Humanoid_Batch.fk_batch()`把SMPL pose通过**G1骨架的FK**算出来存好的
- Predicted = G1机器人在IsaacSim里实际仿真的14个body link位置
- MPJPE_g/l/pa = 14个link之间的Euclidean距离平均

**也就是**：encoder输入是"人看到的人体动作"，GT是"如果用G1去重现这套动作，G1每个link应该在哪"。Policy的任务=学会把人体语义"翻译"成G1 link位置控制。Eval只看G1骨架空间的tracking精度。

### 3.5 对parquet streamer的实操含义

parquet里的 `teleop.smpl_pose` 是63D（21 joints×3，无root），调 `fk_batch` 前要拼成72D（root_aa 3 + body 63 + hands 0×6=72，或者从 `teleop.body_quat_w` 转axis-angle作root）。pose_aa+transl送进fk_batch后输出的 `dof_pos[T, 29]` 就是streamer要发的 `joint_pos`。GT用同样的fk_batch输出的 `body_pos_w` 切14个link。

streamer发ZMQ v3的字段映射：
- `smpl_joints` ← `teleop.smpl_joints`（直接用，人体语义）
- `smpl_pose` ← `teleop.smpl_pose`（直接用，人体语义；shape保持[T,21,3]，deploy支持任意num_smpl_poses）
- `joint_pos[:, 23:29]` ← fk_batch retarget的G1 wrist DOF（机器人语义，必须经FK算）
- `joint_pos[:, 0:23]` ← 可填fk_batch完整retarget结果或零（deploy SMPL encoder只读{23..28}，不影响policy输入）
- `body_quat_w` ← `teleop.body_quat_w`（人体SMPL anchor）
- `body_pos_w` ← 零或SMPL pelvis（deploy SMPL anchor obs只读`body_quat_w[0]`，不读body_pos_w）

**用户选"GT=SMPL→G1 retarget"和这套实现是一致的**：streamer的retargeted dof_pos、metrics GT的14个link位置，都从同一次`fk_batch`调用的输出取，完全自洽。

---

## 4. 已有可复用资产

| 资产 | 路径 | 作用 |
|---|---|---|
| SMPL→G1 FK | `gear_sonic/utils/motion_lib/motion_lib_base.py:1873` + `torch_humanoid_batch.py:fk_batch` | 把parquet里的`teleop.smpl_pose`转成G1 29-DOF |
| Humanoid_Batch | `motion_lib_robot.py` | FK引擎实例化 |
| 现有stream | `tools/sonic_eval/stream_motionlib_to_deploy.py` | `_prepend_stand_transition`/`_finite_difference`/`PackedPublisher._send`可参考；`send_pose`硬编码v=1头不能直接复用 |
| metrics工具 | `tools/sonic_eval/compute_mujoco_tracking_metrics.py` | encoder-agnostic，只读14 link world position + source_frame_index |
| stale帧检测 | `gear_sonic/scripts/process_dataset.py:build_stale_mask` | （**用户选不做清理**，仅备查） |
| MuJoCo logger | `gear_sonic/utils/mujoco_sim/base_sim.py` | step_sync_body_pos_w_14.csv写入，已经encoder-agnostic |
| IsaacSim eval | `gear_sonic/eval_agent_trl.py` + `+use_encoder=smpl` | 已支持SMPL encoder评测 |
| parquet→motion | `tools/sonic_eval/parquet_to_mujoco_motion.py` | 现成的parquet读取+MuJoCo→IsaacLab关节顺序映射，可参考但用途不同（它走robot encoder） |

---

## 4.5 外部 retargeter（**仅 IsaacSim eval / metrics GT 路径需要**）

> **重要前置**（见第 0 节）：retargeter 不是 deploy 推理链路（ZMQ → encoder → policy）的依赖。pico sender 已经在 ZMQ 字节流里把 6 个 G1 wrist DOF 解析式算好了，parquet 是这条 ZMQ 流的镜像落盘，可直接重放给 deploy。
>
> 本节的 GMR 讨论**只在以下场景才相关**：需要算 14-link MPJPE 的 motion_lib GT、或者要跑 IsaacSim `eval_agent_trl.py +use_encoder=smpl`（这两个都通过 motion_lib 加载 pkl，必须有完整 G1 29-DOF retargeted motion）。
>
> 如果你只想跑 deploy SMPL encoder 推理（本任务），跳到 5.1。

### 4.5.1 本repo官方立场（已确认）

`docs/source/user_guide/new_embodiments.md:340-386` 明确：

> "**SOMA Retargeter** (recommended) — NVIDIA's BVH-to-humanoid motion retargeting library... **This is the same tool used to produce the Bones-SEED G1 retargeted data.**"
> "**GMR** (General Motion Retargeting) — retargets human motions to arbitrary humanoid robots..."

也就是说：
- Bones-SEED的**G1侧**数据（`robot_filtered/`）是用 **NVIDIA SOMA Retargeter** 离线产的
- Bones-SEED的**SMPL侧**数据（`smpl_filtered/`）由Bones Studio内部pipeline产，**未开源**
- 用户被默认走"下载预生成数据"路径；要做自定义就需要引入外部retargeter

| 数据集 | 路径 | 内容 | 来源 |
|---|---|---|---|
| robot_filtered | `data/motion_lib_bones_seed/robot_filtered/` | G1 fitted `pose_aa` | BVH → SOMA Retargeter → CSV → `convert_soma_csv_to_motion_lib.py` 转pkl格式 |
| smpl_filtered | `data/smpl_filtered/` | 原始SMPL pose_aa + smpl_joints + transl | HuggingFace下载（Bones Studio内部pipeline生成） |

IsaacSim eval加载时：
- `motion_lib_cfg.motion_file` → robot_filtered（提供G1 fitted pose_aa，fk_batch输入，14 link GT输出）
- `motion_lib_cfg.smpl_motion_file` → smpl_filtered（提供SMPL encoder观测输入）

按sequence name配对消费。

### 4.5.2 parquet场景的retargeter选择：**GMR**

parquet里已经有现成的SMPL pose_aa（`teleop.smpl_pose` 21×3），不是BVH格式。所以：
- **SOMA Retargeter不适用**：input是BVH，要先转SMPL→BVH多一道工序
- **GMR适用**：定位就是"human motion → humanoid robot"，直接接SMPL pose

**官方推荐路径**（按 new_embodiments.md 的语义）：parquet→GMR→G1 fitted pose_aa → 跟parquet里的SMPL data绑成motion pkl → 输入IsaacSim eval / sim2sim streamer / metrics tool。

### 4.5.3 GMR使用前必须验证的点

未实际用过GMR，使用前需验证：
1. **输入format**：GMR能不能直接吃 `pose_aa[T, 24, 3]` (axis-angle) + `transl[T, 3]`？还是要先转其他中间格式？
2. **机器人配置**：GMR是否原生支持G1 MJCF/URDF？还是需要写robot config？官方支持的humanoid list里有没有G1？
3. **输出format**：GMR输出是不是G1 joint angles序列？fps？是否需要对齐我们的IsaacLab joint order？
4. **依赖与运行环境**：是否依赖smplx body model、IsaacLab、cuda？能否在 sonic_eval/sonic_backup conda env跑？
5. **质量评估**：跑一两条parquet看输出G1 motion是否合理（关节范围内、无奇异、足底不穿模），跟bones_seed的G1 motion做spot check对比

如果GMR有阻塞（不支持G1、依赖冲突、retargeting质量差），fallback选项：
- 复刻PHC风格的optimization-based fitting：用smplx body model FK做SMPL keypoint生成，loss = ||SMPL_kp - G1_FK_kp||²，Adam/LBFGS优化G1 joint angles + root transform
- 或者跟Bones Studio团队询问smpl_filtered的generator

---

## 5. 实现方案

> **路径选择**（见第 0 节）：
>
> - **5.0.deploy-only（推荐，最小路径）**：parquet → 直接拼 ZMQ v3 包 → deploy。零 retargeter 依赖，零 pkl 中转。**对应 5.1 节新建的 `stream_parquet_smpl_to_deploy.py`**。
> - **5.0.eval-also（可选，需 GMR）**：parquet → GMR retarget → robot pkl + smpl pkl → 既可走 IsaacSim eval、也可走 metrics GT。**对应原 5.0.A/5.0.B/5.2/5.3 节**。
>
> 两条路径互斥地满足不同任务。本任务（让 deploy human motion encoder 跑起来）只走 deploy-only 路径就够了。

### 5.0.deploy-only 字段映射表（推荐路径，零 retarget）

新 streamer 直接从 parquet 拼 ZMQ Protocol v3 包，字段映射：

| ZMQ v3 字段 | 来源（parquet 字段） | 处理 |
|---|---|---|
| `smpl_joints[T,24,3]` | `teleop.smpl_joints` reshape | 直接用；按 `--smpl-joints-mode canonicalized` 做 frame 变换（复用第 10 节修复） |
| `smpl_pose[T,21,3]` | `teleop.smpl_pose` reshape `(T,63) → (T,21,3)` | 直接用 |
| `body_quat_w[T,4]` | `teleop.body_quat_w` | 按 `--smpl-anchor-mode` 做 frame 变换（复用第 10、11 节修复） |
| `joint_pos[T,29]` indices `{23,25,27}` | `teleop.left_wrist_joints[:,0:3]` | 拷贝（pico sender 已 retarget） |
| `joint_pos[T,29]` indices `{24,26,28}` | `teleop.right_wrist_joints[:,0:3]` | 拷贝（pico sender 已 retarget） |
| `joint_pos[T,29]` 其他 indices | 填 0 | deploy SMPL encoder 不读，无影响 |
| `joint_vel[T,29]` | `_finite_difference(joint_pos)` 或 0 | 复用 motionlib streamer 的 helper |
| `body_pos_w[T,3]` | 填 0 或 `teleop.smpl_joints[:,0,:]`（pelvis） | deploy SMPL anchor obs 不读 body_pos_w |
| `frame_index[T]` | streamer 内部 row index 或 `teleop.smpl_frame_index` | 跟 motionlib streamer 一致 |

→ **零 GMR、零 fk_batch、零 motion_lib、零 pkl 中转**。

---

### 阶段1（仅 5.0.eval-also 路径需要）：准备**两个pkl**（严格IsaacSim eval语义）

motion_lib只读pkl（`joblib.load`），所以parquet→eval必须经过pkl中转。按bones_seed的官方双文件配对：

#### 5.0.A `<name>_smpl.pkl` (smpl_filtered格式)：纯字段重组

目标schema（对齐 `sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl`）：
```
pose_aa:           (T, 72)   f32   # 24×3 SMPL axis-angle，含root
transl:            (T, 3)    f32   # SMPL pelvis Y-up world translation
smpl_joints:       (T, 24, 3) f32  # SMPL joint world position
fps:               <float>         # target_fps
original_pose_aa:  (T0, 72)  f32   # 插值前原始（如果parquet本身就是50Hz可省）
original_fps:      <float>         # 原始fps
```

parquet字段映射（无retarget）：
- `pose_aa`: parquet `teleop.smpl_pose`（63D=21×3 body axis-angle）前面拼root axis-angle（从 `teleop.body_quat_w` 转），后面补hands零；最终 [T, 72]
- `transl`: parquet `teleop.smpl_joints[:, 0, :]`（取SMPL pelvis的3D position，需Y-up确认）
- `smpl_joints`: parquet `teleop.smpl_joints` reshape [T, 24, 3]
- `fps`: 50.0（parquet本身就是50Hz）
- `original_pose_aa`/`original_fps`: 可省或填同值

工具：`tools/sonic_eval/parquet_to_smpl_pkl.py`，纯numpy重组，无外部retargeter依赖。

#### 5.0.B `<name>_robot.pkl` (robot_filtered格式)：**需要GMR retargeting**

目标schema（对齐 `sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl` 顶层嵌套dict）：
```
{
  "<motion_name>": {
    root_trans_offset: (T, 3)     f32  # G1 root world position
    pose_aa:           (T, 30, 3) f32  # G1 fitted axis-angle (30 = 1 root + 29 joints)
    dof:               (T, 29)    f32  # G1 actuated DOF角度 scalar
    root_rot:          (T, 4)     f32  # G1 root四元数
    smpl_joints:       (T, 24, 3) f32  # 对应SMPL reference (从parquet teleop.smpl_joints)
    fps:               <int>            # 一般是30或50
  }
}
```

数据来源：parquet `teleop.smpl_pose` + `teleop.body_quat_w` + `teleop.smpl_joints` + `transl` → **GMR retargeting** → G1 fitted pose_aa / dof / root变换

工具：`tools/sonic_eval/parquet_to_robot_pkl.py`，必须依赖外部GMR（按4.5.3验证后接入）。

#### 5.0.C 假设阶段

按用户决策，**plan从"假设我们已经有这两个pkl"展开**。retargeter integration细节作为外挂工序在4.5节描述，本plan的核心实现是后续streamer + metrics + IsaacSim eval recipe。

如果验证GMR可用，5.0.A + 5.0.B 可以合并成一个 `tools/sonic_eval/parquet_to_motion_pkls.py`（输出双文件）。

---

### 5.1.deploy-only 新建 streamer：`tools/sonic_eval/stream_parquet_smpl_to_deploy.py`

**职责**：直接读 parquet → 按 5.0.deploy-only 字段映射表拼 ZMQ Protocol v3 包 → 发送给 deploy（触发 encoder_mode=2）。零 retarget、零 pkl。

**对齐参考实现**：
- pico 真机链路 [pico_manager_thread_server.py:1346-1403](gear_sonic/scripts/pico_manager_thread_server.py#L1346-L1403)（sender 内联 retarget + ZMQ pose 发送）
- pkl-based SMPL streamer [tools/sonic_eval/stream_motionlib_smpl_to_deploy.py](tools/sonic_eval/stream_motionlib_smpl_to_deploy.py)（已实现，frame 变换、ZMQ v3 打包、blend prefix 都已经踩过坑）

**零侵入式扩展原则**（用户要求）：本任务只新增文件，**不修改任何现有代码**。

**复用契约**（import 不修改）：
```python
from tools.sonic_eval.stream_motionlib_to_deploy import (
    HEADER_SIZE,           # ZMQ 头 padding
    _finite_difference,    # dof_vel
    _prepend_stand_transition,  # stand→motion blend
)
from tools.sonic_eval.stream_motionlib_smpl_to_deploy import (
    _smpl_root_ytoz_up,
    _remove_smpl_base_rot_wxyz,
    _compute_smpl_root_quat_w,
    _canonicalize_smpl_joints,
    # 以及 PackedPublisherSMPL（ZMQ v3 header 打包）
)
```

**CLI（拟）**：复用 motionlib smpl streamer 的全部 flags（host/port/target-fps/chunk-size/realtime/send-command/initial-burst-frames/blend-from-stand-frames/smpl-anchor-mode/smpl-joints-mode/...），改输入参数：
- `--parquet <path>`（替代 `--motion-file` / `--smpl-motion-file`）
- `--episode-index INT`（如果 parquet 是多 episode）

**数据流（单 episode）**：
```
parquet (T 行)
  ↓ pd.read_parquet + 字段抽取
{
  smpl_joints:   teleop.smpl_joints  → (T,24,3) float32
  smpl_pose:     teleop.smpl_pose    → reshape (T,21,3) float32
  body_quat_w:   teleop.body_quat_w  → (T,4) float32
  joint_pos:     np.zeros((T,29)) → 写入 indices {23,25,27,24,26,28} from teleop.{left,right}_wrist_joints
  joint_vel:     _finite_difference(joint_pos)
  frame_index:   np.arange(T) 或 teleop.smpl_frame_index
}
  ↓ frame 变换（复用第 10/11 节的 helper）
{
  body_quat_w:   robot_root 或 smpl_processed（按 --smpl-anchor-mode）
  smpl_joints:   canonicalized（按 --smpl-joints-mode）
}
  ↓ _prepend_stand_transition + chunk + ZMQ Protocol v3 header
deploy
```

**关键工程细节**：
- v3 要求所有 motion 字段帧数严格一致（`zmq_endpoint_interface.hpp:985-994`），所以从 parquet 抽出来的字段直接共享 T，不需要额外对齐
- `body_quat_w` frame 变换：parquet 的 `teleop.body_quat_w` 是 SMPL world body quaternion（Y-up），处理方式与 `stream_motionlib_smpl_to_deploy.py` 完全一致
- pico 链路里 `joint_pos[23..28]` 是**当前帧的瞬时值**，写入 parquet 后是 T 帧的连续序列，可直接用作 `joint_pos[T,29]`
- 不需要走 IsaacSim app（无 fk_batch、无 motion_lib）

**端到端验证（实跑时核对）**：
1. 起 MuJoCo + deploy + 这个新 streamer
2. deploy 日志确认 `active_protocol_version_=3` 和 `encoder_mode=2`
3. MuJoCo G1 不摔倒、policy 输出合理（visually 跟 parquet 里的 SMPL 动作相似）
4. 对照：同一段 parquet，跟 pico 真机重放该 episode 的 deploy log 对比 encoder 输入数值（数量级一致即可）

---

### 5.1.eval-also 新建 streamer：`tools/sonic_eval/stream_smpl_pkl_to_deploy.py`

**职责**：读两个pkl → 构造ZMQ Protocol v3包 → 发送给deploy（触发encoder_mode=2）

**数据流**：
```
<name>_robot.pkl                              <name>_smpl.pkl
  └── pose_aa[T, 30, 3]                         └── pose_aa[T, 72]
        ↓ axis-angle.norm按actuated_idx               ↓ reshape [T, 24, 3] (注意GMR输出排序)
      dof[T, 29] (或直接读pkl里的dof字段)         smpl_pose[T, 21, 3] (取body部分给ZMQ)
                                                  smpl_joints[T, 24, 3]
                                                  transl[T, 3]
        ↓                                                 ↓
        ZMQ.joint_pos[T, 29]               ZMQ.smpl_joints / ZMQ.smpl_pose
        ZMQ.body_pos_w (取root_trans_offset 或 smpl transl)
        ZMQ.body_quat_w (取root_rot 或 teleop.body_quat_w)
        ZMQ.frame_index[T]
  
ZMQ Protocol v3 header (header version=3):
  {
    "v": 3, "endian": "le", "count": N, "motion_start_frame": <int>,
    "fields": [
      {"name": "joint_pos",   "dtype": "f32", "shape": [N, 29]},   # FK retargeted
      {"name": "joint_vel",   "dtype": "f32", "shape": [N, 29]},   # finite_difference
      {"name": "smpl_joints", "dtype": "f32", "shape": [N, 24, 3]},
      {"name": "smpl_pose",   "dtype": "f32", "shape": [N, 21, 3]},
      {"name": "body_pos_w",  "dtype": "f32", "shape": [N, 3]},     # 零或SMPL anchor
      {"name": "body_quat_w", "dtype": "f32", "shape": [N, 4]},     # teleop.body_quat_w
      {"name": "frame_index", "dtype": "i64", "shape": [N]},
      {"name": "catch_up",    "dtype": "u8",  "shape": [1]}
    ]
  }
```

**CLI（拟）**：复用motionlib streamer的flags（host/port/target-fps/chunk-size/realtime/send-command/initial-burst-frames/blend-from-stand-frames/...），新增：
- `--parquet <path>` (必填)
- `--episode-index INT`（如果parquet是单episode可省）
- 不清理stale帧（按用户要求）

**与现有motionlib streamer的复用契约**：
- `_prepend_stand_transition` 和 `_finite_difference` 是模块级函数，可直接import
- `PackedPublisher` 因为header硬编码`v=1`不能复用，需要复制一份做`PackedPublisherSMPL`，header设`v=3`
- 主循环（prestart_frames / send_command / heartbeat / realtime pacing）结构性照抄

**关键工程细节**：
- v3要求所有motion字段帧数严格一致（hpp:985-994），SMPL→G1 FK输出的帧数要==smpl_joints帧数
- `motion_start_frame`语义：取`prepend_stand_frames + blend_from_stand_frames`，跟motionlib streamer保持一致
- `body_quat_w[0]`是SMPL anchor orientation的源头（deploy `GatherMotionAnchorOrientationMutiFrame`从`BodyQuaternions(target_frame)[0]`读），所以prefix stand段的`body_quat_w`要保持等于第一帧实际SMPL anchor，否则anchor会突变
- 不需要走IsaacSim app（fk_batch可以CPU/GPU独立运行，不依赖IsaacSim SimulationApp）

#### 5.2 metrics：直接复用 `compute_mujoco_tracking_metrics.py --gt-format motionlib`

阶段1产的pkl已经是motion_lib消费格式，**零修改**复用现有命令：
```bash
python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
  --gt-format motionlib \
  --motion-file /tmp/sonic_eval/parquet_motion_pkl/episode_000000.pkl \
  --motion-name episode_000000 \
  --logs-dir /tmp/sonic_logs/parquet_smpl_ep0 \
  --out-json /tmp/parquet_smpl_ep0_metrics.json \
  --no-motionlib-robot \
  --ignore-motion-playing-mask \
  --streamed-only \
  --align-mode source_frame_index \
  --actual-source step_sync_body_pos_w_14 \
  --sim-valid-only \
  --use-isaacsim-app
```

GT计算路径：`motionlib.load_motion(pkl)` → `fk_batch(pose_aa, trans)` → `body_pos[T, num_bodies, 3]` → 切14个link indices = GT。完全复用robot encoder链路已经验证过的代码。

#### 5.3 IsaacSim eval recipe（同一份pkl）

```bash
python gear_sonic/eval_agent_trl.py \
  +use_encoder=smpl \
  +checkpoint=/path/to/last.pt \
  ++eval_callbacks=im_eval \
  ++num_envs=1 \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=/tmp/sonic_eval/parquet_motion_pkl/episode_000000.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/tmp/sonic_eval/parquet_motion_pkl/smpl/episode_000000.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=['episode_000000']" \
  "++eval_output_dir=/tmp/isaac_eval_smpl_ep0"
```

注意：sonic_release.yaml要求 `motion_file` 和 `smpl_motion_file` 分别指向G1侧和SMPL侧pkl。所以阶段1的pkl要么单文件含全部，要么分两个文件（按训练config的约定来）。

#### 5.4 端到端sanity check：跑robot encoder链路对照

用同一份阶段1的pkl，去掉SMPL streaming，直接当motion pkl喂给现有的 `stream_motionlib_to_deploy.py`（ZMQ v1 + robot encoder mode），跑一次robot encoder的MPJPE。

预期：
- robot encoder MPJPE 和 smpl encoder MPJPE 量级相近（同一份motion，policy只是encoder不同）
- robot encoder应该更精确（直接喂dof_pos，没有SMPL→G1间接信息损失）
- 两者跑出的MuJoCo轨迹应该visually相似

这步是验证**整个pipeline没有奇怪的bug污染**，跟sim2sim.md里的三大教训对应（body ID prefix、anchor lock、GT污染）。

---

## 6. 风险与未知点（实现前必查）

1. **SMPL坐标系约定**：sonic_release.yaml:74 `smpl_y_up: true`。parquet里的`teleop.smpl_joints`/`teleop.smpl_pose`是数采端SMPL stream落盘的，需用python读一帧确认是Y-up还是Z-up。直接错坐标会导致FK结果全错。
   - 验证方法：读一帧`teleop.smpl_joints[0]`（pelvis），如果Y≈0.9, Z≈0 → Y-up；如果Z≈0.9, Y≈0 → Z-up
   
2. **SMPL pose维度**：parquet是63D（21 joints × 3，不含root），但`Humanoid_Batch.fk_batch`签名要求72D（24 joints × 3）。需要查清楚fk_batch对 `pose_aa[T,72]` 的layout（root3 + 21 body + 0 hands 3+3+3 全零？或root3 + 21 body? 或root_orient_3 + 22 joints × 3？）
   - 验证方法：读 `torch_humanoid_batch.py` 的 `fk_batch` 输入处理，特别是 `pose_aa.shape` 断言

3. **deploy的SMPL encoder是否真训练过**：observation_config.yaml声明mode=2存在，但需要确认release的`model_encoder.onnx`真的有smpl encoder权重。
   - 验证方法：在deploy debug log里搜"SMPL encoder loaded"或类似输出；或者跑空streamer发v3包看deploy是否报"unknown encoder mode"

4. **数据时序对齐**：parquet里 `teleop.smpl_frame_index` 和 `frame_index` 不一定连续（teleop可能跳帧/stale）。用户选了不清理，所以streamer发送时frame_index语义要明确：
   - 用 `teleop.smpl_frame_index` 作为frame_index（IsaacSim eval语义）
   - 还是用 row index（streamer内部连续编号，metrics端跟streamer对齐）
   - 这两种语义对应不同的align-mode；推荐用streamer内部row index，metrics端就能用 `--align-mode source_frame_index` 跟MuJoCo log对齐

5. **smpl_pose=0的stale帧（用户不清理）**：意味着送v3包给deploy的某些帧SMPL pose全零，FK出来dof_pos会跑到T-pose附近，policy可能在这些帧产生奇怪动作。这是用户明确的选择，要在metrics分析时注意区分有效段和stale段。
   - 建议：streamer log每一帧的`smpl_pose_l2_norm`到一个csv，metrics端可以滤掉norm=0的帧

6. **deploy fall-down**：之前sim2sim.md提到"机器人不是处于一个静止的状态，而是一直在起立摔倒"。SMPL encoder路径下，初始blend策略可能要重新调（`--blend-from-stand-frames`、`--reference-motion-align-delay-frames`），因为：
   - SMPL pose的"第0帧"和G1 default standing pose关节配置差异大（teleop开始时人可能在任意姿态）
   - 直接突变到第0帧的retargeted joints会导致MuJoCo G1摔倒
   - 需要从G1 default站立姿态 → 平滑过渡到第0帧retargeted姿态（已有 `_prepend_stand_transition` 逻辑，关键是blend帧数要够长）

---

## 7. 关键文件清单（实现时按此核对）

| 文件 | 关键内容 |
|---|---|
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp:905-1015` | v1/v2/v3字段校验、帧数一致性 |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:615-815` | SMPL anchor从`body_quat_w[0]`、wrist从`joint_pos`indices{23..28} |
| `gear_sonic_deploy/policy/release/observation_config.yaml:74-80` | smpl mode_id=2 |
| `gear_sonic/utils/motion_lib/motion_lib_base.py:1873-1932` | SMPL pkl加载 + fk_batch调用 |
| `gear_sonic/utils/motion_lib/torch_humanoid_batch.py:360-453` | fk_batch实现，返回dof_pos和body_pos |
| `gear_sonic/utils/motion_lib/motion_lib_robot.py:5-9` | Humanoid_Batch实例化 |
| `gear_sonic/envs/manager_env/mdp/observations.py:1567-1588` | `joint_pos_multi_future_select_joints_for_smpl` |
| `gear_sonic/envs/manager_env/mdp/commands.py:1696-1699` | `TrackingCommand.joint_pos_multi_future_for_smpl` |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml:88-105` | SMPL encoder训练config |
| `gear_sonic/eval_agent_trl.py:148-156` | `+use_encoder=smpl` CLI dispatch |
| `tools/sonic_eval/stream_motionlib_to_deploy.py` | streamer骨架可照抄 |
| `tools/sonic_eval/compute_mujoco_tracking_metrics.py:469-558` | `--gt-format motionlib`加载GT |
| `tools/sonic_eval/parquet_to_mujoco_motion.py` | parquet读取 + 关节顺序映射可参考 |
| `gear_sonic/scripts/run_data_exporter.py:410-494, 667` | 数采端SMPL落盘逻辑，反向参考 |

---

## 8. 端到端验证设想（实现完成后怎么验）

1. **第一步：单纯验证deploy能识别v3包**
   - 起Terminal A (MuJoCo) + Terminal B (deploy)
   - 跑Terminal C发一个minimal v3 dummy包（用 `teleop.body_quat_w` 全identity、`teleop.smpl_joints` 全零、`teleop.smpl_pose` 全零、joint_pos用G1 default）
   - 检查deploy日志：应该有 "active_protocol_version_=3" 或 "encoder mode=2"
   - 检查MuJoCo G1是否不摔倒、policy输出是否合理

2. **第二步：用真实parquet数据，先不做FK retargeting**
   - 用 `observation.state[:29]` 直接作为 joint_pos，跑一遍看policy能不能跟（这等同于"robot encoder路径走人体数据的robot端"，作为baseline）
   - 然后切回FK retargeted joints，对比metrics差异
   - 这一步可验证FK pipeline是否正确

3. **第三步：metrics对比**
   - 跑 `compute_mujoco_tracking_metrics.py --gt-format motionlib --motion-file <converted_smpl.pkl> --actual-source step_sync_body_pos_w_14 --align-mode source_frame_index`
   - 检查输出JSON的 MPJPE_g/l/pa 量级（参考sim2sim.md，应在几十到几百毫米区间）
   - 如果数值过小（个位数mm），说明GT被污染（按sim2sim.md的经验教训自查anchor lock、body_id前缀、deploy-corrected target三大bug）

4. **第四步：IsaacSim eval交叉验证**
   - 同样的converted_smpl.pkl，跑 `eval_agent_trl.py +use_encoder=smpl`
   - 对比IsaacSim eval和MuJoCo sim2sim的MPJPE
   - 一致性差应该在合理范围（仿真器本身差异 + control delay/PD差异）
   - 这一步是验证整条SMPL encoder链路在两个仿真器下行为对齐

---

## 9. 本次会话的代码改动清单（changelog）

会话日期：2026-05-12。目标：用sample_data的官方paired pkl走通SMPL encoder链路（仅基础设施层，retargeting tool依赖未实现）。

### 9.1 新增文件

| 文件路径 | 角色 | 行数 | 说明 |
|---|---|---|---|
| `tools/sonic_eval/stream_motionlib_smpl_to_deploy.py` | streamer（新建sibling） | ~310 | ZMQ Protocol v3 streamer，加载robot+smpl配对pkl，触发deploy encoder_mode=2 |
| `sim2sim_human_encoder_plan.md` | 方案文档备份 | 442+ | 本文档；从 `/home/lab/.claude/plans/sim2sim-error-robot-motion-encoder-huma-compiled-kazoo.md` 拷贝过来，并追加本changelog |

### 9.2 修改文件

| 文件路径 | 改动位置 | 改动内容 |
|---|---|---|
| `sim2sim.md` | 第786行 "## mujoco 运行指令（单文件处理模式, human motion encoder输入）" 段下方 | 新增完整的human motion encoder单文件运行指令（终端A/B/C/D），含与robot encoder链路的对照实验设计 |

### 9.3 关键复用（零侵入式扩展）

新streamer通过以下import直接复用现有helper，避免逻辑重复：

```python
from tools.sonic_eval.stream_motionlib_to_deploy import (
    HEADER_SIZE,
    _finite_difference,
    _prepend_stand_transition,
)
from tools.sonic_eval.motionlib_provider import load_motionlib_sequence
```

| 复用资产 | 来自 | 用途 |
|---|---|---|
| `HEADER_SIZE` (=1280) | `stream_motionlib_to_deploy.py:22` | ZMQ头padding对齐 |
| `_finite_difference()` | `stream_motionlib_to_deploy.py:189` | dof_vel有限差分 |
| `_prepend_stand_transition()` | `stream_motionlib_to_deploy.py:199` | 站立姿态blend prefix |
| `load_motionlib_sequence()` | `motionlib_provider.py:422` | 加载robot pkl（走IsaacSim TrackingCommand offline或Humanoid_Batch fallback） |

### 9.4 未触碰的代码（重要）

为保证robot encoder链路完全不受影响，全部走"零侵入式扩展"：

| 资产 | 改动 | 原因 |
|---|---|---|
| `tools/sonic_eval/stream_motionlib_to_deploy.py` | **0改动** | 仍硬编码ZMQ v=1，robot encoder链路不变 |
| `tools/sonic_eval/compute_mujoco_tracking_metrics.py` | **0改动** | encoder-agnostic，直接复用 `--gt-format motionlib` |
| `tools/sonic_eval/motionlib_provider.py` | **0改动** | robot侧加载逻辑直接复用 |
| `gear_sonic_deploy/` 整目录 | **0改动** | deploy本身根据protocol版本自动切encoder mode |
| `gear_sonic_deploy/policy/release/observation_config.yaml` | **0改动** | smpl mode_id=2已在release config中enabled |
| `gear_sonic/utils/mujoco_sim/base_sim.py` | **0改动** | MuJoCo sim侧encoder-agnostic |

### 9.5 验证步骤（实际跑端到端时按此核对）

1. 起 `run_sim_loop.py`（终端A，见sim2sim.md的"human motion encoder输入"小节）
2. 起 `deploy.sh ... sim`（终端B，跟robot encoder完全一样）
3. 起 `stream_motionlib_smpl_to_deploy.py --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl --smpl-motion-file sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl ...`（终端C，**关键**）
4. 在终端B的deploy日志里搜：
   - `Protocol version 3` 或 `active_protocol_version_=3`：确认收到v3包
   - `encoder_mode=2`：确认切到SMPL encoder
5. 跑 `compute_mujoco_tracking_metrics.py --gt-format motionlib ...`（终端D）输出MPJPE JSON
6. 对照实验：同一份motion再跑一次robot encoder（用 `stream_motionlib_to_deploy.py`，不同 `--logs-dir`），对比两份MPJPE的MPJPE_g/l/pa量级

### 9.6 仍未做的部分（后续TODO）

按用户在plan讨论中的决策，本次会话仅完成"用sample_data走通基础设施层"。以下未做：

| 缺失项 | 描述 | plan章节 |
|---|---|---|
| GMR retargeter integration | parquet里raw SMPL → G1 fitted pose_aa | 4.5.2-4.5.3 |
| `parquet_to_smpl_pkl.py` | parquet字段重组成smpl_filtered格式 | 5.0.A |
| `parquet_to_robot_pkl.py` | 调GMR输出robot_filtered格式 | 5.0.B |
| IsaacSim eval recipe验证 | 用sample_data跑一次 `eval_agent_trl.py +use_encoder=smpl` 看是否能起来 | 5.3 |
| 端到端实跑+metrics量级核对 | 见9.5 | 8 |

按下次工作时优先级：先实跑9.5验证整条链路通了，再做GMR integration把parquet数据接入。

---

## 10. Bug 修复记录（2026-05-12 会话续）

### Bug：脚横着走 / 扭曲（feet walking sideways）

#### 现象
用户跑通基础链路后，MuJoCo 中 actual G1 跟 reference 动作差异巨大——脚横着走、姿态扭曲。

#### 根因（frame 语义错位）

通过对比 deploy C++ gather 函数和 IsaacSim 训练观测函数，发现 streamer 侧少了**两步训练时存在的预处理**：

| 维度 | 训练侧 (`gear_sonic/envs/manager_env/mdp/{commands,observations}.py`) | deploy侧 (`g1_deploy_onnx_ref.cpp:615-694, 909-966`) | 修复前 streamer 实际发送 |
|---|---|---|---|
| SMPL anchor orientation | `command.smpl_root_quat_w` = `remove_smpl_base_rot(Y→Z(quat(pose_aa[:,:3])))` | 直接读 ZMQ `body_quat_w[0]`，C++ 不做转换 | G1 robot 的 root_quat（错） |
| SMPL joints obs | `smpl_joints_multi_future_local`: `quat_apply(quat_inv(smpl_root_quat_w), pkl_joints)` per-frame | 直接读 ZMQ `smpl_joints`，C++ 不做转换 | pkl 原始 body-canonical 值（错） |

数据语义证据（实跑 inspect）：
- `sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl` 的 `smpl_joints[:, 0, :]`（pelvis）在所有 2002 帧完全静止（span=0.000），证明 pkl 里的 smpl_joints 是 body-canonical frame（root rotation/translation 都没 apply）
- `pose_aa[:, :3]` 跨帧变化 + `transl` 跨帧变化且 Y≈1.26m 是 Y-up 高度，证明 pose_aa/transl 是 Y-up 世界系
- `sonic_release.yaml:74` 配置 `smpl_y_up: true` 触发训练侧的 Y→Z 转换；deploy 侧不做这个转换，所以 streamer 必须做

修复前 90° 朝向 + SMPL T-pose base rot offset 的复合误差导致 encoder 收到的 anchor orientation 完全错位，policy 把"前进方向"投影到错坐标系下 → 脚横着走。

#### 修复

[tools/sonic_eval/stream_motionlib_smpl_to_deploy.py](tools/sonic_eval/stream_motionlib_smpl_to_deploy.py) 新增三个 CLI flag + 默认参数对齐 IsaacSim：

- `--smpl-y-up` (默认 True)：把 SMPL pkl 当 Y-up 处理
- `--smpl-anchor-mode smpl_processed` (默认)：`body_quat_w` 字段填 `remove_smpl_base_rot(Y→Z(quat(pose_aa[:,:3])))` per-frame，复刻 `TrackingCommand.smpl_root_quat_w`
- `--smpl-joints-mode canonicalized` (默认)：`smpl_joints` 字段填 `quat_apply(quat_inv(smpl_root_quat_w), pkl_joints)` per-frame，复刻 `smpl_joints_multi_future_local`

实现的新工具函数：`_smpl_root_ytoz_up`、`_remove_smpl_base_rot_wxyz`、`_compute_smpl_root_quat_w`、`_canonicalize_smpl_joints`、`_quat_apply_wxyz`、`_quat_mul_wxyz`、`_quat_conjugate_wxyz`。直接对应 `gear_sonic/isaac_utils/rotations.py:704, 711-719` 和 `gear_sonic/envs/manager_env/mdp/commands.py:1267-1282, 1308-1322` 里的训练侧实现。

修复后默认行为已经对齐 IsaacSim eval 语义，不需要新增 CLI flag。保留旧行为做 A/B 测试：`--smpl-anchor-mode robot_root --smpl-joints-mode raw`（这是修复前的 default）。

[sim2sim.md:848](sim2sim.md#L848) 的说明段已同步更新，加了"frame语义对齐（必读）"小节解释这两步预处理为什么必须做。

#### 经验教训

跨进程异步 streaming 系统下的 encoder observation 对齐，要**逐 observation 端到端追**：
- 训练侧 observation function 是什么
- 它从 motion_lib 取的什么字段
- 这个字段在 pkl 里是什么 frame
- 训练侧 observation function 做了什么变换才喂给 encoder
- deploy 侧的 C++ gather 函数读的是哪个 ZMQ 字段，做不做变换
- streamer 端必须把训练侧"observation function 做的变换"+pkl 原始值的差值补上

任何一步遗漏都会导致 encoder 拿到错误的输入空间，policy 输出无意义。这次的 bug 印证了 sim2sim.md 早先的总结：「IsaacSim 里的对齐是天然存在的，sim2sim 里的对齐是后天重建的」。

---

## 11. Bug 修复迭代二（2026-05-13）：ref viz 朝向与 actual G1 对齐

### 11.1 现象与诊断

迭代一修复后 policy 工作正常（feet 不再横着走），但用户反馈 **ref G1 pelvis 朝向跟 actual G1 有 1-3° 的恒定 pitch/roll 偏差**——位置正常但视觉上 ref 是"歪的"。

经数值验证：
- 我的 `_compute_smpl_root_quat_w` 实现跟 IsaacSim 训练侧 `command.smpl_root_quat_w` **完全一致**（max abs diff = 1.19e-07，浮点精度）
- quat 约定一致（都是 wxyz）：robot encoder 的 `seq.root_quat_w` 是 wxyz，我的 SMPL processed root 也是 wxyz
- 但 `smpl_processed_root` 和 `G1_root_quat`（motionlib 从 robot_filtered.pkl 读出来）**逐帧都差 ~2-3°**（点积 0.99977，主要在 pitch/roll，yaw 只差 0.3°）

差异本质：
- `remove_smpl_base_rot` 用的是**所有 SMPL 数据通用**的 `[0.5,0.5,0.5,0.5]` 基准（SMPL T-pose convention）
- `G1 root_quat` 是 SOMA Retargeter 给每段动作独立 fit 的 G1 rest pose alignment
- 这两套 convention 描述同一个物理身体的朝向但有约 2-3° 残差，**配对数据下也无法消除**

### 11.2 为什么 sim 端 anchor 不能修复

`base_sim.py` 现有 `_set_latest_pose` + `_transform_reference_root_pose`：
- 只锁定 `(ref_anchor_xy, actual_anchor_xy, yaw_delta)` 三元组
- 应用时只调 yaw + XY 位移
- **pitch/roll 直接用 streamed 值，不做修正**

所以 streamer 喂进去什么 pitch/roll，ref viz 就显示什么 pitch/roll。`smpl_processed_root` 带的那 ~2-3° pitch/roll 偏差就这样穿透到了 ref viz 上。

### 11.3 解决方案：streamer 端用 G1 root_quat（"复用 sim 现有 anchor"）

robot encoder 链路视觉对齐工作的关键不是"它在 sim 端做了 3D 对齐"，而是 **streamer 喂的 `body_quat_w` 的 pitch/roll 就是 policy 想要 actual G1 摆出来的 pitch/roll**（同源 G1 motion 数据）。sim 只补一个起始 yaw+XY 的偏移就够了。

SMPL encoder 链路要做到一样视觉效果，就让 streamer 也喂 G1 root_quat：

```python
# stream_motionlib_smpl_to_deploy.py 默认改成
--smpl-anchor-mode robot_root   # body_quat_w = seq.root_quat_w (G1 motion 的 root quat)
--smpl-joints-mode canonicalized # canonicalize 用同一个 root (强制耦合，保证 encoder 两个 SMPL 观测内部一致)
```

**唯一代价**：encoder 看到的 anchor_orientation obs 比训练分布偏 ~2-3°。但训练时 reward 把 actual_base 推向 ≈ G1_root_quat（不是精确等于 smpl_processed_root），所以 trained policy 本来就在 robust 处理这 2-3° 差异。把扰动直接喂给 encoder 应该不会破坏 tracking。

### 11.4 配套：sim 端 anchor 锁定时机改成 auto-detect（blend 最后一帧）

`base_sim.py:178-180, 442-451` 早就有 auto-detect 机制：sim 启动时若传 `--reference-motion-align-delay-frames 0`，进入 auto 模式，从第一个 ZMQ 包 header 读 `motion_start_frame`（streamer 写入 = `prepend_stand_frames + blend_from_stand_frames`），把 anchor lock 时机设到正好 blend 阶段最后一帧。

之前 sim2sim.md 示例硬编码 `--reference-motion-align-delay-frames 30`，在 blend 200 帧场景下会锁在过渡半路（15% 进度）。改成 `0` 让 auto-detect 接管，**精确锁在 blend 最后一帧 = actual G1 从 standing pose 平滑过渡到第一帧 motion 姿态完成的那一瞬间**。这是"在插值的最后一帧做一次性 fixed 数值对齐"最严格的实现。

### 11.5 改动清单

| 文件 | 改动 |
|---|---|
| `tools/sonic_eval/stream_motionlib_smpl_to_deploy.py` | (1) `--smpl-anchor-mode` 默认 `smpl_processed` → `robot_root` (2) 重构 `main()`：`reference_root_quat` 由 anchor mode 选择，**canonicalize root 强制和 anchor 用同一个**（内部一致性）(3) 更新 CLI help 说明设计取舍 |
| `sim2sim.md` | (1) MuJoCo 启动示例 `--reference-motion-align-delay-frames 30` → `0` (2) frame语义对齐小节重写：突出新默认 `robot_root` 的"复用 robot encoder 链路对齐机制"的设计意图 (3) 加 auto-detect log 验证方法 |
| `sim2sim_human_encoder_plan.md` | 追加本第 11 节 |

零修改：sim 端代码（`base_sim.py`、`link_error_plot.py`、`run_sim_loop.py`）、deploy 端、robot encoder streamer (`stream_motionlib_to_deploy.py`)、metrics 工具。

### 11.6 验证计划

跑修改后的链路：

```bash
# 终端A (MuJoCo): 注意 --reference-motion-align-delay-frames 0
python gear_sonic/scripts/run_sim_loop.py \
    --interface sim --simulator mujoco --env-name default \
    --reference-motion-align-delay-frames 0

# 终端B (deploy): 不变

# 终端C (streamer): 用新默认 (anchor_mode=robot_root + canonicalize 同源)
python tools/sonic_eval/stream_motionlib_smpl_to_deploy.py \
    --motion-file sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl \
    --smpl-motion-file sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl \
    --motion-name walk_forward_amateur_001__A001 \
    --host 127.0.0.1 --port 5596 --target-fps 50 \
    --initial-burst-frames 20 --blend-from-stand-frames 200 \
    --chunk-size 30 --realtime --send-command --use-isaacsim-app
```

验证项：
1. **MuJoCo 终端输出** `[ReferenceMotionVisualizer] auto align_delay_frames=220 (motion_start_frame=220 from stream)`，证明 auto-detect 起作用、锁在 blend 最后一帧
2. **视觉**：ref G1 pelvis 朝向（pitch/roll/yaw）跟 actual G1 完全对齐，无可见偏差
3. **policy 仍正常 tracking**：feet 走路正常、姿态不扭曲
4. **metrics**：跑 `compute_mujoco_tracking_metrics.py`，MPJPE 量级在几十～几百毫米合理区间
5. **A/B 对照**：再跑一次 `--smpl-anchor-mode smpl_processed`，看两份 metrics 是否量级相近
   - 若两者 MPJPE 接近（< 10% 差异）→ `robot_root` 方案成立，可作为生产默认
   - 若 `smpl_processed` 明显更准 → 说明 encoder 对 anchor obs 2-3° 扰动不够 robust，回滚 default 并升级到方案 A（base_sim.py 加新 3D anchor mode，off by default）

### 11.7 风险与回滚

风险：encoder 对 anchor obs 2-3° 扰动的鲁棒性是个 unverified 假设。若实测发现 policy 退化明显（MPJPE 翻倍以上），方案需要升级为：

**Plan B**（仅在 Plan A 失败时启用）：
- base_sim.py 加新 anchor mode `full3d_align`，**默认 off**（robot encoder 完全不受影响）
- SMPL 链路启用 `full3d_align`：用 frame 0 (or blend 最后一帧) 时的 `actual_G1_quat` 跟 `smpl_processed_root` 做一次性 3D 旋转 offset 锁定
- 之后每一帧 ref pose 都用该 offset 做 full quat 变换（不只是 yaw）
- 这样 encoder 仍然看 smpl_processed_root（训练分布精确匹配），但 ref viz 在 sim 端做 3D 校正后跟 actual G1 完全对齐
- 此方案代价：base_sim.py 加新代码分支（用户 OK，因为不改 robot encoder 默认）

回滚命令：`--smpl-anchor-mode smpl_processed`（CLI 一键切回）。

---

## 12. 修订记录（2026-05-18）：澄清 deploy 推理链路无需 retarget

### 12.1 动机

用户在 review 4.5 / 5.0.B 节时指出：parquet 里 `teleop.smpl_*` 是 human motion 数据，重放走 human motion encoder 推理时是否真的必须先 retarget 成 G1 robot pkl？应当对齐 pico 真机摇操的处理方式。

### 12.2 关键代码证据

逐行追完 pico → deploy → parquet 三段链路后确认：

1. **pico sender 内联做 wrist retarget**（[pico_manager_thread_server.py:1346-1403](gear_sonic/scripts/pico_manager_thread_server.py#L1346-L1403)）：
   - 取 SMPL elbow(17,18) + wrist(19,20) axis-angle
   - `decompose_rotation_aa()` 分解 swing/twist
   - 合成 6 个 G1 wrist DOF（roll/pitch/yaw × 左右）
   - 直接写入 `joint_pos[23..28]`
   - 注意左右非对称的 sign 处理（左 pitch 二次翻转、右 roll 整体翻转）—— 是 G1 硬件 axis convention 镜像，已在 sender 端正确处理
   - **不调 `Humanoid_Batch.fk_batch`，是解析式 Euler 合成**

2. **ZMQ pose 消息发出 `joint_pos[29]`**（[pico_manager_thread_server.py:1442, 1468-1469](gear_sonic/scripts/pico_manager_thread_server.py#L1442)）

3. **deploy 端 SMPL encoder 直接消费**：从 ZMQ `joint_pos` 切 indices `{23..28}` 作为 `motion_joint_positions_wrists_10frame_step1` 观测，**deploy 端无任何 retarget**

4. **data_exporter 镜像落盘**（[run_data_exporter.py:410-447](gear_sonic/scripts/run_data_exporter.py#L410-L447)）：
   - 从 pose_data["joint_pos"] 切 `[23,25,27]` → `teleop.left_wrist_joints (3,)`
   - 从 pose_data["joint_pos"] 切 `[24,26,28]` → `teleop.right_wrist_joints (3,)`
   - **parquet 里这两个字段就是 deploy 当时收到的 G1 wrist DOF**，零加工

### 12.3 推论

| 用途 | retarget 需求 | pkl 需求 |
|---|---|---|
| parquet → deploy SMPL encoder 推理（本任务） | 否（parquet 已含 G1 wrist DOF） | 否 |
| Metrics 14-link GT | 是（除非用 `observation.state` 当 GT） | 是 |
| IsaacSim eval `+use_encoder=smpl` | 是 | 是 |

### 12.4 文档改动

| 章节 | 改动 |
|---|---|
| 0（新增） | "关键澄清" 章节，明确两条互斥链路、本任务的最小路径 |
| 4.5（修订标题 + 加 caveat） | 标题改为 "外部 retargeter（仅 IsaacSim eval / metrics GT 路径需要）"，开头加导引段 |
| 5.0.deploy-only（新增小节） | 字段映射表：parquet 字段 → ZMQ v3 字段，零 retarget |
| 5.1.deploy-only（新增小节） | 新 streamer `stream_parquet_smpl_to_deploy.py` 的设计 |
| 阶段 1 / 5.1.eval-also / 5.2 / 5.3（加 caveat） | 标注 "仅 5.0.eval-also 路径适用"，保留原内容 |

### 12.5 零侵入式扩展原则

按用户要求，新增功能不动现有代码：

| 资产 | 改动 |
|---|---|
| `tools/sonic_eval/stream_parquet_smpl_to_deploy.py` | **新建**（5.1.deploy-only 路径） |
| `tools/sonic_eval/stream_motionlib_smpl_to_deploy.py` | **0 改动**（5.1.eval-also 路径继续走 pkl） |
| `tools/sonic_eval/stream_motionlib_to_deploy.py` | **0 改动**（robot encoder 链路） |
| `gear_sonic/scripts/pico_manager_thread_server.py` | **0 改动**（真机链路） |
| `gear_sonic/scripts/run_data_exporter.py` | **0 改动**（数采链路） |
| `gear_sonic_deploy/` | **0 改动** |
| `gear_sonic/utils/mujoco_sim/base_sim.py` | **0 改动** |

新 streamer 通过 import 复用现有 helper，不修改任何被 import 的源文件。

---

## 13. Bug 修复（2026-05-19）：parquet streamer 双重 canonicalization

### 13.1 现象

用户跑 `stream_parquet_smpl_to_deploy.py` 重放多条 motion（`episode_000003`、`episode_000192`、`episode_000041`），policy 跟踪行为扭曲：脚和手都怪异，跟原 SMPL 动作明显不像。

### 13.2 根因：parquet 和 pkl 的 smpl_joints 语义**相反**

追完 pico → parquet → streamer → deploy 的完整数据流，对照 IsaacSim 训练侧 [observations.py:smpl_joints_multi_future_local](gear_sonic/envs/manager_env/mdp/observations.py#L1716-L1745)：

| 链路 | smpl_joints 内容 | pelvis 验证 |
|---|---|---|
| **pkl smpl_filtered** | FK 原始输出，**未** apply root rotation | pelvis 永远 = `J[0] = (0.003, -0.351, 0.012)`，std ≈ 0 |
| **parquet teleop** | [pico_manager_thread_server.py:476-477](gear_sonic/scripts/pico_manager_thread_server.py#L476-L477) 已 apply `quat_apply(quat_inv(processed_root), FK_output)` | pelvis 随帧变化，std ~ 0.06 |

empirical 验证（用 `data_0424/.../episode_000000.parquet` + `sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl`）：
- pkl pelvis range: 跨 2002 帧静止于 `(0.003, -0.351, 0.012)`，std `[2e-8, 6e-6, 1e-7]`
- parquet pelvis range: `(-0.314, -0.300, -0.111)` ~ `(0, 0, 0.016)`，std `[0.07, 0.06, 0.04]`

进一步对照 `human_joints_info.pkl`：pkl pelvis 数值 = `J[0]` 完全一致，证实 pkl smpl_joints 就是 FK 不带 rotation 的输出。parquet pelvis 偏离且变化，证实 R^-1 已 apply。

训练侧 `smpl_joints_multi_future_local` 做：
```python
ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)  # ref_joints from motion_lib (= pkl smpl_joints)
                = R^-1 * FK_output                                  # encoder 看到的分布
```

### 13.3 修复前的 streamer 默认行为（错）

[stream_parquet_smpl_to_deploy.py 修复前 250-254 行](tools/sonic_eval/stream_parquet_smpl_to_deploy.py#L250-L254)：

```python
if args.smpl_joints_mode == "canonicalized":   # ← 默认 = canonicalized
    smpl_joints = _canonicalize_smpl_joints(data["smpl_joints"], reference_root_quat)
# 实际执行:
#   = quat_apply(quat_inv(R_pico), parquet_smpl_joints)
#   = quat_apply(quat_inv(R_pico), quat_apply(quat_inv(R_pico), FK_output))
#   = R_pico^-2 * FK_output                            ← 双重旋转，错位 R_pico^-1
```

`_canonicalize_smpl_joints` 来自 [stream_motionlib_smpl_to_deploy.py](tools/sonic_eval/stream_motionlib_smpl_to_deploy.py)。pkl 链路那边数据**未** canonicalize，streamer 端补这步是对的；parquet 这边 pico 已经做了，再做就是 bug。

### 13.4 修复

[stream_parquet_smpl_to_deploy.py](tools/sonic_eval/stream_parquet_smpl_to_deploy.py)：
- `--smpl-joints-mode` 选项从 `{canonicalized, raw}` 改成 `{passthrough, re_canonicalize}`
- 默认从 `canonicalized` 改成 `passthrough`
- `passthrough`：直接传 `teleop.smpl_joints`（已是 `R^-1 * FK_output`，匹配训练分布）
- `re_canonicalize`：仅诊断/A-B 对照，复现 pre-fix bug 行为

[sim2sim.md](sim2sim.md) 终端C 注释同步更新，明确两条链路 smpl_joints 处理方式相反的设计意图。

### 13.5 零侵入

| 资产 | 改动 |
|---|---|
| `tools/sonic_eval/stream_parquet_smpl_to_deploy.py` | 仅改 CLI choices + default + 一个分支 if 条件 |
| `tools/sonic_eval/stream_motionlib_smpl_to_deploy.py` | **0 改动**（pkl 链路 canonicalize 是对的） |
| `gear_sonic/scripts/pico_manager_thread_server.py` | **0 改动** |
| 其余 deploy / data_exporter / motion_lib / sim 端 | **0 改动** |

### 13.6 经验教训

跟第 10 节 (pkl streamer feet-walking-sideways) 是一个镜像问题：

| | pre-fix bug | 根因 |
|---|---|---|
| **第 10 节（pkl 链路）** | streamer 没做 canonicalize | pkl 数据是 FK_output，需要 streamer 补 quat_inv |
| **第 13 节（parquet 链路）** | streamer 默认做了 canonicalize | parquet 数据 pico 已经 apply quat_inv，streamer 不能再做 |

两个 bug 都源于"想当然地复用 helper 函数"，没有追到数据源头确认 frame 语义。下次新增 streamer：**先 dump 一帧数据**，对比训练侧 motion_lib 加载的 pkl 一帧，逐字段确认 frame 语义后再决定是否要 transform。

empirical 验证用的脚本（仅 dump，不动数据）：
```python
import joblib, numpy as np, pandas as pd, torch
J = torch.load('gear_sonic/data/human/human_joints_info.pkl', weights_only=False)['J']
pkl = joblib.load('sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl')
df = pd.read_parquet('data_0424/data/chunk-000/episode_000000.parquet')
pkl_j = np.asarray(pkl['smpl_joints'])
pq_j = np.stack([np.asarray(v) for v in df['teleop.smpl_joints'].to_numpy()]).reshape(-1,24,3)
print('SMPL rest J[0]:', J[0].numpy())          # (0.003, -0.351, 0.012)
print('pkl pelvis range:', pkl_j[:,0,:].std(0)) # [~0, ~0, ~0]  → 未 canonicalize
print('parquet pelvis range:', pq_j[:,0,:].std(0)) # [0.07, 0.06, 0.04] → 已 canonicalize
```


