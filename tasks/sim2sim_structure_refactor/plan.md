# sim2sim 结构优化计划

更新时间：2026-06-24

文档位置：`tasks/sim2sim_structure_refactor/plan.md`

配套工作流文档：

- `tasks/sim2sim_structure_refactor/status.md`：当前阶段状态、门禁结论、下一步。
- `tasks/sim2sim_structure_refactor/test_matrix.md`：各阶段必须执行的测试矩阵和通过标准。
- `tasks/sim2sim_structure_refactor/log.md`：阶段执行日志、命令、环境、结果摘要。
- `tasks/sim2sim_structure_refactor/artifacts/`：小体积报告、索引或保留说明；大体积原始日志仍放 `tmp/`。

## -1. 最关键执行门禁

本节优先级高于后续所有阶段计划，整个结构优化过程中必须持续遵守。

- 每一个阶段开始前，必须明确本阶段目标、改动范围、测试范围和退出标准。
- 每一个阶段开始前，必须在 `test_matrix.md` 或本阶段记录中明确本阶段“挑选测试数据清单”，包括数据路径、motion name、数据类型、是否需要端到端运行、是否只做 deterministic/manifest 验证。
- 每一个阶段完成代码优化后，必须跑完该阶段计划内的所有测试，并且所有测试都必须成功完成；测试命令、环境、结果必须写入本文档。
- 每一个阶段必须对本阶段挑选测试数据清单中的所有数据完成计划内的完整测试运行，不能只跑其中一部分、不能只记录成功样例、不能把单条 smoke 结果当作全数据结果。
- 如果某阶段因为时间、显存、机器资源或环境限制不能对某类数据执行完整端到端运行，必须在阶段开始前明确降级为哪一类验证（例如 deterministic replay、streamer manifest、adapter/manifest smoke），说明原因和风险，并得到用户确认；否则该阶段不能标记为完成。
- 只有确认本阶段结构优化对 sim2sim、真机部署、训练/eval、工具入口、日志/metrics、时间帧对齐等既有功能没有影响后，才能进入下一个阶段。
- 如果任一测试失败、结果不确定、环境缺失、性能数据不足、C++ 风险未闭环，或无法确认“对任何既有功能没有影响”，必须停止进入下一阶段，并向用户报告：
  - 具体失败项或不确定项。
  - 初步原因分析。
  - 建议解决方案。
  - 需要补充的测试或数据。
- 每一个 phase 的全部计划内测试都成功完成、确认没有问题、阶段结果已经写入文档并完成提交/push 后，可以清理该阶段测试产生的大体积临时数据，避免长期占用磁盘空间。
- 清理测试数据前必须确认：
  - 已保留必要的小体积 summary、metrics JSON 摘要、失败原因说明或复现实验索引。
  - 文档中已经记录原始数据路径、清理时间、清理范围和保留内容。
  - 本阶段不再需要原始日志做复查；如后续仍可能需要对比，只能压缩或保留，不直接删除。
- 进入下一个阶段之前，必须把本阶段代码提交并 push 一版。
- commit message 必须详细、具体，说明：
  - 本阶段目标。
  - 改动的结构边界。
  - 是否修改 C++。
  - 已跑的测试和结果。
- 只允许 push 到 `origin`，且 `origin` 必须是：

```text
git@github.com:qs778899-lab/GR00T-WholeBodyControl.git
```

- 禁止 push 到 `upstream2`、`upstream3` 或其他远程。
- 如果某阶段测试不完整、环境缺失、性能数据不足、C++ 风险未解释清楚，或无法证明结构优化没有影响既有功能，不能进入下一阶段，也不能把该阶段标记为完成。
- 所有计划，包括 C++ 去侵入方案，只维护在 `tasks/sim2sim_structure_refactor/` 下的工作流文档中；旧路径 `workspace/plan_/sim2sim_structure_refactor_plan.md` 不再维护。
- `tmp/` 只放验证数据、manifest、原始日志和阶段报告；阶段确认完成后按本文门禁清理或压缩大体积数据。

## 0. 目标与约束

目标：整理当前 `GR00T-WholeBodyControl` 中 sim2sim 相关实现，使其后续可以低侵入合入 base 仓库，保持结构清晰、职责边界明确，并尽量不影响真机部署和其他核心功能。

base 仓库定义：

- base 仓库 = `/home/lab/Desktop/LHM-Robot/S0` 在 `/home/lab/Desktop/LHM-Robot` 仓库的 `feat/s0_training` 分支内容。
- 后续所有“新增 / 修改 / 缺失 / 差异”均以当前 `GR00T-WholeBodyControl` 相比 base 仓库为准。

硬约束：

- 先计划，确认后再执行代码结构优化。
- 不破坏现有 sim2sim 行为逻辑，尤其是 reference visualization、step-sync logging、metrics 时间对齐语义。
- 尽量不改 C++ 部署核心代码；如果必须依赖 C++ 已有能力，要把依赖边界写清楚，优先通过独立工具、配置、日志旁路实现。
- 相比 base 仓库纯新增的文件，暂时可以先不做修改/结构优化；只有当新增文件迫使 base 已有核心文件发生差异时，才回头整理它们的依赖边界。
- 最关键优先级：
  1. C++ 文件尽量不要和 base 仓库不同。
  2. base 仓库已有、当前也存在但代码块不同的文件，必须重点关注并做结构优化。
- 每次结构调整后都要记录测试结果、关键指标、差异解释。
- 不能用“逐字节一致”作为唯一标准；以同类 metrics、帧数、source frame 覆盖率、日志完整性和可视化行为为主。
- 所有验证需要的数据、临时日志、manifest、对比报告统一放在项目根目录下的 `tmp/` 文件夹中，不写入源码目录或外部散落路径。

非目标：

- 不重新设计 policy / encoder。
- 不改训练侧 motion lib 语义。
- 不把 sim2sim 调试功能混入真机运行默认路径。

## 1. 已知现状

主要说明文档：

- `workspace/notes/sim2sim.md`
- `workspace/plan_/sim2sim_human_encoder_plan.md`

当前 sim2sim 主链路：

1. Terminal A：`gear_sonic/scripts/run_sim_loop.py` 启动 MuJoCo。
2. Terminal B：`gear_sonic_deploy/deploy.sh ... --input-type zmq_manager ...` 启动 deploy。
3. Terminal C：`tools/sonic_eval/stream_motionlib_to_deploy.py` 或 `tools/sonic_eval/stream_*_smpl_to_deploy.py` 发送 motion stream。
4. Terminal D：`tools/sonic_eval/compute_mujoco_tracking_metrics.py` 读取 sim/deploy 日志计算 metrics。

当前关键代码分布：

- `tools/sonic_eval/`
  - motion streamers、metrics、batch eval、parallel eval、parquet recorder。
- `gear_sonic/utils/mujoco_sim/base_sim.py`
  - reference robot visualization。
  - sim2sim body position logging。
  - step-sync `actual/ref/source_frame_index` CSV。
  - viewer overlay。
- `gear_sonic/utils/mujoco_sim/link_error_plot.py`
  - 实时 link error plot 子进程。
- `gear_sonic/utils/mujoco_sim/configs.py`
  - sim2sim CLI/config 参数。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/...`
  - ZMQ protocol、source frame index、debug output、state logger 等部署侧支持。

base 仓库初步情况：

- `/home/lab/Desktop/LHM-Robot` 当前工作区检出分支是 `master`，但本地 `feat/s0_training` 分支存在；本计划以 `git show feat/s0_training:S0/...` 读取 base 内容，不依赖当前 checkout。
- `/home/lab/Desktop/LHM-Robot/S0/eval_benchmark` 在当前 checkout 下不存在；base 分支是否纳入测试数据需后续继续确认。
- S0 已有部分 SMPL / Protocol v3 / motion lib 代码，不适合简单整目录覆盖。

## 1.1 相比 base 仓库的初步差异盘点

盘点口径：

- base：`/home/lab/Desktop/LHM-Robot` 的 `feat/s0_training:S0/`
- current：当前仓库 tracked files，即 `git ls-files`
- 不把 `.venv_sim`、build 产物、未跟踪数据产物作为结构优化依据。

初步结果：

- common tracked files：376
- base 已有且 current 内容不同：64
- current 纯新增 tracked files：2791
- base 有但 current 缺失：90
- base 已有且 current 内容不同的 C++/header 文件：16
- current 纯新增 C++/header tracked files：789，主要集中在 `thirdparty` / `external_dependencies` / `decoupled_wbc` 等，暂不作为 sim2sim 结构优化优先项。

当前相对 base 纯新增、且与 sim2sim 直接相关的文件包括：

- `gear_sonic/utils/mujoco_sim/link_error_plot.py`
- `tools/sonic_eval/compute_eef_accuracy_offline.py`
- `tools/sonic_eval/compute_mujoco_tracking_metrics.py`
- `tools/sonic_eval/inspect_motionlib_encoder_inputs.py`
- `tools/sonic_eval/motionlib_provider.py`
- `tools/sonic_eval/parquet_recorder.py`
- `tools/sonic_eval/parquet_to_mujoco_motion.py`
- `tools/sonic_eval/run_mujoco_batch_eval.sh`
- `tools/sonic_eval/run_mujoco_batch_eval_parallel.sh`
- `tools/sonic_eval/run_mujoco_chain.sh`
- `tools/sonic_eval/run_mujoco_multi_instance_parallel.sh`
- `tools/sonic_eval/stream_motionlib_smpl_to_deploy.py`
- `tools/sonic_eval/stream_motionlib_to_deploy.py`
- `tools/sonic_eval/stream_parquet_smpl_to_deploy.py`
- `tools/sonic_eval/visualize_realtime_error.py`

处理原则：这些纯新增文件暂时不优先重构；重点是减少它们对 base 已有文件的侵入。

base 已有且 current 内容不同、需要重点审查的 sim2sim 相关 Python/配置文件：

- `gear_sonic/utils/mujoco_sim/base_sim.py`
- `gear_sonic/utils/mujoco_sim/configs.py`
- `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`
- `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`
- `gear_sonic/utils/motion_lib/motion_lib_base.py`
- `gear_sonic/scripts/pico_manager_thread_server.py`
- `gear_sonic/scripts/run_data_exporter.py`
- `gear_sonic/data/features_sonic_vla.py`
- `gear_sonic/data/exporter.py`
- `gear_sonic/envs/manager_env/mdp/commands.py`

其中 `base_sim.py` 当前相比 base 的 numstat 约为 `+1456/-9`，说明 current 在 base 文件上堆入了大量 sim2sim 逻辑。后续不能直接以当前文件覆盖 base，必须基于 base 文件做最小 hook。

base 已有且 current 内容不同的 C++/header 文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/gamepad.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/input_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/interface_manager.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/keyboard_handler.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/ros2_input_handler.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/motion_data_reader.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/output_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/state_logger.cpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/test_ros2.cpp`

处理原则：这些 C++ 差异是最高风险项。结构优化目标不是“整理这些 C++ 文件”，而是尽量让最终合入 base 时无需带入这些差异；如果某个 C++ 差异被证明是 sim2sim 必需，要单独列出原因、替代方案、默认关闭方式和性能测试。

## 2. 主要问题拆分

### 2.1 C++ 侵入风险

当前 sim2sim 依赖 deploy 侧 C++ 的能力包括：

- ZMQ Protocol v1/v2/v3 解析。
- `source_frame_index` / `applied_source_frame_index` 的记录与 debug 输出。
- `g1_debug` 里输出参考帧游标、关节、姿态等。
- `StateLogger` 写 deploy 侧 CSV。

风险：

- 如果为了 sim2sim 继续修改部署 C++ 主路径，可能影响真机时延、日志 I/O、控制频率、协议兼容。
- 合入团队分支时，C++ 冲突和部署风险最高。

策略：

- 第一轮结构优化不新增 C++ 功能，并优先尝试让最终迁移不修改 base C++ 文件。
- 对上面列出的 16 个同名 C++ 差异逐个做块级审查：区分 sim2sim 必需、SMPL/protocol 通用能力、非本任务相关改动。
- 对非 sim2sim 必需的 C++ 差异，迁移到 base 时不携带。
- 对 sim2sim 只用于离线评估/调试的能力，优先改为 Python 旁路、独立 recorder、独立 debug subscriber，而不是修改 deploy 主循环。
- 保留 base C++ 行为作为默认路径，通过 Python 工具层读取和验证。
- 后续如发现必须改 C++，先提出单独方案：默认关闭、编译/运行时开关、零开销 fast path、性能对比数据齐全后再考虑。

### 2.1.1 C++ 去侵入结构优化具体方案

目标不是“整理现有 C++ diff 后直接合入”，而是让 base 仓库的真机部署主路径尽量回到 base 行为；sim2sim 只通过默认关闭的旁路能力获取调试/对齐信息。

核心判断：

- sim2sim 需要的 C++ 能力本质只有两类：
  1. 输入侧：知道 streamer 发送的 `frame_index` / `motion_start_frame` / raw pose 字段。
  2. 输出/日志侧：知道 deploy 实际消费到的 `source_frame_index`，用于严格时间对齐。
- 当前 16 个 C++ diff 把上述能力混入了输入接口、状态日志、debug output、键盘/手柄行为、planner/start 行为和默认端口等多个不相关点，合入风险过高。
- 结构优化必须先拆掉“功能耦合”，再决定是否保留极小的 C++ hook。

#### C++ 目标结构

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/
  include/
    input_interface/
      input_interface.hpp              # 保持 base 主接口，不暴露 sim2sim 调试字段
      zmq_endpoint_interface.hpp       # 保持 base 协议解析主路径
      zmq_manager.hpp                  # 保持 base start/planner 行为
    output_interface/
      output_interface.hpp             # 保持 base debug/output schema
      zmq_output_handler.hpp           # 保持 base debug/output schema
    state_logger.hpp                   # 保持 base 日志字段

  # 只有在 Python 旁路无法满足严格对齐时，才新增以下默认关闭模块：
  include/sim2sim_debug/
    source_frame_tracker.hpp           # 只记录 frame_index -> consumed/applied frame
    sim2sim_debug_config.hpp           # 编译/运行时默认 false
    sim2sim_debug_sink.hpp             # CSV/ZMQ sink 抽象，只有启用时创建
  src/sim2sim_debug/
    source_frame_tracker.cpp
    sim2sim_debug_sink.cpp
```

设计原则：

- base 原有接口签名不变，避免所有 `InputInterface` 子类被迫实现 sim2sim 方法。
- sim2sim debug 不进入默认构造路径；默认不开文件、不分配大 buffer、不发 ZMQ 字段。
- 如必须记录 deploy consumed frame，只在 `--enable-sim2sim-debug` 或已有 `--enable-csv-logs` 且明确开启 sim2sim 选项时启用。
- 不修改 keyboard/gamepad/ros2 planner 行为；这类差异全部回退 base。
- 不修改默认 ZMQ port；sim2sim 入口通过 CLI 参数指定端口。
- 不把 battery SOC、BMS topic 等真机日志字段删除；这些与 sim2sim 无关。

#### 16 个同路径 C++ 文件的处理表

| 文件 | 当前差异角色 | 结构优化处理 |
|---|---|---|
| `include/input_interface/input_interface.hpp` | 新增 `GetSourceFrameIndex(...)` 虚接口，侵入所有输入源 | 回退 base。不要在主接口暴露 sim2sim 方法；如必须读取 frame，用 `Sim2SimSourceFrameTracker` 在 ZMQ 专用实现内部旁路记录。 |
| `include/input_interface/interface_manager.hpp` | 转发 source frame，扩大接口影响面 | 回退 base。manager 不感知 sim2sim debug 字段。 |
| `include/input_interface/zmq_manager.hpp` | 同时改 start/planner 行为和 source frame | 拆分。start/planner 行为回退 base；source frame 只允许进入默认关闭 tracker。 |
| `include/input_interface/zmq_endpoint_interface.hpp` | 大块协议/parser/debug 改动，含 body_pos/source frame/日志变化 | 不整块合入。保留 base parser；raw pose/body_pos 优先由 Python packed stream 旁路读取。如果 C++ 必需，只增加最小字段解析并 gated。 |
| `include/input_interface/streamed_motion_merger.hpp` | SMPL/protocol merge 行为变化 | 不纳入 sim2sim 结构优化；单独作为 SMPL protocol 需求评审。 |
| `include/motion_data_reader.hpp` | 增加 source_frame storage/accessor | 回退 base；如果必须保留，迁入 `sim2sim_debug/source_frame_tracker`，不污染 reader 公共语义。 |
| `include/state_logger.hpp` | 新增 source frame CSV，同时删除 battery sink | 回退 base。sim2sim CSV 写入放到 `Sim2SimDebugSink`，且不能删除原有 battery 字段。 |
| `src/state_logger.cpp` | 写 source frame CSV，同时删除 battery SOC 写入 | 回退 base。必要时新增独立 sink 文件，不修改 StateLogger 默认字段。 |
| `include/output_interface/zmq_output_handler.hpp` | 修改 `g1_debug` schema，加入 source/applied frame | 默认回退 base schema。若 Python 无法旁路获取 consumed frame，可新增 `sim2sim_debug` topic 或 schema-version 字段，默认关闭。 |
| `src/g1_deploy_onnx_ref.cpp` | 混合 SMPL future-frame、source frame logging、默认端口、logger 变化 | 不整块合入。拆成独立 patch：端口回退、logger 回退、SMPL 行为单独评审、sim2sim debug 只保留默认关闭 hook。 |
| `include/output_interface/output_interface.hpp` | 添加 raw reference target 输出字段 | 回退 base。raw reference GT 应来自 Python pose stream，不来自 deploy target。 |
| `include/input_interface/gamepad.hpp` | planner/start 行为变化 | 回退 base；与 sim2sim 无关。 |
| `include/input_interface/keyboard_handler.hpp` | stop/pause/reset 行为变化 | 回退 base；与 sim2sim 无关。 |
| `include/input_interface/ros2_input_handler.hpp` | 小行为/注释变化 | 默认回退 base；如有独立 ROS2 需求另开任务。 |
| `include/robot_parameters.hpp` | 删除 BMS/pause 字段 | 回退 base；真机部署相关，不可随 sim2sim 合入。 |
| `tests/test_ros2.cpp` | 测试删除 | 回退 base；非 sim2sim 必需。 |

#### C++ Phase C0：恢复/比对 base 主路径

目标：建立“最终合入 base 时 C++ 文件可以不变”的可验证前提。

任务：

- 从 base commit 导出 16 个 C++ 文件的原始内容到 `tmp/sim2sim_refactor/<run_id>/cpp_base_snapshot/`。
- 对 current C++ 差异生成 patch 分类：
  - `drop_non_sim2sim.patch`
  - `candidate_sim2sim_debug.patch`
  - `smpl_protocol_separate_review.patch`
- 在不修改 current C++ 的前提下，先判断 Python 旁路能否满足：
  - streamer manifest 提供 sent `frame_index`
  - MuJoCo visualizer exact raw pose buffer 提供 GT frame
  - sim step-sync CSV 提供 actual/ref 同步行

验收：

- 如果 deterministic replay 能证明 metrics 不依赖 deploy-emitted `applied_source_frame_index`，最终迁移时 16 个 C++ 文件全部回退 base。
- 如果不能证明，进入 C++ Phase C1。

#### C++ Phase C1：最小默认关闭 source-frame debug hook

只在 C0 证明 Python 旁路不足时执行。

最小需求：

- deploy 能在 debug 开启时输出“当前控制 tick 实际消费的 source frame”。
- 不改变 policy 输入、不改变 start/planner、不改变 keyboard/gamepad、不改变默认 debug schema。

建议接口：

```cpp
struct Sim2SimDebugConfig {
  bool enabled = false;
  bool write_csv = false;
  bool publish_zmq = false;
  std::string logs_dir;
};

class Sim2SimSourceFrameTracker {
 public:
  void OnStreamFrame(int64_t frame_index);
  void OnControlTickConsumedFrame(int64_t frame_index);
  int64_t LatestConsumedFrame() const;
};
```

接入点：

- 只在 ZMQ streamed-motion 输入实现内部更新 tracker。
- 只在 `g1_deploy_onnx_ref.cpp` 已经进入 debug/log 输出分支时读取 tracker。
- CSV/ZMQ 输出由 `Sim2SimDebugSink` 完成，不修改 `StateLogger` 的默认字段集合。

性能要求：

- `enabled=false` 时只允许一次布尔判断，无文件 I/O、无动态分配、无字符串拼接、无 ZMQ publish。
- `enabled=true` 时记录频率随已有 debug/log 频率，不额外提高主循环 I/O 频率。
- 需要在 base C++ 和开启 sim2sim debug 两种配置下记录控制循环耗时统计。

#### C++ Phase C2：迁移到 base 前的验证

必须通过：

- C++ 默认配置编译通过，输出 schema 与 base 一致。
- 真机/部署默认命令不新增 sim2sim CSV 或 debug 字段。
- `--enable-sim2sim-debug` 开启后才产生额外 source-frame 输出。
- 与 Phase 1/2 Python step-sync deterministic manifest 对齐：
  - streamer `frame_index`
  - deploy consumed frame
  - MuJoCo `sim_source_frame_index`
  - `sim2sim_step_sync_body_pos_w_14.csv`
- 性能报告写入 `tmp/sim2sim_refactor/<run_id>/cpp_perf_report.md`。

### 2.2 base 已有 Python 核心文件过载

`gear_sonic/utils/mujoco_sim/base_sim.py` 同时承担：

- MuJoCo stepping。
- reference visualization。
- sim2sim eval logging。
- tracking overlay。
- link error plot。

风险：

- sim2sim 调试逻辑混入仿真主循环，后续 merge 冲突大。
- 真实控制 / 普通 sim / sim2sim eval 的职责边界不清楚。

策略：

- 不以当前 `base_sim.py` 覆盖 base；后续基于 base 的 `base_sim.py` 做最小 hook。
- 把 sim2sim 独有逻辑放到纯新增模块中，`base_sim.py` 只保留薄挂接点。
- 保持挂接点默认关闭或低开销。
- 不改变现有配置名和 CLI 行为，先做内部重排。

### 2.3 工具脚本职责混杂

`tools/sonic_eval/` 目前同时包含：

- 协议 pack/publish。
- motionlib adapter。
- parquet adapter。
- metrics reader。
- batch orchestration。
- recorder。

策略：

- 由于 `tools/sonic_eval/*` 相比 base 是纯新增，第一轮不优先重构这些文件本身。
- 只有当工具脚本逻辑需要侵入 base 已有文件时，才把公共逻辑下沉到新增包内，减少对 base 文件的修改。
- 保留现有 entrypoint 路径，减少用户命令变化。

## 3. 推荐目标结构

第一阶段建议结构调整重点不是重写纯新增 `tools/sonic_eval/*`，而是让 base 已有文件只依赖一个小的新增 sim2sim 包。候选结构：

```text
tools/sonic_eval/
  stream_motionlib_to_deploy.py          # 保留 CLI，薄 wrapper
  stream_motionlib_smpl_to_deploy.py     # 保留 CLI，薄 wrapper
  stream_parquet_smpl_to_deploy.py       # 保留 CLI，薄 wrapper
  compute_mujoco_tracking_metrics.py     # 保留 CLI，薄 wrapper 或分阶段瘦身
  run_mujoco_batch_eval*.sh              # 保留入口，后续只编排

gear_sonic/sim2sim/
  __init__.py
  constants.py                           # 14 link、DOF order、默认参数
  protocol/
    packed_zmq.py                        # PackedPublisher/PackedSubscriber
    schemas.py                           # v1/v3 field schema 校验
  adapters/
    motionlib.py                         # motionlib pkl -> standard sequence
    smpl_parquet.py                      # parquet -> standard sequence
  visualization/
    reference_motion.py                  # ReferenceMotionVisualizer
    link_error_plot.py                   # Sim2SimLinkErrorPlot
    overlay.py                           # viewer overlay
  logging/
    eval_logger.py                       # Sim2SimEvalLogger
    readers.py                           # CSV readers, validation
  metrics/
    mujoco_tracking.py                   # metrics core
    alignment.py                         # source_frame_index / index / auto_q29
  workflow/
    stream.py                            # stream workflow
    batch_eval.py                        # batch workflow, Python 部分
```

`gear_sonic/utils/mujoco_sim/base_sim.py` 最终只做：

- 根据 config 创建 sim2sim hooks。
- 在 `mj_step` 后传入 actual body pos / ref pose / source frame。
- close 时释放 hooks。

不在 `base_sim.py` 中继续堆 CSV 格式、plot 子进程、exact reference 匹配细节。

## 4. 分阶段执行计划

### Phase 0：base 差异冻结与块级审查

目标：在改代码前保留当前行为证据，并明确哪些同名文件差异必须消除/隔离。

任务：

- 记录当前 git commit、base commit `0a751ee86aab866a6bf483b41e1b8a48a0787d44`、环境、命令参数。
- 生成 current vs base 的 tracked diff 清单。
- 对 16 个同名 C++ 差异做块级审查，输出：
  - 是否 sim2sim 必需。
  - 是否可不合入 base。
  - 是否可由 Python 新增工具替代。
  - 若必须保留，是否默认关闭且无实时路径开销。
- 对 `base_sim.py`、`configs.py`、`motion_lib_base.py`、`pico_manager_thread_server.py` 等同名 Python 差异做块级审查。
- 确认测试数据来源：当前仓库有 `eval_benchmark/robot`、`eval_benchmark/smpl`、`data/smpl_filtered`；目标 S0 当前没有 `eval_benchmark`。

输出：

- 在本文档追加 `Baseline Notes`。
- 生成可重复运行的 baseline 命令记录。
- 生成 `同名差异处理表`：keep-as-base / thin-hook / move-to-new-module / must-review-cpp / ignore-new-file。

### Phase 1：先隔离 base 已有文件差异，不动纯新增工具

目标：降低 base 已有文件差异，保持 CLI 和输出不变。

测试边界说明：

- Phase 1 主要是源码搬移和导入边界调整，因此最关键的验证包括“搬移前后类/常量源码一致”和真实 motion 固定日志下的确定性链路 replay。
- Phase 1 已补齐第一层确定性链路验证，覆盖 streamer manifest、packed ZMQ round-trip、deploy cursor、reference pose buffer、step-sync alignment、metrics replay、数据集 manifest smoke。
- Phase 1 的数据 smoke 只验证数据可加载、字段/shape/finite 合法；数据 smoke 本身不能替代完整链路对比。
- Phase 1 的单条端到端 strict alignment smoke 验证链路能运行、step-sync 日志可消费、`source_frame_index` 对齐为 0 lag；随机 policy 端到端多次统计分布验证仍由后续 phase 根据改动风险决定。

优先处理：

- `SIM2SIM_BODY_FRAMES`、`G1_ISAACLAB_TO_MUJOCO_DOF` -> `gear_sonic/sim2sim/constants.py`
- `PackedZMQSubscriber` -> `gear_sonic/sim2sim/protocol/packed_zmq.py`
- `Sim2SimEvalLogger` -> `gear_sonic/sim2sim/logging/eval_logger.py`
- `Sim2SimLinkErrorPlot` -> `gear_sonic/sim2sim/visualization/link_error_plot.py`

暂不优先处理：

- `tools/sonic_eval/*` 中的纯新增脚本，除非它们要求修改 base 已有文件。
- `gear_sonic/utils/mujoco_sim/link_error_plot.py` 作为纯新增文件可以先保留，后续如需要统一命名再移动。

验收：

- `python gear_sonic/scripts/run_sim_loop.py --help` 正常。
- `python tools/sonic_eval/stream_motionlib_to_deploy.py --help` 正常。
- `python tools/sonic_eval/compute_mujoco_tracking_metrics.py --help` 正常。
- `python -m compileall gear_sonic/sim2sim tools/sonic_eval gear_sonic/utils/mujoco_sim` 通过。

### Phase 2：基于 base 的 `base_sim.py` 做最小 hook

目标：让最终合入 base 时，`base_sim.py` 只新增少量 hook 调用，不承载 sim2sim 业务细节。

建议接口：

```python
hook = Sim2SimMujocoHook(config, mj_model, mj_data, body_joint_names, hand_joint_names)
hook.after_mj_step(actual_body_pos_pre_ref, is_new_control_frame)
hook.update_reference_visualization()
hook.close()
```

注意：

- `actual_body_pos_pre_ref` 仍应在 reference robot update 前取，避免 ref robot 的 `mj_forward` 污染 actual。
- step-sync 写入仍只在 new control frame 且 source frame 有效时发生。
- exact reference pose 必须来自 raw pose stream，不从 deploy target fallback 当 GT。

验收：

- 单条 `eval_benchmark/robot_test/reach-2-003_chr00.pkl` 可跑通完整 A/B/C/D，或在端到端环境不稳定时复用 Phase 1 fixed logs 完成确定性 replay 并明确边界。
- 输出 CSV 文件名和字段不变：
  - `body_pos_w_14.csv`
  - `sim_source_frame_index.csv`
  - `sim2sim_step_sync_body_pos_w_14.csv`
  - deploy 侧 `source_frame_index.csv`
- metrics JSON 中 `actual_source=step_sync_body_pos_w_14`，`gt_body_source=mujoco_ref_body_pos_w_14`。
- `base_sim.py` 中不再直接管理 sim2sim logger、plot、overlay、reference visualizer 的业务细节。
- C++/header diff 为空。

Phase 2 具体执行范围：

- 新增 `gear_sonic/sim2sim/mujoco_hook.py`。
- 从 `base_sim.py` 移入 hook：
  - `_init_sim2sim_eval_logger`
  - `_init_link_error_plot`
  - `_init_reference_visualizer`
  - `_log_sim2sim_eval_frame`
  - `update_reference_motion_visualization`
  - tracking overlay render/update
  - reference scene XML clone/prefix 逻辑
- `base_sim.py` 保留原有时序：
  - `mj_step`
  - capture actual body position before reference robot update
  - reference visualizer `poll/apply`
  - reference applied 后 `mj_forward`
  - step-sync log on new control frame

Phase 2 必须复跑测试：

```bash
python -m compileall gear_sonic/sim2sim tools/sonic_eval gear_sonic/utils/mujoco_sim
python gear_sonic/scripts/run_sim_loop.py --help
/home/lab/miniconda3/envs/sonic/bin/python tools/sonic_eval/stream_motionlib_to_deploy.py --help
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py --help
bash tools/sonic_eval/run_mujoco_batch_eval.sh --help
bash tools/sonic_eval/run_mujoco_batch_eval_parallel.sh --help
bash tools/sonic_eval/run_mujoco_multi_instance_parallel.sh --help
git diff --check
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help
git diff --name-only | rg '\.(cpp|hpp|cc|hh|h)$' || true
```

Phase 2 确定性链路验证：

- 将 Phase 1 的完整确定性链路验证脚本复制到 `tmp/sim2sim_refactor/<phase2_run_id>/deterministic_full/`。
- 使用 pre-Phase2 commit 作为 old/current 对比基准。
- 复用 Phase 1 fixed logs 时，必须读取：
  - `tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_sim_logs`
  - `tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_deploy_logs`
  - `tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_combined_logs`
- 通过标准与 Phase 1 完整确定性链路补测一致。

### 2026-06-24 Phase 2 完成记录

实现结果：

- 新增 `gear_sonic/sim2sim/mujoco_hook.py`。
- `gear_sonic/utils/mujoco_sim/base_sim.py` 中 sim2sim 业务细节已移动到 `Sim2SimMujocoHook`。
- `base_sim.py` 保留最小 hook 生命周期调用和时间顺序，不再直接管理 logger、plot、overlay、reference visualizer、reference XML clone/prefix。
- 本阶段没有修改 C++/header 文件。

关键时序保持不变：

```text
mj_step
  -> capture actual body pos before reference robot update
  -> reference visualizer poll/apply
  -> mj_forward if reference pose applied
  -> hook.log_frame(write_step_sync=is_new_control_frame)
```

测试结果摘要：

- 通用门禁：全部通过。
- C++ 默认 deploy target build/help：通过。
- C++/header diff gate：无输出。
- Phase 2 deterministic replay：通过。
- fixed logs metrics replay：与 Phase 1 fixed metrics 关键字段一致。
- 新 hook 真实 E2E strict alignment smoke：通过，`lag_frames_log_vs_gt=0`。

新 hook E2E metrics：

- `num_frames=743`
- `source_frame_valid_rows=743`
- `step_sync_rows=743`
- `lag_frames_log_vs_gt=0`
- `mpjpe_g=112.0429219746846`
- `mpjpe_l=70.81177089518766`
- `mpjpe_pa=30.77083514480582`

差异说明：

- 新 hook E2E 有效帧数少于 Phase 1 fixed run，是因为本次运行在 early stream 阶段多次 fall/reset，reference visualizer 在 source frame `338` 后才稳定锁定。
- 该差异属于端到端 runtime 随机性和启动状态差异；固定日志 deterministic replay 覆盖完整 `995` rows 并确认时间帧对齐逻辑未变。

Phase 2 门禁结论：

- 本阶段所有计划内测试均已成功完成。
- 可以提交并 push Phase 2。
- push 完成前不能进入 Phase 3。

### Phase 3：streamer / metrics 内部模块化

目标：只有在 Phase 1/2 发现纯新增工具内部重复过多或影响 base 已有文件时，再整理 `tools/sonic_eval/*.py`。保留命令不变。

建议拆分：

- motionlib 读取、stand/blend prefix、finite difference 放入 adapter/workflow。
- parquet SMPL 过滤、field stack、wrist DOF 构造放入 adapter。
- CSV readers、source frame filter、metrics core 放入 `gear_sonic/sim2sim/metrics`。

验收：

- 现有 robot encoder 单文件命令不变。
- 现有 human/SMPL encoder parquet 命令不变。
- metrics 的数值与 Phase 0 baseline 同量级，差异可解释。

### Phase 4：base 仓库迁移适配

目标：面向 base 仓库做最小合入。

策略：

- 不覆盖 base 已有 C++ 文件；除非 Phase 0 标记为 must-review-cpp 并获得单独确认。
- 纯新增文件可以先整体合入或暂存，不作为结构优化重点。
- 对 base 已有文件只做薄挂接，尤其是 `gear_sonic/utils/mujoco_sim/base_sim.py`、`configs.py`。
- 如果目标分支没有 `eval_benchmark`，以文档说明使用当前仓库数据或复制为测试资产，是否纳入 repo 需团队确认。

迁移前置：

- 在 LHM-Robot 上基于 `feat/s0_training` 建独立实验分支或 worktree。
- 跑目标分支原有 smoke test，确认未引入前已通过。
- 逐文件移植并记录冲突。

## 5. 测试计划

### 5.0 随机性下的验证原则

关键问题：policy 每次运行同一条 motion 的结果可能不同，不能只用单次 metrics 数值是否完全一致来判断结构优化是否破坏逻辑。

解决原则：把验证拆成两层。

第一层：确定性链路验证，尽量绕开 policy 随机性。

- 对纯数据处理模块做 golden / regression：
  - motionlib pkl -> dof/root/body_pos 序列。
  - parquet -> SMPL encoder 输入字段。
  - packed ZMQ header/schema。
  - CSV reader / source_frame_index alignment。
  - metrics 输入 `actual/ref/source_frame_index` -> metrics JSON。
- 对 MuJoCo sim logging 做 record/replay：
  - 使用同一份已录制的 `sim2sim_step_sync_body_pos_w_14.csv`、`body_pos_w_14.csv`、`sim_source_frame_index.csv`、`source_frame_index.csv` 重跑 metrics。
  - 这一步不启动 policy，不启动 MuJoCo，只验证 metrics/alignment 重构没有改变。
- 对 reference visualization / GT 做无 policy 检查：
  - 用同一条 raw pose stream 或离线构造的 pose frame buffer，检查 `source_frame_index -> ref 14 link body_pos` 是否一致。
  - 检查 `actual_body_pos` 读取时机仍在 reference robot `mj_forward` 之前，避免 ref robot 污染 actual。

时间帧对齐的逐环节确定性验证：

1. streamer 序列生成阶段
   - 输入同一条 `motion_file`、`motion_name`、`target_fps`、`start_frame`、`end_frame`、`prepend_stand_frames`、`blend_from_stand_frames`、`initial_burst_frames`。
   - 落盘一个 dry-run manifest，不连接 deploy：
     - `stream_frame_index[i]`
     - `motion_start_frame`
     - `is_prefix_frame`
     - `source_motion_frame`
     - `root_pos_w/root_quat_w` hash
     - `dof_pos/dof_vel` hash
   - 验证 refactor 前后：
     - frame_index 单调递增且完全一致。
     - `motion_start_frame == prepend_stand_frames + blend_from_stand_frames`，或与 auto align 规则一致。
     - prefix/blend/motion 三段边界一致。
     - 同一 source frame 对应的 dof/root/body 序列 hash 一致。

2. ZMQ packed message 阶段
   - 对 streamer 输出做离线 pack/unpack round-trip，不启动 deploy。
   - 验证 header 字段和 shape 不变：
     - v1 robot encoder：`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`frame_index`、`catch_up`。
     - v3 SMPL encoder：`joint_pos`、`joint_vel`、`smpl_joints`、`smpl_pose`、`body_quat_w`、`frame_index` 等。
   - 验证 unpack 后：
     - `frame_index` 与 streamer manifest 完全一致。
     - chunk 首尾 frame、chunk size、initial burst 后的连续性一致。
     - `motion_start_frame` 在 header 中保持一致。

3. deploy 播放游标阶段
   - 在不依赖 policy 动作结果的情况下，读取 deploy 侧已录制的：
     - `source_frame_index.csv`
     - `applied_source_frame_index.csv`（如果存在）
     - `target_motion.csv` / `policy_input.csv` 中可对应的时间列。
   - 验证：
     - `source_frame_index` 与 streamer frame_index 的交集、首尾、缺口数量一致。
     - `applied_source_frame_index` 相对 `source_frame_index` 的固定延迟/映射规则一致。
     - `source_frame_index` 没有回退，除非新 motion 重启且日志明确记录。
     - start command 之前的 `-1` 或 invalid 区间不进入 metrics 有效区间。

4. MuJoCo reference pose buffer 阶段
   - 对 `ReferenceMotionVisualizer` 增加或使用离线测试入口，直接喂同一批 pose messages 和 debug source frames。
   - 验证：
     - `_pose_frame_buffer` 中 frame_index 集合一致。
     - 对每个 debug `source_frame_index`，必须命中 exact raw pose；不能因为 fallback 到 latest/old pose 而参与 GT。
     - auto align delay 的锁定帧一致：
       - 显式 `reference_motion_align_delay_frames > 0` 时等于该值。
       - auto 时等于 stream header 的 `motion_start_frame` 或既定推断值。
     - `raw_global`、`start_aligned_xy`、`delayed_align` 等 translation mode 的 anchor frame 和 anchor transform 一致。

5. MuJoCo step-sync 写入阶段
   - 使用同一段已录制或离线模拟的：
     - actual 14 link body positions
     - exact ref 14 link body positions
     - source_frame_index 序列
     - `is_new_control_frame` 序列
   - 直接调用 logger/hook，生成 CSV。
   - 验证：
     - `body_pos_w_14.csv` 行数、`index`、`sim_time` 单调性一致。
     - `sim_source_frame_index.csv` 行数和 invalid/valid 分段一致。
     - `sim2sim_step_sync_body_pos_w_14.csv` 只在 `is_new_control_frame && source_frame_index >= 0 && ref_body_pos is not None` 时写入。
     - 同一个 `source_frame_index` 不重复写 step-sync 行，除非原逻辑明确允许。
     - step-sync 行里的 `actual_*` 来自 reference update 前的 actual snapshot。
     - step-sync 行里的 `ref_*` 来自 exact raw pose 计算，不来自显示 fallback。

6. metrics 读取与切片阶段
   - 固定输入同一组 CSV：
     - `sim2sim_step_sync_body_pos_w_14.csv`
     - `body_pos_w_14.csv`
     - `sim_source_frame_index.csv`
     - `source_frame_index.csv`
   - 验证：
     - `filter_step_sync_body_pos_by_source_frame` 的有效 frame 集合一致。
     - `--streamed-only`、`--sim-valid-only`、`--ignore-motion-playing-mask` 对有效区间的影响一致。
     - `align-mode=source_frame_index` 的 ref/actual 配对表一致，可以输出 `(metric_row, source_frame_index, actual_row, ref_row)` manifest 对比。
     - metrics 输入数组 `actual_body_pos[T,14,3]`、`gt_body_pos[T,14,3]` 的 shape、source_frame_index 序列、hash 一致。

7. 可视化与 metrics 的同源性验证
   - 对同一帧 `source_frame_index=k`，同时记录：
     - viewer reference body positions
     - step-sync CSV 中 `ref_*`
     - offline exact reference function 输出
   - 验证三者在同一 translation/alignment mode 下数值一致或只有明确的显示偏移差异。
   - 如果显示为 `delayed_align`，metrics GT 必须使用同一锁定 anchor；不能出现 viewer 和 metrics 各自重新对齐。

建议新增的确定性检查文件：

- `tmp/sim2sim_refactor/<run_id>/stream_manifest.csv/jsonl`：streamer dry-run 每帧索引、prefix/motion 边界、hash。
- `tmp/sim2sim_refactor/<run_id>/packed_message_manifest.jsonl`：每个 ZMQ chunk 的 header、shape、首尾 frame。
- `tmp/sim2sim_refactor/<run_id>/alignment_manifest.csv`：metrics 最终使用的 `(metric_row, source_frame_index, actual_row, ref_row)`。
- `tmp/sim2sim_refactor/<run_id>/step_sync_invariant_report.json`：step-sync 行数、重复 source frame、invalid frame 分段、exact pose miss 次数。

验证数据目录约定：

- 根目录：`tmp/sim2sim_refactor/`
- 每次验证运行使用独立 `<run_id>`，建议格式：`YYYYMMDD_HHMMSS_<phase>_<scope>`。
- baseline 原始日志：`tmp/sim2sim_refactor/<run_id>/baseline_logs/`
- refactor 原始日志：`tmp/sim2sim_refactor/<run_id>/refactor_logs/`
- deterministic replay 输入：`tmp/sim2sim_refactor/<run_id>/replay_inputs/`
- deterministic replay 输出：`tmp/sim2sim_refactor/<run_id>/replay_outputs/`
- 多次端到端统计结果：`tmp/sim2sim_refactor/<run_id>/metric_runs/`
- 汇总报告：`tmp/sim2sim_refactor/<run_id>/summary.json` 和 `summary.csv`

清理要求：

- `tmp/` 下只保留当前阶段需要的验证数据和最终报告。
- 每个 phase 的所有计划内测试都成功完成、确认没有问题、阶段文档更新完成并完成提交/push 后，可以清理该 phase 产生的大体积中间日志，避免占用磁盘。
- 清理前必须保留小体积 summary、关键 metrics JSON、失败/差异解释、测试命令、环境和原始路径索引。
- 如果某个测试失败、结果不确定、后续仍需复查，不能删除对应原始数据；只能压缩或继续保留。
- 大体积中间日志需要在阶段报告确认后清理或压缩，并把清理记录写入 `tasks/sim2sim_structure_refactor/log.md`。
- 不把 `tmp/` 内容作为默认合入内容；如需提交测试报告，单独整理小体积 summary。

确定性验证的判定标准：

- 对齐相关 manifest 必须逐行一致；如果不一致，先解释每一个差异，不进入端到端随机统计验证。
- 数值 hash 可以允许浮点序列化导致的极小误差，但 frame index、边界、有效区间、配对关系必须完全一致。
- 任何 fallback pose 参与 metrics GT 都判定为失败。

第二层：端到端统计验证，承认 policy 随机性。

- 每个关键样本至少重复运行 N 次 baseline 和 N 次 refactor，建议 smoke N=3，全量抽样 N=5。
- 比较分布而不是比较单点：
  - mean / std / min / max。
  - median 和 p90。
  - `step_sync_rows`、valid source frame 覆盖范围、有效帧比例。
  - per-link MPJPE 排名是否发生明显异常变化。
- 接受标准：
  - refactor 后均值落在 baseline 的自然波动范围内，或差异有明确解释。
  - `step_sync_rows`、source frame 覆盖范围不能明显减少。
  - 不能出现系统性偏移，例如所有 link error 同时扩大、时间对齐整体错位、前几帧/末尾有效帧大量丢失。
- 若端到端 metrics 变差但确定性链路验证全部通过，应优先怀疑 policy/runtime 随机性、启动状态、落地/弹力带状态、ZMQ 时序；不要直接判定结构优化破坏逻辑。

建议新增两类验证产物：

- `replay_metrics_from_logs`：只消费已录制 CSV，重算 metrics，作为确定性回归。
- `compare_metric_runs`：汇总多次 baseline/refactor 运行，输出均值、方差、相对变化、异常样本。

### 5.1 静态 / 轻量测试

每个阶段都跑：

```bash
python gear_sonic/scripts/run_sim_loop.py --help
python tools/sonic_eval/stream_motionlib_to_deploy.py --help
python tools/sonic_eval/compute_mujoco_tracking_metrics.py --help
python -m compileall gear_sonic/sim2sim tools/sonic_eval gear_sonic/utils/mujoco_sim
```

如涉及 shell：

```bash
bash tools/sonic_eval/run_mujoco_batch_eval.sh --help
bash tools/sonic_eval/run_mujoco_batch_eval_parallel.sh --help
bash tools/sonic_eval/run_mujoco_multi_instance_parallel.sh --help
```

### 5.2 eval_benchmark 全量测试

数据范围：

- `eval_benchmark/robot/*.pkl`：当前约 20 条。
- `eval_benchmark/smpl/*.pkl`：当前约 29 条。
- `eval_benchmark/robot_test/*.pkl`、`eval_benchmark/smpl_test/*.pkl`：保留为快速 smoke。

建议流程：

1. Phase 0 记录 baseline results，并保留原始 logs 用于 deterministic replay。
2. 每个重构阶段跑 smoke：
   - `eval_benchmark/robot_test/reach-2-003_chr00.pkl`
   - `eval_benchmark/smpl_test/reach-2-003_chr00.pkl`
3. 每个重构阶段先用 Phase 0 logs 做 replay metrics，确认 metrics/alignment 纯逻辑一致。
4. 完成 Phase 2 和 Phase 3 后跑全量 robot，多次重复取统计结果。
5. 完成 human/SMPL 相关改动后跑全量 smpl，多次重复取统计结果。

结果记录：

- 每条 motion 的 metrics JSON。
- 汇总 CSV。
- 每条 motion 的多次运行统计 summary。
- replay metrics 的 deterministic 对比结果。
- 失败样本、失败阶段、日志路径。
- `step_sync_rows`、valid source frame 范围、是否有空洞。
- MPJPE / max error / per-link error 的 baseline vs refactor 对比。

所有上述结果均写入 `tmp/sim2sim_refactor/<run_id>/` 对应子目录。

### 5.3 `data/smpl_filtered` 抽样测试

从 `data/smpl_filtered` 选覆盖不同动作类型的样本：

- locomotion：`walk_forward_relax_001__A006_M.pkl`
- jump：`high_jump_R_002__A442_M.pkl`
- crouch：`idle_crouch_loop_104__A128.pkl`
- reach/manipulation：`reaching_up_R_003__A065.pkl`
- dance/large motion：`dancing_routine_2_003__A125_M.pkl`

如 robot pkl 配对不完整，先只做 streamer / adapter / metrics 可用性检查；需要完整 sim2sim metrics 时必须确认 robot motion 对应文件。

### 5.4 性能和侵入性检查

重点观察：

- MuJoCo heartbeat 中 `post_step_ms`、`total_ms` 是否与 baseline 同量级。
- 开启 sim2sim logging 与关闭 logging 的差异。
- link error plot 开启后是否有 queue 堆积、主循环阻塞。
- 默认运行不启用 plot 时不能启动额外 GUI/进程。
- 不产生大量临时文件或缓存。

## 6. 接受标准

结构标准：

- sim2sim 专属逻辑集中在 `gear_sonic/sim2sim/` 和 `tools/sonic_eval/`。
- base 已有文件的 diff 显著收敛；`base_sim.py` 只保留少量 hook 调用。
- C++ 文件默认与 base 保持一致；若不得不改，必须单独评审。
- 纯新增文件不强制重构；它们不能反向要求大量修改 base 已有核心文件。
- 原 CLI 命令和输出文件兼容。

行为标准：

- baseline 与 refactor 后同一批数据 metrics 同量级。
- `source_frame_index` 时间对齐没有退化。
- step-sync rows 不明显减少。
- reference visualization 与 metrics GT 同源。
- robot encoder 和 SMPL encoder 两条链路互不污染。

合入标准：

- base 仓库 `feat/s0_training` 实验分支可编译 / help / smoke。
- base 原有真机部署相关 C++ 快路径无新增默认日志开销。
- 测试报告记录在本文档，包含命令、数据、结果路径、差异说明。

## 7. 风险与回滚

风险：

- `base_sim.py` 抽 hook 时改变 `mj_step` 后、`mj_forward` 前后的数据读取顺序。
- reference pose exact matching 被 fallback 逻辑污染。
- metrics 读取 CSV 时字段顺序不兼容。
- LHM-Robot S0 已有同名功能，直接覆盖会丢失目标分支改动。
- 多实例脚本涉及端口、DDS domain、日志目录，重构后容易出现串扰。

回滚策略：

- 每个 phase 单独 commit 或至少单独 diff。
- 先保留旧 entrypoint 和旧输出格式。
- 出现 metrics 大幅偏移时，优先对比：
  - `source_frame_index.csv`
  - `sim_source_frame_index.csv`
  - `sim2sim_step_sync_body_pos_w_14.csv`
  - reference raw pose stream frame index。

## 8. 后续迭代记录

### 2026-06-23 初版计划

已完成：

- 阅读 `workspace/notes/sim2sim.md` 和已有 human encoder 计划。
- 初步定位当前 sim2sim 相关文件。
- 初步查看目标仓库 `/home/lab/Desktop/LHM-Robot`：当前分支 `master`，本地存在 `feat/s0_training`，S0 目录下没有 `eval_benchmark`。

待确认：

- 是否允许在 LHM-Robot 上创建独立 worktree / 实验分支。
- `eval_benchmark` 测试数据是否需要迁移到 LHM-Robot，还是仅作为本地验证数据使用。

### 2026-06-23 base 差异优先级更新

根据用户补充，后续将 `/home/lab/Desktop/LHM-Robot/S0` 的 `feat/s0_training` 称为 base 仓库。

新增执行原则：

- 相比 base 纯新增的文件，暂时可以先不做结构优化。
- 最关键的是让 C++ 文件尽量不要和 base 不同。
- 对 base 已有且 current 内容不同的代码文件，要特别关注并做结构优化。

已完成只读盘点：

- base commit：`0a751ee86aab866a6bf483b41e1b8a48a0787d44`
- base/current common tracked files：376
- base 已有且 current 内容不同：64
- base 已有且 current 内容不同的 C++/header：16
- current 纯新增 sim2sim 相关工具主要在 `tools/sonic_eval/*` 和 `gear_sonic/utils/mujoco_sim/link_error_plot.py`，暂不作为第一优先级。

下一步 Phase 0 应先产出 16 个 C++ 差异和关键 Python 同名差异的块级处理表，再开始代码迁移/结构调整。

### 2026-06-24 随机性验证策略更新

补充关键验收原则：由于 policy 端到端运行存在随机性，结构优化不能只靠单次 metrics 精确一致判断。

后续测试按两层执行：

- 确定性验证：用已录制 logs / raw pose / CSV 重放，验证 adapter、alignment、metrics、reference GT 逻辑不变。
- 统计验证：baseline/refactor 分别多次运行同一批 motion，比较 metrics 分布、source frame 覆盖、step_sync rows 和 per-link 异常，而不是逐字节或单次数值一致。


### 2026-06-24 Phase 0 执行记录

Run ID：`20260624_113337_phase0_base_diff`

产物目录：`tmp/sim2sim_refactor/20260624_113337_phase0_base_diff/`

已完成：

- 记录 base/current commit：
  - base：`0a751ee86aab866a6bf483b41e1b8a48a0787d44`
  - current：`63301c530353fd7ef68b27c915990a4ada30f06d`
- 生成 tracked file manifest、same-path modified manifest、C++ manifest、Python/config manifest。
- 生成 C++ 块级审查报告：`tmp/sim2sim_refactor/20260624_113337_phase0_base_diff/reports/cpp_diff_review.md`。
- 生成 Python/config 同名差异审查报告：`tmp/sim2sim_refactor/20260624_113337_phase0_base_diff/reports/python_config_diff_review.md`。

关键统计：

- common tracked files：376
- base 已有且 current 内容不同：64
- current 纯新增 tracked files：2791
- base 有但 current 缺失：90
- common C++/header files：51
- base 已有且 current 内容不同的 C++/header：16
- current 纯新增 C++/header：789

Phase 0 结论：

- 16 个同路径 C++/header 差异全部先标记为 `must-review-cpp`，Phase 1 不修改、不继续扩大这些 C++ 差异。
- 部署侧 `source_frame_index`、`applied_source_frame_index`、ZMQ debug schema 等能力虽然对严格对齐有帮助，但当前不能整块带入 base；后续先验证 Python/MuJoCo 侧是否能替代。
- `gear_sonic/utils/mujoco_sim/base_sim.py` 是 Phase 1 最合适的结构优化目标：current 相比 base 约 `+1456/-9`，包含大量 sim2sim 专属类和常量，可以先搬到纯新增 Python 包，降低对 base 已有文件的侵入。
- `tools/sonic_eval/*` 和 `gear_sonic/utils/mujoco_sim/link_error_plot.py` 多为 current 纯新增，先保留入口和行为，不作为第一刀重构目标。

Phase 1 执行边界：

- 只做 Python 结构隔离。
- 不修改 C++。
- 不修改 metrics 语义、CSV 文件名/列名、CLI 参数名。
- 不处理训练/SMPL motion lib 的大块同名差异。

### 2026-06-24 Phase 1 第一刀执行记录

目标：先降低 `gear_sonic/utils/mujoco_sim/base_sim.py` 对 sim2sim 专属实现的承载，不改变运行逻辑和输出格式。

已完成：

- 新增 `gear_sonic/sim2sim/` 包，作为 sim2sim 专属 Python 逻辑的边界。
- 从 `base_sim.py` 原样搬出以下代码块：
  - `REFERENCE_NAME_PREFIX`、`PACKED_ZMQ_HEADER_SIZE`、`G1_ISAACLAB_TO_MUJOCO_DOF`、`SIM2SIM_BODY_FRAMES` -> `gear_sonic/sim2sim/constants.py`
  - `PackedZMQSubscriber` -> `gear_sonic/sim2sim/protocol/packed_zmq.py`
  - `ReferenceMotionVisualizer` -> `gear_sonic/sim2sim/visualization/reference_motion.py`
  - `Sim2SimEvalLogger` -> `gear_sonic/sim2sim/logging/eval_logger.py`
  - `Sim2SimTrackingOverlay` -> `gear_sonic/sim2sim/visualization/overlay.py`
- `base_sim.py` 保留现有配置名、hook 调用、step-sync 写入顺序、reference visualization 调用顺序和 close 顺序。
- 本阶段没有修改任何 C++ 文件。

当前结构变化：

- `base_sim.py` 中不再定义 packed ZMQ、reference visualizer、eval logger、viewer overlay 等 sim2sim 类。
- `base_sim.py` 仍然包含 reference scene XML 复制/加前缀逻辑；这是下一步可选继续隔离的内容，因为它仍属于 MuJoCo scene 构造 hook。
- `gear_sonic/utils/mujoco_sim/link_error_plot.py` 暂时保留原路径，因为它相比 base 是纯新增文件，且不是第一优先级。

已执行验证：

```bash
python -m compileall gear_sonic/sim2sim tools/sonic_eval gear_sonic/utils/mujoco_sim
python gear_sonic/scripts/run_sim_loop.py --help
python tools/sonic_eval/stream_motionlib_to_deploy.py --help
python tools/sonic_eval/compute_mujoco_tracking_metrics.py --help
```

结果：

- `compileall` 通过。
- `run_sim_loop.py --help` 通过，sim2sim CLI 参数仍可见。
- `stream_motionlib_to_deploy.py --help` 通过。
- `compute_mujoco_tracking_metrics.py --help` 使用 notes 中的 metrics 环境通过：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_backup/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py --help
```

环境注意：

- `conda run -n sonic_backup ...` 在当前 shell 中解析到了 `.venv_sim/bin/python`，导致最初误报缺少 `pandas`。
- 直接使用 `/home/lab/miniconda3/envs/sonic_backup/bin/python` 后，`pandas` 存在且 metrics help 正常。

最初失败信息如下，保留用于说明环境串扰：

```text
ModuleNotFoundError: No module named 'pandas'
```

格式化/静态检查限制：

- `python -m black ...` 失败：当前 `.venv_sim` 缺少 `black`。
- `python -m ruff check ...` 失败：当前 `.venv_sim` 缺少 `ruff`。
- 尝试用 `uv pip install --python .venv_sim/bin/python black ruff` 补齐；首次受沙箱缓存目录限制失败，提权重试后下载卡住，已终止挂起进程。
- 因此本阶段以 `compileall`、入口 help 和 `git diff --check` 作为可执行检查。

下一步建议：

- 继续把 `base_sim.py` 中 reference scene 构造相关的 `_maybe_create_reference_visualization_scene` / `_prefix_reference_subtree` 移到 `gear_sonic/sim2sim/visualization/reference_scene.py`，让 `base_sim.py` 进一步收敛为薄 hook。
- 在有完整依赖环境后补跑 `compute_mujoco_tracking_metrics.py --help`。
- 进入端到端运行前，先实现或补齐 deterministic replay manifest，避免 policy 随机性掩盖时间帧对齐问题。

### 2026-06-24 Phase 1 最终门禁验证记录

Run ID：`20260624_142140_phase1_validation`

产物目录：`tmp/sim2sim_refactor/20260624_142140_phase1_validation/`

本阶段代码范围：

- 修改 `gear_sonic/utils/mujoco_sim/base_sim.py`，只保留 sim2sim 薄挂接和原有调用顺序。
- 新增 `gear_sonic/sim2sim/` 包，承载 packed ZMQ、reference visualization、eval logger、tracking overlay、常量。
- 未修改任何 C++/header 文件。
- `.gitignore` 当前有无关工作区改动，非本阶段改动，不纳入本阶段提交。

最终通过的测试：

```bash
python -m compileall gear_sonic/sim2sim tools/sonic_eval gear_sonic/utils/mujoco_sim
python gear_sonic/scripts/run_sim_loop.py --help
/home/lab/miniconda3/envs/sonic/bin/python tools/sonic_eval/stream_motionlib_to_deploy.py --help
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py --help
bash tools/sonic_eval/run_mujoco_batch_eval.sh --help
bash tools/sonic_eval/run_mujoco_batch_eval_parallel.sh --help
bash tools/sonic_eval/run_mujoco_multi_instance_parallel.sh --help
git diff --check
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help
```

结果：

- Python compileall：通过。
- sim2sim 相关 Python / shell 入口 help：全部通过。
- `git diff --check`：通过。
- C++ 默认 deploy target build：通过。
- deploy 二进制 help：通过。
- `git diff --name-only | rg '\.(cpp|hpp|cc|hh|h)$'`：无输出，确认本阶段没有 C++/header diff。

确定性 moved-helper 验证：

```bash
python tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic/phase1_moved_helpers_check.py
```

结果：

- `PackedZMQSubscriber`、`ReferenceMotionVisualizer`、`Sim2SimEvalLogger`、`Sim2SimTrackingOverlay` 与迁移前 `base_sim.py` 中对应类源码一致。
- `REFERENCE_NAME_PREFIX`、`PACKED_ZMQ_HEADER_SIZE`、`G1_ISAACLAB_TO_MUJOCO_DOF`、`SIM2SIM_BODY_FRAMES` 与迁移前常量一致。
- packed ZMQ 离线 unpack、logger CSV 写入、overlay error 计算均通过。
- logger 检查写出 `body_rows=3`、`source_rows=3`、`step_sync_rows=1`。

数据 smoke 验证：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic/bin/python \
  tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic/motionlib_dataset_smoke.py
```

结果：

- 总数 `50`，通过 `50`，失败 `0`。
- 覆盖 `eval_benchmark/robot/*.pkl` 全部 19 条。
- 覆盖 `eval_benchmark/smpl/*.pkl` 全部 27 条。
- 覆盖 `data/smpl_filtered` 抽样 4 条：
  - `Idle_Left_001__A017.pkl`
  - `Jump_002__A017.pkl`
  - `Loop_Forward_Walk_001__A017.pkl`
  - `Neutral_stoop_down_001__A057.pkl`

端到端严格对齐 smoke：

第一轮手动端到端使用了 `--output-type none`，导致 deploy 不发布 `g1_debug`，sim 端 `sim_source_frame_index.csv` 全部为 `-1`，`sim2sim_step_sync_body_pos_w_14.csv` 为空。该轮判定为测试配置失败，不作为通过结果。

第二轮改为 `--output-type zmq` 后通过：

```bash
# Terminal A
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py \
  --interface sim \
  --simulator mujoco \
  --env-name default \
  --domain-id 102 \
  --sim2sim-eval-logs-dir "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_sim_logs" \
  --reference-motion-zmq-port 5718 \
  --reference-motion-pose-zmq-port 5726 \
  --no-enable-onscreen \
  --no-enable-offscreen

# Terminal B
cd gear_sonic_deploy
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic
just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx /tmp/sonic_motion_action_only \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host localhost \
  --zmq-port 5726 \
  --zmq-out-port 5718 \
  --dds-domain-id 102 \
  --enable-csv-logs \
  --logs-dir "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_deploy_logs" \
  --target-motion-logfile "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_deploy_logs/target_motion.csv" \
  --policy-input-logfile "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_deploy_logs/policy_input.csv" \
  --enable-motion-recording \
  --disable-crc-check

# Terminal C
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic/bin/python \
  tools/sonic_eval/stream_motionlib_to_deploy.py \
  --motion-file eval_benchmark/robot_test/reach-2-003_chr00.pkl \
  --motion-name reach-2-003_chr00 \
  --host 127.0.0.1 \
  --port 5726 \
  --target-fps 50 \
  --initial-burst-frames 20 \
  --blend-from-stand-frames 200 \
  --chunk-size 30 \
  --realtime \
  --send-command \
  --no-motionlib-robot

# Terminal D
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py \
  --gt-format motionlib \
  --motion-file eval_benchmark/robot_test/reach-2-003_chr00.pkl \
  --motion-name reach-2-003_chr00 \
  --logs-dir "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_combined_logs" \
  --out-json "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_results/reach-2-003_chr00_metrics.json" \
  --no-motionlib-robot \
  --ignore-motion-playing-mask \
  --streamed-only \
  --align-mode source_frame_index \
  --actual-source step_sync_body_pos_w_14 \
  --sim-valid-only \
  --stream-blend-from-stand-frames 200
```

端到端结果：

- streamer：`source_frames=930`，`sent_frames=1130`，endpoint `tcp://127.0.0.1:5726`。
- sim `sim_source_frame_index.csv`：`234188` 行，有效 `5964` 行，有效范围 `41..1083`。
- sim `sim2sim_step_sync_body_pos_w_14.csv`：`995` 行，有效范围 `41..1083`，无空文件问题。
- deploy `q.csv`：`1670` 行。
- deploy `source_frame_index.csv`：`1670` 行，有效范围 `30..1083`。
- metrics JSON：`tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_results/reach-2-003_chr00_metrics.json`
- metrics 关键字段：
  - `actual_source=step_sync_body_pos_w_14`
  - `gt_body_source=mujoco_ref_body_pos_w_14`
  - `alignment.mode=source_frame_index`
  - `lag_frames_log_vs_gt=0`
  - `num_frames=995`
  - `source_frame_valid_rows=995`
  - `step_sync_rows=995`
  - `mpjpe_g=179.1855`
  - `mpjpe_l=87.7680`
  - `mpjpe_pa=43.5100`
  - `progress_rate=0.0643`

解释：

- 本阶段结构调整是源码搬移和导入边界调整；deterministic moved-helper 检查确认核心类和常量源码一致。
- 端到端 metrics 数值用于确认链路可运行和时间帧严格对齐，不能作为单次随机 policy 质量结论。
- `success_rate=0` 来自该 motion 的跟踪阈值判定和运行状态，不影响本阶段“结构搬移是否破坏日志/对齐链路”的判断；严格对齐字段已通过。
- 第一轮 `--output-type none` 失败说明 strict step-sync 依赖 deploy debug ZMQ 输出，后续文档和脚本必须明确 sim2sim strict metrics 需要 `--output-type zmq` 或等效 debug 输出。

Phase 1 门禁结论：

- 本阶段计划内测试均已成功完成。
- 没有 C++/header 修改。
- Python 结构隔离没有改变已搬移类/常量源码。
- 入口、编译、数据读取、严格 frame 对齐、metrics 消费链路均通过。
- 可以提交并 push 本阶段，但不能进入 Phase 2 前遗漏 commit/push 门禁。

边界修正：

- 上述“本阶段计划内测试”指 Phase 1 源码搬移范围内的测试和本次补齐的完整确定性链路补测。
- 数据 smoke 本身不是实际结果对比；本阶段通过结论依赖完整确定性链路补测和固定日志 metrics replay。
- Phase 1 没有做随机 policy 多次端到端统计分布验证；该项不是本次源码搬移的必要门禁，但后续 phase 如扩大 hook 或 C++ 边界，必须按该 phase 风险重新评估是否加入。
- Phase 2 代码重构前必须重新明确 Phase 2 自身的完整测试命令、数据范围和退出标准，不能直接复用 Phase 1 结论替代 Phase 2 验证。

### 2026-06-24 Phase 1 完整确定性链路补测记录

补测 run：

- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/`
- 固定 motion：`eval_benchmark/robot_test/reach-2-003_chr00.pkl`
- `motion_name=reach-2-003_chr00`
- pre-refactor 对比 commit：`63301c530353fd7ef68b27c915990a4ada30f06d`

完整确定性链路脚本：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/phase1_full_deterministic_validation.py
```

产物：

- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/phase1_full_deterministic_validation_summary.json`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/stream_manifest.csv`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/packed_message_manifest.jsonl`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/alignment_manifest.csv`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/dataset_manifest_smoke.json`

关键结果：

- streamer manifest：`sent_frames=1130`，`source_frames=930`，`blend_frames=200`，`motion_start_frame=200`，frame range `0..1129`。
- packed ZMQ round-trip：`chunks=38`，old/current unpack 输出完全一致。
- deploy cursor：source/applied valid rows 均为 `1670`，frame range `30..1083`，均为 streamer manifest 子集且单调。
- reference pose buffer：`buffer_frames=1130`，step-sync exact pose hits `995`，misses `0`，old/current 抽样 frame buffer 一致。
- step-sync alignment：`raw_rows=995`，`filtered_rows=995`，source range `41..1083`，unique source frames `995`，无相邻重复。
- 数据集 manifest smoke：`eval_benchmark/robot/*.pkl` 19 条、`eval_benchmark/smpl/*.pkl` 27 条、`data/smpl_filtered` 抽样 4 条全部通过。

metrics replay：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py \
  --gt-format motionlib \
  --motion-file eval_benchmark/robot_test/reach-2-003_chr00.pkl \
  --motion-name reach-2-003_chr00 \
  --logs-dir "$ROOT/tmp/sim2sim_refactor/$RUN_ID/manual_e2e_sync_combined_logs" \
  --out-json "$ROOT/tmp/sim2sim_refactor/$RUN_ID/deterministic_full/metrics_replay_from_fixed_logs.json" \
  --no-motionlib-robot \
  --ignore-motion-playing-mask \
  --streamed-only \
  --align-mode source_frame_index \
  --actual-source step_sync_body_pos_w_14 \
  --sim-valid-only \
  --stream-blend-from-stand-frames 200
```

metrics replay 结果：

- `num_frames=995`
- `alignment.lag_frames_log_vs_gt=0`
- `alignment.source_frame_valid_rows=995`
- `alignment.step_sync_rows=995`
- `metrics_all.mpjpe_g=179.1855428355443`
- `metrics_all.mpjpe_l=87.76796772732865`
- `metrics_all.mpjpe_pa=43.50997555797189`
- 上述字段与原 `manual_e2e_sync_results/reach-2-003_chr00_metrics.json` 一致。

通用门禁复跑结果：

- Python compileall：通过。
- sim2sim Python/shell entrypoint help：全部通过。
- `git diff --check`：通过。
- C++ 默认 deploy target build/help：通过。
- C++/header diff 检查：无输出。

补测结论：

- Phase 1 完整确定性链路补测全部通过。
- 本次补测确认 Phase 1 的 Python 结构搬移没有改变真实 motion 下的 stream frame、packed message、reference pose buffer、step-sync frame alignment 和 metrics replay 逻辑。
- 可以完成 Phase 1 文档提交/push；后续进入 Phase 2 前必须重新定义 Phase 2 的测试矩阵。
