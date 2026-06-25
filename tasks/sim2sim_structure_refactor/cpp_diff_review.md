# C++ diff review and latency root cause analysis

更新时间：2026-06-25

## 结论摘要

用户已通过真机测试对比确认：加入 sim2sim 版代码后，真机时延严重变差。因此 C0 的意义不是普通 diff 归档，而是定位 current 相比 base 的 C++ 改动中哪些进入了真机热路径，并给出拆除方案。

当前最高风险结论：

1. `StateLogger` 新增的 `source_frame_index` / `applied_source_frame_index` 记录进入 control loop，每 tick 增加互斥锁访问、source frame 查询和 CSV 写入；CSV sink 使用 flush 型输出，真机上风险最高。
2. `g1_deploy_onnx_ref.cpp` 在每个 control tick 中两次调用 `input_interface_->GetSourceFrameIndex(...)`，且每 tick 调用 `UpdateAppliedSourceFrameIndex(...)`，这属于 sim2sim 对齐调试逻辑进入部署主循环。
3. `output_interface.hpp` / `zmq_output_handler.hpp` 将 raw reference target 和 source frame 字段加入 debug output，每次 publish 都会扩展 map/vector copy 和 msgpack payload；如果真机命令使用 `--output-type zmq` 或 `all`，会直接增加周期负担。
4. `LowStateHandler` 在 500Hz DDS callback 中每次调用 `steady_clock::now()`，并每秒 `std::endl` flush 一次 heartbeat；这虽然不是最大项，但属于不应默认进入真机回调的调试逻辑。
5. `enable_motion_recording_`、`enable_csv_logs`、`output-type zmq/all` 如果在真机命令中开启，会把大量磁盘 I/O 和 debug publish 带进实时路径；sim2sim 版本新增字段会进一步放大问题。

## 热路径证据

### 1. Control loop source frame 查询和 StateLogger 更新

文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`

关键位置：

- `3973-3987`：每个 control tick 在 `LogPostState(...)` 前调用 `GetSourceFrameIndex(...)`。
- `4000-4009`：每个 control tick 在 `CreatePolicyCommand()` 后再次调用 `GetSourceFrameIndex(...)`，然后调用 `UpdateAppliedSourceFrameIndex(...)`。

风险：

- 同一个 tick 查询两次 source frame，且无 sim2sim/debug 开关保护。
- 非 sim2sim 真机路径也会经过这些分支，只要 `state_logger_` 存在就执行。
- 这个逻辑本质只服务 strict alignment/debug，不应进入真机默认控制路径。

处理方案：

- C++ 合入 base 时原则上删除这段 source-frame 查询和 `UpdateAppliedSourceFrameIndex(...)` 调用。
- 如确实需要保留，只允许进入默认关闭的 `Sim2SimDebug` 路径，并且每 tick 最多查询一次。

### 2. StateLogger 每 tick CSV 写入和重复 applied frame 写入

文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/state_logger.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/state_logger.cpp`

关键位置：

- `state_logger.cpp:141-176`：`LogPostState(...)` 持有 `ring_mutex_`，写 token state、motion metadata、`source_frame_index`，若 `enable_csv_` 为真则调用 `appendTokenStateToCSV_(...)`。
- `state_logger.cpp:181-205`：`UpdateAppliedSourceFrameIndex(...)` 再次持有 `ring_mutex_`，并在 CSV enabled 时写 `applied_source_frame_index.csv`。
- `state_logger.cpp:376-384`：`appendTokenStateToCSV_(...)` 已经写了一次 `source_frame_index` 和 `applied_source_frame_index`。

风险：

- `applied_source_frame_index.csv` 在一个 tick 内可能被写两次：一次在 `appendTokenStateToCSV_(...)`，一次在 `UpdateAppliedSourceFrameIndex(...)`。
- `FileSink::writeLine(...)` 使用 `std::endl` 风格 flush 输出；新增两个 sink 会显著放大 CSV enabled 时的阻塞 I/O。
- `StateLogger` 原本是真机观测和调试基础设施，sim2sim 专属字段不应修改默认结构，更不应删除 base 中与真机相关的 battery/BMS 字段。

处理方案：

- C++ 合入 base 时回退 `StateLogger` 到 base 行为，不携带 sim2sim source-frame 字段。
- sim2sim 需要的 frame 对齐优先由 Python streamer manifest、MuJoCo step-sync CSV 和 metrics replay 提供。
- 如果 C++ 必须输出 consumed source frame，把现有 source-frame CSV/debug 写入代码块整体迁入默认关闭 `sim2sim_debug.cpp`，不修改 `StateLogger` 默认字段；CSV 写入必须异步、缓冲或低频，不能每 tick flush。

### 3. ZMQ debug output payload 扩展

文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/output_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp`

关键位置：

- `output_interface.hpp:151-153`：新增 `ref_base_trans_raw`、`ref_base_quat_raw`、`ref_body_q_raw`。
- `output_interface.hpp:266-271`：每次 output update 都把 target/raw/measured 数组 assign 到 `output_data_map_`。
- `zmq_output_handler.hpp:299-303`：msgpack 中加入 `source_frame_index` 和 `applied_source_frame_index`。
- `zmq_output_handler.hpp:403-410`：遍历 `output_data_map_` 打包全部 visualisation fields。

风险：

- 如果真机运行启用了 ZMQ debug output，payload 字段和 per-frame vector/map copy 增加。
- raw reference target 是 sim2sim 可视化/评测需要，不应默认进入真机 debug schema。

处理方案：

- 合入 base 时回退 debug schema。
- raw reference GT 从 Python pose stream / MuJoCo reference buffer 获取，不从 deploy debug output 获取。
- 如果必须保留，使用单独 `sim2sim_debug` topic 或 schema-version 字段，默认关闭。

### 4. DDS LowState callback heartbeat

文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`

关键位置：

- `2617-2654`：`LowStateHandler` 是 500Hz DDS callback，当前每次 callback 更新计数并调用 `steady_clock::now()`，每秒用 `std::endl` 输出 heartbeat。

风险：

- DDS callback 是真机输入热路径，不应默认执行周期性 debug 计时和 flush 输出。
- 单项开销可能不如 CSV 大，但会增加实时路径抖动。

处理方案：

- 回退 base 行为或挂到显式 debug flag。
- 真机默认路径不打印 heartbeat，不在每个 callback 做非必要时间查询。

### 5. Motion recording / CSV logging / output type 命令风险

文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`

关键位置：

- `enable_motion_recording_` 控制的 recorder 路径。
- `state_logger_` + `enable_csv_` 控制的 CSV 写盘路径。
- `output-type zmq/all` 控制的 realtime debug output 路径。

风险：

- 如果真机测试命令沿用了 sim2sim 命令中的 debug/recording 参数，会把大量磁盘和 ZMQ 输出放进实时路径。
- 即使 base 已有 logging，sim2sim 版额外 source-frame 字段和 raw payload 会放大开销。

处理方案：

- 真机命令默认必须关闭 motion recording、CSV debug、ZMQ realtime debug output，除非专门做日志采样。
- 代码层面也要保证 sim2sim debug 参数默认 false，并且关闭时没有 per-tick 文件、map copy、ZMQ pack、source-frame 查询。

## 逐文件分类初稿

| 文件 | 时延风险 | 建议分类 | 处理方案 |
|---|---|---|---|
| `src/g1_deploy_onnx_ref.cpp` | 高 | `drop_non_sim2sim` + `must_keep_default_off_debug_hook` 备选 | 删除默认 control loop source-frame 查询、`UpdateAppliedSourceFrameIndex`、LowState heartbeat；如 C++ 必须保留 consumed frame，单独默认关闭 hook。 |
| `include/state_logger.hpp` | 高 | `drop_non_sim2sim` | 回退 base，不在 StateLogger 默认结构加入 sim2sim frame 字段，不删除 battery/BMS 字段。 |
| `src/state_logger.cpp` | 高 | `drop_non_sim2sim` | 回退 base；sim2sim CSV 改 Python 旁路或默认关闭独立 sink。 |
| `include/output_interface/output_interface.hpp` | 中高 | `python_side_replacement` | raw reference target 不从 deploy 输出，改由 Python pose stream / MuJoCo buffer 提供。 |
| `include/output_interface/zmq_output_handler.hpp` | 中高 | `python_side_replacement` | 回退 base debug schema；source/applied frame 不进入默认 ZMQ payload。 |
| `include/input_interface/input_interface.hpp` | 中 | `drop_non_sim2sim` | 删除主接口 `GetSourceFrameIndex`，避免所有输入源被 sim2sim 字段污染。 |
| `include/input_interface/interface_manager.hpp` | 中 | `drop_non_sim2sim` | manager 不转发 sim2sim debug 字段。 |
| `include/input_interface/zmq_manager.hpp` | 中 | `smpl_protocol_separate_review` + `must_keep_default_off_debug_hook` 备选 | start/planner 行为回退；SMPL/protocol 与 sim2sim 去侵入分开评审。 |
| `include/input_interface/zmq_endpoint_interface.hpp` | 中 | `smpl_protocol_separate_review` + `python_side_replacement` | parser 大改不整块合入；raw pose/source frame 优先 Python 旁路。 |
| `include/input_interface/streamed_motion_merger.hpp` | 待确认 | `smpl_protocol_separate_review` | 与 SMPL/protocol 能力绑定，单独评审，不作为真机默认路径改动。 |
| `include/motion_data_reader.hpp` | 中 | `drop_non_sim2sim` | source frame storage 不进入 motion reader 公共语义。 |
| `include/robot_parameters.hpp` | 功能风险 | `drop_non_sim2sim` | 回退 base，不能删除真机相关参数。 |
| `include/input_interface/gamepad.hpp` | 功能风险 | `drop_non_sim2sim` | 回退 base，与 sim2sim 时延无直接关系。 |
| `include/input_interface/keyboard_handler.hpp` | 功能风险 | `drop_non_sim2sim` | 回退 base，与 sim2sim 时延无直接关系。 |
| `include/input_interface/ros2_input_handler.hpp` | 低 | `drop_non_sim2sim` | 默认回退 base。 |
| `tests/test_ros2.cpp` | 低 | `drop_non_sim2sim` | 回退 base 测试。 |

## 优化方案优先级

### P0：先恢复真机默认路径，不丢失 sim2sim 能力

- 真机部署目标中，原则上让 base 原有 C++ 文件的默认行为回到 base。
- 不把 sim2sim source-frame、raw target、debug output schema、CSV sink 带入默认真机路径。
- 不直接删除原有 sim2sim 对齐能力；必须迁移到独立、默认关闭的 `sim2sim_debug` C++ 模块，或由 Python/MuJoCo 旁路提供等价信息。
- 先确认 base C++ build/help 和真机命令参数。

### P1：优先用 Python/MuJoCo 旁路保留可旁路信息

- streamer manifest 记录发送帧号。
- packed pose stream 记录 raw pose / root / joint 数据。
- MuJoCo hook 记录 exact reference pose 和 step-sync actual/ref。
- metrics replay 用 `source_frame_index` 对齐，不依赖 deploy 默认 debug output。

### P2：必须保留的 deploy 对齐信息放入独立默认关闭 C++ hook

目标：

- 保留完整原有 sim2sim 时间帧对齐语义。
- 避免 sim2sim 调试逻辑进入真机默认热路径。
- 避免把功能实现写在 base 原有 C++ 文件中。

要求：

- 新增 hook 默认 false。
- 实现主体默认只放在两个文件中：
  - `include/sim2sim_debug/sim2sim_debug.hpp`
  - `src/sim2sim_debug/sim2sim_debug.cpp`
- 最保险的做法是把 base 原有 C++ 文件中 sim2sim 相关修改代码块整体迁出到 `sim2sim_debug.cpp`，不要过度拆分为多个新 C++ 文件，以减少控制流、时序和变量语义漂移。
- base 原有 C++ 文件只允许最小接入：
  - include hook 头文件。
  - 构造默认 disabled config/hook。
  - 调用 `OnStreamFrameDecoded(...)`、`OnControlTickObserved(...)`、`OnControlTickApplied(...)`。
- 关闭时不查询 source frame、不写 CSV、不 pack ZMQ 字段、不改变 StateLogger schema、不改变默认 ZMQ debug schema。
- 开启时每 tick 最多一次 source-frame 查询。
- CSV 不使用 per-line flush；优先异步、缓冲或低频 summary。

## 独立 hook 设计草案

建议文件：

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/
  include/sim2sim_debug/
    sim2sim_debug.hpp
  src/sim2sim_debug/
    sim2sim_debug.cpp
```

建议接口：

```cpp
struct Sim2SimDebugConfig {
  bool enabled = false;
  bool write_csv = false;
  bool publish_zmq = false;
  std::string logs_dir;
};

class Sim2SimDebugHook {
 public:
  explicit Sim2SimDebugHook(Sim2SimDebugConfig config);
  bool Enabled() const;

  void OnStreamFrameDecoded(int64_t stream_frame_index,
                            int64_t source_frame_index);
  void OnControlTickObserved(uint64_t tick_index,
                             int motion_frame,
                             int64_t source_frame_index);
  void OnControlTickApplied(uint64_t tick_index,
                            int64_t applied_source_frame_index);
};
```

必须逐帧保留的 mapping：

```text
stream_chunk/frame_index -> motion/source_frame_index
control_tick             -> source_frame_index
control_tick             -> applied_source_frame_index
metrics_row              -> source_frame_index
source_frame_index       -> reference pose
```

## 不影响原有逻辑功能的测试门禁

### 默认关闭路径

必须证明：

- C++ build/help 通过。
- 不新增 sim2sim CSV。
- 不创建 sim2sim ZMQ socket。
- 默认 `g1_debug` schema 与 base 一致。
- `StateLogger` 默认字段集合与 base 一致。
- `InputInterface` 公共虚接口与 base 一致。
- policy input tensor 和 action command 不变。
- keyboard/gamepad/ros2 行为不变。
- 控制循环耗时与 base 同量级，差异必须解释。

### sim2sim debug 开启路径

必须证明：

- hook 输出 `source_frame_index` / `applied_source_frame_index`。
- 旧逻辑与新 hook 的逐帧 mapping 一致：
  - `control_tick -> source_frame_index`
  - `control_tick -> applied_source_frame_index`
  - `metrics_row -> source_frame_index`
  - `source_frame_index -> reference pose`
- `robot_test` 真实 E2E strict alignment 保持 `lag_frames_log_vs_gt == 0`。
- 有效帧覆盖不能异常减少；如果减少，必须定位到动作边界、blend frame、控制 tick 或日志条件变化，不能模糊解释。
- `eval_benchmark/robot/*.pkl` 19 条、`eval_benchmark/smpl/*.pkl` 27 条、`data/smpl_filtered` 固定抽样 4 条全部完成 C1 计划内测试。

## 待确认信息

为了把根因从“高概率”确认成“实测结论”，需要记录用户真机测试命令中的以下参数：

- 是否开启 `--enable-csv-logs` 或类似 CSV logging。
- 是否开启 `--enable-motion-recording`。
- `--output-type` 是 `none`、`ros2`、`zmq` 还是 `all`。
- 是否使用 sim2sim 相关 ZMQ input/output 参数。
- 真机时延指标：control loop 周期、P95/P99、超时次数、CPU 占用、磁盘写入量。

在拿到命令前，当前最优先的代码方向仍然是：不要优化这些 C++ diff，而是默认从真机路径拆掉它们。
