# sim2sim 结构优化状态

更新时间：2026-06-25

## 当前门禁

- 当前阶段：C++ Phase C1b 已完成，代码和文档已提交并 push 到 `origin main`。
- Phase 1 代码提交：`3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 远程：`origin main`
- Phase 1 完整测试文档提交：`ce69a40 Record completed Phase 1 deterministic sim2sim validation`
- Phase 2 提交：`26cac18 Isolate sim2sim MuJoCo hook from base simulator`
- Phase 2 数据覆盖门禁补充提交：`ba53983 Clarify per-phase full data coverage gate`
- Phase 3 状态：取消执行。`tools/sonic_eval/*.py` 均视为增量拓展文件，本轮不做结构优化、不修改。
- C++ Phase C1b 提交：`385fbfb Complete C++ sim2sim hook isolation`
- 允许进入下一阶段：是。当前没有未闭环测试阻塞。
- C++/header 状态：相对 base 仓库 `/home/lab/Desktop/LHM-Robot` 的 `feat/s0_training`，C++ diff allowlist gate 通过，unexpected list 为空。
- 未提交的无关工作区改动：`.gitignore` 中 `tasks/` ignore 规则，非 Phase 1 提交内容。

## C++ Phase C1b 完成记录

Run ID：`20260625_cpp_c1_final_validation`

结构结论：

- 已将 13 个非 hook C++/header/test 差异恢复到 base 等价，避免把 keyboard/gamepad/ros2/motion reader/robot parameters/test 等非 sim2sim 改动带入团队分支。
- 保留默认关闭的 C++ sim2sim debug hook。
- sim2sim 需要的 source-frame window tracking、raw reference output helper、enabled-only CSV/ZMQ 字段集中在 `sim2sim_debug` 新模块和最小接入点。
- `streamed_motion_merger.hpp` 只保留 streamed motion 必需的 `body_pos` 承接，避免 enabled E2E raw reference 全局位置退化为零。

7 项最终验证全部通过：

- default-off ZMQ schema：通过，`31` keys，不含 `source_frame_index` / `applied_source_frame_index`，deploy logs 不生成对应 CSV。
- deterministic fixed replay：通过，`num_frames=995`，`lag_frames_log_vs_gt=0`，source/applied cursor 单调且属于 fixed stream manifest。
- StateLogger unit：通过，default-off 不生成 source-frame CSV，enabled 生成两份 CSV。
- ZMQOutputHandler schema：通过，default-off `31` keys，enabled `36` keys。
- C++ diff allowlist gate：通过，unexpected list 为空。
- default-off timing 多样本：通过，解析到 `43` 条 timing，3 个 5-sample 窗口 obs-to-motor 平均值为 `536.8us`、`593.2us`、`484.6us`。
- enabled E2E 多 motion：通过，4 条 `eval_benchmark/robot/*.pkl` 全部 `0 lag`：
  - `reach-1-001_chr00`：`735` frames，MPJPE-G `112.477mm`。
  - `reach-2-003_chr00`：`1007` frames，MPJPE-G `175.410mm`。
  - `reach-3-002_chr00`：`859` frames，MPJPE-G `121.746mm`。
  - `reach-4-004_chr00`：`827` frames，MPJPE-G `83.722mm`。

后续清理：

- 已完成提交和 push。
- 可以按门禁清理本阶段 `tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/results/` 中大体积原始运行日志，只保留 summary/metrics 索引或按需要压缩归档。

## C++ Phase C1 Final Validation 阻塞

Run ID：`20260625_cpp_c1_final_validation`

已通过：

- default-off ZMQ schema：通过。
- deterministic fixed replay：通过。
- StateLogger unit：通过。
- ZMQOutputHandler default-off/enabled schema：通过。
- default-off timing 多样本：通过。
- enabled E2E 多 motion：通过，`eval_benchmark/robot` 下 4 条 motion 均 `0 lag`。

未通过：

- C++ diff allowlist gate：失败。

阻塞原因：

- C1 允许保留的 C++ 差异应集中在默认关闭 sim2sim debug hook 及最小接入点。
- 当前分支除 hook allowlist 外，仍有 `input_interface/*`、`motion_data_reader.hpp`、`robot_parameters.hpp`、`output_interface.hpp`、`test_ros2.cpp` 等非 hook 文件与 base 不一致。
- 这不满足“尽量不要修改 base C++ 文件、避免影响真机部署和时延”的核心目标。

下一步：

- 逐文件移除或迁移上述非 hook C++ diff。用户确认这些文件也必须遵循“整体迁出到新 C++ 文件”的原则：
  - 不允许把 source-frame、stream debug、teleop latency、pause/stop 行为改动继续散落在 base 原文件中。
  - 需要保留的 sim2sim 功能必须整体迁入 `sim2sim_debug` 相关新增 C++ 文件。
  - 不属于 sim2sim 必需功能的热路径改动恢复到 base 等价。
- 保留默认关闭 C++ sim2sim debug hook 的最小接入。
- 重跑 C1 final validation 全部 7 项测试。
- 只有 7 项全部通过后，才能 commit/push 并标记阶段完成。

## C++ Phase C1b 迁出规则

C1b 目标：

- 不扩大 C++ allowlist。
- 将 13 个非 hook 文件中的 sim2sim/source-frame/stream debug 逻辑整体迁出到新增 C++ 模块。
- 原 base 文件只保留最小接入点；不能在原文件中保留大段 sim2sim 专属逻辑。

初步分类：

- 需要迁出到新 C++ 模块：
  - source-frame 查询链路：`InputInterface::GetSourceFrameIndex`、`InterfaceManager::GetSourceFrameIndex`、`ZMQManager::GetSourceFrameIndex`、`ZMQEndpointInterface::GetSourceFrameIndex`。
  - source-frame 存储链路：`MotionSequence::SourceFrameIndices`、`StreamedMotionMerger` 写入/复制 source-frame indices。
  - streamed body position debug 数据：`StreamedMotionMerger::IncomingData::body_pos` 及复制逻辑。
  - raw reference ZMQ output fields：`ref_base_trans_raw`、`ref_base_quat_raw`、`ref_body_q_raw`。
- 应恢复到 base 等价，除非后续证明是 sim2sim 必需：
  - keyboard/gamepad/ros2/zmq 中 `pause_control` 删除、`O/X/S/Q` 语义变化。
  - `teleop_latency_logger.hpp` 删除及 ZMQ latency cap 参数变化。
  - `DEBUG_LOGGING=true` 默认变化。
  - motion reset 时从 temporary motion 改回 reader shared motion 的行为变化。
  - `HG_BMS_STATE_TOPIC` 删除。
  - `test_ros2.cpp` include 清理。

C1b 测试门禁：

- C++ diff allowlist gate 必须通过：非 hook 文件不能再出现在 unexpected list。
- C1 final validation 7 项全部重跑且全部通过。
- 如果 C1b 为保持 enabled sim2sim 功能新增独立 C++ 文件，这些文件必须只在 `--enable-sim2sim-debug` 或 sim2sim 专用路径中启用。

## 数据覆盖门禁

- 从下一阶段开始，每个 phase 开始前必须明确本阶段挑选测试数据清单。
- 清单中的所有数据都必须完成本阶段计划内的完整测试运行；不能只跑部分数据，不能只记录成功样例，不能把单条 smoke 结果当作全数据结果。
- 如果某类数据不能完整跑 sim/deploy/policy A/B/C/D 端到端链路，必须提前写明降级为 deterministic replay、manifest smoke 或 adapter smoke 的原因、风险和覆盖边界。
- 任一挑选数据未运行、失败、结果不确定，或有效帧覆盖异常减少且无法解释，本 phase 不能标记为完成，不能进入下一阶段。
- 该规则已写入 `plan.md` 和 `test_matrix.md`，后续 phase 必须按该规则执行。

## Phase 3 取消记录

- Phase 3 原计划处理 `tools/sonic_eval/*.py` 内部模块化。
- 用户明确要求：第三阶段不需要执行，`tools/sonic_eval/*.py` 都是增量拓展文件，不修改。
- 因此 Phase 3 标记为 skipped，不做代码修改，不跑 Phase 3 数据清单，不进入 Phase 3 提交。

## C++ Phase C0 启动门禁

C0 目标：

- 针对相对 base 仓库不同的 16 个 C++/header 同路径文件，先定位真机时延回归风险，再做逐文件、逐代码块去侵入审查。
- 优先证明最终合入 base 时可以不携带这些 C++ diff。
- C0 不直接修改 C++；只生成审查报告、patch 分类和验证证据。
- C0 不能再被理解为普通“差异归档”。用户已经通过真机对比确认 sim2sim 版代码严重影响时延，因此 C0 必须输出当前 C++ diff 进入控制热路径的证据、优先级和拆除方案。

C0 必须产出：

- `tmp/sim2sim_refactor/<run_id>/cpp_base_snapshot/`：base 分支 16 个 C++/header 文件快照。
- `tmp/sim2sim_refactor/<run_id>/cpp_diff_patches/`：
  - `drop_non_sim2sim.patch`
  - `python_side_replacement.patch`
  - `smpl_protocol_separate_review.patch`
  - `must_keep_default_off_debug_hook.patch`
- `tasks/sim2sim_structure_refactor/cpp_diff_review.md`：逐文件、逐代码块分类表和结论。
- `tasks/sim2sim_structure_refactor/cpp_phase_c0_test_report.md`：C0 测试命令、环境、结果。

C0 数据清单：

- `eval_benchmark/robot_test/*.pkl`：全部 1 条，用于 deterministic replay、metrics replay、必要时真实 E2E strict alignment。
- `eval_benchmark/robot/*.pkl`：全部 19 条，用于 streamer/data manifest smoke，证明 Python 旁路可覆盖 robot 数据。
- `eval_benchmark/smpl/*.pkl`：全部 27 条，用于 SMPL adapter/manifest smoke，标记 SMPL/protocol 相关 C++ diff 是否属于单独评审。
- `data/smpl_filtered` 固定抽样 4 条，用于 SMPL filtered adapter/manifest smoke：
  - `Idle_Left_001__A017.pkl`
  - `Jump_002__A017.pkl`
  - `Loop_Forward_Walk_001__A017.pkl`
  - `Neutral_stoop_down_001__A057.pkl`

C0 通过标准：

- 16 个 C++/header diff 都有明确分类：drop / python-side replacement / SMPL separate review / must-keep default-off debug hook。
- 对于每个 proposed drop 的 C++ diff，都必须说明 sim2sim 当前由哪个 Python 旁路、manifest 或 fixed log replay 覆盖。
- 对真机时延相关 diff 必须明确：
  - 是否在 DDS callback、control loop、StateLogger、ZMQ debug output、文件 I/O 等热路径执行。
  - 是否默认启用或会被常用真机命令启用。
  - 拆除、默认关闭或替代到 Python 旁路后的预期影响。
- 默认 C++ build/help 通过。
- C++ 默认运行路径性能风险结论明确：C0 本身不新增修改，但必须说明 current 相比 base 已存在的 sim2sim C++ diff 是否已经造成默认路径或常用真机路径开销。
- 如果无法证明某个 C++ diff 可丢弃，必须进入 C1 方案评审，不能直接修改代码。

## C++ Phase C0 完成记录

Run ID：`20260625_cpp_c0_review`

报告：

- `tasks/sim2sim_structure_refactor/cpp_phase_c0_test_report.md`
- `tasks/sim2sim_structure_refactor/cpp_diff_review.md`

产物：

- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_base_snapshot/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_current_snapshot/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_raw/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_patches/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/`

测试结果：

- C++ build/help：通过。
- C++/header diff gate：无输出。
- 16 个 C++/header diff 分类：完成。
- deterministic replay：通过。
- metrics replay：通过。
- 数据覆盖：
  - `robot_test` 1/1。
  - `robot` 19/19。
  - `smpl` 27/27。
  - `smpl_filtered` 固定抽样 4/4。

关键指标：

- `num_frames=995`
- `lag_frames_log_vs_gt=0`
- `step_sync_rows=995`
- `source_frame_valid_rows=995`
- `reference_pose_exact_hits=995`
- `reference_pose_misses=0`

结论：

- C0 没有修改 C++/header 源码。
- C0 测试全部成功完成。
- 可以进入 C1，但 C1 必须严格按“少量文件整块迁出 + 默认关闭 hook + 逐帧 mapping 等价测试 + 性能对比”执行。

## C++ Phase C1 方向修正

用户确认：

- 从保留完整原有逻辑和功能的角度，保留一个默认关闭的 C++ sim2sim debug hook 是合理方案。
- 这些功能模块的代码不应该写在原有 base C++ 文件中，应单独写功能 C++ 文件，再通过最小 include/call 接入 base 原有文件。

当前 C1 方案：

- 新增独立模块，但默认只允许少量文件整块承接旧逻辑：
  - `include/sim2sim_debug/sim2sim_debug.hpp`
  - `src/sim2sim_debug/sim2sim_debug.cpp`
- C++ 修改最保险的方式是把 base 原有 C++ 文件中 sim2sim 相关修改代码块整体迁出，不把这些代码过度拆分到多个新文件中。
- base 原有 C++ 文件只允许最小接入：
  - include hook 头文件。
  - 构造默认 disabled config/hook。
  - 在必要位置调用 `OnStreamFrameDecoded(...)`、`OnControlTickObserved(...)`、`OnControlTickApplied(...)`。
- 默认关闭时必须与 base 真机路径等价：
  - 不新增文件 I/O。
  - 不新增 ZMQ publish。
  - 不改默认 debug schema。
  - 不改 policy 输入、action command、keyboard/gamepad/ros2 行为。
  - 不改 `StateLogger` 默认字段集合。
- 开启时必须保留旧 sim2sim 时间帧对齐语义，并通过逐帧 mapping 对比验证：
  - `control_tick -> source_frame_index`
  - `control_tick -> applied_source_frame_index`
  - `metrics_row -> source_frame_index`
  - `source_frame_index -> reference pose`

状态：

- C1 仍未开始代码修改。
- C0 必须先完成 hook 需求冻结、base 文件最小接入点冻结、旧逻辑等价测试设计。

## Phase 1 结论

Phase 1 目标是把 `gear_sonic/utils/mujoco_sim/base_sim.py` 中的 sim2sim 专属 Python 逻辑迁移到新增包 `gear_sonic/sim2sim/`，降低对 base 已有文件的侵入。

已确认：

- `base_sim.py` 只保留现有 hook 调用和原执行顺序。
- 搬移的类与常量源码通过 deterministic check 确认一致。
- Python compile/help、shell help、C++ build/help、数据 smoke、端到端 strict frame alignment 均通过。
- 端到端 strict metrics 产物显示 `lag_frames_log_vs_gt=0`、`step_sync_rows=995`。
- Phase 1 完整确定性链路补测已通过，覆盖 streamer manifest、packed ZMQ round-trip、deploy cursor、reference pose buffer、step-sync alignment、metrics replay、数据集 manifest smoke。

Phase 1 补测结果：

- 固定 motion：`eval_benchmark/robot_test/reach-2-003_chr00.pkl`，`motion_name=reach-2-003_chr00`。
- `stream_manifest.csv`：`sent_frames=1130`，`blend_frames=200`，`motion_frames=930`，`motion_start_frame=200`，frame range `0..1129`。
- packed ZMQ round-trip：`chunks=38`，old/current unpack 输出完全一致。
- deploy cursor：source/applied rows 均为 `1670` valid，frame range `30..1083`，均为 streamer manifest 子集且单调。
- reference pose buffer：`buffer_frames=1130`，step-sync exact pose hits `995`，misses `0`，抽样 old/current frame buffer 一致。
- step-sync alignment：`raw_rows=995`，`filtered_rows=995`，source range `41..1083`，unique source frames `995`，无相邻重复。
- metrics replay：从固定 CSV 日志重算后 `num_frames=995`、`lag_frames_log_vs_gt=0`、`step_sync_rows=995`，`mpjpe_g/l/pa` 与原 metrics JSON 一致。
- 数据集 manifest smoke：`eval_benchmark/robot/*.pkl` 19 条、`eval_benchmark/smpl/*.pkl` 27 条、`data/smpl_filtered` 抽样 4 条全部通过。

边界说明：

- Phase 1 已补齐的是绕开 policy 随机性的确定性链路验证，不是 policy 随机端到端多次统计分布验证。
- 数据 smoke 仍不能单独替代实际链路对比；本次通过依赖完整确定性补测脚本和固定端到端日志 replay。
- Phase 2 如修改更深 hook 或 C++ 边界，必须重新按 Phase 2 计划跑该阶段所有测试。

## Phase 2 执行范围

Phase 2 目标：

- 基于 base 的 `base_sim.py` 继续收敛为最小 hook。
- 保持 sim2sim reference visualization、step-sync logging、metrics 对齐语义不变。
- 不新增 C++ 修改。

Phase 2 本次改动范围：

- 新增 `gear_sonic/sim2sim/mujoco_hook.py`，集中管理 sim2sim MuJoCo hook 生命周期。
- 从 `gear_sonic/utils/mujoco_sim/base_sim.py` 移出：
  - sim2sim eval logger 初始化和每帧写入编排。
  - link error plot 初始化和 push 编排。
  - tracking overlay 初始化、更新、render。
  - reference visualizer 初始化、poll/apply、toggle/reset/close。
  - reference visualization scene XML 复制和 prefix 逻辑。
- `base_sim.py` 只保留：
  - reference scene 创建的单行调用。
  - joint discovery 时跳过 reference prefix。
  - hook 初始化。
  - `capture_actual_body_pos_pre_ref -> update_reference_visualization -> mj_forward -> log_frame` 的原顺序。
  - viewer render、keyboard toggle、reset、close 的 hook 转发。
- Phase 2 不修改 C++/header，不修改 streamer/metrics CLI，不修改 CSV 文件名和字段。

Phase 2 已完成改动：

- 新增 `gear_sonic/sim2sim/mujoco_hook.py`。
- `base_sim.py` 删除 sim2sim eval logger、link error plot、tracking overlay、reference visualizer、reference XML clone/prefix 的具体业务实现。
- `base_sim.py` 保留的 sim2sim 相关逻辑收敛为：
  - reference scene 创建调用。
  - `REFERENCE_NAME_PREFIX` 跳过 reference robot joints。
  - `Sim2SimMujocoHook` 生命周期调用。
  - `capture_actual_body_pos_pre_ref -> update_reference_visualization -> mj_forward -> log_frame` 时序。
  - viewer、keyboard、reset、close 的 hook 转发。

Phase 2 测试结果：

- 通用门禁测试：全部通过。
- C++ 默认 deploy target build/help：通过。
- C++/header diff gate：无输出。
- Phase 2 deterministic replay：通过。
- fixed logs metrics replay：关键字段与 Phase 1 fixed metrics 一致。
- 新 hook 真实 E2E strict alignment smoke：通过，`lag_frames_log_vs_gt=0`。

Phase 2 关键产物：

- Run ID：`20260624_200458_phase2_hook_validation`
- 确定性链路 summary：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/deterministic_full/phase1_full_deterministic_validation_summary.json`
- fixed logs metrics replay：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/deterministic_full/metrics_replay_from_fixed_logs.json`
- 新 hook E2E metrics：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/e2e_hook_results/reach-2-003_chr00_metrics.json`
- 新 hook E2E summary：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/e2e_hook_results/summary.json`

Phase 2 必须执行的测试：

- 通用门禁测试全部执行。
- C++ 默认 deploy target build/help 与 C++ diff gate 全部执行。
- Phase 1 完整确定性链路脚本复制为 Phase 2 run 并执行，确认 streamer、packed ZMQ、deploy cursor、reference pose buffer、step-sync alignment、metrics replay、dataset manifest smoke 全部通过。
- 使用 Phase 1 fixed logs 重新 replay metrics，确认关键字段一致。
- 单条 `eval_benchmark/robot_test/reach-2-003_chr00.pkl` strict frame alignment smoke 如环境允许重新跑；如果复用 Phase 1 fixed logs，必须在 `log.md` 明确原因和覆盖边界。

Phase 2 退出标准：

- 所有 Phase 2 计划内测试退出码为 0。
- `base_sim.py` sim2sim 业务细节显著减少，只保留 hook 调用。
- CSV 文件名、字段、step-sync 写入规则、exact reference pose 逻辑不变。
- C++/header diff 为空。
- 文档记录命令、环境、结果、失败原因或差异解释。
- commit 并只 push 到 `origin main`。

## C++ Phase C1 Status

状态：完成，允许提交和 push。

Run ID：`20260625_cpp_c1_debug_hook`

改动范围：

- 新增 `sim2sim_debug` C++ hook 文件。
- 默认关闭 `g1_deploy_onnx_ref` 中的 source-frame 查询、applied-source update、source-frame CSV、ZMQ source-frame fields。
- 通过 `--enable-sim2sim-debug` 显式恢复 sim2sim debug 输出。
- `deploy.sh` 支持转发该开关。

测试结论：

- C++ build/help：通过。
- `deploy.sh` syntax/help：通过。
- 静态默认关闭 gate：通过。
- 确定性 replay：通过，`995` rows，`lag_frames_log_vs_gt=0`。
- fixed-log metrics replay：通过，`995` rows，`lag_frames_log_vs_gt=0`。
- enabled E2E：通过，`769` rows，`lag_frames_log_vs_gt=0`。
- default-off stream smoke：通过，无 source-frame CSV，loop timing 同量级。

关键产物：

- `tasks/sim2sim_structure_refactor/cpp_phase_c1_test_report.md`
- `tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/deterministic_full/phase1_full_deterministic_validation_summary.json`
- `tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/deterministic_full/metrics_replay_from_fixed_logs.json`
- `tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/e2e_hook_results/summary.json`
- `tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/default_off_stream_smoke/summary.json`

注意：

- `default_off_smoke` 和 `default_off_smoke_rerun` 是测试配置不足，不作为通过结果；最终有效结果是 `default_off_stream_smoke`。
- C1 后进入 C2 前必须先 commit 并只 push 到 `origin`。
