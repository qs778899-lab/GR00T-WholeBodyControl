# sim2sim 结构优化状态

更新时间：2026-06-24

## 当前门禁

- 当前阶段：Phase 2 已完成，等待提交和 push。
- Phase 1 代码提交：`3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 远程：`origin main`
- Phase 1 完整测试文档提交：`ce69a40 Record completed Phase 1 deterministic sim2sim validation`
- 允许进入下一阶段：不允许进入 Phase 3，直到 Phase 2 提交并 push 到 `origin main`。
- C++/header 状态：Phase 2 未修改任何 C++/header 文件，C++ diff gate 无输出。
- 未提交的无关工作区改动：`.gitignore` 中 `tasks/` ignore 规则，非 Phase 1 提交内容。

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
