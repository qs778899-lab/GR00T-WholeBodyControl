# sim2sim 结构优化状态

更新时间：2026-06-24

## 当前门禁

- 当前阶段：C++ Phase C0 已开始，当前只做 C++ 差异审查和去侵入验证计划，不直接修改 C++。
- Phase 1 代码提交：`3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 远程：`origin main`
- Phase 1 完整测试文档提交：`ce69a40 Record completed Phase 1 deterministic sim2sim validation`
- Phase 2 提交：`26cac18 Isolate sim2sim MuJoCo hook from base simulator`
- Phase 2 数据覆盖门禁补充提交：`ba53983 Clarify per-phase full data coverage gate`
- Phase 3 状态：取消执行。`tools/sonic_eval/*.py` 均视为增量拓展文件，本轮不做结构优化、不修改。
- 允许进入下一阶段：进入 C++ Phase C0；C0 完成前不允许修改 C++，不允许进入 C1。
- C++/header 状态：Phase 2 未修改任何 C++/header 文件，C++ diff gate 无输出。
- 未提交的无关工作区改动：`.gitignore` 中 `tasks/` ignore 规则，非 Phase 1 提交内容。

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

- 针对相对 base 仓库不同的 16 个 C++/header 同路径文件，做逐文件、逐代码块去侵入审查。
- 优先证明最终合入 base 时可以不携带这些 C++ diff。
- C0 不直接修改 C++；只生成审查报告、patch 分类和验证证据。

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
- 默认 C++ build/help 通过。
- C++ 默认运行路径性能风险结论明确：C0 不修改 C++，因此默认路径无新增开销。
- 如果无法证明某个 C++ diff 可丢弃，必须进入 C1 方案评审，不能直接修改代码。

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
