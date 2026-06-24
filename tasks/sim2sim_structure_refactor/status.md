# sim2sim 结构优化状态

更新时间：2026-06-24

## 当前门禁

- 当前阶段：Phase 1 已完成，完整确定性链路补测已通过。
- Phase 1 代码提交：`3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 远程：`origin main`
- 允许进入下一阶段：Phase 1 测试门禁已满足；进入 Phase 2 前仍必须先明确 Phase 2 的改动范围、测试命令和退出标准，并在完成 Phase 1 文档提交/push 后才能开始。
- C++/header 状态：Phase 1 未修改任何 C++/header 文件。
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

## 下一阶段

Phase 2 目标：

- 基于 base 的 `base_sim.py` 继续收敛为最小 hook。
- 保持 sim2sim reference visualization、step-sync logging、metrics 对齐语义不变。
- 不新增 C++ 修改。

进入 Phase 2 前必须：

- 明确 Phase 2 修改范围。
- 明确 Phase 2 全部测试命令。
- 明确 Phase 2 需要复用哪些 Phase 1 固定日志，哪些测试必须重新跑 baseline/refactor。
- 确认 Phase 1 大体积测试数据是否需要保留；如清理，必须先在 `log.md` 记录保留摘要和清理范围。
