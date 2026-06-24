# sim2sim 结构优化状态

更新时间：2026-06-24

## 当前门禁

- 当前阶段：Phase 1 已完成。
- 当前提交：`3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 远程：`origin main`
- 允许进入下一阶段：仅允许进入 Phase 2 的计划细化和基线补测；不允许直接开始更深代码重构，直到 `test_matrix.md` 中 Phase 2 必需的确定性链路验证补齐并通过。
- C++/header 状态：Phase 1 未修改任何 C++/header 文件。
- 未提交的无关工作区改动：`.gitignore` 中 `tasks/` ignore 规则，非 Phase 1 提交内容。

## Phase 1 结论

Phase 1 目标是把 `gear_sonic/utils/mujoco_sim/base_sim.py` 中的 sim2sim 专属 Python 逻辑迁移到新增包 `gear_sonic/sim2sim/`，降低对 base 已有文件的侵入。

已确认：

- `base_sim.py` 只保留现有 hook 调用和原执行顺序。
- 搬移的类与常量源码通过 deterministic check 确认一致。
- Python compile/help、shell help、C++ build/help、数据 smoke、端到端 strict frame alignment 均通过。
- 端到端 strict metrics 产物显示 `lag_frames_log_vs_gt=0`、`step_sync_rows=995`。

未完成 / 不能误判为已完成：

- Phase 1 的“数据 smoke”只验证数据可加载、字段/shape/finite 正常，不是实际运行结果对比。
- Phase 1 的 deterministic check 只覆盖“代码搬移后类/常量源码一致、局部 logger/ZMQ/overlay 行为一致”，不是 `plan.md` 中第一层 7 个环节的完整确定性链路验证。
- Phase 1 的端到端测试是单条 motion 的 strict alignment smoke，只证明链路能跑通且本次 `source_frame_index` 对齐为 0 lag；它不是 baseline/refactor 全量数据对比，也不证明 policy 随机运行 metrics 分布不变。
- 因此，后续 Phase 2 不能把 Phase 1 的数据 smoke 当作“所有实际数据对比已经完成”。Phase 2 开始前必须先补齐基线/重构后的确定性对比计划和命令。

## 下一阶段

Phase 2 目标：

- 基于 base 的 `base_sim.py` 继续收敛为最小 hook。
- 保持 sim2sim reference visualization、step-sync logging、metrics 对齐语义不变。
- 不新增 C++ 修改。

进入 Phase 2 前必须：

- 明确 Phase 2 修改范围。
- 明确 Phase 2 全部测试命令。
- 补齐第一层确定性链路验证的可执行测试脚本或命令，至少覆盖 streamer manifest、packed ZMQ round-trip、reference pose buffer、step-sync CSV invariant、metrics replay/alignment manifest。
- 明确哪些测试使用已录制 Phase 1 日志，哪些测试需要重新跑 baseline/refactor。
- 确认 Phase 1 大体积测试数据是否需要保留；如清理，必须先在 `log.md` 记录保留摘要和清理范围。
