# sim2sim 结构优化状态

更新时间：2026-06-24

## 当前门禁

- 当前阶段：Phase 1 已完成。
- 当前提交：`3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 远程：`origin main`
- 允许进入下一阶段：允许，但进入 Phase 2 前必须重新确认本文件和 `test_matrix.md` 中的 Phase 2 测试范围。
- C++/header 状态：Phase 1 未修改任何 C++/header 文件。
- 未提交的无关工作区改动：`.gitignore` 中 `tasks/` ignore 规则，非 Phase 1 提交内容。

## Phase 1 结论

Phase 1 目标是把 `gear_sonic/utils/mujoco_sim/base_sim.py` 中的 sim2sim 专属 Python 逻辑迁移到新增包 `gear_sonic/sim2sim/`，降低对 base 已有文件的侵入。

已确认：

- `base_sim.py` 只保留现有 hook 调用和原执行顺序。
- 搬移的类与常量源码通过 deterministic check 确认一致。
- Python compile/help、shell help、C++ build/help、数据 smoke、端到端 strict frame alignment 均通过。
- 端到端 strict metrics 产物显示 `lag_frames_log_vs_gt=0`、`step_sync_rows=995`。

## 下一阶段

Phase 2 目标：

- 基于 base 的 `base_sim.py` 继续收敛为最小 hook。
- 保持 sim2sim reference visualization、step-sync logging、metrics 对齐语义不变。
- 不新增 C++ 修改。

进入 Phase 2 前必须：

- 明确 Phase 2 修改范围。
- 明确 Phase 2 全部测试命令。
- 确认 Phase 1 大体积测试数据是否需要保留；如清理，必须先在 `log.md` 记录保留摘要和清理范围。
