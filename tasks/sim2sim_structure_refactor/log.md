# sim2sim 结构优化执行日志

更新时间：2026-06-24

## 2026-06-24 文档迁移

用户要求：

- 在文档中补充：每一个 phase 的测试全部完成、确认没有问题后，可以清理测试产生的数据，避免占用磁盘。
- 将旧计划文档 `workspace/plan_/sim2sim_structure_refactor_plan.md` 迁移到 `tasks/sim2sim_structure_refactor/`。
- 按新结构组织文档内容，便于严格按照工作流执行。

已执行：

- 将旧计划文档迁移为 `tasks/sim2sim_structure_refactor/plan.md`。
- 新增/补齐：
  - `tasks/sim2sim_structure_refactor/status.md`
  - `tasks/sim2sim_structure_refactor/test_matrix.md`
  - `tasks/sim2sim_structure_refactor/log.md`
- 删除空占位文件：
  - `tasks/sim2sim_structure_refactor/plan.md`
  - `tasks/sim2sim_structure_refactor/status.md`
  - `tasks/sim2sim_structure_refactor/log.md`
  - `tasks/sim2sim_structure_refactor/test_matrix.md `，该文件名末尾有空格。
- 在 `plan.md` 最高优先级门禁和清理要求中补充 phase 测试数据清理规则。

注意：

- `tasks/` 当前被 `.gitignore` 忽略；如需提交这些文档，必须使用 `git add -f tasks/sim2sim_structure_refactor/...`。
- 旧路径 `workspace/plan_/sim2sim_structure_refactor_plan.md` 不再维护。

## 2026-06-24 测试层级修正

用户指出：

- Phase 1 中的数据检测是数据 smoke，并没有实际对比数据结果。
- `plan.md` 中第一层确定性链路验证需要大量实际数据运行测试和逐环节对比。
- 当前文档容易让工作流误以为数据 smoke 已经覆盖完整确定性链路。

修正：

- 在 `status.md` 中明确 Phase 1 的数据 smoke 和单条端到端 smoke 不能等价于完整第一层确定性链路验证。
- 在 `test_matrix.md` 中新增测试层级说明，区分数据 smoke、确定性链路验证、端到端统计验证。
- 在 `test_matrix.md` 中列出 Phase 2 前必须补齐的 7 个确定性链路环节：
  - streamer manifest
  - packed ZMQ round-trip
  - deploy 播放游标
  - reference pose buffer
  - step-sync logger
  - metrics replay
  - visualization/metrics 同源
- 在 `plan.md` 的 Phase 1 和 Phase 1 门禁结论中补充边界说明：Phase 1 通过只适用于源码搬移范围，不代表完整 baseline/refactor 数据结果对比完成。

## 2026-06-24 Phase 1 已完成摘要

提交：

- `3ce6143 Refactor sim2sim helpers out of mujoco base sim`
- 已 push 到 `origin main`。

核心测试结果：

- Python compileall：通过。
- sim2sim Python/shell entrypoint help：通过。
- `git diff --check`：通过。
- C++ deploy target build/help：通过。
- deterministic moved-helper check：通过。
- 数据 smoke：`50/50` 通过。
- strict frame alignment E2E：
  - `num_frames=995`
  - `source_frame_valid_rows=995`
  - `step_sync_rows=995`
  - `lag_frames_log_vs_gt=0`

测试产物：

- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/`

清理状态：

- 暂未清理 Phase 1 原始日志。
- 后续如确认不再需要复查，可按 `test_matrix.md` 的清理规则清理大体积日志，并保留 summary 与 metrics JSON 摘要。
