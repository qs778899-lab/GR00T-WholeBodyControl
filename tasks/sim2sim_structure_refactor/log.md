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

## 2026-06-24 Phase 1 完整确定性链路补测

用户指出：

- Phase 1 数据检测不能停留在数据 smoke。
- 第一层确定性链路验证需要用实际数据和固定日志，对 sim2sim 过程中每一个时间帧对齐相关步骤做输出对比。

补测环境：

- 项目根目录：`/home/lab/Desktop/GR00T-WholeBodyControl`
- Python 环境：`/home/lab/miniconda3/envs/sonic_eval/bin/python`
- `PYTHONPATH=/home/lab/Desktop/IsaacLab/source`
- 固定 run：`tmp/sim2sim_refactor/20260624_142140_phase1_validation/`
- 固定 motion：`eval_benchmark/robot_test/reach-2-003_chr00.pkl`
- pre-refactor 对比 commit：`63301c530353fd7ef68b27c915990a4ada30f06d`

执行命令：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/phase1_full_deterministic_validation.py
```

结果：

- streamer manifest：`sent_frames=1130`，`source_frames=930`，`blend_frames=200`，`motion_start_frame=200`，frame range `0..1129`。
- packed ZMQ round-trip：`chunks=38`，old/current unpack 输出完全一致。
- deploy cursor：source/applied valid rows 均为 `1670`，frame range `30..1083`，均为 streamer manifest 子集且单调。
- reference pose buffer：`buffer_frames=1130`，step-sync exact pose hits `995`，misses `0`，old/current 抽样 frame buffer 一致。
- step-sync alignment：`raw_rows=995`，`filtered_rows=995`，source range `41..1083`，unique source frames `995`，无相邻重复。
- 数据集 manifest smoke：`eval_benchmark/robot/*.pkl` 19 条、`eval_benchmark/smpl/*.pkl` 27 条、`data/smpl_filtered` 抽样 4 条全部通过。

metrics replay 命令：

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
- 与原 `manual_e2e_sync_results/reach-2-003_chr00_metrics.json` 对应关键字段一致。

通用门禁复跑：

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

通用门禁结果：

- 全部命令退出码为 0。
- C++ 默认 deploy target build/help 通过。
- C++/header diff 检查无输出，确认 Phase 1 未产生 C++/header 修改。

结论：

- Phase 1 的完整确定性链路补测已通过。
- 本次补测绕开 policy 随机性，验证了 streamer、packed ZMQ、deploy cursor、reference pose buffer、step-sync、metrics replay 和数据 manifest 的时间帧对齐逻辑没有因 Phase 1 结构搬移发生变化。
- Phase 1 可以按门禁提交/push 文档结果；进入 Phase 2 前仍必须重新明确 Phase 2 自身的测试范围和退出标准。
