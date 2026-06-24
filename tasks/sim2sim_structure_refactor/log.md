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

## 2026-06-24 Phase 2 启动记录

用户确认进入下一阶段。

Phase 2 范围冻结：

- 本阶段继续处理 `gear_sonic/utils/mujoco_sim/base_sim.py` 的 sim2sim 侵入。
- 新增 `gear_sonic/sim2sim/mujoco_hook.py`，把 sim2sim 的 reference visualization、eval logging、tracking overlay、link error plot 和 reference scene XML 生成集中到 hook 内。
- `base_sim.py` 只保留 hook 生命周期调用和原有时间顺序，不再承载 sim2sim CSV、plot、overlay、reference visualizer 细节。
- 本阶段不修改 C++/header 文件，不修改 deploy 主路径，不修改 streamer/metrics CLI。

Phase 2 测试计划：

- 通用门禁测试全部复跑。
- C++ 默认 deploy target build/help 与 C++ diff gate 全部复跑。
- 复用 Phase 1 fixed logs 做完整确定性链路 replay，并将产物写入新的 Phase 2 run 目录。
- 复用 Phase 1 fixed logs 重算 metrics，确认 `num_frames`、`lag_frames_log_vs_gt`、`step_sync_rows`、`mpjpe_g/l/pa` 关键字段一致。
- 如端到端环境稳定，重新跑一条 `eval_benchmark/robot_test/reach-2-003_chr00.pkl` strict alignment smoke；如复用 fixed logs，必须说明边界。

## 2026-06-24 Phase 2 完成记录

Run ID：`20260624_200458_phase2_hook_validation`

代码改动：

- 新增 `gear_sonic/sim2sim/mujoco_hook.py`。
- `base_sim.py` 不再直接实现 sim2sim eval logger、link error plot、tracking overlay、reference visualizer、reference XML clone/prefix 的业务细节。
- `base_sim.py` 只保留 hook 调用和原有关键时序：
  - `mj_step`
  - 在 reference robot update 前 capture actual body position
  - reference visualizer `poll/apply`
  - reference applied 后 `mj_forward`
  - new control frame 时写 step-sync
- 本阶段未修改任何 C++/header 文件。

通用门禁命令：

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

- Python compileall：通过。
- `run_sim_loop.py --help`：通过。
- `stream_motionlib_to_deploy.py --help`：通过。
- `compute_mujoco_tracking_metrics.py --help`：通过。
- 三个 shell wrapper `--help`：通过。
- `git diff --check`：通过。
- C++ 默认 deploy target build/help：通过。
- C++/header diff gate：无输出。

确定性链路 replay：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/deterministic_full/phase2_full_deterministic_validation.py
```

结果：

- streamer manifest：`sent_frames=1130`，`source_frames=930`，`blend_frames=200`，`motion_start_frame=200`。
- packed ZMQ round-trip：`chunks=38`，old/current unpack 输出完全一致。
- deploy cursor：source/applied valid rows 均为 `1670`，frame range `30..1083`，均为 streamer manifest 子集且单调。
- reference pose buffer：`buffer_frames=1130`，step-sync exact pose hits `995`，misses `0`。
- step-sync alignment：`raw_rows=995`，`filtered_rows=995`，source range `41..1083`，unique source frames `995`，无相邻重复。
- 数据集 manifest smoke：robot `19/19`，smpl `27/27`，smpl_filtered sampled `4/4`。

fixed logs metrics replay：

- `num_frames=995`
- `alignment.lag_frames_log_vs_gt=0`
- `alignment.source_frame_valid_rows=995`
- `alignment.step_sync_rows=995`
- `metrics_all.mpjpe_g=179.1855428355443`
- `metrics_all.mpjpe_l=87.76796772732865`
- `metrics_all.mpjpe_pa=43.50997555797189`
- 与 Phase 1 fixed metrics 对应字段一致。

新 hook 真实 E2E strict alignment smoke：

- 端口：pose `5826`，debug `5818`
- DDS domain：`112`
- motion：`eval_benchmark/robot_test/reach-2-003_chr00.pkl`
- output：`--output-type zmq`
- sim logs：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/e2e_hook_sim_logs`
- deploy logs：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/e2e_hook_deploy_logs`

结果：

- streamer 正常完成：`sent_frames=1130`。
- 新 hook 写出 `body_pos_w_14.csv`、`sim_source_frame_index.csv`、`sim2sim_step_sync_body_pos_w_14.csv`。
- `sim2sim_step_sync_body_pos_w_14.csv`：`743` data rows，source range `338..1080`。
- metrics JSON：`tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/e2e_hook_results/reach-2-003_chr00_metrics.json`
- metrics summary：
  - `num_frames=743`
  - `actual_source=step_sync_body_pos_w_14`
  - `gt_body_source=mujoco_ref_body_pos_w_14`
  - `alignment.mode=source_frame_index`
  - `lag_frames_log_vs_gt=0`
  - `source_frame_valid_rows=743`
  - `step_sync_rows=743`
  - `mpjpe_g=112.0429219746846`
  - `mpjpe_l=70.81177089518766`
  - `mpjpe_pa=30.77083514480582`

差异解释：

- 新 hook E2E 的有效帧数 `743` 少于 Phase 1 fixed run 的 `995`，但 strict alignment 仍为 `0 lag`。
- 本次 E2E stdout 显示 streamer 开始前和 early stream 阶段 robot 多次 fall/reset，reference visualizer 在 source frame `338` 后才稳定锁定，因此有效 step-sync 覆盖范围减少。
- 这属于端到端 policy/runtime 初始状态随机性和 reset 时机差异，不是 hook 结构调整导致的时间帧对齐逻辑变化。
- 固定日志 deterministic replay 和 metrics replay 已覆盖完整 `995` rows，并确认对齐逻辑和 metrics replay 未变。

进程清理：

- 本次 E2E 后已终止 domain `112`、ports `5818/5826` 对应 sim/deploy 进程。
- `ps` 检查未发现残留 `run_sim_loop.py` 或 `g1_deploy_onnx_ref` 测试进程。

Phase 2 结论：

- Phase 2 计划内测试全部成功完成。
- 本阶段结构优化没有改变 sim2sim CSV 字段、step-sync 写入规则、exact reference pose 对齐和 metrics replay 结果。
- 本阶段没有 C++/header 修改。
- 满足提交和 push 门禁；完成提交/push 后才允许进入 Phase 3。
