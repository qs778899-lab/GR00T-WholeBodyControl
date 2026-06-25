# sim2sim 结构优化执行日志

更新时间：2026-06-25

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

## 2026-06-24 Phase 3 启动记录

用户确认：如果 Phase 2 成功，可以进入下一阶段。

确认结果：

- Phase 2 已完成。
- Phase 2 commit：`26cac18 Isolate sim2sim MuJoCo hook from base simulator`
- 数据覆盖门禁补充 commit：`ba53983 Clarify per-phase full data coverage gate`
- 两个提交均已 push 到 `origin main`。
- 当前只进入 Phase 3 的范围冻结和数据清单确认，不开始代码修改。

Phase 3 初始目标：

- 评估并整理 `tools/sonic_eval/*.py` 中 streamer / metrics / adapter 的内部重复逻辑。
- 保持现有 CLI、输出文件名、CSV/JSON 字段不变。
- 不修改 C++/header，不修改 deploy 主路径。

Phase 3 挑选测试数据清单：

- `eval_benchmark/robot_test/*.pkl`：全部 1 条。
- `eval_benchmark/robot/*.pkl`：全部 19 条。
- `eval_benchmark/smpl_test/*.pkl`：全部 1 条。
- `eval_benchmark/smpl/*.pkl`：全部 27 条。
- `data/smpl_filtered` 抽样 4 条：
  - `Idle_Left_001__A017.pkl`
  - `Jump_002__A017.pkl`
  - `Loop_Forward_Walk_001__A017.pkl`
  - `Neutral_stoop_down_001__A057.pkl`

Phase 3 测试层级：

- `robot_test`：deterministic replay、metrics replay、至少 1 条真实 A/B/C/D strict alignment E2E。
- `robot`：全量 streamer/data manifest smoke；如修改 robot streamer/adapter，必须执行 refactor 前后 manifest 对比。
- `smpl_test`：SMPL adapter/manifest smoke；如修改 SMPL streamer，必须执行 streamer manifest 对比。
- `smpl`：全量 SMPL adapter/manifest smoke；如修改 SMPL adapter，必须执行 refactor 前后 manifest 对比。
- `smpl_filtered` 抽样：4/4 全部 adapter/manifest smoke；如修改 filtered loader/adapter，必须扩大抽样范围并记录原因。

Phase 3 进入代码修改前必须先完成：

- 审查 streamer / metrics / adapter 当前重复点。
- 判断是否确有必要做代码重构。
- 若需要改代码，先明确具体文件和测试命令。

## 2026-06-24 Phase 3 取消与 C++ Phase C0 计划更新

用户要求：

- 第三阶段不需要执行。
- `tools/sonic_eval/*.py` 都是增量拓展文件，不修改。
- 现在完善 C++ Phase 的计划。

执行调整：

- Phase 3 标记为 skipped，不做代码修改，不跑 Phase 3 数据清单。
- 当前后续阶段切换为 C++ Phase C0。
- C++ Phase C0 是审查和验证阶段，不直接修改 C++/header。

C++ Phase C0 计划补充：

- 导出 base 分支 16 个 C++/header 文件快照。
- 生成 current vs base 原始 diff。
- 逐文件、逐代码块分类：
  - `drop_non_sim2sim`
  - `python_side_replacement`
  - `smpl_protocol_separate_review`
  - `must_keep_default_off_debug_hook`
- 输出：
  - `tasks/sim2sim_structure_refactor/cpp_diff_review.md`
  - `tasks/sim2sim_structure_refactor/cpp_phase_c0_test_report.md`
  - `tmp/sim2sim_refactor/<run_id>/cpp_base_snapshot/`
  - `tmp/sim2sim_refactor/<run_id>/cpp_diff_raw/`
  - `tmp/sim2sim_refactor/<run_id>/cpp_diff_patches/`

C0 数据覆盖：

- `eval_benchmark/robot_test/*.pkl` 全部 1 条：deterministic replay、metrics replay、必要时真实 E2E strict alignment。
- `eval_benchmark/robot/*.pkl` 全部 19 条：streamer/data manifest smoke。
- `eval_benchmark/smpl/*.pkl` 全部 27 条：SMPL adapter/manifest smoke。
- `data/smpl_filtered` 固定抽样 4 条：
  - `Idle_Left_001__A017.pkl`
  - `Jump_002__A017.pkl`
  - `Loop_Forward_Walk_001__A017.pkl`
  - `Neutral_stoop_down_001__A057.pkl`

C0 退出标准：

- 16 个同路径 C++/header diff 都有明确分类和证据。
- C0 不产生 C++ 修改。
- 默认 C++ build/help 通过。
- 若 Python 旁路可覆盖 sim2sim 必需信息，最终迁移原则是 C++ 全部回退 base。
- 若 Python 旁路不能覆盖，必须进入 C1 方案评审，不能直接修改 C++。

## 2026-06-24 C0 语义修正与真机时延根因初查

用户反馈：

- 已做过真机测试对比，确认增加 sim2sim 版代码后严重影响真机时延。
- C0 不应只是普通差异分类；必须找到原因并分析优化方案。

文档修正：

- `status.md`：C0 目标改为“真机时延回归风险定位 + C++ 去侵入审查”。
- `plan.md`：C0 增加热路径风险排序要求，包括 per-tick 文件写入、source-frame 查询、StateLogger 锁、ZMQ debug payload、DDS callback debug 输出等。
- 新增 `cpp_diff_review.md`：记录当前 C++ diff 的时延风险证据和优化方案。

初步根因排序：

1. `StateLogger` 新增 `source_frame_index` / `applied_source_frame_index`，进入 control loop，并在 CSV enabled 时每 tick 写盘；`applied_source_frame_index` 还存在同 tick 重复写入风险。
2. `g1_deploy_onnx_ref.cpp` 在每个 control tick 两次调用 `input_interface_->GetSourceFrameIndex(...)`，并每 tick 调用 `UpdateAppliedSourceFrameIndex(...)`。
3. `output_interface.hpp` / `zmq_output_handler.hpp` 扩展 raw reference target 和 source-frame debug payload；如果真机使用 ZMQ output，会增加 per-frame map/vector copy 和 msgpack 打包。
4. `LowStateHandler` 在 500Hz DDS callback 中加入 `steady_clock::now()` 和每秒 `std::endl` heartbeat，属于不应默认进入真机回调的 debug 逻辑。
5. 如果真机命令启用了 motion recording、CSV logging 或 `output-type zmq/all`，sim2sim 版新增字段会进一步放大 I/O 和 debug publish 开销。

优化方向：

- P0：最终合入 base 时，原则上让 16 个同路径 C++/header 文件回退 base，先恢复真机默认路径。
- P1：sim2sim 所需 strict alignment 尽量由 Python streamer manifest、packed pose stream、MuJoCo step-sync CSV 和 metrics replay 旁路提供。
- P2：为了保留完整原有 sim2sim 时间帧对齐逻辑，可以进入 C1；C1 必须是独立默认关闭 C++ hook，关闭时无 source-frame 查询、无 CSV 写盘、无 ZMQ payload 扩展，开启时逐帧对齐 mapping 必须与旧逻辑一致。

待确认：

- 用户真机测试命令中是否开启 `--enable-csv-logs`、`--enable-motion-recording`、`--output-type zmq/all`。
- 真机时延指标包括 control loop 周期、P95/P99、超时次数、CPU 占用和磁盘写入量。

## 2026-06-25 C++ hook 方案完善

用户要求：

- 从保留完整原有逻辑和功能的角度，保留一个默认关闭的 C++ sim2sim debug hook 是合理方案。
- sim2sim 功能模块代码不应该写在原有 base 仓库 C++ 文件中，应单独写功能 C++ 文件，再通过最小 include/call 接入 base 原有文件。
- 进一步完善计划和测试，确保不会影响原有任何逻辑功能。

已更新：

- `plan.md`：
  - 新增“完整原有时间帧对齐逻辑保留策略”。
  - 新增 base C++ 最小接入点表。
  - C1 调整为“独立默认关闭 sim2sim debug hook”，不再表述为只有 Python 旁路失败才执行。
- `test_matrix.md`：
  - C1 必跑测试增加默认关闭行为等价检查。
  - 增加源码边界检查。
  - 增加旧逻辑逐帧 mapping 等价测试。
  - 增加 base/current/C1 disabled/C1 enabled 四类性能对比要求。
- `status.md`：
  - 明确 C1 仍未开始代码修改。
  - C0 必须先冻结 hook 需求、base 文件最小接入点和旧逻辑等价测试设计。
- `cpp_diff_review.md`：
  - 新增独立 hook 设计草案。
  - 新增默认关闭路径和 sim2sim debug 开启路径的测试门禁。

关键门禁：

- `enabled=false` 必须证明真机默认路径与 base 等价，不新增文件 I/O、ZMQ publish、schema、policy/action/input 行为变化。
- `enabled=true` 必须证明保留旧 sim2sim 时间帧对齐语义：
  - `control_tick -> source_frame_index`
  - `control_tick -> applied_source_frame_index`
  - `metrics_row -> source_frame_index`
  - `source_frame_index -> reference pose`
- 如果逐帧 mapping 不一致，不能进入下一阶段，必须先定位差异来源并修复。

## 2026-06-25 C++ 整块迁出策略修正

用户进一步确认：

- C++ 修改最保险的方式，是将 base 仓库中被修改的 C++ 文件里的 sim2sim 修改代码块整体迁出。
- 不应把这些逻辑过度拆分到多个新 C++ 文件中；这样更不容易破坏原有功能。

已更新：

- `plan.md`：
  - C++ 目标结构从多个 `config/hook/tracker/sink` 文件收敛为默认两个文件：
    - `include/sim2sim_debug/sim2sim_debug.hpp`
    - `src/sim2sim_debug/sim2sim_debug.cpp`
  - 明确 C1 默认只新增一个头文件和一个实现文件；只有编译依赖或文件规模确实失控时，才允许增加内部 helper 文件，并必须说明原因。
  - 将 `StateLogger`、`g1_deploy_onnx_ref.cpp`、`zmq_manager` 等处理策略改为“sim2sim 相关代码块整体迁入 `sim2sim_debug.cpp`”。
- `test_matrix.md`：
  - 源码边界检查改为默认只允许 `sim2sim_debug.hpp/.cpp` 承载实现主体。
  - 增加 `find include/sim2sim_debug src/sim2sim_debug -type f` 检查，防止实现阶段过度拆分。
- `status.md` 和 `cpp_diff_review.md`：
  - 同步记录“少量文件整块迁出”作为 C1 默认方案。

执行含义：

- C1 不是重新设计一个新的 C++ debug 子系统。
- C1 是把 current C++ diff 中已经验证过的 sim2sim 逻辑代码块，从 base 原有文件中迁出到少量独立文件。
- base 原有文件只保留默认关闭 hook 的最小调用点。
- 逐帧 mapping 等价测试仍是 C1 的核心通过标准。

## 2026-06-25 C++ Phase C0 执行完成

Run ID：`20260625_cpp_c0_review`

本阶段性质：

- 只做 C++ 差异审查、产物生成和验证。
- 未修改任何 C++/header 源码文件。
- 未进入 C1 实现。

C++ diff 产物：

- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_base_snapshot/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_current_snapshot/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_raw/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_patches/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_manifest.json`

分类结果：

- `drop_non_sim2sim`：7 个文件。
- `python_side_replacement`：2 个文件。
- `smpl_protocol_separate_review`：2 个文件。
- `must_keep_default_off_debug_hook`：5 个文件。

C++ build/help：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help
git diff --name-only | rg '\.(cpp|hpp|cc|hh|h)$' || true
```

结果：

- build 通过。
- help 通过。
- C++/header diff gate 无输出。

deterministic replay：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/c0_full_deterministic_validation.py
```

执行说明：

- 第一次运行失败：新 C0 run 目录缺少 `manual_e2e_sync_deploy_logs/source_frame_index.csv`。
- 第二次运行失败：新 C0 run 目录缺少 `manual_e2e_sync_results/reach-2-003_chr00_metrics.json`。
- 修复方式：复制 Phase 1 固定端到端日志和 metrics 结果到 C0 run 目录。
- 第三次运行通过。
- 失败原因是 run 目录固定产物缺失，不是 sim2sim 对齐逻辑失败。

deterministic replay 结果：

- `streamer sent_frames=1130`
- `source_frames=930`
- `blend_frames=200`
- `packed_zmq chunks=38`
- `old_new_unpack_equal=true`
- `deploy source_valid_rows=1670`
- `deploy applied_valid_rows=1670`
- `reference_pose_exact_hits=995`
- `reference_pose_misses=0`
- `step_sync_rows_raw=995`
- `step_sync_rows_filtered=995`
- `source_frame_valid_rows=995`
- `lag_frames_log_vs_gt=0`

metrics replay：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py \
  --gt-format motionlib \
  --motion-file eval_benchmark/robot_test/reach-2-003_chr00.pkl \
  --motion-name reach-2-003_chr00 \
  --logs-dir /home/lab/Desktop/GR00T-WholeBodyControl/tmp/sim2sim_refactor/20260625_cpp_c0_review/manual_e2e_sync_combined_logs \
  --out-json /home/lab/Desktop/GR00T-WholeBodyControl/tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/metrics_replay_from_fixed_logs.json \
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
- `alignment.step_sync_rows=995`
- `alignment.source_frame_valid_rows=995`
- `metrics_all.mpjpe_g=179.1855428355443`
- `metrics_all.mpjpe_l=87.76796772732865`
- `metrics_all.mpjpe_pa=43.50997555797189`

数据覆盖：

- `eval_benchmark/robot_test/*.pkl`：1/1，通过 deterministic replay / metrics replay。
- `eval_benchmark/robot/*.pkl`：19/19，通过 streamer/data manifest smoke。
- `eval_benchmark/smpl/*.pkl`：27/27，通过 SMPL adapter/manifest smoke。
- `data/smpl_filtered` 固定抽样：4/4，通过 adapter/manifest smoke。

阶段报告：

- `tasks/sim2sim_structure_refactor/cpp_phase_c0_test_report.md`

结论：

- C0 计划内测试全部成功完成。
- C0 没有修改 C++/header。
- 允许进入 C1，但 C1 必须先冻结文件清单、最小接入点、默认关闭策略、旧逻辑逐帧等价测试和性能测试命令。

## 2026-06-25 C++ Phase C1 执行完成

Run ID：`20260625_cpp_c1_debug_hook`

本阶段改动：

- 新增默认关闭的 C++ sim2sim debug hook：
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/sim2sim_debug/sim2sim_debug.hpp`
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/sim2sim_debug/sim2sim_debug.cpp`
- `g1_deploy_onnx_ref.cpp` 只保留最小接入点：解析 `--enable-sim2sim-debug`、创建 hook、在 enabled 时才执行 source-frame 查询和 applied-source update。
- `StateLogger` 的 source-frame CSV sink 和写入默认关闭。
- `ZMQOutputHandler` 的 source-frame 字段默认不进入 `g1_debug` schema。
- `deploy.sh` 增加 `--enable-sim2sim-debug` 转发。

构建与入口测试：

```bash
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help | rg 'enable-sim2sim-debug|enable-csv-logs|output-type'
bash -n gear_sonic_deploy/deploy.sh
gear_sonic_deploy/deploy.sh --help | rg 'enable-sim2sim-debug|enable-csv-logs|output-type'
```

结果：全部通过。

静态隔离检查：

- `GetSourceFrameIndex(...)` 只在 `sim2sim_debug_hook_->Enabled()` 后调用。
- `UpdateAppliedSourceFrameIndex(...)` 只在 enabled block 内调用。
- `StateLogger` source-frame CSV sink 使用 `enable_csv && enable_sim2sim_debug`。
- ZMQ source-frame fields 只在 `enable_sim2sim_debug_` 时 pack。
- `deploy.sh` 转发 `--enable-sim2sim-debug`。

结果：通过。

确定性 replay：

```bash
/home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/deterministic_full/c1_full_deterministic_validation.py
```

执行说明：

- 第一次误用系统 Python，失败：`ModuleNotFoundError: No module named 'pandas'`。
- 使用 `sonic_eval` 环境重跑，通过。

结果：

- `old_new_unpack_equal=true`
- `sent_frames=1130`
- `step_sync_rows_filtered=995`
- `source_frame_valid_rows=995`
- `reference_pose_exact_hits=995`
- `reference_pose_misses=0`
- `lag_frames_log_vs_gt=0`
- 数据 manifest smoke：`robot=19/19`，`smpl=27/27`，`smpl_filtered=4/4`。

fixed-log metrics replay：

- `num_frames=995`
- `alignment.lag_frames_log_vs_gt=0`
- `alignment.step_sync_rows=995`
- `alignment.source_frame_valid_rows=995`

enabled E2E strict alignment：

- domain：`122`
- pose port：`5926`
- debug port：`5918`
- motion：`eval_benchmark/robot_test/reach-2-003_chr00.pkl`
- deploy：显式传入 `--enable-sim2sim-debug`

结果：

- `num_frames=769`
- `alignment.lag_frames_log_vs_gt=0`
- `alignment.source_frame_valid_rows=769`
- `alignment.step_sync_rows=769`
- `metrics_all.mpjpe_g=95.7565672488709`
- `metrics_all.mpjpe_l=37.10884109482132`
- `metrics_all.mpjpe_pa=30.266596272019257`

默认关闭 runtime smoke：

- 前两次配置不计为通过：
  - `default_off_smoke`：deploy 在完成 control loop 初始化前被 timeout 中断。
  - `default_off_smoke_rerun`：deploy 完成初始化但未发送 stream，没有产生 policy loop timing。
- 最终有效测试：`default_off_stream_smoke`
  - domain：`125`
  - pose port：`5986`
  - debug port：`5978`
  - deploy：不传 `--enable-sim2sim-debug`
  - streamer：`eval_benchmark/robot_test/reach-2-003_chr00.pkl`

结果：

- deploy logs 中没有 `source_frame_index.csv` 或 `applied_source_frame_index.csv`。
- stdout 未出现 `sim2sim debug hook enabled`。
- 控制循环进入 policy path，代表性样本：
  - `Obs: 335us, Policy: 81us, Obs 2 Motor Command: 417us, Post processing: 221us`
  - `Obs: 496us, Policy: 95us, Obs 2 Motor Command: 592us, Post processing: 53us`
  - `Obs: 418us, Policy: 143us, Obs 2 Motor Command: 561us, Post processing: 30us`
  - `Obs: 412us, Policy: 92us, Obs 2 Motor Command: 504us, Post processing: 36us`
  - `Obs: 388us, Policy: 94us, Obs 2 Motor Command: 482us, Post processing: 88us`

进程清理：

- domain `122`、`125`，ports `5918/5926/5978/5986` 对应 sim/deploy/streamer 进程已停止。

阶段报告：

- `tasks/sim2sim_structure_refactor/cpp_phase_c1_test_report.md`

阶段内结论：

- C1 计划内测试全部成功完成。
- 默认关闭路径不再产生 source-frame CSV，也不执行 source-frame 查询/更新。
- 显式 enabled 路径 strict alignment 为 `0 lag`，保持原 sim2sim 时间帧对齐逻辑。

## 2026-06-25 C++ Phase C1 Final Validation 补测

Run ID：`20260625_cpp_c1_final_validation`

用户要求补充 7 个最高风险测试，尤其是当前 enabled E2E 不能只跑 `reach-2-003_chr00`，需要多选 `eval_benchmark/robot/*.pkl`。

测试结果汇总：

- 默认关闭 ZMQ schema 精确检查：通过。`g1_debug` schema 共 `34` 个 key，不包含 `source_frame_index` / `applied_source_frame_index`，deploy logs 不生成对应 CSV。
- deterministic source-frame replay：通过。固定日志 replay `num_frames=995`，`lag_frames_log_vs_gt=0`，source/applied cursor 均单调且属于固定 stream manifest。
- StateLogger unit deterministic test：通过。默认关闭不生成 source-frame CSV，enabled 生成两份 CSV。
- ZMQOutputHandler schema deterministic test：通过。default-off `34` keys，enabled `36` keys，enabled 时 source-frame fields 存在。
- C++ diff allowlist gate：失败。相对 base 仓库 `/home/lab/Desktop/LHM-Robot` 的 `feat/s0_training`，除 C1 hook allowlist 外仍存在 13 个非 hook C++/header/test 文件差异。
- 默认关闭 timing 多样本：通过。解析到 `40` 条控制循环 timing，3 个 5-sample 窗口平均 obs-to-motor 分别为 `536.8us`、`593.2us`、`484.6us`。
- enabled E2E 多 motion：通过。`eval_benchmark/robot` 下 4 条 motion 全部 strict alignment `0 lag`：
  - `reach-1-001_chr00`: `743` frames
  - `reach-2-003_chr00`: `1007` frames
  - `reach-3-002_chr00`: `855` frames
  - `reach-4-004_chr00`: `823` frames

阻塞文件清单：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/gamepad.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/input_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/interface_manager.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/keyboard_handler.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/ros2_input_handler.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/streamed_motion_merger.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/teleop_latency_logger.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/motion_data_reader.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/output_interface.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tests/test_ros2.cpp`

结论：

- runtime/default-off/enabled sim2sim hook 行为测试通过。
- C++ 结构集成门禁未通过。当前分支仍有非 hook C++ diff，不能按流程标记 C1 完成，不能 push 作为阶段完成版本。
- 下一步必须移除或迁移这些非 hook C++ diff，然后重跑全部 7 项 final validation。

## 2026-06-25 C++ Phase C1b 迁出与最终复测

Run ID：`20260625_cpp_c1_final_validation`

代码处理：

- 将 allowlist 外 13 个非 hook C++/header/test 文件恢复到 base 等价，避免把非 sim2sim 的 keyboard/gamepad/ros2/motion reader/robot parameters/test 改动带入团队分支。
- 保留默认关闭 C++ sim2sim debug hook。
- 新增/保留独立 sim2sim helper：
  - `include/sim2sim_debug/source_frame_tracker.hpp`
  - `src/sim2sim_debug/source_frame_tracker.cpp`
  - `include/sim2sim_debug/reference_output_fields.hpp`
- 最小接入点：
  - `g1_deploy_onnx_ref.cpp`：enabled 时启用 source-frame tracker 和 hook。
  - `zmq_endpoint_interface.hpp` / `streamed_motion_merger.hpp`：streamed motion 数据接收与 `body_pos` 保留。
  - `zmq_manager.hpp`：streamed motion start control 的最小接入。
  - `output_interface.hpp`：enabled 时填充 raw reference fields。
  - `state_logger` / `zmq_output_handler`：默认关闭 source-frame CSV/schema，enabled 时输出。

关键修正：

- 第一次 enabled 多 motion 复测发现 `body_pos` 丢失会导致 raw reference global position 退化，MPJPE-G 出现米级异常。
- 将 `body_pos` 解码、校验、传入 `StreamedMotionMerger` 并写入 `MotionSequence` 后，enabled E2E 恢复到正常毫米级同量级结果。
- 测试脚本加严：
  - enabled E2E 每条 motion 必须 `num_frames >= 500`。
  - 每条 motion 必须 `lag_frames_log_vs_gt == 0`。
  - 每条 motion 必须满足 MPJPE 阈值。
  - schema 捕获样例必须包含 source-frame 字段。

最终测试命令与结果：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help | rg 'enable-sim2sim-debug|enable-csv-logs|output-type'
bash -n gear_sonic_deploy/deploy.sh && gear_sonic_deploy/deploy.sh --help | rg 'enable-sim2sim-debug|enable-csv-logs|output-type'
/home/lab/miniconda3/envs/sonic_eval/bin/python tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/scripts/deterministic_cpp_boundary_checks.py
g++ -std=c++20 -I gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include \
  tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/scripts/state_logger_sim2sim_gate_test.cpp \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/state_logger.cpp \
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/file_sink.cpp \
  -o tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/scripts/state_logger_sim2sim_gate_test
tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/scripts/state_logger_sim2sim_gate_test \
  tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/results/state_logger_unit
/home/lab/miniconda3/envs/sonic_eval/bin/python tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/deterministic_full/c1_full_deterministic_validation.py
bash tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/scripts/run_enabled_multi_motion_e2e.sh
/home/lab/miniconda3/envs/sonic_eval/bin/python tmp/sim2sim_refactor/20260625_cpp_c1_final_validation/scripts/summarize_final_validation.py
```

通过项：

- build/help/deploy help：通过。
- default-off ZMQ schema/runtime：通过，`31` keys，无 `source_frame_index` / `applied_source_frame_index`，deploy logs 无 source-frame CSV。
- deterministic fixed replay：通过，`num_frames=995`，`lag_frames_log_vs_gt=0`。
- StateLogger unit：通过。
- ZMQOutputHandler schema：通过，default-off `31` keys，enabled `36` keys。
- C++ diff allowlist gate：通过，unexpected list 为空。
- default-off timing 多样本：通过，解析到 `43` 条 timing。
- enabled E2E 多 motion：通过：
  - `reach-1-001_chr00`: `735` frames, `0 lag`, MPJPE-G `112.477mm`
  - `reach-2-003_chr00`: `1007` frames, `0 lag`, MPJPE-G `175.410mm`
  - `reach-3-002_chr00`: `859` frames, `0 lag`, MPJPE-G `121.746mm`
  - `reach-4-004_chr00`: `827` frames, `0 lag`, MPJPE-G `83.722mm`

最终结论：

- C1b 结构优化和计划内全部测试成功完成。
- 可以按流程 commit 并 push 到 `origin`。
