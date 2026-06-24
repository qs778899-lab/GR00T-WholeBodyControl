# sim2sim 结构优化测试矩阵

更新时间：2026-06-24

## 通用门禁测试

每个 phase 完成后必须执行并记录：

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
```

如果涉及 deploy 或可能影响 C++/真机路径，还必须执行：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help
git diff --name-only | rg '\.(cpp|hpp|cc|hh|h)$' || true
```

通过标准：

- 所有命令退出码为 0。
- C++/header diff 必须为空，除非该 phase 明确是 C++ 默认关闭 hook 阶段并已单独评审。
- 入口 help 中 sim2sim 参数仍可见且没有 import/runtime error。

## 测试层级说明

测试分为三层，不能互相替代：

- 数据 smoke：只验证数据文件存在、可加载、字段/shape/finite 合法；不比较实际 sim2sim 结果。
- 确定性链路验证：绕开 policy 随机性，固定输入/日志/manifest，对 streamer、ZMQ、reference pose、step-sync CSV、metrics alignment 做逐环节输出对比。
- 端到端统计验证：承认 policy 随机性，跑 baseline/refactor 多次，比较 metrics 分布、有效帧覆盖和 per-link 异常。

Phase 1 已完成的是“代码搬移一致性 + 数据 smoke + 单条端到端 strict alignment smoke”，不是完整的第一层确定性链路验证。

## 确定性验证

Phase 1 已使用的最小确定性检查：

```bash
python tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic/phase1_moved_helpers_check.py
```

通过标准：

- 搬移类/常量源码一致。
- packed ZMQ unpack、logger CSV、overlay error 计算通过。
- `Sim2SimEvalLogger` 的 step-sync 写入规则不变。

覆盖边界：

- 该检查只适用于 Phase 1 的“源码搬移不改变局部逻辑”判断。
- 该检查不覆盖真实 motion 的 streamer manifest 对比、packed chunk 序列对比、reference pose buffer 命中对比、metrics replay 对比。

Phase 2 代码重构前必须补齐的第一层确定性链路验证：

| 环节 | 必需输入 | 必需输出 | 通过标准 |
|---|---|---|---|
| streamer manifest | 固定 `motion_file/motion_name/target_fps/start/end/prepend/blend/chunk` | `stream_manifest.csv/jsonl` | frame_index、prefix/blend/motion 边界、motion_start_frame、source frame 映射一致 |
| packed ZMQ round-trip | streamer 输出 chunk | `packed_message_manifest.jsonl` | header 字段、shape、dtype、chunk 首尾 frame、frame_index 序列一致 |
| deploy 播放游标 | 已录制 `source_frame_index.csv` / `applied_source_frame_index.csv` | source/applied frame 对比 summary | 有效区间、首尾、缺口、延迟/映射规则一致 |
| reference pose buffer | raw pose messages + debug source frames | exact pose hit/miss report | 每个 metrics GT frame 必须命中 exact raw pose；fallback pose 不能参与 GT |
| step-sync logger | actual/ref body pos、source_frame_index、is_new_control_frame | `step_sync_invariant_report.json` | 写入条件、去重规则、actual/ref 来源、行数和有效 source frame 一致 |
| metrics replay | 固定 CSV 日志 | metrics JSON + alignment manifest | `(metric_row, source_frame_index, actual_row, ref_row)`、输入 shape、有效 frame 集合一致 |
| visualization/metrics 同源 | 同一 source frame 的 viewer/ref/offline 输出 | per-frame ref body pos 对比 | 同一 translation/alignment mode 下 ref body positions 一致或差异有明确解释 |

Phase 2 通过前至少要有：

- 一条 `eval_benchmark/robot_test` motion 的完整确定性链路验证。
- `eval_benchmark/robot/*.pkl` 全量 streamer/data manifest smoke。
- `eval_benchmark/smpl/*.pkl` 全量 adapter/manifest smoke。
- `data/smpl_filtered` 抽样 adapter/manifest smoke。

## 数据 smoke

Phase 1 已使用：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic/bin/python \
  tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic/motionlib_dataset_smoke.py
```

覆盖：

- `eval_benchmark/robot/*.pkl`
- `eval_benchmark/smpl/*.pkl`
- `data/smpl_filtered` 抽样：
  - `Idle_Left_001__A017.pkl`
  - `Jump_002__A017.pkl`
  - `Loop_Forward_Walk_001__A017.pkl`
  - `Neutral_stoop_down_001__A057.pkl`

通过标准：

- 所有样本加载/字段/shape/finite 检查通过。
- 当前 Phase 1 结果为 `50/50` 通过。

限制：

- 数据 smoke 不运行 MuJoCo/deploy/policy。
- 数据 smoke 不比较 baseline/refactor metrics。
- 数据 smoke 不能替代确定性链路验证或端到端统计验证。

## 端到端 strict frame alignment smoke

最小 smoke motion：

- `eval_benchmark/robot_test/reach-2-003_chr00.pkl`
- `motion_name=reach-2-003_chr00`

必需条件：

- deploy 必须打开 debug ZMQ 输出：`--output-type zmq`。
- sim 侧 `--reference-motion-zmq-port` 必须指向 deploy `--zmq-out-port`。
- sim 侧 `--reference-motion-pose-zmq-port` 必须指向 streamer/deploy pose 端口。
- metrics 使用 `--actual-source step_sync_body_pos_w_14 --align-mode source_frame_index --sim-valid-only`。

通过标准：

- streamer 正常结束。
- `sim_source_frame_index.csv` 有有效 source frame。
- `sim2sim_step_sync_body_pos_w_14.csv` 非空。
- metrics JSON 正常写出。
- `alignment.lag_frames_log_vs_gt == 0`。
- `alignment.step_sync_rows > 0`，且覆盖范围没有明显异常减少。

Phase 1 通过结果：

- `num_frames=995`
- `source_frame_valid_rows=995`
- `step_sync_rows=995`
- `lag_frames_log_vs_gt=0`

## 测试数据清理规则

每个 phase 的所有计划内测试都成功完成、确认没有问题、阶段文档更新并完成提交/push 后，可以清理该 phase 测试产生的大体积临时数据。

清理前必须保留：

- 小体积 summary。
- metrics JSON 或摘要。
- 测试命令、环境、结果。
- 原始数据路径索引。
- 失败原因或差异解释。

不能清理的情况：

- 任何测试失败。
- 结果不确定。
- 后续还需要复查原始日志。
- 阶段尚未 commit/push。
