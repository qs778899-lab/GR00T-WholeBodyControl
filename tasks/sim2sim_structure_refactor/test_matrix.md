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

Phase 1 已补齐并通过第一层确定性链路验证。该验证使用固定 motion、固定已录制端到端日志和 pre-refactor/current old-new 对比，绕开 policy 随机性；它不能替代后续 phase 自己的随机端到端统计验证。

## 数据覆盖硬规则

每个 phase 开始前必须定义本阶段的“挑选测试数据清单”。清单至少包含：

- 数据路径。
- motion name 或数据集条目名。
- 数据类型：`robot_test` / `robot` / `smpl` / `smpl_filtered` / 其他。
- 本阶段对该数据执行的测试层级：数据 smoke、确定性链路验证、端到端 strict alignment、端到端统计验证。
- 是否必须跑完整 sim/deploy/policy A/B/C/D 链路。
- 如果不能跑完整端到端链路，必须写明降级原因和风险。

通过标准：

- 清单中的所有数据都必须完成本阶段计划内的完整测试运行。
- 所有测试必须成功完成；不能只跑部分数据，不能只保留成功样例。
- 每条数据的结果必须有可追踪产物，至少包含 summary、metrics JSON 或 manifest。
- 如果任一条数据失败、未运行、结果不确定、有效帧覆盖异常减少且原因无法解释，本 phase 不能标记为完成，不能进入下一阶段。
- 数据 smoke 只能证明数据可加载，不能替代确定性链路验证或端到端运行。

默认挑选数据范围：

- `eval_benchmark/robot_test/*.pkl`：优先用于完整 A/B/C/D 端到端 strict alignment 和 deterministic replay。
- `eval_benchmark/robot/*.pkl`：至少全量执行 streamer/data manifest smoke；若该 phase 改动 streamer、metrics、reference pose 或时间对齐逻辑，必须扩大到完整确定性链路验证或端到端运行。
- `eval_benchmark/smpl/*.pkl`：至少全量执行 adapter/manifest smoke；若该 phase 改动 SMPL adapter、streamer 或 metrics 输入逻辑，必须扩大到完整确定性链路验证或端到端运行。
- `data/smpl_filtered/`：必须在阶段开始前明确抽样清单；抽样数据全部执行计划内测试。若该 phase 改动 SMPL filtered 相关 loader/adapter，抽样范围必须扩大，并说明覆盖依据。

阶段记录要求：

- `log.md` 必须记录实际运行的数据清单、成功/失败数量、失败原因或差异解释。
- `status.md` 必须写明本阶段数据覆盖是否满足门禁。
- 所有验证数据、manifest、metrics 和 summary 仍统一放在项目根目录 `tmp/` 下。

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
- 真实 motion 的 streamer manifest、packed chunk 序列、reference pose buffer 命中、metrics replay 对比由 Phase 1 完整确定性链路补测覆盖。

Phase 1 完整确定性链路补测：

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/phase1_full_deterministic_validation.py
```

产物：

- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/phase1_full_deterministic_validation_summary.json`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/stream_manifest.csv`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/packed_message_manifest.jsonl`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/alignment_manifest.csv`
- `tmp/sim2sim_refactor/20260624_142140_phase1_validation/deterministic_full/dataset_manifest_smoke.json`

通过结果：

- streamer manifest：`sent_frames=1130`，`source_frames=930`，`blend_frames=200`，`motion_start_frame=200`。
- packed ZMQ round-trip：`chunks=38`，old/current unpack 输出完全一致。
- deploy cursor：source/applied frame 均为 streamer manifest 子集，valid rows `1670`，frame range `30..1083`，单调。
- reference pose buffer：`buffer_frames=1130`，step-sync exact pose hits `995`，misses `0`，old/current 抽样 buffer 一致。
- step-sync alignment：`raw_rows=995`，`filtered_rows=995`，source range `41..1083`，unique source frames `995`，无相邻重复。
- metrics replay：固定日志重算后 `num_frames=995`、`lag_frames_log_vs_gt=0`、`step_sync_rows=995`，`mpjpe_g/l/pa` 与原 JSON 一致。
- 数据集 manifest smoke：robot `19/19`，smpl `27/27`，smpl_filtered sampled `4/4`。

每个 phase 都必须覆盖的第一层确定性链路验证：

| 环节 | 必需输入 | 必需输出 | 通过标准 |
|---|---|---|---|
| streamer manifest | 固定 `motion_file/motion_name/target_fps/start/end/prepend/blend/chunk` | `stream_manifest.csv/jsonl` | frame_index、prefix/blend/motion 边界、motion_start_frame、source frame 映射一致 |
| packed ZMQ round-trip | streamer 输出 chunk | `packed_message_manifest.jsonl` | header 字段、shape、dtype、chunk 首尾 frame、frame_index 序列一致 |
| deploy 播放游标 | 已录制 `source_frame_index.csv` / `applied_source_frame_index.csv` | source/applied frame 对比 summary | 有效区间、首尾、缺口、延迟/映射规则一致 |
| reference pose buffer | raw pose messages + debug source frames | exact pose hit/miss report | 每个 metrics GT frame 必须命中 exact raw pose；fallback pose 不能参与 GT |
| step-sync logger | actual/ref body pos、source_frame_index、is_new_control_frame | `step_sync_invariant_report.json` | 写入条件、去重规则、actual/ref 来源、行数和有效 source frame 一致 |
| metrics replay | 固定 CSV 日志 | metrics JSON + alignment manifest | `(metric_row, source_frame_index, actual_row, ref_row)`、输入 shape、有效 frame 集合一致 |
| visualization/metrics 同源 | 同一 source frame 的 viewer/ref/offline 输出 | per-frame ref body pos 对比 | 同一 translation/alignment mode 下 ref body positions 一致或差异有明确解释 |

后续 phase 通过前至少要有：

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

## C++ Phase 专项测试矩阵

C++ Phase 的优先目标是证明默认真机/部署 C++ 主路径不受 sim2sim 影响。C++ Phase 不与 Python 工具重构混合提交。

### C++ Phase C0：审查与验证，不改 C++

必跑命令：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help
git diff --name-only | rg '\.(cpp|hpp|cc|hh|h)$' || true
```

必需产物：

- `tmp/sim2sim_refactor/<run_id>/cpp_base_snapshot/`
- `tmp/sim2sim_refactor/<run_id>/cpp_diff_raw/`
- `tmp/sim2sim_refactor/<run_id>/cpp_diff_patches/`
- `tasks/sim2sim_structure_refactor/cpp_diff_review.md`
- `tasks/sim2sim_structure_refactor/cpp_phase_c0_test_report.md`

数据覆盖：

- `eval_benchmark/robot_test/*.pkl` 全部 1 条：deterministic replay、metrics replay、必要时真实 E2E strict alignment。
- `eval_benchmark/robot/*.pkl` 全部 19 条：streamer/data manifest smoke。
- `eval_benchmark/smpl/*.pkl` 全部 27 条：SMPL adapter/manifest smoke。
- `data/smpl_filtered` 固定抽样 4 条：adapter/manifest smoke。

通过标准：

- C0 不产生 C++/header 工作区修改。
- 16 个同路径 C++/header diff 全部完成逐文件、逐代码块分类。
- 每个 proposed drop 都有 Python 旁路、manifest 或 fixed-log replay 证据。
- 如果任何 diff 无法分类或无法证明可丢弃，C0 不能标记为完成；必须报告问题并决定是否进入 C1。

### C++ Phase C1：最小默认关闭 hook，仅在 C0 失败时执行

进入条件：

- C0 证明 Python 旁路不足。
- 用户确认允许修改 C++。
- C1 文档先列出具体文件、默认关闭策略、性能测试命令和回滚方案。

必跑测试：

- C++ 默认 debug 关闭路径 build/help。
- C++ 默认 debug 关闭路径部署 smoke。
- 控制循环耗时统计，与 base/default 路径对比。
- sim2sim debug 开启路径至少 1 条 `robot_test` 真实 E2E strict alignment。
- C0 数据清单全部重新按受影响层级执行。

通过标准：

- `enabled=false` 默认路径无新增文件 I/O、无 ZMQ publish、无动态分配、无 schema 变化。
- `enabled=false` 控制循环耗时与 base/default 同量级，差异有明确解释。
- `enabled=true` 才产生 sim2sim debug 输出。
- C1 修改单独 commit，不能混入 Python 工具重构。

### C++ Phase C2：迁移到 base 前验证

必需结论：

- 最终合入 base 的 C++ 文件列表。
- 最终不合入 base 的 C++ diff 列表和原因。
- 如果 C++ 全部回退 base，明确写出“不合入任何 C++ diff”的结论。
- 如果存在 C++ diff 需要合入，必须有默认关闭、性能报告、真机部署风险说明和单独 patch/PR 方案。

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
