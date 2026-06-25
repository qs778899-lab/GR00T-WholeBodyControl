# C++ Phase C0 test report

更新时间：2026-06-25

## 阶段结论

C++ Phase C0 已完成计划内审查和测试。本阶段没有修改任何 C++/header 源码文件。

结论：

- 16 个相对 base 不同的同路径 C++/header 文件已生成 base/current 快照、raw diff 和分类 patch。
- C++ build/help 通过。
- C++/header 工作区 diff 为空。
- 固定日志 deterministic replay 通过。
- metrics replay 通过，`lag_frames_log_vs_gt=0`。
- C0 数据清单覆盖完成：
  - `eval_benchmark/robot_test/*.pkl`：1/1。
  - `eval_benchmark/robot/*.pkl`：19/19。
  - `eval_benchmark/smpl/*.pkl`：27/27。
  - `data/smpl_filtered` 固定抽样：4/4。

C0 不进入 C++ 代码修改。下一阶段 C1 如开始，必须按 `plan.md` 当前方案执行：少量文件整块迁出，默认只新增 `include/sim2sim_debug/sim2sim_debug.hpp` 和 `src/sim2sim_debug/sim2sim_debug.cpp`，base 原有 C++ 文件只保留最小 hook 接入点。

## 环境

- 项目根目录：`/home/lab/Desktop/GR00T-WholeBodyControl`
- 当前分支：`main`
- 当前 commit：`9a7839ae17bc61c64fb1f9e8c002311d70536335`
- base 仓库：`/home/lab/Desktop/LHM-Robot`
- base 分支：`feat/s0_training`
- base commit：`0a751ee86aab866a6bf483b41e1b8a48a0787d44`
- Python 环境：`/home/lab/miniconda3/envs/sonic_eval/bin/python`
- `PYTHONPATH=/home/lab/Desktop/IsaacLab/source`
- Run ID：`20260625_cpp_c0_review`
- Run 目录：`tmp/sim2sim_refactor/20260625_cpp_c0_review`

## C++ diff 产物

生成命令：使用本地 Python 一次性导出 base/current 16 个文件快照，并生成 raw/classified patches。

产物：

- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_base_snapshot/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_current_snapshot/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_raw/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_patches/`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/cpp_diff_manifest.json`

分类结果：

| 分类 | 文件数 |
|---|---:|
| `drop_non_sim2sim` | 7 |
| `python_side_replacement` | 2 |
| `smpl_protocol_separate_review` | 2 |
| `must_keep_default_off_debug_hook` | 5 |

## C++ build/help

命令：

```bash
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help
git diff --name-only | rg '\.(cpp|hpp|cc|hh|h)$' || true
```

结果：

- `g1_deploy_onnx_ref` build 通过。
- deploy `--help` 通过。
- C++/header diff gate 无输出，确认 C0 未修改 C++/header 源码。

## Deterministic replay

命令：

```bash
mkdir -p tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full
cp tmp/sim2sim_refactor/20260624_200458_phase2_hook_validation/deterministic_full/phase2_full_deterministic_validation.py \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/c0_full_deterministic_validation.py

cp -a tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_deploy_logs \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/
cp -a tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_sim_logs \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/
cp -a tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_combined_logs \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/
cp -a tmp/sim2sim_refactor/20260624_142140_phase1_validation/manual_e2e_sync_results \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/

env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/c0_full_deterministic_validation.py
```

执行记录：

- 第一次运行失败：新 C0 run 目录缺少 `manual_e2e_sync_deploy_logs/source_frame_index.csv`。
- 第二次运行失败：新 C0 run 目录缺少 `manual_e2e_sync_results/reach-2-003_chr00_metrics.json`。
- 修复方式：复制 Phase 1 固定端到端日志和 metrics 结果到 C0 run 目录。
- 第三次运行通过。

失败性质：run 目录固定产物缺失，不是对齐逻辑失败。C0 是审查阶段，不重新运行 policy；本阶段使用 Phase 1 固定日志绕开 policy 随机性做 deterministic replay。

通过结果：

| 项 | 结果 |
|---|---:|
| streamer sent frames | 1130 |
| source frames | 930 |
| blend frames | 200 |
| packed ZMQ chunks | 38 |
| old/current unpack equal | true |
| deploy source valid rows | 1670 |
| deploy applied valid rows | 1670 |
| reference pose exact hits | 995 |
| reference pose misses | 0 |
| step sync raw rows | 995 |
| step sync filtered rows | 995 |
| source frame valid rows | 995 |
| metrics replay input frames | 995 |
| lag frames log vs gt | 0 |

数据清单：

| 数据范围 | 计划数量 | 完成数量 | 结果 |
|---|---:|---:|---|
| `eval_benchmark/robot_test/*.pkl` | 1 | 1 | 通过 deterministic replay / metrics replay |
| `eval_benchmark/robot/*.pkl` | 19 | 19 | 通过 streamer/data manifest smoke |
| `eval_benchmark/smpl/*.pkl` | 27 | 27 | 通过 SMPL adapter/manifest smoke |
| `data/smpl_filtered` 固定抽样 | 4 | 4 | 通过 adapter/manifest smoke |

产物：

- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/phase1_full_deterministic_validation_summary.json`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/dataset_manifest_smoke.json`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/stream_manifest.csv`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/packed_message_manifest.jsonl`
- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/alignment_manifest.csv`

## Metrics replay

命令：

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

结果：

| 字段 | 值 |
|---|---:|
| `num_frames` | 995 |
| `alignment.lag_frames_log_vs_gt` | 0 |
| `alignment.step_sync_rows` | 995 |
| `alignment.source_frame_valid_rows` | 995 |
| `metrics_all.mpjpe_g` | 179.1855428355443 |
| `metrics_all.mpjpe_l` | 87.76796772732865 |
| `metrics_all.mpjpe_pa` | 43.50997555797189 |

产物：

- `tmp/sim2sim_refactor/20260625_cpp_c0_review/deterministic_full/metrics_replay_from_fixed_logs.json`

## C0 gate

通过：

- 16 个 C++/header diff 全部有分类。
- C0 没有产生 C++/header 工作区修改。
- C++ build/help 通过。
- C0 数据清单全部完成计划内验证。
- deterministic replay 和 metrics replay 均通过。

允许进入下一阶段：是。进入 C1 前必须先冻结 C1 文件清单、base 最小接入点、默认关闭策略、旧逻辑逐帧等价测试和性能测试命令。
