# C++ Phase C1 Test Report

Run ID: `20260625_cpp_c1_debug_hook`

## Scope

- Add a default-off C++ sim2sim debug hook boundary.
- Keep true-robot/default deploy behavior free of sim2sim source-frame lookup, source-frame CSV, and default `g1_debug` source-frame fields.
- Preserve original sim2sim source-frame behavior when `--enable-sim2sim-debug` is explicitly enabled.

## Code Boundary

- New files:
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/sim2sim_debug/sim2sim_debug.hpp`
  - `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/sim2sim_debug/sim2sim_debug.cpp`
- Minimal touched base files:
  - `g1_deploy_onnx_ref.cpp`: parse `--enable-sim2sim-debug`, create hook, guard source-frame lookup/update.
  - `state_logger.hpp/.cpp`: gate source-frame CSV creation/writes behind `enable_sim2sim_debug`.
  - `zmq_output_handler.hpp`: gate source-frame fields behind `enable_sim2sim_debug`.
  - `CMakeLists.txt`: include new source files via existing src glob.
  - `deploy.sh`: forward `--enable-sim2sim-debug`.

## Tests

### Build and Entrypoints

```bash
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref -j2
cd gear_sonic_deploy && ./target/release/g1_deploy_onnx_ref --help | rg 'enable-sim2sim-debug|enable-csv-logs|output-type'
bash -n gear_sonic_deploy/deploy.sh
gear_sonic_deploy/deploy.sh --help | rg 'enable-sim2sim-debug|enable-csv-logs|output-type'
```

Result: passed.

### Static Default-Off Gate

Checked:

- CLI flag exists.
- `GetSourceFrameIndex(...)` is guarded by `sim2sim_debug_hook_->Enabled()`.
- `UpdateAppliedSourceFrameIndex(...)` is only called in the enabled block.
- ZMQ `source_frame_index` / `applied_source_frame_index` fields are only packed when enabled.
- StateLogger source-frame CSV sinks are constructed with `enable_csv && enable_sim2sim_debug`.
- default-off `UpdateAppliedSourceFrameIndex(...)` is no-op.
- `deploy.sh` forwards `--enable-sim2sim-debug`.

Result: passed.

### Deterministic Replay

```bash
/home/lab/miniconda3/envs/sonic_eval/bin/python \
  tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/deterministic_full/c1_full_deterministic_validation.py
```

First attempt used system Python and failed with `ModuleNotFoundError: No module named 'pandas'`; rerun in `sonic_eval` passed.

Key results:

- `old_new_unpack_equal=true`
- `sent_frames=1130`
- `step_sync_rows_filtered=995`
- `source_frame_valid_rows=995`
- `reference_pose_exact_hits=995`
- `reference_pose_misses=0`
- `lag_frames_log_vs_gt=0`
- dataset manifest smoke: `robot=19/19`, `smpl=27/27`, `smpl_filtered=4/4`

### Fixed-Log Metrics Replay

```bash
env PYTHONPATH=/home/lab/Desktop/IsaacLab/source \
  /home/lab/miniconda3/envs/sonic_eval/bin/python \
  tools/sonic_eval/compute_mujoco_tracking_metrics.py \
  --gt-format motionlib \
  --motion-file eval_benchmark/robot_test/reach-2-003_chr00.pkl \
  --motion-name reach-2-003_chr00 \
  --logs-dir tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/manual_e2e_sync_combined_logs \
  --out-json tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/deterministic_full/metrics_replay_from_fixed_logs.json \
  --no-motionlib-robot \
  --ignore-motion-playing-mask \
  --streamed-only \
  --align-mode source_frame_index \
  --actual-source step_sync_body_pos_w_14 \
  --sim-valid-only \
  --stream-blend-from-stand-frames 200
```

Result: passed.

- `num_frames=995`
- `alignment.lag_frames_log_vs_gt=0`
- `alignment.step_sync_rows=995`
- `alignment.source_frame_valid_rows=995`

### Enabled E2E Strict Alignment

Run config:

- domain: `122`
- pose port: `5926`
- debug port: `5918`
- motion: `eval_benchmark/robot_test/reach-2-003_chr00.pkl`
- deploy flag: `--enable-sim2sim-debug`

Result: passed.

- `num_frames=769`
- `alignment.lag_frames_log_vs_gt=0`
- `alignment.source_frame_valid_rows=769`
- `alignment.step_sync_rows=769`
- `metrics_all.mpjpe_g=95.7565672488709`
- `metrics_all.mpjpe_l=37.10884109482132`
- `metrics_all.mpjpe_pa=30.266596272019257`

The effective frame count differs from the fixed-log replay because this is a live policy/runtime E2E run with reset timing variability. Strict source-frame alignment is still `0 lag`, and metrics are same-order or better than Phase 2 E2E.

### Default-Off Runtime Smoke

Two setup attempts were not counted as passing:

- `default_off_smoke`: deploy was interrupted before policy/control loop initialization completed.
- `default_off_smoke_rerun`: deploy initialized but no stream was sent, so no policy loop timing was produced.

Final valid run:

- directory: `tmp/sim2sim_refactor/20260625_cpp_c1_debug_hook/default_off_stream_smoke`
- domain: `125`
- pose port: `5986`
- debug port: `5978`
- deploy flag: no `--enable-sim2sim-debug`
- motion stream: `eval_benchmark/robot_test/reach-2-003_chr00.pkl`

Result: passed.

- No `source_frame_index.csv` or `applied_source_frame_index.csv` was created in deploy logs.
- `sim2sim debug hook enabled` was not printed.
- Representative loop timing samples:
  - `Obs: 335us, Policy: 81us, Obs 2 Motor Command: 417us, Post processing: 221us`
  - `Obs: 496us, Policy: 95us, Obs 2 Motor Command: 592us, Post processing: 53us`
  - `Obs: 418us, Policy: 143us, Obs 2 Motor Command: 561us, Post processing: 30us`
  - `Obs: 412us, Policy: 92us, Obs 2 Motor Command: 504us, Post processing: 36us`
  - `Obs: 388us, Policy: 94us, Obs 2 Motor Command: 482us, Post processing: 88us`

## Conclusion

C1 planned tests passed after correcting test configuration. Default-off deploy path no longer performs sim2sim source-frame CSV output and source-frame lookup is guarded. Enabled sim2sim debug path preserves strict source-frame alignment.
