# SONIC MuJoCo Inference Test Utilities

These scripts are standalone helpers for your requested workflow without modifying project core logic.

## Files

- `parquet_to_mujoco_motion.py`
  - Read one official exported parquet and generate deploy-compatible motion CSV folder.
  - Handles joint-order conversion for deploy motion input.
  - Supports `--joint-source action.wbc` to use collected robot control as encoder motion input.
- `visualize_realtime_error.py`
  - Compute and print instantaneous error + end-effector accuracy.
  - Supports strict EEF GT from action/state FK with `--eef-gt-source from_gt_q43`.
  - Supports reusing converted motion folder as GT with `--gt-motion-dir`.
  - Modes:
    - `live_zmq`: read `g1_debug` directly (needs `pyzmq + msgpack_numpy`)
    - `tail_logs`: realtime-like tailing of deploy `q.csv/action.csv` (no zmq python deps)
    - `offline_logs`: post-hoc metrics from complete logs.
- `compute_eef_accuracy_offline.py`
  - Offline FK consistency check: `observation.state` vs `observation.eef_state`.
- `compute_mujoco_tracking_metrics.py`
  - Compute Isaac-like tracking metrics on MuJoCo deploy logs (`q.csv`) vs parquet GT:
    - `mpjpe_g`, `mpjpe_l`, `mpjpe_pa`
    - subsets: `legs`, `vr_3points`, `other_upper_bodies`, `foot`
- `run_mujoco_chain.sh`
  - Convenience script: convert parquet + start sim + start deploy.
- `run_mujoco_batch_eval.sh`
  - Batch helper for existing sim2sim chain (keeps A/B unchanged, loops C+D per motion).
  - Reads motion list CSV (`motion_file,motion_name`) and writes summary JSON/CSV.
- `run_mujoco_batch_eval_parallel.sh`
  - Parallel dispatcher that launches multiple `run_mujoco_batch_eval.sh` workers.
  - Requires isolated A/B runtime per worker (separate ZMQ ports and logs).

## 1) Convert parquet -> deploy motion folder

```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic

python tools/sonic_eval/parquet_to_mujoco_motion.py \
  --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
  --meta-info-json /home/lab/Desktop/data/meta/info.json \
  --output-root /tmp/sonic_motions_from_parquet \
  --motion-name episode_000000_from_parquet \
  --joint-source action.wbc
```

Generated folder:
- `/tmp/sonic_motions_from_parquet/episode_000000_from_parquet`

## 2) Run MuJoCo + deploy policy (manual)

Terminal A (sim loop):
```bash
# use your existing environment that has unitree_sdk2py + tyro + mujoco runtime
python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default
```

Terminal B (deploy):
```bash
cd gear_sonic_deploy
# NOTE: deploy build needs TensorRT_ROOT + just + successful cmake build
./deploy.sh sim \
  --motion-data /tmp/sonic_motions_from_parquet \
  --obs-config policy/release/observation_config.yaml \
  --input-type manager \
  --output-type all \
  --zmq-host localhost \
  --zmq-port 5596
```

Note:
- `--motion-data` must point to the parent directory containing one or more motion folders.
- For strict "control-to-encoder" tests, always convert with `--joint-source action.wbc`.

## 3) Realtime instantaneous error + end-effector accuracy

### Option A: tail deploy logs (works without pyzmq)

Run after deploy starts and writes logs (`--enable-csv-logs --logs-dir ...` in deploy argv if needed):

```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic

python tools/sonic_eval/visualize_realtime_error.py \
  --mode tail_logs \
  --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
  --gt-source action.wbc \
  --logs-dir /path/to/deploy/logs/xx-xx-xx/xx-xx-xx \
  --print-every 10 \
  --out-json /tmp/sonic_error_metrics_tail.json
```

### Option B: direct ZMQ stream

```bash
python tools/sonic_eval/visualize_realtime_error.py \
  --mode live_zmq \
  --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
  --gt-source action.wbc \
  --eef-gt-source from_gt_q43 \
  --gt-motion-dir /tmp/sonic_motion_action_only/episode_000000_action \
  --zmq-host 127.0.0.1 --zmq-port 5608 --zmq-topic g1_debug \
  --print-every 10 \
  --out-json /tmp/sonic_error_metrics_zmq.json
```

## 4) Offline EEF sanity metric (dataset-level)

```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic

python tools/sonic_eval/compute_eef_accuracy_offline.py \
  --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
  --out-json /tmp/sonic_eef_offline.json
```

## 5) MuJoCo sim2sim tracking metrics (Isaac-like)

After deploy has produced CSV logs (`q.csv`):

```bash
source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic

python tools/sonic_eval/compute_mujoco_tracking_metrics.py \
  --parquet /home/lab/Desktop/data/data/chunk-000/episode_000000.parquet \
  --logs-dir /path/to/deploy/logs/<date>/<time> \
  --gt-source action.wbc \
  --gt-motion-dir /tmp/sonic_motion_action_only/episode_000000_action \
  --out-json /tmp/sonic_mujoco_tracking_metrics.json
```

## 6) Batch C+D for motionlib list (no codepath changes)

Keep your existing Terminal A/B running, then execute batch C+D in another terminal:

```bash
cat >/tmp/motion_list.csv <<EOF
sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl,walk_forward_amateur_001__A001
sample_data/robot_filtered/210531/walk_forward_amateur_001__A001_M.pkl,walk_forward_amateur_001__A001_M
EOF

source /home/lab/miniconda3/etc/profile.d/conda.sh
conda activate sonic

tools/sonic_eval/run_mujoco_batch_eval.sh \
  --motion-list /tmp/motion_list.csv \
  --logs-root /tmp/sonic_logs/batch_zmq \
  --results-json /tmp/sonic_batch_metrics_summary.json \
  --results-csv /tmp/sonic_batch_metrics_summary.csv
```

### Parallel batch (N workers, isolated ports)

```bash
tools/sonic_eval/run_mujoco_batch_eval_parallel.sh \
  --motion-list /tmp/motion_list.csv \
  --workers 2 \
  --port-base 5556 \
  --port-step 10 \
  --results-root /tmp/sonic_batch_parallel
```

Notes:
- Worker 0 uses port `5556`, worker 1 uses `5566` (with `--port-step 10`).
- You must provide matching isolated A/B instances for each worker port.
- This adds parallel throughput without changing original single-worker behavior.

## Environment caveats observed on this machine

Current `sonic` env has:
- yes: `numpy/pandas/pyarrow/scipy/matplotlib/pinocchio`
- no: `pyzmq/msgpack_numpy`
- no: `tyro`
- no: `unitree_sdk2py`

Current system missing for deploy build/runtime:
- `TensorRT_ROOT` unresolved (cmake fails)
- `just` missing

So you can already run conversion/offline metrics now; full MuJoCo+deploy runtime needs your deploy/runtime env fixed first.
