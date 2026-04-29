#!/usr/bin/env bash
set -euo pipefail

# Parallel dispatcher for MuJoCo sim2sim batch eval.
# This script DOES NOT change existing logic. It only launches multiple
# run_mujoco_batch_eval.sh workers in parallel with isolated ports/log dirs.
#
# You must prepare matching running A/B pipelines per worker (port-isolated).
#
# Example worker ports:
#   worker0 -> 5556
#   worker1 -> 5566

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MOTION_LIST=""
MOTION_DIR=""
WORKERS=2
PORT_BASE=5556
PORT_STEP=10
LOGS_ROOT_BASE="/tmp/sonic_logs/batch_parallel"
RESULTS_ROOT="/tmp/sonic_batch_parallel"
HOST="127.0.0.1"
DEPLOY_LOGS_DIR_BASE=""

# passthrough args for worker script
TARGET_FPS="50"
CHUNK_SIZE="20"
START_FRAME="1215"
PREPEND_STAND_FRAMES="50"
BLEND_FROM_STAND_FRAMES="100"
INITIAL_BURST_FRAMES="160"
COMMAND_REPEAT="10"
COMMAND_INTERVAL="0.1"
COMMAND_HEARTBEAT_INTERVAL="0.5"
USE_ISAACSIM_APP="true"
ALIGN_MODE="source_frame_index"
METRICS_CONDA_ENV=""
ALLOW_FALLBACK_METRICS="false"
STRICT_WORKER_READY_CHECK="false"
EXPECTED_A_INSTANCES=""
PROGRESS_INTERVAL_SEC="10"

usage() {
  cat <<EOF
Usage:
  $0 --motion-list <csv> [--workers 2]
  $0 --motion-dir <dir> [--workers 2]

Required:
  One of:
    --motion-list PATH
    --motion-dir DIR

Optional:
  --workers N                          Parallel workers (default: ${WORKERS})
  --port-base N                        First worker port (default: ${PORT_BASE})
  --port-step N                        Port increment per worker (default: ${PORT_STEP})
  --host HOST                          ZMQ host (default: ${HOST})
  --logs-root-base DIR                 Default: ${LOGS_ROOT_BASE}
  --results-root DIR                   Default: ${RESULTS_ROOT}
  --deploy-logs-dir-base DIR           Optional base dir for running deploy logs.
                                       If set, worker i uses: DIR/worker_i
  --target-fps N                       Default: ${TARGET_FPS}
  --chunk-size N                       Default: ${CHUNK_SIZE}
  --start-frame N                      Default: ${START_FRAME}
  --prepend-stand-frames N             Default: ${PREPEND_STAND_FRAMES}
  --blend-from-stand-frames N          Default: ${BLEND_FROM_STAND_FRAMES}
  --initial-burst-frames N             Default: ${INITIAL_BURST_FRAMES}
  --command-repeat N                   Default: ${COMMAND_REPEAT}
  --command-interval SEC               Default: ${COMMAND_INTERVAL}
  --command-heartbeat-interval SEC     Default: ${COMMAND_HEARTBEAT_INTERVAL}
  --use-isaacsim-app / --no-use-isaacsim-app
  --align-mode MODE                    source_frame_index|index|auto_q29
  --metrics-conda-env NAME             Pass through to worker metrics stage
  --allow-fallback-metrics             Pass through to worker metrics stage
  --strict-worker-ready-check          Validate per-worker deploy logs freshness and A-instance count
  --expected-a-instances N             Expected number of run_sim_loop.py processes (default: --workers when strict check is enabled)
  --progress-interval-sec N            Progress print interval in seconds (default: ${PROGRESS_INTERVAL_SEC}; 0 disables)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-list) MOTION_LIST="$2"; shift 2 ;;
    --motion-dir) MOTION_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --port-step) PORT_STEP="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --logs-root-base) LOGS_ROOT_BASE="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --deploy-logs-dir-base) DEPLOY_LOGS_DIR_BASE="$2"; shift 2 ;;
    --target-fps) TARGET_FPS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --prepend-stand-frames) PREPEND_STAND_FRAMES="$2"; shift 2 ;;
    --blend-from-stand-frames) BLEND_FROM_STAND_FRAMES="$2"; shift 2 ;;
    --initial-burst-frames) INITIAL_BURST_FRAMES="$2"; shift 2 ;;
    --command-repeat) COMMAND_REPEAT="$2"; shift 2 ;;
    --command-interval) COMMAND_INTERVAL="$2"; shift 2 ;;
    --command-heartbeat-interval) COMMAND_HEARTBEAT_INTERVAL="$2"; shift 2 ;;
    --use-isaacsim-app) USE_ISAACSIM_APP="true"; shift ;;
    --no-use-isaacsim-app) USE_ISAACSIM_APP="false"; shift ;;
    --align-mode) ALIGN_MODE="$2"; shift 2 ;;
    --metrics-conda-env) METRICS_CONDA_ENV="$2"; shift 2 ;;
    --allow-fallback-metrics) ALLOW_FALLBACK_METRICS="true"; shift ;;
    --strict-worker-ready-check) STRICT_WORKER_READY_CHECK="true"; shift ;;
    --expected-a-instances) EXPECTED_A_INSTANCES="$2"; shift 2 ;;
    --progress-interval-sec) PROGRESS_INTERVAL_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$MOTION_LIST" && -n "$MOTION_DIR" ]]; then
  echo "[ERROR] use only one of --motion-list / --motion-dir"
  exit 1
fi
if [[ -z "$MOTION_LIST" && -z "$MOTION_DIR" ]]; then
  echo "[ERROR] one of --motion-list / --motion-dir is required"
  exit 1
fi
if [[ -n "$MOTION_LIST" && ! -f "$MOTION_LIST" ]]; then
  echo "[ERROR] --motion-list file not found: ${MOTION_LIST}"
  exit 1
fi
if [[ -n "$MOTION_DIR" && ! -d "$MOTION_DIR" ]]; then
  echo "[ERROR] --motion-dir dir not found: ${MOTION_DIR}"
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "$LOGS_ROOT_BASE"

rm -f "${RESULTS_ROOT}"/summary.json "${RESULTS_ROOT}"/summary.csv
rm -f "${RESULTS_ROOT}"/worker_*.json "${RESULTS_ROOT}"/worker_*.csv "${RESULTS_ROOT}"/worker_*.log

TMPDIR="$(mktemp -d /tmp/sonic_batch_parallel.XXXXXX)"
PIDS=()
progress_pid=""
cleanup() {
  if [[ -n "$progress_pid" ]]; then
    kill "$progress_pid" >/dev/null 2>&1 || true
  fi
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

if [[ -n "$DEPLOY_LOGS_DIR_BASE" ]]; then
  for ((i=0; i<WORKERS; i++)); do
    d="${DEPLOY_LOGS_DIR_BASE}/worker_${i}"
    if [[ ! -d "$d" ]]; then
      echo "[ERROR] missing deploy logs dir for worker ${i}: $d"
      echo "[HINT] start deploy worker ${i} with --enable-csv-logs --logs-dir $d"
      exit 1
    fi
  done
fi

# Build clean list
CLEAN_LIST="$TMPDIR/motion_list.clean.csv"
if [[ -n "$MOTION_LIST" ]]; then
  awk 'NF && $1 !~ /^#/' "$MOTION_LIST" > "$CLEAN_LIST"
else
  find "$MOTION_DIR" -type f -name "*.pkl" | sort | while read -r f; do
    echo "$f,"
  done > "$CLEAN_LIST"
fi

total_lines="$(wc -l < "$CLEAN_LIST" | tr -d ' ')"
if [[ "$total_lines" -eq 0 ]]; then
  echo "[ERROR] motion list is empty after filtering"
  exit 1
fi

echo "[INFO] total motions: $total_lines"
echo "[INFO] workers: $WORKERS"
echo "[INFO] results_root: $RESULTS_ROOT"

if [[ "$STRICT_WORKER_READY_CHECK" == "true" ]]; then
  expected_a="$EXPECTED_A_INSTANCES"
  if [[ -z "$expected_a" ]]; then
    expected_a="$WORKERS"
  fi
  if command -v rg >/dev/null 2>&1; then
    sim_count="$(ps -ef | rg -c 'gear_sonic/scripts/run_sim_loop.py' || true)"
  else
    sim_count="$(ps -ef | grep -c 'gear_sonic/scripts/run_sim_loop.py' || true)"
  fi
  sim_count="${sim_count:-0}"
  if [[ "$sim_count" -lt "$expected_a" ]]; then
    echo "[ERROR] strict check failed: run_sim_loop.py instances=${sim_count}, expected at least ${expected_a}"
    echo "[HINT] true parallel sim requires one MuJoCo A instance per worker"
    exit 1
  fi
  if [[ -n "$DEPLOY_LOGS_DIR_BASE" ]]; then
    now_ts="$(date +%s)"
    for ((i=0; i<WORKERS; i++)); do
      d="${DEPLOY_LOGS_DIR_BASE}/worker_${i}"
      if [[ ! -d "$d" ]]; then
        echo "[ERROR] strict check failed: missing deploy logs dir for worker ${i}: $d"
        exit 1
      fi

      marker_file="${d}/.launch_marker"
      marker_ts=""
      if [[ -f "$marker_file" ]]; then
        marker_ts="$(stat -c %Y "$marker_file" 2>/dev/null || true)"
      fi

      latest_ready_ts=""
      metadata_ts="$(stat -c %Y "${d}/metadata.json" 2>/dev/null || true)"
      latest_csv_ts="$(find "$d" -maxdepth 1 -type f -name "*.csv" -printf '%T@\n' 2>/dev/null | sort -nr | head -n1 | cut -d. -f1)"

      if [[ -n "$metadata_ts" ]]; then
        latest_ready_ts="$metadata_ts"
      fi
      if [[ -n "$latest_csv_ts" && ( -z "$latest_ready_ts" || "$latest_csv_ts" -gt "$latest_ready_ts" ) ]]; then
        latest_ready_ts="$latest_csv_ts"
      fi

      if [[ -z "$latest_ready_ts" ]]; then
        echo "[ERROR] strict check failed: no deploy readiness artifacts found for worker ${i} in $d"
        echo "[HINT] ensure deploy worker ${i} is alive and writes metadata.json / csv logs"
        exit 1
      fi

      if [[ -n "$marker_ts" ]]; then
        if [[ "$latest_ready_ts" -lt "$marker_ts" ]]; then
          echo "[ERROR] strict check failed: worker ${i} deploy logs are from an older run: $d"
          echo "[HINT] current deploy worker ${i} has not produced fresh metadata/csv after this run started"
          exit 1
        fi
      else
        age=$((now_ts - latest_ready_ts))
        if [[ "$age" -gt 120 ]]; then
          echo "[ERROR] strict check failed: worker ${i} deploy logs look stale (age=${age}s): $d"
          echo "[HINT] confirm deploy worker ${i} is alive and bound to its dedicated ports"
          exit 1
        fi
      fi
    done
  fi
  echo "[INFO] strict worker-ready check passed"
fi

for ((i=0; i<WORKERS; i++)); do
  awk -F, -v mod="$WORKERS" -v idx="$i" '((NR-1) % mod) == idx {print $0}' "$CLEAN_LIST" > "$TMPDIR/worker_${i}.csv"
done

for ((i=0; i<WORKERS; i++)); do
  worker_list="$TMPDIR/worker_${i}.csv"
  count_i="$(wc -l < "$worker_list" | tr -d ' ')"
  if [[ "$count_i" -eq 0 ]]; then
    continue
  fi
  worker_port=$((PORT_BASE + i * PORT_STEP))
  worker_logs="${LOGS_ROOT_BASE}/worker_${i}"
  worker_json="${RESULTS_ROOT}/worker_${i}.json"
  worker_csv="${RESULTS_ROOT}/worker_${i}.csv"
  worker_log="${RESULTS_ROOT}/worker_${i}.log"
  mkdir -p "$worker_logs"

  cmd=(
    tools/sonic_eval/run_mujoco_batch_eval.sh
    --motion-list "$worker_list"
    --logs-root "$worker_logs"
    --results-json "$worker_json"
    --results-csv "$worker_csv"
    --host "$HOST"
    --port "$worker_port"
    --target-fps "$TARGET_FPS"
    --chunk-size "$CHUNK_SIZE"
    --start-frame "$START_FRAME"
    --prepend-stand-frames "$PREPEND_STAND_FRAMES"
    --blend-from-stand-frames "$BLEND_FROM_STAND_FRAMES"
    --initial-burst-frames "$INITIAL_BURST_FRAMES"
    --command-repeat "$COMMAND_REPEAT"
    --command-interval "$COMMAND_INTERVAL"
    --command-heartbeat-interval "$COMMAND_HEARTBEAT_INTERVAL"
    --align-mode "$ALIGN_MODE"
  )
  if [[ -n "$DEPLOY_LOGS_DIR_BASE" ]]; then
    cmd+=(--deploy-logs-dir "${DEPLOY_LOGS_DIR_BASE}/worker_${i}")
  fi
  if [[ "$USE_ISAACSIM_APP" == "true" ]]; then
    cmd+=(--use-isaacsim-app)
  else
    cmd+=(--no-use-isaacsim-app)
  fi
  if [[ -n "$METRICS_CONDA_ENV" ]]; then
    cmd+=(--metrics-conda-env "$METRICS_CONDA_ENV")
  fi
  if [[ "$ALLOW_FALLBACK_METRICS" == "true" ]]; then
    cmd+=(--allow-fallback-metrics)
  fi

  echo "[INFO] launch worker=${i} motions=${count_i} port=${worker_port} log=${worker_log}"
  "${cmd[@]}" > "$worker_log" 2>&1 &
  PIDS+=("$!")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[ERROR] no workers launched"
  exit 1
fi

if [[ "$PROGRESS_INTERVAL_SEC" -gt 0 ]]; then
  (
    while true; do
      if command -v rg >/dev/null 2>&1; then
        done_count="$(rg -c '^\[OK\] done motion:' "${RESULTS_ROOT}"/worker_*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')"
      else
        done_count="$(
          grep -h -c '^\[OK\] done motion:' "${RESULTS_ROOT}"/worker_*.log 2>/dev/null \
          | awk '{s+=$1} END {print s+0}'
        )"
      fi
      running=0
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
          running=$((running + 1))
        fi
      done
      echo "[INFO] progress: ${done_count}/${total_lines} done, running_workers=${running}/${#PIDS[@]}"
      if [[ "$running" -eq 0 ]]; then
        break
      fi
      sleep "$PROGRESS_INTERVAL_SEC"
    done
  ) &
  progress_pid="$!"
fi

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

python - <<PY
import csv
import json
from pathlib import Path

root = Path("${RESULTS_ROOT}")
rows = []
for p in sorted(root.glob("worker_*.json")):
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            rows.extend(data)
    except Exception:
        pass

out_json = root / "summary.json"
out_csv = root / "summary.csv"
out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

fields = [
    "motion_file", "motion_name", "logs_dir", "metric_backend", "num_frames",
    "align_mode", "align_lag_frames",
    "mpjpe_g", "mpjpe_l", "mpjpe_pa", "vel_dist", "accel_dist", "success_rate"
]
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

print(f"[OK] merged rows: {len(rows)}")
print(f"[OK] {out_json}")
print(f"[OK] {out_csv}")
PY

if [[ "$fail" -ne 0 ]]; then
  echo "[WARN] one or more workers failed; check ${RESULTS_ROOT}/worker_*.log"
  exit 1
fi

echo "[OK] parallel batch complete"
