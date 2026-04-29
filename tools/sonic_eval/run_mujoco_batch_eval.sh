#!/usr/bin/env bash
set -euo pipefail

# Batch helper for MuJoCo sim2sim eval.
# It keeps the existing pipeline unchanged and only orchestrates:
#   C) stream motionlib to deploy
#   D) compute offline tracking metrics
#
# Expected to run while:
#   - Terminal A: run_sim_loop.py (MuJoCo)
#   - Terminal B: deploy.sh with --input-type zmq_manager (listening on host:port)
#
# Motion list format (CSV, no header):
#   motion_file,motion_name
#   sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl,walk_forward_amateur_001__A001
#   sample_data/robot_filtered/210531/walk_forward_amateur_001__A001_M.pkl,walk_forward_amateur_001__A001_M
#
# Optional:
#   - motion_name can be empty if pkl contains a single motion.
#   - lines starting with '#' are ignored.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MOTION_LIST=""
MOTION_DIR=""
LOGS_ROOT="/tmp/sonic_logs/batch_zmq"
RESULTS_JSON="/tmp/sonic_batch_metrics_summary.json"
RESULTS_CSV="/tmp/sonic_batch_metrics_summary.csv"
DEPLOY_LOGS_DIR=""

# Stream args (C)
HOST="127.0.0.1"
PORT="5556"
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
METRICS_CONDA_ENV=""
ALLOW_FALLBACK_METRICS="false"

# Metric args (D)
ALIGN_MODE="source_frame_index"

usage() {
  cat <<EOF
Usage:
  $0 --motion-list <csv>
  $0 --motion-dir <dir>

Required:
  One of:
    --motion-list PATH                   CSV: motion_file,motion_name
    --motion-dir DIR                     auto-scan *.pkl under DIR (recursive)

Optional:
  --logs-root DIR                        Per-motion logs root (default: ${LOGS_ROOT})
  --results-json PATH                    Summary json (default: ${RESULTS_JSON})
  --results-csv PATH                     Summary csv (default: ${RESULTS_CSV})
  --deploy-logs-dir DIR                  If set, copy CSV logs from this running deploy dir
                                         (useful when Terminal B has fixed --logs-dir)
  --host HOST                            ZMQ host (default: ${HOST})
  --port PORT                            ZMQ port (default: ${PORT})
  --target-fps N                         Stream target fps (default: ${TARGET_FPS})
  --chunk-size N                         Stream chunk size (default: ${CHUNK_SIZE})
  --start-frame N                        Stream start frame (default: ${START_FRAME})
  --prepend-stand-frames N               Default: ${PREPEND_STAND_FRAMES}
  --blend-from-stand-frames N            Default: ${BLEND_FROM_STAND_FRAMES}
  --initial-burst-frames N               Default: ${INITIAL_BURST_FRAMES}
  --command-repeat N                     Default: ${COMMAND_REPEAT}
  --command-interval SEC                 Default: ${COMMAND_INTERVAL}
  --command-heartbeat-interval SEC       Default: ${COMMAND_HEARTBEAT_INTERVAL}
  --use-isaacsim-app / --no-use-isaacsim-app
  --metrics-conda-env NAME               Conda env for D/metrics. If omitted, auto-detect one with smpl_sim.
  --allow-fallback-metrics               Pass through to compute_mujoco_tracking_metrics.py
  --align-mode MODE                      source_frame_index|index|auto_q29 (default: ${ALIGN_MODE})

Notes:
  - This script does not start/stop Terminal A/B.
  - It only runs C + D for each motion row.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-list) MOTION_LIST="$2"; shift 2 ;;
    --motion-dir) MOTION_DIR="$2"; shift 2 ;;
    --logs-root) LOGS_ROOT="$2"; shift 2 ;;
    --results-json) RESULTS_JSON="$2"; shift 2 ;;
    --results-csv) RESULTS_CSV="$2"; shift 2 ;;
    --deploy-logs-dir) DEPLOY_LOGS_DIR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
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
    --metrics-conda-env) METRICS_CONDA_ENV="$2"; shift 2 ;;
    --allow-fallback-metrics) ALLOW_FALLBACK_METRICS="true"; shift ;;
    --align-mode) ALIGN_MODE="$2"; shift 2 ;;
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
  usage
  exit 1
fi

mkdir -p "$LOGS_ROOT"
mkdir -p "$(dirname "$RESULTS_JSON")" "$(dirname "$RESULTS_CSV")"

if [[ -f /home/lab/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/lab/miniconda3/etc/profile.d/conda.sh
fi
conda activate sonic

detect_metrics_env() {
  if [[ -n "$METRICS_CONDA_ENV" ]]; then
    return 0
  fi
  if python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("smpl_sim") else 1)
PY
  then
    METRICS_CONDA_ENV="sonic"
    return 0
  fi
  for cand in sonic_eval sonic_backup; do
    if conda run -n "$cand" python - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("smpl_sim") else 1)
PY
    then
      METRICS_CONDA_ENV="$cand"
      return 0
    fi
  done
}

detect_metrics_env
if [[ -z "$METRICS_CONDA_ENV" ]]; then
  echo "[ERROR] no conda env with smpl_sim found; use --metrics-conda-env or install smpl_sim"
  exit 1
fi

TMP_SUMMARY_JSONL="$(mktemp /tmp/sonic_batch_summary.XXXXXX.jsonl)"
TMP_MOTION_LIST="$(mktemp /tmp/sonic_batch_motion_list.XXXXXX.csv)"
trap 'rm -f "$TMP_SUMMARY_JSONL" "$TMP_MOTION_LIST"' EXIT

if [[ -n "$MOTION_LIST" ]]; then
  if [[ ! -f "$MOTION_LIST" ]]; then
    echo "[ERROR] motion list file not found: $MOTION_LIST"
    exit 1
  fi
  cp "$MOTION_LIST" "$TMP_MOTION_LIST"
else
  if [[ ! -d "$MOTION_DIR" ]]; then
    echo "[ERROR] motion dir not found: $MOTION_DIR"
    exit 1
  fi
  # auto-scan pkl; leave motion_name empty so downstream loader can use
  # the single key inside pkl (avoids filename/key mismatch like "copy 2").
  find "$MOTION_DIR" -type f -name "*.pkl" | sort | while read -r f; do
    echo "$f,"
  done > "$TMP_MOTION_LIST"
fi

echo "[INFO] batch start"
if [[ -n "$MOTION_LIST" ]]; then
  echo "[INFO] motion_list=${MOTION_LIST}"
else
  echo "[INFO] motion_dir=${MOTION_DIR}"
fi
echo "[INFO] logs_root=${LOGS_ROOT}"
echo "[INFO] metrics_conda_env=${METRICS_CONDA_ENV}"
if [[ -n "$DEPLOY_LOGS_DIR" ]]; then
  echo "[INFO] deploy_logs_dir=${DEPLOY_LOGS_DIR}"
fi

while IFS=, read -r motion_file motion_name _rest; do
  motion_file="$(echo "${motion_file:-}" | xargs)"
  motion_name="$(echo "${motion_name:-}" | xargs)"

  if [[ -z "$motion_file" ]]; then
    continue
  fi
  if [[ "${motion_file:0:1}" == "#" ]]; then
    continue
  fi

  if [[ ! -f "$motion_file" ]]; then
    echo "[WARN] skip missing file: $motion_file"
    continue
  fi

  motion_key="$motion_name"
  if [[ -z "$motion_key" ]]; then
    motion_key="$(basename "$motion_file" .pkl)"
  fi
  safe_key="$(echo "$motion_key" | sed 's#[^A-Za-z0-9._-]#_#g')"
  ts="$(date +%Y%m%d_%H%M%S)"
  logs_dir="${LOGS_ROOT}/${safe_key}_${ts}"
  out_json="${logs_dir}/metrics.json"
  mkdir -p "$logs_dir"

  echo "[INFO] stream motion: file=${motion_file} name=${motion_name:-<auto>} logs=${logs_dir}"

  stream_args=(
    python tools/sonic_eval/stream_motionlib_to_deploy.py
    --motion-file "$motion_file"
    --host "$HOST"
    --port "$PORT"
    --target-fps "$TARGET_FPS"
    --chunk-size "$CHUNK_SIZE"
    --start-frame "$START_FRAME"
    --prepend-stand-frames "$PREPEND_STAND_FRAMES"
    --blend-from-stand-frames "$BLEND_FROM_STAND_FRAMES"
    --initial-burst-frames "$INITIAL_BURST_FRAMES"
    --realtime
    --send-command
    --command-repeat "$COMMAND_REPEAT"
    --command-interval "$COMMAND_INTERVAL"
    --command-heartbeat-interval "$COMMAND_HEARTBEAT_INTERVAL"
  )
  if [[ -n "$motion_name" ]]; then
    stream_args+=(--motion-name "$motion_name")
  fi
  if [[ "$USE_ISAACSIM_APP" == "true" ]]; then
    stream_args+=(--use-isaacsim-app)
  fi

  "${stream_args[@]}"

  if [[ -n "$DEPLOY_LOGS_DIR" ]]; then
    if [[ ! -d "$DEPLOY_LOGS_DIR" ]]; then
      echo "[ERROR] deploy logs dir not found: $DEPLOY_LOGS_DIR"
      echo "[HINT] start deploy with --enable-csv-logs --logs-dir $DEPLOY_LOGS_DIR"
      exit 1
    fi
    # Copy current deploy CSV logs snapshot to per-motion folder for stable per-motion metrics.
    find "$DEPLOY_LOGS_DIR" -maxdepth 1 -type f -name "*.csv" -exec cp {} "$logs_dir/" \;
  fi

  echo "[INFO] compute metrics: ${out_json}"
  metric_args=(
    python tools/sonic_eval/compute_mujoco_tracking_metrics.py
    --gt-format motionlib
    --motion-file "$motion_file"
    --logs-dir "$logs_dir"
    --out-json "$out_json"
    --no-motionlib-robot
    --ignore-motion-playing-mask
    --streamed-only
    --stream-start-frame "$START_FRAME"
    --stream-prepend-stand-frames "$PREPEND_STAND_FRAMES"
    --stream-blend-from-stand-frames "$BLEND_FROM_STAND_FRAMES"
    --align-mode "$ALIGN_MODE"
  )
  if [[ -n "$motion_name" ]]; then
    metric_args+=(--motion-name "$motion_name")
  fi
  if [[ "$ALLOW_FALLBACK_METRICS" == "true" ]]; then
    metric_args+=(--allow-fallback-metrics)
  fi
  conda run -n "$METRICS_CONDA_ENV" "${metric_args[@]}"
  if [[ ! -f "$out_json" ]]; then
    echo "[ERROR] metrics output not found: $out_json"
    exit 1
  fi

  python - <<PY >> "$TMP_SUMMARY_JSONL"
import json
p = "${out_json}"
d = json.load(open(p))
m = d.get("metrics_all", {})
a = d.get("alignment", {})
row = {
  "motion_file": "${motion_file}",
  "motion_name": "${motion_name}",
  "logs_dir": "${logs_dir}",
  "metric_backend": d.get("metric_backend"),
  "num_frames": d.get("num_frames"),
  "align_mode": a.get("mode"),
  "align_lag_frames": a.get("lag_frames_log_vs_gt"),
  "mpjpe_g": m.get("mpjpe_g"),
  "mpjpe_l": m.get("mpjpe_l"),
  "mpjpe_pa": m.get("mpjpe_pa"),
  "vel_dist": m.get("vel_dist"),
  "accel_dist": m.get("accel_dist"),
  "success_rate": d.get("metrics_success", {}).get("success_rate"),
}
print(json.dumps(row, ensure_ascii=False))
PY

  echo "[OK] done motion: ${motion_key}"
done < "$TMP_MOTION_LIST"

python - <<PY
import csv
import json
from pathlib import Path

jsonl = Path("${TMP_SUMMARY_JSONL}")
rows = []
if jsonl.exists():
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

Path("${RESULTS_JSON}").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

fields = [
    "motion_file", "motion_name", "logs_dir", "metric_backend", "num_frames",
    "align_mode", "align_lag_frames",
    "mpjpe_g", "mpjpe_l", "mpjpe_pa", "vel_dist", "accel_dist", "success_rate"
]
with open("${RESULTS_CSV}", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

print(f"[OK] wrote {len(rows)} rows to ${RESULTS_JSON}")
print(f"[OK] wrote {len(rows)} rows to ${RESULTS_CSV}")
PY

echo "[OK] batch complete"
