#!/usr/bin/env bash
set -euo pipefail

# One-command orchestrator:
#  - Launch N MuJoCo sim instances (A workers) with isolated DDS domains
#  - Launch N deploy instances (B workers) with isolated ZMQ ports/log dirs/domains
#  - Run existing parallel batch pipeline (C)
#
# Default behavior is additive and does not change existing single/batch scripts.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
RUN_START_TS="$(date +%s)"

MOTION_DIR=""
MOTION_LIST=""
WORKERS=2
HOST="127.0.0.1"
PORT_BASE=5556
PORT_STEP=10
ZMQ_OUT_BASE=5557
ZMQ_OUT_STEP=10
DOMAIN_BASE=100

LOGS_ROOT_BASE="/tmp/sonic_logs/batch_parallel"
DEPLOY_LOGS_DIR_BASE="/tmp/sonic_logs/parallel_deploy"
RESULTS_ROOT="/tmp/sonic_batch_parallel"
RUN_ROOT="/tmp/sonic_multi_instance_run"

TARGET_FPS="50"
CHUNK_SIZE="20"
START_FRAME="1215"
PREPEND_STAND_FRAMES="50"
BLEND_FROM_STAND_FRAMES="100"
INITIAL_BURST_FRAMES="160"
COMMAND_REPEAT="10"
COMMAND_INTERVAL="0.1"
COMMAND_HEARTBEAT_INTERVAL="0.5"
ALIGN_MODE="source_frame_index"
SIM_CONDA_ENV="sonic"
DEPLOY_CONDA_ENV="sonic"
BATCH_CONDA_ENV="sonic"
METRICS_CONDA_ENV="sonic_backup"
SIM_PYTHON_MODE="auto"
SIM_VENV_PATH=".venv_sim"
USE_ISAACSIM_APP="true"
PROGRESS_INTERVAL_SEC="10"

DEPLOY_AUTO_BUILD="false"
SKIP_STRICT_CHECK="false"

usage() {
  cat <<EOF
Usage:
  $0 --motion-dir <dir> [--workers 4]
  $0 --motion-list <csv> [--workers 4]

Required:
  One of:
    --motion-dir DIR
    --motion-list PATH

Optional:
  --workers N
  --host HOST
  --port-base N
  --port-step N
  --zmq-out-base N
  --zmq-out-step N
  --domain-base N
  --logs-root-base DIR
  --deploy-logs-dir-base DIR
  --results-root DIR
  --run-root DIR
  --sim-conda-env NAME
  --deploy-conda-env NAME
  --batch-conda-env NAME
  --sim-conda-env NAME                  Conda env for A workers (default: ${SIM_CONDA_ENV})
  --deploy-conda-env NAME               Conda env for B workers (default: ${DEPLOY_CONDA_ENV})
  --batch-conda-env NAME                Conda env for C parallel dispatcher (default: ${BATCH_CONDA_ENV})
  --metrics-conda-env NAME              Conda env for D metrics stage (default: ${METRICS_CONDA_ENV})
  --sim-python-mode MODE                auto|conda|venv (default: ${SIM_PYTHON_MODE})
  --sim-venv-path PATH                  Venv for A workers fallback/default (default: ${SIM_VENV_PATH})
  --use-isaacsim-app / --no-use-isaacsim-app
  --progress-interval-sec N
  --align-mode MODE
  --target-fps N
  --chunk-size N
  --start-frame N
  --prepend-stand-frames N
  --blend-from-stand-frames N
  --initial-burst-frames N
  --command-repeat N
  --command-interval SEC
  --command-heartbeat-interval SEC
  --deploy-auto-build                      Run 'just build' once before launching deploy workers
  --skip-strict-check                      Skip strict multi-instance validation in parallel batch script
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-dir) MOTION_DIR="$2"; shift 2 ;;
    --motion-list) MOTION_LIST="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --port-step) PORT_STEP="$2"; shift 2 ;;
    --zmq-out-base) ZMQ_OUT_BASE="$2"; shift 2 ;;
    --zmq-out-step) ZMQ_OUT_STEP="$2"; shift 2 ;;
    --domain-base) DOMAIN_BASE="$2"; shift 2 ;;
    --logs-root-base) LOGS_ROOT_BASE="$2"; shift 2 ;;
    --deploy-logs-dir-base) DEPLOY_LOGS_DIR_BASE="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --sim-conda-env) SIM_CONDA_ENV="$2"; shift 2 ;;
    --deploy-conda-env) DEPLOY_CONDA_ENV="$2"; shift 2 ;;
    --batch-conda-env) BATCH_CONDA_ENV="$2"; shift 2 ;;
    --metrics-conda-env) METRICS_CONDA_ENV="$2"; shift 2 ;;
    --sim-python-mode) SIM_PYTHON_MODE="$2"; shift 2 ;;
    --sim-venv-path) SIM_VENV_PATH="$2"; shift 2 ;;
    --use-isaacsim-app) USE_ISAACSIM_APP="true"; shift ;;
    --no-use-isaacsim-app) USE_ISAACSIM_APP="false"; shift ;;
    --progress-interval-sec) PROGRESS_INTERVAL_SEC="$2"; shift 2 ;;
    --align-mode) ALIGN_MODE="$2"; shift 2 ;;
    --target-fps) TARGET_FPS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --prepend-stand-frames) PREPEND_STAND_FRAMES="$2"; shift 2 ;;
    --blend-from-stand-frames) BLEND_FROM_STAND_FRAMES="$2"; shift 2 ;;
    --initial-burst-frames) INITIAL_BURST_FRAMES="$2"; shift 2 ;;
    --command-repeat) COMMAND_REPEAT="$2"; shift 2 ;;
    --command-interval) COMMAND_INTERVAL="$2"; shift 2 ;;
    --command-heartbeat-interval) COMMAND_HEARTBEAT_INTERVAL="$2"; shift 2 ;;
    --deploy-auto-build) DEPLOY_AUTO_BUILD="true"; shift ;;
    --skip-strict-check) SKIP_STRICT_CHECK="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$MOTION_DIR" && -n "$MOTION_LIST" ]]; then
  echo "[ERROR] use only one of --motion-dir / --motion-list"
  exit 1
fi
if [[ -z "$MOTION_DIR" && -z "$MOTION_LIST" ]]; then
  echo "[ERROR] one of --motion-dir / --motion-list is required"
  exit 1
fi
if [[ -n "$MOTION_DIR" && ! -d "$MOTION_DIR" ]]; then
  echo "[ERROR] motion-dir not found: $MOTION_DIR"
  exit 1
fi
if [[ -n "$MOTION_LIST" && ! -f "$MOTION_LIST" ]]; then
  echo "[ERROR] motion-list not found: $MOTION_LIST"
  exit 1
fi

TOTAL_MOTIONS=0
if [[ -n "$MOTION_DIR" ]]; then
  TOTAL_MOTIONS="$(find "$MOTION_DIR" -type f -name "*.pkl" | wc -l | tr -d ' ')"
else
  TOTAL_MOTIONS="$(awk 'NF && $1 !~ /^#/' "$MOTION_LIST" | wc -l | tr -d ' ')"
fi
echo "[INFO] total motions planned: ${TOTAL_MOTIONS}"

mkdir -p "$RUN_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT_BASE" "$DEPLOY_LOGS_DIR_BASE"

rm -f "${RESULTS_ROOT}"/summary.json "${RESULTS_ROOT}"/summary.csv
rm -f "${RESULTS_ROOT}"/worker_*.json "${RESULTS_ROOT}"/worker_*.csv "${RESULTS_ROOT}"/worker_*.log

if [[ -f /home/lab/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/lab/miniconda3/etc/profile.d/conda.sh
fi

resolve_sim_launcher() {
  local mode="$SIM_PYTHON_MODE"
  if [[ "$mode" == "conda" ]]; then
    echo "conda"
    return 0
  fi
  if [[ "$mode" == "venv" ]]; then
    echo "venv"
    return 0
  fi
  if conda run -n "$SIM_CONDA_ENV" python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("unitree_sdk2py") else 1)' >/dev/null 2>&1; then
    echo "conda"
    return 0
  fi
  if [[ -x "${SIM_VENV_PATH}/bin/python" ]] && "${SIM_VENV_PATH}/bin/python" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("unitree_sdk2py") else 1)' >/dev/null 2>&1; then
    echo "venv"
    return 0
  fi
  echo "none"
}

SIM_LAUNCHER="$(resolve_sim_launcher)"
if [[ "$SIM_LAUNCHER" == "none" ]]; then
  echo "[ERROR] unable to launch A workers: neither conda env '${SIM_CONDA_ENV}' nor ${SIM_VENV_PATH} provides unitree_sdk2py"
  exit 1
fi
if [[ "$SIM_LAUNCHER" == "venv" ]]; then
  echo "[INFO] sim launcher fallback: using ${SIM_VENV_PATH} because conda env '${SIM_CONDA_ENV}' lacks unitree_sdk2py"
else
  echo "[INFO] sim launcher: using conda env '${SIM_CONDA_ENV}'"
fi

SIM_PIDS=()
DEPLOY_PIDS=()
progress_pid=""
cleanup() {
  set +e
  if [[ -n "$progress_pid" ]]; then
    kill "$progress_pid" >/dev/null 2>&1 || true
  fi
  for pid in "${DEPLOY_PIDS[@]}"; do kill "$pid" >/dev/null 2>&1 || true; done
  for pid in "${SIM_PIDS[@]}"; do kill "$pid" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT

kill_matching_cmds() {
  local pattern="$1"
  local label="$2"
  mapfile -t stale_pids < <(ps -eo pid=,args= | awk -v pat="$pattern" '$0 ~ pat {print $1}')
  if [[ "${#stale_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[WARN] cleaning stale ${label} processes: ${stale_pids[*]}"
  kill "${stale_pids[@]}" >/dev/null 2>&1 || true
  sleep 2

  mapfile -t stubborn_pids < <(ps -eo pid=,args= | awk -v pat="$pattern" '$0 ~ pat {print $1}')
  if [[ "${#stubborn_pids[@]}" -gt 0 ]]; then
    echo "[WARN] force killing stale ${label} processes: ${stubborn_pids[*]}"
    kill -9 "${stubborn_pids[@]}" >/dev/null 2>&1 || true
    sleep 1
  fi
}

cleanup_stale_workers() {
  echo "[INFO] cleaning stale workers for requested domains/ports"
  for ((i=0; i<WORKERS; i++)); do
    local domain port
    domain=$((DOMAIN_BASE + i))
    port=$((PORT_BASE + i * PORT_STEP))
    kill_matching_cmds "gear_sonic/scripts/run_sim_loop.py.*--domain-id ${domain}([[:space:]]|$)" "sim(domain=${domain})"
    kill_matching_cmds "g1_deploy_onnx_ref.*--dds-domain-id ${domain}([[:space:]]|$)" "deploy(domain=${domain})"
    kill_matching_cmds "g1_deploy_onnx_ref.*--zmq-port ${port}([[:space:]]|$)" "deploy(zmq-port=${port})"
  done
}

wait_for_worker_ready() {
  local worker_id="$1"
  local deploy_log="$2"
  local deploy_dir="$3"
  local timeout_sec="$4"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if ! kill -0 "${DEPLOY_PIDS[$worker_id]}" >/dev/null 2>&1; then
      echo "[ERROR] deploy worker ${worker_id} exited during startup"
      echo "[HINT] inspect log: ${deploy_log}"
      return 1
    fi

    if grep -q "Init Done" "$deploy_log" 2>/dev/null; then
      if [[ -f "${deploy_dir}/metadata.json" ]]; then
        return 0
      fi
    fi

    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_sec )); then
      echo "[ERROR] deploy worker ${worker_id} not ready within ${timeout_sec}s"
      echo "[HINT] inspect log: ${deploy_log}"
      return 1
    fi
    sleep 1
  done
}

cleanup_stale_workers

echo "[INFO] launching ${WORKERS} MuJoCo sim workers"
for ((i=0; i<WORKERS; i++)); do
  domain=$((DOMAIN_BASE + i))
  sim_log="${RUN_ROOT}/sim_worker_${i}.log"
  if [[ "$SIM_LAUNCHER" == "venv" ]]; then
    (
      cd "$REPO_ROOT"
      source "${SIM_VENV_PATH}/bin/activate"
      python gear_sonic/scripts/run_sim_loop.py \
        --interface sim \
        --simulator mujoco \
        --env-name default \
        --domain-id "$domain" \
        --no-enable-onscreen \
        --no-enable-offscreen
    ) >"$sim_log" 2>&1 &
  else
    (
      cd "$REPO_ROOT"
      source /home/lab/miniconda3/etc/profile.d/conda.sh
      conda activate "$SIM_CONDA_ENV"
      python gear_sonic/scripts/run_sim_loop.py \
        --interface sim \
        --simulator mujoco \
        --env-name default \
        --domain-id "$domain" \
        --no-enable-onscreen \
        --no-enable-offscreen
    ) >"$sim_log" 2>&1 &
  fi
  SIM_PIDS+=("$!")
  echo "[INFO] sim worker=${i} domain=${domain} log=${sim_log}"
done

if [[ "$DEPLOY_AUTO_BUILD" == "true" ]]; then
  echo "[INFO] building deploy binaries once"
  (
    cd "$REPO_ROOT/gear_sonic_deploy"
    source /home/lab/miniconda3/etc/profile.d/conda.sh
    conda activate "$DEPLOY_CONDA_ENV"
    just build
  )
fi

echo "[INFO] launching ${WORKERS} deploy workers"
for ((i=0; i<WORKERS; i++)); do
  domain=$((DOMAIN_BASE + i))
  port=$((PORT_BASE + i * PORT_STEP))
  out_port=$((ZMQ_OUT_BASE + i * ZMQ_OUT_STEP))
  dlog="${DEPLOY_LOGS_DIR_BASE}/worker_${i}"
  mkdir -p "$dlog"
  find "$dlog" -maxdepth 1 -type f \( -name "*.csv" -o -name "*.json" -o -name ".launch_marker" \) -delete
  touch "${dlog}/.launch_marker"
  target_log="${dlog}/target_motion.csv"
  policy_log="${dlog}/policy_input.csv"
  deploy_log="${RUN_ROOT}/deploy_worker_${i}.log"

  (
    cd "$REPO_ROOT/gear_sonic_deploy"
    source /home/lab/miniconda3/etc/profile.d/conda.sh
    conda activate "$DEPLOY_CONDA_ENV"
    just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx /tmp/sonic_motion_action_only \
      --obs-config policy/release/observation_config.yaml \
      --encoder-file policy/release/model_encoder.onnx \
      --planner-file planner/target_vel/V2/planner_sonic.onnx \
      --input-type zmq_manager \
      --output-type none \
      --zmq-host localhost \
      --zmq-port "$port" \
      --zmq-out-port "$out_port" \
      --dds-domain-id "$domain" \
      --enable-csv-logs \
      --logs-dir "$dlog" \
      --target-motion-logfile "$target_log" \
      --policy-input-logfile "$policy_log" \
      --enable-motion-recording \
      --disable-crc-check
  ) >"$deploy_log" 2>&1 &
  DEPLOY_PIDS+=("$!")
  echo "[INFO] deploy worker=${i} domain=${domain} zmq=${port} zmq_out=${out_port} log=${deploy_log}"
done

echo "[INFO] waiting for deploy workers to become ready"
for ((i=0; i<WORKERS; i++)); do
  if ! wait_for_worker_ready "$i" "${RUN_ROOT}/deploy_worker_${i}.log" "${DEPLOY_LOGS_DIR_BASE}/worker_${i}" 180; then
    exit 1
  fi
  echo "[INFO] deploy worker ${i} ready"
done

parallel_cmd=(
  conda run -n "$BATCH_CONDA_ENV"
  tools/sonic_eval/run_mujoco_batch_eval_parallel.sh
  --workers "$WORKERS"
  --host "$HOST"
  --port-base "$PORT_BASE"
  --port-step "$PORT_STEP"
  --logs-root-base "$LOGS_ROOT_BASE"
  --deploy-logs-dir-base "$DEPLOY_LOGS_DIR_BASE"
  --results-root "$RESULTS_ROOT"
  --metrics-conda-env "$METRICS_CONDA_ENV"
  --target-fps "$TARGET_FPS"
  --chunk-size "$CHUNK_SIZE"
  --start-frame "$START_FRAME"
  --prepend-stand-frames "$PREPEND_STAND_FRAMES"
  --blend-from-stand-frames "$BLEND_FROM_STAND_FRAMES"
  --initial-burst-frames "$INITIAL_BURST_FRAMES"
  --command-repeat "$COMMAND_REPEAT"
  --command-interval "$COMMAND_INTERVAL"
  --command-heartbeat-interval "$COMMAND_HEARTBEAT_INTERVAL"
  --progress-interval-sec "$PROGRESS_INTERVAL_SEC"
  --align-mode "$ALIGN_MODE"
)
if [[ -n "$MOTION_DIR" ]]; then
  parallel_cmd+=(--motion-dir "$MOTION_DIR")
else
  parallel_cmd+=(--motion-list "$MOTION_LIST")
fi
if [[ "$USE_ISAACSIM_APP" == "true" ]]; then
  parallel_cmd+=(--use-isaacsim-app)
else
  parallel_cmd+=(--no-use-isaacsim-app)
fi
if [[ "$SKIP_STRICT_CHECK" != "true" ]]; then
  parallel_cmd+=(--strict-worker-ready-check --expected-a-instances "$WORKERS")
fi

echo "[INFO] running parallel batch C/D"
if [[ "$PROGRESS_INTERVAL_SEC" -gt 0 ]]; then
  (
    while true; do
      done_count="$(grep -h -c '^\[OK\] done motion:' "${RESULTS_ROOT}"/worker_*.log 2>/dev/null | awk '{s+=$1} END {print s+0}')"
      json_count="$(ls "${RESULTS_ROOT}"/worker_*.json >/dev/null 2>&1 && wc -l < <(find "${RESULTS_ROOT}" -maxdepth 1 -type f -name 'worker_*.json') || echo 0)"
      echo "[INFO] batch progress: motions=${done_count}/${TOTAL_MOTIONS}, finished_workers=${json_count}/${WORKERS}"
      sleep "$PROGRESS_INTERVAL_SEC"
    done
  ) &
  progress_pid="$!"
fi
"${parallel_cmd[@]}"
if [[ -n "$progress_pid" ]]; then
  kill "$progress_pid" >/dev/null 2>&1 || true
  progress_pid=""
fi

echo "[OK] multi-instance run complete"
echo "[OK] summary: ${RESULTS_ROOT}/summary.json"
echo "[OK] logs: ${RUN_ROOT}"
RUN_END_TS="$(date +%s)"
RUN_ELAPSED_SEC=$((RUN_END_TS - RUN_START_TS))
echo "[OK] total elapsed: ${RUN_ELAPSED_SEC}s"
