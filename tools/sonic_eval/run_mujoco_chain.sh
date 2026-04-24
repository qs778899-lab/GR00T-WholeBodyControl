#!/usr/bin/env bash
set -euo pipefail

# Run chain helper:
# 1) convert parquet -> deploy motion folder
# 2) start MuJoCo sim loop (background)
# 3) start deploy policy in sim mode with converted motion
#
# Notes:
# - This script does not modify project source files.
# - If your environment has no 'just'/build artifact, deploy.sh may fail at build step.
# - In that case, you can still use generated motion folder for manual runs later.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARQUET_PATH="${1:-/home/lab/Desktop/data/data/chunk-000/episode_000000.parquet}"
MOTION_NAME="${2:-episode_000000_from_parquet}"
MOTION_ROOT="${3:-/tmp/sonic_motions_from_parquet}"

cd "$REPO_ROOT"

# Prefer conda sonic env for pandas/pyarrow/scipy/pinocchio.
if [[ -f /home/lab/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/lab/miniconda3/etc/profile.d/conda.sh
  conda activate sonic
fi

python tools/sonic_eval/parquet_to_mujoco_motion.py \
  --parquet "$PARQUET_PATH" \
  --output-root "$MOTION_ROOT" \
  --motion-name "$MOTION_NAME" \
  --meta-info-json /home/lab/Desktop/data/meta/info.json

MOTION_DIR="$MOTION_ROOT/$MOTION_NAME"
echo "[INFO] motion dir: $MOTION_DIR"

# Start simulator in background
python gear_sonic/scripts/run_sim_loop.py --interface sim --simulator mujoco --env-name default > /tmp/sonic_sim_loop.log 2>&1 &
SIM_PID=$!
echo "[INFO] started sim loop pid=$SIM_PID (log: /tmp/sonic_sim_loop.log)"

# Wait a bit for sim init
sleep 3

cd gear_sonic_deploy
# Auto-confirm deploy.sh prompt
printf 'Y\n' | bash ./deploy.sh sim \
  --motion-data "$MOTION_DIR" \
  --obs-config policy/release/observation_config.yaml \
  --input-type manager \
  --output-type all \
  --zmq-host localhost

