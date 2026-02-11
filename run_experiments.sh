#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
#######################################
# Paths
#######################################
BASE_DIR="/workspace/flwr-enas-nofix"
PROJECT_TOML="$BASE_DIR/pyproject.toml"

#######################################
# Flower / Python / Tensorflow (CRÍTICO PARA DOCKER)
#######################################
export RICH_DISABLE=1
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=1

#######################################
# Experiment configurations
#######################################
NON_IID_LIST=(false true true true)
ALPHA_LIST=(0.0 1.0 0.5 0.1)
AGG_LIST=("fedavg" "fedavg" "fedavg" "fedavg")
MU_LIST=(0.0 0.0 0.0 0.0)

NUM_EXPS=${#NON_IID_LIST[@]}

#######################################
# Runtime PIDs
#######################################
SUPERLINK_PID=""
SERVER_PID=""

#######################################
# Functions
#######################################
update_project_toml() {
  sed -i \
    -e "s/^non_iid = .*/non_iid = ${1}/" \
    -e "s/^alpha = .*/alpha = ${2}/" \
    -e "s/^agg_method = .*/agg_method = \"${3}\"/" \
    -e "s/^mu = .*/mu = ${4}/" \
    "$PROJECT_TOML"
}

cleanup() {
  echo "🛑 Cleanup"

  # [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  # [[ -n "${SUPERLINK_PID:-}" ]] && kill "$SUPERLINK_PID" 2>/dev/null || true

  # pkill -9 flower-supernode 2>/dev/null || true
  # pkill -9 flower-superlink 2>/dev/null || true
  # pkill -9 -f "flwr run" 2>/dev/null || true

  # Matar servidor y todos sus hijos
  if [[ -n "${SERVER_PID:-}" ]]; then
      kill -- -"$SERVER_PID" 2>/dev/null || true
  fi

  [[ -n "${SUPERLINK_PID:-}" ]] && kill -- -"$SUPERLINK_PID" 2>/dev/null || true
  [[ -n "${SUPERNODES_PID:-}" ]] && kill -- -"$SUPERNODES_PID" 2>/dev/null || true


  # Matar superlink y supernodes
  pkill -9 -f flower-supernode 2>/dev/null || true
  pkill -9 -f flower-superlink 2>/dev/null || true
  pkill -9 -f "flwr run" 2>/dev/null || true
}

trap cleanup SIGINT SIGTERM

#######################################
# Main loop
#######################################
for ((i=0; i<NUM_EXPS; i++)); do
  echo "========================================"
  echo "🧪 EXPERIMENT $((i+1)) / $NUM_EXPS"
  echo "========================================"

  update_project_toml \
    "${NON_IID_LIST[$i]}" \
    "${ALPHA_LIST[$i]}" \
    "${AGG_LIST[$i]}" \
    "${MU_LIST[$i]}"

  #######################################
  # Create experiment
  #######################################
  EXP_DIR=$(./create_experiment.sh | tail -n 1)
  LOGDIR="$EXP_DIR/logs"

  #######################################
  # Superlink
  #######################################
  echo "🔗 Launching flower-superlink"
  setsid flower-superlink --insecure > "$LOGDIR/superlink.log" 2>&1 &
  SUPERLINK_PID=$!
  sleep 3

  #######################################
  # Supernodes
  #######################################
  echo "🖥️ Launching supernodes"
  setsid ./flwr_super_nodes_4090.sh &
  SUPERNODES_PID=$!
  sleep 5

  #######################################
  # Server
  #######################################
  echo "🚀 Launching server Flower"
  flwr run "$BASE_DIR" local-deployment \
      > "$LOGDIR/server.log" 2>&1 &
  SERVER_PID=$!
  sleep 2

  echo "🧠 Executing experiment"
  echo "👉 Ctrl+C to stop the experiment"

  #######################################
  # Safe wait loop (NO wait)
  #######################################
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep 1
  done

  cleanup
  echo "✅ Experiment finished"
  sleep 2
done

echo "🎉 All experiments completed succesfully!"
