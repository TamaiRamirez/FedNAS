#!/usr/bin/env bash
# This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)
set -euo pipefail
ulimit -c 0

#######################################
# Paths
#######################################
BASE_DIR="/workspace/flwr-enas-nofix"
PROJECT_TOML="$BASE_DIR/pyproject.toml"
# NODES_SCRIPT="./flwr_super_nodes_4090.sh"

#######################################
# Environment
#######################################
export RICH_DISABLE=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export ABSL_LOGGING_MIN_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID

#######################################
# Experiment configurations
#######################################
NON_IID_LIST=(true)
ALPHA_LIST=(0.5)
AGG_LIST=("qfedavg")
MU_LIST=(0.0)
BETA_LIST=(0.0)
Q_PARAM_LIST=(0.1)

NUM_EXPS=${#NON_IID_LIST[@]}

#######################################
# Runtime PID
#######################################
NODES_PID=""

#######################################
# Functions
#######################################
update_project_toml() {
  sed -i \
    -e "s/^non_iid = .*/non_iid = ${1}/" \
    -e "s/^alpha = .*/alpha = ${2}/" \
    -e "s/^agg_method = .*/agg_method = \"${3}\"/" \
    -e "s/^mu = .*/mu = ${4}/" \
    -e "s/^beta = .*/beta = ${5}/" \
    -e "s/^q_param = .*/q_param = ${6}/" \
    "$PROJECT_TOML"
}


#######################################
# Main experiment loop
#######################################
for ((i=0; i<NUM_EXPS; i++)); do
  echo "========================================"
  echo "🧪 EXPERIMENT $((i+1)) / $NUM_EXPS"
  echo "========================================"

  update_project_toml \
    "${NON_IID_LIST[$i]}" \
    "${ALPHA_LIST[$i]}" \
    "${AGG_LIST[$i]}" \
    "${MU_LIST[$i]}" \
    "${BETA_LIST[$i]}" \
    "${Q_PARAM_LIST[$i]}"

  #######################################
  # Create experiment
  #######################################
  EXP_DIR=$(./create_experiment.sh | tail -n 1)
  LOGDIR="$EXP_DIR/logs"

  echo "📁 Experiment directory: $EXP_DIR"

  #######################################
  # Run server
  #######################################
  echo "🚀 Launching Flower server"
  
  if ! flwr run "$BASE_DIR" local-deployment --stream 2>&1 | tee "$LOGDIR/server.log"; then
    echo "❌ Experiment $((i+1)) failed"
    continue
  fi

  # if ! flwr run "$BASE_DIR" --federation-config="num-supernodes=6 client-resources-num-cpus=8 client-resources-num-gpus=0.5" --stream 2>&1 | tee "$LOGDIR/server.log"; then
  #   echo "❌ Experiment $((i+1)) failed"
  #   continue
  # fi

  echo "✅ Experiment $((i+1)) finished"
  
  # Cleanup controller weights
  rm -rf "$EXP_DIR/controller_weights" 2>/dev/null || true

  
  echo "Running Selection Best ROUND..."
  python3 "$BASE_DIR"/manual_early_stopping.py
  
  sleep 2
done

echo "🎉 All experiments completed!"
