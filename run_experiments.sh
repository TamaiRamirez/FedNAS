#!/usr/bin/env bash
# This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirez)
set -euo pipefail
ulimit -c 0

#######################################
# Paths
#######################################
BASE_DIR="/workspace/flwr-enas-nofix"
PROJECT_TOML="$BASE_DIR/pyproject.toml"

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

#######################################
# Default configuration
#######################################
DEFAULT_NUM_ROUNDS=40
DEFAULT_NUM_NODES=6
DEFAULT_LOCAL_EPOCHS=10
DEFAULT_BATCH_SIZE=128
DEFAULT_NON_IID=true
DEFAULT_ALPHA=0.1
DEFAULT_DATASET="uoft-cs/cifar10"  # uoft-cs/cifar10, uoft-cs/cifar100, ylecun/mnist, flwrlabs/femnist
DEFAULT_DATASET_NAME="cifar10"      # cifar10, cifar100, mnist, femnist
DEFAULT_NUM_CLASSES=10              # 10, 100, 10, 62
DEFAULT_AGG="custom"
DEFAULT_MU=1.0
DEFAULT_ABLATION=false
DEFAULT_BETA=0.0
DEFAULT_Q_PARAM=0.0
DEFAULT_TVD=1.0
DEFAULT_CLIP=true
DEFAULT_CLIP_VALUE=3
DEFAULT_SEED=42
DEFAULT_MODE="local-deployment"
DEFAULT_FRACTION_FIT=1.0
DEFAULT_FRACTION_EVALUATE=1.0

#######################################
# Experiments
#######################################

# "mode|parametro=valor|parametro=valor"
EXPERIMENTS=(
  # Q ablation
  "local-deployment|q_param=0.0"
)

NUM_EXPS=${#EXPERIMENTS[@]}

#######################################
# Runtime PID
#######################################
NODES_PID=""

#######################################
# Functions
#######################################
update_project_toml() {
  local tmp_file
  tmp_file=$(mktemp)

  sed \
    -e "s/^num-server-rounds = .*/num-server-rounds = ${1}/" \
    -e "s/^num-nodes = .*/num-nodes = ${2}/" \
    -e "s/^local-epochs = .*/local-epochs = ${3}/" \
    -e "s/^batch-size = .*/batch-size = ${4}/" \
    -e "s/^non_iid = .*/non_iid = ${5}/" \
    -e "s/^alpha = .*/alpha = ${6}/" \
    -e "s|^dataset = .*|dataset = \"${7}\"|" \
    -e "s/^dataset_name = .*/dataset_name = \"${8}\"/" \
    -e "s/^num-classes = .*/num-classes = ${9}/" \
    -e "s/^agg_method = .*/agg_method = \"${10}\"/" \
    -e "s/^mu = .*/mu = ${11}/" \
    -e "s/^ablation = .*/ablation = ${12}/" \
    -e "s/^beta = .*/beta = ${13}/" \
    -e "s/^q_param = .*/q_param = ${14}/" \
    -e "s/^seed = .*/seed = ${15}/" \
    -e "s/^tvd = .*/tvd = ${16}/" \
    -e "s/^clip = .*/clip = ${17}/" \
    -e "s/^clip_value = .*/clip_value = ${18}/" \
    -e "s/^fraction_fit = .*/fraction_fit = ${19}/" \
    -e "s/^fraction_evaluate = .*/fraction_evaluate = ${20}/" \
    -e "s/^mode = .*/mode = \"${21}\"/" \
    "$PROJECT_TOML" > "$tmp_file"

  cat "$tmp_file" > "$PROJECT_TOML"
  rm -f "$tmp_file"
}

load_experiment() {
  local experiment="$1"
  local field key value

  #######################################
  # Reset to default configuration
  #######################################
  NUM_ROUNDS="$DEFAULT_NUM_ROUNDS"
  NUM_NODES="$DEFAULT_NUM_NODES"
  LOCAL_EPOCHS="$DEFAULT_LOCAL_EPOCHS"
  BATCH_SIZE="$DEFAULT_BATCH_SIZE"
  NON_IID="$DEFAULT_NON_IID"
  ALPHA="$DEFAULT_ALPHA"
  DATASET="$DEFAULT_DATASET"
  DATASET_NAME="$DEFAULT_DATASET_NAME"
  NUM_CLASSES="$DEFAULT_NUM_CLASSES"
  AGG="$DEFAULT_AGG"
  MU="$DEFAULT_MU"
  ABLATION="$DEFAULT_ABLATION"
  BETA="$DEFAULT_BETA"
  Q_PARAM="$DEFAULT_Q_PARAM"
  SEED="$DEFAULT_SEED"
  TVD="$DEFAULT_TVD"
  CLIP="$DEFAULT_CLIP"
  CLIP_VALUE="$DEFAULT_CLIP_VALUE"
  FRACTION_FIT="$DEFAULT_FRACTION_FIT"
  FRACTION_EVALUATE="$DEFAULT_FRACTION_EVALUATE"

  #######################################
  # Parse experiment configuration
  #######################################
  local -a fields
  IFS='|' read -r -a fields <<< "$experiment"

  MODE="${fields[0]:-}"

  case "$MODE" in
    simulation|local-deployment)
      ;;
    *)
      echo "❌ Invalid execution mode: $MODE" >&2
      return 1
      ;;
  esac

  #######################################
  # Apply experiment overrides
  #######################################
  for field in "${fields[@]:1}"; do

    if [[ "$field" != *=* ]]; then
      echo "❌ Invalid configuration: $field" >&2
      return 1
    fi

    key="${field%%=*}"
    value="${field#*=}"

    if [[ -z "$value" ]]; then
      echo "❌ Empty value for: $key" >&2
      return 1
    fi

    case "$key" in
      num-server-rounds) NUM_ROUNDS="$value" ;;
      num-nodes) NUM_NODES="$value" ;;
      local-epochs) LOCAL_EPOCHS="$value" ;;
      batch-size) BATCH_SIZE="$value" ;;
      non_iid)    NON_IID="$value" ;;
      alpha)      ALPHA="$value" ;;
      dataset) DATASET="$value" ;;
      dataset_name) DATASET_NAME="$value" ;;
      num-classes) NUM_CLASSES="$value" ;;
      agg_method) AGG="$value" ;;
      mu)         MU="$value" ;;
      ablation)         ABLATION="$value" ;;
      beta)       BETA="$value" ;;
      q_param)    Q_PARAM="$value" ;;
      seed)    SEED="$value" ;;
      tvd) TVD="$value" ;;
      clip) CLIP="$value" ;;
      clip_value) CLIP_VALUE="$value" ;;
      fraction_fit) FRACTION_FIT="$value" ;;
      fraction_evaluate) FRACTION_EVALUATE="$value" ;;
      *)
        echo "❌ Unknown parameter: $key" >&2
        return 1
        ;;
    esac
  done
}

#######################################
# Main experiment loop
#######################################
for ((i=0; i<NUM_EXPS; i++)); do
  echo "========================================"
  echo "🧪 EXPERIMENT $((i+1)) / $NUM_EXPS"
  echo "========================================"

  #######################################
  # Load experiment configuration
  #######################################
  load_experiment "${EXPERIMENTS[$i]}"

  update_project_toml \
    "$NUM_ROUNDS" \
    "$NUM_NODES" \
    "$LOCAL_EPOCHS" \
    "$BATCH_SIZE" \
    "$NON_IID" \
    "$ALPHA" \
    "$DATASET" \
    "$DATASET_NAME" \
    "$NUM_CLASSES" \
    "$AGG" \
    "$MU" \
    "$ABLATION" \
    "$BETA" \
    "$Q_PARAM" \
    "$SEED" \
    "$TVD" \
    "$CLIP" \
    "$CLIP_VALUE" \
    "$FRACTION_FIT" \
    "$FRACTION_EVALUATE" \
    "$MODE"

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

  case "$MODE" in
    simulation)
      FLWR_CMD=(
        flwr run "$BASE_DIR"
        --federation-config="num-supernodes=${NUM_NODES} client-resources-num-cpus=10 client-resources-num-gpus=0.1"
        --stream
      )
      ;;
    local-deployment)
      FLWR_CMD=(
        flwr run "$BASE_DIR" local-deployment
        --stream
      )
      ;;
    *)
      echo "❌ Invalid execution mode: $MODE" >&2
      exit 1
      ;;
  esac

  if ! "${FLWR_CMD[@]}" 2>&1 | tee "$LOGDIR/server.log"; then
    echo "❌ Experiment $((i+1)) failed"
    continue
  fi
  
  echo "Running Selection Best ROUND..."
  python3 "$BASE_DIR"/manual_early_stopping.py

  # Cleanup controller weights
  rm -rf "$EXP_DIR/controller_weights" 2>/dev/null || true

  echo "✅ Experiment $((i+1)) finished"
  
  sleep 2
done

echo "🎉 All experiments completed!"
