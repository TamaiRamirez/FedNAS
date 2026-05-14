#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/workspace/flwr-enas-nofix/ablation"
TODAY=$(date +"%Y%m%d")

mkdir -p "$BASE_DIR"

# Buscar el último experimento del día
LAST_EXP=$(ls -d "$BASE_DIR"/"$TODAY"_* 2>/dev/null | sort | tail -n 1 || true)

if [[ -z "$LAST_EXP" ]]; then
  EXP_NUM=1
else
  LAST_NUM=${LAST_EXP##*_}
  EXP_NUM=$((LAST_NUM + 1))
fi

EXP_ID="${TODAY}_${EXP_NUM}"
EXP_DIR="$BASE_DIR/$EXP_ID"

mkdir -p "$EXP_DIR/logs"
mkdir -p "$EXP_DIR/results"
mkdir -p "$EXP_DIR/controller_weights"
mkdir -p "$EXP_DIR/label_distributions"

echo "Creado experimento: $EXP_ID"
echo "$EXP_DIR"