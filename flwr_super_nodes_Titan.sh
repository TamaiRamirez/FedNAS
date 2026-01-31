#!/usr/bin/env bash

set -euo pipefail

NODES=2
PIDS=()

BASE_DIR="/workspace/flwr-enas-nofix/experiments"
TODAY=$(date +"%Y%m%d")

# Buscar subcarpetas del día con formato YYYYMMDD_N
LAST_EXP=$(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -name "${TODAY}_*" \
           -printf "%f\n" 2>/dev/null | sort -V | tail -n 1)

if [[ -z "$LAST_EXP" ]]; then
  echo "No se encontró experimento para hoy"
  exit 1
fi

EXP_DIR="$BASE_DIR/$LAST_EXP"

echo "Usando EXPERIMENT_DIR=$EXP_DIR"
LOGDIR="$EXP_DIR/logs"

cleanup() {
  echo "Cerrando supernodes Titan..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
}

trap cleanup SIGINT SIGTERM

export PYTHONUNBUFFERED=1

for i in $(seq 0 $((NODES-1))); do
  echo "Lanzando supernode $((4+i))"

  flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:$((9098+i)) \
    --node-config "partition-id=$((4+i)) num-partitions=6" \
    > "$LOGDIR/supernode_$((4+i)).log" 2>&1 \
    &

  PIDS+=($!)
done

wait