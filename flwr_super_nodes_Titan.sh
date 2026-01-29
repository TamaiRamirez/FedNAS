#!/usr/bin/env bash

set -euo pipefail

NODES=2
PIDS=()
LOGDIR=/workspace/flwr-enas-nofix/logs

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