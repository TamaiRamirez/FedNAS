#!/usr/bin/env bash
set -Eeuo pipefail

#######################################
# Config
#######################################
NODES=4
LOGDIR="/workspace/flwr-enas-nofix/logs"
RESTART_DELAY=5
MAX_RESTARTS=5
HEALTH_CHECK_INTERVAL=3

mkdir -p "$LOGDIR"

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
# PID tracking
#######################################
SUPERLINK_PID=""
declare -a SUPERNODE_PIDS=()
declare -a RESTART_COUNTS=()
declare -a NODE_LOGFILES=()

for i in $(seq 0 $((NODES-1))); do
  RESTART_COUNTS+=("0")
  NODE_LOGFILES+=("$LOGDIR/supernode_${i}_$(date +%Y%m%d_%H%M%S).log")
done

#######################################
# Utility: Kill entire process group
#######################################
kill_group() {
  local pid="$1"

  # Skip sentinel
  if [[ "$pid" == "-1" ]]; then
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "⏹  Killing process group $pid"
    kill -TERM "-$pid" 2>/dev/null || true

    for _ in {1..10}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "$pid" 2>/dev/null; then
      echo "⚠  Force killing process group $pid"
      kill -KILL "-$pid" 2>/dev/null || true
    fi

    wait "$pid" 2>/dev/null || true
  fi
}

#######################################
# Cleanup handler
#######################################
cleanup() {
  echo ""
  echo "🛑 Cleaning up Flower infrastructure..."

  for pid in "${SUPERNODE_PIDS[@]:-}"; do
    kill_group "$pid"
  done

  if [[ -n "${SUPERLINK_PID:-}" ]]; then
    kill_group "$SUPERLINK_PID"
  fi

  echo "✅ Cleanup complete"
}

trap cleanup SIGINT SIGTERM EXIT

#######################################
# Launch a single supernode by index
#######################################
launch_supernode() {
  local i="$1"
  local logfile="${NODE_LOGFILES[$i]}"

  echo "  ▶ Launching supernode $i → $logfile"

  setsid flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:$((9094 + i)) \
    --node-config "partition-id=$i num-partitions=$NODES" \
    >> "$logfile" 2>&1 &

  SUPERNODE_PIDS[$i]=$!
}

#######################################
# Launch Superlink (new process group)
#######################################
echo "🔗 Launching flower-superlink"

setsid flower-superlink --insecure \
  > "$LOGDIR/superlink.log" 2>&1 &

SUPERLINK_PID=$!

sleep 3

if ! kill -0 "$SUPERLINK_PID" 2>/dev/null; then
  echo "❌ Superlink failed to start"
  exit 1
fi

echo "✅ Superlink started (PID: $SUPERLINK_PID)"

#######################################
# Initial launch of all supernodes
#######################################
echo "🖥️  Starting $NODES Flower supernodes"

for i in $(seq 0 $((NODES-1))); do
  launch_supernode "$i"
  echo "  ✓ Supernode $i started (PID: ${SUPERNODE_PIDS[$i]})"
  sleep 2
done

echo "✅ All supernodes launched"
echo "⏳ Entering monitoring loop (max $MAX_RESTARTS restarts per node)..."

#######################################
# Monitoring loop
#######################################
ALL_DEAD=false

while true; do
  ALL_DEAD=true

  for i in $(seq 0 $((NODES-1))); do
    pid="${SUPERNODE_PIDS[$i]}"

    # Skip nodes that have been given up on
    if [[ "$pid" == "-1" ]]; then
      continue
    fi

    # ── Node is alive ─────────────────────────────────
    if kill -0 "$pid" 2>/dev/null; then
      ALL_DEAD=false
      continue
    fi

    # ── Node is dead — reap zombie immediately ────────
    wait "$pid" 2>/dev/null || true
    EXIT_CODE=$?

    echo "⚠  Supernode $i (PID $pid) exited with code $EXIT_CODE"

    count="${RESTART_COUNTS[$i]}"

    if (( count >= MAX_RESTARTS )); then
      echo "💀 Supernode $i exceeded MAX_RESTARTS ($MAX_RESTARTS). Giving up on this node."
      SUPERNODE_PIDS[$i]=-1
      continue
    fi

    RESTART_COUNTS[$i]=$(( count + 1 ))
    echo "🔄 Restarting supernode $i (attempt ${RESTART_COUNTS[$i]} / $MAX_RESTARTS) in ${RESTART_DELAY}s..."
    sleep "$RESTART_DELAY"

    launch_supernode "$i"
    ALL_DEAD=false

    echo "  ✓ Supernode $i restarted (new PID: ${SUPERNODE_PIDS[$i]})"
  done

  # ── Superlink watchdog ────────────────────────────
  if ! kill -0 "$SUPERLINK_PID" 2>/dev/null; then
    echo "💥 Superlink (PID $SUPERLINK_PID) died. Cannot continue — exiting."
    exit 1
  fi

  # ── All nodes exhausted their budgets ────────────
  if $ALL_DEAD; then
    echo "💀 All supernodes have died and exhausted restarts. Shutting down."
    exit 1
  fi

  sleep "$HEALTH_CHECK_INTERVAL"
done

# #!/usr/bin/env bash
# set -Eeuo pipefail

# #######################################
# # Config
# #######################################
# NODES=6
# LOGDIR="/workspace/flwr-enas-nofix/logs"
# mkdir -p "$LOGDIR"

# #######################################
# # Environment
# #######################################
# export RICH_DISABLE=1
# export PYTHONUNBUFFERED=1
# export TF_CPP_MIN_LOG_LEVEL=3
# export ABSL_LOGGING_MIN_LEVEL=3
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# #######################################
# # PID tracking
# #######################################
# SUPERLINK_PID=""
# declare -a SUPERNODE_PIDS=()

# #######################################
# # Utility: Kill entire process group
# #######################################
# kill_group() {
#   local pid="$1"
#   if kill -0 "$pid" 2>/dev/null; then
#     echo "⏹ Killing process group $pid"
#     kill -TERM "-$pid" 2>/dev/null || true

#     # Wait up to 10 seconds
#     for _ in {1..10}; do
#       if ! kill -0 "$pid" 2>/dev/null; then
#         break
#       fi
#       sleep 1
#     done

#     # Force kill if still alive
#     if kill -0 "$pid" 2>/dev/null; then
#       echo "⚠ Force killing process group $pid"
#       kill -KILL "-$pid" 2>/dev/null || true
#     fi

#     # Reap process
#     wait "$pid" 2>/dev/null || true
#   fi
# }

# #######################################
# # Cleanup handler
# #######################################
# cleanup() {
#   echo ""
#   echo "🛑 Cleaning up Flower infrastructure..."

#   # Stop supernodes
#   for pid in "${SUPERNODE_PIDS[@]:-}"; do
#     kill_group "$pid"
#   done

#   # Stop superlink
#   if [[ -n "${SUPERLINK_PID:-}" ]]; then
#     kill_group "$SUPERLINK_PID"
#   fi

#   echo "✅ Cleanup complete"
# }

# trap cleanup SIGINT SIGTERM EXIT

# #######################################
# # Launch Superlink (new process group)
# #######################################
# echo "🔗 Launching flower-superlink"

# setsid flower-superlink --insecure \
#   > "$LOGDIR/superlink.log" 2>&1 &

# SUPERLINK_PID=$!

# sleep 3

# if ! kill -0 "$SUPERLINK_PID" 2>/dev/null; then
#   echo "❌ Superlink failed to start"
#   exit 1
# fi

# echo "✅ Superlink started (PGID: $SUPERLINK_PID)"

# #######################################
# # Launch Supernodes
# #######################################
# echo "🖥️  Starting $NODES Flower supernodes"

# for i in $(seq 0 $((NODES-1))); do
#   LOGFILE="$LOGDIR/supernode_${i}_$(date +%Y%m%d_%H%M%S).log"

#   setsid flower-supernode \
#     --insecure \
#     --superlink 127.0.0.1:9092 \
#     --clientappio-api-address 127.0.0.1:$((9094+i)) \
#     --node-config "partition-id=$i num-partitions=$NODES" \
#     > "$LOGFILE" 2>&1 &

#   pid=$!
#   SUPERNODE_PIDS+=("$pid")

#   echo "  ✓ Supernode $i started (PGID: $pid)"
#   sleep 2
# done

# echo "✅ All supernodes launched"
# echo "⏳ Waiting for processes..."

# #######################################
# # Wait for ALL processes (no wait -n)
# #######################################
# for pid in "${SUPERNODE_PIDS[@]}"; do
#   wait "$pid" 2>/dev/null || true
# done

# wait "$SUPERLINK_PID" 2>/dev/null || true

# echo "⚠ A Flower process exited"
