#!/bin/bash

# This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirez)

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: ./run.sh <gpu_device> <container_name> <image_name>"
    exit 1
fi

export containerName="$2"

#######################################
# GPU selection
#######################################

case "$1" in
    0)
        gpu_device='"device=0"'
        ;;
    1)
        gpu_device='"device=1"'
        ;;
    all)
        gpu_device=all
        ;;
    *)
        echo "Error: Invalid GPU. Use '0', '1', or 'all'."
        exit 1
        ;;
esac

#######################################
# Project configuration
#######################################

PROJECT_DIR="/workspace/flwr-enas-nofix"

case "$containerName" in
    experiments)
        PROJECT_TOML="$PWD/pyproject_experiments.toml"
        ;;
    ablations)
        PROJECT_TOML="$PWD/pyproject_ablations.toml"
        ;;
    *)
        echo "Error: Invalid container name."
        echo "Use 'experiments' or 'ablations'."
        exit 1
        ;;
esac

if [[ ! -f "$PROJECT_TOML" ]]; then
    echo "Error: Configuration file not found: $PROJECT_TOML"
    exit 1
fi

#######################################
# Container information
#######################################

echo "Container: $containerName"
echo "GPU device: $1"
echo "Configuration: $PROJECT_TOML"

sleep 3 && \
    xhost +local:$(docker inspect --format='{{ .Config.Hostname }}' "$containerName") \
    >/dev/null 2>&1 &

#######################################
# Run Docker container
#######################################

docker run --rm -it \
    --gpus "$gpu_device" \
    --init \
    --sig-proxy=true \
    --ipc=private \
    --shm-size=128g \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network bridge \
    --workdir="/workspace" \
    --name "$containerName" \
    --runtime=nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e "TERM=xterm-256color" \
    --volume="$PWD:/workspace:rw" \
    --volume="$PROJECT_TOML:$PROJECT_DIR/pyproject.toml:rw" \
    "$3" bash