#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: ./run.sh <gpu_device> <container_name> <image_name>"
    exit 1
fi

export containerName="$2"
sleep 3 && \
        xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerName` >/dev/null 2>&1 &

if [ "$1" = "0" ]; then
    gpu_device="device=0"
    cuda_device="0"
elif [ "$1" = "1" ]; then
    gpu_device="device=1"
    cuda_device="1"
elif [ "$1" = "2" ]; then
    gpu_device="device=2"
    cuda_device="2"
elif [ "$1" = "3" ]; then
    gpu_device="device=3"
    cuda_device="3"
elif [ "$1" = "all" ]; then
    gpu_device="all"
    cuda_device="0,1,2,3"
else
    echo "Error: Invalid argument. Use 0,1,2,3, or all."
    exit 1
fi

# Print the result (or use it however you want)
echo "GPU Device set to: RTX $1"

docker run --rm -it \
    --gpus ${gpu_device} \
    -e CUDA_VISIBLE_DEVICES=${cuda_device} \
    --ipc=host -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network host \
    --workdir="/workspace" \
    --name $containerName \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e TERM=xterm-256color \
    --volume="$PWD:/workspace:rw" \
    "$3" bash
