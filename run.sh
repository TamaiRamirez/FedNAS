#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: ./run.sh <gpu_device> <container_name> <image_name>"
    exit 1
fi

export containerName="$2"
sleep 3 && \
        xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerName` >/dev/null 2>&1 &

if [ "$1" = "Titan" ]; then
    gpu_device='"device=0"'
elif [ "$1" = "4090" ]; then
    gpu_device='"device=1"'
elif [ "$1" = "all" ]; then
    gpu_device=all
else
    echo "Error: Invalid argument. Use 'Titan' or '4090'."
    exit 1
fi

# Print the result (or use it however you want)
echo "GPU Device set to: RTX $1"

docker run --rm -it --gpus ${gpu_device} --ipc=host -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw --network host \
    --workdir="/workspace" \
    --name $containerName --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e "TERM=xterm-256color" \
    --volume="$PWD:/workspace:rw" \
    "$3" bash