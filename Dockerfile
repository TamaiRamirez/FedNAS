# Include a base image
FROM ubuntu:22.04
# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)

ARG USERNAME
ARG USER_UID
ARG USER_GID=$USER_UID

ARG DEBIAN_FRONTEND=noninteractive

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
USER $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git, vim, and nano
RUN sudo apt install -y git vim nano

RUN sudo apt install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libprotobuf-dev \
    protobuf-compiler \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx

# RUN sudo apt-get install -y nvidia-cuda-toolkit

RUN pip install -U tensorflow[and-cuda]
RUN pip install -U absl-py
RUN pip install -U tensorboard
RUN pip install -U scikit-learn
RUN pip install -U np_utils
RUN pip install -U matplotlib
RUN pip install -U pandas
RUN pip install -U opencv-python
RUN pip install -U flwr[simulation]
RUN pip install -U flwr-datasets[vision]
RUN pip install -U hydra-core
RUN pip install -U memory_profiler

USER $USERNAME


