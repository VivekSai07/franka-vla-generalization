# Base: CUDA 12.8 + cuDNN on Ubuntu 22.04
# RTX 5090 is Blackwell (sm_120). PyTorch stable (2.6) only compiled for up to sm_90.
# PyTorch nightly cu128 includes sm_120 kernels — required for proper Blackwell performance.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # MuJoCo / rendering
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    libosmesa6 \
    libosmesa6-dev \
    # Video encoding (OpenCV + lerobot video ops)
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Pinocchio / pin build deps
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# PyTorch nightly for CUDA 12.8 — nightly is required for RTX 5090 (Blackwell sm_120).
# Stable PyTorch 2.6 only includes kernels up to sm_90 (H100); without nightly,
# Blackwell falls back to slow PTX JIT compilation for every operation.
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# LeRobot + physics / robotics stack
# Install pin (real pinocchio) BEFORE lerobot so lerobot doesn't pull in the fake stub
RUN pip install --no-cache-dir pin

# Install lerobot with the smolvla extra so pip resolves the correct
# transformers version (>=4.57.1,<5.0.0). Installing bare lerobot then
# adding transformers separately risks pulling in transformers 5.x which breaks groot/smolvla.
RUN pip install --no-cache-dir "lerobot[smolvla]==0.4.4"

RUN pip install --no-cache-dir accelerate

RUN pip install --no-cache-dir mujoco

# Inference server + client dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    requests \
    opencv-python-headless \
    Pillow \
    matplotlib \
    PyOpenGL \
    PyOpenGL_accelerate

# HuggingFace + experiment tracking
RUN pip install --no-cache-dir \
    huggingface_hub \
    wandb

# Create a user matching the host UID/GID so bind-mounted files are owned
# correctly and the shell prompt shows a real name instead of "I have no name!".
RUN groupadd -g 1001 vivek && useradd -u 1001 -g 1001 -m -s /bin/bash vivek

WORKDIR /vla
