# ==============================================================================
# Ultra-Optimized Real-CUGAN Serverless Worker
# Architecture: Real Cascade U-Nets for Anime Image & Video Super-Resolution (Bilibili AI Lab)
# Universal GPU Support: Turing (sm_75) -> Ampere (sm_80/86) -> Ada (sm_89) -> Hopper (sm_90) -> Blackwell (sm_120)
# ==============================================================================
ARG CUDA_VERSION="12.8.0"
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# 1. Install system tools, ffmpeg, build packages, jemalloc, and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libjemalloc2 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# 2. Install PyTorch with CUDA 12.8 explicitly (Universal Blackwell sm_120 support)
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch torchvision torchaudio

# 3. Install worker dependencies
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /app/requirements.txt

# 4. Pre-download Real-CUGAN v3 model weights into image layer
# Baking models directly eliminates runtime downloads (Zero Cold-Start!)
COPY download_models.py /app/download_models.py
RUN python3 /app/download_models.py

# 5. Copy handler, model architecture, and documentation
COPY upcunet_v3.py /app/upcunet_v3.py
COPY handler.py /app/handler.py
COPY README.md /app/README.md

RUN python3 -m compileall -q /app /usr/local/lib/python3.11/dist-packages || true

CMD ["python3", "-u", "/app/handler.py"]
