#!/usr/bin/env bash
set -e

echo "================================================================"
echo "🚀 Starting Real-CUGAN Serverless Worker"
echo "================================================================"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=0
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Check GPU capability and log hardware
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'🎯 Hardware: {name} (sm_{cap[0]}{cap[1]}, {vram:.2f} GB VRAM) — PyTorch {torch.__version__}', flush=True)
else:
    print('⚠️ No CUDA GPU detected, running in CPU fallback mode', flush=True)
" 2>/dev/null || true

# Preload jemalloc to prevent memory fragmentation on long video upscaling
if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD}
fi

# Support persistent network volume if mounted and populated with weights
for VOL in "/runpod-volume" "/workspace" "/volume"; do
    if [ -f "$VOL/weights_v3/up2x-latest-conservative.pth" ]; then
        echo "⚡ Found external weights in $VOL/weights_v3, using volume models..."
        export WEIGHTS_DIR="$VOL/weights_v3"
        break
    fi
done

cd /app
exec python3 -u handler.py
