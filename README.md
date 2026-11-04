# Real-CUGAN RunPod Serverless Worker 🚀

An ultra-optimized, production-ready **RunPod Serverless Worker** for **Real-CUGAN** (Real Cascade U-Nets for Anime Image & Video Super-Resolution, developed by Bilibili AI Lab).

Optimized for **blazing-fast video upscaling** (30–60 FPS on modern GPUs) and **high-fidelity image upscaling** with universal architecture compatibility—from NVIDIA Turing to modern Blackwell (`sm_120`).

---

## 🌟 Key Highlights

- **⚡ Streaming Video Super-Resolution:** Streams raw frames directly from GPU into an `ffmpeg` encoding pipeline (`stdin` pipe). **Zero intermediate frame caching to disk**, eliminating disk space limits and boosting throughput.
- **🎵 Lossless Audio Synchronization:** Automatically extracts and remuxes original audio streams with synchronized PTS/DTS timestamps.
- **🎨 Image Super-Resolution:** Upscales anime illustrations with full RGBA alpha transparency support.
- **🛡️ Dynamic Auto-Tiling:** Automatically calculates memory-safe tiles based on resolution (720p, 1080p, 2K, 4K) to eliminate Out-Of-Memory (OOM) errors.
- **🔥 Zero Cold-Start:** All 11 official Real-CUGAN v3 model weights (`2x`, `3x`, `4x`) are pre-baked directly into the container image (~58 MB total).
- **🌍 Universal Blackwell & CUDA 12.8 Support:** PyTorch CUDA 12.8 with universal PTX architecture targeting:
  - `sm_75` (Turing: T4, RTX 2080)
  - `sm_80`, `sm_86` (Ampere: A100, RTX 3090, A5000)
  - `sm_89` (Ada Lovelace: RTX 4090, L4)
  - `sm_90` (Hopper: H100)
  - `sm_100`, `sm_120` (Blackwell: RTX 6000 Ada, B200)
- **☁️ Multi-Destination Cloud Uploads:** Automatically uploads output media to **Catbox.moe**, **Backblaze B2**, **Uguu.se**, or returns direct **Base64**.

---

## 📦 Model Scales & Denoise Modes

Real-CUGAN v3 offers fine-grained models suited for different levels of compression and style preservation:

| Scale | Available Denoise Modes | Best Used For |
| :---: | :--- | :--- |
| **2x** | `conservative`, `no-denoise`, `1x`, `2x`, `3x` | General anime clips, maintaining original line weight (`conservative`) or heavy compression cleanup (`3x`). |
| **3x** | `conservative`, `no-denoise`, `3x` | 480p/720p to 1440p upscaling. |
| **4x** | `conservative`, `no-denoise`, `3x` | 720p/1080p to 4K Ultra HD. |

---

## 📡 API Reference

### Input Payload (`job["input"]`)

```json
{
  "input": {
    "video": "https://example.com/anime_clip.mp4",
    "scale": 2,
    "denoise": "conservative",
    "tile_mode": "auto",
    "crf": 18,
    "upload_destination": "catbox"
  }
}
```

#### Parameters:

| Field | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `video` | string | `null` | Video URL, base64 data URI (`data:video/mp4;base64,...`), or raw base64. |
| `image` | string | `null` | Image URL, base64 data URI (`data:image/png;base64,...`), or raw base64. |
| `scale` | integer | `2` | Upscale factor: `2`, `3`, or `4`. |
| `denoise` | string | `"conservative"` | Denoise mode: `"conservative"`, `"no-denoise"`, `"1x"`, `"2x"`, or `"3x"`. |
| `tile_mode` | string / int | `"auto"` | Tiling mode: `"auto"`, `0` (no tiling), `1` (half long edge), `2`, `3` (grid tiles). |
| `crf` | integer | `18` | H.264 video compression quality (lower = higher quality, 16–23 recommended). |
| `fps` | float | `0` | Override output video FPS (`0` or omitted = keep original video FPS). |
| `cache_mode` | integer | `0` | CUGAN cache mode: `0` (full cache), `1` (8-bit quantized), `2` (fast rough). |
| `alpha` | float | `1.0` | Feature blend weight factor (0.0 – 1.0). |
| `upload_destination` | string | `"auto"` | Destination: `"catbox"`, `"b2"`, `"uguu"`, `"base64"`, or `"auto"`. |

---

### Output Response (`job["output"]`)

#### Video Output Example:
```json
{
  "status": "COMPLETED",
  "media_type": "video",
  "scale": 2,
  "denoise": "conservative",
  "file_name": "upscaled_2x_a1b2c3d4.mp4",
  "size_bytes": 14205812,
  "url": "https://files.catbox.moe/example.mp4",
  "input_resolution": "1280x720",
  "output_resolution": "2560x1440",
  "frames_processed": 300,
  "video_fps": 29.97,
  "fps_processing": 34.5,
  "execution_time_seconds": 8.7,
  "tile_mode_used": 1
}
```

#### Image Output Example:
```json
{
  "status": "COMPLETED",
  "media_type": "image",
  "scale": 4,
  "denoise": "3x",
  "file_name": "upscaled_4x_e5f6g7h8.png",
  "size_bytes": 4120930,
  "url": "https://files.catbox.moe/example.png",
  "base64": "iVBORw0KGgo...",
  "input_resolution": "512x512",
  "output_resolution": "2048x2048",
  "execution_time_seconds": 0.42,
  "tile_mode_used": 0
}
```

---

## 🛠️ Local Development & Testing

### 1. Download Model Weights
```bash
python3 download_models.py
```

### 2. Run Local Test
```bash
# Test Image Upscaling locally
python3 test_endpoint.py --local --image ./sample.png --scale 2

# Test Video Upscaling locally
python3 test_endpoint.py --local --video ./sample.mp4 --scale 2 --denoise conservative
```

---

## 🐳 Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t your-dockerhub-username/runpod-real-cugan:latest .
```

### 2. Push to Docker Hub / Container Registry
```bash
docker push your-dockerhub-username/runpod-real-cugan:latest
```

### 3. Deploy on RunPod Serverless
1. Open the [RunPod Serverless Console](https://www.runpod.io/console/serverless).
2. Create a new **Template**:
   - **Container Image:** `your-dockerhub-username/runpod-real-cugan:latest`
   - **Container Disk:** `20 GB`
3. Create an **Endpoint**:
   - Select recommended GPU types: **RTX 4090**, **RTX 3090**, **A5000**, or **L4**.
   - Set **Max Execution Timeout**: `600s`.
   - Set **Idle Timeout**: `5s`.

---

## ⚙️ Environment Variables (Optional)

| Variable | Description |
| :--- | :--- |
| `MAX_CONCURRENCY` | Maximum concurrent jobs per worker instance (default: `1`). |
| `CATBOX_USER_HASH` | Optional Catbox.moe user hash for managing uploads. |
| `B2_KEY_ID` | Backblaze B2 Application Key ID. |
| `B2_APPLICATION_KEY` | Backblaze B2 Application Key. |
| `B2_BUCKET_NAME` | Backblaze B2 Bucket Name. |
| `B2_ENDPOINT_URL` | Backblaze B2 S3-compatible endpoint URL (e.g. `https://s3.us-west-004.backblazeb2.com`). |
| `B2_PUBLIC_URL_PREFIX` | Optional custom CDN / public URL prefix for B2 bucket. |

---

## 📄 Credits & Acknowledgements

- **Real-CUGAN:** Developed and open-sourced by **Bilibili AI Lab** ([bilibili/ailab](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)).
- **RunPod:** Serverless GPU infrastructure ([runpod.io](https://runpod.io)).
