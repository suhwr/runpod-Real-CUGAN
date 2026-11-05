#!/usr/bin/env python3
"""
handler.py — RunPod Serverless Worker for Real-CUGAN
Architecture: Real Cascade U-Nets for Anime Image & Video Super-Resolution (Bilibili AI Lab)
Universal GPU Support: Turing (sm_75) -> Ampere (sm_80/86) -> Ada (sm_89) -> Hopper (sm_90) -> Blackwell (sm_120)
Features:
  - Video Super-Resolution with streaming FFmpeg pipeline (Zero intermediate disk frame caching, lossless audio sync)
  - Image Super-Resolution with RGBA alpha channel preservation
  - Scales: 2x, 3x, 4x
  - Denoise modes: conservative, no-denoise, 1x, 2x, 3x
  - Automatic dynamic tiling to prevent Out-Of-Memory (OOM) on 2K/4K resolutions
  - Multi-destination upload: Backblaze B2, Uguu.se, Tmpfiles.org, Catbox.moe, or direct Base64
"""

import os
import sys

# Ensure immediate line-buffered logging on Docker / RunPod daemon
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "0")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import io
import time
import uuid
import base64
import shutil
import tempfile
import urllib.parse
import mimetypes
import subprocess
import traceback
import requests
import cv2
import numpy as np
import torch

# GPU Performance Configuration
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        _gpu_name = torch.cuda.get_device_name(0)
        _gpu_cap = torch.cuda.get_device_capability(0)
        _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 Active GPU: {_gpu_name} (sm_{_gpu_cap[0]}{_gpu_cap[1]}, {_vram_gb:.2f} GB VRAM) — PyTorch {torch.__version__}", flush=True)
    except Exception:
        pass

import runpod
from upcunet_v3 import RealWaifuUpScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(BASE_DIR, "weights_v3"))
DEFAULT_CATBOX_USER_HASH = os.environ.get("CATBOX_USER_HASH", "")
MAX_CONCURRENCY = max(1, int(os.environ.get("MAX_CONCURRENCY", "1")))
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

# Global in-memory cache for loaded model instances
UPSCALER_CACHE = {}


# ==============================================================================
# Model Weight Resolution & Caching
# ==============================================================================
def resolve_model_path(scale: int, denoise: str) -> str:
    """Resolve the exact .pth model weight filename from scale and denoise mode."""
    scale = int(scale)
    if scale not in (2, 3, 4):
        scale = 2

    denoise = str(denoise).strip().lower()

    if scale == 2:
        if "cons" in denoise:
            fname = "up2x-latest-conservative.pth"
        elif "no" in denoise or "none" in denoise or "0" in denoise:
            fname = "up2x-latest-no-denoise.pth"
        elif "1" in denoise:
            fname = "up2x-latest-denoise1x.pth"
        elif "2" in denoise:
            fname = "up2x-latest-denoise2x.pth"
        elif "3" in denoise:
            fname = "up2x-latest-denoise3x.pth"
        else:
            fname = "up2x-latest-conservative.pth"
    elif scale == 3:
        if "cons" in denoise:
            fname = "up3x-latest-conservative.pth"
        elif "no" in denoise or "none" in denoise or "0" in denoise:
            fname = "up3x-latest-no-denoise.pth"
        else:
            fname = "up3x-latest-denoise3x.pth"
    else:  # scale == 4
        if "cons" in denoise:
            fname = "up4x-latest-conservative.pth"
        elif "no" in denoise or "none" in denoise or "0" in denoise:
            fname = "up4x-latest-no-denoise.pth"
        else:
            fname = "up4x-latest-denoise3x.pth"

    candidates = [
        os.path.join(WEIGHTS_DIR, fname),
        os.path.join("/app", "weights_v3", fname),
        os.path.join(BASE_DIR, "weights_v3", fname),
        os.path.join("/workspace", "weights_v3", fname),
        os.path.join("/runpod-volume", "weights_v3", fname),
        os.path.join(BASE_DIR, fname),
        os.path.join(os.getcwd(), "weights_v3", fname),
        os.path.join(os.getcwd(), fname),
    ]

    for p in candidates:
        if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
            return p

    raise FileNotFoundError(
        f"Real-CUGAN model weight not found for {fname}. Checked: {candidates[:4]}. Run download_models.py first."
    )


def get_upscaler(scale: int, denoise: str, half: bool = True, device: str = "cuda:0") -> RealWaifuUpScaler:
    """Retrieve or initialize a cached RealWaifuUpScaler instance."""
    if not torch.cuda.is_available():
        device = "cpu"
        half = False

    weight_path = resolve_model_path(scale, denoise)
    cache_key = (scale, weight_path, half, device)

    if cache_key in UPSCALER_CACHE:
        return UPSCALER_CACHE[cache_key]

    print(f"⚡ Loading Real-CUGAN {scale}x ({os.path.basename(weight_path)}) [half={half}, device={device}]...", flush=True)
    t0 = time.time()
    upscaler = RealWaifuUpScaler(scale=scale, weight_path=weight_path, half=half, device=device)
    print(f"✅ Loaded {os.path.basename(weight_path)} in {time.time() - t0:.2f}s", flush=True)

    UPSCALER_CACHE[cache_key] = upscaler
    return upscaler


# ==============================================================================
# Dynamic Tiling Selector (Prevents VRAM OOM on 2K / 4K)
# ==============================================================================
def determine_tile_mode(width: int, height: int, user_tile_mode=None) -> int:
    """Select the optimal tile_mode based on resolution to balance speed & VRAM."""
    if user_tile_mode is not None and str(user_tile_mode).strip().lower() not in ("auto", "none", "-1", ""):
        try:
            return int(user_tile_mode)
        except ValueError:
            pass

    pixels = width * height
    # <= 720p (e.g. 1280x720): tile_mode 0 (single pass, maximum speed)
    if pixels <= 1280 * 720:
        return 0
    # <= 1080p (e.g. 1920x1080): tile_mode 1 (halves long side, saves 40% VRAM)
    elif pixels <= 1920 * 1080:
        return 1
    # <= 1440p (2K): tile_mode 2 (halves both sides)
    elif pixels <= 2560 * 1440:
        return 2
    # 4K or above: tile_mode 3 or 4 (4x4 tiles, zero OOM risk)
    else:
        return 3


# ==============================================================================
# Cloud Upload Utilities (Backblaze B2, Uguu.se, Tmpfiles.org, Catbox.moe)
# ==============================================================================
def upload_b2(file_path: str, folder: str = "cugan") -> str:
    """Upload to Backblaze B2 (boto3 S3 or native REST)."""
    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket = os.environ.get("B2_BUCKET_NAME")
    endpoint = os.environ.get("B2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")

    if not (key_id and app_key and bucket):
        raise ValueError("Backblaze B2 credentials (B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME) not set")

    fname = os.path.basename(file_path)
    clean_folder = folder.strip("/")
    target_key = f"{clean_folder}/{int(time.time())}_{uuid.uuid4().hex[:8]}_{fname}"
    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        content_type = "video/mp4" if fname.endswith(".mp4") else "image/png"

    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket, target_key, ExtraArgs={"ContentType": content_type})

    public_prefix = os.environ.get("B2_PUBLIC_URL_PREFIX", "").rstrip("/")
    if public_prefix:
        return f"{public_prefix}/{target_key}"

    try:
        return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": target_key}, ExpiresIn=86400)
    except Exception:
        parsed = urllib.parse.urlsplit(endpoint)
        return f"{parsed.scheme}://{bucket}.{parsed.netloc}/{target_key}"


def upload_uguu(file_path: str) -> str:
    """Upload to Uguu.se (files up to 100MB)."""
    headers = {"User-Agent": BROWSER_UA}
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://uguu.se/upload",
            params={"output": "json"},
            files={"files[]": (os.path.basename(file_path), f)},
            headers=headers,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files")
        if files and len(files) > 0 and files[0].get("url"):
            return files[0]["url"]
        raise ValueError(f"Uguu response invalid: {data}")


def upload_tmpfiles(file_path: str) -> str:
    """Upload to Tmpfiles.org (files up to 10GB)."""
    headers = {"User-Agent": BROWSER_UA}
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (os.path.basename(file_path), f)},
            headers=headers,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success" and "data" in data and "url" in data["data"]:
            url = data["data"]["url"]
            return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        raise ValueError(f"Tmpfiles upload failed: {data}")


def upload_catbox(file_path: str, user_hash: str = None) -> str:
    """Upload to Catbox.moe (files up to 200MB)."""
    data = {"reqtype": "fileupload"}
    h = user_hash or DEFAULT_CATBOX_USER_HASH
    if h:
        data["userhash"] = h

    headers = {"User-Agent": BROWSER_UA}
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data=data,
            files={"fileToUpload": (os.path.basename(file_path), f)},
            headers=headers,
            timeout=180,
        )
        resp.raise_for_status()
        url = resp.text.strip()
        if url.startswith("http"):
            return url
        raise ValueError(f"Catbox upload failed: {resp.text}")


def upload_file_auto(file_path: str, destination: str = "auto") -> str:
    """Upload output file to requested destination or auto-fallback chain."""
    dest = (destination or "auto").strip().lower()

    if dest == "b2":
        return upload_b2(file_path)

    # If Uguu requested specifically
    if dest == "uguu":
        try:
            return upload_uguu(file_path)
        except Exception as e:
            print(f"⚠️ Uguu upload failed ({e}), trying tmpfiles fallback...", flush=True)
            try:
                return upload_tmpfiles(file_path)
            except Exception:
                return upload_catbox(file_path)

    # If Catbox requested specifically
    if dest == "catbox":
        try:
            return upload_catbox(file_path)
        except Exception as e:
            print(f"⚠️ Catbox upload failed ({e}), falling back to Uguu...", flush=True)
            try:
                return upload_uguu(file_path)
            except Exception:
                return upload_tmpfiles(file_path)

    # If Tmpfiles requested specifically
    if dest in ("tmpfiles", "tmpfile"):
        try:
            return upload_tmpfiles(file_path)
        except Exception as e:
            print(f"⚠️ Tmpfiles upload failed ({e}), trying Uguu fallback...", flush=True)
            return upload_uguu(file_path)

    # Auto mode: try B2 if configured, then Uguu, then tmpfiles, then Catbox
    if os.environ.get("B2_KEY_ID") and os.environ.get("B2_BUCKET_NAME"):
        try:
            return upload_b2(file_path)
        except Exception as e:
            print(f"⚠️ B2 upload failed ({e}), falling back to Uguu...", flush=True)

    # Try Uguu first
    try:
        return upload_uguu(file_path)
    except Exception as e1:
        print(f"⚠️ Uguu upload failed ({e1}), trying tmpfiles...", flush=True)

    # Try tmpfiles second
    try:
        return upload_tmpfiles(file_path)
    except Exception as e2:
        print(f"⚠️ Tmpfiles upload failed ({e2}), trying Catbox...", flush=True)

    # Try Catbox third
    try:
        return upload_catbox(file_path)
    except Exception as e3:
        raise RuntimeError(f"All cloud upload providers failed: Uguu ({e1}), Tmpfiles ({e2}), Catbox ({e3})")


# ==============================================================================
# Ingestion: Download or Decode Input Media
# ==============================================================================
def download_or_decode_media(raw_input: str, temp_dir: str) -> tuple[str, str]:
    """
    Downloads or decodes media from URL, base64 data URI, or raw base64.
    Returns (local_file_path, detected_type: 'video' | 'image').
    """
    raw = raw_input.strip()

    # 1. Base64 Data URI (e.g. data:image/png;base64,... or data:video/mp4;base64,...)
    if raw.startswith("data:"):
        header, b64data = raw.split(",", 1)
        media_type = "video" if "video" in header else "image"
        ext = ".mp4" if media_type == "video" else ".png"
        out_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex[:8]}{ext}")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64data))
        return out_path, media_type

    # 2. HTTP/HTTPS URL
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        path_lower = parsed.path.lower()

        video_exts = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts")
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

        if any(path_lower.endswith(e) for e in video_exts):
            media_type = "video"
            ext = ".mp4"
        elif any(path_lower.endswith(e) for e in image_exts):
            media_type = "image"
            ext = ".png"
        else:
            media_type = "unknown"
            ext = ".tmp"

        out_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex[:8]}{ext}")
        print(f"⬇️ Downloading input from {raw}...", flush=True)
        headers = {"User-Agent": BROWSER_UA}
        resp = requests.get(raw, stream=True, timeout=(15, 300), headers=headers)
        resp.raise_for_status()

        # Check content-type header if extension was ambiguous
        ctype = resp.headers.get("content-type", "").lower()
        if media_type == "unknown":
            if "video" in ctype:
                media_type = "video"
                ext = ".mp4"
            else:
                media_type = "image"
                ext = ".png"
            new_path = os.path.splitext(out_path)[0] + ext
            out_path = new_path

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return out_path, media_type

    # 3. Raw Base64 string
    decoded = base64.b64decode(raw)
    if decoded.startswith(b"\x89PNG") or decoded.startswith(b"\xff\xd8\xff") or b"WEBP" in decoded[:16]:
        media_type = "image"
        ext = ".png"
    elif b"ftyp" in decoded[:32] or decoded.startswith(b"\x1a\x45\xdf\xa3"):
        media_type = "video"
        ext = ".mp4"
    else:
        media_type = "image"
        ext = ".png"

    out_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex[:8]}{ext}")
    with open(out_path, "wb") as f:
        f.write(decoded)

    return out_path, media_type


# ==============================================================================
# Image Super-Resolution Pipeline
# ==============================================================================
def process_image(
    input_path: str,
    output_path: str,
    upscaler: RealWaifuUpScaler,
    tile_mode: int,
    cache_mode: int = 0,
    alpha: float = 1.0,
) -> dict:
    """Upscale an image with Real-CUGAN, handling RGB and RGBA transparency."""
    t0 = time.time()
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image from {input_path}")

    h, w = img.shape[:2]
    has_alpha = img.ndim == 3 and img.shape[2] == 4

    if has_alpha:
        bgr = img[:, :, :3]
        alpha_channel = img[:, :, 3]
    elif img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha_channel = None
    else:
        bgr = img
        alpha_channel = None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    effective_tile = determine_tile_mode(w, h, tile_mode)
    upscaled_rgb = upscaler(rgb, tile_mode=effective_tile, cache_mode=cache_mode, alpha=alpha)
    upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)

    out_h, out_w = upscaled_bgr.shape[:2]

    if has_alpha:
        upscaled_alpha = cv2.resize(alpha_channel, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        final_img = cv2.merge([upscaled_bgr[:, :, 0], upscaled_bgr[:, :, 1], upscaled_bgr[:, :, 2], upscaled_alpha])
    else:
        final_img = upscaled_bgr

    cv2.imwrite(output_path, final_img)
    duration = time.time() - t0

    return {
        "input_resolution": f"{w}x{h}",
        "output_resolution": f"{out_w}x{out_h}",
        "execution_time_seconds": round(duration, 3),
        "tile_mode_used": effective_tile,
    }


# ==============================================================================
# Video Super-Resolution Pipeline (Streaming FFmpeg Pipe)
# ==============================================================================
def process_video(
    input_path: str,
    output_path: str,
    upscaler: RealWaifuUpScaler,
    scale: int,
    tile_mode: int,
    cache_mode: int = 0,
    alpha: float = 1.0,
    crf: int = 18,
    custom_fps: float = None,
    job_id: str = "cugan-video",
) -> dict:
    """
    High-performance streaming video super-resolution pipeline.
    Streams frames from GPU directly into FFmpeg stdin pipe (zero disk frame caching).
    Preserves audio streams losslessly with synchronized PTS/DTS timestamps.
    """
    t0 = time.time()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or np.isnan(fps):
        fps = 24.0
    if custom_fps and custom_fps > 0:
        fps = float(custom_fps)

    # Force even dimensions for H.264 yuv420p encoder compatibility
    out_w = ((in_w * scale) // 2) * 2
    out_h = ((in_h * scale) // 2) * 2

    effective_tile = determine_tile_mode(in_w, in_h, tile_mode)

    print(f"[{job_id}] Video Info: {in_w}x{in_h} @ {fps:.2f}fps ({total_frames} frames) -> Target: {out_w}x{out_h} (scale={scale}x, tile={effective_tile})", flush=True)

    # 1. Extract audio losslessly to temporary file if audio stream exists
    temp_audio = None
    has_audio = False
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ]
        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if probe_res.stdout.strip():
            temp_audio = os.path.join(os.path.dirname(output_path), f"audio_{uuid.uuid4().hex[:6]}.m4a")
            extract_cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-c:a", "copy", temp_audio]
            ext_res = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ext_res.returncode == 0 and os.path.isfile(temp_audio) and os.path.getsize(temp_audio) > 0:
                has_audio = True
            else:
                # Fallback to AAC transcode
                extract_cmd_aac = ["ffmpeg", "-y", "-i", input_path, "-vn", "-c:a", "aac", "-b:a", "256k", temp_audio]
                ext_aac = subprocess.run(extract_cmd_aac, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if ext_aac.returncode == 0 and os.path.isfile(temp_audio) and os.path.getsize(temp_audio) > 0:
                    has_audio = True
    except Exception as e:
        print(f"[{job_id}] Audio extraction check failed ({e}), proceeding without audio...", flush=True)

    # 2. Build FFmpeg command to read raw BGR frames from stdin
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",  # Read from stdin
    ]

    if has_audio and temp_audio and os.path.isfile(temp_audio):
        ffmpeg_cmd.extend([
            "-i", temp_audio,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:a", "copy",
            "-shortest",
        ])
    else:
        ffmpeg_cmd.extend([
            "-map", "0:v:0",
        ])

    ffmpeg_cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", str(crf),
        "-movflags", "+faststart",
        output_path,
    ])

    print(f"[{job_id}] Launching FFmpeg encoder process...", flush=True)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # 3. Stream frames through Real-CUGAN into FFmpeg pipe
    frames_processed = 0
    t_loop = time.time()

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            upscaled_rgb = upscaler(rgb_frame, tile_mode=effective_tile, cache_mode=cache_mode, alpha=alpha)
            upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)

            # Ensure exact target dimensions match ffmpeg -s {out_w}x{out_h}
            if upscaled_bgr.shape[0] != out_h or upscaled_bgr.shape[1] != out_w:
                upscaled_bgr = cv2.resize(upscaled_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            # Write raw frame buffer directly to FFmpeg stdin
            try:
                proc.stdin.write(upscaled_bgr.tobytes())
            except (BrokenPipeError, IOError):
                print(f"[{job_id}] FFmpeg pipe closed prematurely.", flush=True)
                break

            frames_processed += 1

            if frames_processed % 30 == 0 or frames_processed == total_frames:
                elapsed = time.time() - t_loop
                fps_rate = frames_processed / max(0.001, elapsed)
                pct = (frames_processed / total_frames * 100) if total_frames > 0 else 0
                print(f"[{job_id}] Processed {frames_processed}/{total_frames} frames ({pct:.1f}%) — {fps_rate:.1f} FPS", flush=True)

    finally:
        cap.release()
        stderr_text = ""
        try:
            # DO NOT call proc.stdin.close() manually before communicate()!
            # In Python 3.11+, communicate() automatically flushes and closes stdin cleanly.
            _, stderr_bytes = proc.communicate(timeout=180)
            if stderr_bytes:
                stderr_text = stderr_bytes.decode("utf-8", errors="ignore")
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_bytes = proc.communicate()
            stderr_text = stderr_bytes.decode("utf-8", errors="ignore") if stderr_bytes else "FFmpeg timed out"
        except Exception as comm_err:
            try:
                proc.kill()
            except Exception:
                pass
            stderr_text = str(comm_err)

        if proc.returncode != 0:
            print(f"[{job_id}] FFmpeg error (exit code {proc.returncode}): {stderr_text[-500:]}", flush=True)
            raise RuntimeError(f"FFmpeg encoding failed with exit code {proc.returncode}: {stderr_text[-300:]}")

        if temp_audio and os.path.isfile(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass

    total_time = time.time() - t0
    avg_fps = frames_processed / max(0.001, total_time)

    return {
        "input_resolution": f"{in_w}x{in_h}",
        "output_resolution": f"{out_w}x{out_h}",
        "frames_processed": frames_processed,
        "fps_processing": round(avg_fps, 2),
        "video_fps": round(fps, 2),
        "execution_time_seconds": round(total_time, 2),
        "tile_mode_used": effective_tile,
    }


# ==============================================================================
# Main RunPod Serverless Handler
# ==============================================================================
def handler(job: dict) -> dict:
    job_id = job.get("id", str(uuid.uuid4()))
    job_input = job.get("input", {})

    print(f"\n⚡ [{job_id}] Received job request", flush=True)

    # 1. Parse Input Parameters
    raw_media = (
        job_input.get("video")
        or job_input.get("image")
        or job_input.get("url")
        or job_input.get("input")
        or job_input.get("media")
    )
    if not raw_media:
        return {"error": "Missing input: provide 'video', 'image', or 'url' in job input payload."}

    scale = int(job_input.get("scale", 2))
    if scale not in (2, 3, 4):
        scale = 2

    denoise = str(job_input.get("denoise", "conservative")).strip().lower()
    tile_mode = job_input.get("tile_mode", "auto")
    cache_mode = int(job_input.get("cache_mode", 0))
    alpha = float(job_input.get("alpha", 1.0))
    crf = int(job_input.get("crf", 18))
    custom_fps = job_input.get("fps", None)
    half = bool(job_input.get("half", True))
    upload_dest = str(job_input.get("upload_destination", "auto")).strip().lower()

    # 2. Setup Temporary Workspace
    temp_dir = tempfile.mkdtemp(prefix=f"cugan_{job_id}_")

    try:
        # 3. Ingest Media
        input_file, detected_type = download_or_decode_media(raw_media, temp_dir)
        print(f"[{job_id}] Input media detected as '{detected_type}' -> {input_file}", flush=True)

        # 4. Initialize Upscaler
        upscaler = get_upscaler(scale=scale, denoise=denoise, half=half)

        # 5. Process Image or Video
        if detected_type == "video":
            output_file = os.path.join(temp_dir, f"upscaled_{scale}x_{job_id[:8]}.mp4")
            stats = process_video(
                input_path=input_file,
                output_path=output_file,
                upscaler=upscaler,
                scale=scale,
                tile_mode=tile_mode,
                cache_mode=cache_mode,
                alpha=alpha,
                crf=crf,
                custom_fps=custom_fps,
                job_id=job_id,
            )
            media_type = "video"
        else:
            output_file = os.path.join(temp_dir, f"upscaled_{scale}x_{job_id[:8]}.png")
            stats = process_image(
                input_path=input_file,
                output_path=output_file,
                upscaler=upscaler,
                tile_mode=tile_mode,
                cache_mode=cache_mode,
                alpha=alpha,
            )
            media_type = "image"

        if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
            return {"error": "Processing completed but output file is missing or empty."}

        size_bytes = os.path.getsize(output_file)
        file_name = os.path.basename(output_file)

        # 6. Prepare Output (Cloud URL or Base64)
        output_response = {
            "status": "COMPLETED",
            "media_type": media_type,
            "scale": scale,
            "denoise": denoise,
            "file_name": file_name,
            "size_bytes": size_bytes,
            **stats,
        }

        # If user explicitly requested base64
        if upload_dest == "base64":
            with open(output_file, "rb") as f:
                output_response["base64"] = base64.b64encode(f.read()).decode("utf-8")
        else:
            # Upload to cloud
            print(f"[{job_id}] Uploading output ({size_bytes / (1024*1024):.2f} MB) with dest '{upload_dest}'...", flush=True)
            upload_url = upload_file_auto(output_file, destination=upload_dest)
            output_response["url"] = upload_url
            print(f"[{job_id}] Uploaded successfully: {upload_url}", flush=True)

            # For images under 5MB, also attach base64 for zero-download instant display
            if media_type == "image" and size_bytes < 5 * 1024 * 1024:
                with open(output_file, "rb") as f:
                    output_response["base64"] = base64.b64encode(f.read()).decode("utf-8")

        return output_response

    except Exception as e:
        print(f"❌ [{job_id}] Error: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("🚀 Starting Real-CUGAN Serverless Worker...", flush=True)
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda _: MAX_CONCURRENCY,
        }
    )
