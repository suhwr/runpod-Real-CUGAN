#!/usr/bin/env python3
"""
download_models.py — Real-CUGAN v3 Model Weights Downloader & Verifier
Ensures all 11 official Real-CUGAN v3 model weights are present.
If weights are already included locally, verifies them instantly (0s).
If missing, downloads with retry, timeout, and custom User-Agent.
"""

import os
import sys
import zipfile
import subprocess
import shutil

TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_v3")

REQUIRED_MODELS = [
    "up2x-latest-conservative.pth",
    "up2x-latest-no-denoise.pth",
    "up2x-latest-denoise1x.pth",
    "up2x-latest-denoise2x.pth",
    "up2x-latest-denoise3x.pth",
    "up3x-latest-conservative.pth",
    "up3x-latest-no-denoise.pth",
    "up3x-latest-denoise3x.pth",
    "up4x-latest-conservative.pth",
    "up4x-latest-no-denoise.pth",
    "up4x-latest-denoise3x.pth",
]

MIRROR_URLS = [
    "https://github.com/ZHCSOFT/Real-CUGAN/releases/download/Real-CUGAN/weights_v3.zip",
    "https://github.com/suhwr/runpod-Real-CUGAN/releases/download/v1.0.0/weights_v3.zip",
]


def download_with_curl(url: str, output_path: str) -> bool:
    """Download file using curl with timeouts and retries."""
    cmd = [
        "curl", "-sSL",
        "--retry", "5",
        "--retry-delay", "3",
        "--connect-timeout", "20",
        "--max-time", "300",
        "-A", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-o", output_path,
        url,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 10 * 1024 * 1024
    except Exception as e:
        print(f"⚠️ curl error: {e}", file=sys.stderr)
        return False


def download_with_python(url: str, output_path: str) -> bool:
    """Fallback download using python requests with stream & timeout."""
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0 Safari/537.36"}
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 10 * 1024 * 1024
    except Exception as e:
        print(f"⚠️ Python requests error: {e}", file=sys.stderr)
        return False


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. Fast path: check if all 11 models already exist and are non-empty
    existing = [m for m in REQUIRED_MODELS if os.path.isfile(os.path.join(TARGET_DIR, m)) and os.path.getsize(os.path.join(TARGET_DIR, m)) > 1024 * 1024]
    if len(existing) == len(REQUIRED_MODELS):
        print(f"✅ All {len(REQUIRED_MODELS)} Real-CUGAN v3 weights already exist in {TARGET_DIR}. Instant verification complete!")
        return

    print(f"📦 Missing {len(REQUIRED_MODELS) - len(existing)} models. Downloading weights_v3...")

    zip_tmp = os.path.join(TARGET_DIR, "weights_v3_tmp.zip")
    success = False

    for url in MIRROR_URLS:
        print(f"🌐 Trying mirror: {url}...")
        # Try curl first (installed in image), then python requests
        if shutil.which("curl"):
            if download_with_curl(url, zip_tmp):
                success = True
                break
        if download_with_python(url, zip_tmp):
            success = True
            break

    if not success or not os.path.isfile(zip_tmp):
        print("❌ Failed to download weights from all mirrors!", file=sys.stderr)
        sys.exit(1)

    print("📦 Extracting weights archive...")
    try:
        with zipfile.ZipFile(zip_tmp, 'r') as zf:
            zf.extractall(TARGET_DIR)
        print(f"✅ Successfully extracted models into {TARGET_DIR}!")
    finally:
        if os.path.exists(zip_tmp):
            os.remove(zip_tmp)

    # Final verification
    missing = [m for m in REQUIRED_MODELS if not os.path.isfile(os.path.join(TARGET_DIR, m))]
    if missing:
        print(f"❌ Error: Still missing models: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"🎉 Real-CUGAN v3 weights verified successfully ({len(REQUIRED_MODELS)} models ready)!")


if __name__ == "__main__":
    main()
