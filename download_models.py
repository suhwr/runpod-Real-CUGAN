#!/usr/bin/env python3
"""
download_models.py — Real-CUGAN v3 Model Weights Downloader
Downloads the official 11 Real-CUGAN v3 model weights (2x, 3x, 4x with denoise levels).
"""

import os
import sys
import zipfile
import urllib.request
import shutil

WEIGHTS_URL = "https://github.com/ZHCSOFT/Real-CUGAN/releases/download/Real-CUGAN/weights_v3.zip"
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


def download_progress_hook(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r⬇️  Downloading weights_v3.zip: {percent}% ({count * block_size / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB)")
        sys.stdout.flush()


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Check if all models already exist
    all_exist = all(os.path.isfile(os.path.join(TARGET_DIR, m)) for m in REQUIRED_MODELS)
    if all_exist:
        print(f"✅ All {len(REQUIRED_MODELS)} Real-CUGAN v3 weights already exist in {TARGET_DIR}.")
        return

    zip_tmp = os.path.join(TARGET_DIR, "weights_v3_tmp.zip")
    print(f"📦 Downloading Real-CUGAN v3 weights from {WEIGHTS_URL}...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, zip_tmp, reporthook=download_progress_hook)
        print("\n📦 Extracting weights...")
        with zipfile.ZipFile(zip_tmp, 'r') as zf:
            zf.extractall(TARGET_DIR)
        print(f"✅ Successfully extracted all models into {TARGET_DIR}!")
    finally:
        if os.path.exists(zip_tmp):
            os.remove(zip_tmp)

    # Verify extracted models
    missing = [m for m in REQUIRED_MODELS if not os.path.isfile(os.path.join(TARGET_DIR, m))]
    if missing:
        print(f"⚠️ Warning: Missing expected models: {missing}", file=sys.stderr)
    else:
        print(f"🎉 Real-CUGAN v3 weights verified ({len(REQUIRED_MODELS)} models ready)!")


if __name__ == "__main__":
    main()
