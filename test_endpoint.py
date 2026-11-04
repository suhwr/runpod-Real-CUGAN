#!/usr/bin/env python3
"""
test_endpoint.py — Test Real-CUGAN RunPod Serverless Endpoint or Local Handler
Usage:
  1. Local test (no RunPod endpoint needed):
     python3 test_endpoint.py --local --image input.png
     python3 test_endpoint.py --local --video input.mp4

  2. Cloud endpoint test:
     export RUNPOD_API_KEY="your_api_key"
     export RUNPOD_ENDPOINT_ID="your_endpoint_id"
     python3 test_endpoint.py --image input.png
     python3 test_endpoint.py --video input.mp4
"""

import os
import sys
import json
import time
import base64
import argparse
import requests


def encode_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Test Real-CUGAN RunPod Serverless Worker")
    parser.add_argument("--local", action="store_true", help="Run handler locally directly in Python")
    parser.add_argument("--image", type=str, help="Path or URL to test image")
    parser.add_argument("--video", type=str, help="Path or URL to test video")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor (2, 3, 4)")
    parser.add_argument("--denoise", type=str, default="conservative", help="Denoise mode (conservative, no-denoise, 1x, 2x, 3x)")
    parser.add_argument("--tile-mode", default="auto", help="Tiling mode (auto, 0, 1, 2, 3, 4)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("RUNPOD_API_KEY", ""), help="RunPod API Key")
    parser.add_argument("--endpoint-id", type=str, default=os.environ.get("RUNPOD_ENDPOINT_ID", ""), help="RunPod Endpoint ID")

    args = parser.parse_args()

    payload_input = {
        "scale": args.scale,
        "denoise": args.denoise,
        "tile_mode": args.tile_mode,
        "upload_destination": "catbox",
    }

    if args.video:
        if args.video.startswith("http://") or args.video.startswith("https://"):
            payload_input["video"] = args.video
        else:
            payload_input["video"] = encode_file_to_base64(args.video)
    elif args.image:
        if args.image.startswith("http://") or args.image.startswith("https://"):
            payload_input["image"] = args.image
        else:
            payload_input["image"] = encode_file_to_base64(args.image)
    else:
        # Default fallback sample image
        payload_input["image"] = "https://raw.githubusercontent.com/bilibili/ailab/main/Real-CUGAN/demos/comic.jpg"
        print("ℹ️ No media provided, using default sample anime image from GitHub.")

    # Local Direct Execution Mode
    if args.local:
        print("💻 Running test locally with handler.py...")
        from handler import handler

        mock_job = {
            "id": "local-test-001",
            "input": payload_input,
        }
        t0 = time.time()
        result = handler(mock_job)
        print(f"⏱️ Finished in {time.time() - t0:.2f}s")
        print("\n=== Result ===")
        print(json.dumps(result, indent=2))
        return

    # RunPod Serverless Cloud API Mode
    api_key = args.api_key
    endpoint_id = args.endpoint_id
    if not api_key or not endpoint_id:
        print("❌ Error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set for cloud testing.", file=sys.stderr)
        sys.exit(1)

    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"🚀 Submitting job to RunPod Serverless endpoint ({endpoint_id})...")
    t0 = time.time()
    resp = requests.post(url, headers=headers, json={"input": payload_input}, timeout=300)
    resp.raise_for_status()

    data = resp.json()
    print(f"⏱️ Response received in {time.time() - t0:.2f}s")
    print("\n=== RunPod Response ===")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
