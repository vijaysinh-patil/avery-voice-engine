#!/usr/bin/env python3
"""Download Kokoro model files to /data/models/ on first run."""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/data/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = {
    "kokoro-v1.0.int8.onnx": f"{_RELEASE}/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin":        f"{_RELEASE}/voices-v1.0.bin",
}

for filename, url in FILES.items():
    dest = MODEL_DIR / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"✓ {filename} already present ({dest.stat().st_size:,} bytes)")
        continue
    print(f"Downloading {filename} …", flush=True)
    def _progress(count, block, total):
        pct = min(100, int(count * block * 100 / total)) if total > 0 else 0
        print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r✓ {filename} ({dest.stat().st_size:,} bytes)")

print("Models ready.", flush=True)
