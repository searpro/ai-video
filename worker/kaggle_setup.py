"""Bring the media worker up on a Kaggle/Colab GPU session.

Paste into one notebook cell:

    !git clone -q https://github.com/<you>/ai-video /kaggle/working/ai-video 2>/dev/null; \
     python /kaggle/working/ai-video/worker/kaggle_setup.py

or, without a repo, upload worker/ and run `python kaggle_setup.py`.

It installs what the base image lacks, fetches the weights, starts the worker
and opens a Cloudflare tunnel, then prints the URL to paste into Viceroy's
provider rows. Weights are the slow part on a cold session (~20GB), so once a
run works, snapshot WORKER_MODELS into a private Kaggle Dataset and set
WORKER_MODELS to its mount path: a dataset is attached instantly and does not
count against the session clock.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("WORKER_ROOT", "/kaggle/working/media"))
MODELS = Path(os.environ.get("WORKER_MODELS", str(ROOT / "models")))
PORT = os.environ.get("PORT", "8000")

# Only what a Kaggle image does not already carry; torch is preinstalled and
# reinstalling it would pull a CPU wheel over the CUDA one.
PIP = ["diffusers>=0.40", "transformers>=4.56", "accelerate", "fastapi", "uvicorn",
       "python-multipart", "imageio", "imageio-ffmpeg", "ftfy", "sentencepiece", "protobuf"]

# repo -> (subdir under MODELS, allow_patterns). Trimmed to inference files:
# the full repos carry training states and duplicate formats.
WEIGHTS = {
    "Tongyi-MAI/Z-Image-Turbo": ("z-image-turbo", None),
    "Qwen/Qwen-Image-2.1": ("qwen-image-2.1", None),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": ("wan2.2-ti2v-5b", None),
}


def sh(cmd: str, check: bool = True) -> str:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode and check:
        raise SystemExit(f"failed: {cmd}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout


def install() -> None:
    sh(f"{sys.executable} -m pip install -q --no-input " + " ".join(f'"{p}"' for p in PIP))
    if not Path("/usr/local/bin/cloudflared").exists():
        sh("curl -sL -o /usr/local/bin/cloudflared "
           "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
           "&& chmod +x /usr/local/bin/cloudflared")


def fetch_weights() -> None:
    from huggingface_hub import snapshot_download

    MODELS.mkdir(parents=True, exist_ok=True)
    for repo, (subdir, patterns) in WEIGHTS.items():
        target = MODELS / subdir
        if target.exists() and any(target.rglob("*.safetensors")):
            print(f"= {subdir} already present", flush=True)
            continue
        print(f"+ {repo} -> {target}", flush=True)
        snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=patterns,
                          max_workers=8, ignore_patterns=["*.pth", "*.ckpt", "*.onnx"])


def serve() -> None:
    env = {**os.environ, "WORKER_ROOT": str(ROOT), "WORKER_MODELS": str(MODELS), "PORT": PORT}
    here = Path(__file__).parent
    server = subprocess.Popen([sys.executable, str(here / "server.py")], env=env)

    # Wait for the worker before opening the tunnel, so the printed URL is
    # never a 502 that looks like a tunnel problem.
    import urllib.request

    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            break
        except Exception:
            if server.poll() is not None:
                raise SystemExit("worker exited during startup")
            time.sleep(2)
    else:
        raise SystemExit("worker did not become healthy")

    tunnel = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    url = None
    assert tunnel.stdout
    for line in tunnel.stdout:
        found = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
        if found:
            url = found.group(0)
            break

    print("\n" + "=" * 68)
    print(f"  worker URL: {url}")
    print("  point Viceroy's image and video providers at it:")
    print(f'    cd ~/projects/viceroy && sqlite3 data/viceroy.db "update providers '
          f"set base_url='{url}' where is_default=1 and kind in ('image','video');\"")
    print("=" * 68 + "\n", flush=True)

    # Hold the cell open: when this returns, Kaggle tears the session down.
    for line in tunnel.stdout:
        if any(w in line for w in ("ERR", "error", "Retrying")):
            print(line, end="", flush=True)


if __name__ == "__main__":
    install()
    fetch_weights()
    serve()
