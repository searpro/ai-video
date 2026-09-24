"""Lip-sync a clip to a spoken line with LatentSync (MPS).

  uv run lipsync.py --video shot.mp4 --audio line.wav --out shot-synced.mp4

Same pass make_video.py runs for a shot with "audio"; exposed on its own so a
clip can be re-synced to a new take without regenerating the video.
"""

import argparse
import os
import subprocess
from pathlib import Path

LIPSYNC = Path(__file__).parent / "third_party/latentsync"

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=8)
args = p.parse_args()

subprocess.run(
    [".venv/bin/python", "-m", "scripts.inference",
     "--unet_config_path", "configs/unet/stage2.yaml",
     "--inference_ckpt_path", "checkpoints/v1.5/latentsync_unet.pt",
     "--inference_steps", str(args.steps), "--guidance_scale", "1.5", "--enable_deepcache",
     "--video_path", str(Path(args.video).resolve()), "--audio_path", str(Path(args.audio).resolve()),
     "--video_out_path", str(Path(args.out).resolve())],
    cwd=LIPSYNC, check=True,
    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "LATENTSYNC_DTYPE": "bfloat16",
         "MPS_ATTENTION_BUDGET_MB": "256", "VAE_CHUNK_SIZE": "2"})
print(f"saved {args.out}")
