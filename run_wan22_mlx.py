#!/usr/bin/env python3
"""Wan 2.2 TI2V-5B through MLX instead of PyTorch MPS.

    uv run run_wan22_mlx.py --image images/baker.png --prompt "..." \
        --size 512 --frames 49 --steps 6 --out out/shots/baker-mlx.mp4

Same job as run_wan22.py, different runtime. MLX is Apple's own framework and
uses unified memory directly, where PyTorch's MPS backend adapts a CUDA-shaped
design onto Metal; on this hardware that gap is the point of the exercise.

This is a thin wrapper around mlx-video, plus one fix without which it cannot
run on a 24GB machine at all — see keep_t5_small() below.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
MODEL_DIR = HERE / "models" / "wan22-mlx-q8"


def keep_t5_small() -> None:
    """Stop mlx-video upcasting the T5 encoder to fp32 as it loads it.

    Its loader does `weights.astype(mx.float32)` for "maximum precision",
    reasoning that the encoder runs once per generation so the cost is
    negligible. That is true of time and false of memory: umt5-xxl is 11.4GB in
    bf16 and 22.7GB in fp32, which on a 24GB Mac does not merely swap — the
    single `mx.eval` over all of it overruns Metal's command-buffer watchdog
    and the run dies with a GPU timeout.

    bf16 is the dtype Wan ships and the one the PyTorch path uses, so this
    gives up nothing real. The eval is also done a layer at a time, since one
    enormous command buffer is what trips the watchdog.
    """
    import mlx.core as mx
    from mlx_video.models.wan_2 import generate as gen
    from mlx_video.models.wan_2.text_encoder import T5Encoder

    def load_t5_encoder(model_path, config):
        encoder = T5Encoder(
            vocab_size=config.t5_vocab_size, dim=config.t5_dim,
            dim_attn=config.t5_dim_attn, dim_ffn=config.t5_dim_ffn,
            num_heads=config.t5_num_heads, num_layers=config.t5_num_layers,
            num_buckets=config.t5_num_buckets, shared_pos=False,
        )
        weights = mx.load(str(model_path))
        encoder.load_weights([(k, v.astype(mx.bfloat16)) for k, v in weights.items()])
        for name, child in encoder.children().items():
            mx.eval(child)
        mx.eval(encoder.parameters())
        return encoder

    gen.load_t5_encoder = load_t5_encoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default=None)
    p.add_argument("--size", type=int, default=512, help="short side")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--frames", type=int, default=49, help="4n+1")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheduler", default="unipc", choices=["euler", "dpm++", "unipc"])
    p.add_argument("--tiling", default="auto")
    p.add_argument("--lora", nargs=2, action="append", metavar=("PATH", "STRENGTH"),
                   help="e.g. a Lightning LoRA to cut the step count")
    p.add_argument("--out", default="out_wan22_mlx.mp4")
    a = p.parse_args()

    # Match run_wan22.py: derive the frame from the driving image's aspect so a
    # portrait still does not get letterboxed into a landscape clip.
    if a.width and a.height:
        w, h = a.width, a.height
    else:
        iw, ih = Image.open(a.image).size
        if iw <= ih:
            w, h = a.size, round(a.size * ih / iw)
        else:
            w, h = round(a.size * iw / ih), a.size
    w, h = (round(v / 32) * 32 for v in (w, h))

    keep_t5_small()
    from mlx_video.models.wan_2.generate import generate_video

    print(f"mlx wan22 i2v: {w}x{h} {a.frames}f {a.steps} steps cfg {a.cfg}", flush=True)
    t0 = time.time()
    generate_video(
        model_dir=str(MODEL_DIR), prompt=a.prompt, negative_prompt=a.negative,
        image=a.image, width=w, height=h, num_frames=a.frames, steps=a.steps,
        guide_scale=a.cfg, seed=a.seed, output_path=a.out, scheduler=a.scheduler,
        tiling=a.tiling,
        loras=[[path, float(strength)] for path, strength in (a.lora or [])] or None,
    )
    print(f"total {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
