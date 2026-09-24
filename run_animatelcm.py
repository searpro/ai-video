"""AnimateDiff (SD1.5, GGUF UNet) + AnimateLCM — t2v / i2v / video edit on Apple Silicon.

Examples:
  uv run run_animatelcm.py t2v  --prompt "a red fox running through a snowy forest"
  uv run run_animatelcm.py i2v  --prompt "the fox jumps" --image fox.png [--strength 0.7]
  uv run run_animatelcm.py edit --prompt "a fox under a starry sky" --video in.mp4 [--strength 0.6]

i2v: AnimateDiff has no native image conditioning, so the image is repeated into a still clip
and animated via video-to-video (lower --strength = closer to the image, less motion).
"""

import argparse
import time

import gguf
import numpy as np
import torch
from diffusers import (AnimateDiffPipeline, AnimateDiffVideoToVideoPipeline, LCMScheduler,
                       MotionAdapter, UNet2DConditionModel)
from diffusers.models.attention_processor import SlicedAttnProcessor
from diffusers.utils import export_to_video, load_image, load_video

BASE = "models/sd15-base"
UNET_GGUF = "models/sd15-unet-gguf/unet-q4_0.gguf"
ADAPTER = "models/animatelcm"
LORA = "models/animatelcm/AnimateLCM_sd15_t2v_lora.safetensors"
NEG = "bad quality, worse quality, low resolution, blurry, deformed"


def load_gguf_unet(path, dtype):
    """Dequantize a ComfyUI-style GGUF UNet (diffusers key names) into a regular UNet."""
    r = gguf.GGUFReader(path)
    shapes = {k.removeprefix("comfy.gguf.orig_shape."): [int(v) for v in f.contents()]
              for k, f in r.fields.items() if k.startswith("comfy.gguf.orig_shape.")}
    sd = {}
    for t in r.tensors:
        w = torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(t.data, t.tensor_type)))
        sd[t.name] = w.reshape(shapes.get(t.name, w.shape)).to(dtype)
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet"))
    unet.load_state_dict(sd, strict=True)
    return unet.to(dtype)


p = argparse.ArgumentParser()
p.add_argument("mode", choices=["t2v", "i2v", "edit"])
p.add_argument("--prompt", required=True)
p.add_argument("--negative", default=NEG)
p.add_argument("--image", help="i2v: input image")
p.add_argument("--video", help="edit: source video")
p.add_argument("--strength", type=float, default=None, help="i2v/edit denoise strength (0-1)")
p.add_argument("--width", type=int, default=512)
p.add_argument("--height", type=int, default=512)
p.add_argument("--frames", type=int, default=16)
p.add_argument("--steps", type=int, default=6)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--lora-scale", type=float, default=0.8)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--fps", type=int, default=8)
p.add_argument("--out", default="out_animatelcm.mp4")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16
W, H, N = args.width, args.height, args.frames
t0 = time.time()

adapter = MotionAdapter.from_pretrained(ADAPTER, variant="fp16", torch_dtype=dtype)
pipe = AnimateDiffPipeline.from_pretrained(
    BASE, unet=load_gguf_unet(UNET_GGUF, dtype), motion_adapter=adapter,
    variant="fp16", torch_dtype=dtype, feature_extractor=None, image_encoder=None,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights(LORA, adapter_name="lcm")
pipe.set_adapters(["lcm"], [args.lora_scale])
pipe.fuse_lora()
pipe.unload_lora_weights()
# MPS: full attention tries a 16GB buffer, and SDPA is slow at SD1.5's head_dim=40; sliced bmm is fastest.
pipe.unet.set_attn_processor(SlicedAttnProcessor(slice_size=32))
pipe.to(device)
print(f"loaded in {time.time() - t0:.0f}s")

common = dict(prompt=args.prompt, negative_prompt=args.negative, height=H, width=W,
              num_inference_steps=args.steps, guidance_scale=args.cfg,
              generator=torch.Generator().manual_seed(args.seed))
if args.mode == "t2v":
    out = pipe(num_frames=N, **common).frames[0]
else:
    if args.mode == "i2v":
        assert args.image, "--image required for i2v"
        video = [load_image(args.image).convert("RGB").resize((W, H))] * N
        strength = args.strength or 0.7
    else:
        assert args.video, "--video required for edit"
        video = [f.convert("RGB").resize((W, H)) for f in load_video(args.video)[:N]]
        strength = args.strength or 0.6
    v2v = AnimateDiffVideoToVideoPipeline.from_pipe(pipe)
    out = v2v(video=video, strength=strength, **common).frames[0]

export_to_video(out, args.out, fps=args.fps)
print(f"saved {args.out} ({len(out)} frames, {W}x{H}) in {time.time() - t0:.0f}s")
