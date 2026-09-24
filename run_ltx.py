"""LTX-Video 0.9.8 distilled (2B self-made GGUF, or 13B GGUF) — t2v / i2v / video edit on Apple Silicon.

Multi-scale (Lightricks' distilled recipe): 7 steps at half res -> 2x latent upscale -> 3 refine steps, no CFG.
Output size follows the input image/video aspect (long side --size) unless --width/--height are given.

Fidelity (i2v), measured against the source image:
  default                         fast, 576x1024
  --supersample                   render at 4/3 size and downscale: sharper, ~2x slower
  --model 13b                     far better temporal consistency (face/texture hold), ~6x slower
  --upscale 2                     Real-ESRGAN to 2x res; the LTX VAE caps detail at ~50% of the source

Examples:
  uv run run_ltx.py t2v  --prompt "a red fox running through a snowy forest"
  uv run run_ltx.py i2v  --prompt "the fox jumps" --image fox.png
  uv run run_ltx.py edit --prompt "a fox at night under a starry sky" --video in.mp4 [--strength 0.6]
  --single-pass : skip the latent-upscale pass (faster, softer)
"""

import argparse
import gc
import time

import torch
from PIL import Image
from diffusers import GGUFQuantizationConfig, LTXConditionPipeline, LTXLatentUpsamplePipeline, LTXVideoTransformer3DModel
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video
from transformers import T5Config, T5EncoderModel

BASE = "models/ltx-base"
T5_GGUF = "models/t5-gguf/t5-v1_1-xxl-encoder-Q3_K_S.gguf"
UPSCALER = "models/ltx-upscaler"
# distilled schedules from Lightricks' ltxv-2b-0.9.8-distilled.yaml
PASS1 = [1000, 993, 987, 981, 975, 909, 725, 0.03]
PASS2 = [1000, 909, 725, 421, 0]  # with denoise_strength=0.999 -> starts at 909

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["t2v", "i2v", "edit"])
p.add_argument("--prompt", required=True)
p.add_argument("--image", help="i2v: first frame")
p.add_argument("--video", help="edit: source video")
p.add_argument("--strength", type=float, default=0.6, help="edit: how much to change (0-1)")
p.add_argument("--width", type=int, help="multiple of 64 (default: from input aspect, else 768)")
p.add_argument("--height", type=int, help="multiple of 64 (default: from input aspect, else 512)")
p.add_argument("--size", type=int, default=1024, help="long side when size comes from the input's aspect")
p.add_argument("--model", default="2b", choices=["2b", "13b"], help="13b: much better fidelity, ~4x slower")
p.add_argument("--quant", default=None, help="GGUF quant in models/ltx-gguf (default: Q8_0 for 2b, Q4_K_M for 13b)")
p.add_argument("--supersample", action="store_true",
               help="Lightricks' recipe: render at 4/3 size, downscale to target (sharper, ~1.8x slower pass 2)")
p.add_argument("--frames", type=int, default=97, help="8k+1")
p.add_argument("--single-pass", action="store_true")
p.add_argument("--upscale", type=float, default=0, help="Real-ESRGAN post-upscale factor (e.g. 2); 0 = off")
p.add_argument("--sr", default="best", choices=["best", "fast"], help="upscaler: best=x4plus (~9s/frame), fast=x2 (~2.6s/frame)")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--fps", type=int, default=24)
p.add_argument("--out", default="out_ltx.mp4")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16
N = args.frames
src = load_image(args.image) if args.image else (load_video(args.video)[0] if args.video else None)
if args.width and args.height:
    W, H = args.width, args.height
elif src is not None:  # keep the input's aspect ratio: long side = --size, both sides multiples of 64
    r = src.width / src.height
    W, H = (args.size, args.size / r) if r >= 1 else (args.size * r, args.size)
    W, H = max(64, round(W / 64) * 64), max(64, round(H / 64) * 64)
else:
    W, H = 768, 512
t0 = time.time()

# 1) T5 from GGUF (dequantized on load), encode, free.
te = T5EncoderModel.from_pretrained(
    "models/t5-gguf", gguf_file=T5_GGUF.split("/")[-1],
    config=T5Config.from_pretrained(f"{BASE}/text_encoder"), torch_dtype=dtype,
).to(device)
pipe = LTXConditionPipeline.from_pretrained(BASE, transformer=None, text_encoder=te, torch_dtype=dtype)
with torch.no_grad():
    pe, pm, _, _ = pipe.encode_prompt(args.prompt, do_classifier_free_guidance=False,
                                      max_sequence_length=256, device=device, dtype=dtype)
pipe.text_encoder = None
del te
gc.collect()
torch.mps.empty_cache() if device == "mps" else None
print(f"prompt encoded in {time.time() - t0:.0f}s")

# 2) GGUF transformer (stays Q4_0 in memory) + VAE tiling to cap decode memory.
gguf_path, cfg = {
    "2b": (f"models/ltx-gguf/ltxv-2b-0.9.8-distilled-{args.quant or 'Q8_0'}.gguf", BASE),
    "13b": (f"models/ltx-gguf/LTXV-13B-0.9.8-distilled-{args.quant or 'Q4_K_M'}.gguf", "models/ltx13b-cfg"),
}[args.model]
pipe.transformer = LTXVideoTransformer3DModel.from_single_file(
    gguf_path, config=cfg, subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype,
)
pipe.to(device)
pipe.vae.enable_tiling()

conditions, latents, denoise = None, None, 1.0
if args.mode == "i2v":
    conditions = [LTXVideoCondition(image=src.convert("RGB"), frame_index=0)]
elif args.mode == "edit":
    frames = load_video(args.video)
    N = min(N, (len(frames) - 1) // 8 * 8 + 1)
    frames = [f.convert("RGB") for f in frames[:N]]

common = dict(prompt_embeds=pe, prompt_attention_mask=pm, num_frames=N, guidance_scale=1.0,
              decode_timestep=0.05, decode_noise_scale=0.025, image_cond_noise_scale=0.0,
              generator=torch.Generator().manual_seed(args.seed), frame_rate=args.fps)


def encode_video(frames, w, h):
    """SDEdit-style edit: encode the source video to normalized latents."""
    v = pipe.video_processor.preprocess_video(frames, h, w).to(device, dtype)
    with torch.no_grad():
        lat = pipe.vae.encode(v).latent_dist.mode()
    return pipe._normalize_latents(lat, pipe.vae.latents_mean, pipe.vae.latents_std).float()


if args.single_pass:
    if args.mode == "edit":
        latents, denoise = encode_video(frames, W, H), args.strength
    out = pipe(conditions=conditions, width=W, height=H, timesteps=PASS1, latents=latents,
               denoise_strength=denoise, **common).frames[0]
else:
    f = 2 / 3 if args.supersample else 1 / 2  # pass-1 scale; the 2x upscale lands on W x H (or 4/3 of it)
    w1, h1 = round(W * f / 32) * 32, round(H * f / 32) * 32
    if args.mode == "edit":
        latents, denoise = encode_video(frames, w1, h1), args.strength
    lat = pipe(conditions=conditions, width=w1, height=h1, timesteps=PASS1, latents=latents,
               denoise_strength=denoise, output_type="latent", **common).frames
    print(f"pass 1 ({w1}x{h1}) done at {time.time() - t0:.0f}s")
    up = LTXLatentUpsamplePipeline(
        vae=pipe.vae,
        latent_upsampler=LTXLatentUpsamplerModel.from_pretrained(UPSCALER, subfolder="latent_upsampler",
                                                                  torch_dtype=dtype).to(device),
    )
    lat = up(latents=lat, adain_factor=1.0, output_type="latent").frames
    out = pipe(conditions=conditions, width=w1 * 2, height=h1 * 2, timesteps=PASS2, latents=lat,
               denoise_strength=0.999, **common).frames[0]
    if (w1 * 2, h1 * 2) != (W, H):
        out = [fr.resize((W, H), Image.LANCZOS) for fr in out]

if args.upscale:
    # Real-ESRGAN per frame, then Lanczos to the requested factor (gets past the VAE's detail ceiling)
    import numpy as np
    from spandrel import ModelLoader

    del pipe
    gc.collect()
    torch.mps.empty_cache() if device == "mps" else None
    sr = ModelLoader().load_from_file(
        f"models/esrgan/RealESRGAN_{'x4plus' if args.sr == 'best' else 'x2'}.safetensors").model.to(device).eval().half()
    W, H = round(W * args.upscale / 2) * 2, round(H * args.upscale / 2) * 2
    with torch.no_grad():
        for i, fr in enumerate(out):
            x = torch.from_numpy(np.asarray(fr)).permute(2, 0, 1)[None].to(device, torch.float16) / 255
            y = (sr(x)[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            out[i] = Image.fromarray(y).resize((W, H), Image.LANCZOS)
    print(f"upscaled to {W}x{H} at {time.time() - t0:.0f}s")

export_to_video(out, args.out, fps=args.fps, quality=9)
print(f"saved {args.out} ({len(out)} frames, {W}x{H}) in {time.time() - t0:.0f}s")
