"""Wan2.2 TI2V-5B Turbo (GGUF) — t2v / i2v on Apple Silicon. 4 steps, no CFG.

  uv run run_wan22.py i2v --image shot.png --prompt "she turns and smiles"
  uv run run_wan22.py t2v --prompt "a neon-lit alley in the rain"
  --upscale 2 : Real-ESRGAN pass (same as run_ltx.py)
"""

import argparse
import gc
import time

import torch
from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from PIL import Image
from transformers import T5Config, UMT5EncoderModel

BASE = "models/wan22-base"
T5_GGUF = "models/umt5-gguf/umt5-xxl-encoder-Q3_K_S.gguf"
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
       "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
       "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["t2v", "i2v"])
p.add_argument("--prompt", required=True)
p.add_argument("--negative", default=NEG)
p.add_argument("--image", help="i2v: first frame")
p.add_argument("--width", type=int, help="default: from image aspect at --size")
p.add_argument("--height", type=int)
p.add_argument("--size", type=int, default=704, help="short side (model likes 1280x704)")
p.add_argument("--frames", type=int, default=49, help="4k+1")
p.add_argument("--steps", type=int, default=4)
p.add_argument("--cfg", type=float, default=1.0)
p.add_argument("--quant", default="Q5_K_M")
p.add_argument("--upscale", type=float, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--fps", type=int, default=24)
p.add_argument("--out", default="out_wan22.mp4")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16
MOD = 32  # vae spatial 16 * patch 2
t0 = time.time()

src = load_image(args.image).convert("RGB") if args.image else None
if args.width and args.height:
    W, H = args.width, args.height
elif src is not None:  # keep aspect: short side = --size
    r = src.width / src.height
    W, H = (args.size * r, args.size) if r >= 1 else (args.size, args.size / r)
    W, H = max(MOD, round(W / MOD) * MOD), max(MOD, round(H / MOD) * MOD)
else:
    W, H = 1280, 704

# 1) umt5 from GGUF, encode, free.
te = UMT5EncoderModel.from_pretrained(
    "models/umt5-gguf", gguf_file=T5_GGUF.split("/")[-1],
    config=T5Config.from_pretrained(f"{BASE}/text_encoder"), torch_dtype=dtype,
).to(device)
# bf16 VAE: decode is the bottleneck on MPS (386s vs 1487s in fp32 for 49f), output stays stable
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=dtype)
pipe = WanImageToVideoPipeline.from_pretrained(
    BASE, transformer=None, vae=vae, text_encoder=te, image_encoder=None, image_processor=None, torch_dtype=dtype)
with torch.no_grad():
    pe, npe = pipe.encode_prompt(args.prompt, args.negative, do_classifier_free_guidance=args.cfg > 1,
                                 device=device)
pipe.text_encoder = None
del te
gc.collect()
torch.mps.empty_cache() if device == "mps" else None
print(f"prompt encoded in {time.time() - t0:.0f}s")

# 2) GGUF transformer (stays quantized) + Wan 2.2 VAE.
pipe.transformer = WanTransformer3DModel.from_single_file(
    f"models/wan22-gguf/Wan2_2-TI2V-5B-Turbo-{args.quant}.gguf", config=BASE, subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype,
)
pipe.to(device)

out = pipe(image=src, prompt_embeds=pe, negative_prompt_embeds=npe if args.cfg > 1 else None,
           height=H, width=W, num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=args.cfg,
           output_type="pil", generator=torch.Generator().manual_seed(args.seed)).frames[0]

if args.upscale:
    import numpy as np
    from spandrel import ModelLoader

    del pipe
    gc.collect()
    torch.mps.empty_cache() if device == "mps" else None
    sr = ModelLoader().load_from_file("models/esrgan/RealESRGAN_x4plus.safetensors").model.to(device).eval().half()
    W, H = round(W * args.upscale / 2) * 2, round(H * args.upscale / 2) * 2
    with torch.no_grad():
        for i, fr in enumerate(out):
            x = torch.from_numpy(np.asarray(fr)).permute(2, 0, 1)[None].to(device, torch.float16) / 255
            y = (sr(x)[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            out[i] = Image.fromarray(y).resize((W, H), Image.LANCZOS)

export_to_video(out, args.out, fps=args.fps, quality=9)
print(f"saved {args.out} ({len(out)} frames, {W}x{H}) in {time.time() - t0:.0f}s")
