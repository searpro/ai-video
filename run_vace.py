"""Wan2.1 VACE 1.3B (GGUF) — t2v / i2v / video edit on Apple Silicon.

Examples:
  uv run run_vace.py t2v  --prompt "a cat surfing a wave"
  uv run run_vace.py i2v  --prompt "the cat starts dancing" --image cat.png
  uv run run_vace.py edit --prompt "make it snowy" --video in.mp4 [--mask mask.mp4]
  (any mode) --ref ref.png   # extra reference image(s) for subject/style
"""

import argparse
import gc
import json
import os
import struct
import time

import torch
from diffusers import GGUFQuantizationConfig, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.loaders.single_file_utils import load_single_file_checkpoint
from diffusers.utils import export_to_video, load_image, load_video
from huggingface_hub import get_session, hf_hub_url
from PIL import Image
from transformers import UMT5Config, UMT5EncoderModel

BASE = "models/wan-vace-diffusers"
TRANSFORMER_GGUF = "models/wan-vace-gguf/Wan2.1-VACE-1.3B-Q3_K_S.gguf"
TEXT_ENCODER_GGUF = "models/umt5-gguf/umt5-xxl-encoder-Q3_K_S.gguf"
# The GGUF conversion dropped this 5D conv weight (it feeds VACE's image/video context), so we
# range-fetch just that ~1MB tensor from the official safetensors once and cache it.
PATCH = "models/wan-vace-gguf/vace_patch_embedding.safetensors"


def fetch_vace_patch_embedding():
    if os.path.exists(PATCH):
        return
    from safetensors.torch import load, save_file
    url = hf_hub_url("Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                     "transformer/diffusion_pytorch_model-00001-of-00002.safetensors")
    get = lambda a, b: get_session().get(url, headers={"Range": f"bytes={a}-{b}"}).content
    n = struct.unpack("<Q", get(0, 7))[0]
    info = json.loads(get(8, 8 + n - 1))["vace_patch_embedding.weight"]
    a, b = info["data_offsets"]
    raw = get(8 + n + a, 8 + n + b - 1)
    hdr = json.dumps({"vace_patch_embedding.weight": {**info, "data_offsets": [0, b - a]}}).encode()
    t = load(struct.pack("<Q", len(hdr)) + hdr + raw)
    save_file(t, PATCH)
NEG = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["t2v", "i2v", "edit"])
p.add_argument("--prompt", required=True)
p.add_argument("--negative", default=NEG)
p.add_argument("--image", help="i2v: first frame")
p.add_argument("--video", help="edit: source/control video")
p.add_argument("--mask", help="edit: mask video or image (white = regenerate). Default: regenerate all")
p.add_argument("--ref", nargs="*", default=[], help="reference image(s)")
p.add_argument("--width", type=int, default=832)
p.add_argument("--height", type=int, default=480)
p.add_argument("--frames", type=int, default=33, help="must be 4k+1")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--cfg", type=float, default=5.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--fps", type=int, default=16)
p.add_argument("--out", default="out.mp4")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16
W, H, N = args.width, args.height, args.frames
t0 = time.time()

# 1) Text encoding with the GGUF umt5 (dequantized on load), then free it — it's the memory hog.
text_encoder = UMT5EncoderModel.from_pretrained(
    "models/umt5-gguf", gguf_file=TEXT_ENCODER_GGUF.split("/")[-1],
    config=UMT5Config.from_pretrained(f"{BASE}/text_encoder"), torch_dtype=dtype,
).to(device)
pipe = WanVACEPipeline.from_pretrained(BASE, transformer=None, text_encoder=text_encoder, torch_dtype=dtype)
with torch.no_grad():
    pe, npe = pipe.encode_prompt(args.prompt, args.negative, do_classifier_free_guidance=args.cfg > 1,
                                 max_sequence_length=512, device=device, dtype=dtype)
pipe.text_encoder = None
del text_encoder
gc.collect()
if device == "mps":
    torch.mps.empty_cache()
print(f"prompt encoded in {time.time() - t0:.0f}s")

# 2) GGUF transformer (stays quantized, dequantized per-layer at compute time).
fetch_vace_patch_embedding()
from safetensors.torch import load_file
sd = load_single_file_checkpoint(TRANSFORMER_GGUF)
sd.update(load_file(PATCH))
pipe.transformer = WanVACETransformer3DModel.from_single_file(
    sd, config=BASE, subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype,
)
pipe.to(device)

# 3) Build VACE conditioning: video frames + mask (white = generate, black = keep).
gray = Image.new("RGB", (W, H), (128, 128, 128))
white, black = Image.new("L", (W, H), 255), Image.new("L", (W, H), 0)
video = mask = None
if args.mode == "i2v":
    assert args.image, "--image required for i2v"
    first = load_image(args.image).convert("RGB").resize((W, H))
    video = [first] + [gray] * (N - 1)
    mask = [black] + [white] * (N - 1)
elif args.mode == "edit":
    assert args.video, "--video required for edit"
    video = [f.convert("RGB").resize((W, H)) for f in load_video(args.video)[:N]]
    N = len(video) - (len(video) - 1) % 4  # snap to 4k+1
    video = video[:N]
    if args.mask:
        m = load_video(args.mask) if args.mask.endswith((".mp4", ".mov", ".gif")) else [load_image(args.mask)] * N
        mask = [f.convert("L").resize((W, H)) for f in m[:N]]
refs = [load_image(r).convert("RGB") for r in args.ref] or None

out = pipe(
    video=video, mask=mask, reference_images=refs,
    prompt_embeds=pe, negative_prompt_embeds=npe,
    height=H, width=W, num_frames=N, num_inference_steps=args.steps, guidance_scale=args.cfg,
    generator=torch.Generator().manual_seed(args.seed),
).frames[0]
export_to_video(out, args.out, fps=args.fps)
print(f"saved {args.out} ({N} frames, {W}x{H}) in {time.time() - t0:.0f}s")
