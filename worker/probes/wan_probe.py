"""Does Wan 2.2 survive fp16 on a T4, and what does a clip cost?

Z-Image does not: its activations pass fp16's 65504 ceiling in several places,
so it needs fp32 at ~215s for a still on either Kaggle card. Video is the
expensive half, so the same question for Wan decides whether Kaggle is usable
at all — a 6x dtype tax on a still is annoying, on a clip it is fatal.

Wan is widely run from GGUF in fp16 on Turing, so the expectation here is
different from Z-Image. This measures rather than assumes: fp16 first, with
per-step NaN checks and a hook naming any module that overflows, and a timed
decode. If fp16 is clean this is the whole answer for video.
"""

import gc
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                "gguf>=0.10.0", "diffusers>=0.40", "transformers>=4.56", "accelerate",
                "hf_xet", "imageio", "imageio-ffmpeg"], check=True)

import numpy as np
import torch
from PIL import Image

OUT = Path("/kaggle/working")
MODELS = Path("/kaggle/temp/media/models")
DT = torch.float16
ENCODER_GPU = 1 if torch.cuda.device_count() > 1 else 0
report: dict = {}


def say(*a):
    print(*a, flush=True)


for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    say(f"cuda:{i} {p.name} cc={p.major}.{p.minor} {p.total_memory/1e9:.1f}GB")

from huggingface_hub import snapshot_download

for repo, sub, pats in [
    ("hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF", "wan2.2-ti2v-5b/gguf", ["Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf"]),
    ("city96/umt5-xxl-encoder-gguf", "wan2.2-ti2v-5b/te", ["umt5-xxl-encoder-Q5_K_M.gguf"]),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b",
     ["model_index.json", "vae/*", "scheduler/*", "transformer/config.json", "tokenizer/*", "text_encoder/config.json"]),
]:
    say(f"+ {repo}")
    snapshot_download(repo_id=repo, local_dir=str(MODELS / sub), allow_patterns=pats, max_workers=8)

from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig, WanImageToVideoPipeline,
                       WanTransformer3DModel)
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, T5Config, UMT5EncoderModel

BASE = MODELS / "wan2.2-ti2v-5b"
W, H, FRAMES, STEPS = 704, 1280, 49, 6
PROMPT = ("a village baker kneading dough, her hands working the flour, warm morning light, "
          "gentle camera push in, cinematic")

# A driving frame. Content does not matter for numerics or timing, and
# generating a real one here would cost another four minutes of Z-Image.
rng = np.random.default_rng(0)
base = np.linspace(40, 210, H, dtype=np.float32)[:, None, None] * np.ones((1, W, 3), dtype=np.float32)
first = Image.fromarray(np.clip(base + rng.normal(0, 12, (H, W, 3)), 0, 255).astype("uint8"))
first.save(OUT / "driving_frame.png")

# Encode first, then throw the encoder away. umt5-xxl dequantizes from GGUF to
# about 11GB and killed three earlier runs via the HOST RAM OOM killer, which
# is a separate and tighter ceiling than VRAM (~29GB on a Kaggle session). Held
# alongside the transformer it does not fit; used and released it does. It also
# loads straight onto the second T4, which is the first real use for that card.
say(f"encoding prompt on cuda:{ENCODER_GPU}")
t0 = time.time()
te = UMT5EncoderModel.from_pretrained(
    str(BASE / "te"), gguf_file=next((BASE / "te").glob("*.gguf")).name,
    config=T5Config.from_pretrained(str(BASE / "text_encoder")), dtype=DT,
    low_cpu_mem_usage=True, device_map={"": f"cuda:{ENCODER_GPU}"})
tok = AutoTokenizer.from_pretrained(str(BASE / "tokenizer"))
ids = tok(PROMPT, padding="max_length", max_length=512, truncation=True,
          return_tensors="pt").to(f"cuda:{ENCODER_GPU}")
with torch.no_grad():
    embeds = te(ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state
embeds = embeds.to("cuda:0", DT)
report["encode_s"] = round(time.time() - t0)
report["embed_absmax"] = round(embeds[torch.isfinite(embeds)].abs().max().item(), 1)
report["embed_nan"] = bool(torch.isnan(embeds).any() or torch.isinf(embeds).any())
say(f"  embeds {tuple(embeds.shape)} absmax={report['embed_absmax']} nan={report['embed_nan']}"
    f" in {report['encode_s']}s")

del te
gc.collect()
torch.cuda.empty_cache()

say(f"building Wan pipeline in {DT}  ({W}x{H}, {FRAMES} frames, {STEPS} steps)")
t0 = time.time()
# The VAE must exist before the pipeline: its scale factor is read at
# construction and a missing one silently halves the latent grid.
vae = AutoencoderKLWan.from_pretrained(str(BASE), subfolder="vae", dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    str(BASE), transformer=None, vae=vae, text_encoder=None, tokenizer=None,
    image_encoder=None, image_processor=None, dtype=DT)
pipe.transformer = WanTransformer3DModel.from_single_file(
    str(next((BASE / "gguf").glob("*.gguf"))), config=str(BASE), subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=DT), dtype=DT)
say("__call__ takes prompt_embeds:",
    "prompt_embeds" in inspect.signature(WanImageToVideoPipeline.__call__).parameters)
pipe.enable_model_cpu_offload()
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)
report["build_s"] = round(time.time() - t0)
say("built in", report["build_s"], "s")

seen: dict[str, dict] = {}


def nonfinite(xs):
    if torch.is_tensor(xs):
        return xs.is_floating_point() and bool(torch.isnan(xs).any() or torch.isinf(xs).any())
    return isinstance(xs, (tuple, list)) and any(nonfinite(x) for x in xs)


def absmax(xs):
    t = xs if torch.is_tensor(xs) else next((x for x in (xs or []) if torch.is_tensor(x)), None)
    if t is None or not t.is_floating_point():
        return None
    f = t[torch.isfinite(t)]
    return round(f.abs().max().item(), 1) if f.numel() else None


def make_hook(name):
    def fn(mod, inp, out):
        if name not in seen and nonfinite(out) and not nonfinite(inp):
            seen[name] = {"type": type(mod).__name__, "in": absmax(inp), "out": absmax(out)}
    return fn


handles = [m.register_forward_hook(make_hook(n)) for n, m in pipe.transformer.named_modules() if n]

nan_steps, times, last = [], [], [time.time()]


def tick(_p, i, _t, kw):
    lat = kw.get("latents")
    times.append(round(time.time() - last[0], 1)); last[0] = time.time()
    if lat is not None:
        b = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
        f = lat[torch.isfinite(lat)]
        say(f"   step {i+1}: nan={b} absmax={f.abs().max().item() if f.numel() else float('nan'):.1f} ({times[-1]}s)")
        if b:
            nan_steps.append(i + 1)
    return kw


t0 = time.time()
try:
    frames = pipe(image=first, prompt_embeds=embeds, width=W, height=H,
                  num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=1.0,
                  output_type="pil", generator=torch.Generator("cpu").manual_seed(42),
                  callback_on_step_end=tick).frames[0]
    report["generate_s"] = round(time.time() - t0)
    report["step_times"] = times
    report["nan_steps"] = nan_steps
    export_to_video(frames, str(OUT / "wan_fp16.mp4"), fps=24, quality=9)
    a = np.array(frames[len(frames) // 2])
    report["mid_frame"] = {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2)}
    report["frames"] = len(frames)
    say("frames:", len(frames), "| mid frame:", report["mid_frame"], "| total", report["generate_s"], "s")
except Exception as e:
    say("failed:", str(e)[:400])
    report["error"] = str(e)[:400]

for h in handles:
    h.remove()
report["overflowing"] = seen
say("\n--- modules turning finite input into non-finite output ---")
for k, v in seen.items():
    say("  ", k, v)
if not seen:
    say("   none")

(OUT / "report.json").write_text(json.dumps(report, indent=2))
say("\n" + "=" * 70)
say(json.dumps(report, indent=2)[:3000])
