"""How fast is Z-Image in fp32, and is the P100 the better card for it?

fp16 is a dead end for this model. Chasing the overflow moved it rather than
fixing it: upcasting the refiner blocks cleared them and it reappeared at
layers.0.feed_forward.w2. The activations broadly exceed fp16's 65504, which
is consistent with bf16 (same exponent range as fp32) being clean throughout.

So the question is no longer "how do we get fp16 to work" but "what is the
cheapest correct dtype". On paper the T4 does 3.9 fp32 TFLOP/s against 2.2 for
emulated bf16, and the P100 does ~9.3 fp32 with 732GB/s of bandwidth against
the T4's 320. If that holds, fp32 on a P100 beats every T4 option.

The text encoder stays in fp16 — it was measured clean at absmax 143.9, and in
fp32 it would not fit alongside the transformer.
"""


import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                "gguf>=0.10.0", "diffusers>=0.40", "transformers>=4.56", "accelerate", "hf_xet"],
               check=True)

import numpy as np
import torch
from PIL import Image

OUT = Path("/kaggle/working")
MODELS = Path("/kaggle/temp/media/models")
DT = torch.float32          # the transformer and VAE
TE_DT = torch.float16       # measured clean, and fp32 would not fit
report: dict = {}


def say(*a):
    print(*a, flush=True)


from huggingface_hub import snapshot_download

for repo, sub, pats in [
    ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo/gguf", ["z-image-turbo-Q6_K.gguf"]),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "z-image-turbo/te", ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"]),
    ("Tongyi-MAI/Z-Image-Turbo", "z-image-turbo",
     ["model_index.json", "vae/*", "scheduler/*", "transformer/config.json", "tokenizer/*", "text_encoder/config.json"]),
]:
    snapshot_download(repo_id=repo, local_dir=str(MODELS / sub), allow_patterns=pats, max_workers=8)

from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                       ZImagePipeline, ZImageTransformer2DModel)
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = MODELS / "z-image-turbo"
PROMPT = ("a village baker in her forties, flour dusted on her apron, warm morning light "
          "through a bakery window, photorealistic portrait, shallow depth of field")

say("building pipeline")
pipe = ZImagePipeline(
    transformer=ZImageTransformer2DModel.from_single_file(
        str(next((BASE / "gguf").glob("*.gguf"))), config=str(BASE), subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), dtype=DT),
    text_encoder=AutoModelForCausalLM.from_pretrained(
        str(BASE / "te"), gguf_file=next((BASE / "te").glob("*.gguf")).name, dtype=TE_DT),
    tokenizer=AutoTokenizer.from_pretrained(str(BASE / "tokenizer")),
    vae=AutoencoderKL.from_pretrained(str(BASE), subfolder="vae", dtype=DT),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(BASE), subfolder="scheduler"),
)


def cast_tree(x, dtype):
    """Cast every floating tensor in a nested structure, leaving the rest alone."""
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, tuple):
        return tuple(cast_tree(i, dtype) for i in x)
    if isinstance(x, list):
        return [cast_tree(i, dtype) for i in x]
    if isinstance(x, dict):
        return {k: cast_tree(v, dtype) for k, v in x.items()}
    return x


# Bridge the fp16 encoder to the fp32 transformer.
_orig_encode = pipe.encode_prompt


def _encode(*a, **k):
    out = _orig_encode(*a, **k)
    return cast_tree(out, DT)


pipe.encode_prompt = _encode
report["dtypes"] = {"transformer": str(DT), "text_encoder": str(TE_DT)}
say("transformer/vae in", DT, "| text encoder in", TE_DT)

pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)

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
    img = pipe(prompt=PROMPT, width=768, height=1024, num_inference_steps=8, guidance_scale=1.0,
               generator=torch.Generator("cpu").manual_seed(42),
               callback_on_step_end=tick, output_type="pil").images[0]
    report["denoise_s"] = round(time.time() - t0)
    report["step_times"] = times
    report["nan_steps"] = nan_steps
    img.save(OUT / "fp32_transformer.png")
    a = np.array(img)
    report["image"] = {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
                       "unique": int(len(np.unique(a.reshape(-1, 3), axis=0)))}
    say("image stats:", report["image"], f"in {report['denoise_s']}s")
except Exception as e:
    say("failed:", str(e)[:400])
    report["error"] = str(e)[:400]

for h in handles:
    h.remove()
report["still_overflowing"] = seen
say("\n--- still turning finite input into non-finite output ---")
for k, v in seen.items():
    say("  ", k, v)
if not seen:
    say("   none")

(OUT / "report.json").write_text(json.dumps(report, indent=2))
say("\n" + "=" * 70)
say(json.dumps(report, indent=2)[:3000])
