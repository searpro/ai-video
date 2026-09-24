"""Is fp16 usable on a T4 for Z-Image, and if so where does it need fp32?

The previous run established:
  - bf16 is numerically clean on sm75 but SOFTWARE EMULATED: 2.2 TFLOP/s
    against fp16's 22.9, and 377s for eight steps. Correct and unusable.
  - so fp16 is the only fast path, and the black image has to be fixed rather
    than avoided by switching dtype.

This run builds ONE pipeline (two in a single kernel tripped the host RAM OOM
killer) in fp16, logs NaN per denoising step to place the blame, then decodes
the same latents three ways to find the cheapest decode that survives.
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
DT = torch.float16
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
    say(f"+ {repo}")
    snapshot_download(repo_id=repo, local_dir=str(MODELS / sub), allow_patterns=pats, max_workers=8)

from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                       ZImagePipeline, ZImageTransformer2DModel)
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = MODELS / "z-image-turbo"
PROMPT = ("a village baker in her forties, flour dusted on her apron, warm morning light "
          "through a bakery window, photorealistic portrait, shallow depth of field")

say("building fp16 pipeline")
t0 = time.time()
pipe = ZImagePipeline(
    transformer=ZImageTransformer2DModel.from_single_file(
        str(next((BASE / "gguf").glob("*.gguf"))), config=str(BASE), subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), dtype=DT),
    text_encoder=AutoModelForCausalLM.from_pretrained(
        str(BASE / "te"), gguf_file=next((BASE / "te").glob("*.gguf")).name, dtype=DT),
    tokenizer=AutoTokenizer.from_pretrained(str(BASE / "tokenizer")),
    vae=AutoencoderKL.from_pretrained(str(BASE), subfolder="vae", dtype=DT),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(BASE), subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
report["build_s"] = round(time.time() - t0)
say("built in", report["build_s"], "s")

nan_steps, step_times = [], []
last = [time.time()]


def tick(_p, i, _t, kw):
    lat = kw.get("latents")
    step_times.append(round(time.time() - last[0], 1))
    last[0] = time.time()
    if lat is not None:
        bad = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
        fin = lat[torch.isfinite(lat)]
        mx = fin.abs().max().item() if fin.numel() else float("nan")
        say(f"   step {i+1}: nan/inf={bad} absmax={mx:.1f} ({step_times[-1]}s)")
        if bad:
            nan_steps.append(i + 1)
    return kw


say("--- denoising in fp16 ---")
t0 = time.time()
lat = pipe(prompt=PROMPT, width=768, height=1024, num_inference_steps=8, guidance_scale=1.0,
           generator=torch.Generator("cpu").manual_seed(42),
           callback_on_step_end=tick, output_type="latent").images
report["denoise_s"] = round(time.time() - t0)
report["step_times"] = step_times
report["nan_steps"] = nan_steps
report["latent_nan"] = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
say(f"latents NaN: {report['latent_nan']}  in {report['denoise_s']}s")

# Free the transformer and encoder before the VAE work; offload hooks leave
# them resident otherwise and the fp32 VAE needs the room.
pipe.maybe_free_model_hooks()
torch.cuda.empty_cache()

scaling = pipe.vae.config.scaling_factor


def decode(device, dtype, tag):
    """Decode the SAME latents and report whether anything survived."""
    try:
        t0 = time.time()
        vae = pipe.vae.to(device=device, dtype=dtype)
        with torch.no_grad():
            img = vae.decode((lat.to(device=device, dtype=dtype) / scaling)).sample
        arr = img.float().cpu().numpy()
        st = {"s": round(time.time() - t0, 1), "nan": bool(np.isnan(arr).any()),
              "min": round(float(np.nanmin(arr)), 3), "max": round(float(np.nanmax(arr)), 3),
              "std": round(float(np.nanstd(arr)), 4)}
        report[tag] = st
        say(f"  {tag}: {st}")
        a = np.nan_to_num(arr[0].transpose(1, 2, 0))
        Image.fromarray(((a.clip(-1, 1) + 1) * 127.5).astype("uint8")).save(OUT / f"{tag}.png")
    except Exception as e:
        say(f"  {tag} failed: {str(e)[:200]}")
        report[tag] = {"error": str(e)[:200]}
    torch.cuda.empty_cache()


say("--- decoding the same latents three ways ---")
decode("cuda", torch.float16, "vae_cuda_fp16")
decode("cuda", torch.float32, "vae_cuda_fp32")
decode("cpu", torch.float32, "vae_cpu_fp32")

(OUT / "report.json").write_text(json.dumps(report, indent=2))
say("\n" + "=" * 70)
say(json.dumps(report, indent=2))
