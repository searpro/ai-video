"""One shot, end to end on Kaggle: Z-Image still -> Wan clip, with real timings.

What the component probes established:
  - Z-Image needs fp32 on Turing (fp16 overflows); 215s for a 768x1024 still.
  - Wan 2.2 is fp16-clean: no NaN anywhere, 92s for six steps at 704x1280x49.
  - Of Wan's 506s, only 92s was denoising — 414s went to the fp32 VAE decode.
  - Host RAM (~29GB) is tighter than VRAM and kills runs silently, so a text
    encoder is used and freed rather than held.

The open question is quality, and the last run could not answer it: the driving
frame was synthetic gradient-and-noise, which is fine for numerics but tells
you nothing about whether the clip looks like anything. So this generates a
real still first and animates that.

It also exercises the thing production actually does and no probe has yet: two
large models in one process, the first fully evicted before the second loads.
"""

import gc
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

OUT = Path("/kaggle/working")
MODELS = Path("/kaggle/temp/media/models")
W, H, FRAMES, STEPS = 704, 1280, 49, 6
ENCODER_GPU = 1 if torch.cuda.device_count() > 1 else 0
report: dict = {}


def say(*a):
    print(*a, flush=True)


def ram_gb():
    with open("/proc/meminfo") as f:
        m = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return round((m["MemTotal"] - m["MemAvailable"]) / 1e6, 1)


def vram_gb():
    return round(torch.cuda.memory_allocated() / 1e9, 1)


def mark(tag):
    say(f"   [{tag}] host RAM {ram_gb()}GB used, VRAM {vram_gb()}GB")


from huggingface_hub import snapshot_download

for repo, sub, pats in [
    ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo/gguf", ["z-image-turbo-Q6_K.gguf"]),
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "z-image-turbo/te", ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"]),
    ("Tongyi-MAI/Z-Image-Turbo", "z-image-turbo",
     ["model_index.json", "vae/*", "scheduler/*", "transformer/config.json", "tokenizer/*", "text_encoder/config.json"]),
    ("hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF", "wan2.2-ti2v-5b/gguf", ["Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf"]),
    ("city96/umt5-xxl-encoder-gguf", "wan2.2-ti2v-5b/te", ["umt5-xxl-encoder-Q5_K_M.gguf"]),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b",
     ["model_index.json", "vae/*", "scheduler/*", "transformer/config.json", "tokenizer/*", "text_encoder/config.json"]),
]:
    snapshot_download(repo_id=repo, local_dir=str(MODELS / sub), allow_patterns=pats, max_workers=8)

STILL = ("a village baker in her forties, flour dusted on her apron, standing at a wooden "
         "counter in a stone bakery, warm morning light through a tall window, "
         "photorealistic, shallow depth of field, 35mm")
MOTION = ("she kneads the dough, hands pressing and folding, shoulders shifting with the "
          "effort, dust of flour in the sunlight, slow camera push in, cinematic")

# ------------------------------------------------------------ 1. the still
from diffusers import (AutoencoderKL, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler,
                       GGUFQuantizationConfig, WanImageToVideoPipeline, WanTransformer3DModel,
                       ZImagePipeline, ZImageTransformer2DModel)
from diffusers.utils import export_to_video
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Config, UMT5EncoderModel

ZB = MODELS / "z-image-turbo"


def make_still():
    """Generate the driving frame, leaving nothing behind.

    Everything lives in this function's scope on purpose. The previous attempt
    kept the pipeline alive through a module-level wrapper that closed over
    `zpipe.encode_prompt` — a bound method, so the whole pipeline stayed
    reachable — and `del zpipe` released 5GB of 19GB. umt5 then pushed the
    process past the host RAM ceiling and it was killed with no traceback.
    Accelerate's offload hooks hold references of their own, so those come off
    explicitly too.
    """
    pipe = ZImagePipeline(
        transformer=ZImageTransformer2DModel.from_single_file(
            str(next((ZB / "gguf").glob("*.gguf"))), config=str(ZB), subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
            dtype=torch.float32),
        text_encoder=AutoModelForCausalLM.from_pretrained(
            str(ZB / "te"), gguf_file=next((ZB / "te").glob("*.gguf")).name, dtype=torch.float16),
        tokenizer=AutoTokenizer.from_pretrained(str(ZB / "tokenizer")),
        vae=AutoencoderKL.from_pretrained(str(ZB), subfolder="vae", dtype=torch.float32),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(ZB), subfolder="scheduler"),
    )

    # fp16 encoder, fp32 transformer: bridge on the way out.
    inner = pipe.encode_prompt

    def encode_fp32(*a, **k):
        out = inner(*a, **k)
        seq = out if isinstance(out, (tuple, list)) else (out,)
        cast = tuple(x.to(torch.float32) if torch.is_tensor(x) else x for x in seq)
        return cast if isinstance(out, (tuple, list)) else cast[0]

    pipe.encode_prompt = encode_fp32
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    t = time.time()
    out = pipe(prompt=STILL, width=W, height=H, num_inference_steps=8, guidance_scale=1.0,
               generator=torch.Generator("cpu").manual_seed(7)).images[0]
    report["still_s"] = round(time.time() - t)

    pipe.maybe_free_model_hooks()
    pipe.encode_prompt = inner          # drop the closure's hold on the pipeline
    for name in ("transformer", "text_encoder", "vae", "tokenizer", "scheduler"):
        setattr(pipe, name, None)
    return out.copy()


say("=== Z-Image (fp32 transformer, fp16 text encoder) ===")
mark("start")
still = make_still()
still.save(OUT / "still.png")
_a = np.array(still)
report["still_stats"] = {"mean": round(float(_a.mean()), 1), "std": round(float(_a.std()), 1)}
say(f"  still in {report['still_s']}s  {report['still_stats']}")
mark("after z-image")

gc.collect()
torch.cuda.empty_cache()
# CPython holds freed arenas; hand them back or the next model sees a full host.
try:
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception as exc:
    say("  malloc_trim unavailable:", exc)
mark("after evicting z-image")

# ------------------------------------------------------------ 2. the clip
WB = MODELS / "wan2.2-ti2v-5b"
say("=== Wan 2.2 (fp16 transformer, fp32 VAE) ===")
t0 = time.time()
te = UMT5EncoderModel.from_pretrained(
    str(WB / "te"), gguf_file=next((WB / "te").glob("*.gguf")).name,
    config=T5Config.from_pretrained(str(WB / "text_encoder")), dtype=torch.float16,
    low_cpu_mem_usage=True, device_map={"": f"cuda:{ENCODER_GPU}"})
wtok = AutoTokenizer.from_pretrained(str(WB / "tokenizer"))
ids = wtok(MOTION, padding="max_length", max_length=512, truncation=True,
           return_tensors="pt").to(f"cuda:{ENCODER_GPU}")
with torch.no_grad():
    embeds = te(ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state
embeds = embeds.to("cuda:0", torch.float16)
report["wan_encode_s"] = round(time.time() - t0)
say(f"  prompt encoded in {report['wan_encode_s']}s")
del te
gc.collect()
torch.cuda.empty_cache()
mark("after freeing umt5")

t0 = time.time()
vae = AutoencoderKLWan.from_pretrained(str(WB), subfolder="vae", dtype=torch.float32)
wpipe = WanImageToVideoPipeline.from_pretrained(
    str(WB), transformer=None, vae=vae, text_encoder=None, tokenizer=None,
    image_encoder=None, image_processor=None, dtype=torch.float16)
wpipe.transformer = WanTransformer3DModel.from_single_file(
    str(next((WB / "gguf").glob("*.gguf"))), config=str(WB), subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16), dtype=torch.float16)
wpipe.enable_model_cpu_offload()
if hasattr(wpipe.vae, "enable_tiling"):
    wpipe.vae.enable_tiling()
wpipe.set_progress_bar_config(disable=True)
say(f"  built in {round(time.time()-t0)}s")
mark("wan loaded")

steps_t, last = [], [time.time()]


def tick(_p, i, _t, kw):
    steps_t.append(round(time.time() - last[0], 1)); last[0] = time.time()
    return kw


t0 = time.time()
frames = wpipe(image=still, prompt_embeds=embeds, width=W, height=H, num_frames=FRAMES,
               num_inference_steps=STEPS, guidance_scale=1.0, output_type="pil",
               generator=torch.Generator("cpu").manual_seed(7),
               callback_on_step_end=tick).frames[0]
report["wan_total_s"] = round(time.time() - t0)
report["wan_step_times"] = steps_t
report["wan_denoise_s"] = round(sum(steps_t))
report["wan_decode_s"] = report["wan_total_s"] - report["wan_denoise_s"]
export_to_video(frames, str(OUT / "clip.mp4"), fps=24, quality=9)

# Does it actually move, and does it hold up? A clip that collapses to mush
# still reports "no NaN", so measure the frames themselves.
stats = [{"i": i, "mean": round(float(np.array(f).mean()), 1),
          "std": round(float(np.array(f).std()), 1)}
         for i, f in enumerate(frames) if i % 12 == 0 or i == len(frames) - 1]
report["frame_stats"] = stats
diffs = [round(float(np.abs(np.array(frames[i], dtype=np.int16)
                            - np.array(frames[i - 1], dtype=np.int16)).mean()), 2)
         for i in range(1, len(frames))]
report["mean_abs_frame_delta"] = round(sum(diffs) / len(diffs), 2)
report["frame_delta_range"] = [min(diffs), max(diffs)]
for i in (0, len(frames) // 2, len(frames) - 1):
    frames[i].save(OUT / f"frame_{i:02d}.png")

say(f"  clip in {report['wan_total_s']}s "
    f"(denoise {report['wan_denoise_s']}s, decode {report['wan_decode_s']}s)")
say("  frame stats:", stats)
say("  mean abs frame-to-frame delta:", report["mean_abs_frame_delta"], report["frame_delta_range"])
mark("done")

report["totals"] = {"still_s": report["still_s"], "clip_s": report["wan_total_s"],
                    "shot_s": report["still_s"] + report["wan_encode_s"] + report["wan_total_s"]}
(OUT / "report.json").write_text(json.dumps(report, indent=2))
say("\n" + "=" * 70)
say(json.dumps(report, indent=2)[:3000])
