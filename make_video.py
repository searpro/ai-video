"""Shot-list -> stitched video. Wan2.2 TI2V-5B Turbo (GGUF), Apple Silicon.

  uv run make_video.py shots.json [--out story.mp4] [--upscale 2] [--dry-run]

shots.json (this is the contract Viceroy generates):
{
  "fps": 24, "size": 704,            # short side; model likes 1280x704
  "audio": "audio/voice.wav",        # optional, muxed over the finished cut
  "shots": [
    {"image": "chars/hero.png", "prompt": "she turns and smiles", "frames": 49},
    {"prompt": "she walks out of frame", "frames": 49, "continue": true},   # starts on the previous shot's last frame
    {"image": "chars/hero.png", "prompt": "she speaks to camera", "audio": "lines/hero_01.wav"}  # lip-synced shot
  ]
}
Models load once for the whole list: all prompts are encoded in one text-encoder pass, then the
transformer renders every shot, then one optional upscale pass, then ffmpeg concatenates.

A shot with "audio" is lip-synced with LatentSync (third_party/latentsync, its own venv) and its
frame count is derived from the audio length. "audio" at the top level instead lays one track
(music/VO) over the finished cut.

Quality notes (measured on this machine):
  * keep "size" >= 512 — at 384 the model mangles faces by the end of a shot
  * prefer a fresh "image" per shot; "continue" chains generation loss, so keep chained shots short
    and give them simple prompts (a camera move or a walk, not "he takes off his sunglasses")
  * --steps 6 is steadier than the turbo default of 4; cfg stays at 1
"""

import argparse
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image
from spandrel import ModelLoader
from transformers import T5Config, UMT5EncoderModel

BASE = "models/wan22-base"
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
       "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
       "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
MOD = 32

p = argparse.ArgumentParser()
p.add_argument("shots", help="shot list json")
p.add_argument("--out", default="story.mp4")
p.add_argument("--workdir", default="shots_out", help="where per-shot mp4s are written")
p.add_argument("--upscale", type=float, default=0, help="Real-ESRGAN factor for every shot (e.g. 2)")
p.add_argument("--quant", default="Q5_K_M")
p.add_argument("--steps", type=int, default=4)
p.add_argument("--size", type=int, help="override the shot list's short side (the model wants >=512; faces break below that)")
p.add_argument("--no-refresh", action="store_true",
               help="don't clean up the frame handed to a 'continue' shot (chaining then compounds blur/colour drift)")
p.add_argument("--no-lipsync", action="store_true", help="ignore per-shot audio (skip LatentSync)")
p.add_argument("--dry-run", action="store_true", help="validate the shot list and print the plan only")
args = p.parse_args()

LIPSYNC = Path("third_party/latentsync")


def audio_frames(path, fps):
    """Shot length from the audio: 4k+1 frames covering the whole clip."""
    out = subprocess.run([get_ffmpeg_exe(), "-i", path], capture_output=True, text=True).stderr
    h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    n = round((int(h) * 3600 + int(m) * 60 + float(sec)) * fps)
    return max(5, n + (1 - n % 4) % 4)


def lipsync(video, audio, out):
    """LatentSync (ByteDance) in its own venv: re-animates the mouth to match the audio."""
    subprocess.run(
        [".venv/bin/python", "-m", "scripts.inference",
         "--unet_config_path", "configs/unet/stage2.yaml",
         "--inference_ckpt_path", "checkpoints/v1.5/latentsync_unet.pt",
         "--inference_steps", "8", "--guidance_scale", "1.5", "--enable_deepcache",
         "--video_path", str(Path(video).resolve()), "--audio_path", str(Path(audio).resolve()),
         "--video_out_path", str(Path(out).resolve())],
        cwd=LIPSYNC, check=True, capture_output=True,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "LATENTSYNC_DTYPE": "bfloat16",
             "MPS_ATTENTION_BUDGET_MB": "256", "VAE_CHUNK_SIZE": "2"})


spec = json.loads(Path(args.shots).read_text())
shots, fps = spec["shots"], spec.get("fps", 24)
size = args.size or spec.get("size", 704)
if any(sh.get("audio") for sh in shots) and not args.no_lipsync and fps != 25:
    fps = 25  # LatentSync resamples to 25fps; mixing rates would desync the cut
    print("lip-sync in use -> fps forced to 25")
work = Path(args.workdir)
work.mkdir(exist_ok=True)

for i, s in enumerate(shots):  # validate up front: a 20-minute run should not die on shot 4
    if i == 0 and not s.get("image"):
        raise SystemExit("shot 0 needs an 'image' (nothing to continue from)")
    if s.get("image") and not Path(s["image"]).exists():
        raise SystemExit(f"shot {i}: missing image {s['image']}")
    if s.get("audio"):
        if not Path(s["audio"]).exists():
            raise SystemExit(f"shot {i}: missing audio {s['audio']}")
        s.setdefault("frames", audio_frames(s["audio"], fps))
    if s.get("frames", 49) % 4 != 1:
        raise SystemExit(f"shot {i}: frames must be 4k+1, got {s.get('frames')}")
if (a := spec.get("audio")) and not Path(a).exists():
    raise SystemExit(f"missing audio {a}")
print(f"{len(shots)} shots, {sum(s.get('frames', 49) for s in shots) / fps:.1f}s at {fps}fps")
if args.dry_run:
    for i, s in enumerate(shots):
        print(f"  {i}: {'continue' if s.get('continue') else s.get('image')} | {s.get('frames', 49)}f | {s['prompt'][:60]}")
    raise SystemExit(0)

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16
t0 = time.time()
free = lambda: (gc.collect(), torch.mps.empty_cache() if device == "mps" else None)

# 1) One text-encoder pass for every prompt in the list, then drop it (it is the memory hog).
te = UMT5EncoderModel.from_pretrained(
    "models/umt5-gguf", gguf_file="umt5-xxl-encoder-Q3_K_S.gguf",
    config=T5Config.from_pretrained(f"{BASE}/text_encoder"), torch_dtype=dtype).to(device)
# bf16 VAE: decode is the bottleneck on MPS (386s vs 1487s in fp32 for 49f), output stays stable
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=dtype)
pipe = WanImageToVideoPipeline.from_pretrained(
    BASE, transformer=None, vae=vae, text_encoder=te, image_encoder=None, image_processor=None, torch_dtype=dtype)
with torch.no_grad():
    embeds = [pipe.encode_prompt(s["prompt"], NEG, do_classifier_free_guidance=False, device=device)[0]
              for s in shots]
pipe.text_encoder = None
del te
free()
print(f"{len(shots)} prompts encoded in {time.time() - t0:.0f}s")

# 2) Render every shot with one load of the transformer + VAE.
pipe.transformer = WanTransformer3DModel.from_single_file(
    f"models/wan22-gguf/Wan2_2-TI2V-5B-Turbo-{args.quant}.gguf", config=BASE, subfolder="transformer",
    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype)
pipe.to(device)

def color_match(img, ref):
    """Match per-channel mean/std to the anchor image: stops colour/saturation drifting across cuts."""
    a, b = np.asarray(img).astype(np.float32), np.asarray(ref.resize(img.size)).astype(np.float32)
    for c in range(3):
        sa, sb = a[..., c].std(), b[..., c].std()
        a[..., c] = (a[..., c] - a[..., c].mean()) * (sb / max(sa, 1e-5)) + b[..., c].mean()
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def refresh(img):
    """Re-detail a handoff frame (upscale then back down) so chained shots don't accumulate softness."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].to(device, torch.float16) / 255
        y = (sr_model(x)[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y).resize(img.size, Image.LANCZOS)

sr_model = None if args.no_refresh else (
    ModelLoader().load_from_file("models/esrgan/RealESRGAN_x2.safetensors").model.to(device).eval().half())

clips, last, anchor = [], None, None
for i, s in enumerate(shots):
    if s.get("continue"):
        first = last if args.no_refresh else color_match(refresh(last), anchor)
    else:
        first = anchor = load_image(s["image"]).convert("RGB")
    r = first.width / first.height
    W, H = (size * r, size) if r >= 1 else (size, size / r)
    W, H = max(MOD, round(W / MOD) * MOD), max(MOD, round(H / MOD) * MOD)
    frames = pipe(image=first, prompt_embeds=embeds[i], height=H, width=W, num_frames=s.get("frames", 49),
                  num_inference_steps=args.steps, guidance_scale=1.0, output_type="pil",
                  generator=torch.Generator().manual_seed(s.get("seed", 42))).frames[0]
    clips.append(frames)
    last = frames[-1]
    print(f"shot {i} done ({len(frames)}f {W}x{H}) at {time.time() - t0:.0f}s")

# 3) Optional detail pass (Real-ESRGAN loads once for all shots).
if args.upscale:
    del pipe, sr_model
    free()
    sr = ModelLoader().load_from_file("models/esrgan/RealESRGAN_x4plus.safetensors").model.to(device).eval().half()
    with torch.no_grad():
        for frames in clips:
            w, h = frames[0].size
            w, h = round(w * args.upscale / 2) * 2, round(h * args.upscale / 2) * 2
            for j, fr in enumerate(frames):
                x = torch.from_numpy(np.asarray(fr)).permute(2, 0, 1)[None].to(device, torch.float16) / 255
                y = (sr(x)[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                frames[j] = Image.fromarray(y).resize((w, h), Image.LANCZOS)
    print(f"upscaled at {time.time() - t0:.0f}s")

# 4) Per-shot files: lip-sync the ones with audio, give the rest a silent track so concat matches.
W, H = clips[0][0].size
talking = [s.get("audio") for s in shots] if not args.no_lipsync else [None] * len(shots)
for i, frames in enumerate(clips):
    if frames[0].size != (W, H):
        frames = [f.resize((W, H), Image.LANCZOS) for f in frames]
    mute = work / f"shot_{i:02d}_v.mp4"
    export_to_video(frames, str(mute), fps=fps, quality=9)
    shot_mp4 = work / f"shot_{i:02d}.mp4"
    if talking[i]:
        lipsync(mute, talking[i], shot_mp4)
        print(f"shot {i} lip-synced at {time.time() - t0:.0f}s")
    elif any(talking):  # concat needs every part to have the same streams
        subprocess.run([get_ffmpeg_exe(), "-y", "-i", str(mute), "-f", "lavfi", "-i",
                        "anullsrc=r=16000:cl=mono", "-shortest", "-c:v", "copy", "-c:a", "aac",
                        str(shot_mp4)], check=True, capture_output=True)
    else:
        mute.replace(shot_mp4)
(work / "list.txt").write_text("".join(f"file 'shot_{i:02d}.mp4'\n" for i in range(len(clips))))

cmd = [get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt")]
if spec.get("audio"):  # one track over the whole cut
    cmd += ["-i", spec["audio"], "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
elif any(talking):
    cmd += ["-c:a", "aac"]
cmd += ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", args.out]
subprocess.run(cmd, check=True, capture_output=True)
print(f"saved {args.out} ({W}x{H}) in {time.time() - t0:.0f}s")
