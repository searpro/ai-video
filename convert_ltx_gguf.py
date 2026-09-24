"""One-off: turn ltxv-2b-0.9.8-distilled.safetensors into
  models/ltx-gguf/ltxv-2b-0.9.8-distilled-Q4_0.gguf   (transformer, 2D weights -> Q4_0, rest kept)
  models/ltx-base/vae/diffusion_pytorch_model.safetensors  (0.9.8 VAE, bf16, diffusers format)
No public GGUF exists for 0.9.8 2B; gguf-py can only write Q4_0/Q5_0/Q8_0 (K-quants need llama.cpp).

  uv run convert_ltx_gguf.py [--qtype Q4_0|Q5_0|Q8_0]
"""

import argparse
import os

import gguf
import torch
from diffusers import AutoencoderKLLTXVideo
from safetensors.torch import load_file

SRC = "models/_ltx-src/ltxv-2b-0.9.8-distilled.safetensors"
BASE = "models/ltx-base"

p = argparse.ArgumentParser()
p.add_argument("--qtype", default="Q4_0", choices=["Q4_0", "Q5_0", "Q8_0"])
args = p.parse_args()
qtype = gguf.GGMLQuantizationType[args.qtype]
out = f"models/ltx-gguf/ltxv-2b-0.9.8-distilled-{args.qtype}.gguf"
os.makedirs("models/ltx-gguf", exist_ok=True)

sd = load_file(SRC)

# Transformer -> GGUF. Keep the original key names; diffusers' single-file converter renames them.
w = gguf.GGUFWriter(out, arch="ltxv")
w.add_file_type(qtype)
for k, t in sd.items():
    if not k.startswith("model.diffusion_model."):
        continue
    x = t.float().numpy()
    # quantize only large Linear weights; norms, biases, scale_shift tables stay F32
    if k.endswith(".weight") and x.ndim == 2 and x.shape[1] % 32 == 0 and x.size > 4096:
        w.add_tensor(k, gguf.quants.quantize(x, qtype), raw_dtype=qtype)
    else:
        w.add_tensor(k, x)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")

# VAE -> diffusers format bf16 (the 0.9.8 VAE, not the older 0.9.5 one from the base repo).
vae = AutoencoderKLLTXVideo.from_single_file(
    {k: v for k, v in sd.items() if k.startswith("vae.")}, config=BASE, subfolder="vae", torch_dtype=torch.bfloat16,
)
vae.to(torch.bfloat16).save_pretrained(f"{BASE}/vae")
print("wrote VAE to", f"{BASE}/vae")
