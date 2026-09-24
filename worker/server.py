"""Media worker — image and video generation behind sd-api's contract.

Runs where the GPU is (Kaggle, Colab, RunPod); Viceroy stays on the laptop and
talks to it over a tunnel. The routes are the ones Viceroy's sd-api client
already calls, so pointing a provider row at this worker's URL is the whole
integration:

    POST   /v1/jobs            {kind, model, prompt, width, height, ...} -> job
    GET    /v1/jobs/{id}                                                 -> job
    DELETE /v1/jobs/{id}       abandon a running job
    POST   /v1/inputs          multipart "file" -> {inputs:[{name}]}   (references)
    HEAD   /v1/inputs/{name}   does the host still hold it
    GET    /v1/outputs/{name}  the finished image or clip
    GET    /health

One job at a time, because there is one GPU: a second request queues rather
than fighting the first for VRAM. Models load on first use and the previous
one is evicted, since a 16GB card holds exactly one of these at a time.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(os.environ.get("WORKER_ROOT", "/kaggle/working/media"))
MODELS = Path(os.environ.get("WORKER_MODELS", str(ROOT / "models")))
INPUTS, OUTPUTS = ROOT / "inputs", ROOT / "outputs"
for d in (INPUTS, OUTPUTS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# Turing (T4) has no bfloat16; Ampere and later do. Picking the wrong one is a
# silent 10x slowdown on the T4 Kaggle usually hands out.
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class JobRequest(BaseModel):
    kind: str = "image"
    model: str = "z-image"
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 768
    height: int = 1024
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    sampler: str | None = None
    ref_images: list[str] = []
    increase_ref_index: bool = False
    # video only
    image: str | None = None       # an /v1/inputs name to animate
    num_frames: int = 49
    fps: int = 24


@dataclass
class Job:
    id: str
    request: JobRequest
    status: str = "queued"
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    cancelled: bool = False
    created: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id, "status": self.status, "progress": round(self.progress, 3),
            "step": self.step, "totalSteps": self.total_steps,
        }
        if self.result:
            out["result"] = self.result
        if self.error:
            out["error"] = self.error
        return out


JOBS: dict[str, Job] = {}
QUEUE: Queue[str] = Queue()
_loaded: tuple[str, Any] | None = None  # (model id, pipeline) — one at a time


def pipeline_for(model: str):
    """Load a pipeline, evicting whatever else is on the card.

    Assembled from quantized parts rather than `from_pretrained` on the full
    repo: these models ship fp32 (~33GB each), which neither a 16GB card nor a
    notebook's disk can take. The transformer comes from a GGUF single file and
    the text encoder from a GGUF too where transformers can read that
    architecture, which puts each model in the 8-9GB range.
    """
    global _loaded
    if _loaded and _loaded[0] == model:
        return _loaded[1]

    if _loaded:
        _loaded = None
        gc.collect()
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    from diffusers import GGUFQuantizationConfig

    quant = GGUFQuantizationConfig(compute_dtype=DTYPE)

    if model.startswith("z-image"):
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = MODELS / "z-image-turbo"
        pipe = ZImagePipeline(
            transformer=ZImageTransformer2DModel.from_single_file(
                str(next((base / "gguf").glob("*.gguf"))), config=str(base), subfolder="transformer",
                quantization_config=quant, torch_dtype=DTYPE),
            text_encoder=AutoModelForCausalLM.from_pretrained(
                str(base / "te"), gguf_file=next((base / "te").glob("*.gguf")).name, torch_dtype=DTYPE),
            tokenizer=AutoTokenizer.from_pretrained(str(base / "tokenizer")),
            vae=AutoencoderKL.from_pretrained(str(base), subfolder="vae", torch_dtype=DTYPE),
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(base), subfolder="scheduler"),
        )
    elif model.startswith("wan"):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
        from transformers import T5Config, UMT5EncoderModel

        base = MODELS / "wan2.2-ti2v-5b"
        te = UMT5EncoderModel.from_pretrained(
            str(base / "te"), gguf_file=next((base / "te").glob("*.gguf")).name,
            config=T5Config.from_pretrained(str(base / "text_encoder")), torch_dtype=DTYPE)
        # The VAE has to exist before the pipeline is built: its scale factor is
        # read at construction, and a missing one silently halves the latent grid.
        vae = AutoencoderKLWan.from_pretrained(str(base), subfolder="vae", torch_dtype=DTYPE)
        pipe = WanImageToVideoPipeline.from_pretrained(
            str(base), transformer=None, vae=vae, text_encoder=te,
            image_encoder=None, image_processor=None, torch_dtype=DTYPE)
        pipe.transformer = WanTransformer3DModel.from_single_file(
            str(next((base / "gguf").glob("*.gguf"))), config=str(base), subfolder="transformer",
            quantization_config=quant, torch_dtype=DTYPE)
    elif model.startswith("qwen-image"):
        # Full precision only: its encoder is Qwen3-VL, which transformers
        # cannot read from GGUF, so this one needs a card with room for it.
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(str(MODELS / "qwen-image-2.1"), torch_dtype=DTYPE)
    elif model.startswith("ltx"):
        from diffusers import LTXConditionPipeline

        pipe = LTXConditionPipeline.from_pretrained(str(MODELS / "ltx-0.9.8"), torch_dtype=DTYPE)
    else:
        raise ValueError(f"unknown model {model!r}")

    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    _loaded = (model, pipe)
    return pipe


def run_image(job: Job) -> dict[str, Any]:
    from diffusers.utils import load_image

    r = job.request
    pipe = pipeline_for(r.model)
    steps = r.steps or (8 if r.model.startswith("z-image") else 20)
    cfg = r.cfg_scale if r.cfg_scale is not None else (1.0 if r.model.startswith("z-image") else 4.0)
    job.total_steps = steps

    def tick(_pipe, i, _t, kw):
        job.step, job.progress = i + 1, (i + 1) / steps
        if job.cancelled:
            raise RuntimeError("cancelled")
        return kw

    kwargs: dict[str, Any] = dict(
        prompt=r.prompt, negative_prompt=r.negative_prompt or None,
        width=r.width, height=r.height, num_inference_steps=steps, true_cfg_scale=cfg,
        generator=torch.Generator(device="cpu").manual_seed(r.seed if r.seed is not None else 42),
        callback_on_step_end=tick,
    )
    if r.ref_images:
        kwargs["image"] = [load_image(str(INPUTS / n)) for n in r.ref_images]

    out = pipe(**kwargs).images[0]
    name = f"{job.id}.png"
    out.save(OUTPUTS / name)
    return {"image_url": f"/v1/outputs/{name}", "image_path": str(OUTPUTS / name)}


def run_video(job: Job) -> dict[str, Any]:
    from diffusers.utils import export_to_video, load_image

    r = job.request
    pipe = pipeline_for(r.model)
    steps = r.steps or (6 if r.model.startswith("wan") else 8)
    job.total_steps = steps
    frames = max(5, round((r.num_frames - 1) / 4) * 4 + 1)

    def tick(_pipe, i, _t, kw):
        job.step, job.progress = i + 1, (i + 1) / steps
        if job.cancelled:
            raise RuntimeError("cancelled")
        return kw

    first = load_image(str(INPUTS / r.image)) if r.image else None
    out = pipe(
        image=first, prompt=r.prompt, negative_prompt=r.negative_prompt or None,
        height=r.height, width=r.width, num_frames=frames, num_inference_steps=steps,
        guidance_scale=r.cfg_scale if r.cfg_scale is not None else 1.0, output_type="pil",
        generator=torch.Generator(device="cpu").manual_seed(r.seed if r.seed is not None else 42),
        callback_on_step_end=tick,
    ).frames[0]

    name = f"{job.id}.mp4"
    export_to_video(out, str(OUTPUTS / name), fps=r.fps, quality=9)
    return {"video_url": f"/v1/outputs/{name}", "video_path": str(OUTPUTS / name)}


def worker_loop() -> None:
    while True:
        job = JOBS[QUEUE.get()]
        if job.cancelled:
            continue
        job.status = "running"
        started = time.time()
        try:
            job.result = run_video(job) if job.request.kind == "video" else run_image(job)
            job.result["metadata"] = {"duration_ms": int((time.time() - started) * 1000),
                                      "model": job.request.model, "device": DEVICE, "dtype": str(DTYPE)}
            job.status, job.progress = "completed", 1.0
        except Exception as exc:  # a failed job must not take the worker down
            job.status = "failed"
            job.error = {"code": "GENERATION_FAILED", "message": str(exc)[:500]}
        finally:
            QUEUE.task_done()


app = FastAPI(title="media worker")
threading.Thread(target=worker_loop, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True, "device": DEVICE, "dtype": str(DTYPE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "loaded": _loaded[0] if _loaded else None,
        "queued": QUEUE.qsize(),
        "models": sorted(p.name for p in MODELS.iterdir() if p.is_dir()),
    }


@app.post("/v1/jobs")
def create_job(request: JobRequest) -> dict[str, Any]:
    job = Job(id=str(uuid.uuid4()), request=request)
    JOBS[job.id] = job
    QUEUE.put(job.id)
    return job.public()


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id].public()


@app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str) -> dict[str, Any]:
    if job_id in JOBS:
        JOBS[job_id].cancelled = True
    return {"ok": True}


@app.post("/v1/inputs")
async def upload_input(file: UploadFile) -> dict[str, Any]:
    name = f"{uuid.uuid4().hex[:12]}-{Path(file.filename or 'ref.png').name}"
    (INPUTS / name).write_bytes(await file.read())
    return {"inputs": [{"name": name}]}


@app.api_route("/v1/inputs/{name}", methods=["GET", "HEAD"])
def get_input(name: str):
    path = INPUTS / Path(name).name
    if not path.exists():
        raise HTTPException(404, "no such input")
    return FileResponse(path)


@app.get("/v1/outputs/{name}")
def get_output(name: str):
    path = OUTPUTS / Path(name).name
    if not path.exists():
        raise HTTPException(404, "no such output")
    return FileResponse(path)


@app.exception_handler(HTTPException)
def http_error(_request: Request, exc: HTTPException):
    return JSONResponse({"error": {"code": exc.status_code, "message": exc.detail}}, status_code=exc.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")
