# Media worker

Image and video generation behind sd-api's contract, so Viceroy can talk to a
GPU it does not own. The laptop keeps Viceroy, the LLM, TTS and ASR; this runs
wherever the GPU is that week — Kaggle and Colab now, RunPod later — and the
only thing that changes between them is the URL.

## On Kaggle (T4 x2, 30h/week)

New notebook, GPU accelerator on, internet on, then one cell:

    !pip -q install huggingface_hub
    !git clone -q https://github.com/<you>/ai-video /kaggle/working/ai-video
    !python /kaggle/working/ai-video/worker/kaggle_setup.py

It installs the few packages the image lacks, pulls the weights, starts the
worker and opens a Cloudflare tunnel, then prints the URL and the exact
sqlite3 line to point Viceroy at it. Keep the cell running: when it ends, the
session and the tunnel go with it.

Weights are ~20GB on a cold session. Once a run works, save them into a
private Kaggle Dataset and set `WORKER_MODELS` to its mount path — a dataset
attaches instantly and does not spend the session clock.

## Anywhere else

    WORKER_MODELS=/path/to/models python worker/server.py

RunPod, Vast, a desktop with a 3090: same server, same contract. Point the
provider rows at it.

## What it serves

    POST   /v1/jobs            {kind: image|video, model, prompt, width, height, ...}
    GET    /v1/jobs/{id}       status, progress, result
    DELETE /v1/jobs/{id}       abandon
    POST   /v1/inputs          multipart "file" -> reference images / driving frames
    GET    /v1/outputs/{name}  the finished image or clip
    GET    /health             device, dtype, loaded model, queue depth

Models: `z-image` (Z-Image Turbo, photoreal, 8 steps), `qwen-image-2.1`
(reference-conditioned, keeps a face across shots), `wan` (Wan 2.2 TI2V-5B),
`ltx`. One job at a time and one model resident: a 16GB card holds exactly one
of these, so the worker evicts rather than thrashes.
