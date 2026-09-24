#!/usr/bin/env python3
"""Render ONE short shot and show you what you got, for tuning prompts.

    uv run shot.py --image images/baker.png --prompt "she kneads the dough, ..."
    uv run shot.py --image a.png --prompt "..." --steps 4 6 --size 512 640

Whole videos are the wrong unit for iterating: a 30s piece is hours, and almost
every question you actually have — does it follow the prompt, does the face
hold, does anything move — is answerable from two seconds. This renders one
shot at a review-grade size, times it, and writes a contact sheet so you can
judge it at a glance instead of scrubbing an mp4.

Pass several --steps or --size values and it renders the cross product as
variants and puts them on one sheet, which is the only honest way to tell
whether a setting helped.

It shells out to run_wan22.py rather than re-implementing it: that script owns
the model, and a second copy of the loading logic would drift.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
OUT = HERE / "out" / "shots"


def frames_of(path: Path) -> np.ndarray:
    # imageio-ffmpeg, not pyav: ffmpeg is already a dependency here because the
    # render scripts write video with it, and pyav is a second binary wheel for
    # no gain.
    with iio.get_reader(str(path), format="FFMPEG") as r:
        return np.stack([f for f in r])


def measure(path: Path) -> dict:
    """Numbers that catch the two failure modes worth catching automatically.

    A clip can be perfectly free of NaN and still be useless: either nothing
    moves (an expensive still) or it dissolves partway through. Frame-to-frame
    delta catches the first; comparing detail at the start against the end
    catches the second, which is the degradation seen on longer stitches.
    """
    f = frames_of(path).astype(np.int16)
    deltas = [float(np.abs(f[i] - f[i - 1]).mean()) for i in range(1, len(f))]
    # Laplacian-ish detail proxy: mean gradient magnitude per frame.
    def detail(x):
        g = np.asarray(x, dtype=np.float32).mean(axis=2)
        return float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)

    head, tail = detail(f[1]), detail(f[-1])
    return {
        "frames": int(len(f)),
        "motion": round(sum(deltas) / len(deltas), 2),
        "motion_min": round(min(deltas), 2),
        "motion_max": round(max(deltas), 2),
        "detail_start": round(head, 2),
        "detail_end": round(tail, 2),
        # <1 means the clip is losing detail as it runs; that is the degradation
        # that shows up as mush at the end of a shot.
        "detail_ratio": round(tail / head, 2) if head else None,
    }


def sheet(clips: list[tuple[str, Path, dict]], dest: Path) -> None:
    """One row per variant: first, middle and last frame, labelled."""
    rows = []
    for label, clip, m in clips:
        f = frames_of(clip)
        picks = [f[0], f[len(f) // 2], f[-1]]
        h = 360
        imgs = [Image.fromarray(p).resize((int(p.shape[1] * h / p.shape[0]), h)) for p in picks]
        row = Image.new("RGB", (sum(i.width for i in imgs), h + 28), "black")
        x = 0
        for i in imgs:
            row.paste(i, (x, 28))
            x += i.width
        ImageDraw.Draw(row).text(
            (6, 7), f"{label}   motion {m['motion']}  detail {m['detail_start']}->{m['detail_end']}",
            fill="white")
        rows.append(row)
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows)), "black")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(dest)


def render(args, size: int, steps: int, tag: str) -> tuple[Path, float]:
    clip = OUT / f"{tag}.mp4"
    cmd = [
        "uv", "run", "run_wan22.py", "i2v",
        "--prompt", args.prompt, "--image", args.image,
        "--size", str(size), "--steps", str(steps),
        "--frames", str(args.frames), "--seed", str(args.seed),
        "--out", str(clip),
    ]
    if args.negative:
        cmd += ["--negative", args.negative]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE)
    if proc.returncode:
        sys.exit(f"render failed for {tag}")
    return clip, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="the shot's first frame")
    p.add_argument("--prompt", required=True, help="what happens in the shot")
    p.add_argument("--negative", default="")
    # 2 seconds: long enough to read as a shot, short enough to iterate on.
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--size", type=int, nargs="+", default=[512],
                   help="short side; faces mangle below about 512")
    p.add_argument("--steps", type=int, nargs="+", default=[6])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="shot")
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for size, steps in itertools.product(a.size, a.steps):
        tag = f"{a.name}-s{size}-st{steps}"
        clip, secs = render(a, size, steps, tag)
        m = measure(clip)
        m |= {"seconds": round(secs), "size": size, "steps": steps}
        results.append((f"{size}px {steps}st {round(secs)}s", clip, m))
        print(f"  -> {clip.name}  {json.dumps(m)}", flush=True)

    dest = OUT / f"{a.name}-sheet.png"
    sheet(results, dest)
    (OUT / f"{a.name}-report.json").write_text(
        json.dumps([{"label": l, "clip": str(c), **m} for l, c, m in results], indent=2))

    print(f"\ncontact sheet: {dest}")
    for label, _, m in results:
        flag = ""
        if m["motion"] < 1.0:
            flag = "  <- barely moves"
        elif m["detail_ratio"] and m["detail_ratio"] < 0.8:
            flag = "  <- losing detail toward the end"
        print(f"  {label}: motion {m['motion']}, detail ratio {m['detail_ratio']}{flag}")


if __name__ == "__main__":
    main()
