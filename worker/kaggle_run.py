#!/usr/bin/env -S uv run --quiet --with requests --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Run a script on a Kaggle GPU and bring the results back, without a browser.

    uv run worker/kaggle_run.py probe.py               # push, wait, print the log
    uv run worker/kaggle_run.py probe.py --out ./res   # also download the output files

Kaggle's API runs notebooks in *batch*: push a script, it queues, runs to
completion on a real GPU, and saves whatever it wrote to /kaggle/working. That
is the wrong shape for serving Viceroy, but exactly the right shape for an
experiment — no tunnel, no session to babysit, no URL to copy by hand.

Credentials come from ~/.kaggle/kaggle.json and are never printed.

One kernel slug is reused per name, so repeated runs version the same notebook
instead of littering the account.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://www.kaggle.com/api/v1"
CREDS = Path.home() / ".kaggle" / "kaggle.json"

# Kaggle rejects an unknown accelerator outright, so keep to the names it
# publishes. The dual T4 is the interesting one: 2x15GB, no NVLink.
ACCELERATORS = {"none": None, "t4": "nvidiaTeslaT4", "t4x2": "nvidiaTeslaT4x2", "p100": "nvidiaTeslaP100"}

DONE = {"complete", "error", "cancelled", "cancelAcknowledged"}


def auth() -> tuple[str, dict[str, str]]:
    if not CREDS.exists():
        sys.exit(f"no credentials at {CREDS} — download kaggle.json from kaggle.com/settings")
    c = json.loads(CREDS.read_text())
    token = base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    return c["username"], {"Authorization": f"Basic {token}"}


def push(user: str, headers: dict[str, str], slug: str, source: str, accel: str, internet: bool) -> None:
    body = {
        # "id" is Kaggle's numeric kernel id; the human ref goes in "slug".
        "slug": f"{user}/{slug}",
        "newTitle": slug,
        "text": source,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": accel != "none",
        "enableInternet": internet,
        "datasetDataSources": [],
        "competitionDataSources": [],
        "kernelDataSources": [],
        "modelDataSources": [],
        "categoryIds": [],
    }
    if ACCELERATORS[accel]:
        body["acceleratorType"] = ACCELERATORS[accel]
    r = requests.post(f"{API}/kernels/push", headers=headers, json=body, timeout=60)
    r.raise_for_status()
    out = r.json()
    if out.get("error"):
        sys.exit(f"push rejected: {out['error']}")
    print(f"pushed {out.get('ref')} v{out.get('versionNumber')}  {out.get('url')}", flush=True)


def wait(user: str, headers: dict[str, str], slug: str, timeout: int) -> str:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = requests.get(f"{API}/kernels/status", headers=headers,
                         params={"userName": user, "kernelSlug": slug}, timeout=30)
        s = r.json()
        status = s.get("status", "unknown")
        if status != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {status}"
                  + (f" — {s['failureMessage']}" if s.get("failureMessage") else ""), flush=True)
            last = status
        if status in DONE:
            return status
        time.sleep(15)
    return "timeout"


def fetch(user: str, headers: dict[str, str], slug: str, out_dir: Path | None) -> None:
    r = requests.get(f"{API}/kernels/output", headers=headers,
                     params={"userName": user, "kernelSlug": slug}, timeout=120)
    r.raise_for_status()
    data = r.json()

    log = data.get("log")
    if log:
        # The log arrives as a JSON array of {stream, time, data} records.
        try:
            for rec in json.loads(log) if isinstance(log, str) else log:
                sys.stdout.write(rec.get("data", ""))
        except (ValueError, TypeError):
            print(log)
    print(flush=True)

    files = data.get("files", [])
    print(f"--- {len(files)} output file(s) ---")
    for f in files:
        print("  ", f.get("fileName"), f.get("url"))
    if out_dir and files:
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest = out_dir / f["fileName"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(requests.get(f["url"], headers=headers, timeout=300).content)
            print("   saved", dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="local python file to run on Kaggle")
    ap.add_argument("--name", default=None, help="kernel slug (default: the script's stem)")
    ap.add_argument("--accel", default="t4x2", choices=sorted(ACCELERATORS))
    ap.add_argument("--no-internet", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds to wait")
    ap.add_argument("--out", type=Path, default=None, help="directory for output files")
    a = ap.parse_args()

    user, headers = auth()
    slug = a.name or Path(a.script).stem.replace("_", "-")
    push(user, headers, slug, Path(a.script).read_text(), a.accel, not a.no_internet)
    status = wait(user, headers, slug, a.timeout)
    print(f"--- {status} ---", flush=True)
    fetch(user, headers, slug, a.out)
    sys.exit(0 if status == "complete" else 1)


if __name__ == "__main__":
    main()
