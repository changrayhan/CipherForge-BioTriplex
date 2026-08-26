#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download TinyLlama-1.1B-Chat-v1.0 into the HF cache.

China mirror (recommended):  export HF_ENDPOINT=https://hf-mirror.com
Then run: python baseline/scripts/download_model.py
Afterwards set CF_MODEL_PATH to the printed snapshot path:
    export CF_MODEL_PATH=$(python cipherforge/tools/resolve_model_path.py)
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    path = snapshot_download(args.repo)
    snap = Path(path)
    if not (snap / "config.json").exists():
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        name = "models--" + args.repo.replace("/", "--")
        hits = sorted((cache / "hub" / name / "snapshots").glob("*/config.json"))
        if hits:
            snap = hits[0].parent
    print("[ok] snapshot:", snap)
    print("export CF_MODEL_PATH=" + str(snap))
    print(
        "model id for transformers: " + args.repo,
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
