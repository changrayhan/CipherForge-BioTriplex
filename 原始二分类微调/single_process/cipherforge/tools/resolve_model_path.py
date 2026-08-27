#!/usr/bin/env python3
"""Print the local HF-cache snapshot directory for a model repo id.

Usage:
    export CF_MODEL_PATH=$(python cipherforge/tools/resolve_model_path.py \
        TinyLlama/TinyLlama-1.1B-Chat-v1.0)
"""
import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", nargs="?", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = ap.parse_args()
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    name = "models--" + args.repo.replace("/", "--")
    snaps = cache / "hub" / name / "snapshots"
    if not snaps.exists():
        print(
            f"[error] no HF cache snapshot for {args.repo} under {snaps}\n"
            "run:  HF_ENDPOINT=https://hf-mirror.com python baseline/scripts/download_model.py",
            file=sys.stderr,
        )
        sys.exit(1)
    for d in sorted(snaps.iterdir()):
        if (d / "config.json").exists():
            print(d)
            return
    print(f"[error] snapshot dir exists but config.json not found in {snaps}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
