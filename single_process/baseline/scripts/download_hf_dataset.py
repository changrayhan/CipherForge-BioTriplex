#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-download a Hugging Face dataset repo into data/external (mirror-friendly)."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--local_dir", default="")
    args = ap.parse_args()
    local = Path(args.local_dir) if args.local_dir else Path(__file__).resolve().parents[1] / "data" / "external" / args.repo.replace("/", "__")
    local.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(args.repo, repo_type="dataset", local_dir=str(local))
    print("[ok]", path)


if __name__ == "__main__":
    main()
