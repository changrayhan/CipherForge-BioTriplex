#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the latest ClinVar VCF (GRCh38) from NCBI FTP into data/raw/."""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38"


def list_remote():
    req = urllib.request.Request(FTP_BASE, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    return sorted(set(re.findall(r"clinvar_\d{8}\.vcf\.gz", html)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--date", default="", help="explicit YYYYMMDD filename")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)

    names = [f"clinvar_{args.date}.vcf.gz"] if args.date else list_remote()
    if not names:
        sys.exit("Cannot list remote directory")
    name = names[-1]
    dest = out / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} exists ({dest.stat().st_size / 1e9:.2f} GB)")
        print(f"[path] {out}")
        return

    url = f"{FTP_BASE}/{name}"
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=900) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB ({done * 100 // total}%)", end="", flush=True)
    tmp.replace(dest)
    print(f"\n[ok] {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    print(f"[path] {out}")


if __name__ == "__main__":
    main()
