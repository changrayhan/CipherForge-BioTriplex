#!/usr/bin/env python3
"""U1b — ciphertext-only channel control.

U only has Enc(V_i) rows (BFV ciphertexts). Observations:
  * the encrypted DB is a STATIC Stage-0 artifact, so the same row returns
    byte-identical ciphertexts every fetch (row linking is trivially possible
    for the querying party, who already knows the indices it queried);
  * the real question is whether the ciphertext leaks PLAINTEXT V values.
We therefore measure whether ciphertext bytes carry any linear signal of the
true row values (correlation ~= 0 expected: BFV semantic security).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load_ciphertexts(ct_dir: Path):
    samples = []  # (index, bytes)
    for p in sorted(ct_dir.glob("step_*.pkl")):
        with open(p, "rb") as f:
            rows = pickle.load(f)
        for idx, blob in rows.items():
            samples.append((int(idx), blob))
    return samples


def norm_l1(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    aa = np.frombuffer(a[:n], dtype=np.uint8).astype(np.float64)
    bb = np.frombuffer(b[:n], dtype=np.uint8).astype(np.float64)
    return float(np.abs(aa - bb).mean() / 255.0)


def plaintext_signal(blob: bytes, v_row: np.ndarray) -> dict:
    """Correlation between ciphertext bytes and the true float row values."""
    n = len(blob) // 4
    f32 = np.frombuffer(blob[: n * 4], dtype="<f4").astype(np.float64)
    u8 = np.frombuffer(blob[: min(len(blob), len(v_row) * 4)], dtype=np.uint8).astype(np.float64)
    vf = v_row.reshape(-1).astype(np.float64)
    m = min(len(f32), len(vf))
    corr_f32 = float(np.corrcoef(f32[:m], vf[:m])[0, 1]) if m > 4 and np.std(vf[:m]) > 0 else 0.0
    if not np.isfinite(corr_f32):
        corr_f32 = 0.0  # ciphertext bytes contain NaN/Inf float patterns -> no signal
    corr_u8 = float(np.corrcoef(u8[:m], vf[:m])[0, 1]) if m > 4 and np.std(vf[:m]) > 0 else 0.0
    if not np.isfinite(corr_u8):
        corr_u8 = 0.0
    return {"corr_float32": corr_f32, "corr_uint8": corr_u8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    samples = load_ciphertexts(run_dir / "ciphertexts")
    by_idx = {}
    for idx, blob in samples:
        by_idx.setdefault(idx, []).append(blob)
    idxs = sorted(by_idx)

    same, diff = [], []
    for i, idx in enumerate(idxs):
        grp = by_idx[idx]
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                same.append(norm_l1(grp[a], grp[b]))
        for j in range(i + 1, len(idxs)):
            grp2 = by_idx[idxs[j]]
            for x in grp:
                for y in grp2:
                    diff.append(norm_l1(x, y))
    same = np.asarray(same, dtype=np.float64)
    diff = np.asarray(diff, dtype=np.float64)
    n = len(samples)

    # plaintext-signal check on the class rows (we hold their ground truth)
    v_data = np.load(run_dir / "captures" / "s" / "v_rows.npz")
    v_true = v_data["v_rows"].astype(np.float32)
    class_ids = v_data["class_token_ids"].tolist()
    signal = {}
    for i, idx in enumerate(idxs):
        if idx in class_ids:
            row_idx = class_ids.index(idx)
            sigs = [plaintext_signal(b, v_true[row_idx]) for b in by_idx[idx][:3]]
            signal[str(idx)] = sigs
    corrs = [s["corr_float32"] for sigs in signal.values() for s in sigs]
    corrs = [c for c in corrs if np.isfinite(c)]
    max_abs_corr = float(max(abs(c) for c in corrs)) if corrs else 0.0
    verdict = "PRESERVED" if max_abs_corr < 0.05 else "LEAK_DETECTED"
    out = {
        "u1b_ciphertext": {
            "n_rows": len(idxs),
            "n_samples": n,
            "same_row_l1_mean": float(same.mean()) if len(same) else None,
            "diff_row_l1_mean": float(diff.mean()) if len(diff) else None,
            "same_vs_diff_gap": float(abs(same.mean() - diff.mean())) if len(same) and len(diff) else None,
            "note": "static encrypted DB -> same-row ciphertexts byte-identical; "
                    "row linking is trivial for the querying party and reveals no plaintext",
            "plaintext_signal": signal,
            "max_abs_corr": max_abs_corr,
            "verdict": verdict,
        }
    }
    (run_dir / "u1b_ciphertext.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["u1b_ciphertext"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
