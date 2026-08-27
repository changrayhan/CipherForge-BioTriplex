#!/usr/bin/env python3
"""U1c — can U recover S's task head V from the composed oracle?

U only has (H_U -> z_cls) pairs (no h = f_M(H_U), no full logits).
  * blind attack: treat H_U as h and least-squares fit V rows;
  * joint fit: student trunk (same architecture, public base weights) +
    trainable head rows fitted to the oracle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.student import build_student, student_z  # noqa: E402
from common import load_config  # noqa: E402


def load_samples(run_dir: Path):
    samples = []
    for p in sorted((run_dir / "corpus").glob("corpus_q*.npz")):
        d = np.load(p)
        samples.append((
            d["h_u"].astype(np.float32),
            d["positions"].astype(np.int64),
            d["z_cls"].astype(np.float32),
        ))
    for p in sorted((run_dir / "captures" / "u").glob("eval_val_*.npz")):
        d = np.load(p)
        samples.append((
            d["h_u_full"].astype(np.float32),
            d["score_pos"].astype(np.int64),
            d["z_cls"].astype(np.float32),
        ))
    return samples


def blind_fit(samples, v_true):
    """Treat H_U at positions as h; least-squares recover V rows."""
    Xs, Ys = [], []
    for h, pos, z in samples:
        B, S, H = h.shape
        rows = h.reshape(B * S, H)[pos]  # (n,H)
        Xs.append(rows)
        Ys.append(z)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    v_hat, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (H,C)
    v_hat = v_hat.T  # (C,H)
    re = float(np.linalg.norm(v_hat - v_true) / np.linalg.norm(v_true))
    cos = float(np.mean([
        np.dot(v_hat[i], v_true[i]) / (np.linalg.norm(v_hat[i]) * np.linalg.norm(v_true[i]) + 1e-12)
        for i in range(v_true.shape[0])
    ]))
    return {"re_f": re, "row_cosine": cos, "X_rows": int(X.shape[0])}


def fit_student(model, head, opt, train, epochs, device):
    model.train()
    for _ in range(epochs):
        for h, pos, z in train:
            opt.zero_grad()
            h_t = torch.from_numpy(np.ascontiguousarray(h)).to(device)
            pos_t = torch.from_numpy(np.ascontiguousarray(pos)).to(device)
            z_t = torch.from_numpy(np.ascontiguousarray(z)).to(device)
            z_hat = student_z(model, head, h_t, pos_t)
            loss = F.mse_loss(z_hat, z_t)
            loss.backward()
            opt.step()


def functional_agreement(model, head, heldout, device):
    model.eval()
    preds, golds = [], []
    corr = []
    with torch.no_grad():
        for h, pos, z in heldout:
            h_t = torch.from_numpy(np.ascontiguousarray(h)).to(device)
            pos_t = torch.from_numpy(np.ascontiguousarray(pos)).to(device)
            z_hat = student_z(model, head, h_t, pos_t).cpu()
            p_hat = torch.softmax(z_hat, dim=-1)[:, 0]
            p_true = torch.softmax(torch.from_numpy(z), dim=-1)[:, 0]
            preds.append((p_hat > 0.5).numpy())
            golds.append((p_true > 0.5).numpy())
            if len(p_hat) and len(p_true):
                corr.append(float(np.corrcoef(p_hat.numpy(), p_true.numpy())[0, 1]))
    preds = np.concatenate(preds) if preds else np.zeros(0)
    golds = np.concatenate(golds) if golds else np.zeros(0)
    agree = float(np.mean(preds == golds)) if len(preds) else 0.0
    return agree, float(np.nanmean(corr)) if corr else 0.0


def head_metrics(head, v_true):
    v_hat = head.detach().float().cpu().numpy()
    re = float(np.linalg.norm(v_hat - v_true) / np.linalg.norm(v_true))
    cos = float(np.mean([
        np.dot(v_hat[i], v_true[i]) / (np.linalg.norm(v_hat[i]) * np.linalg.norm(v_true[i]) + 1e-12)
        for i in range(v_true.shape[0])
    ]))
    return {"re_f": re, "row_cosine": cos}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg", default="/root/CipherForge/CipherForge-ClinVar/three_party/coordinator/three_party_config.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(args.cfg)
    device = args.device if torch.cuda.is_available() else "cpu"

    v_data = np.load(run_dir / "captures" / "s" / "v_rows.npz")
    v_true = v_data["v_rows"].astype(np.float32)  # (C,H)
    samples = load_samples(run_dir)
    print(f"[u1c] samples={len(samples)} pairs="
          f"{sum(s[2].shape[0] for s in samples)}", flush=True)

    # ---- blind ----
    blind = blind_fit(samples, v_true)
    print(f"[u1c] blind RE_F={blind['re_f']:.4f} cosine={blind['row_cosine']:.4f}", flush=True)

    # ---- joint fit with query-complexity curve ----
    rng = np.random.default_rng(0)
    order = rng.permutation(len(samples))
    n_hold = max(1, int(len(samples) * 0.2))
    hold_idx = set(order[:n_hold].tolist())
    train_all = [s for i, s in enumerate(samples) if i not in hold_idx]
    heldout = [samples[i] for i in sorted(hold_idx)]
    n_q = len(train_all)

    curve = []
    for frac in (0.1, 0.3, 1.0):
        n_use = max(1, int(n_q * frac))
        train = train_all[:n_use]
        model, head = build_student(cfg, device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad] + [head], lr=1e-3)
        t0 = time.time()
        epochs = 2 if frac < 1.0 else 3
        fit_student(model, head, opt, train, epochs, device)
        hm = head_metrics(head, v_true)
        agree, corr = functional_agreement(model, head, heldout, device)
        curve.append({
            "queries": n_use, "epochs": epochs,
            "re_f_v": hm["re_f"], "row_cosine": hm["row_cosine"],
            "func_agree": agree, "func_corr": corr,
            "fit_seconds": round(time.time() - t0, 1),
        })
        print(f"[u1c] q={n_use}: RE_F={hm['re_f']:.4f} cos={hm['row_cosine']:.4f} "
              f"agree={agree:.4f} corr={corr:.4f}", flush=True)

    best = curve[-1]
    if best["re_f_v"] < 0.01:
        verdict = "LEAK"
    elif best["func_agree"] > 0.9 or best["func_corr"] > 0.9:
        verdict = "PARTIAL"
    else:
        verdict = "PRESERVED"
    out = {
        "u1c_recover_v": {
            "n_samples": len(samples),
            "n_train_queries": n_q,
            "n_heldout_queries": len(heldout),
            "blind": blind,
            "joint_fit_curve": curve,
            "verdict": verdict,
        }
    }
    (run_dir / "u1c_recover_v.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[u1c] verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
