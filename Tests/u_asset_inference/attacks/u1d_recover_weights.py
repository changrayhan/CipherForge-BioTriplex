#!/usr/bin/env python3
"""U1d — can U recover M's per-step LoRA weight updates?

U observes per-step snapshots of the composed oracle (crafted H_U -> z_cls).
An incremental student is fine-tuned per snapshot step; the student's LoRA
update direction/magnitude is compared to the true W_M(t) trajectory.
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

from attacks.student import (  # noqa: E402
    build_student,
    composite_vector,
    freeze_lora_state_dict,
    student_z,
)
from common import load_config  # noqa: E402


def load_true_weights(cap_m: Path):
    """{step: composite vector} from M capture."""
    steps = {}
    for p in sorted(cap_m.glob("w_step_*.pt")):
        step = int(p.stem.split("_")[-1])
        state = torch.load(p, map_location="cpu", weights_only=False)
        comp = {}
        names = {}
        for k, v in state.items():
            if "lora" not in k:
                continue
            names[k] = v
        for k, v in names.items():
            if k.endswith("lora_B"):
                base = k[: -len("lora_B")]
                A = names.get(base + "lora_A")
                if A is not None:
                    comp[base] = (v @ A).float()
        steps[step] = composite_vector(comp).numpy()
    return steps


def load_snapshots(run_dir: Path, step: int):
    out = []
    for p in sorted((run_dir / "active").glob(f"active_step_{step:04d}_q*.npz")):
        d = np.load(p)
        out.append((
            d["h_u"].astype(np.float32),
            d["positions"].astype(np.int64),
            d["z_cls"].astype(np.float32),
        ))
    return out


def fit(model, head, opt, data, epochs, device):
    model.train()
    for _ in range(epochs):
        for h, pos, z in data:
            opt.zero_grad()
            h_t = torch.from_numpy(np.ascontiguousarray(h)).to(device)
            pos_t = torch.from_numpy(np.ascontiguousarray(pos)).to(device)
            z_t = torch.from_numpy(np.ascontiguousarray(z)).to(device)
            z_hat = student_z(model, head, h_t, pos_t)
            loss = F.mse_loss(z_hat, z_t)
            loss.backward()
            opt.step()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg", default="/root/CipherForge/CipherForge-ClinVar/three_party/coordinator/three_party_config.json")
    ap.add_argument("--epochs0", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(args.cfg)
    device = args.device if torch.cuda.is_available() else "cpu"

    true_w = load_true_weights(run_dir / "captures" / "m")
    snap_steps = sorted({
        int(p.stem.split("_")[2]) for p in (run_dir / "active").glob("active_step_*.npz")
    })
    snap_steps = [s for s in snap_steps if s in true_w]
    print(f"[u1d] snapshot steps with truth: {snap_steps}", flush=True)
    if len(snap_steps) < 2:
        raise SystemExit("need >=2 snapshot steps with ground truth")

    model, head = build_student(cfg, device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + [head], lr=1e-3)
    w_hat = {}
    for i, t in enumerate(snap_steps):
        data = load_snapshots(run_dir, t)
        if not data:
            continue
        if i == 0:
            fit(model, head, opt, data, args.epochs0, device)
        else:
            fit(model, head, opt, data, args.epochs, device)
        w_hat[t] = composite_vector(freeze_lora_state_dict(model)).numpy()
        print(f"[u1d] fitted step {t} (data={len(data)})", flush=True)

    steps = sorted(w_hat)
    per_step = []
    dW_true_all, dW_hat_all = [], []
    for i in range(1, len(steps)):
        t_prev, t = steps[i - 1], steps[i]
        d_true = true_w[t] - true_w[t_prev]
        d_hat = w_hat[t] - w_hat[t_prev]
        dW_true_all.append(d_true)
        dW_hat_all.append(d_hat)
        re = float(np.linalg.norm(d_hat - d_true) / (np.linalg.norm(d_true) + 1e-12))
        cos = float(np.dot(d_hat, d_true) / (
            np.linalg.norm(d_hat) * np.linalg.norm(d_true) + 1e-12))
        per_step.append({"from": t_prev, "to": t, "re_f": re, "direction_cosine": cos})
        print(f"[u1d] {t_prev}->{t}: RE_F={re:.4f} cos={cos:.4f}", flush=True)

    dW_true_all = np.concatenate(dW_true_all)
    dW_hat_all = np.concatenate(dW_hat_all)
    traj_cos = float(np.dot(dW_hat_all, dW_true_all) / (
        np.linalg.norm(dW_hat_all) * np.linalg.norm(dW_true_all) + 1e-12))
    mean_re = float(np.mean([p["re_f"] for p in per_step]))
    mean_cos = float(np.mean([p["direction_cosine"] for p in per_step]))
    verdict = "LEAK" if (mean_re < 0.01 or traj_cos > 0.9) else "PRESERVED"
    out = {
        "u1d_recover_weights": {
            "snapshot_steps": steps,
            "per_step": per_step,
            "mean_re_f": mean_re,
            "mean_direction_cosine": mean_cos,
            "trajectory_cosine": traj_cos,
            "verdict": verdict,
        }
    }
    (run_dir / "u1d_recover_weights.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[u1d] mean RE_F={mean_re:.4f} mean cos={mean_cos:.4f} "
          f"traj cos={traj_cos:.4f} verdict={verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
