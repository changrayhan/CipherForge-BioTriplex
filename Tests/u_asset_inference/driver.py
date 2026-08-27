#!/usr/bin/env python3
"""T1 driver: run a real three-party training with U-side captures.

Stages:
  1. INIT handshake (replicates the coordinator control plane);
  2. U1b: sample PIR ciphertexts for fixed indices across steps (attacker queries);
  3. train N steps; after selected steps take ACTIVE oracle snapshots
     (crafted H_U -> class logits) for per-step weight tracking (U1d);
  4. bulk active corpus for V recovery (U1c);
  5. passive eval (U node captures (H_U, z_cls) pairs itself);
  6. shutdown.

The driver impersonates U's active oracle: it holds the same M/S URLs U holds
and calls the same unauthenticated endpoints.
"""
from __future__ import annotations

import base64
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from common import init_nodes, load_config, query_val

DEFAULT_CFG = "/root/CipherForge/CipherForge-ClinVar/three_party/coordinator/three_party_config.json"


def snapshot_active(m_cli, trace, step, out_dir: Path, Q: int, H: int):
    os.makedirs(out_dir, exist_ok=True)
    for q in range(Q):
        h, positions, z = query_val(m_cli, trace, step, H=H)
        np.savez(
            out_dir / f"active_step_{int(step):04d}_q{q:03d}.npz",
            h_u=h.astype(np.float16), positions=positions, z_cls=z.astype(np.float32),
        )


def main() -> int:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CFG
    cfg = load_config(cfg_path)
    pir_mode = os.environ.get("PIR_MODE", "block")
    run_dir = Path(os.environ["RUN_DIR"])
    trace = "t1_uai"

    u_cli, m_cli, s_cli, wc, u_info = init_nodes(cfg, pir_mode, run_dir, trace)
    H = int(wc["hidden_dim"])
    print(f"[driver] nodes ready, hidden={H}, pir_mode={pir_mode}", flush=True)

    # ---- U1b: ciphertext sampling (same rows across steps) ----
    ct_dir = run_dir / "ciphertexts"
    ct_dir.mkdir(parents=True, exist_ok=True)
    indices = [0, 3869, 1939, 12345, 29999]
    for step in range(10):
        r = s_cli.action(trace, "BACKWARD", step, "fetch_rows", {"indices": indices})
        if not r.get("ok"):
            raise RuntimeError(f"fetch_rows failed: {r.get('error')}")
        rows = {
            idx: base64.b64decode(b) for idx, b in zip(indices, r["result"]["rows"])
        }
        with open(ct_dir / f"step_{step:02d}.pkl", "wb") as f:
            pickle.dump(rows, f)
    print(f"[driver] U1b ciphertext samples saved ({len(indices)} rows x 10 steps)", flush=True)

    # ---- training + per-step active snapshots ----
    N = int(os.environ.get("N_STEPS", "20"))
    SNAP_EVERY = int(os.environ.get("SNAP_EVERY", "5"))
    Q_SNAP = int(os.environ.get("Q_SNAP", "20"))
    MAX_SNAP_STEP = int(os.environ.get("MAX_SNAP_STEP", "20"))
    active_dir = run_dir / "active"
    for step in range(N):
        t0 = time.time()
        r = u_cli.action(trace, "TRAIN", step, "train_step", {"step": step})
        if not r.get("ok"):
            raise RuntimeError(f"train_step failed: {r.get('error')}")
        if step % SNAP_EVERY == 0 and step <= MAX_SNAP_STEP:
            snapshot_active(m_cli, trace, step, active_dir, Q_SNAP, H)
            print(f"[driver] step {step}: train ok + snapshot({Q_SNAP}) in {time.time()-t0:.1f}s", flush=True)
        elif step % 20 == 0:
            print(f"[driver] step {step}: train ok", flush=True)
    print(f"[driver] training done ({N} steps)", flush=True)

    # ---- U1c: bulk active corpus ----
    Q_CORPUS = int(os.environ.get("Q_CORPUS", "300"))
    corpus_dir = run_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for q in range(Q_CORPUS):
        h, positions, z = query_val(m_cli, trace, N, H=H)
        np.savez(
            corpus_dir / f"corpus_q{q:04d}.npz",
            h_u=h.astype(np.float16), positions=positions, z_cls=z.astype(np.float32),
        )
    print(f"[driver] U1c corpus done ({Q_CORPUS} queries)", flush=True)

    # ---- passive eval (captured by the U node) ----
    r = u_cli.action(trace, "EVAL", 0, "run_eval", {"kind": "val", "max_batches": 50})
    if not r.get("ok"):
        raise RuntimeError(f"run_eval failed: {r.get('error')}")
    print(f"[driver] passive eval done: {r.get('result', {}).get('val_samples')} samples", flush=True)

    for cli in (u_cli, m_cli, s_cli):
        try:
            cli.action(trace, "EVAL", 0, "shutdown", {})
        except Exception:
            pass
    print("[driver] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
