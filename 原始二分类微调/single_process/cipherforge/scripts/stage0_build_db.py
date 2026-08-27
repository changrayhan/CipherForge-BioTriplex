#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CipherForge Stage 0 for the ClinVar/TinyLlama path.

1. Generate the M-side BFV keypair (N=4096, plain_bits=30, scale=10000),
   persist sk (0600) and pk to ``cache_dir``.
2. Build the full encrypted lm_head DB (vocab_size rows x hidden_dim) with the
   public key, in place (progress printed every 1000 rows).
3. Decrypt-verify sampled rows with the (reloaded) secret key.
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.bfv_privselect_v2_adapter import (  # noqa: E402
    BFVPrivSelectV2Backend,
    _seal_ciphertext_from_bytes,
    _seal_to_bytes,
    create_bfv_context,
)


def load_lm_head(model_path: str, hidden_dim: int) -> np.ndarray:
    import glob

    from safetensors.torch import load_file

    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        sd = load_file(f, device="cpu")
        for k, v in sd.items():
            if k == "lm_head.weight":
                arr = v.float().numpy().astype(np.float64)
                if arr.shape[1] != hidden_dim:
                    raise ValueError(f"lm_head dim {arr.shape[1]} != {hidden_dim}")
                return arr
    raise FileNotFoundError("lm_head.weight not found in " + model_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--poly_degree", type=int, default=4096)
    ap.add_argument("--plain_bits", type=int, default=30)
    ap.add_argument("--scale", type=int, default=10000)
    ap.add_argument("--verify_rows", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    sk_path = os.path.join(args.cache_dir, "bfv_sk.bin")
    pk_path = os.path.join(args.cache_dir, "bfv_pk.bin")

    print("[Stage0] generating BFV keypair ...", flush=True)
    backend = BFVPrivSelectV2Backend(
        n_entries=args.vocab_size,
        vec_dim=args.hidden_dim,
        shared_seed=os.urandom(32),
        cache_dir=args.cache_dir,
        poly_degree=args.poly_degree,
        plain_bits=args.plain_bits,
        scale=args.scale,
        pk_path=None,
        force_new_keys=True,
    )
    with open(sk_path, "wb") as f:
        f.write(_seal_to_bytes(backend._secret_key))
    with open(pk_path, "wb") as f:
        f.write(backend.public_key_bytes)
    os.chmod(sk_path, 0o600)
    print(f"[Stage0] keypair saved: sk={sk_path} pk={pk_path}", flush=True)

    print("[Stage0] loading lm_head ...", flush=True)
    V = load_lm_head(args.model_path, args.hidden_dim)
    print(f"[Stage0] V shape = {V.shape}", flush=True)

    print("[Stage0] building encrypted DB ...", flush=True)
    t0 = __import__("time").time()
    backend.build_encrypted_database(V, force=False)
    dt = __import__("time").time() - t0
    print(f"[Stage0] DB built in {dt:.1f}s", flush=True)

    enc_db = backend._ensure_db()
    ctx = create_bfv_context(poly_degree=args.poly_degree, plain_bits=args.plain_bits)
    from seal import BatchEncoder, Ciphertext, Plaintext, SecretKey, Decryptor

    p = ctx.key_context_data().parms().plain_modulus().value()
    batch = BatchEncoder(ctx)
    half = p // 2

    # Reload the persisted secret key (proves the file is usable for Stage 1).
    sk2 = SecretKey()
    sk2.load(ctx, sk_path)
    dec2 = Decryptor(ctx, sk2)

    rows = sorted({int(i * (args.vocab_size - 1) / max(1, args.verify_rows - 1)) for i in range(args.verify_rows)})
    max_err = 0.0
    for i in rows:
        ct = _seal_ciphertext_from_bytes(ctx, enc_db.get_encrypted_row(i))
        pt = Plaintext()
        dec2.decrypt(ct, pt)
        raw = np.array(batch.decode_uint64(pt), dtype=np.int64)
        centered = np.where(raw > half, raw - p, raw)
        negV = centered[: args.hidden_dim]
        recovered = -negV.astype(np.float64) / args.scale
        err = float(np.max(np.abs(recovered - V[i])))
        tail_ok = bool((centered[args.hidden_dim :] == 0).all())
        max_err = max(max_err, err)
        print(f"[verify] row {i}: max_err={err:.2e} tail_zero={tail_ok}", flush=True)

    ok = max_err < 1e-3
    print(f"[Stage0] VERIFY {'OK' if ok else 'FAIL'} max_err={max_err:.2e}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
