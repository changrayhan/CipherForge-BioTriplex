#!/usr/bin/env python3
"""RMS-PIR v2 round-trip verification against the real BFV encrypted DB.

Mirrors the v2 responsibility split:
  - U (offline server): hint parities and replenishment halves are ciphertext
    sums over a local encrypted DB copy;
  - S (online server): answers the two online subsets with plaintext
    aggregation (per-row fixed-point ints summed, then encoded + encrypted
    once under pk_M) — bit-exact with the ciphertext-sum path;
  - M: decrypts the recovered Enc(-V_y).
Loads the real 4.2 GB encrypted DB, the M-side keys and the plaintext V
matrix, runs real queries, decrypts with sk_M and compares against direct DB
row decryption.

Usage (from the repo root, WSL):
  PYTHON=... bash -c 'cd /root/cipherforge-three-party && \
    $PYTHON -u scripts/rms_roundtrip_verify.py --keys_dir party_m/keys \
    --db_dir party_s/db --model_path $CF_MODEL_PATH --lam 12'
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from shared.core.bfv_privselect_v2_adapter import (
    BFVEncryptedDatabase,
    _seal_ciphertext_from_bytes,
    _seal_ciphertext_to_bytes,
    create_bfv_context,
)
from shared.core.rms_pir import (
    RMSHintParams,
    RMSHintStore,
    hint_half_rows,
    pick_replenish_half,
)


def load_sk_pk(keys_dir: str):
    from seal import PublicKey, SecretKey
    ctx = create_bfv_context(poly_degree=4096, plain_bits=30)
    sk = SecretKey()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sk") as f:
        f.write(Path(keys_dir, "bfv_sk.bin").read_bytes())
        sk_path = f.name
    try:
        sk.load(ctx, sk_path)
    finally:
        os.unlink(sk_path)
    pk = PublicKey()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pk") as f:
        f.write(Path(keys_dir, "bfv_pk.bin").read_bytes())
        pk_path = f.name
    try:
        pk.load(ctx, pk_path)
    finally:
        os.unlink(pk_path)
    return ctx, sk, pk


def decrypt_float(ctx, sk, ct_bytes: bytes, scale: float = 10000.0, dim: int = 2048):
    from seal import BatchEncoder, Decryptor, Plaintext
    dec = Decryptor(ctx, sk)
    pt = Plaintext()
    dec.decrypt(_seal_ciphertext_from_bytes(ctx, ct_bytes), pt)
    vals = np.asarray(BatchEncoder(ctx).decode(pt), dtype=np.float64)
    return vals[:dim] / scale


def parity_bytes(ctx, enc_db, rows):
    """U-local hint/replenishment parity: ciphertext sum over encrypted rows."""
    from shared.core.bfv_privselect_v2_adapter import (
        _seal_ciphertext_from_bytes,
        _seal_ciphertext_to_bytes,
    )
    valid = [int(i) for i in rows if 0 <= int(i) < enc_db.n_entries]
    acc = _seal_ciphertext_from_bytes(ctx, enc_db.get_encrypted_row(valid[0]))
    for i in valid[1:]:
        ct = _seal_ciphertext_from_bytes(ctx, enc_db.get_encrypted_row(i))
        enc_db.evaluator.add_inplace(acc, ct)
    return _seal_ciphertext_to_bytes(acc)


def s_parity_bytes(ctx, pk, V, rows, scale=10000.0, poly_degree=4096, dim=2048):
    """S-side online parity: plaintext int aggregation + one encryption.

    int_sum = -Σ round(V_i · scale) — bit-exact with Σ Enc(-V_i).
    """
    from seal import BatchEncoder, Ciphertext, Encryptor
    from shared.core.bfv_privselect_v2_adapter import _seal_to_bytes
    valid = [int(i) for i in rows if 0 <= int(i) < V.shape[0]]
    if not valid:
        ints = np.zeros(poly_degree, dtype=np.int64)
    else:
        int_sum = -np.round(V[valid] * scale).astype(np.int64).sum(axis=0)
        ints = np.zeros(poly_degree, dtype=np.int64)
        ints[: min(dim, len(int_sum))] = int_sum[:dim]
    pt = BatchEncoder(ctx).encode(ints)
    ct = Ciphertext()
    Encryptor(ctx, pk).encrypt(pt, ct)
    return _seal_to_bytes(ct)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys_dir", default=str(ROOT / "party_m" / "keys"))
    ap.add_argument("--db_dir", default=str(ROOT / "party_s" / "db"))
    ap.add_argument("--model_path", default=os.environ.get("CF_MODEL_PATH", ""))
    ap.add_argument("--lam", type=int, default=12)
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("--model_path (CF_MODEL_PATH) is required")

    ctx, sk, pk = load_sk_pk(args.keys_dir)
    db_path = Path(args.db_dir) / "bfv_ct_db_n32000_d2048_p4096.bin"
    enc_db = BFVEncryptedDatabase.from_cache(
        context=ctx, n_entries=32000, vec_dim=2048,
        cache_path=str(db_path), public_key=pk,
    )
    print(f"[roundtrip] DB loaded: {enc_db.n_entries} rows", flush=True)
    from shared.model.model_splitting import detect_model_spec, load_s_submodel
    spec = detect_model_spec(args.model_path)
    V_model = load_s_submodel(spec=spec, model_path=args.model_path, device="cpu")
    V = V_model.weight.detach().float().numpy().astype(np.float64)
    print(f"[roundtrip] V loaded: {V.shape}", flush=True)

    params = RMSHintParams(n_entries=32000, partition_size=200, lam=args.lam)
    seed = os.urandom(32)
    labels = [259, 3869, 1939]      # space / Yes / No
    store = RMSHintStore(seed, params, tempfile.mkdtemp(prefix="rms_verify_"))
    req, topups = store.build_initial_pool(labels, min_coverage=4)
    req.update({j: topups[j]["row_list"] for j in topups})
    for j, t in topups.items():
        store.hints[j] = {
            "rows": t["picked_rows"], "extra": t["y"],
            "parity_path": str(store._parity_path(j)), "_pending": True,
        }

    # Compute parities only for hints that contain a test label (cheap subset
    # of the full offline phase).
    needed = set()
    for j, row_list in req.items():
        if any(y in row_list for y in labels):
            needed.add(j)
    print(f"[roundtrip] scanning {len(req)} hints, {len(needed)} contain a label", flush=True)
    for j in needed:
        store.complete_hint(j, parity_bytes(ctx, enc_db, req[j]))
    for y in labels:
        assert len(store.label_hints.get(y, [])) > 0, f"no hint for label {y}"

    max_err = 0.0
    n_queries = 0
    for y in labels:
        for trial in range(6):
            j, rows, extra, hint_ct = store.pop_hint(y)
            real, dummy, perm = store.build_query(j, rows, extra, y)
            q_real_ct = s_parity_bytes(ctx, pk, V, real)
            q_dummy_ct = s_parity_bytes(ctx, pk, V, dummy)
            q_ct = q_real_ct if perm == 0 else q_dummy_ct

            # U-side recovery: hint_parity - real_query_parity
            hint_ct_obj = _seal_ciphertext_from_bytes(ctx, hint_ct)
            q_ct_obj = _seal_ciphertext_from_bytes(ctx, q_ct)
            recovered = enc_db.evaluator.sub(hint_ct_obj, q_ct_obj)
            recovered_bytes = _seal_ciphertext_to_bytes(recovered)
            got = decrypt_float(ctx, sk, recovered_bytes)
            want = decrypt_float(ctx, sk, enc_db.get_encrypted_row(y))
            err = float(np.max(np.abs(got - want)))
            max_err = max(max_err, err)
            n_queries += 1
            if err > 0.01:
                raise RuntimeError(f"label {y} trial {trial}: err={err} > 0.01")

            # Replenish: pick half not containing y's partition, new parity =
            # picked half + recovered, then use the new hint for another query.
            J, half_a, half_b = store.plan_replenish(y)
            rows_a, rows_b, _, _ = hint_half_rows(seed, params, J)
            picked_rows, picked_idx = pick_replenish_half(params, rows_a, rows_b, y)
            half_ct = parity_bytes(ctx, enc_db, half_a if picked_idx == 0 else half_b)
            half_obj = _seal_ciphertext_from_bytes(ctx, half_ct)
            new_par = enc_db.evaluator.add(recovered, half_obj)
            store.add_replenished(J, y, picked_rows, _seal_ciphertext_to_bytes(new_par))

            # Second query with the replenished hint (it must contain y)
            h2 = store.hints[J]
            assert h2["extra"] == y
            real2, dummy2, perm2 = store.build_query(J, h2["rows"], h2["extra"], y)
            q2 = s_parity_bytes(ctx, pk, V, real2 if perm2 == 0 else dummy2)
            rec2 = enc_db.evaluator.sub(
                _seal_ciphertext_from_bytes(ctx, Path(h2["parity_path"]).read_bytes()),
                _seal_ciphertext_from_bytes(ctx, q2),
            )
            got2 = decrypt_float(ctx, sk, _seal_ciphertext_to_bytes(rec2))
            err2 = float(np.max(np.abs(got2 - want)))
            max_err = max(max_err, err2)
            n_queries += 1
            if err2 > 0.01:
                raise RuntimeError(f"replenished label {y}: err={err2} > 0.01")

    print(f"[roundtrip] OK: {n_queries} queries, max |err| = {max_err:.6f} (threshold 0.01)")


if __name__ == "__main__":
    main()
