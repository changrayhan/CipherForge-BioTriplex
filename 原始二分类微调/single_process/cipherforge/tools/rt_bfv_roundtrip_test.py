#!/usr/bin/env python3
"""Real-SEAL round-trip test for the ClinVar BFV/S3PIR pipeline.

Uses the PERSISTED keypair (bfv_pk.bin / bfv_sk.bin), the REAL encrypted DB
(bfv_ct_db_n32000_d2048_p4096.bin) and the REAL TinyLlama lm_head rows.
Replicates exactly what the runtime does per training step:

  U: fetch Enc(-V_y * scale), homomorphically add PRG mask r_t
  M: decrypt + decode -> masked_arr, plaintext-add s_share = a_t*scale - r_t

Then compares the reconstructed gradient against the plaintext
``(a_t - V_y)`` and reports per-slot error statistics for both the old
(centre-then-add) and new (add-mod-then-centre) reconstruction formulas.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CACHE = "/root/cipherforge/checkpoints/clinvar_bfv_cache"
DB = os.path.join(CACHE, "bfv_ct_db_n32000_d2048_p4096.bin")
PK = os.path.join(CACHE, "bfv_pk.bin")
SK = os.path.join(CACHE, "bfv_sk.bin")
MODEL = (
    "/root/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
    "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
)
SCALE = 10000
PM = 1 << 30
HALF = PM // 2

from seal import (  # noqa: E402
    BatchEncoder,
    Ciphertext,
    Decryptor,
    Evaluator,
    Plaintext,
    PublicKey,
    SecretKey,
)

from src.core.bfv_privselect_v2_adapter import (  # noqa: E402
    BFVEncryptedDatabase,
    PRGShareProtocolBFV,
    _seal_load_ciphertext,
    create_bfv_context,
    decode_ints_as_vector,
)


def load_key(path: str, cls):
    raw = open(path, "rb").read()
    fd, p = tempfile.mkstemp(suffix=".key")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        k = cls()
        k.load(ctx, p)
        return k
    finally:
        os.unlink(p)


def decrypt_ints(ct_bytes, r_t):
    ct = _seal_load_ciphertext(ctx, ct_bytes)
    pt_mask = encoder.encode(np.asarray(r_t, dtype=np.int64))
    evaluator.add_plain_inplace(ct, pt_mask)
    pt = Plaintext()
    decryptor.decrypt(ct, pt)
    return np.array(encoder.decode(pt), dtype=np.int64)


def recon_formula(ints2048, s_share, mode):
    if mode == "old":
        c = np.where(ints2048 > HALF, ints2048 - PM, ints2048)
        diff = c + s_share
        return diff.astype(np.float64) / SCALE
    diff_mod = (ints2048 + s_share) % PM
    c = np.where(diff_mod > HALF, diff_mod - PM, diff_mod)
    return c.astype(np.float64) / SCALE


def run_case(y, step, t_flat, r_t, label):
    a_t = (0.7 * V[y] + 0.3 * V[(y + 137) % 32000]).astype(np.float32)
    s_share = shS.server_make_share(step, t_flat, a_t)
    ints = decrypt_ints(enc_db.get_encrypted_row(y), r_t)
    ints2048 = ints[:2048]
    target = (
        np.round(a_t * SCALE).astype(np.int64)
        - np.round(V[y] * SCALE).astype(np.int64)
    ) / SCALE
    old_exact = recon_formula(ints2048, s_share, "old")
    new_exact = recon_formula(ints2048, s_share, "new")
    # Pipeline-realistic: CryptoMWorker returns float32 (ints / scale).
    masked_f32 = decode_ints_as_vector(ints2048, SCALE).astype(np.float32)
    masked_int_f32 = np.round(masked_f32 * SCALE).astype(np.int64)
    old_pipe = recon_formula(masked_int_f32, s_share, "old")
    new_pipe = recon_formula(masked_int_f32, s_share, "new")
    return {
        "y": y,
        "step": step,
        "t": t_flat,
        "label": label,
        "old_exact_max": float(np.abs(old_exact - target).max()),
        "new_exact_max": float(np.abs(new_exact - target).max()),
        "old_pipe_max": float(np.abs(old_pipe - target).max()),
        "new_pipe_max": float(np.abs(new_pipe - target).max()),
        "old_pipe_bad": int((np.abs(old_pipe - target) > 0.05).sum()),
        "new_pipe_bad": int((np.abs(new_pipe - target) > 0.05).sum()),
        "new_pipe_bad005": int((np.abs(new_pipe - target) > 0.005).sum()),
    }


ctx = create_bfv_context(poly_degree=4096, plain_bits=30)
encoder = BatchEncoder(ctx)
evaluator = Evaluator(ctx)

pk = load_key(PK, PublicKey)
sk = load_key(SK, SecretKey)
decryptor = Decryptor(ctx, sk)
print("keys loaded: pk ok, sk ok", flush=True)

enc_db = BFVEncryptedDatabase.from_cache(
    context=ctx,
    n_entries=32000,
    vec_dim=2048,
    cache_path=DB,
    public_key=pk,
)
print("DB loaded:", len(enc_db._ct_list), "rows", flush=True)

V = None
for f in sorted(glob.glob(os.path.join(MODEL, "*.safetensors"))):
    from safetensors.torch import load_file

    sd = load_file(f, device="cpu")
    for k, v in sd.items():
        if k == "lm_head.weight":
            V = v.float().numpy().astype(np.float64)
if V is None:
    raise SystemExit("lm_head not found")
V = V[:32000]
print("V:", V.shape, "max|V|=", float(np.abs(V).max()), flush=True)

seed = bytes(range(32))
shU = PRGShareProtocolBFV(prg_seed=seed, vec_dim=4096, plain_modulus=PM, scale=SCALE)
shS = PRGShareProtocolBFV(prg_seed=seed, vec_dim=2048, plain_modulus=PM, scale=SCALE)

# ---------------- Test A: DB integrity (decrypt w/o mask) ----------------
rows_a = [0, 1, 2, 5, 100, 1000, 5000, 15000, 29999, 31999]
top = np.argsort(np.abs(V).max(axis=1))[::-1][:4]
rows_a += [int(t) for t in top]
max_err_a = 0.0
print("\n=== Test A: DB rows decode vs -round(V*scale) (first 2048 slots) ===", flush=True)
for y in rows_a:
    ct = enc_db.get_encrypted_row(y)
    ints = decrypt_ints(ct, [0] * 4096)
    expect = -np.round(V[y] * SCALE).astype(np.int64)
    err = float(np.abs(ints[:2048] - expect).max())
    max_err_a = max(max_err_a, err)
    print(f"  y={y:6d} max|int err|={err}  (float err={err / SCALE:.2e})", flush=True)
print(f"DB integrity max int err = {max_err_a}  ({max_err_a / SCALE:.2e} float)", flush=True)

# ---------------- Test B: full round trip with mask + s_share ----------------
rng = np.random.default_rng(7)
ys = [0, 1, 2, 29800, 31999] + [int(t) for t in top] + rng.integers(0, 32000, 12).tolist()
ys = list(dict.fromkeys(ys))
boundary = [(-HALF, "r=-pm/2"), (HALF - 1, "r=pm/2-1"), (HALF - 2, "r=pm/2-2"), (0, "r=0")]

print("\n=== Test B: full round trip (mask+decrypt+s_share) vs plaintext ===", flush=True)
stats = {"old_exact": [], "new_exact": [], "old_pipe": [], "new_pipe": []}
bad_cases = []
n_cases = 0
for step in [0, 1]:
    for t_flat in [0, 13, 255, 511]:
        for y in ys:
            r_t = shU.generate_mask_ints(step, t_flat)
            r = run_case(y, step, t_flat, r_t, f"rng t={t_flat}")
            n_cases += 1
            for k in stats:
                stats[k].append(r[k + "_max"])
            if r["old_pipe_bad"] or r["new_pipe_bad005"]:
                bad_cases.append(r)

print(f"cases={n_cases}", flush=True)
for k in stats:
    a = np.array(stats[k])
    print(
        f"  {k:12s}: max={a.max():.4f}  mean={a.mean():.6f}  p90={np.percentile(a, 90):.4f}",
        flush=True,
    )

print("\nworst cases with real random r_t (old_bad>0 or new_bad005>0):", flush=True)
for r in sorted(bad_cases, key=lambda x: -x["old_pipe_max"])[:15]:
    print(
        f"  y={r['y']:6d} step={r['step']} t={r['t']:3d} {r['label']:12s} "
        f"old_exact={r['old_exact_max']:.3f} new_exact={r['new_exact_max']:.5f} "
        f"old_pipe={r['old_pipe_max']:.3f} new_pipe={r['new_pipe_max']:.5f} "
        f"old_bad={r['old_pipe_bad']} new_bad={r['new_pipe_bad']} "
        f"new_bad005={r['new_pipe_bad005']}",
        flush=True,
    )

# ---------------- Test D: real block-PIR round trip ----------------
# U builds a real+dummy block (real target y hidden among 7 random dummies),
# S returns the encrypted rows of the whole block, U extracts the real row via
# its private permutation, adds r_t, M decrypts + adds s_share. S never learns
# which row is the target.
print("\n=== Test D: real block PIR (block=8, U extracts) ===", flush=True)
import secrets as _secrets

rngd = _secrets.SystemRandom()
max_err_d = 0.0
n_slots_d = 0
for trial in range(40):
    step_d = trial % 3
    t_d = trial
    y = int(rngd.randrange(32000))
    a_t = (0.7 * V[y] + 0.3 * V[(y + 137) % 32000]).astype(np.float32)
    others = [i for i in range(32000) if i != y]
    dummies = rngd.sample(others, 7)
    block = dummies + [y]
    rngd.shuffle(block)
    real_pos = block.index(y)
    rows = {i: enc_db.get_encrypted_row(i) for i in set(block)}
    real_ct = rows[block[real_pos]]          # U picks its private position
    r_t = shU.generate_mask_ints(step_d, t_d)
    ints = decrypt_ints(real_ct, r_t)[:2048]
    s_share = shS.server_make_share(step_d, t_d, a_t)
    target = (
        np.round(a_t * SCALE).astype(np.int64)
        - np.round(V[y] * SCALE).astype(np.int64)
    ) / SCALE
    diff_mod = (ints + s_share) % PM
    cent = np.where(diff_mod > HALF, diff_mod - PM, diff_mod) / SCALE
    e = float(np.abs(cent - target).max())
    max_err_d = max(max_err_d, e)
    n_slots_d += int(cent.size)
print(
    f"block-PIR trials=40 block=8 slots={n_slots_d} max_abs_err={max_err_d:.6f}",
    flush=True,
)

# ---------------- Test C: artificial boundary probe (documents the artifact) ----------------
print("\n=== Test C: artificial probe with constant r_t near pm/2 (mechanism check) ===", flush=True)
for val, lab in boundary:
    r_t = np.full(4096, val, dtype=np.int64)
    r = run_case(ys[0], 0, 0, r_t, lab)
    print(
        f"  {lab:12s}: old_exact={r['old_exact_max']:.3f} new_exact={r['new_exact_max']:.3f} "
        f"new_bad005={r['new_pipe_bad005']}",
        flush=True,
    )

# ---------------- Deep dive: one failing case ----------------
print("\n=== Deep dive: y=2 step=0 t=0 (worst new-formula residual) ===", flush=True)
y_dd, step_dd, t_dd = 2, 0, 0
a_t = (0.7 * V[y_dd] + 0.3 * V[(y_dd + 137) % 32000]).astype(np.float32)
s_share = shS.server_make_share(step_dd, t_dd, a_t)
r_t = shU.generate_mask_ints(step_dd, t_dd)
ints = decrypt_ints(enc_db.get_encrypted_row(y_dd), r_t)
ints2048 = ints[:2048]
target = (
    np.round(a_t * SCALE).astype(np.int64)
    - np.round(V[y_dd] * SCALE).astype(np.int64)
)
new_pipe = recon_formula(
    np.round(decode_ints_as_vector(ints2048, SCALE).astype(np.float32) * SCALE).astype(np.int64),
    s_share,
    "new",
) * SCALE
err = np.abs(new_pipe - target)
worst = int(np.argmax(err))
print(f"worst slot={worst}  err(float)={err[worst]:.4f}  err(int)={err[worst]:.0f}", flush=True)
print(f"  decoded int      = {ints2048[worst]}", flush=True)
print(f"  decoded centred  = {ints2048[worst] - PM if ints2048[worst] > HALF else ints2048[worst]}", flush=True)
print(f"  r_t (U/S PRG)    = {int(r_t[worst])}", flush=True)
print(f"  s_share          = {int(s_share[worst])}", flush=True)
print(f"  target int       = {int(target[worst])}", flush=True)
print(f"  -round(V*scale)  = {int(-np.round(V[y_dd] * SCALE).astype(np.int64)[worst])}", flush=True)
print(f"  round(a*scale)   = {int(np.round(a_t * SCALE).astype(np.int64)[worst])}", flush=True)
true_masked = (-np.round(V[y_dd] * SCALE).astype(np.int64)[worst] + int(r_t[worst])) % PM
print(f"  true masked mod  = {true_masked}", flush=True)
print(f"  decoded - true   = {int(ints2048[worst]) - true_masked}  (mod-pm delta)", flush=True)
diff_mod = (ints2048 + s_share) % PM
cent = np.where(diff_mod > HALF, diff_mod - PM, diff_mod)
print(f"  new-formula int  = {int(cent[worst])}  vs target {int(target[worst])}", flush=True)

dom = ints2048
n_neg = int((dom < 0).sum())
n_high = int((dom > HALF).sum())
n_low = int(((dom >= 0) & (dom <= HALF)).sum())
print(f"decode domain over 2048 slots: negative={n_neg}  [0,pm/2]={n_low}  (pm/2,pm)={n_high}", flush=True)
multi = np.unique(np.abs(cent - target))
print("unique |new - target| int deltas (first 12):", multi[:12].tolist(), flush=True)

# ---------------- Full scan: per-slot new-formula residual structure ----------------
print("\n=== Full scan: per-slot |new_exact - target| int deltas over all cases ===", flush=True)
import collections
delta_hist = collections.Counter()
big_slots = []
for step in [0, 1]:
    for t_flat in [0, 13, 255, 511]:
        for y in ys:
            r_t = shU.generate_mask_ints(step, t_flat)
            a_t = (0.7 * V[y] + 0.3 * V[(y + 137) % 32000]).astype(np.float32)
            s_share = shS.server_make_share(step, t_flat, a_t)
            ints = decrypt_ints(enc_db.get_encrypted_row(y), r_t)[:2048]
            target = (
                np.round(a_t * SCALE).astype(np.int64)
                - np.round(V[y] * SCALE).astype(np.int64)
            )
            diff_mod = (ints + s_share) % PM
            cent = np.where(diff_mod > HALF, diff_mod - PM, diff_mod)
            err = np.abs(cent - target)
            for d in np.unique(err):
                delta_hist[int(d)] += 1
            for slot in np.flatnonzero(err > 50000):
                big_slots.append((y, step, t_flat, int(slot), int(err[slot]),
                                  int(ints[slot]), int(r_t[slot]), int(s_share[slot]),
                                  int(target[slot])))
print("delta histogram (int error -> #slots), top 20:", flush=True)
for d, c in sorted(delta_hist.items(), key=lambda kv: -kv[1])[:20]:
    print(f"  {d:>12} -> {c}", flush=True)
print(f"slots with |err| > 50000 ints: {len(big_slots)}", flush=True)
for s in big_slots[:20]:
    y, step, t_flat, slot, err, dec, r, sh, tgt = s
    print(
        f"  y={y:6d} step={step} t={t_flat:3d} slot={slot:4d} err={err} "
        f"decoded={dec} r_t={r} s_share={sh} target={tgt}",
        flush=True,
    )
