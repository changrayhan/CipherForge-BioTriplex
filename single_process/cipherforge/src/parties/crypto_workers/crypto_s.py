"""
CryptoSWorker — CPU worker that handles S-side ``process_logits`` post-processing.

Responsibilities
----------------
The GPU Fusion driver computes ``logits = H_M @ V^T`` and
``a_t = softmax @ V`` on the GPU (V is too large to live in the CPU worker
heap twice). What remains for the CPU worker is:

  1. Generate ``s_share = a_t - R_t`` for each token (PRG share with U).
  2. Look up ``Enc(-V_{y_t})`` in the mmap-backed encrypted DB.
  3. Return the per-token response dicts.

Privacy
-------
This worker is created with:

  * ``bfv_pk_pem`` only — never ``sk_M``.
  * ``prg_seed`` — same seed U uses; M never sees it.
  * ``enc_db_path`` — the on-disk encrypted DB; mmap'd read-only.

The worker cannot decrypt anything. It only produces ciphertext bytes
(zero-copy slices of the mmap) and the plaintext share ``s_share``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class CryptoSWorker:
    """CPU worker for S-side response + share generation."""

    @classmethod
    def init_state(
        cls,
        bfv_pk_pem: bytes,
        prg_seed: bytes,
        bfv_cache_dir: str,
        poly_degree: int,
        plain_bits: int,
        scale: int,
        plain_modulus: int,
        n_entries: int,
        vec_dim: int,
        partition_size: int,
        lam: int,
    ) -> Dict[str, Any]:
        """One-time per-worker init. Loads the mmap'd encrypted DB + PRG state."""
        import os
        from pathlib import Path

        from src.core.bfv_privselect_v2_adapter import (
            PRGShareProtocolBFV,
            BFVEncryptedDatabase,
            create_bfv_context,
        )
        from src.core.s3pir_hints import HintTable
        from seal import PublicKey  # type: ignore
        import tempfile
        import pickle as _pickle

        ctx = create_bfv_context(poly_degree=poly_degree, plain_bits=plain_bits)

        # Determine cache path (mirrors build_encrypted_db.py naming).
        cache_path = (
            Path(bfv_cache_dir)
            / f"bfv_ct_db_n{n_entries}_d{vec_dim}_p{poly_degree}.bin"
        )

        # Unwrap pickled pk.
        _pk_data = _pickle.loads(bfv_pk_pem)
        pk_raw_bytes = _pk_data["pk_bytes"]

        pk = PublicKey()
        fd, pk_path = tempfile.mkstemp(suffix=".pub")
        try:
            with open(pk_path, "wb") as f:
                f.write(pk_raw_bytes)
            pk.load(ctx, pk_path)
        finally:
            try:
                os.remove(pk_path)
            except OSError:
                pass

        enc_db = BFVEncryptedDatabase.from_cache(
            context=ctx,
            n_entries=n_entries,
            vec_dim=vec_dim,
            cache_path=str(cache_path),
            public_key=pk,
        )

        shares = PRGShareProtocolBFV(
            prg_seed=prg_seed,
            vec_dim=vec_dim,
            plain_modulus=plain_modulus,
            scale=scale,
        )

        # Optional HintTable (Design-2 only uses |real|=1; hint parity isn't
        # needed for the per-row mmap fetch).
        hints_dir = Path(bfv_cache_dir) / "s3pir_hints"
        hint_table = None
        if (hints_dir / "hint_table.json").exists():
            try:
                hint_table = HintTable.from_cache_files(str(hints_dir))
            except Exception:
                hint_table = None

        logger.info(
            "[CryptoSWorker] init done: n_entries=%d vec_dim=%d poly_degree=%d "
            "enc_db=%s hints=%s",
            n_entries, vec_dim, poly_degree, cache_path,
            "loaded" if hint_table else "none",
        )

        return {
            "enc_db": enc_db,
            "shares": shares,
            "hint_table": hint_table,
            "n_entries": n_entries,
            "vec_dim": vec_dim,
        }

    @classmethod
    def handle_request(
        cls,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle two label-free request modes.

        mode="make_shares": compute ``s_share = a_t - r_t`` for every position
            (S never sees the supervised positions or gold tokens).
        mode="fetch_rows":  return the encrypted rows for a real+dummy PIR
            query block (S cannot tell which row is the target).
        """
        enc_db = state["enc_db"]
        shares = state["shares"]
        hint_table = state["hint_table"]

        mode = payload.get("mode", "make_shares")
        step = int(payload.get("step", 0))

        if mode == "fetch_rows":
            indices = [int(i) for i in payload.get("indices", [])]
            cache: Dict[int, bytes] = {}
            rows: List[bytes] = []
            for i in indices:
                if i not in cache:
                    cache[i] = enc_db.get_encrypted_row(i)
                rows.append(cache[i])
            return {"rows": rows}

        # ---- mode = "make_shares" ----
        a_t_list: List[np.ndarray] = payload["a_t_list"]
        t_flats: List[int] = payload.get("t_flats") or list(range(len(a_t_list)))
        all_s_shares: List[List[int]] = []
        for i, a_t in enumerate(a_t_list):
            t_flat = int(t_flats[i]) if i < len(t_flats) else i
            # s_share = a_t - R_t — pure CPU, scales linearly with tokens.
            s_share = shares.server_make_share(step, t_flat, a_t)
            all_s_shares.append(s_share)

        return {
            "s_shares": all_s_shares,
        }
