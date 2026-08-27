"""
CryptoUWorker — CPU worker that handles U-side ``add_mask``.

Responsibilities
----------------
For each token ``t`` in the current step:

  1. Pick the S3PIR parity blob (real or dummy) according to ``permutation``.
  2. Generate the per-token mask ``R_t`` from the PRG seed shared with S.
  3. Homomorphically add ``R_t`` (plaintext) to the ciphertext, yielding
     ``Enc(-V_{y_t} + R_t)``.

Privacy
-------
This worker is created with:

  * ``bfv_pk_pem`` only — it never sees ``sk_M``.
  * ``prg_seed`` — only U and S share this; M never receives it.

The worker therefore *cannot* decrypt anything — it can only operate on
already-encrypted blobs and add public-key-preserving masks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class CryptoUWorker:
    """CPU worker for U-side ``add_mask``."""

    @classmethod
    def init_state(
        cls,
        bfv_pk_pem: bytes,
        prg_seed: bytes,
        poly_degree: int,
        plain_bits: int,
        scale: int,
        plain_modulus: int,
        rms_db_path: str = "",
        rms_n_entries: int = 0,
    ) -> Dict[str, Any]:
        """One-time per-worker init. Builds the SEAL context and PRG state."""
        from seal import (  # type: ignore
            BatchEncoder,
            Evaluator,
            PublicKey,
            SEALContext,
        )

        from shared.core.bfv_privselect_v2_adapter import (
            PRGShareProtocolBFV,
            create_bfv_context,
        )

        ctx = create_bfv_context(poly_degree=poly_degree, plain_bits=plain_bits)
        encoder = BatchEncoder(ctx)
        evaluator = Evaluator(ctx)

        # Unwrap pickled pk format produced by ``BFVPrivSelectV2Backend.get_he_pubkey_pem``.
        # Format: ``pickle.dumps({"pk_bytes": raw_seal_bytes, ...})``.
        import pickle as _pickle
        _pk_data = _pickle.loads(bfv_pk_pem)
        pk_raw_bytes = _pk_data["pk_bytes"]

        pk = PublicKey()
        fd, pk_path = tempfile.mkstemp(suffix=".pub")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(pk_raw_bytes)
            pk.load(ctx, pk_path)
        finally:
            try:
                os.remove(pk_path)
            except OSError:
                pass

        shares = PRGShareProtocolBFV(
            prg_seed=prg_seed,
            vec_dim=poly_degree,
            plain_modulus=plain_modulus,
        )

        # RMS-PIR v2: U holds a local encrypted DB copy and is the offline
        # server (hint parities + replenishment halves computed here, never
        # shown to S).  Loaded lazily by the worker; requires rms_db_path.
        enc_db = None
        if rms_db_path:
            from shared.core.bfv_privselect_v2_adapter import BFVEncryptedDatabase
            enc_db = BFVEncryptedDatabase.from_cache(
                context=ctx,
                n_entries=int(rms_n_entries),
                vec_dim=poly_degree,
                cache_path=rms_db_path,
                public_key=pk,
            )
            logger.info(
                "[CryptoUWorker] RMS local encrypted DB loaded: %s (%d rows)",
                rms_db_path, enc_db.n_entries,
            )

        logger.info(
            "[CryptoUWorker] init done: poly_degree=%d plain_bits=%d scale=%d",
            poly_degree, plain_bits, scale,
        )
        return {
            "ctx": ctx,
            "encoder": encoder,
            "evaluator": evaluator,
            "public_key": pk,
            "shares": shares,
            "poly_degree": poly_degree,
            "vec_dim": poly_degree,
            "plain_modulus": plain_modulus,
            "enc_db": enc_db,
        }

    @classmethod
    def handle_request(
        cls,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add the PRG mask ``r_t`` to the *selected* PIR row of each token.

        Args:
            payload: ``{"ct_bytes": [Enc(-V_y) ...], "t_flats": [...],
                        "step": int}`` — one selected row per valid token;
                        or ``{"mode": "rms_recover_and_mask", "items": [...]}``
                        for the RMS-PIR path.

        Returns:
            ``{"ct_list": List[bytes]}`` — one masked ciphertext per token.
        """
        from seal import Plaintext  # type: ignore

        from shared.core.bfv_privselect_v2_adapter import (
            _seal_load_ciphertext,
            _seal_to_bytes,
        )

        mode = payload.get("mode", "add_mask")
        if mode == "rms_local_parity":
            return cls._rms_local_parity(state, payload)
        if mode == "rms_recover_and_mask_v2":
            return cls._rms_recover_and_mask_v2(state, payload)
        if mode == "rms_recover_and_mask":
            return cls._rms_recover_and_mask(state, payload)

        ctx = state["ctx"]
        encoder = state["encoder"]
        evaluator = state["evaluator"]
        shares = state["shares"]
        poly_degree = state["poly_degree"]

        ct_bytes: List[bytes] = payload.get("ct_bytes") or []
        t_flats: List[int] = payload.get("t_flats") or []
        step: int = int(payload.get("step", 0))

        ct_list: List[bytes] = [b""] * len(ct_bytes)
        for i, selected in enumerate(ct_bytes):
            if not selected:
                continue
            t_flat = int(t_flats[i]) if i < len(t_flats) else i
            ct = _seal_load_ciphertext(ctx, selected)
            r_t = shares.generate_mask_ints(step, t_flat)
            # Pad to poly_degree slots if necessary (BFV requires full slot count).
            if len(r_t) < poly_degree:
                r_t = r_t + [0] * (poly_degree - len(r_t))
            else:
                r_t = r_t[:poly_degree]
            # Per docs §3.3 SVG formula: result_U = Enc(−Ṽ_y_t) + Enc(r_t).
            # r_t is generated in the *strictly open* range (−pm/2, +pm/2), so
            # passing it directly to ``encoder.encode`` keeps SEAL's centred
            # representation intact and avoids the +49151 wrap-around error
            # SEAL introduces when an input x ≥ pm/2 is internally reduced
            # mod pm.  (We previously did ``pos_r_t = x % pm`` here, which
            # forced that wrap and produced a constant +49151 residual that
            # leaked into ``g_accum``.)
            pt = encoder.encode(r_t)
            evaluator.add_plain_inplace(ct, pt)
            ct_list[i] = _seal_to_bytes(ct)
            del ct, pt, r_t
        import gc as _gc
        _gc.collect()

        return {"ct_list": ct_list}

    @classmethod
    def _rms_recover_and_mask(
        cls,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """RMS-PIR recovery + masking for a batch of tokens.

        For each token:
          recovered   = hint_parity - real_query_parity   (= Enc(-V_y))
          masked      = recovered + r_t                   (sent to M)
          new_parity  = recovered + picked_half_parity    (replenished hint)

        Args:
            payload: ``{"items": [{"hint_parity": bytes, "q_parity": bytes,
                        "half_parity": bytes, "t_flat": int, "step": int}]}``

        Returns:
            ``{"ct_list": List[bytes], "new_parities": List[bytes]}``
        """
        from shared.core.bfv_privselect_v2_adapter import (
            _seal_load_ciphertext,
            _seal_to_bytes,
        )

        ctx = state["ctx"]
        encoder = state["encoder"]
        evaluator = state["evaluator"]
        shares = state["shares"]
        poly_degree = state["poly_degree"]

        items = payload.get("items") or []
        ct_list: List[bytes] = []
        new_parities: List[bytes] = []
        for it in items:
            hint_ct = _seal_load_ciphertext(ctx, it["hint_parity"])
            q_ct = _seal_load_ciphertext(ctx, it["q_parity"])
            half_ct = _seal_load_ciphertext(ctx, it["half_parity"])
            recovered = evaluator.sub(hint_ct, q_ct)       # Enc(-V_y)
            t_flat = int(it["t_flat"])
            step = int(it.get("step", 0))
            r_t = shares.generate_mask_ints(step, t_flat)
            if len(r_t) < poly_degree:
                r_t = r_t.tolist() + [0] * (poly_degree - len(r_t))
            else:
                r_t = r_t.tolist()[:poly_degree]
            pt = encoder.encode(r_t)
            masked = evaluator.add_plain(recovered, pt)    # Enc(-V_y + r_t)
            ct_list.append(_seal_to_bytes(masked))
            new_par = evaluator.add(recovered, half_ct)    # Enc(-V_y + half)
            new_parities.append(_seal_to_bytes(new_par))
            del hint_ct, q_ct, half_ct, recovered, pt, masked, new_par, r_t
        import gc as _gc
        _gc.collect()
        return {"ct_list": ct_list, "new_parities": new_parities}

    @classmethod
    def _rms_local_parity(
        cls,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute BFV ciphertext sums over U's local encrypted DB copy.

        Used by RMS-PIR v2 for offline hint parities and per-query
        replenishment halves.  S never sees these row lists.
        """
        from shared.core.bfv_privselect_v2_adapter import (
            _seal_load_ciphertext,
            _seal_to_bytes,
        )
        enc_db = state.get("enc_db")
        if enc_db is None:
            raise RuntimeError("rms_local_parity requires rms_db_path (U local DB)")
        row_lists = [
            [int(i) for i in lst] for lst in payload.get("row_lists", [])
        ]
        parities: List[bytes] = []
        for rows in row_lists:
            valid = [i for i in rows if 0 <= i < enc_db.n_entries]
            if not valid:
                from seal import BatchEncoder, Ciphertext, Encryptor
                zero = BatchEncoder(state["ctx"]).encode(
                    np.zeros(int(state["poly_degree"]), dtype=np.int64)
                )
                ct = Ciphertext()
                Encryptor(state["ctx"], state["public_key"]).encrypt(zero, ct)
                parities.append(_seal_to_bytes(ct))
                del zero, ct
                continue
            acc = _seal_load_ciphertext(state["ctx"], enc_db.get_encrypted_row(valid[0]))
            for i in valid[1:]:
                ct = _seal_load_ciphertext(state["ctx"], enc_db.get_encrypted_row(i))
                state["evaluator"].add_inplace(acc, ct)
                del ct
            parities.append(_seal_to_bytes(acc))
            del acc
        import gc as _gc
        _gc.collect()
        return {"parities": parities}

    @classmethod
    def _rms_recover_and_mask_v2(
        cls,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """RMS-PIR v2 per-token: local replenish + recover + mask.

        For each token:
          1. recovered = hint_parity - q_parity          (= Enc(-V_y));
          2. masked = recovered + r_t                     (sent to M);
          3. new_parity = fresh local parity over the picked replenishment
             half PLUS y (from U's local encrypted DB).  This is the
             critical fix: the old ``recovered + half_ct`` construction
             inherited the popped hint's accumulated ciphertext noise, so
             noise grew linearly with replenishment generations and blew
             M's BFV decryption budget after a few hundred steps (RMS
             divergence).  A fresh 81-row sum keeps every hint's noise
             bounded at ~one generation, so M decrypts correctly forever.
        """
        from shared.core.bfv_privselect_v2_adapter import (
            _seal_load_ciphertext,
            _seal_to_bytes,
        )
        items = payload.get("items") or []
        ctx = state["ctx"]
        evaluator = state["evaluator"]
        encoder = state["encoder"]
        shares = state["shares"]
        poly_degree = state["poly_degree"]
        ct_list: List[bytes] = []
        new_parities: List[bytes] = []
        for it in items:
            half_rows = [int(i) for i in it.get("half_rows", [])]
            y = int(it.get("y", -1))
            hint_ct = _seal_load_ciphertext(ctx, it["hint_parity"])
            q_ct = _seal_load_ciphertext(ctx, it["q_parity"])
            recovered = evaluator.sub(hint_ct, q_ct)
            t_flat = int(it["t_flat"])
            step = int(it.get("step", 0))
            r_t = shares.generate_mask_ints(step, t_flat)
            if len(r_t) < poly_degree:
                r_t = r_t.tolist() + [0] * (poly_degree - len(r_t))
            else:
                r_t = r_t.tolist()[:poly_degree]
            pt = encoder.encode(r_t)
            masked = evaluator.add_plain(recovered, pt)
            ct_list.append(_seal_to_bytes(masked))
            # 新鲜 parity：picked half + y（本地密文库直接求和，噪声有界）
            if y >= 0 and half_rows:
                fresh_par = cls._rms_local_parity(
                    state, {"row_lists": [half_rows + [y]]}
                )["parities"][0]
                new_par = _seal_load_ciphertext(ctx, fresh_par)
            else:
                new_par = recovered
            new_parities.append(_seal_to_bytes(new_par))
            # Drop per-token SEAL handles so the glibc arena can recycle the
            # 64 KB chunks instead of stacking them across steps.
            del hint_ct, q_ct, recovered, pt, masked, new_par, r_t
        import gc as _gc
        _gc.collect()
        return {"ct_list": ct_list, "new_parities": new_parities}
