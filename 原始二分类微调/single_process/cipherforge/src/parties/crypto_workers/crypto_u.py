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
    ) -> Dict[str, Any]:
        """One-time per-worker init. Builds the SEAL context and PRG state."""
        from seal import (  # type: ignore
            BatchEncoder,
            Evaluator,
            PublicKey,
            SEALContext,
        )

        from src.core.bfv_privselect_v2_adapter import (
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
                        "step": int}`` — one selected row per valid token.

        Returns:
            ``{"ct_list": List[bytes]}`` — one masked ciphertext per token.
        """
        from seal import Plaintext  # type: ignore

        from src.core.bfv_privselect_v2_adapter import (
            _seal_load_ciphertext,
            _seal_to_bytes,
        )

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

        return {"ct_list": ct_list}
