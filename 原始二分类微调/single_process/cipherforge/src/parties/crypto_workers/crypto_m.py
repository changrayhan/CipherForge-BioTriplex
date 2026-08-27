"""
CryptoMWorker — CPU worker that handles M-side ``decrypt_only``.

Responsibilities
----------------
Decrypt a batch of ciphertexts (which encode ``-V_{y_t} + R_t`` after U's
homomorphic addition) using ``sk_M``.

Privacy
-------
This worker is the **only** process that holds ``sk_M`` after
``_drop_secret_key()`` is called on the main ``BFVPrivSelectV2Backend``. The
worker therefore cannot leak ``sk_M`` because:

  * The forked worker process inherits a snapshot of the parent's heap —
    but ``_drop_secret_key()`` has already nulled it before fork.
  * The driver only sends ``sk_pem`` (a serialized form) to this worker
    during init; no other worker ever receives it.

After decryption, the worker returns the decrypted vectors to the driver,
which then plaintext-adds ``s_share`` to recover ``a_t - V_y``. The worker
itself never sees ``s_share`` or ``R_t``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class CryptoMWorker:
    """CPU worker for M-side ``decrypt_only``."""

    @classmethod
    def init_state(
        cls,
        bfv_sk_pem: bytes,
        bfv_pk_pem: bytes,
        poly_degree: int,
        plain_bits: int,
        scale: int,
        vec_dim: int,
    ) -> Dict[str, Any]:
        """One-time per-worker init. Builds SEAL context + decryptor."""
        from seal import (  # type: ignore
            BatchEncoder,
            Decryptor,
            PublicKey,
            SecretKey,
        )

        from src.core.bfv_privselect_v2_adapter import create_bfv_context

        ctx = create_bfv_context(poly_degree=poly_degree, plain_bits=plain_bits)
        encoder = BatchEncoder(ctx)

        # Unwrap pickled pk format produced by ``BFVPrivSelectV2Backend.get_he_pubkey_pem``.
        import pickle as _pickle
        _pk_data = _pickle.loads(bfv_pk_pem)
        pk_raw_bytes = _pk_data["pk_bytes"]

        # sk_pem is the **raw** SEAL SecretKey bytes (produced by
        # ``_serialize_sk`` → ``_seal_to_bytes(secret_key)``). Unlike pk,
        # sk is not pickled.
        fd, sk_path = tempfile.mkstemp(suffix=".sk")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(bfv_sk_pem)
            sk = SecretKey()
            sk.load(ctx, sk_path)
        finally:
            try:
                os.remove(sk_path)
            except OSError:
                pass

        fd, pk_path = tempfile.mkstemp(suffix=".pub")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(pk_raw_bytes)
            pk = PublicKey()
            pk.load(ctx, pk_path)
        finally:
            try:
                os.remove(pk_path)
            except OSError:
                pass

        decryptor = Decryptor(ctx, sk)

        logger.info(
            "[CryptoMWorker] init done: poly_degree=%d plain_bits=%d scale=%d vec_dim=%d",
            poly_degree, plain_bits, scale, vec_dim,
        )
        return {
            "ctx": ctx,
            "encoder": encoder,
            "decryptor": decryptor,
            "scale": scale,
            "vec_dim": vec_dim,
            "poly_degree": poly_degree,
        }

    @classmethod
    def handle_request(
        cls,
        state: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Decrypt a list of ciphertexts.

        Args:
            payload: ``{"ct_list": List[bytes], "scale": int, "vec_dim": int}``
                — all optional overrides; defaults come from state.

        Returns:
            ``{"decrypted": np.ndarray}`` — shape ``(n, vec_dim)``, dtype
            ``float32``.
        """
        from seal import Plaintext  # type: ignore

        from src.core.bfv_privselect_v2_adapter import (
            _seal_load_ciphertext,
            decode_ints_as_vector,
        )

        ctx = state["ctx"]
        encoder = state["encoder"]
        decryptor = state["decryptor"]
        scale = int(payload.get("scale", state["scale"]))
        vec_dim = int(payload.get("vec_dim", state["vec_dim"]))

        ct_list: List[bytes] = payload.get("ct_list", []) or []
        out = np.zeros((len(ct_list), vec_dim), dtype=np.float32)

        decoded_pt = Plaintext()
        for i, ct_bytes in enumerate(ct_list):
            if not ct_bytes:
                continue
            ct = _seal_load_ciphertext(ctx, ct_bytes)
            decryptor.decrypt(ct, decoded_pt)
            ints = list(encoder.decode(decoded_pt))
            v = decode_ints_as_vector(ints, scale=scale)
            if v.size < vec_dim:
                v = np.pad(v, (0, vec_dim - v.size))
            out[i] = v[:vec_dim]

        return {"decrypted": out}
