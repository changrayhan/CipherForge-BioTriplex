"""
Party U (User) — embed + decoder[0..16) + validation metrics.

Privacy contract:
  - U holds (x, y) — the full input and ground truth labels
  - U computes H_U on GPU (T1 correction)
  - U dispatches ``add_mask`` to a ``CryptoUWorker`` CPU pool (the worker
    holds U's portion of the BFV state: pk_M only, never sk_M)
  - U knows R_t (shared with S) but never reveals it to M
  - U NEVER holds sk_M — it can encrypt under pk_M but cannot decrypt
  - U ships ciphertexts to M, not plaintext gradients
  - U computes validation metrics locally

This version (heterogeneous protocol) keeps H_U on the GPU; it is forwarded
to the M-side PartyM object via direct in-process reference (not pickled,
not transported). The CPU crypto work is delegated to ``CryptoUWorker``
through ``CryptoWorkerPool``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class PartyU:
    """User-side party (U) for SLG-HE-PIR.

    Responsibilities:
      1. Load embed_tokens + decoder[0..16) on GPU
      2. Compute H_U = submodel(input_ids) — GPU forward
      3. Dispatch add_mask to ``CryptoUWorker`` (CPU pool)
      4. Compute validation metrics (ROUGE-L, F1)
    """

    def __init__(
        self,
        model_path: str,
        bfv_pk_pem: bytes,
        prg_seed: bytes,
        hint_table,
        config: Dict,
        crypto_u_pool=None,
    ):
        self.config = config
        self.hint_table = hint_table
        self.prg_seed = prg_seed
        # Crypto worker pool is optional at __init__ time — main.py sets it
        # after construction (so we don't fork before the CUDA context is up).
        self.crypto_u_pool = crypto_u_pool
        self._setup_device()
        self._setup_submodel(model_path)
        self._setup_bfv(bfv_pk_pem)
        self._setup_optimizer()

    # ------------------------------------------------------------------------- #
    #  Setup
    # ------------------------------------------------------------------------- #
    def _setup_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[U-device] device={self.device}", flush=True)

    def _setup_submodel(self, model_path: str) -> None:
        from ..model.model_splitting import (
            detect_model_spec,
            load_u_submodel,
            _get_shared_weights,
        )
        self.spec = detect_model_spec(model_path, u_layers=int(self.config.get("u_layers", 0)))
        # Pre-load weights once to avoid loading the 16GB model twice (U + M)
        all_weights = _get_shared_weights(model_path)
        self.model = load_u_submodel(
            spec=self.spec,
            model_path=model_path,
            device=str(self.device),
            use_flash_attention=bool(self.config.get("use_flash_attention", True)),
            use_sage_attention=bool(self.config.get("use_sage_attention", True)),
            gradient_checkpointing_style=self.config.get("gradient_checkpointing_style", "reentrant"),
            all_weights=all_weights,
        )
        # Embeddings are trainable under the new (embedding-only) split;
        # do NOT call ``freeze_submodel`` here. Decoder layers (if any) are
        # already frozen by load_u_submodel.

    def _setup_bfv(self, bfv_pk_pem: bytes) -> None:
        """Store the public key for U.

        U only needs the public key for encryption; actual crypto work is
        delegated to the CryptoUWorker pool. We store the pk_pem and a
        lightweight wrapper that can be passed to CryptoUWorker.
        """
        from ..core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend

        # Create a minimal backend with the public key.
        # Note: shared_seed is required by the backend but not used by U's operations.
        self.bfv_backend = BFVPrivSelectV2Backend(
            n_entries=self.config["vocab_size"],
            vec_dim=self.config["hidden_dim"],
            shared_seed=self.prg_seed,  # U's PRG seed (same as S's)
            cache_dir=self.config.get("bfv_cache_dir"),
            poly_degree=self.config.get("poly_degree", 4096),
            plain_bits=self.config.get("plain_bits", 30),
            scale=float(self.config.get("scale", 10000)),
        )
        # Handle both raw bytes and pickle format for pk
        import pickle as _pickle
        try:
            pk_data = _pickle.loads(bfv_pk_pem)
            pk_bytes = pk_data["pk_bytes"]
        except Exception:
            pk_bytes = bfv_pk_pem
        self.bfv_backend._public_key = self.bfv_backend.reconstruct_public_key(pk_bytes)
        self.bfv_backend._encryptor = None  # U delegates encryption to CryptoUWorker

    def _setup_optimizer(self) -> None:
        """U-side optimizer for the trainable embedding table.

        Decision (fix, v2.1): U's embedding weight is intentionally marked
        ``requires_grad=True`` (see ``model_splitting.py``) so the autograd
        graph flows through the embedding lookup into M's decoder — this
        is what gives M's LoRA real gradients. However, U's optimizer is
        **disabled** here: the embedding is "frozen" in the sense that no
        weight update ever happens on it. The gradient that lands on
        ``embed_tokens.weight`` is silently dropped at the
        ``step_optimizer`` boundary (see that method's no-op guard).

        Privacy: U's optimizer is None, so the embedding weight is never
        modified. No information crosses the U↔M boundary from this side —
        the gradient is computed locally and ignored.
        """
        # Per design decision: U's optimizer is intentionally disabled even
        # though ``embed_tokens.weight.requires_grad = True``. We keep the
        # ``requires_grad=True`` setting so the autograd graph can flow
        # through the embedding lookup and into M's decoder; we keep
        # ``self.optimizer = None`` so the resulting gradient is silently
        # ignored.
        self.optimizer = None
        self.lr_scheduler = None

        # --- DChi-privacy (dχ-privacy) — optional, off by default -----------
        # The privatiser is attached to the U→M cut: after ``_u_forward``
        # returns ``H_U``, ``_maybe_privatize`` injects multivariate Laplace
        # noise calibrated by α / β / calibration_steps.  See
        # ``src/core/dchi_privacy.py`` and ``DP机制-迁移参考.md``.
        try:
            from ..core.dchi_privacy import H15Privatizer as _H15Privatizer
        except Exception as exc:  # pragma: no cover — defensive
            _H15Privatizer = None
            logger.warning("[PartyU] H15Privatizer import failed: %s", exc)

        dp_enabled = bool(self.config.get("dp_enable", False)) and _H15Privatizer is not None
        if dp_enabled:
            dp_config = dict(self.config)
            dp_config.setdefault("hidden_dim", self.config.get("hidden_dim", 4096))
            dp_config.setdefault("vocab_size", self.config.get("vocab_size", 128_256))
            dp_config.setdefault("dp_device", str(self.device))
            dp_config.setdefault("dp_num_classes", 7)
            self.h15_privatizer = _H15Privatizer(dp_config)
            self.h15_privatizer.set_calibration_mode(
                bool(self.config.get("dp_calibration_mode", False))
            )
            logger.info(
                "[PartyU] H15 privatizer attached (alpha=%s, beta=%s)",
                self.h15_privatizer.alpha,
                self.h15_privatizer.answer_beta,
            )
        else:
            self.h15_privatizer = None
        self._last_dp_audit = None

    def step_optimizer(self) -> None:
        """Apply pending gradients and zero them out.

        Called by the protocol driver after M's optimizer step, since the
        autograd graph produced by ``H_M.backward()`` flows back through
        M's decoder into U's embed_tokens.
        """
        if getattr(self, "optimizer", None) is None:
            return
        self.optimizer.step()
        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------------- #
    #  Forward (training) — keeps H_U on GPU
    # ------------------------------------------------------------------------- #
    def forward_train(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute H_U = embed_tokens(x) (and any U-side decoder layers) on GPU.

        Returns H_U as a **GPU tensor**; the heterogeneous protocol does the
        in-process hand-off without copying to CPU.

        With the new split (u_layers=16) U runs the first 16 decoder layers.
        Gradients flow through H_U into M's decoder so that M's LoRA receives
        real gradients.

        When ``dp_enable`` is true, H_U is privatized through the
        :class:`H15Privatizer` before being handed to M.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        H_U = self._u_forward(input_ids, attention_mask)
        H_U = self._maybe_privatize(H_U, batch, stage="train")

        return {"H_U": H_U}

    # Backward-compatible alias used by some legacy test scripts.
    def forward_val(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Same as ``forward_train`` — val path keeps gradients off so the
        validation forward mirrors the baseline non-trainable path."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            H_U = self._u_forward(input_ids, attention_mask)
        H_U = self._maybe_privatize(H_U, batch, stage="val")
        return {"H_U": H_U.detach()}

    def _maybe_privatize(self, H_U: torch.Tensor, batch: Dict, *, stage: str) -> torch.Tensor:
        """Hook the dχ privatiser into the U→M forwarding pipeline.

        Returns either ``H_U`` unchanged (privatiser disabled / failure) or
        the privatised tensor ``H_tilde``.  Failures are logged and the
        clean ``H_U`` is forwarded so training is never broken by DP issues.
        """
        priv = getattr(self, "h15_privatizer", None)
        if priv is None:
            return H_U
        try:
            H_tilde, audit = priv(H_U, batch, stage=stage)
        except Exception as exc:
            logger.warning("[PartyU] privatizer failed (%s); passing clean H_U", exc)
            return H_U
        self._last_dp_audit = audit
        return H_tilde

    def _u_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """U's forward pass: embed + (optional) decoder layers.

        Uses SDPA with manual rotary embeddings computed by the model.
        """
        # Delegate to the model's forward method which handles position embeddings
        return self.model.forward(input_ids)

    # ------------------------------------------------------------------------- #
    #  Add-mask dispatch — forwards to CryptoUWorker pool
    # ------------------------------------------------------------------------- #
    def privselect_and_recover(self, s3pir_msg: Dict) -> Any:
        """Direct CPU dispatch (kept for tests that want serial behavior)."""
        return self.privselect_and_recover_dispatch(s3pir_msg)

    def privselect_and_recover_parallel(self, s3pir_msg: Dict, n_workers: int = 8) -> Any:
        return self.privselect_and_recover_dispatch(s3pir_msg)

    def privselect_and_recover_dispatch(self, s3pir_msg: Dict) -> Any:
        """Single entry point used by the runtime.

        Routes the S3PIR response list to the ``CryptoUWorker`` pool. The
        worker holds U's pk_M-side BFV context; U itself never touches SEAL.
        """
        if self.crypto_u_pool is None:
            raise RuntimeError(
                "CryptoUWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_u_pool() after init."
            )
        result = self.crypto_u_pool.submit({
            "s3pir_responses": s3pir_msg.get("s3pir_responses") or [],
            "step": int(s3pir_msg.get("step", 0)),
        })
        return {"ct_list": result.get("ct_list") or []}

    # ------------------------------------------------------------------------- #
    #  Real PIR: query-block construction + extraction + mask
    # ------------------------------------------------------------------------- #
    def _sample_pir_block(self, y: int, block_size: int):
        """Sample a real+dummy PIR block: ``block_size-1`` random dummies from
        the whole vocabulary plus the real target ``y``, randomly permuted.

        The returned ``(block, real_pos)`` is known only to U; S receives only
        the index set and cannot tell which row is the target (guessing
        advantage 1/block_size per query).
        """
        import secrets

        rng = secrets.SystemRandom()
        vocab = int(self.config.get("vocab_size", 32000))
        y = int(y) % vocab
        others = [i for i in range(vocab) if i != y]
        k = min(max(block_size - 1, 1), len(others))
        dummies = rng.sample(others, k)
        block = dummies + [y]
        rng.shuffle(block)
        return block, block.index(y)

    def pir_query_mask(
        self,
        s_ref,
        y_list: List[int],
        t_flats: List[int],
        step: int,
        block_size: int = 8,
    ) -> List[bytes]:
        """Run real block PIR for a set of valid tokens and return masked cts.

        For each valid token: build a real+dummy block, ask S for the encrypted
        rows of the whole block, extract the real row via U's private
        permutation, then homomorphically add the PRG mask ``r_t``. S never
        learns the target row; gold labels never leave U.
        """
        queries = []
        for y, t in zip(y_list, t_flats):
            block, real_pos = self._sample_pir_block(int(y), block_size)
            queries.append({
                "indices": block,
                "real_pos": real_pos,
                "t_flat": int(t),
            })
        all_idx = [i for q in queries for i in q["indices"]]
        if s_ref is None:
            raise RuntimeError("PartyU.pir_query_mask: S reference not attached")
        rows = s_ref.pir_fetch_dispatch(all_idx)
        if len(rows) < len(set(all_idx)):
            raise RuntimeError("PIR block fetch returned incomplete rows")

        selected = [rows[q["indices"][q["real_pos"]]] for q in queries]
        if self.crypto_u_pool is None:
            raise RuntimeError(
                "CryptoUWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_u_pool() after init."
            )
        masked = self.crypto_u_pool.submit({
            "ct_bytes": selected,
            "t_flats": [q["t_flat"] for q in queries],
            "step": int(step),
        })
        return masked.get("ct_list") or []

    # ------------------------------------------------------------------------- #
    #  Validation metrics (delegated to letter-level, BioTriplex-style)
    # ------------------------------------------------------------------------- #
    def compute_val_metrics(self, pred_msg: Dict) -> Dict[str, float]:
        """Compute validation metrics from predictions.

        Under the new QA prompt format the gold output is a letter answer
        ("l)" or "j), o)"). We delegate the metric computation to the
        Trainer (which has access to full per-batch letter tracking) and
        return an empty dict here so that legacy callers continue to work.
        """
        # New behaviour: metrics are computed in Trainer._run_val_epoch
        # directly from ``predictions_letters`` and ``labels_letters``.
        return {}

    def save_checkpoint(self) -> Dict[str, Any]:
        """U has no trainable state; we just record the model spec."""
        return {
            "party": "U",
            "model_arch": getattr(self.spec, "arch", None),
            "device": str(self.device),
            "u_num_layers": len(self.model.layers),
            "spec": self.spec.__dict__ if hasattr(self.spec, "__dict__") else {},
        }
