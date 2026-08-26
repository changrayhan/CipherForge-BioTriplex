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
        self.rms_store = None
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
        self._dp_logged = False

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
        """Validation forward (no gradient, no DP noise).

        ``forward_train`` privatizes H_U when dp_enable=true; validation must
        NOT do the same, otherwise val metrics measure the noise instead of
        the real model (it silently underreports/inverts AUPRC). The U shard
        is also switched to eval() so dropout etc. are disabled.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                H_U = self._u_forward(input_ids, attention_mask)
        finally:
            if was_training:
                self.model.train()
        # 验证路径坚决不加 DP 噪声：评估必须反映真实模型能力。
        return {"H_U": H_U.detach()}

    def _maybe_privatize(self, H_U: torch.Tensor, batch: Dict, *, stage: str) -> torch.Tensor:
        """Hook the dχ privatiser into the U→M forwarding pipeline.

        Returns either ``H_U`` unchanged (privatiser disabled / failure) or
        the privatised tensor ``H_tilde``.  Failures are logged and the
        clean ``H_U`` is forwarded so training is never broken by DP issues.

        ClinVar adaptation: the trainer batch carries ``output_ids`` (raw
        token ids, -100 = ignore), while ``H15Privatizer`` expects a class
        label tensor in ``batch["labels"]``.  We map the answer tokens to
        binary classes (Yes=1 / No=0, -100 elsewhere) so the CTI and the
        answer-position noise semantics work correctly.
        """
        priv = getattr(self, "h15_privatizer", None)
        if priv is None:
            return H_U
        try:
            dp_batch = dict(batch) if isinstance(batch, dict) else batch
            if isinstance(dp_batch, dict) and "labels" not in dp_batch:
                lab = dp_batch.get("output_ids")
                if lab is not None:
                    cls = torch.full_like(lab, -100)
                    # Generic multi-class mapping (BioTriplex 7/21-class): the
                    # config maps answer tokens (e.g. the letter of " j)") to
                    # class indices. Falls back to the ClinVar Yes/No binary
                    # mapping when the config has no answer_token_to_class.
                    tok2cls = self.config.get("answer_token_to_class") or {}
                    if tok2cls:
                        for tok_s, cls_s in tok2cls.items():
                            tok = int(tok_s)
                            clsi = int(cls_s)
                            cls = torch.where(
                                lab == tok, torch.full_like(lab, clsi), cls)
                    else:
                        yes_id = int(self.config.get("yes_token_id", -1))
                        no_id = int(self.config.get("no_token_id", -1))
                        cls = torch.where(lab == yes_id, torch.ones_like(lab), cls)
                        cls = torch.where(lab == no_id, torch.zeros_like(lab), cls)
                    dp_batch["labels"] = cls
                else:
                    dp_batch["labels"] = None
            H_tilde, audit = priv(H_U, dp_batch, stage=stage)
        except Exception as exc:
            logger.warning("[PartyU] privatizer failed (%s); passing clean H_U", exc)
            return H_U
        self._last_dp_audit = audit
        if (
            audit is not None
            and getattr(audit, "activated", False)
            and not getattr(self, "_dp_logged", False)
        ):
            self._dp_logged = True
            logger.info(
                "[PartyU] DP activated: alpha=%.3f eta_ctx=%.3g eta_ans=%.3g "
                "noise_l2_ctx=%.3g noise_l2_ans=%.3g",
                getattr(audit, "alpha", 0.0),
                getattr(audit, "eta_used_context", 0.0),
                getattr(audit, "eta_used_answer", 0.0),
                getattr(audit, "noise_l2_context", 0.0),
                getattr(audit, "noise_l2_answer", 0.0),
            )
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
    def _pir_dummy_weights(self) -> List[tuple]:
        """Dummy sampling distribution, aligned to the real label marginal.

        Config key ``pir_dummy_weights`` is a list of ``[token_id, weight]``
        pairs computed by the coordinator from the actual train labels (e.g.
        [[29871, 0.5], [3869, 0.25], [1939, 0.25]]).  Falling back to uniform
        over the vocabulary keeps the old behaviour when the key is absent.
        """
        weights = self.config.get("pir_dummy_weights")
        if weights:
            total = sum(float(w) for _, w in weights)
            if total > 0:
                return [(int(t), float(w) / total) for t, w in weights]
        vocab = int(self.config.get("vocab_size", 32000))
        return [(i, 1.0 / vocab) for i in range(vocab)]

    def _sample_weighted_rows(self, k: int, rng) -> List[int]:
        """Draw ``k`` rows iid from the label-aligned dummy distribution.

        Sampling is with replacement on purpose: the per-row marginal of the
        dummy rows is exactly the label marginal, so a real row adds only a
        small (1/block_size) deviation to the block's count vector — the
        frequency signal the server could otherwise exploit is suppressed.
        """
        dist = self._pir_dummy_weights()
        tokens = [t for t, _ in dist]
        probs = [w for _, w in dist]
        return [int(t) for t in rng.choices(tokens, weights=probs, k=k)]

    def _sample_pir_block(self, y: int, block_size: int):
        """Sample a real+dummy PIR block: ``block_size-1`` dummies drawn from
        the label-aligned marginal plus the real target ``y``, randomly
        permuted.  ``real_pos`` is tracked explicitly (dummies may legally
        coincide with ``y``).

        The returned ``(block, real_pos)`` is known only to U; S receives only
        the index set and cannot tell which row is the target (guessing
        advantage 1/block_size per query).
        """
        import secrets

        rng = secrets.SystemRandom()
        vocab = int(self.config.get("vocab_size", 32000))
        y = int(y) % vocab
        dummies = self._sample_weighted_rows(max(block_size - 1, 0), rng)
        block = dummies + [y]
        real_pos = len(block) - 1
        perm = list(range(len(block)))
        rng.shuffle(perm)
        block = [block[i] for i in perm]
        real_pos = perm.index(len(block) - 1)
        return block, real_pos

    def _sample_fake_block(self, block_size: int):
        """Sample a fully-fake query block: ``block_size`` dummy rows from the
        same label-aligned marginal (no real target inside).  U discards the
        result; S cannot distinguish fake blocks from real ones statistically.
        """
        import secrets

        rng = secrets.SystemRandom()
        return self._sample_weighted_rows(block_size, rng)

    def pir_query_mask(
        self,
        s_ref,
        y_list: List[int],
        t_flats: List[int],
        step: int,
        block_size: int = 8,
    ) -> List[bytes]:
        """Run real block PIR (v2) for a set of valid tokens and return masked cts.

        When ``pir_mode == "rms"`` the RMS-PIR backup path is used instead
        (hint-based parity queries, see :meth:`rms_query_mask`).

        For each valid token: build a real+dummy block, ask S for the encrypted
        rows of the whole block, extract the real row via U's private
        permutation, then homomorphically add the PRG mask ``r_t``.  Per
        config, dummy rows are drawn from the label-aligned marginal
        (``pir_dummy_weights``) and ``pir_fake_ratio`` fake query blocks are
        interleaved (real:fake = 8:2 by default) so that S cannot tell which
        requests carry a target.  S never learns the target row; gold labels
        never leave U.
        """
        if self.config.get("pir_mode") == "rms":
            return self.rms_query_mask(s_ref, y_list, t_flats, step)

        queries = []
        for y, t in zip(y_list, t_flats):
            block, real_pos = self._sample_pir_block(int(y), block_size)
            queries.append({
                "indices": block,
                "real_pos": real_pos,
                "t_flat": int(t),
                "fake": False,
            })

        fake_ratio = float(self.config.get("pir_fake_ratio", 0.0))
        if fake_ratio > 0:
            n_fakes = int(round(len(queries) * fake_ratio))
            # Insert fake queries at random positions so S cannot distinguish
            # them by order, but keep the relative order of real queries
            # unchanged. (Critical: real_queries must follow the same t_flat
            # order as the consensus t_flats list forwarded to S/M for share
            # computation, otherwise the PRG mask ``r_t`` mismatches between
            # U's add_mask and S's make_shares and the encrypted gradient
            # explodes.)
            import secrets
            rng = secrets.SystemRandom()
            interleaved: List[Dict] = list(queries)
            for _ in range(n_fakes):
                insert_at = rng.randint(0, len(interleaved))
                interleaved.insert(
                    insert_at,
                    {
                        "indices": self._sample_fake_block(block_size),
                        "real_pos": -1,
                        "t_flat": -1,
                        "fake": True,
                    },
                )
            queries = interleaved

        all_idx = [i for q in queries for i in q["indices"]]
        if s_ref is None:
            raise RuntimeError("PartyU.pir_query_mask: S reference not attached")
        rows = s_ref.pir_fetch_dispatch(all_idx, step=step)
        if len(rows) < len(set(all_idx)):
            raise RuntimeError("PIR block fetch returned incomplete rows")

        real_queries = [q for q in queries if q["real_pos"] >= 0]
        selected = [rows[q["indices"][q["real_pos"]]] for q in real_queries]
        if self.crypto_u_pool is None:
            raise RuntimeError(
                "CryptoUWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_u_pool() after init."
            )
        masked = self.crypto_u_pool.submit({
            "ct_bytes": selected,
            "t_flats": [q["t_flat"] for q in real_queries],
            "step": int(step),
        })
        return masked.get("ct_list") or []

    # ------------------------------------------------------------------------- #
    #  RMS-PIR (backup mode): hint-based parity queries
    # ------------------------------------------------------------------------- #
    def rms_query_mask(
        self,
        s_ref,
        y_list: List[int],
        t_flats: List[int],
        step: int,
    ) -> List[bytes]:
        """RMS-PIR v2 online phase for a batch of valid tokens.

        For each token y:
          1. pop a hint containing y from the local store;
          2. build the real subset (hint minus y) + dummy subset covering all
             remaining partitions, send both (permuted) to S;
          3. plan replenishment locally (next hint ID's two halves are summed
             by U's own crypto worker over U's local encrypted DB copy);
          4. recover Enc(-V_y) = hint_parity - real_parity, add r_t, and build
             the replenished hint's parity (picked half + Enc(-V_y)).

        S only ever sees the two online subsets (row-index lists); it never
        sees hint subsets or replenishment halves — the condition that makes
        RMS-PIR's multi-query privacy (paper §3.5) hold.
        """
        from ..core.rms_pir import hint_half_rows, pick_replenish_half

        store = self.rms_store
        if store is None:
            raise RuntimeError("RMS mode enabled but rms_store not attached")

        tokens = []
        row_lists = []
        def _auto_replenish(bad_y: int) -> None:
            """Generate a fresh hint for ``bad_y`` from the U-local DB and
            register it in the pool. Called when pop_hint runs out of hints
            for that label, e.g. when 29871 shows up 7+ times in one batch
            and exhausts the 20-element minimum-coverage top-up.

            Correctness fix: the replenished hint's parity must include the
            extra index ``bad_y`` (Enc(-V_y)) — otherwise a later query that
            pops this hint recovers Enc(-V_y) = hint_parity - real_parity = 0
            and corrupts the gradient.  We therefore ask the local worker for
            the parity of ``picked_half + [bad_y]`` in one call.
            """
            from ..core.rms_pir import hint_half_rows, pick_replenish_half
            J = store.next_j
            store.next_j += 1
            rows_a, rows_b, _, _ = hint_half_rows(
                store.seed, store.params, J,
            )
            picked_rows, picked_idx = pick_replenish_half(
                store.params, rows_a, rows_b, bad_y,
            )
            picked_list = list(picked_rows.values())
            worker_out = self.crypto_u_pool.submit({
                "mode": "rms_local_parity",
                "row_lists": [picked_list + [bad_y]],
            })
            pars = worker_out.get("parities") or []
            if len(pars) < 1:
                raise RuntimeError(
                    f"rms_local_parity returned {len(pars)} parities (need 1)"
                )
            store.add_hint(J, picked_rows, bad_y, pars[0])

        for y, t in zip(y_list, t_flats):
            y = int(y)
            # Pop with auto-replenish: when pool is empty for label y,
            # synthesise one extra hint locally and retry up to N times.
            for _attempt in range(64):
                try:
                    j, rows, extra, hint_parity = store.pop_hint(y)
                    break
                except RuntimeError as e:
                    if "pool empty" not in str(e):
                        raise
                    _auto_replenish(y)
            else:
                # Exhausted retries — treat as fatal.
                raise RuntimeError(
                    f"RMS: could not replenish hint for label {y} after 64 "
                    "auto-attempts"
                )
            first, second, perm = store.build_query(j, rows, extra, y)
            J, half_a, half_b = store.plan_replenish(y)
            row_lists += [first, second]
            tokens.append({
                "j": j, "J": J, "y": y, "t_flat": int(t), "perm": perm,
                "hint_parity": hint_parity,
                "half_a": half_a, "half_b": half_b,
            })

        result = s_ref.action(
            "rms_parity", {"row_lists": row_lists}, stage="BACKWARD", step=step,
        )
        parities = result.get("parities") or []
        if len(parities) != len(row_lists):
            raise RuntimeError(
                f"rms_parity returned {len(parities)} parities for {len(row_lists)} lists"
            )
        import base64
        parities = [base64.b64decode(p) for p in parities]

        items = []
        picked_info = []
        for idx, tok in enumerate(tokens):
            base = idx * 2
            q_parity = parities[base] if tok["perm"] == 0 else parities[base + 1]
            rows_a, rows_b, _, _ = hint_half_rows(store.seed, store.params, tok["J"])
            picked_rows, picked_idx = pick_replenish_half(
                store.params, rows_a, rows_b, tok["y"]
            )
            picked_rows_list = (
                tok["half_a"] if picked_idx == 0 else tok["half_b"]
            )
            items.append({
                "hint_parity": tok["hint_parity"],
                "q_parity": q_parity,
                "half_rows": picked_rows_list,
                "y": int(tok["y"]),
                "t_flat": tok["t_flat"],
                "step": int(step),
            })
            picked_info.append((tok["J"], tok["y"], picked_rows))

        if self.crypto_u_pool is None:
            raise RuntimeError(
                "CryptoUWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_u_pool() after init."
            )
        pool = self.crypto_u_pool
        n_workers = max(1, int(getattr(pool, "n_workers", 1)))
        ct_list: List[bytes] = []
        new_parities: List[bytes] = []
        if n_workers > 1 and len(items) > 1:
            import math
            chunk = int(math.ceil(len(items) / n_workers))
            futures = [
                pool.submit_async({
                    "mode": "rms_recover_and_mask_v2",
                    "items": items[s:s + chunk],
                })
                for s in range(0, len(items), chunk)
            ]
            for fut in futures:
                out = fut.get()
                ct_list += out.get("ct_list") or []
                new_parities += out.get("new_parities") or []
        else:
            worker_out = pool.submit({
                "mode": "rms_recover_and_mask_v2",
                "items": items,
            })
            ct_list = worker_out.get("ct_list") or []
            new_parities = worker_out.get("new_parities") or []
        if len(new_parities) != len(picked_info):
            raise RuntimeError(
                f"rms worker returned {len(new_parities)} new parities for "
                f"{len(picked_info)} tokens"
            )
        for (J, y, picked_rows), parity in zip(picked_info, new_parities):
            store.add_replenished(J, y, picked_rows, parity)
        return ct_list

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
