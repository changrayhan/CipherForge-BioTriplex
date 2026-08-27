"""
Party S (Server) — V matrix + encrypted DB + S3PIR response.

Privacy contract (design 2):
  - S holds V matrix (lm_head.weight) — never sent to other parties
  - S holds BFV encrypted database D[y] = Enc_M(-V_y)
  - S generates r_t = PRG(seed, t) — shared with U, unknown to M
  - S computes logits = H_M @ V^T and a = softmax(logits) @ V on GPU
  - S dispatches ``_respond_for_position`` to a ``CryptoSWorker`` (CPU pool)
    which generates ``s_share`` and fetches ``Enc(-V_y)`` from mmap DB
  - S never sees (x, y) — only logits and V[y]

In the heterogeneous protocol, the GPU computes the heavy matmuls (logits and
``a = softmax @ V``); the CPU worker handles the per-token BFV mmap fetch
and PRG share generation, both of which are CPU-bound.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["PartyS"]

logger = logging.getLogger(__name__)


class PartyS:
    """Server-side party (S) for SLG-HE-PIR (heterogeneous protocol)."""

    def __init__(
        self,
        lm_head_path: str,
        bfv_pk_pem: bytes,
        prg_seed: bytes,
        bfv_backend,
        hint_table,
        config: Dict,
        crypto_s_pool=None,
    ):
        self.config = config
        self.prg_seed = prg_seed
        self.bfv_backend = bfv_backend
        self.hint_table = hint_table
        self.crypto_s_pool = crypto_s_pool
        self._setup_device()
        self._setup_lm_head(lm_head_path)
        self._setup_bfv(bfv_pk_pem)

    def _setup_device(self) -> None:
        # S keeps V on GPU: per docs §3.2.1, compute_logits_gpu / compute_a_t_gpu
        # perform Z = H_M @ V.T and softmax(Z) @ V entirely on GPU.
        # CryptoSWorker (PIR / ciphertext mask) runs in CPU subprocess pool, so
        # the CPU↔GPU boundary only happens for numpy payloads between phases.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("PartyS device: %s (V matrix stays on GPU for fast matmul)", self.device)

    def _setup_lm_head(self, lm_head_path: str) -> None:
        from ..model.model_splitting import detect_model_spec, load_s_submodel
        self.spec = detect_model_spec(lm_head_path)
        self.spec.model_path = lm_head_path   # not in ModelSpec; cache for tokenizer
        self.V = load_s_submodel(
            spec=self.spec,
            model_path=lm_head_path,
            device=str(self.device),
        )
        self.V_weight = self.V.weight
        self.V_weight.requires_grad = False
        logger.info(
            "PartyS loaded V matrix: shape=%s",
            tuple(self.V_weight.shape),
        )

    def _setup_bfv(self, bfv_pk_pem: bytes) -> None:
        """Attach S-side BFV backend (no sk_M). Mostly metadata here; the
        real BFV work happens in ``CryptoSWorker``."""
        # Handle both raw bytes and pickle format
        import pickle as _pickle
        try:
            pk_data = _pickle.loads(bfv_pk_pem)
            pk_bytes = pk_data["pk_bytes"]
        except Exception:
            # Fallback: assume raw bytes
            pk_bytes = bfv_pk_pem
        # Reconstruct the public key from bytes distributed by M
        self.bfv_backend._public_key = self.bfv_backend.reconstruct_public_key(pk_bytes)
        logger.info("PartyS BFV public key set (from M)")

    # ------------------------------------------------------------------------- #
    #  Logits + a_t on GPU (heavy matmuls stay on GPU)
    # ------------------------------------------------------------------------- #
    def compute_logits_gpu(self, H_M: torch.Tensor) -> torch.Tensor:
        """Compute logits = H_M @ V^T on GPU (no fp32 intermediates).

        H_M and V are both on the same device. PyTorch's matmul does
        internal accumulation in fp32 regardless of input dtype, so we
        don't need an explicit upcast — which would otherwise allocate a
        2 GB intermediate copy of V^T.
        """
        H_M_t = H_M.to(self.device)
        V_T = self.V_weight.T
        if H_M_t.dtype != V_T.dtype:
            H_M_t = H_M_t.to(V_T.dtype)
        return torch.matmul(H_M_t, V_T)

    def compute_logits_for_eval(self, H_M: torch.Tensor) -> torch.Tensor:
        """Validation-time logits returning a CPU-resident tensor.

        Same matmul as ``compute_logits_gpu`` but explicitly moves the
        result to CPU to avoid leaking GPU memory across many val batches.
        Validation does not need the heavy ``softmax @ V`` matmul that
        ``process_logits_dispatch`` runs during training, so this is cheap.
        """
        logits = self.compute_logits_gpu(H_M)
        return logits.detach().cpu()

    def compute_a_t_gpu(self, logits: torch.Tensor) -> tuple:
        """Compute ``a_t = softmax @ V`` and ``y_t = argmax`` on GPU.

        Computes argmax first (cheap), then runs softmax + matmul in
        per-row chunks to keep memory pressure bounded. Returns:
          * a_all_flat: (B*S, hidden_dim), fp16
          * y_all: (B*S,), int64

        Note: B*S may be very large (e.g. 10000 tokens). We must NOT upcast
        the whole logits tensor to fp32 (that alone would need ~5 GB for
        10000×128256). Instead we process chunks: softmax in original dtype
        (bf16) for stability, then matmul against V_weight.
        """
        # argmax stays in logits's dtype — no precision loss.
        y_all = logits.argmax(dim=-1).flatten()
        B, S, V = logits.shape
        H = self.V_weight.shape[1]
        n_tokens = B * S
        device = logits.device

        # -------------------------------------------------------------------------
        # Memory-adaptive chunk sizing.
        #
        # We allocate intermediate buffers per chunk:
        #   probs_chunk = chunk_size × V × dtype_bytes
        #   matmul_out  = chunk_size × H × dtype_bytes
        #
        # For very long sequences we keep the chunk small enough that the
        # softmax + matmul fits comfortably. We deliberately do NOT cast
        # the full logits tensor to fp32 (which would explode memory for
        # long sequences); each chunk's softmax is computed in the original
        # dtype which is fine for bf16.
        # -------------------------------------------------------------------------
        # Estimate: keep chunk's intermediate buffers under ~1 GB
        max_chunk_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
        # probs + matmul_out per row ≈ V*dtype + H*dtype bytes
        dtype_bytes = logits.element_size()
        per_row_bytes = (V + H) * dtype_bytes
        chunk_size = max(1, min(n_tokens, max_chunk_bytes // max(per_row_bytes, 1)))
        chunk_size = min(chunk_size, 128)  # cap at 128 rows per chunk

        try:
            torch.cuda.empty_cache()
            logits_view = logits.view(n_tokens, V)
            a_all_flat = torch.empty(
                (n_tokens, H), dtype=self.V_weight.dtype, device=device
            )
            for start in range(0, n_tokens, chunk_size):
                end = min(start + chunk_size, n_tokens)
                chunk = logits_view[start:end]
                # Softmax in original dtype (bf16) — fine for stability since
                # the dominant operation is matmul against V_weight anyway.
                probs = F.softmax(chunk, dim=-1)
                a_all_flat[start:end] = torch.matmul(probs, self.V_weight)
                del probs, chunk
            del logits_view
            return a_all_flat, y_all

        except (torch.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                "GPU OOM/Error in compute_a_t_gpu — falling back to CPU "
                "(batch=%d, seq=%d, vocab=%d, hidden=%d, err=%s).",
                B, S, V, H, str(e)[:80],
            )
            torch.cuda.empty_cache()

        # -------------------------------------------------------------------------
        # CPU path: transfer logits + V to CPU in dtype-matched chunks, compute
        # softmax fp32 per chunk, then copy result back to GPU.
        #
        # We allocate result tensor on CPU first to avoid GPU OOM, then move to GPU.
        # -------------------------------------------------------------------------
        chunk_size = max(1, min(n_tokens, 64))
        # Allocate on CPU to avoid GPU OOM during fallback
        a_all_flat_cpu = torch.empty(
            (n_tokens, H), dtype=torch.float32, device="cpu"
        )
        # Copy logits to CPU before deleting GPU tensor
        logits_cpu = logits.to(device="cpu", dtype=torch.float32)   # (B, S, V) fp32 on CPU
        # Delete logits from GPU to free memory
        del logits
        torch.cuda.empty_cache()
        V_cpu = self.V_weight.to(device="cpu", dtype=torch.float32)  # (V, H) fp32 on CPU
        logits_cpu_view = logits_cpu.view(n_tokens, V)

        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            chunk_cpu = logits_cpu_view[start:end]          # (chunk, V) fp32 on CPU
            probs_cpu = F.softmax(chunk_cpu, dim=-1)       # fp32 softmax on CPU
            chunk_result = torch.matmul(probs_cpu, V_cpu)   # (chunk, H) fp32 on CPU
            a_all_flat_cpu[start:end] = chunk_result
            del probs_cpu, chunk_cpu, chunk_result

        del logits_cpu, V_cpu, logits_cpu_view
        # Move result to GPU and convert to expected dtype
        a_all_flat = a_all_flat_cpu.to(self.V_weight.device, dtype=self.V_weight.dtype)
        del a_all_flat_cpu
        torch.cuda.empty_cache()

        return a_all_flat, y_all

    # ------------------------------------------------------------------------- #
    #  S3PIR response + s_share (delegates to CryptoSWorker)
    # ------------------------------------------------------------------------- #
    def process_logits_dispatch(self, payload: Dict) -> Dict[str, Any]:
        """Training-time entry point — **label-free**.

        S computes the softmax-weighted embedding ``a_t`` and the complementary
        share ``s_share = a_t - r_t`` for EVERY position, without knowing which
        token is supervised (gold labels never leave U). The encrypted row for
        the gold token is retrieved separately via real block PIR
        (``pir_fetch_dispatch``), where S only sees a real+dummy block and
        cannot tell which row is the target.

        Args:
            payload: ``{"H_M": Tensor, "step": int,
                        "share_positions": Optional[List[int]],
                        "monitor_positions": Optional[List[int]]}``
                — ``share_positions`` are the supervised flat positions for
                  which S computes ``s_share`` (S learns *positions*, never
                  the label values).

        Returns:
            ``{"s_shares": List[List[int]],       # all B*S positions
                "monitor_p_yes": Optional[List[float]],
                "monitor_positions": Optional[List[int]],
                "n_tokens": int, "step": int}``
        """
        H_M = payload["H_M"]
        step = int(payload.get("step", 0))
        share_positions = payload.get("share_positions")
        monitor_positions = payload.get("monitor_positions")

        # 1. GPU matmul: logits = H_M @ V^T
        logits = self.compute_logits_gpu(H_M)
        # 2. GPU matmul: a = softmax @ V, y = argmax
        a_all_flat, y_all = self.compute_a_t_gpu(logits)

        B, S, V_dim = logits.shape
        n_tokens = B * S

        # 3. Materialize a_t on CPU for the worker (this is the only GPU→CPU
        # copy). Shares are computed only for the supervised positions
        # requested by U, so the PRG cost scales with n_pir, not B*S.
        # V.weight is loaded as bfloat16 by transformers; numpy can't directly
        # convert bfloat16 → float32 (unsupported ScalarType on torch 2.x
        # NumPy bridge), so we cast to float32 on GPU first.
        a_all_cpu = (
            a_all_flat.detach().to(torch.float32).cpu().numpy().astype("float32")
        )
        if share_positions is not None:
            share_pos = [int(i) for i in share_positions]
            sel = share_pos
        else:
            sel = list(range(n_tokens))

        # 4. Dispatch share generation to CryptoSWorker pool (CPU).
        if self.crypto_s_pool is None:
            raise RuntimeError(
                "CryptoSWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_s_pool() after init."
            )
        result = self.crypto_s_pool.submit({
            "mode": "make_shares",
            "a_t_list": [a_all_cpu[i] for i in sel],
            "t_flats": sel,
            "step": step,
        })

        # 5. Optional label-free training monitor: P(Yes) at the answer-adjacent
        # positions requested by U (same convention as evaluate_auprc.py).
        monitor_p_yes = None
        if monitor_positions is not None and len(monitor_positions) == B:
            try:
                if not hasattr(self, "_tokenizer"):
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.spec.model_path, trust_remote_code=True, use_fast=True,
                    )
                    if self._tokenizer.pad_token is None:
                        self._tokenizer.pad_token = self._tokenizer.eos_token
                yes_id = self._tokenizer("Yes", add_special_tokens=False).input_ids[0]
                no_id = self._tokenizer("No", add_special_tokens=False).input_ids[0]
                pos = torch.tensor(
                    monitor_positions, dtype=torch.long, device=logits.device,
                )
                rows = torch.arange(B, device=logits.device)
                z = logits[rows, pos]  # (B, V)
                sc = torch.stack([z[:, yes_id], z[:, no_id]], dim=1).float()
                monitor_p_yes = torch.softmax(sc, dim=1)[:, 0].cpu().tolist()
            except Exception:
                monitor_p_yes = None

        return {
            "s_shares": result.get("s_shares") or [],
            "monitor_p_yes": monitor_p_yes,
            "monitor_positions": monitor_positions,
            "n_tokens": n_tokens,
            "step": step,
        }

    # ------------------------------------------------------------------------- #
    #  Real PIR block serving (label-free; S never learns the target row)
    # ------------------------------------------------------------------------- #
    def pir_fetch_dispatch(self, indices: List[int]) -> Dict[int, bytes]:
        """Return encrypted rows for a real+dummy query block.

        U builds the block (target row y hidden among ``block_size-1`` random
        dummies) and sends only the *index set*; S returns the encrypted rows
        for every index in the block. S cannot tell which index is the real
        target, giving 1/block_size guessing advantage per query.
        """
        indices = [int(i) for i in indices]
        if self.crypto_s_pool is None:
            raise RuntimeError(
                "CryptoSWorker pool not attached; "
                "HeterogeneousProtocol must call set_crypto_s_pool() after init."
            )
        result = self.crypto_s_pool.submit({
            "mode": "fetch_rows",
            "indices": indices,
            "step": 0,
        })
        rows = result.get("rows") or []
        if len(rows) != len(indices):
            raise RuntimeError(
                f"PIR block fetch: got {len(rows)} rows for {len(indices)} indices"
            )
        return {idx: b for idx, b in zip(indices, rows)}

    # ------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------- #
    #  Task-aware option-letter projection helpers
    # ------------------------------------------------------------------------- #
    def _get_option_token_ids(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Map ``['a','b','c','d','e','f','g']`` to single token ids of the LM head.

        Mirrors the baseline strategy from
        ``baseline/classification_genrel/scripts/infer_and_save.py:60-78``:
        for each letter try ``f"{letter})"``, ``letter``, ``f" {letter})"``,
        ``f" {letter}"`` and pick the first encoding that is a *single* token.

        Cached on ``self._option_token_ids_cache`` keyed by tokenizer identity.
        Returned tensor is shaped ``[7]`` long; ``device`` controls placement.
        """
        cache_key = getattr(self.spec, "model_path", None) or "default"
        if (
            not hasattr(self, "_option_token_ids_cache")
            or self._option_token_ids_cache.get("key") != cache_key
        ):
            from transformers import AutoTokenizer
            if not hasattr(self, "_tokenizer"):
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.spec.model_path, use_fast=True
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            tokenizer = self._tokenizer
            ids: List[int] = []
            for letter in "abcdefg":
                chosen: Optional[int] = None
                for cand in (f"{letter})", letter, f" {letter})", f" {letter}"):
                    enc = tokenizer.encode(cand, add_special_tokens=False)
                    if len(enc) == 1:
                        chosen = enc[0]
                        break
                if chosen is None:
                    # Fallback: take the first id of the canonical `{letter})`
                    enc = tokenizer.encode(f"{letter})", add_special_tokens=False)
                    chosen = enc[0]
                ids.append(chosen)
            self._option_token_ids_cache = {
                "key": cache_key,
                "ids": torch.tensor(ids, dtype=torch.long),
            }
        ids_t = self._option_token_ids_cache["ids"]
        if device is not None:
            ids_t = ids_t.to(device)
        return ids_t

    @staticmethod
    def _get_last_nonpad_index(attention_mask: torch.Tensor) -> torch.Tensor:
        """Return per-row index of the *last* valid (non-pad) position.

        ``attention_mask`` shape ``[B, S]``; result shape ``[B]`` long.
        Falls back to ``S - 1`` if a row is all-zero (shouldn't happen in
        healthy validation batches).
        """
        am = attention_mask.long()
        sums = am.sum(dim=1)
        last = sums - 1
        # guard against degenerate all-zero rows
        if last.dim() == 0:
            last = last.unsqueeze(0)
        seq_len = am.size(1)
        last = torch.clamp(last, min=0, max=seq_len - 1)
        return last

    # ------------------------------------------------------------------------- #
    #  Validation: 7-class projection for classification OR greedy decoding for NER
    # ------------------------------------------------------------------------- #
    def generate_predictions(
        self,
        H_M_or_logits: Any,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "classification",
        max_new_tokens: int = 128,
    ) -> Dict[str, List[Any]]:
        """Generate predictions via the standard forward pass (validation only).

        ``task_type == 'classification'``
            Project the LM logits at the *last non-pad* position onto the
            7 option token ids (``a..g``) and return ``"a)" / "b)" / ... / "g)"``
            strings plus 7-dim logits per sample for downstream ROC AUC.

        ``task_type == 'generation'``
            Appending-style greedy decoding starting from the last prompt
            token. Returns ``predictions: List[str]`` plus ``token_ids``.

        Args:
            H_M_or_logits: hidden states ``[B, S, H]`` *or* pre-computed
                logits ``[B, S, V]``. Dict form ``{"H_M": ...}`` is also
                accepted.
            attention_mask: ``[B, S]`` (0/1). When supplied, the *last
                non-pad* position is used for classification projection.
                ``None`` falls back to ``S - 1`` (matches baseline behaviour).
            task_type: ``'classification'`` (default) or ``'generation'``.
            max_new_tokens: max generated length for the NER path.

        Returns:
            For classification:
              ``{"predictions": List[str], "logits": List[List[float]]}``
            For generation:
              ``{"predictions": List[str], "token_ids": List[List[int]],
                 "logits": None}``
        """
        if isinstance(H_M_or_logits, dict) and "H_M" in H_M_or_logits:
            H_M = H_M_or_logits["H_M"]
            logits = self.compute_logits_gpu(H_M)
        elif isinstance(H_M_or_logits, torch.Tensor):
            logits = H_M_or_logits
        else:
            raise ValueError("Unknown input type for generate_predictions")

        if task_type == "classification":
            return self._classify_from_logits(logits, attention_mask)
        elif task_type == "clinvar":
            return self._clinvar_predict_from_logits(logits, attention_mask)
        elif task_type == "generation":
            return self._greedy_decode_from_logits(
                logits, attention_mask, max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown task_type for generate_predictions: {task_type!r}")

    # ------------------------------------------------------------------------- #
    #  7-class classification projection (last non-pad position → 7 option ids)
    # ------------------------------------------------------------------------- #
    def _classify_from_logits(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, List[Any]]:
        # logits: [B, S, V] on self.device (bf16 or matching dtype)
        device = logits.device
        opt_ids = self._get_option_token_ids(device=device)  # [7] long
        if attention_mask is not None:
            last_idx = self._get_last_nonpad_index(attention_mask.to(device))  # [B]
        else:
            last_idx = torch.full(
                (logits.size(0),), logits.size(1) - 1, dtype=torch.long, device=device,
            )
        # Gather last-position logits: [B, V]
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        last_logits = logits.gather(1, gather_idx).squeeze(1)  # [B, V]
        # Project to 7 classes
        option_logits = last_logits.index_select(-1, opt_ids).float()  # [B, 7]
        best_idx = option_logits.argmax(dim=-1)  # [B]
        option_letters = [chr(ord("a") + i) for i in range(7)]
        predictions = [f"{option_letters[i]})" for i in best_idx.cpu().tolist()]
        return {
            "predictions": predictions,
            "logits": option_logits.detach().cpu().tolist(),
        }

    # ------------------------------------------------------------------------- #
    #  ClinVar binary answer prediction (last non-pad position → "Yes"/"No")
    # ------------------------------------------------------------------------- #
    def _clinvar_predict_from_logits(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, List[Any]]:
        """Predict the answer token at the last non-pad position (ClinVar).

        The ClinVar QA prompt is ``question\\n\\ninput\\n\\nAnswer:`` followed
        by the gold ``Yes``/``No`` token, so the *last non-pad* position is
        the answer position (matches ``clinvar_plain/scripts/evaluate_auprc.py``).
        Returns the decoded token strings so the trainer can compute
        accuracy / P-R against the gold ``Yes``/``No`` labels.
        """
        device = logits.device
        if attention_mask is not None:
            last_idx = self._get_last_nonpad_index(attention_mask.to(device))  # [B]
        else:
            last_idx = torch.full(
                (logits.size(0),), logits.size(1) - 1, dtype=torch.long, device=device,
            )
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        last_logits = logits.gather(1, gather_idx).squeeze(1)  # [B, V]
        try:
            if not hasattr(self, "_tokenizer"):
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.spec.model_path, trust_remote_code=True, use_fast=True,
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            # Same convention as evaluate_auprc.py: compare the Yes/No logits
            # at the last non-pad position (the argmax token there is the
            # space token, which is not what the eval wants).
            yes_id = self._tokenizer("Yes", add_special_tokens=False).input_ids[0]
            no_id = self._tokenizer("No", add_special_tokens=False).input_ids[0]
            pred_bin = last_logits[:, yes_id] >= last_logits[:, no_id]
            predictions = ["Yes" if p else "No" for p in pred_bin.cpu().tolist()]
        except Exception:
            predictions = [""] * logits.size(0)
        return {
            "predictions": predictions,
            "logits": None,
        }

    # ------------------------------------------------------------------------- #
    #  Greedy decoding for NER / generation (replaces whole-sequence argmax decode)
    # ------------------------------------------------------------------------- #
    def _greedy_decode_from_logits(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int = 128,
    ) -> Dict[str, List[Any]]:
        """Appending-style greedy decoding starting from the last prompt token.

        Baseline replace-by-char ``argmax(dim=-1)`` decode produces
        unrelated tokens because the language model has not been trained
        teacher-forcing-free at every position. Here we *append* the next
        argmax token step by step until EOS or ``max_new_tokens``.

        For each batch row:
          - Determine the response start (last non-pad prompt position).
          - At step t: pick argmax over V at position ``start + t``.
          - Stop on EOS or when ``max_new_tokens`` reached.
        Returns ``{"predictions": List[str], "token_ids": List[List[int]],
                   "logits": None}``.
        """
        device = logits.device
        B, S, V = logits.shape
        eos_id = self._get_eos_token_id()

        if attention_mask is not None:
            last_idx_cpu = self._get_last_nonpad_index(attention_mask).cpu().tolist()
        else:
            last_idx_cpu = [S - 1] * B

        # We accept extra token positions beyond S by greedy autocompletion.
        # Spec is fixed-length (S); in practice we'd extend this with a
        # decoder-only LM, but for a *validation* path we stop at S − start.
        per_batch_predictions: List[str] = []
        per_batch_token_ids: List[List[int]] = []
        for b in range(B):
            start = last_idx_cpu[b]
            gen_ids: List[int] = []
            for t in range(start + 1, S):
                with torch.no_grad():
                    next_logits = logits[b, t, :].float().cpu()
                tid = int(next_logits.argmax().item())
                gen_ids.append(tid)
                if eos_id is not None and tid == eos_id:
                    break
            # strip past `max_new_tokens`
            gen_ids = gen_ids[:max_new_tokens]
            per_batch_token_ids.append(gen_ids)
            per_batch_predictions.append(self._decode_tokens(gen_ids))

        return {
            "predictions": per_batch_predictions,
            "token_ids": per_batch_token_ids,
            "logits": None,
        }

    def _get_eos_token_id(self) -> Optional[int]:
        """Best-effort EOS id lookup against the cached tokenizer."""
        from transformers import AutoTokenizer
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_path, use_fast=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        tok = self._tokenizer
        eos = tok.eos_token_id
        if eos is None:
            return None
        return int(eos)

    def _decode_tokens(self, token_ids: List[int]) -> str:
        from transformers import AutoTokenizer
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_path, use_fast=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def save_checkpoint(self) -> Dict[str, Any]:
        """S has no trainable state; record metadata only."""
        return {
            "party": "S",
            "v_shape": tuple(self.V_weight.shape),
            "device": str(self.device),
            "note": "V is frozen; encrypted DB is on disk",
        }
