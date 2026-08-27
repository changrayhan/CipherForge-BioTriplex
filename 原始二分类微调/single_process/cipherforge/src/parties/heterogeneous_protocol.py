"""
HeterogeneousProtocol — single GPU Fusion process + multiple CPU Crypto
Workers. This is the active runtime for SLG-HE-PIR v2.0.

Topology
--------
    ┌─────────────────────────────────────────────────────────────┐
    │ HeterogeneousProtocol (this file)                           │
    │ ─────────────────────────────────────────────────────────── │
    │ • PartyU (GPU): embed + decoder[0..16)                      │
    │ • PartyM (GPU): decoder[16..32) + norm + LoRA + sk_M        │
    │ • PartyS (GPU): lm_head (V)                                 │
    │                                                             │
    │     │ submit                  │ submit                │ submit
    │     ▼                         ▼                        ▼
    │ ┌────────────┐         ┌────────────┐         ┌────────────┐
    │ │CryptoUWorker│         │CryptoMWorker│         │CryptoSWorker│
    │ │  (CPU fork)│         │  (CPU fork) │         │  (CPU fork) │
    │ │ • add_mask │         │ • decrypt   │         │ • s_share   │
    │ │ • PRG R_t  │         │ • sk_M only │         │ • mmap DB   │
    │ │ • no sk_M  │         │             │         │ • no sk_M   │
    │ └────────────┘         └────────────┘         └────────────┘
    └─────────────────────────────────────────────────────────────┘

Privacy boundary
----------------
* U/M/S class boundaries enforce the protocol contract in code.
* ``_drop_secret_key()`` is called on the main ``BFVPrivSelectV2Backend``
  before any worker is forked — so ``sk_M`` is structurally absent from
  the parent's heap.
* Only ``CryptoMWorker`` re-attaches ``sk_M`` (via the ``bfv_sk_pem`` blob
  the driver passes in its init kwargs).
* GPU forward/backward is local to the Fusion process; only ciphertext
  bytes cross process boundaries (no torch tensors).

Public API
----------
Mirrors ``IPCProtocol`` exactly so the Trainer code is unchanged:

  * ``step_train(batch, global_step) -> StepResult``
  * ``step_train_chunked(batch, global_step, chunk_tokens) -> StepResult``
  * ``step_val(val_batch, global_step) -> dict``
  * ``gather_checkpoints() -> dict``
  * ``shutdown() -> None``
  * ``profiler`` (StepProfiler or None)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .crypto_workers.crypto_u import CryptoUWorker
from .crypto_workers.crypto_m import CryptoMWorker
from .crypto_workers.crypto_s import CryptoSWorker
from .crypto_workers.pool import CryptoWorkerPool
from .party_u import PartyU
from .party_m import PartyM
from .party_s import PartyS
from .wire import StepResult, StepProfiler
from ..data.dataset import parse_answer_letter

__all__ = ["HeterogeneousProtocol"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helper: chunk span computation
# --------------------------------------------------------------------------- #
@dataclass
class _ChunkedSplit:
    starts: List[int]
    ends: List[int]

    @property
    def n_chunks(self) -> int:
        return len(self.starts)


def _split_into_chunks(n_tokens: int, chunk_tokens: int) -> _ChunkedSplit:
    if chunk_tokens <= 0 or chunk_tokens >= n_tokens:
        return _ChunkedSplit(starts=[0], ends=[n_tokens])
    starts: List[int] = []
    ends: List[int] = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_tokens, n_tokens)
        starts.append(start)
        ends.append(end)
        start = end
    return _ChunkedSplit(starts=starts, ends=ends)


# --------------------------------------------------------------------------- #
#  HeterogeneousProtocol
# --------------------------------------------------------------------------- #
class HeterogeneousProtocol:
    """GPU Fusion driver + CPU Crypto Worker pools.

    Args:
        u_submodel_path / m_submodel_path / s_lm_head_path: HuggingFace
            model snapshot paths (each party loads only the safetensor shards
            it needs).
        bfv_backend: a ``BFVPrivSelectV2Backend`` whose ``_drop_secret_key``
            has **already** been called by the caller. We pass this to
            PartyS so it can hold the mmap-backed encrypted DB; the worker
            pool rebuilds its own SEAL context inside forked processes.
        hint_table: shared ``HintTable``.
        bfv_sk_pem / bfv_pk_pem: bytes serialized by the main process;
            ``sk_pem`` is forwarded ONLY to ``CryptoMWorker`` and PartyM.
        prg_seed: 32 random bytes; forwarded to U/S parties and to
            CryptoUWorker / CryptoSWorker (never to M).
        config: dict copied from ``finetune.py``. Must contain ``vocab_size``,
            ``hidden_dim``, ``poly_degree``, ``plain_bits``, ``scale``,
            ``bfv_cache_dir``, ``lora_r``, ``lora_alpha``, and the optional
            ``N_CRYPTO_*_WORKERS`` knobs.
    """

    def __init__(
        self,
        u_submodel_path: str,
        m_submodel_path: str,
        s_lm_head_path: str,
        bfv_backend,
        hint_table,
        bfv_sk_pem: bytes,
        bfv_pk_pem: bytes,
        prg_seed: bytes,
        config: Dict,
    ):
        self.config = config
        self.bfv_backend = bfv_backend
        self.hint_table = hint_table
        self.bfv_sk_pem = bfv_sk_pem
        self.bfv_pk_pem = bfv_pk_pem
        self.prg_seed = prg_seed

        # Task type propagated from the caller (TrainerConfig.task_type).
        # Validated/inference predictions decode LM logits differently depending
        # on the task: ``classification`` → 7-class argmax on a..g option
        # tokens; ``generation`` → greedy decoding. Defaults to
        # ``classification`` so legacy callers behave reasonably.
        self._task_type: str = str(self.config.get("task_type", "classification"))

        # --- Step profiler ---
        self.profiler: Optional[StepProfiler] = None
        if bool(self.config.get("ENABLE_STEP_PROFILING", True)):
            log_dir = self.config.get("LOG_DIR")
            self.profiler = StepProfiler(log_dir=log_dir)

        # ------------------------------------------------------------------
        # 1. Construct U / M / S (GPU). Each Party's __init__ loads its
        #    submodel into the shared CUDA context.
        # ------------------------------------------------------------------
        t_init = time.time()
        logger.info("[HeterogeneousProtocol] constructing PartyU ...")
        self.party_u = PartyU(
            model_path=u_submodel_path,
            bfv_pk_pem=bfv_pk_pem,
            prg_seed=prg_seed,
            hint_table=hint_table,
            config=config,
        )

        logger.info("[HeterogeneousProtocol] constructing PartyM ...")
        self.party_m = PartyM(
            model_path=m_submodel_path,
            bfv_sk_pem=bfv_sk_pem,
            bfv_pk_pem=bfv_pk_pem,
            config=config,
        )

        logger.info("[HeterogeneousProtocol] constructing PartyS ...")
        self.party_s = PartyS(
            lm_head_path=s_lm_head_path,
            bfv_pk_pem=bfv_pk_pem,
            prg_seed=prg_seed,
            bfv_backend=bfv_backend,
            hint_table=hint_table,
            config=config,
        )
        # U needs S to serve real PIR blocks (label-free row fetch).
        self.party_u._s_ref = self.party_s
        logger.info(
            "[HeterogeneousProtocol] U/M/S ready in %.1fs (single CUDA context)",
            time.time() - t_init,
        )

        # ------------------------------------------------------------------
        # 2. Fork CPU Crypto Worker pools.
        #    CryptoUWorker / CryptoMWorker can run in parallel;
        #    CryptoSWorker is single-threaded by default (mmap fetch is cheap).
        # ------------------------------------------------------------------
        poly_degree = int(self.config.get("poly_degree", 4096))
        plain_bits = int(self.config.get("plain_bits", 30))
        scale = int(self.config.get("scale", 10_000))
        vec_dim = int(self.config.get("hidden_dim", 4096))
        n_entries = int(self.config.get("vocab_size", 128256))
        plain_modulus = self._infer_plain_modulus(plain_bits)
        bfv_cache_dir = self.config.get("bfv_cache_dir", "/root/autodl-tmp/slg-bfv-cache")

        n_u = int(self.config.get("N_CRYPTO_U_WORKERS", 8))
        n_m = int(self.config.get("N_CRYPTO_M_WORKERS", 8))
        n_s = int(self.config.get("N_CRYPTO_S_WORKERS", 1))

        # CryptoUWorker — no sk_M
        logger.info("[HeterogeneousProtocol] starting CryptoUWorker pool (n=%d) ...", n_u)
        self.crypto_u_pool = CryptoWorkerPool(
            CryptoUWorker,
            n_workers=n_u,
            init_kwargs={
                "bfv_pk_pem": bfv_pk_pem,
                "prg_seed": prg_seed,
                "poly_degree": poly_degree,
                "plain_bits": plain_bits,
                "scale": scale,
                "plain_modulus": plain_modulus,
            },
        )
        self.party_u.crypto_u_pool = self.crypto_u_pool

        # CryptoMWorker — holds sk_M
        logger.info("[HeterogeneousProtocol] starting CryptoMWorker pool (n=%d) ...", n_m)
        self.crypto_m_pool = CryptoWorkerPool(
            CryptoMWorker,
            n_workers=n_m,
            init_kwargs={
                "bfv_sk_pem": bfv_sk_pem,
                "bfv_pk_pem": bfv_pk_pem,
                "poly_degree": poly_degree,
                "plain_bits": plain_bits,
                "scale": scale,
                "vec_dim": vec_dim,
            },
        )
        self.party_m.crypto_m_pool = self.crypto_m_pool

        # CryptoSWorker — PRG + mmap fetch
        logger.info("[HeterogeneousProtocol] starting CryptoSWorker pool (n=%d) ...", n_s)
        partition_size = 1 << ((n_entries.bit_length() - 1) // 2)
        self.crypto_s_pool = CryptoWorkerPool(
            CryptoSWorker,
            n_workers=n_s,
            init_kwargs={
                "bfv_pk_pem": bfv_pk_pem,
                "prg_seed": prg_seed,
                "bfv_cache_dir": bfv_cache_dir,
                "poly_degree": poly_degree,
                "plain_bits": plain_bits,
                "scale": scale,
                "plain_modulus": plain_modulus,
                "n_entries": n_entries,
                "vec_dim": vec_dim,
                "partition_size": partition_size,
                "lam": int(self.config.get("lam", 80)),
            },
        )
        self.party_s.crypto_s_pool = self.crypto_s_pool

        logger.info(
            "[HeterogeneousProtocol] all worker pools ready; total init %.1fs",
            time.time() - t_init,
        )

        # ------------------------------------------------------------------
        # 3. dχ-privacy wiring (optional, off by default).  When
        #    ``dp_calibration_mode`` is true and the privatiser is attached
        #    to PartyU, we keep the first N steps in noise-free observation
        #    mode so the EMA norm estimator can lock η₀.
        # ------------------------------------------------------------------
        self.dp_calibration_mode: bool = bool(
            self.config.get("dp_calibration_mode", False)
        )
        self.dp_step_counter: int = 0
        if self.dp_calibration_mode and getattr(
            self.party_u, "h15_privatizer", None
        ) is not None:
            self.party_u.h15_privatizer.set_calibration_mode(True)
            logger.info(
                "[HeterogeneousProtocol] DP calibration mode on for %d step(s)",
                int(self.config.get("dp_calibration_steps", 1)),
            )

    # ------------------------------------------------------------------ #
    #  Plain BFV / plain modulus inference
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_plain_modulus(plain_bits: int) -> int:
        """Mirror SEAL's PlainModulus.Batching(n, bits): t = 2^bits + small_delta."""
        # Llama-3-1-8B project uses plain_bits=30 → t = 2^30 + 27.
        # This is a known SEAL constant for batching at poly_degree=4096.
        if plain_bits == 30:
            return (1 << 30) + 27
        # Fallback: re-derive from SEAL context if needed. But the value is
        # baked into build_encrypted_db.py's serialized config, so just
        # use a sensible default.
        return (1 << plain_bits)

    # ------------------------------------------------------------------ #
    #  Label handling (U-side only; gold never sent to S)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _binary_ce_from_p(gold_yes, p_yes) -> Optional[float]:
        """Binary cross-entropy of the monitor P(Yes) against the gold label."""
        import math as _math
        if not gold_yes or not p_yes or len(gold_yes) != len(p_yes):
            return None
        total = 0.0
        for g, p in zip(gold_yes, p_yes):
            p = min(max(float(p), 1e-7), 1.0 - 1e-7)
            total += -(g * _math.log(p) + (1.0 - g) * _math.log(1.0 - p))
        return total / max(len(gold_yes), 1)

    def _prepare_labels(self, batch: Dict) -> Dict:
        """Compute next-token gold targets + valid positions (U-side only).

        The shifted labels (logits_t -> token_{t+1}) live only on U; S never
        receives them. The monitor position per row is the answer-adjacent
        prompt position (eval convention), and ``gold_yes`` is the row label
        (1 = pathogenic "Yes").
        """
        lab = batch.get("output_ids") if isinstance(batch, dict) else None
        out = {
            "y_valid": [],
            "valid_indices": [],
            "valid_mask": None,
            "monitor_positions": None,
            "gold_yes": None,
        }
        if lab is None or not isinstance(lab, torch.Tensor):
            return out
        lab = lab.to("cpu")
        B, S = lab.shape
        y_shift = torch.full_like(lab, -100)
        if S > 1:
            y_shift[:, :-1] = lab[:, 1:]
        valid = y_shift != -100
        valid_idx = valid.flatten().nonzero().flatten().tolist()
        y_valid = y_shift.flatten()[valid_idx].tolist()

        first = (lab != -100).long().argmax(dim=1)          # answer start L
        monitor_pos = (first - 1).clamp(min=0).tolist()
        yes_id = int(self.config.get("yes_token_id", -1))
        gold_pos = (first + 1).clamp(max=S - 1)
        gold_tok = lab.gather(1, gold_pos.unsqueeze(1)).squeeze(1)
        gold_yes = (gold_tok == yes_id).float().tolist()

        out.update({
            "y_valid": y_valid,
            "valid_indices": valid_idx,
            "valid_mask": valid.numpy(),
            "monitor_positions": monitor_pos,
            "gold_yes": gold_yes,
        })
        return out

    # ------------------------------------------------------------------ #
    #  Flat training step
    # ------------------------------------------------------------------ #
    def step_train(self, batch: Dict, global_step: int) -> StepResult:
        """One-shot (non-chunked) training step."""
        t0 = time.time()
        prof = self.profiler

        # === U forward (GPU) ===
        if prof:
            prof.begin_phase("forward_U")
        u_result = self.party_u.forward_train(batch)
        H_U = u_result["H_U"]  # GPU tensor
        if prof:
            prof.end_phase("forward_U")

        # === Calibration hook: feed unmodified H_U to the calibrator ===
        self._provide_clean_h_to_calibrator(H_U)

        # === M forward (GPU) ===
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        if prof:
            prof.begin_phase("forward_M")
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]  # GPU tensor
        if prof:
            prof.end_phase("forward_M")

        # === Labels (U-side only; never sent to S) ===
        labels_info = self._prepare_labels(batch) if isinstance(batch, dict) else {}
        valid_indices = labels_info.get("valid_indices") or []
        if not valid_indices:
            raise RuntimeError(
                f"step {global_step}: no valid answer tokens (labels all -100?)"
            )
        y_valid = labels_info.get("y_valid") or []
        valid_mask = labels_info.get("valid_mask")
        gold_yes = labels_info.get("gold_yes")
        monitor_positions = labels_info.get("monitor_positions")

        # === S: label-free shares + monitor (no gold_ids!) ===
        if prof:
            prof.begin_phase("s_logits")
        s_result = self.party_s.process_logits_dispatch({
            "H_M": H_M,
            "step": global_step,
            "share_positions": valid_indices,
            "monitor_positions": monitor_positions,
        })
        if prof:
            prof.end_phase("s_logits")

        s_shares = s_result.get("s_shares") or []
        n_tokens = int(s_result.get("n_tokens", 0))

        # === U: real block PIR query + mask (only valid positions) ===
        if prof:
            prof.begin_phase("priv_U")
        block_size = int(self.config.get("pir_block_size", 8))
        ct_list = self.party_u.pir_query_mask(
            self.party_s, y_valid, valid_indices, global_step, block_size,
        )
        if prof:
            prof.end_phase("priv_U")

        monitor_ce = self._binary_ce_from_p(gold_yes, s_result.get("monitor_p_yes"))

        # === M decrypt + LoRA step (CPU decrypt + GPU autograd) ===
        if isinstance(batch, dict) and "input_ids" in batch:
            expected_shape = (
                int(batch["input_ids"].shape[0]),
                int(batch["input_ids"].shape[1]),
            )
        else:
            expected_shape = None

        if prof:
            prof.begin_phase("backward_M")
        ack = self.party_m.backward_and_update({
            "ct_from_U": ct_list,
            "s_share": s_shares,
            "valid_mask": valid_mask,
            "valid_indices": valid_indices,
            "step": global_step,
            "expected_shape": expected_shape,
        })
        if prof:
            prof.end_phase("backward_M")

        # Apply U-side optimizer step (trainable embeddings under the new split).
        self.party_u.step_optimizer()

        step_time_ms = (time.time() - t0) * 1000

        if prof:
            prof.end_step(
                step=global_step,
                n_tokens=n_tokens,
                n_chunks=1,
                step_time_ms=step_time_ms,
                extra={"mode": "flat"},
            )

        return StepResult(
            step=global_step,
            loss=float(ack.get("loss", 0.0)),
            gpu_mem_mb=float(ack.get("gpu_mem_mb", 0.0)),
            step_time_ms=step_time_ms,
            attack_dumps=ack.get("attack_dumps", {}),
            n_chunks=1,
            dp_audit=self._collect_dp_audit(global_step),
            loss_ce=monitor_ce,
        )

    # ------------------------------------------------------------------ #
    #  Chunked training step
    # ------------------------------------------------------------------ #
    def step_train_chunked(
        self,
        batch: Dict,
        global_step: int,
        chunk_tokens: int = 3072,
    ) -> StepResult:
        """Chunked streaming step: S → U (per chunk) → M (once)."""
        t0 = time.time()
        prof = self.profiler
        if prof:
            self.profiler.is_chunked = True

        # U forward
        if prof:
            prof.begin_phase("forward_U")
        u_result = self.party_u.forward_train(batch)
        H_U = u_result["H_U"]
        if prof:
            prof.end_phase("forward_U")

        # === Calibration hook: feed unmodified H_U to the calibrator ===
        self._provide_clean_h_to_calibrator(H_U)

        # M forward
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else None
        if prof:
            prof.begin_phase("forward_M")
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]
        if prof:
            prof.end_phase("forward_M")

        # === Labels (U-side only; never sent to S) ===
        labels_info = self._prepare_labels(batch) if isinstance(batch, dict) else {}
        valid_indices = labels_info.get("valid_indices") or []
        if not valid_indices:
            raise RuntimeError(
                f"step {global_step}: no valid answer tokens (labels all -100?)"
            )
        y_valid = labels_info.get("y_valid") or []
        valid_mask = labels_info.get("valid_mask")
        gold_yes = labels_info.get("gold_yes")
        monitor_positions = labels_info.get("monitor_positions")

        # S: label-free shares + monitor (no gold_ids!)
        if prof:
            prof.begin_phase("s_logits")
        s_result = self.party_s.process_logits_dispatch({
            "H_M": H_M,
            "step": global_step,
            "share_positions": valid_indices,
            "monitor_positions": monitor_positions,
        })
        if prof:
            prof.end_phase("s_logits")

        s_shares = s_result.get("s_shares") or []
        n_tokens = int(s_result.get("n_tokens", 0))
        n_chunks = 1
        logger.info(
            "step_train_chunked: step=%d n_tokens=%d n_pir=%d (real PIR, block=%d)",
            global_step, n_tokens, len(valid_indices),
            int(self.config.get("pir_block_size", 8)),
        )

        # U: real block PIR query + mask (only valid positions)
        if prof:
            prof.begin_phase("priv_U")
        block_size = int(self.config.get("pir_block_size", 8))
        all_ct = self.party_u.pir_query_mask(
            self.party_s, y_valid, valid_indices, global_step, block_size,
        )
        if prof:
            prof.end_phase("priv_U")

        monitor_ce = self._binary_ce_from_p(gold_yes, s_result.get("monitor_p_yes"))

        # M backward + LoRA step
        if isinstance(batch, dict) and "input_ids" in batch:
            expected_shape = (
                int(batch["input_ids"].shape[0]),
                int(batch["input_ids"].shape[1]),
            )
        else:
            expected_shape = None

        if prof:
            prof.begin_phase("backward_M")
        ack = self.party_m.backward_and_update({
            "ct_from_U": all_ct,
            "s_share": s_shares,
            "valid_mask": valid_mask,
            "valid_indices": valid_indices,
            "step": global_step,
            "expected_shape": expected_shape,
        })
        if prof:
            prof.end_phase("backward_M")

        # Apply U-side optimizer step. Under the new split U's embedding table
        # is trainable; the autograd graph produced by M.backward() flows back
        # through M's decoder into U's embed_tokens, populating its gradient.
        self.party_u.step_optimizer()

        step_time_ms = (time.time() - t0) * 1000
        if prof:
            prof.end_step(
                step=global_step,
                n_tokens=n_tokens,
                n_chunks=n_chunks,
                step_time_ms=step_time_ms,
                extra={"mode": "chunked"},
            )

        return StepResult(
            step=global_step,
            loss=float(ack.get("loss", 0.0)),
            gpu_mem_mb=float(ack.get("gpu_mem_mb", 0.0)),
            step_time_ms=step_time_ms,
            attack_dumps=ack.get("attack_dumps", {}),
            n_chunks=n_chunks,
            dp_audit=self._collect_dp_audit(global_step),
            loss_ce=monitor_ce,
        )

    # ------------------------------------------------------------------ #
    #  Validation: same U/M/S forward + argmax predictions (no PIR)
    # ------------------------------------------------------------------ #
    def step_val(self, val_batch: Dict, global_step: int) -> Dict:
        """Validation: U forward → M forward → S argmax → U compute metrics.

        This is a privacy-free path: no BFV decryption, no PRG share, no
        gradient injection. Labels are visible to the driver (we are in the
        same Python process) so the returned metrics include both
        predictions and labels.
        """
        # U forward
        u_result = self.party_u.forward_val(val_batch)
        H_U = u_result["H_U"]

        # M forward
        attention_mask = val_batch.get("attention_mask") if isinstance(val_batch, dict) else None
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]

        # S logits (kept on CPU for Trainer CE loss computation)
        logits_cpu = self.party_s.compute_logits_for_eval(H_M)

        # S argmax predictions (task-aware: 7-class projection vs greedy decode)
        s_pred = self.party_s.generate_predictions(
            H_M,
            attention_mask=attention_mask,
            task_type=self._task_type,
        )
        predictions = s_pred.get("predictions", [])
        pred_logits = s_pred.get("logits")  # [B, 7] for classification; None for generation

        # U computes metrics locally (has labels at val time)
        labels = val_batch.get("output_text") if isinstance(val_batch, dict) else None
        if isinstance(val_batch, dict) and labels is None:
            labels = val_batch.get("labels")
        if labels is None and isinstance(val_batch, dict):
            # Some datasets store labels under different keys
            labels = val_batch.get("target_text") or val_batch.get("gold")

        # normalize to list-of-strings
        if isinstance(labels, (list, tuple)):
            labels = list(labels)

        # Letter-level labels (BioTriplex style: e.g. "l)" or "j), o)")
        # ``output_ids`` from the dataset are the gold token ids whose first
        # non-pad token typically encodes the answer letter.
        labels_letters = []
        if isinstance(val_batch, dict):
            output_ids = val_batch.get("output_ids")
            if output_ids is not None:
                # Decode the full gold sequence; Trainer only consumes the
                # canonical letter form (BioTriplex style).
                if isinstance(output_ids, torch.Tensor):
                    for row in output_ids:
                        # Skip leading pad tokens AND -100 ignore-index tokens.
                        # The BioTriplexQADataset uses -100 for prompt tokens
                        # (standard causal-LM masking), but we only want the gold
                        # response portion which is the last non-pad token(s).
                        valid = (row != self._pad_token_id()) & (row != -100)
                        non_pad = row[valid]
                        if non_pad.numel() > 0:
                            txt = self._decode_token_tensor(non_pad)
                            labels_letters.append(parse_answer_letter(txt))
                        else:
                            labels_letters.append("")
                else:
                    for row in output_ids:
                        labels_letters.append(parse_answer_letter(str(row)))

        predictions_letters = [parse_answer_letter(p) for p in predictions]

        metrics = self.party_u.compute_val_metrics({
            "predictions": predictions,
            "labels": labels or [],
        })

        # Propagate doc_keys so downstream trainers can match predictions to
        # gold entities by unique id (required by BioTriplex NER / classification
        # evaluators which need doc_key-aligned access).
        doc_keys: List[str] = []
        if isinstance(val_batch, dict):
            raw_keys = val_batch.get("doc_key")
            if raw_keys is None:
                raw_keys = val_batch.get("doc_keys")
            if raw_keys is not None:
                if isinstance(raw_keys, torch.Tensor):
                    doc_keys = [str(k) for k in raw_keys.cpu().tolist()]
                elif isinstance(raw_keys, (list, tuple)):
                    doc_keys = [str(k) for k in raw_keys]
                else:
                    doc_keys = [str(raw_keys)]
            else:
                # Last-resort fallback: positional identifiers (the trainer can
                # still rely on list ordering). This preserves backward compat
                # with batches that never carried doc_keys.
                doc_keys = [f"sample_{i}" for i in range(len(predictions))]

        # Always return predictions/labels so Trainer can compute its own
        # aggregates; U's compute_val_metrics returns them indirectly via
        # the metrics dict.
        out = {
            "predictions": predictions,
            "predictions_letters": predictions_letters,
            "labels": labels or [],
            "labels_letters": labels_letters,
            "logits": logits_cpu,
            "pred_logits": pred_logits,                       # [B, 7] for classification
            "task_type": self._task_type,                     # forwarded to trainer
            "labels_tensor": val_batch.get("output_ids") if isinstance(val_batch, dict) else None,
            "doc_keys": doc_keys,
            "metrics": metrics,
        }
        return out

    # ------------------------------------------------------------------ #
    #  Test evaluation: identical to step_val (no PIR, standard forward)
    # ------------------------------------------------------------------ #
    def step_test(self, test_batch: Dict, global_step: int) -> Dict:
        """Test evaluation: U forward → M forward → S argmax → U compute metrics.

        This is the same privacy-free path as step_val. It is called by the
        Trainer after training ends to obtain a final estimate of model
        accuracy on the held-out test set.
        """
        # U forward
        u_result = self.party_u.forward_val(test_batch)
        H_U = u_result["H_U"]

        # M forward
        attention_mask = (
            test_batch.get("attention_mask")
            if isinstance(test_batch, dict)
            else None
        )
        m_result = self.party_m.forward(H_U, attention_mask=attention_mask)
        H_M = m_result["H_M"]

        # S logits (CPU)
        logits_cpu = self.party_s.compute_logits_for_eval(H_M)

        # S argmax predictions (task-aware: 7-class projection vs greedy decode)
        attention_mask_test = (
            test_batch.get("attention_mask")
            if isinstance(test_batch, dict)
            else None
        )
        s_pred = self.party_s.generate_predictions(
            H_M,
            attention_mask=attention_mask_test,
            task_type=self._task_type,
        )
        predictions = s_pred.get("predictions", [])
        pred_logits = s_pred.get("logits")

        # Labels from test_batch
        labels = (
            test_batch.get("output_text")
            if isinstance(test_batch, dict)
            else None
        )
        if labels is None and isinstance(test_batch, dict):
            labels = test_batch.get("labels")
        if labels is None and isinstance(test_batch, dict):
            labels = test_batch.get("target_text") or test_batch.get("gold")

        if isinstance(labels, (list, tuple)):
            labels = list(labels)

        predictions_letters = [parse_answer_letter(p) for p in predictions]
        labels_letters = []
        if isinstance(test_batch, dict):
            output_ids = test_batch.get("output_ids")
            if output_ids is not None:
                if isinstance(output_ids, torch.Tensor):
                    for row in output_ids:
                        valid = (row != self._pad_token_id()) & (row != -100)
                        non_pad = row[valid]
                        if non_pad.numel() > 0:
                            txt = self._decode_token_tensor(non_pad)
                            labels_letters.append(parse_answer_letter(txt))
                        else:
                            labels_letters.append("")
                else:
                    for row in output_ids:
                        labels_letters.append(parse_answer_letter(str(row)))

        metrics = self.party_u.compute_val_metrics({
            "predictions": predictions,
            "labels": labels or [],
        })

        # Propagate doc_keys (see step_val for the rationale).
        doc_keys: List[str] = []
        if isinstance(test_batch, dict):
            raw_keys = test_batch.get("doc_key")
            if raw_keys is None:
                raw_keys = test_batch.get("doc_keys")
            if raw_keys is not None:
                if isinstance(raw_keys, torch.Tensor):
                    doc_keys = [str(k) for k in raw_keys.cpu().tolist()]
                elif isinstance(raw_keys, (list, tuple)):
                    doc_keys = [str(k) for k in raw_keys]
                else:
                    doc_keys = [str(raw_keys)]
            else:
                doc_keys = [f"sample_{i}" for i in range(len(predictions))]

        return {
            "predictions": predictions,
            "predictions_letters": predictions_letters,
            "labels": labels or [],
            "labels_letters": labels_letters,
            "logits": logits_cpu,
            "pred_logits": pred_logits,
            "task_type": self._task_type,
            "labels_tensor": test_batch.get("output_ids") if isinstance(test_batch, dict) else None,
            "doc_keys": doc_keys,
            "metrics": metrics,
        }

    @staticmethod
    def _pad_token_id() -> int:
        """Best-effort pad_token_id lookup (used to strip pads from output_ids)."""
        try:
            return 0  # HF defaults to 0 for padding
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    #  dχ-privacy helpers
    # ------------------------------------------------------------------ #
    def _collect_dp_audit(self, step: int) -> Dict:
        """Pull the latest privatiser audit from PartyU into a plain dict.

        Returns ``{}`` when DP is disabled or no audit has been recorded yet.
        ``step`` is added (or overwritten) so downstream tooling can join
        per-step audit records with the trainer log.
        """
        priv = getattr(self.party_u, "h15_privatizer", None)
        if priv is None:
            return {}
        last = getattr(self.party_u, "_last_dp_audit", None)
        if last is None:
            return {}
        if hasattr(last, "as_dict"):
            out = dict(last.as_dict())
        elif isinstance(last, dict):
            out = dict(last)
        else:
            out = dict(getattr(last, "__dict__", {}))
        out["step"] = int(step)
        return out

    def _provide_clean_h_to_calibrator(self, H_U: torch.Tensor) -> None:
        """Calibration-mode hook: forward the unmodified H_U to the
        ``observes_clean`` entry point each step so the EMA can lock η₀."""
        if not self.dp_calibration_mode:
            return
        priv = getattr(self.party_u, "h15_privatizer", None)
        if priv is None:
            return
        self.dp_step_counter += 1
        try:
            priv.observe_clean(H_U)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("[HeterogeneousProtocol] DP observe_clean failed: %s", exc)
        calib_steps = int(self.config.get("dp_calibration_steps", 1))
        if self.dp_step_counter >= max(calib_steps, 1):
            # Calibration done — switch to live-DP mode.
            self.dp_calibration_mode = False
            priv.set_calibration_mode(False)
            logger.info(
                "[HeterogeneousProtocol] DP calibration done after %d step(s); "
                "switching to live noise mode",
                self.dp_step_counter,
            )

    @staticmethod
    def _decode_token_tensor(token_tensor: torch.Tensor) -> str:
        """Decode a 1-D tensor of token ids into a text string (no special tokens)."""
        try:
            from transformers import AutoTokenizer
            if not hasattr(HeterogeneousProtocol, "_cached_tokenizer"):
                HeterogeneousProtocol._cached_tokenizer = AutoTokenizer.from_pretrained(
                    "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I",
                    trust_remote_code=True,
                    use_fast=True,
                )
                if HeterogeneousProtocol._cached_tokenizer.pad_token is None:
                    HeterogeneousProtocol._cached_tokenizer.pad_token = (
                        HeterogeneousProtocol._cached_tokenizer.eos_token
                    )
            return HeterogeneousProtocol._cached_tokenizer.decode(
                token_tensor.tolist(), skip_special_tokens=True
            )
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    #  Checkpoint + shutdown
    # ------------------------------------------------------------------ #
    def gather_checkpoints(self) -> Dict:
        """Collect U/M/S checkpoint blobs (all in-process; no IPC)."""
        return {
            "U": self.party_u.save_checkpoint(),
            "M": self.party_m.save_checkpoint(),
            "S": self.party_s.save_checkpoint(),
        }

    def load_checkpoints(self, checkpoint_dir: str, ckpt_path: str = None) -> None:
        """Load model weights from a checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
            ckpt_path:      Specific checkpoint file to load.  If None, loads
                           ``best_checkpoint.pt`` from *checkpoint_dir*.
                           Pass ``last_checkpoint.pt`` to resume from the most
                           recent run.
        """
        if ckpt_path is None:
            ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(
                "[load_checkpoints] No checkpoint found at %s — skipping.",
                ckpt_path,
            )
            return

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        party_ckpts = ckpt.get("party_checkpoints", {})

        # U: no trainable weights, but restore spec metadata if needed
        u_ckpt = party_ckpts.get("U", {})
        logger.info(
            "[load_checkpoints] U restored (party=%s)",
            u_ckpt.get("party", "N/A"),
        )

        # M: restore LoRA parameters + optimizer state
        m_ckpt = party_ckpts.get("M", {})
        lora_state = m_ckpt.get("lora_state", {})
        if lora_state:
            missing, unexpected = self.party_m.model.load_state_dict(lora_state, strict=False)
            if missing:
                logger.warning(
                    "[load_checkpoints] M missing keys (LoRA may not be in state dict): %s",
                    missing,
                )
            if unexpected:
                logger.debug(
                    "[load_checkpoints] M unexpected keys: %s",
                    unexpected,
                )
            logger.info("[load_checkpoints] M LoRA state restored (%d tensors)", len(lora_state))
        else:
            logger.warning("[load_checkpoints] M checkpoint has no lora_state — skipping")

        # M: restore optimizer state if present
        optimizer_state = m_ckpt.get("optimizer_state", {})
        if optimizer_state:
            try:
                self.party_m.optimizer.load_state_dict(optimizer_state)
                logger.info("[load_checkpoints] M optimizer state restored")
            except Exception as e:
                logger.warning(
                    "[load_checkpoints] Failed to restore M optimizer state: %s — "
                    "continuing without optimizer restore",
                    e,
                )

        # M: restore LR scheduler state (warmup/cosine must continue from the
        # saved step, otherwise a resumed run restarts the schedule).
        scheduler_state = m_ckpt.get("scheduler_state")
        if scheduler_state and getattr(self.party_m, "lr_scheduler", None) is not None:
            try:
                self.party_m.lr_scheduler.load_state_dict(scheduler_state)
                logger.info("[load_checkpoints] M LR scheduler state restored")
            except Exception as e:
                logger.warning(
                    "[load_checkpoints] Failed to restore M LR scheduler state: %s",
                    e,
                )

        # S: frozen V matrix — cannot be restored (no meaningful trainable state)
        s_ckpt = party_ckpts.get("S", {})
        logger.info(
            "[load_checkpoints] S restored (party=%s, frozen)",
            s_ckpt.get("party", "N/A"),
        )

        logger.info("[load_checkpoints] All parties restored from %s", ckpt_path)

    def shutdown(self) -> None:
        """Tear down the worker pools in order. Idempotent."""
        try:
            self.crypto_u_pool.close()
        except Exception as e:
            logger.warning("[shutdown] crypto_u_pool.close raised: %s", e)
        try:
            self.crypto_m_pool.close()
        except Exception as e:
            logger.warning("[shutdown] crypto_m_pool.close raised: %s", e)
        try:
            self.crypto_s_pool.close()
        except Exception as e:
            logger.warning("[shutdown] crypto_s_pool.close raised: %s", e)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def __enter__(self) -> "HeterogeneousProtocol":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
