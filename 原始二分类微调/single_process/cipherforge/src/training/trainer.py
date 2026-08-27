"""
High-level Trainer for SLG-HE-PIR v2.0 (heterogeneous runtime).

The Trainer is protocol-agnostic: it only depends on the protocol exposing
``step_train_chunked``, ``step_train``, ``step_val``, ``gather_checkpoints``,
and ``shutdown``. Both :class:`HeterogeneousProtocol` (active runtime) and
:class:`LegacyIPCStub` (audit / multi-host preview) satisfy this contract.

Key fixes vs. previous trainer
------------------------------
* :attr:`TrainerConfig.val_metric` now has the correct key
  ``val_entity_micro_f1`` (was: ``entity_micro_f1`` which did not exist in
  the per-epoch metrics dict).
* :attr:`TrainerConfig.batch_size` default raised to 48 to match the
  ``Config.batch_size`` default in ``finetune.py``.
* A one-shot :class:`LegacyIPCStub` warning is emitted if the user instantiates
  the trainer with a legacy runtime (the trainer is intended to be paired with
  :class:`HeterogeneousProtocol`).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..data.dataset import OPTIONS_LETTERS, parse_answer_letter

__all__ = ["Trainer", "TrainerConfig"]

logger = logging.getLogger(__name__)


def _safe_letter_split(letter_str: Optional[str]) -> List[str]:
    """Split a letter answer like ``"l)"`` or ``"j), o)"`` into a list of letters.

    Returns an empty list if ``letter_str`` is None/empty.
    """
    if not letter_str:
        return []
    return [tok.strip() for tok in letter_str.split(",") if tok.strip()]


def _letter_to_coarse(letter: str) -> Optional[str]:
    """Map a single BioTriplex option letter to its coarse relation name.

    The dataset builds an answer prompt like ``"a) ASSOCIATED"`` /
    ``"b) CORRELATES_WITH"`` etc.  When the first decoded token is ``"a"``
    we want ``"ASSOCIATED"`` so the trainer can look it up in
    :data:`GENERAL_REL_TO_IDX`.  This helper does the letter→coarse mapping
    via the canonical :data:`GENERAL_RELATIONS` ordering.
    """
    try:
        from src.data.biotriplex_dataset import GENERAL_RELATIONS
    except Exception:
        return None
    letter = (letter or "").strip().lower()
    if not letter:
        return None
    # Allow multiple letters (e.g. "a, c") → take the first one.
    head = letter.split(",")[0].strip()
    if len(head) != 1 or not head.isalpha():
        return None
    idx = ord(head) - ord("a")
    if 0 <= idx < len(GENERAL_RELATIONS):
        return GENERAL_RELATIONS[idx]
    return None


def make_string_safe_collate():
    """Build a DataLoader collate function that preserves string fields.

    The default PyTorch collate stacks tensors and fails on string fields
    like ``doc_key``, ``output_text``, etc. This custom collate returns
    lists for non-tensor entries and stacks only the tensor entries.
    """
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not batch:
            return out
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                try:
                    out[k] = torch.stack(vals, dim=0)
                except Exception:
                    out[k] = vals
            elif isinstance(vals[0], (list, tuple)):
                # keep as list of lists
                out[k] = list(vals)
            else:
                out[k] = list(vals)
        return out
    return _collate    



@dataclass
class TrainerConfig:
    """Configuration for the Trainer.

    Default values match docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md §6.1.
    """
    max_epochs: int = 10
    patience: int = 999               # No early stopping by default
    train_ratio: float = 0.9
    seed: int = 42
    # Fixed key — _run_val_epoch emits `val_entity_micro_f1`, so this must
    # match it. Previous default was `entity_micro_f1` which silently broke
    # best-metric tracking.
    val_metric: str = "val_entity_micro_f1"
    save_freq: int = 1
    log_freq: int = 10
    checkpoint_dir: str = "/root/autodl-tmp/SLG-HE-PIR/checkpoints"
    log_dir: str = "/root/autodl-tmp/SLG-HE-PIR/logs"
    dump_attacks: bool = False
    batch_size: int = 4               # Small batch to avoid OOM with 32-layer M
    max_seq_length: int = 128        # Max sequence length (matches docs §6.1)
    # Parallel pipeline knobs.
    # USE_CHUNKED_PIPELINE=True routes step_train_chunked (S→U→M streamed
    # by chunk); False keeps the legacy one-shot step_train.
    USE_CHUNKED_PIPELINE: bool = True
    CHUNK_TOKENS: int = 3072
    # Run full evaluation on test_ds after training finishes.
    do_test_eval: bool = False
    # ------------------------------------------------------------------ #
    #  BioTriplex task-specific knobs
    # ------------------------------------------------------------------ #
    # "classification" (GenRel QA letter answer) or "generation" (NER JSON).
    # When set to "generation", ``_run_val_epoch``/``_run_test_epoch`` also
    # compute BioTriplex per-class span P/R/F1 metrics (GENE / DISEASE /
    # RELATION) in addition to the existing letter-level fields.
    task_type: str = "classification"
    # Optional explicit path to the BioTriplex gold JSONL file. When the
    # task is "generation" the trainer uses this file (in addition to the
    # batch-level gold text) to score entities by exact-match span.
    ner_gold_path: Optional[str] = None
    # ------------------------------------------------------------------ #
    #  dχ-privacy knobs (see DP机制-迁移参考.md §3.7 / §4.1)
    # ------------------------------------------------------------------ #
    dp_enable: bool = False
    dp_dump_audit: bool = False
    dp_alpha: float = 0.15
    dp_answer_beta: float = 0.5
    dp_calibration_steps: int = 1
    dp_eta0: Optional[float] = None
    dp_clip_value: Optional[float] = None
    dp_calibration_mode: bool = False
    dp_num_classes: int = 7


class Trainer:
    """High-level trainer coordinating U/M/S training loop."""

    def __init__(
        self,
        config: TrainerConfig,
        ipc_protocol: Any,          # HeterogeneousProtocol (active) or LegacyIPCStub
        train_ds: Any,
        val_ds: Any,
        test_ds: Any,
        tokenizer: Any,
    ):
        self.config = config
        self.ipc = ipc_protocol
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.tokenizer = tokenizer

        self.best_metric = float("-inf") if "f1" in config.val_metric else float("inf")
        self.best_epoch = -1
        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0
        self.start_time = time.time()
        self._completed_epochs: int = 0  # epochs already done when resuming

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        self.metrics_log: List[Dict] = []

        self.step_callback: Optional[Any] = None

        # If the user passed a legacy runtime, emit a one-shot warning so
        # audit-mode use is intentional.
        if type(self.ipc).__name__ in ("LegacyIPCStub", "IPCProtocol"):
            logger.warning(
                "Trainer is wired to a LegacyIPCStub. The active runtime is "
                "HeterogeneousProtocol — only use this for audit/multi-host "
                "preview purposes."
            )

    # ------------------------------------------------------------------------- #
    #  Checkpoint / resume
    # ------------------------------------------------------------------------- #
    def resume_from(self, ckpt_path: str) -> int:
        """Restore LoRA weights, optimizer state, and epoch from a checkpoint file.

        Also saves a ``last_checkpoint.pt`` before loading so the run is always
        recoverable even if the loaded file is subsequently overwritten.

        Returns the 0-based epoch index to resume from (i.e. the epoch that was
        *completed* before the checkpoint was taken).  If no checkpoint exists,
        returns 0 and leaves state unchanged.
        """
        if not os.path.exists(ckpt_path):
            logger.warning("[resume_from] No checkpoint found at %s — starting from scratch", ckpt_path)
            return 0

        # Atomic-ish safety net: copy the file so we always have something to recover.
        import shutil
        safety_path = os.path.join(self.config.checkpoint_dir, "last_checkpoint.pt")
        try:
            shutil.copy2(ckpt_path, safety_path)
            logger.info("[resume_from] Safety copy saved → %s", safety_path)
        except Exception as e:
            logger.warning("[resume_from] Could not write safety copy: %s", e)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_epoch = ckpt.get("epoch", -1)
        saved_global_step = ckpt.get("global_step", 0)
        saved_completed_epochs = ckpt.get("completed_epochs", saved_epoch)

        # Restore LoRA + optimizer via the IPC protocol.
        self.ipc.load_checkpoints(self.config.checkpoint_dir, ckpt_path=ckpt_path)

        self.global_step = saved_global_step
        self._completed_epochs = saved_completed_epochs + 1

        logger.info(
            "[resume_from] Loaded epoch=%d (0-based), global_step=%d, "
            "will resume from epoch %d. Safety copy → %s",
            saved_epoch, saved_global_step, self._completed_epochs, safety_path,
        )
        return self._completed_epochs

    def _save_last_checkpoint(self) -> None:
        """Save a lightweight 'last' checkpoint at the end of each epoch."""
        ckpt_path = os.path.join(self.config.checkpoint_dir, "last_checkpoint.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "completed_epochs": self.epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "party_checkpoints": self.ipc.gather_checkpoints(),
                "config": self.config.__dict__,
            },
            ckpt_path,
        )
        logger.info("Last checkpoint saved → %s (epoch=%d)", ckpt_path, self.epoch)

    # ------------------------------------------------------------------------- #
    #  dχ-privacy helpers
    # ------------------------------------------------------------------------- #
    def _ensure_dp_audit_path(self) -> str:
        """Pick a stable path under ``log_dir`` for the per-step audit JSONL."""
        if getattr(self, "_dp_audit_path", None) is not None:
            return self._dp_audit_path
        os.makedirs(self.config.log_dir, exist_ok=True)
        self._dp_audit_path = os.path.join(self.config.log_dir, "dp_audit.jsonl")
        return self._dp_audit_path

    def _log_dp_audit(self, result: Any) -> None:
        """Append the latest ``StepResult.dp_audit`` to ``dp_audit.jsonl``."""
        import json
        audit = getattr(result, "dp_audit", None)
        if not audit:
            return
        path = self._ensure_dp_audit_path()
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(audit, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[_log_dp_audit] failed to write %s: %s", path, exc)

    def _maybe_fit_cti_for_epoch(self, train_loader: Any) -> None:
        """Feed the first training batch of the epoch to the LabelBasedCTI.

        The CTI uses the per-sample coarse label to update its (C, V)
        conditional token distribution; that is what makes the per-token
        utility importance (UI) shift as the dataset distribution drifts.
        For ``task_type == "generation"`` we skip the call because there is
        no canonical 7-class label to feed back.
        """
        priv = getattr(
            getattr(self.ipc, "party_u", None), "h15_privatizer", None
        )
        if priv is None or not hasattr(priv, "fit_cti"):
            return
        if str(getattr(self.config, "task_type", "classification")) != "classification":
            return
        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            return
        if not isinstance(sample_batch, dict):
            return
        cls_idx = self._derive_class_idx_from_batch(sample_batch)
        if cls_idx is None:
            return
        try:
            import torch as _torch
            cls_tensor = _torch.tensor(cls_idx, dtype=_torch.long)
            input_ids = sample_batch.get("input_ids")
            attn = sample_batch.get("attention_mask")
            if input_ids is None:
                return
            if attn is None:
                attn = _torch.ones_like(input_ids, dtype=_torch.long)
            priv.fit_cti(cls_tensor, input_ids, attn)
        except Exception as exc:
            logger.warning("[Trainer] fit_cti failed: %s", exc)

    @staticmethod
    def _derive_class_idx_from_batch(batch: Dict) -> Optional[List[int]]:
        """Extract coarse class indices from a batch.

        Prefers ``label_idx`` (set by :class:`BioTriplexQADatasetClassification`).
        Falls back to decoding the first non-pad label token and looking it up
        in :data:`GENERAL_REL_TO_IDX` (which is the 7-class GenRel mapping).
        Returns ``None`` when the batch carries no usable class signal.
        """
        if "label_idx" in batch and batch["label_idx"] is not None:
            li = batch["label_idx"]
            if isinstance(li, list):
                return [int(x) for x in li]
            try:
                return [int(x) for x in li.tolist()]
            except Exception:
                pass

        # Fallback: try to decode the first answer letter per sample.
        try:
            from src.data.biotriplex_dataset import GENERAL_REL_TO_IDX
            import torch as _torch

            output_ids = batch.get("output_ids")
            if output_ids is None:
                return None
            if not isinstance(output_ids, _torch.Tensor):
                return None
            out: List[int] = []
            for row in output_ids:
                non_pad = row[row != 0]
                if non_pad.numel() == 0:
                    out.append(-1)
                    continue
                # The first non-pad token in BioTriplex is the letter (a, b, c, ...).
                letter_idx = int(non_pad[0].item())
                # Try numeric first.
                if 0 <= letter_idx < 32:
                    out.append(letter_idx)
                    continue
                # Decode token → letter char.
                try:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(
                        "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I",
                        trust_remote_code=True,
                        use_fast=True,
                    )
                    txt = tok.decode([letter_idx], skip_special_tokens=True).strip()
                    if not txt:
                        out.append(-1)
                        continue
                    # Coarse lookup: the dataset uses letters a..g with bias >= 7.
                    # Match on coarse category rather than letter spelling to
                    # avoid having to map a..g to the relation class.
                    rel = _letter_to_coarse(txt)
                    if rel is None:
                        out.append(-1)
                    else:
                        out.append(GENERAL_REL_TO_IDX.get(rel, -1))
                except Exception:
                    out.append(-1)
            return out
        except Exception:
            return None

    # ------------------------------------------------------------------------- #
    #  Training loop
    # ------------------------------------------------------------------------- #
    def train(self) -> Dict[str, Any]:
        logger.info(
            "Starting training: epochs=%d, patience=%d, train=%d, val=%d",
            self.config.max_epochs,
            self.config.patience,
            len(self.train_ds),
            len(self.val_ds),
        )

        for epoch in range(self.config.max_epochs):
            # Skip epochs already completed by a previous resume run.
            if epoch < self._completed_epochs:
                logger.info("Epoch %d already completed — skipping", epoch)
                continue

            self.epoch = epoch
            epoch_metrics = self._run_epoch(epoch)
            self._log_epoch(epoch, epoch_metrics)
            self._save_last_checkpoint()  # always save after every epoch

            if (epoch + 1) % self.config.save_freq == 0:
                self._save_checkpoint(epoch, epoch_metrics)

            if self._is_best(epoch_metrics):
                self._save_best_checkpoint(epoch, epoch_metrics)
                self.patience_counter = 0
                logger.info(
                    "New best %s=%.4f at epoch %d",
                    self.config.val_metric,
                    epoch_metrics.get(self.config.val_metric, 0),
                    epoch,
                )
            else:
                self.patience_counter += 1
                logger.info("No improvement for %d epochs", self.patience_counter)

            if self.patience_counter >= self.config.patience:
                logger.info(
                    "Early stopping at epoch %d — restoring best model (epoch %d, "
                    "metric=%.4f).",
                    epoch,
                    self.best_epoch,
                    self.best_metric,
                )
                self._load_checkpoint()
                break

        elapsed = time.time() - self.start_time

        # Run test evaluation after training (or after early stopping recovery).
        if self.config.do_test_eval:
            test_metrics = self._run_test_epoch()
            self.metrics_log.append({"phase": "test", **test_metrics})

        return self._finalize()

    def _run_epoch(self, epoch: int) -> Dict[str, Any]:
        total_steps = 0
        epoch_loss = 0.0
        step_times = []
        gpu_mem_samples = []

        from torch.utils.data import DataLoader
        collate = make_string_safe_collate()
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=collate,
        )

        # ---- dχ-privacy: feed the first batch to the CTI at every epoch ----
        # This keeps the per-class token distribution drifting with epoch
        # progression without paying the per-step overhead (reference doc §5).
        if bool(getattr(self.config, "dp_enable", False)):
            self._maybe_fit_cti_for_epoch(train_loader)

        for batch in train_loader:
            if bool(self.config.USE_CHUNKED_PIPELINE):
                chunk_tokens = int(self.config.CHUNK_TOKENS)
                result = self.ipc.step_train_chunked(
                    batch, self.global_step,
                    chunk_tokens=chunk_tokens,
                )
            else:
                result = self.ipc.step_train(batch, self.global_step)
            epoch_loss += result.loss
            step_times.append(result.step_time_ms)
            gpu_mem_samples.append(result.gpu_mem_mb)
            self.global_step += 1
            total_steps += 1

            if self.step_callback is not None:
                try:
                    self.step_callback(
                        epoch=epoch,
                        step_idx=total_steps - 1,
                        batch=batch,
                        result=result,
                    )
                except Exception as cb_e:
                    logger.warning("step_callback raised: %s", cb_e)

            # ---- dχ-privacy: write per-step audit to log_dir/dp_audit.jsonl ----
            if bool(getattr(self.config, "dp_dump_audit", False)):
                self._log_dp_audit(result)

            if total_steps % self.config.log_freq == 0:
                loss_ce = getattr(result, "loss_ce", None)
                logger.info(
                    "Step %d: loss=%.4f loss_ce=%s time=%.1fms mem=%.0fMB",
                    self.global_step, result.loss,
                    "%.4f" % float(loss_ce) if loss_ce is not None else "n/a",
                    result.step_time_ms, result.gpu_mem_mb,
                )

        val_metrics = self._run_val_epoch(epoch)

        avg_loss = epoch_loss / max(total_steps, 1)
        avg_step_time = sum(step_times) / max(len(step_times), 1)
        avg_gpu_mem = sum(gpu_mem_samples) / max(len(gpu_mem_samples), 1)

        return {
            "train_loss": avg_loss,
            "train_steps": total_steps,
            "avg_step_time_ms": avg_step_time,
            "avg_gpu_mem_mb": avg_gpu_mem,
            **val_metrics,
        }

    def _run_val_epoch(self, epoch: int) -> Dict[str, float]:
        """Run Step B: validation set via PIR protocol.

        Computes BioTriplex-QA letter-level micro-F1 (matching Table 5
        in the BioTriplex paper) plus CE loss over validation logits,
        micro-averaged Precision/Recall, Micro Accuracy, Macro F1,
        and Weighted F1.

        When ``config.task_type == "generation"`` the validation loop also
        computes BioTriplex per-class NER span P/R/F1 metrics (GENE /
        DISEASE / RELATION). For ``task_type == "classification"`` the
        additional metrics block is a no-op.
        """
        # Emergency checkpoint before val (in case val crashes and we lose training progress)
        try:
            self._save_last_checkpoint()
            logger.debug("Emergency checkpoint saved before val epoch %d", epoch)
        except Exception as ckpt_e:
            logger.warning("Emergency checkpoint failed: %s", ckpt_e)

        from torch.utils.data import DataLoader
        collate = make_string_safe_collate()
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

        # ClinVar is a single-answer-token binary task; use its own metric
        # branch (answer-position CE + token accuracy + Yes/No P/R) instead of
        # the BioTriplex letter-level aggregators.
        if getattr(self.config, "task_type", "classification") == "clinvar":
            return self._run_val_clinvar(val_loader, epoch)
        return self._run_val_biotriplex(val_loader, epoch)

    def _run_val_clinvar(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate the ClinVar answer token: CE + accuracy + Yes/No P/R.

        ``step_val`` returns the full-vocab logits at every position plus
        ``labels_tensor`` (gold token ids, -100 on prompt/pad). We compute the
        standard masked CE over the gold answer token(s), token-level accuracy,
        and a binary Yes/No P/R using the decoded predictions.
        """
        import torch
        import torch.nn.functional as F

        total_ce = 0.0
        n_ce = 0
        n_correct = 0
        n_samples = 0
        tp = fp = fn = 0  # positive = "Yes"

        for batch in val_loader:
            result = self.ipc.step_val(batch, self.global_step)
            logits_v = result.get("logits")
            labels_tensor = result.get("labels_tensor")
            preds = result.get("predictions") or []
            labels = result.get("labels") or []

            if (
                logits_v is not None
                and labels_tensor is not None
                and isinstance(logits_v, torch.Tensor)
                and isinstance(labels_tensor, torch.Tensor)
            ):
                # Standard causal-LM alignment: logits at t predict token t+1.
                logits_shift = logits_v[:, :-1, :]
                labels_shift = labels_tensor[:, 1:]
                try:
                    ce = F.cross_entropy(
                        logits_shift.reshape(-1, logits_v.size(-1)),
                        labels_shift.reshape(-1),
                        ignore_index=-100,
                    )
                    total_ce += ce.item()
                    n_ce += 1
                except Exception as ce_e:
                    logger.debug("ClinVar val CE skip: %s", ce_e)
                # Token-level accuracy at the gold answer position(s): argmax
                # of logits_t must equal the gold token at t+1.
                pred_tok = logits_v.argmax(dim=-1)[:, :-1]
                valid = labels_shift != -100
                if bool(valid.any()):
                    n_correct += int((pred_tok[valid] == labels_shift[valid]).sum().item())
                    n_samples += int(valid.sum().item())

            for p, g in zip(preds, labels):
                if not g:
                    continue
                gold_pos = str(g).strip().lower().startswith("y")
                pred_pos = str(p).strip().lower().startswith("y")
                if gold_pos:
                    if pred_pos:
                        tp += 1
                    else:
                        fn += 1
                elif pred_pos:
                    fp += 1

        acc = n_correct / n_samples if n_samples else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        avg_ce = total_ce / max(n_ce, 1)
        return {
            # Same keys the trainer/logger consume, so nothing else changes.
            "val_entity_micro_f1": acc,
            "val_letter_micro_f1": acc,
            "val_micro_precision": prec,
            "val_micro_recall": rec,
            "val_micro_accuracy": acc,
            "val_macro_f1": f1,
            "val_weighted_f1": f1,
            "val_ce_loss": avg_ce,
            "val_samples": n_samples,
            "val_clinvar_yes_precision": prec,
            "val_clinvar_yes_recall": rec,
            "val_clinvar_yes_f1": f1,
            "val_clinvar_token_accuracy": acc,
            "val_clinvar_ce_batches": n_ce,
        }

    def _run_val_biotriplex(self, val_loader, epoch: int) -> Dict[str, float]:
        """Original BioTriplex letter-level validation loop."""
        # 累积 TP/FP/FN（论文式 micro-averaging）
        tp_total = fp_total = fn_total = 0
        per_class_tp: Dict[str, int] = {}
        per_class_fp: Dict[str, int] = {}
        per_class_fn: Dict[str, int] = {}
        all_predictions_letters: List[str] = []
        all_labels_letters: List[str] = []
        all_predictions: List[str] = []
        all_labels: List[str] = []
        all_doc_keys: List[str] = []
        all_pred_logits: List[List[float]] = []
        val_ce_loss = 0.0
        val_ce_count = 0
        val_ce_used_fallback = False  # tracks old token-level CE fallback

        for batch in val_loader:
            result = self.ipc.step_val(batch, self.global_step)

            # --- CE Loss (7-class projection OR token-level fallback) ---
            #
            # When ``HeterogeneousProtocol`` is in ``classification`` mode
            # it returns ``pred_logits`` (shape ``[B, 7]``) which is the
            # correct target for CE loss on the 7-class GenRel task. The
            # only label the dataset emits is the gold letter index in
            # ``0..6`` (gold general relation). For NER / generation mode
            # ``pred_logits`` is ``None`` and we still fall back to the
            # token-level CE loss against ``labels_tensor`` (``output_ids``).
            import torch
            import torch.nn.functional as F

            pred_logits_b = result.get("pred_logits")
            logits_v = result.get("logits")
            labels_tensor = result.get("labels_tensor")

            used_classification_ce = False
            if (
                pred_logits_b is not None
                and isinstance(pred_logits_b, list)
                and len(pred_logits_b) > 0
                and isinstance(pred_logits_b[0], list)
            ):
                # 7-class CE against gold letter index. Gold letter index is
                # recoverable from ``labels_letters`` already collected by the
                # protocol layer via ``parse_answer_letter`` regex.
                preds_letters = result.get("predictions_letters") or []
                labs_letters = result.get("labels_letters") or []
                if len(preds_letters) == len(labs_letters):
                    label_idx_list = []
                    skip = False
                    for letter_str in labs_letters:
                        if not letter_str:
                            skip = True
                            break
                        head = letter_str.strip().lower()[:1]
                        if not ("a" <= head <= "g"):
                            skip = True
                            break
                        label_idx_list.append(ord(head) - ord("a"))
                    if not skip and len(label_idx_list) == len(pred_logits_b):
                        try:
                            ce_in = torch.tensor(pred_logits_b, dtype=torch.float32)
                            ce_tgt = torch.tensor(label_idx_list, dtype=torch.long)
                            ce = F.cross_entropy(ce_in, ce_tgt)
                            val_ce_loss += ce.item()
                            val_ce_count += 1
                            used_classification_ce = True
                        except Exception as cb_e:
                            logger.debug("7-class CE skip: %s", cb_e)

            if not used_classification_ce:
                # Fallback path (token-level CE on vocab logits)
                if logits_v is not None and labels_tensor is not None:
                    if isinstance(logits_v, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
                        try:
                            ce = F.cross_entropy(
                                logits_v.view(-1, logits_v.size(-1)),
                                labels_tensor.view(-1),
                                ignore_index=-100,
                            )
                            val_ce_loss += ce.item()
                            val_ce_count += 1
                            val_ce_used_fallback = True
                        except Exception as cb_e:
                            logger.debug("CE loss skip: %s", cb_e)

            # --- Letter-level TP/FP/FN ---
            pred_letters_list = result.get("predictions_letters", []) or []
            gold_letters_list = result.get("labels_letters", []) or []
            if not pred_letters_list:
                pred_letters_list = result.get("predictions", [])
            if not gold_letters_list:
                gold_letters_list = result.get("labels", [])

            # Persist decoded texts for debug logging.
            all_predictions.extend(result.get("predictions", []) or [])
            all_labels.extend(result.get("labels", []) or [])
            all_doc_keys.extend(result.get("doc_keys", []) or [])

            # Persist 7-class logits (classification only) so compute_classification_metrics
            # can compute real ROC AUC instead of the 1-hot fallback.
            pred_logits_b_out = result.get("pred_logits")
            if isinstance(pred_logits_b_out, list):
                all_pred_logits.extend(pred_logits_b_out)

            for pred, gold in zip(pred_letters_list, gold_letters_list):
                all_predictions_letters.append(pred or "")
                all_labels_letters.append(gold or "")

                pred_set = set(_safe_letter_split(pred))
                gold_set = set(_safe_letter_split(gold))

                tp_total += len(pred_set & gold_set)
                fp_total += len(pred_set - gold_set)
                fn_total += len(gold_set - pred_set)

                for rel in pred_set:
                    per_class_fp[rel] = per_class_fp.get(rel, 0) + 1
                    if rel in gold_set:
                        per_class_tp[rel] = per_class_tp.get(rel, 0) + 1
                for rel in gold_set:
                    per_class_fn[rel] = per_class_fn.get(rel, 0) + 1

        # === BioTriplex Table 5 / Table 7 风格的 micro-averaged 指标 ===
        micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        # === Micro Accuracy (eval_ft_llama_qa.py line 239) ===
        correct = sum(
            1 for p, g in zip(all_predictions_letters, all_labels_letters)
            if set(_safe_letter_split(p)) == set(_safe_letter_split(g))
            and bool(p)
        )
        micro_acc = correct / len(all_predictions_letters) if all_predictions_letters else 0.0

        # === Macro F1 ===
        macro_f1_list = []
        for rel in OPTIONS_LETTERS:
            tp_c = per_class_tp.get(rel, 0)
            fp_c = per_class_fp.get(rel, 0)
            fn_c = per_class_fn.get(rel, 0)
            if tp_c + fp_c == 0 and fn_c == 0:
                continue
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            macro_f1_list.append(f1_c)
        macro_f1 = sum(macro_f1_list) / len(macro_f1_list) if macro_f1_list else 0.0

        # === Weighted F1 (eval_ft_llama_qa.py) ===
        weighted_num = 0.0
        weighted_denom = 0
        for rel in OPTIONS_LETTERS:
            tp_c = per_class_tp.get(rel, 0)
            fp_c = per_class_fp.get(rel, 0)
            fn_c = per_class_fn.get(rel, 0)
            support = tp_c + fn_c
            if support == 0 or (tp_c + fp_c) == 0:
                continue
            p_c = tp_c / (tp_c + fp_c)
            r_c = tp_c / (tp_c + fn_c)
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            weighted_num += f1_c * support
            weighted_denom += support
        weighted_f1 = weighted_num / weighted_denom if weighted_denom > 0 else 0.0

        avg_ce_loss = val_ce_loss / max(val_ce_count, 1)

        # === BioTriplex task-specific metrics ===
        task_metrics: Dict[str, float] = {}
        task_type = getattr(self.config, "task_type", "classification")
        if task_type == "generation":
            try:
                from .biotriplex_metrics import compute_ner_metrics, load_ner_gold_entities

                gold_path = getattr(self.config, "ner_gold_path", None)
                gold_map: Dict[str, Dict[str, set]] = {}
                if gold_path and os.path.exists(gold_path):
                    gold_map = load_ner_gold_entities(gold_path)

                # Build per-sample gold entities aligned with predictions by doc_key
                golds_for_pred: List[Dict[str, set]] = []
                pred_aligned: List[str] = []
                for i, pred in enumerate(all_predictions):
                    dk = all_doc_keys[i] if i < len(all_doc_keys) else f"sample_{i}"
                    # Prefer gold from gold-map (entities dict); fallback: parse
                    # the JSON label text in all_labels[i] (it carries entities).
                    if dk in gold_map:
                        golds_for_pred.append(gold_map[dk])
                    else:
                        from .biotriplex_metrics import _parse_entities_json
                        golds_for_pred.append(_parse_entities_json(all_labels[i] if i < len(all_labels) else ""))
                    pred_aligned.append(pred)
                ner = compute_ner_metrics(pred_aligned, golds_for_pred, all_doc_keys)
                m = ner.get("metrics", {})
                task_metrics = {
                    "val_ner_macro_f1": m.get("macro_f1", 0.0),
                    "val_ner_weighted_f1": m.get("weighted_f1", 0.0),
                    "val_ner_micro_f1": m.get("overall_micro_f1", 0.0),
                    "val_ner_macro_precision": m.get("macro_precision", 0.0),
                    "val_ner_macro_recall": m.get("macro_recall", 0.0),
                    "val_ner_micro_precision": m.get("overall_micro_precision", 0.0),
                    "val_ner_micro_recall": m.get("overall_micro_recall", 0.0),
                    "val_ner_n_parse_failures": ner.get("n_parse_failures", 0),
                    # Per-class F1
                    "val_ner_f1_GENE": ner.get("per_class_metrics", {}).get("GENE", {}).get("f1", 0.0),
                    "val_ner_f1_DISEASE": ner.get("per_class_metrics", {}).get("DISEASE", {}).get("f1", 0.0),
                    "val_ner_f1_RELATION": ner.get("per_class_metrics", {}).get("RELATION", {}).get("f1", 0.0),
                }
            except Exception as e:
                logger.warning("NER metric computation failed: %s", e)
        elif task_type == "classification":
            # Compute BioTriplex GenRel multi-label metrics from raw predictions + labels
            try:
                from .biotriplex_metrics import compute_classification_metrics
                bt_metrics = compute_classification_metrics(
                    all_predictions,  # raw model output strings
                    all_labels,        # gold answer strings
                    pred_logits=all_pred_logits or None,
                )
                m = bt_metrics.get("metrics", {})
                task_metrics = {
                    "val_bt_micro_f1": m.get("micro_f1", 0.0),
                    "val_bt_macro_f1": m.get("macro_f1", 0.0),
                    "val_bt_weighted_f1": m.get("weighted_f1", 0.0),
                    "val_bt_multilabel_f1_samples": m.get("multilabel_f1_samples", 0.0),
                    "val_bt_multilabel_f1_macro": m.get("multilabel_f1_macro", 0.0),
                    "val_bt_multilabel_f1_micro": m.get("multilabel_f1_micro", 0.0),
                    "val_bt_macro_roc_auc": bt_metrics.get("macro_roc_auc_ovr") or 0.0,
                    "val_bt_micro_roc_auc": bt_metrics.get("micro_roc_auc_ovr") or 0.0,
                    "val_bt_n_parse_failures": bt_metrics.get("n_parse_failures", 0),
                }
            except Exception as e:
                logger.warning("BioTriplex classification metric computation failed: %s", e)

        return {
            # === 主指标（与论文 Table 5/7 对齐） ===
            "val_entity_micro_f1": micro_f1,    # 兼容 TrainerConfig.val_metric
            "val_letter_micro_f1": micro_f1,    # 与论文 micro-F1 同义
            "val_micro_precision": micro_p,
            "val_micro_recall": micro_r,
            "val_micro_accuracy": micro_acc,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            # === CE loss ===
            "val_ce_loss": avg_ce_loss,
            # === 调试 ===
            "val_samples": len(all_predictions_letters),
            # === BioTriplex task-specific metrics (NER) ===
            **task_metrics,
        }

    def _is_best(self, epoch_metrics: Dict) -> bool:
        metric = self.config.val_metric
        value = epoch_metrics.get(metric, 0.0)
        if "f1" in metric or "recall" in metric or "precision" in metric:
            return value > self.best_metric
        else:
            return value < self.best_metric

    def _log_epoch(self, epoch: int, metrics: Dict) -> None:
        record = {
            "epoch": epoch,
            "timestamp": time.time(),
            "elapsed_s": time.time() - self.start_time,
            **metrics,
        }
        self.metrics_log.append(record)
        logger.info(
            "Epoch %d: train_loss_proxy=%.4f | val_ce_loss=%.4f | "
            "val_micro_F1=%.4f (P=%.4f R=%.4f Acc=%.4f) | "
            "val_macro_F1=%.4f | val_weighted_F1=%.4f | n_samples=%d",
            epoch,
            metrics.get("train_loss", 0),
            metrics.get("val_ce_loss", 0),
            metrics.get("val_entity_micro_f1", 0),
            metrics.get("val_micro_precision", 0),
            metrics.get("val_micro_recall", 0),
            metrics.get("val_micro_accuracy", 0),
            metrics.get("val_macro_f1", 0),
            metrics.get("val_weighted_f1", 0),
            metrics.get("val_samples", 0),
        )
        # Log task-specific BioTriplex metrics
        task_type = getattr(self.config, "task_type", "classification")
        if task_type == "classification":
            logger.info(
                "  [BioTriplex Classification] val_bt_micro_f1=%.4f | val_bt_macro_f1=%.4f | "
                "val_bt_weighted_f1=%.4f | val_bt_multilabel_f1=%.4f | "
                "val_bt_macro_roc_auc=%.4f",
                metrics.get("val_bt_micro_f1", 0),
                metrics.get("val_bt_macro_f1", 0),
                metrics.get("val_bt_weighted_f1", 0),
                metrics.get("val_bt_multilabel_f1_samples", 0),
                metrics.get("val_bt_macro_roc_auc", 0),
            )
        elif task_type == "generation":
            logger.info(
                "  [BioTriplex NER] val_ner_macro_f1=%.4f | val_ner_weighted_f1=%.4f | "
                "val_ner_micro_f1=%.4f | val_ner_macro_roc_auc=%.4f | "
                "GENE_f1=%.4f DISEASE_f1=%.4f RELATION_f1=%.4f",
                metrics.get("val_ner_macro_f1", 0),
                metrics.get("val_ner_weighted_f1", 0),
                metrics.get("val_ner_micro_f1", 0),
                metrics.get("val_ner_macro_roc_auc", 0),
                metrics.get("val_ner_f1_GENE", 0),
                metrics.get("val_ner_f1_DISEASE", 0),
                metrics.get("val_ner_f1_RELATION", 0),
            )
        # Append each epoch to a separate JSONL file so runs never overwrite each other.
        jsonl_path = os.path.join(self.config.log_dir, "epoch_metrics.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------------- #
    #  Checkpointing
    # ------------------------------------------------------------------------- #
    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        ckpts = self.ipc.gather_checkpoints()
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch:03d}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "metrics": metrics,
                "party_checkpoints": ckpts,
                "config": self.config.__dict__,
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved → %s", ckpt_path)

    def _save_best_checkpoint(self, epoch: int, metrics: Dict) -> None:
        self.best_metric = metrics.get(self.config.val_metric, 0)
        self.best_epoch = epoch
        ckpts = self.ipc.gather_checkpoints()
        ckpt_path = os.path.join(
            self.config.checkpoint_dir,
            "best_checkpoint.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "metrics": metrics,
                "best_metric": self.best_metric,
                "party_checkpoints": ckpts,
            },
            ckpt_path,
        )
        logger.info("Best checkpoint saved → %s (metric=%.4f)", ckpt_path, self.best_metric)

    def _load_checkpoint(self) -> None:
        """Restore model weights from the best checkpoint.

        Loads ``best_checkpoint.pt`` from ``self.config.checkpoint_dir`` and
        restores party weights via ``self.ipc.load_checkpoints``.
        """
        ckpt_path = os.path.join(self.config.checkpoint_dir, "best_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(
                "[_load_checkpoint] No best checkpoint found at %s — "
                "model state unchanged.",
                ckpt_path,
            )
            return

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        logger.info(
            "[_load_checkpoint] Restoring best model from epoch %d "
            "(metric=%.4f) at %s",
            ckpt.get("epoch", -1),
            ckpt.get("best_metric", self.best_metric),
            ckpt_path,
        )
        self.ipc.load_checkpoints(self.config.checkpoint_dir)

    # ------------------------------------------------------------------------- #
    #  Finalization
    # ------------------------------------------------------------------------- #
    def _run_test_epoch(self) -> Dict[str, float]:
        """Run final evaluation on the held-out test set.

        Uses ``step_test`` (standard, non-private forward) on each test batch
        and computes letter-level BioTriplex metrics (precision, recall, F1,
        accuracy, macro-F1, weighted-F1).
        """
        from torch.utils.data import DataLoader
        import torch
        import torch.nn.functional as F
        logger.info(
            "Running test evaluation on %d samples (batch_size=%d) ...",
            len(self.test_ds),
            self.config.batch_size,
        )
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_string_safe_collate(),
        )

        # Letter-level TP/FP/FN tracking (BioTriplex Table 5 style)
        tp_total = fp_total = fn_total = 0
        per_class_tp: Dict[str, int] = {}
        per_class_fp: Dict[str, int] = {}
        per_class_fn: Dict[str, int] = {}
        all_predictions_letters: List[str] = []
        all_labels_letters: List[str] = []
        all_predictions: List[str] = []
        all_labels: List[str] = []
        all_doc_keys: List[str] = []
        test_ce_loss = 0.0
        test_ce_count = 0

        for batch in test_loader:
            result = self.ipc.step_test(batch, self.global_step)

            logits = result.get("logits")
            labels_tensor = result.get("labels_tensor")
            if logits is not None and labels_tensor is not None:
                if isinstance(logits, torch.Tensor) and isinstance(labels_tensor, torch.Tensor):
                    try:
                        ce = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels_tensor.view(-1),
                            ignore_index=-100,
                        )
                        test_ce_loss += ce.item()
                        test_ce_count += 1
                    except Exception:
                        pass

            pred_letters = result.get("predictions_letters", []) or []
            gold_letters = result.get("labels_letters", []) or []
            if not pred_letters:
                pred_letters = result.get("predictions", [])
            if not gold_letters:
                gold_letters = result.get("labels", [])

            all_predictions.extend(result.get("predictions", []) or [])
            all_labels.extend(result.get("labels", []) or [])
            all_doc_keys.extend(result.get("doc_keys", []) or [])

            for pred, gold in zip(pred_letters, gold_letters):
                all_predictions_letters.append(pred or "")
                all_labels_letters.append(gold or "")
                pred_set = set(_safe_letter_split(pred))
                gold_set = set(_safe_letter_split(gold))
                tp_total += len(pred_set & gold_set)
                fp_total += len(pred_set - gold_set)
                fn_total += len(gold_set - pred_set)
                for rel in pred_set:
                    per_class_fp[rel] = per_class_fp.get(rel, 0) + 1
                    if rel in gold_set:
                        per_class_tp[rel] = per_class_tp.get(rel, 0) + 1
                for rel in gold_set:
                    per_class_fn[rel] = per_class_fn.get(rel, 0) + 1

        micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        correct = sum(
            1 for p, g in zip(all_predictions_letters, all_labels_letters)
            if set(_safe_letter_split(p)) == set(_safe_letter_split(g))
            and bool(p)
        )
        micro_acc = correct / len(all_predictions_letters) if all_predictions_letters else 0.0

        macro_f1_list = []
        for rel in OPTIONS_LETTERS:
            tp_c = per_class_tp.get(rel, 0)
            fp_c = per_class_fp.get(rel, 0)
            fn_c = per_class_fn.get(rel, 0)
            if tp_c + fp_c == 0 and fn_c == 0:
                continue
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            macro_f1_list.append(f1_c)
        macro_f1 = sum(macro_f1_list) / len(macro_f1_list) if macro_f1_list else 0.0

        weighted_num = 0.0
        weighted_denom = 0
        for rel in OPTIONS_LETTERS:
            tp_c = per_class_tp.get(rel, 0)
            fp_c = per_class_fp.get(rel, 0)
            fn_c = per_class_fn.get(rel, 0)
            support = tp_c + fn_c
            if support == 0 or (tp_c + fp_c) == 0:
                continue
            p_c = tp_c / (tp_c + fp_c)
            r_c = tp_c / (tp_c + fn_c)
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            weighted_num += f1_c * support
            weighted_denom += support
        weighted_f1 = weighted_num / weighted_denom if weighted_denom > 0 else 0.0

        avg_ce_loss = test_ce_loss / max(test_ce_count, 1)

        # === BioTriplex NER metrics on test set (generation task only) ===
        task_metrics: Dict[str, float] = {}
        if getattr(self.config, "task_type", "classification") == "generation":
            try:
                from .biotriplex_metrics import compute_ner_metrics, load_ner_gold_entities
                gold_path = getattr(self.config, "ner_gold_path", None)
                gold_map: Dict[str, Dict[str, set]] = {}
                if gold_path and os.path.exists(gold_path):
                    gold_map = load_ner_gold_entities(gold_path)
                golds_for_pred: List[Dict[str, set]] = []
                pred_aligned: List[str] = []
                for i, pred in enumerate(all_predictions):
                    dk = all_doc_keys[i] if i < len(all_doc_keys) else f"sample_{i}"
                    if dk in gold_map:
                        golds_for_pred.append(gold_map[dk])
                    else:
                        from .biotriplex_metrics import _parse_entities_json
                        golds_for_pred.append(_parse_entities_json(all_labels[i] if i < len(all_labels) else ""))
                    pred_aligned.append(pred)
                ner = compute_ner_metrics(pred_aligned, golds_for_pred, all_doc_keys)
                m = ner.get("metrics", {})
                task_metrics = {
                    "test_ner_macro_f1": m.get("macro_f1", 0.0),
                    "test_ner_weighted_f1": m.get("weighted_f1", 0.0),
                    "test_ner_micro_f1": m.get("overall_micro_f1", 0.0),
                    "test_ner_macro_precision": m.get("macro_precision", 0.0),
                    "test_ner_macro_recall": m.get("macro_recall", 0.0),
                    "test_ner_n_parse_failures": ner.get("n_parse_failures", 0),
                    "test_ner_f1_GENE": ner.get("per_class_metrics", {}).get("GENE", {}).get("f1", 0.0),
                    "test_ner_f1_DISEASE": ner.get("per_class_metrics", {}).get("DISEASE", {}).get("f1", 0.0),
                    "test_ner_f1_RELATION": ner.get("per_class_metrics", {}).get("RELATION", {}).get("f1", 0.0),
                }
            except Exception as e:
                logger.warning("NER test metric computation failed: %s", e)

        metrics = {
            "test_micro_precision": micro_p,
            "test_micro_recall": micro_r,
            "test_micro_f1": micro_f1,
            "test_micro_accuracy": micro_acc,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "test_ce_loss": avg_ce_loss,
            "test_samples": len(all_predictions_letters),
            **task_metrics,
        }
        logger.info(
            "Test results — BioTriplex letter-level: "
            "P=%.4f R=%.4f F1=%.4f Acc=%.4f Macro_F1=%.4f Weighted_F1=%.4f "
            "CE_loss=%.4f (%d samples)",
            micro_p, micro_r, micro_f1, micro_acc, macro_f1, weighted_f1,
            avg_ce_loss, len(all_predictions_letters),
        )
        return metrics
    def _finalize(self) -> Dict:
        # Write summary JSON with a timestamp-based name (never overwritten).
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.config.log_dir, f"training_metrics_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        logger.info("Metrics log → %s", summary_path)

        # === BioTriplex paper Table 5 reference (supervised LLaMA 3.1 8B) ===
        BIOTRIPLEX_REFERENCE = {
            "precision": 0.65,
            "recall": 0.62,
            "f1": 0.63,
        }
        if self.metrics_log:
            best = max(self.metrics_log, key=lambda m: m.get("val_entity_micro_f1", 0))
            logger.info(
                "BioTriplex-QA reference (Table 5, supervised LLaMA 3.1 8B): "
                "P=%.2f R=%.2f F1=%.2f | "
                "Best epoch %d: P=%.4f R=%.4f F1=%.4f | "
                "Macro_F1=%.4f | Weighted_F1=%.4f | CE_loss=%.4f",
                BIOTRIPLEX_REFERENCE["precision"],
                BIOTRIPLEX_REFERENCE["recall"],
                BIOTRIPLEX_REFERENCE["f1"],
                best.get("epoch", -1),
                best.get("val_micro_precision", 0),
                best.get("val_micro_recall", 0),
                best.get("val_entity_micro_f1", 0),
                best.get("val_macro_f1", 0),
                best.get("val_weighted_f1", 0),
                best.get("val_ce_loss", 0),
            )

        return {
            "best_metric": self.best_metric,
            "best_epoch": self.epoch,
            "total_steps": self.global_step,
            "elapsed_s": time.time() - self.start_time,
            "metrics_path": summary_path,
        }

    # ------------------------------------------------------------------------- #
    #  Metrics helpers (legacy — preserved for backward-compat tests)
    # ------------------------------------------------------------------------- #
    def _compute_f1(
        self,
        predictions: List[str],
        labels: List[str],
    ) -> float:
        """Legacy entity-level F1 (BioTriplex-NER style). New code paths
        use the letter-level metrics in :meth:`_run_val_epoch`."""
        from ..data.dataset import parse_gold_entities
        tp = fp = fn = 0
        for pred, label in zip(predictions, labels):
            pred_ents = set(parse_gold_entities(pred))
            label_ents = set(parse_gold_entities(label))
            tp += len(pred_ents & label_ents)
            fp += len(pred_ents - label_ents)
            fn += len(label_ents - pred_ents)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _compute_rouge(
        self,
        predictions: List[str],
        labels: List[str],
    ) -> float:
        """Legacy ROUGE-L (LCS-based). New code paths do not use ROUGE for
        the QA task since BioTriplex eval is letter-level."""
        total = 0.0
        for pred, label in zip(predictions, labels):
            lcs = self._lcs(pred, label)
            denom = max(len(pred), len(label))
            total += lcs / denom if denom > 0 else 0.0
        return total / len(predictions) if predictions else 0.0

    def _lcs(self, a: str, b: str) -> int:
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                else:
                    dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
        return dp[m % 2][n]
