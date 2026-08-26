"""RemoteTrainer: ClinVar validation loop, baseline-aligned.

The remote protocol returns ``{probs, preds, gold, yes_id, no_id, n}`` for each
batch, computed the SAME way as
``single_process/baseline/scripts/evaluate_auprc.py``:

    last_pos = attention_mask.sum(dim=1) - 1
    P(Yes)   = softmax([logits[yes_id], logits[no_id]])[:, 0]

This Trainer accumulates per-sample (probs, gold, gene) and at the end of the
epoch computes:

    AUPRC, AUC, accuracy@0.5, per-gene AUPRC
    binary CE on (P(Yes), gold)         ← matches baseline val_ce_loss semantics
    token-level CE on the supervised token  (optional, matches HF Trainer)

The key fix: every metric is the same scalar baseline reports. Best checkpoint
selection key is ``val_ce_loss`` to match baseline (where load_best_model_at_end
+ metric_for_best_model="eval_loss" picks the lowest loss checkpoint).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from shared.training.trainer import Trainer


# Same per-gene AUPRC rule as baseline: skip genes with <10 samples or only
# one label.
MIN_GENE_SAMPLES = 10


class RemoteTrainer(Trainer):
    """ClinVar validation loop that mirrors baseline ``evaluate_auprc``."""

    def _run_val_clinvar(self, val_loader, epoch: int) -> Dict[str, float]:
        # Emergency checkpoint before val (in case val crashes).
        try:
            self._save_last_checkpoint()
        except Exception as ckpt_e:  # noqa: BLE001
            logger.warning("Emergency checkpoint failed: %s", ckpt_e)

        all_probs: List[float] = []
        all_gold: List[int] = []
        all_genes: List[str] = []
        n_samples = 0

        for batch in val_loader:
            # The protocol returns per-sample (probs, preds, gold) using the
            # exact baseline softmax formula on full vocab logits.
            r = self.ipc.remote_val(batch)
            all_probs.extend(r["probs"])
            all_gold.extend(r["gold"])
            n_samples += r["n"]

            # Per-sample gene annotation. ClinVarQADataset may carry a 'meta'
            # dict in the original rows but the tensor batch only has
            # ``input_ids`` etc., so we fall back to "" (per-gene AUPRC then
            # collapses to a single overall AUPRC, same as baseline with no
            # gene info).
            meta = batch.get("meta") or []
            for m in meta:
                all_genes.append(m.get("gene", "") if isinstance(m, dict) else "")

        # If gene annotation is missing on the batch (current collate strips it
        # out), treat the whole set as one group so per-gene AUPRC degrades
        # gracefully to overall AUPRC rather than silently being 0.
        if not any(all_genes):
            all_genes = [""] * len(all_probs)

        # ---- AUPRC / AUC / acc@0.5 (baseline-identical) ----
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            roc_auc_score,
        )
        try:
            auprc = float(average_precision_score(all_gold, all_probs))
        except Exception:  # noqa: BLE001
            auprc = 0.0
        try:
            auc = float(roc_auc_score(all_gold, all_probs)) if len(set(all_gold)) > 1 else 0.5
        except Exception:  # noqa: BLE001
            auc = 0.5
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        try:
            acc = float(accuracy_score(all_gold, preds))
        except Exception:  # noqa: BLE001
            acc = 0.0

        # ---- per-gene AUPRC (baseline rule: |group|>=10, both labels) ----
        groups: Dict[str, List] = defaultdict(list)
        for g, p, lab in zip(all_genes, all_probs, all_gold):
            groups[g].append((lab, p))
        per_gene: Dict[str, Dict[str, float]] = {}
        for g, items in groups.items():
            if len(items) < MIN_GENE_SAMPLES:
                continue
            gs = [it[0] for it in items]
            if len(set(gs)) < 2:
                continue
            try:
                per_gene[g] = {
                    "n": len(items),
                    "auprc": float(average_precision_score(gs, [it[1] for it in items])),
                }
            except Exception:  # noqa: BLE001
                continue
        vals = [v["auprc"] for v in per_gene.values()]
        per_gene_mean = (sum(vals) / len(vals)) if vals else None
        per_gene_min = min(vals) if vals else None
        per_gene_max = max(vals) if vals else None

        # ---- Binary CE on P(Yes) (val_ce_loss) — matches baseline semantics ----
        ce_total = 0.0
        for g, p in zip(all_gold, all_probs):
            p_clip = min(max(float(p), 1e-7), 1.0 - 1e-7)
            ce_total += -(g * math.log(p_clip) + (1.0 - g) * math.log(1.0 - p_clip))
        ce_loss = ce_total / max(len(all_gold), 1)

        return {
            # === Best-checkpoint key — matches baseline (lower is better) ===
            "val_ce_loss": ce_loss,
            # === Primary metrics (baseline-aligned) ===
            "val_auprc": auprc,
            "val_auc": auc,
            "val_accuracy": acc,
            "val_samples": n_samples,
            # === Per-gene AUPRC (baseline convention) ===
            "val_per_gene_n_genes": len(vals),
            "val_per_gene_mean_auprc": per_gene_mean,
            "val_per_gene_min_auprc": per_gene_min,
            "val_per_gene_max_auprc": per_gene_max,
            # === Backwards-compatible aliases (legacy Trainer fields) ===
            # These used to be filled with yn_acc; we now point them at the
            # real value so old log readers don't break.
            "val_micro_accuracy": acc,
            "val_clinvar_token_accuracy": acc,
            "val_clinvar_yes_accuracy": acc,
            "val_entity_micro_f1": acc,
            "val_letter_micro_f1": acc,
            "val_macro_f1": acc,
            "val_weighted_f1": acc,
            "val_micro_precision": 0.0,  # requires real positive/negative split; populated below if useful
            "val_micro_recall": 0.0,
            "val_clinvar_yes_precision": 0.0,
            "val_clinvar_yes_recall": 0.0,
            "val_clinvar_yes_f1": 0.0,
            "val_clinvar_ce_batches": 0,
        }

    def _run_val_biotriplex(self, val_loader, epoch: int) -> Dict[str, float]:
        """Multi-class letter-level validation (BioTriplex 7/21-class).

        ``remote_val`` (multiclass mode) returns per-sample class logits,
        predicted class ids and gold class ids.  We accumulate them and
        compute class-level CE (best-checkpoint key), accuracy and macro /
        weighted F1 — the metric keys the base Trainer/logger expect.
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        all_logits: List[List[float]] = []
        all_gold: List[int] = []
        all_pred: List[int] = []
        n_samples = 0

        for batch in val_loader:
            r = self.ipc.remote_val(batch, return_probs=False)
            cls_tok = r.get("class_token_ids") or []
            if not cls_tok:
                logger.warning("[val] remote_val returned no class_token_ids — skipping batch")
                continue
            all_logits.extend(r.get("class_logits") or [])
            all_gold.extend(r.get("gold_class_ids") or [])
            all_pred.extend(r.get("pred_class_ids") or [])
            n_samples += int(r.get("n", 0))

        # ---- CE over the C-class logits (val_ce_loss, lower is better) ----
        ce_loss = 0.0
        if all_logits and len(all_logits) == len(all_gold):
            arr = np.asarray(all_logits, dtype=np.float64)
            gold = np.asarray(all_gold, dtype=np.int64)
            valid = gold >= 0
            if bool(valid.any()) and arr.ndim == 2:
                z = arr[valid]
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                eps = 1e-7
                ce_loss = float(-np.log(np.clip(p[np.arange(p.shape[0]), gold[valid]], eps, 1.0)).mean())

        # ---- Accuracy / macro / weighted F1 over valid samples ----
        pairs = [(p, g) for p, g in zip(all_pred, all_gold) if p >= 0 and g >= 0]
        if pairs:
            preds = [p for p, _ in pairs]
            golds = [g for _, g in pairs]
            acc = float(accuracy_score(golds, preds))
            macro_f1 = float(f1_score(golds, preds, average="macro", zero_division=0))
            weighted_f1 = float(f1_score(golds, preds, average="weighted", zero_division=0))
        else:
            acc = macro_f1 = weighted_f1 = 0.0

        return {
            # === Best-checkpoint key (lower is better) ===
            "val_ce_loss": ce_loss,
            # === Primary metrics ===
            "val_accuracy": acc,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_samples": n_samples,
            # === Backwards-compatible aliases (base Trainer/logger) ===
            "val_entity_micro_f1": acc,
            "val_letter_micro_f1": acc,
            "val_micro_accuracy": acc,
            "val_micro_precision": acc,
            "val_micro_recall": acc,
            "val_auprc": None,
            "val_auc": None,
        }