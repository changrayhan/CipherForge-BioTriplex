"""BioTriplex task-specific metric computation for SLG-HE-PIR.

These functions compute the metrics mandated by
``docs/BIOTRIPLEX_FINETUNE_README.md``:

* **Task A (classification, 7-class GenRel QA)**
  - ``compute_classification_metrics()`` returns the full set of
    multi-label / macro / weighted / ROC AUC metrics, keyed by the
    README schema.

* **Task B (generation, NER JSON)**
  - ``compute_ner_metrics()`` returns per-class (GENE / DISEASE /
    RELATION) span-level P / R / F1 plus the macro / weighted / overall
    micro aggregates.

Both are pure-Python / numpy / sklearn — they take the trainer's
``predictions`` (list of raw generated texts) and ``labels`` (list of
gold texts) plus the ``doc_keys`` for entity alignment.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Task A — Classification (GenRel QA)
# ---------------------------------------------------------------------------
GENERAL_RELATIONS: List[str] = [
    "pathological",
    "modulatory",
    "expression change",
    "diagnosis",
    "therapy",
    "no relation",
    "relation undefined",
]
OPTION_LETTERS: List[str] = ["a", "b", "c", "d", "e", "f", "g"]
OPTION_TO_RELATION: Dict[str, str] = {
    chr(ord("a") + i): r for i, r in enumerate(GENERAL_RELATIONS)
}


def _parse_letter_answer(text: Any) -> Optional[str]:
    """Reduce raw model output to a coarse relation string (or None on parse fail)."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if not t:
        return None
    head = t[0]
    if head in OPTION_TO_RELATION:
        return OPTION_TO_RELATION[head]
    return None


def _parse_letters_multilabel(text: Any) -> Set[int]:
    """Parse ``j), o)`` style outputs into a set of 0-based relation indices."""
    if text is None:
        return set()
    out: Set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        ch = part[0].lower()
        if "a" <= ch <= "g":
            out.add(ord(ch) - ord("a"))
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def compute_classification_metrics(
    predictions: List[str],
    labels: List[str],
    pred_logits: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """Compute the full set of classification metrics mandated by the README.

    Args:
        predictions: list of raw model outputs (e.g. ``"a)"`` / ``"j), o)"``).
        labels:      list of gold letter answers in the same format.
        pred_logits: optional ``[N, 7]`` logits from the last non-pad position
            projected onto the seven option token ids. When provided, the
            ROC AUC step is computed from a real softmax instead of the
            one-hot fallback (no logits → keeps the old one-hot behaviour).

    Returns:
        A dict matching the README's ``genrel_<TS>_evaluate_metrics.json``
        schema:

        * ``n_samples``
        * ``n_parse_failures``
        * ``metrics``: {micro_accuracy, macro_precision, macro_recall,
          macro_f1, weighted_f1, micro_f1, multilabel_f1_samples,
          multilabel_f1_macro, multilabel_f1_micro, macro_roc_auc_ovr,
          micro_roc_auc_ovr}
        * ``per_class_metrics``: {relation_name: {precision, recall, f1, support}}
        * ``y_true_distribution`` / ``y_pred_distribution``: {rel_name: count}
        * ``has_logits``: bool indicating whether ``pred_logits`` was used.
    """
    n_samples = len(predictions)
    if n_samples == 0:
        return {
            "task": "GenRel QA (7-class Classification)",
            "n_samples": 0,
            "n_parse_failures": 0,
            "metrics": {},
            "per_class_metrics": {},
            "y_true_distribution": {r: 0 for r in GENERAL_RELATIONS},
            "y_pred_distribution": {r: 0 for r in GENERAL_RELATIONS},
        }

    # ------------------------------------------------------------------ #
    #  Single-label argmax view
    # ------------------------------------------------------------------ #
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    parse_fail = 0
    for pred, label in zip(predictions, labels):
        # Use argmax-style: first letter of each.
        p = _parse_letter_answer(pred)
        g = _parse_letter_answer(label)
        if p is None:
            parse_fail += 1
            y_pred_idx.append(-1)
        else:
            y_pred_idx.append(GENERAL_RELATIONS.index(p))
        if g is None:
            # gold must always parse; treat as missing class
            y_true_idx.append(-1)
        else:
            y_true_idx.append(GENERAL_RELATIONS.index(g))

    # Replace missing predictions with a "default" class so downstream
    # sklearn calls do not crash; track them as parse failures.
    if any(i == -1 for i in y_pred_idx):
        default_idx = GENERAL_RELATIONS.index("relation undefined")
        y_pred_idx = [default_idx if i == -1 else i for i in y_pred_idx]
    valid_pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t != -1]
    if valid_pairs:
        y_t = np.array([t for t, _ in valid_pairs], dtype=np.int64)
        y_p = np.array([p for _, p in valid_pairs], dtype=np.int64)
    else:
        y_t = np.zeros(0, dtype=np.int64)
        y_p = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  Per-class P / R / F1 (micro + macro + weighted via sklearn)
    # ------------------------------------------------------------------ #
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
        confusion_matrix,
        roc_auc_score,
    )

    n_classes = len(GENERAL_RELATIONS)
    labels_range = list(range(n_classes))

    micro_accuracy = float(accuracy_score(y_t, y_p)) if y_t.size else 0.0
    macro_precision = float(precision_score(y_t, y_p, average="macro", labels=labels_range, zero_division=0)) if y_t.size else 0.0
    macro_recall = float(recall_score(y_t, y_p, average="macro", labels=labels_range, zero_division=0)) if y_t.size else 0.0
    macro_f1 = float(f1_score(y_t, y_p, average="macro", labels=labels_range, zero_division=0)) if y_t.size else 0.0
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", labels=labels_range, zero_division=0)) if y_t.size else 0.0
    micro_f1 = float(f1_score(y_t, y_p, average="micro", labels=labels_range, zero_division=0)) if y_t.size else 0.0

    # Per-class support (number of gold samples per class)
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    y_true_dist = {r: 0 for r in GENERAL_RELATIONS}
    y_pred_dist = {r: 0 for r in GENERAL_RELATIONS}
    for t in y_t:
        y_true_dist[GENERAL_RELATIONS[int(t)]] += 1
    for p in y_p:
        y_pred_dist[GENERAL_RELATIONS[int(p)]] += 1
    for i, rel in enumerate(GENERAL_RELATIONS):
        # Per-class P/R/F1 via sklearn's classification_report-style call.
        # zero_division=0 keeps empty classes at 0.
        p_i = float(precision_score(y_t, y_p, labels=[i], average="micro", zero_division=0)) if y_t.size else 0.0
        r_i = float(recall_score(y_t, y_p, labels=[i], average="micro", zero_division=0)) if y_t.size else 0.0
        f1_i = float(f1_score(y_t, y_p, labels=[i], average="micro", zero_division=0)) if y_t.size else 0.0
        per_class_metrics[rel] = {
            "precision": p_i,
            "recall": r_i,
            "f1": f1_i,
            "support": int(y_true_dist[rel]),
        }

    # ------------------------------------------------------------------ #
    #  Multi-label view (treat each relation as a binary problem)
    # ------------------------------------------------------------------ #
    y_true_multi = np.zeros((len(predictions), n_classes), dtype=np.int32)
    y_pred_multi = np.zeros_like(y_true_multi)
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        g_letters = _parse_letters_multilabel(label)
        p_letters = _parse_letters_multilabel(pred)
        for j in g_letters:
            y_true_multi[i, j] = 1
        for j in p_letters:
            y_pred_multi[i, j] = 1

    # multilabel F1 requires sklearn>=0.24; fall back gracefully if missing.
    try:
        from sklearn.metrics import f1_score as _f1
        ml_f1_samples = float(_f1(y_true_multi, y_pred_multi, average="samples", zero_division=0))
        ml_f1_macro = float(_f1(y_true_multi, y_pred_multi, average="macro", zero_division=0))
        ml_f1_micro = float(_f1(y_true_multi, y_pred_multi, average="micro", zero_division=0))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("multilabel F1 failed: %s", e)
        ml_f1_samples = ml_f1_macro = ml_f1_micro = 0.0

    # ------------------------------------------------------------------ #
    #  ROC AUC (one-vs-rest)
    # ------------------------------------------------------------------ #
    macro_roc_auc = None
    micro_roc_auc = None
    has_logits = bool(pred_logits) and len(pred_logits) == len(y_t)
    if y_t.size and y_p.size:
        # Build soft scores. Preferred path: real 7-class logits from the
        # ``PartyS`` 7-class projection — softmax over the 7 option ids.
        # Fallback path: one-hot at argmax (or 1/n_classes uniform when both
        # argmax and label are missing), reproducing the previous contract.
        try:
            if has_logits:
                logits_arr = np.asarray(pred_logits, dtype=np.float64)
                # Row-wise softmax
                shifted = logits_arr - logits_arr.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                y_score = exp / exp.sum(axis=1, keepdims=True)
            else:
                y_score = np.zeros((len(y_t), n_classes), dtype=np.float64)
                for i, p in enumerate(y_p):
                    if 0 <= int(p) < n_classes:
                        y_score[i, int(p)] = 1.0
                # Renormalize for the (unlikely) all-missing case.
                row_sum = y_score.sum(axis=1, keepdims=True)
                all_zero = row_sum == 0
                y_score = np.where(
                    all_zero, 1.0 / n_classes, y_score / np.maximum(row_sum, 1)
                )

            # Per-class AUC. Compute explicitly so we can average across only
            # classes that have at least one positive sample in y_true — this
            # avoids sklearn's "Only one class is present in y_true" ValueError
            # on small/imbalanced val sets (e.g. 3-step post-fix verification
            # where the model collapses to a single class).
            per_class_auc = np.full(n_classes, np.nan, dtype=np.float64)
            for k in labels_range:
                yt_k = (y_t == k).astype(np.int32)
                if yt_k.sum() == 0:
                    # Class missing in y_true → undefined, skip
                    continue
                if yt_k.sum() == len(yt_k):
                    # All positives → undefined, skip
                    continue
                try:
                    per_class_auc[k] = float(roc_auc_score(yt_k, y_score[:, k]))
                except ValueError:
                    per_class_auc[k] = np.nan

            valid_aucs = per_class_auc[~np.isnan(per_class_auc)]
            if valid_aucs.size > 0:
                macro_roc_auc = float(np.mean(valid_aucs))
            # Micro AUC: flatten one-hot and scores then compute a single AUC.
            try:
                yt_onehot = np.eye(n_classes, dtype=np.int32)[y_t]
                micro_roc_auc = float(
                    roc_auc_score(yt_onehot.ravel(), y_score.ravel())
                )
            except ValueError:
                micro_roc_auc = None
        except Exception as e:
            logger.warning("ROC AUC computation failed: %s", e)
            macro_roc_auc = None
            micro_roc_auc = None

    return {
        "task": "GenRel QA (7-class Classification)",
        "n_samples": n_samples,
        "n_parse_failures": int(parse_fail),
        "has_logits": has_logits,
        "metrics": {
            "micro_accuracy": round(micro_accuracy, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "micro_f1": round(micro_f1, 6),
            "multilabel_f1_samples": round(ml_f1_samples, 6),
            "multilabel_f1_macro": round(ml_f1_macro, 6),
            "multilabel_f1_micro": round(ml_f1_micro, 6),
            "macro_roc_auc_ovr": round(macro_roc_auc, 6) if macro_roc_auc is not None else None,
            "micro_roc_auc_ovr": round(micro_roc_auc, 6) if micro_roc_auc is not None else None,
        },
        "per_class_metrics": {
            rel: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}
            for rel, m in per_class_metrics.items()
        },
        "y_true_distribution": y_true_dist,
        "y_pred_distribution": y_pred_dist,
        # also include confusion-matrix for debug
        "confusion_matrix": confusion_matrix(y_t, y_p, labels=labels_range).tolist() if y_t.size else [],
    }


# ---------------------------------------------------------------------------
#  Task B — NER generation
# ---------------------------------------------------------------------------
ENTITY_TYPES: List[str] = ["GENE", "DISEASE", "RELATION"]


def _find_first_json_array(text: str) -> Optional[str]:
    """Extract the first JSON array ``[{...}, ...]`` substring from text.

    The baseline ``ner_infer.py`` strips ``### Response:\\n`` and returns the
    raw completion. Models frequently emit surrounding chatter, so we search
    for the first ``[`` and the matching ``]`` (greedy).
    """
    if not text:
        return None
    s = str(text)
    # Strip assistant/Response framing (mirrors baseline evaluate_metrics.py)
    if "assistant\n\n" in s:
        s = s.split("assistant\n\n")[-1]
    elif "### Response:\n" in s:
        try:
            s = s.split("### Response:\n")[1]
        except IndexError:
            return None
    s = s.strip()
    start = s.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_entities_json(text: str) -> Dict[str, Set[str]]:
    """Parse a JSON list output to ``{entity_type: set(spans)}``.

    Returns an empty dict on parse failure or empty list.
    """
    raw = _find_first_json_array(text)
    if raw is None:
        return {et: set() for et in ENTITY_TYPES}
    try:
        items = json.loads(raw)
    except Exception:
        return {et: set() for et in ENTITY_TYPES}
    bucket: Dict[str, Set[str]] = {et: set() for et in ENTITY_TYPES}
    if not isinstance(items, list):
        return bucket
    for ent in items:
        if not isinstance(ent, dict):
            continue
        span = ent.get("span")
        et = ent.get("entity_type")
        if span is None or et is None:
            continue
        et = str(et).upper()
        if et not in bucket:
            continue
        bucket[et].add(str(span))
    return bucket


def _parse_gold_entities_from_text(sent: str, entities: List[List[int]]) -> Dict[str, Set[str]]:
    """Map gold ``[start, end, entity_type]`` triples to ``{entity_type: set(spans)}``."""
    bucket: Dict[str, Set[str]] = {et: set() for et in ENTITY_TYPES}
    for ent in entities:
        try:
            s, e, et = ent[0], ent[1], ent[2]
        except (IndexError, TypeError):
            continue
        et = str(et).upper()
        if et not in bucket:
            continue
        try:
            span = sent[s:e]
        except Exception:
            continue
        bucket[et].add(span)
    return bucket


def compute_ner_metrics(
    predictions: List[str],
    golds: List[Dict[str, Set[str]]],
    doc_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute BioTriplex NER metrics from raw model outputs and gold entities.

    Args:
        predictions: list of raw model outputs (JSON list text).
        golds:       list aligned with ``predictions``; each entry is
            ``{entity_type: set(span)}`` already lifted from gold entities.
        doc_keys:    optional list of sample ids (used for parse-failure
            diagnostics).

    Returns:
        Dict matching the README's ``ner_<TS>_evaluate_metrics.json`` schema:
        ``metrics``, ``per_class_metrics``, ``per_class_parse_failures``.
    """
    n = len(predictions)
    parse_failures = 0
    per_class_parse_fail = {et: 0 for et in ENTITY_TYPES}

    tp = {et: 0 for et in ENTITY_TYPES}
    fp = {et: 0 for et in ENTITY_TYPES}
    fn = {et: 0 for et in ENTITY_TYPES}

    for i, (pred, gold) in enumerate(zip(predictions, golds)):
        pred_b = _parse_entities_json(pred)
        if not any(len(v) > 0 for v in pred_b.values()):
            # Count as parse failure if we expected something
            parse_failures += 1
        for et in ENTITY_TYPES:
            p_set = pred_b.get(et, set())
            g_set = gold.get(et, set())
            tp[et] += len(p_set & g_set)
            fp[et] += len(p_set - g_set)
            fn[et] += len(g_set - p_set)
            if not p_set and g_set:
                per_class_parse_fail[et] += 1

    # Per-class P/R/F1
    per_class: Dict[str, Dict[str, Any]] = {}
    for et in ENTITY_TYPES:
        p = _safe_div(tp[et], tp[et] + fp[et])
        r = _safe_div(tp[et], tp[et] + fn[et])
        f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
        per_class[et] = {
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1, 6),
            "tp": tp[et],
            "fp": fp[et],
            "fn": fn[et],
        }

    # Macro F1 (avg over classes with support > 0)
    macro_f1_vals = [per_class[et]["f1"] for et in ENTITY_TYPES if (tp[et] + fn[et]) > 0]
    macro_f1 = float(np.mean(macro_f1_vals)) if macro_f1_vals else 0.0

    # Weighted F1 (by per-class support)
    support = {et: tp[et] + fn[et] for et in ENTITY_TYPES}
    total_support = sum(support.values())
    if total_support > 0:
        weighted_f1 = sum(per_class[et]["f1"] * support[et] for et in ENTITY_TYPES) / total_support
    else:
        weighted_f1 = 0.0

    # Overall micro P/R/F1 (sum TP/FP/FN across all classes)
    tot_tp = sum(tp.values())
    tot_fp = sum(fp.values())
    tot_fn = sum(fn.values())
    micro_p = _safe_div(tot_tp, tot_tp + tot_fp)
    micro_r = _safe_div(tot_tp, tot_tp + tot_fn)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "task": "NER (Span-level Exact-match)",
        "n_common_doc_keys": n,
        "n_parse_failures": parse_failures,
        "metrics": {
            "macro_precision": round(_safe_div(sum(per_class[et]["precision"] for et in ENTITY_TYPES), 3), 6),
            "macro_recall": round(_safe_div(sum(per_class[et]["recall"] for et in ENTITY_TYPES), 3), 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "overall_micro_precision": round(micro_p, 6),
            "overall_micro_recall": round(micro_r, 6),
            "overall_micro_f1": round(micro_f1, 6),
        },
        "per_class_metrics": per_class,
        "per_class_parse_failures": per_class_parse_fail,
    }


# ---------------------------------------------------------------------------
#  Gold entities loader (used by evaluate_biotriplex.py at Stage 2)
# ---------------------------------------------------------------------------
def load_ner_gold_entities(
    gold_jsonl_path: str,
    doc_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Set[str]]]:
    """Load gold entities keyed by doc_key.

    Args:
        gold_jsonl_path: path to ``{split}_gold_ner.txt`` written by
            ``BioTriplexQADatasetGeneration``.
        doc_keys: optional filter list; if provided, only docs whose key
            appears here are returned.

    Returns:
        ``{doc_key: {entity_type: set(span)}}``
    """
    out: Dict[str, Dict[str, Set[str]]] = {}
    with open(gold_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            key = item.get("doc_key")
            sent = item.get("input", "")
            ents = item.get("entities", []) or []
            if key is None:
                continue
            if doc_keys is not None and key not in doc_keys:
                continue
            out[key] = _parse_gold_entities_from_text(sent, ents)
    return out