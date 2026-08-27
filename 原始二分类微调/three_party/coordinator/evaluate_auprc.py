#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-aware evaluation for TinyLlama(+LoRA) QA fine-tuning.

* binary  (ClinVar): AUPRC / AUC / accuracy@0.5 / macro-F1 / per-gene AUPRC
* multiclass (BioTriplex 7/21-class letter QA): letter accuracy, macro-F1,
  weighted-F1 and per-class P/R/F1

For multiclass we score the position right after the prompt (the SentencePiece
blank-prefix token position), where the model is trained to predict the class
letter, exactly mirroring ``remote_val`` in shared/remote_protocol.py.
"""
import argparse
import json
import os

# Must be set before importing huggingface_hub / transformers (they read env at
# import time).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def class_token_ids_from(tokenizer, class_outputs, answer_prefix=" "):
    ids = []
    for out in class_outputs:
        tok = tokenizer(answer_prefix + out, add_special_tokens=False).input_ids
        if len(tok) < 2:
            raise ValueError(
                f"class output {out!r} must tokenize with prefix into >=2 tokens, got {tok}"
            )
        ids.append(int(tok[1]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"class answer tokens are not unique: {ids}")
    return ids


def predict_probs(model, tokenizer, rows, batch_size, device, max_seq_length=128):
    """Binary path: P(Yes) at the last prompt position (baseline-identical)."""
    prompts = [f"{r['question']}\n\n{r['input']}\n\nAnswer:" for r in rows]
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids
    no_id = tokenizer("No", add_special_tokens=False).input_ids
    assert len(yes_id) == 1 and len(no_id) == 1, "Yes/No must be single tokens in this tokenizer"
    yes_id, no_id = yes_id[0], no_id[0]
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            last = logits[torch.arange(logits.size(0), device=logits.device), last_pos]
            scores = torch.stack([last[:, yes_id], last[:, no_id]], dim=1).float()
            probs.extend(torch.softmax(scores, dim=1)[:, 0].cpu().tolist())
    return probs


def predict_multiclass(model, tokenizer, rows, batch_size, device, class_outputs, max_seq_length=2048):
    """Multiclass path: logits over the class letters at the blank-prefix position."""
    class_tokens = class_token_ids_from(tokenizer, class_outputs)
    prompts = [f"{r['question']}\n\n{r['input']}\n\nAnswer: " for r in rows]
    logits_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            last = logits[torch.arange(logits.size(0), device=logits.device), last_pos]
            logits_list.append(last[:, class_tokens].float().cpu())
    return torch.cat(logits_list), class_tokens


def report_binary(rows, probs, name):
    labels = [1 if r["output"].strip().lower().startswith("y") else 0 for r in rows]
    genes = [r.get("meta", {}).get("gene", "") for r in rows]
    out = {"name": name, "task_type": "binary", "n": len(rows), "pos_rate": sum(labels) / len(labels)}
    out["auprc"] = average_precision_score(labels, probs)
    if len(set(labels)) > 1:
        out["auc"] = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    out["accuracy@0.5"] = accuracy_score(labels, preds)
    out["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)

    groups = defaultdict(list)
    for r, p, lab in zip(rows, probs, labels):
        groups[r.get("meta", {}).get("gene", "")].append((lab, p))
    per_gene = {}
    for g, items in groups.items():
        if len(items) < 10:
            continue
        gs = [it[0] for it in items]
        if len(set(gs)) < 2:
            continue
        per_gene[g] = {"n": len(items), "auprc": average_precision_score(gs, [it[1] for it in items])}
    vals = [v["auprc"] for v in per_gene.values()]
    out["per_gene"] = {
        "n_genes": len(vals),
        "mean_auprc": sum(vals) / len(vals) if vals else None,
        "min_auprc": min(vals) if vals else None,
        "max_auprc": max(vals) if vals else None,
    }
    out["majority_auprc"] = out["pos_rate"]
    return out


def report_multiclass(rows, class_logits, class_outputs, name,
                     prior_correction=0.0, class_priors=None):
    import numpy as np
    gold = [class_outputs.index(r["output"].strip()) for r in rows]
    pred = class_logits.argmax(dim=1).tolist()
    out = {
        "name": name,
        "task_type": "multiclass",
        "n": len(rows),
        "n_classes": len(class_outputs),
        "class_outputs": class_outputs,
        "prior_correction": float(prior_correction),
    }
    out["accuracy"] = accuracy_score(gold, pred)
    out["macro_f1"] = f1_score(gold, pred, average="macro", zero_division=0)
    out["weighted_f1"] = f1_score(gold, pred, average="weighted", zero_division=0)
    report = classification_report(
        gold, pred, labels=list(range(len(class_outputs))),
        target_names=class_outputs, output_dict=True, zero_division=0,
    )
    out["per_class"] = {
        k: v for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }

    # ---- post-hoc prior correction: logit'_c = logit_c - lam * log(p_c) ----
    # (rare classes get boosted; lambda > 0 favours recall on the long tail)
    if float(prior_correction) > 0.0 and class_priors is not None:
        priors = np.asarray(class_priors, dtype=np.float64)
        if priors.sum() > 0:
            priors = priors / priors.sum()
            logits_np = class_logits.float().numpy()
            adj = logits_np - float(prior_correction) * np.log(priors + 1e-12)
            pred_prior = adj.argmax(axis=1).tolist()
            out["accuracy_prior"] = accuracy_score(gold, pred_prior)
            out["macro_f1_prior"] = f1_score(
                gold, pred_prior, average="macro", zero_division=0)
            out["weighted_f1_prior"] = f1_score(
                gold, pred_prior, average="weighted", zero_division=0)
            rep_prior = classification_report(
                gold, pred_prior, labels=list(range(len(class_outputs))),
                target_names=class_outputs, output_dict=True, zero_division=0,
            )
            out["per_class_prior"] = {
                k: v for k, v in rep_prior.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir; empty = zero-shot base model")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--task_type", default="", help="clinvar | biotriplex21 | biotriplex7")
    ap.add_argument("--class_outputs", default="", help="JSON list of class answer strings")
    ap.add_argument("--max_seq_length", type=int, default=0)
    ap.add_argument("--train_data", default="", help="train.jsonl used to compute class priors")
    ap.add_argument("--prior_correction", type=float, default=0.0,
                    help="post-hoc prior correction strength lambda (0 = off)")
    args = ap.parse_args()

    # Resolve task profile defaults.
    task_type = args.task_type or "clinvar"
    class_outputs = None
    if args.class_outputs:
        class_outputs = json.loads(args.class_outputs)
    if class_outputs is None:
        try:
            from coordinator.task_profiles import get_profile
            profile = get_profile(task_type)
            class_outputs = list(profile["class_outputs"])
        except Exception:
            class_outputs = ["Yes", "No"] if task_type == "clinvar" else None
    # Only ClinVar's Yes/No profile is binary; everything else is multiclass.
    eval_mode = "binary"
    if class_outputs == ["Yes", "No"]:
        eval_mode = "binary"
    elif class_outputs:
        eval_mode = "multiclass"
    max_seq_length = args.max_seq_length or (128 if eval_mode == "binary" else 2048)

    base = Path(__file__).resolve().parents[1]
    data = Path(args.data) if args.data else base / "party_u" / "data" / "qa" / "test.jsonl"
    if args.out:
        out_path = Path(args.out)
    elif args.adapter:
        out_path = Path(args.adapter) / "metrics.json"
    else:
        out_path = base / "runs" / "metrics_zeroshot.json"

    rows = load_rows(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device} n={len(rows)} task_type={task_type} "
          f"eval_mode={eval_mode} adapter={args.adapter or 'zero-shot'}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    if eval_mode == "multiclass":
        class_logits, _ = predict_multiclass(
            model, tokenizer, rows, args.batch_size, device, class_outputs, max_seq_length)
        class_priors = None
        if args.train_data and os.path.exists(args.train_data):
            tr_rows = load_rows(args.train_data)
            counts = [0] * len(class_outputs)
            for r in tr_rows:
                try:
                    counts[class_outputs.index(r["output"].strip())] += 1
                except ValueError:
                    pass
            class_priors = counts
        res = report_multiclass(
            rows, class_logits, class_outputs, name=args.adapter or "zero-shot",
            prior_correction=args.prior_correction, class_priors=class_priors)
    else:
        probs = predict_probs(model, tokenizer, rows, args.batch_size, device, max_seq_length)
        res = report_binary(rows, probs, name=args.adapter or "zero-shot")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
