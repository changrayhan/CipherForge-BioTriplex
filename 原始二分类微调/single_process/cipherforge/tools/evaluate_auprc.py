#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate TinyLlama(+LoRA) on the ClinVar QA test set: AUPRC/AUC/acc + per-gene AUPRC."""
import argparse
import json
import os

# Must be set before importing huggingface_hub / transformers (they read env at import time).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def predict_probs(model, tokenizer, rows, batch_size, device):
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
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            last = logits[torch.arange(logits.size(0), device=logits.device), last_pos]
            scores = torch.stack([last[:, yes_id], last[:, no_id]], dim=1).float()
            probs.extend(torch.softmax(scores, dim=1)[:, 0].cpu().tolist())
    return probs


def report(rows, probs, name):
    labels = [1 if r["output"].strip().lower().startswith("y") else 0 for r in rows]
    genes = [r.get("meta", {}).get("gene", "") for r in rows]
    out = {"name": name, "n": len(rows), "pos_rate": sum(labels) / len(labels)}
    out["auprc"] = average_precision_score(labels, probs)
    if len(set(labels)) > 1:
        out["auc"] = roc_auc_score(labels, probs)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    out["accuracy@0.5"] = accuracy_score(labels, preds)

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", default="", help="LoRA adapter dir; empty = zero-shot base model")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    data = Path(args.data) if args.data else base / "data" / "qa" / "test.jsonl"
    if args.out:
        out_path = Path(args.out)
    elif args.adapter:
        out_path = Path(args.adapter) / "metrics.json"
    else:
        out_path = base / "runs" / "metrics_zeroshot.json"

    rows = load_rows(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device} n={len(rows)} adapter={args.adapter or 'zero-shot'}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    probs = predict_probs(model, tokenizer, rows, args.batch_size, device)
    res = report(rows, probs, name=args.adapter or "zero-shot")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
