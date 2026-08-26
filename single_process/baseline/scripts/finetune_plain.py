#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plaintext LoRA fine-tuning of TinyLlama on the ClinVar QA dataset."""
import argparse
import json
import math
import os
import sys

# Must be set before importing huggingface_hub / transformers (they read env at import time).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class QADataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.examples = []
        for r in rows:
            prompt = f"{r['question']}\n\n{r['input']}\n\nAnswer:"
            p = tokenizer(prompt, add_special_tokens=True)
            a = tokenizer(" " + r["output"], add_special_tokens=False)
            ids = p["input_ids"] + a["input_ids"]
            labels = [-100] * len(p["input_ids"]) + a["input_ids"]
            if len(ids) > max_len:
                # Truncate the PROMPT head, never the answer (else loss=0).
                keep = max_len - len(a["input_ids"])
                if keep <= 0:
                    ids = ids[:max_len]
                    labels = [-100] * max_len
                else:
                    ids = p["input_ids"][-keep:] + a["input_ids"]
                    labels = [-100] * keep + a["input_ids"]
            self.examples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class _Tee:
    """Mirror a stream (stdout/stderr) to a log file as well."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8")

    def write(self, msg):
        self.stream.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def collate(batch, pad_id):
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop after N consecutive evals without eval_loss improvement (0 = disabled)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else base / "data" / "qa"
    out_dir = Path(args.out_dir) if args.out_dir else base / "runs" / "clinvar_tinylama_plain"
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, out_dir / "train.log")
    sys.stderr = _Tee(sys.stderr, out_dir / "train.log")

    train_rows = load_rows(data_dir / "train.jsonl")
    val_rows = load_rows(data_dir / "val.jsonl")
    print(f"[data] train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = QADataset(train_rows, tokenizer, args.max_seq_len)
    val_ds = QADataset(val_rows, tokenizer, args.max_seq_len) if val_rows else None
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)

    targs = dict(
        output_dir=str(out_dir / "ckpt"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.1 * total_steps)),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.eval_steps if val_ds else None,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss" if val_ds else None,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**targs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)] if args.early_stop_patience > 0 else None,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "train_args.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)
    print(f"[ok] adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
