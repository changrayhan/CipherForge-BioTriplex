"""ClinVar variant-pathogenicity QA dataset for the SLG-HE-PIR pipeline.

Reads the BioTriplex-compatible JSONL splits produced by
``clinvar_plain/scripts/build_clinvar_qa.py`` (fields: id/question/input/
output + meta) and emits tensors the HeterogeneousProtocol expects:
  - input_ids / attention_mask
  - output_ids: gold token id ONLY at the answer position, -100 elsewhere
    (prompt and padding) -- this is what PartyS uses to select Enc(-V_gold)
    and PartyM uses to zero non-answer gradients.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


def load_clinvar_samples(data_dir: str) -> tuple:
    """Load train/val/test JSONL rows (list of dicts)."""
    data_dir = Path(data_dir)

    def _load(name: str) -> List[Dict]:
        with open(data_dir / name, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    return _load("train.jsonl"), _load("val.jsonl"), _load("test.jsonl")


class ClinVarQADataset(Dataset):
    """Tokenized ClinVar QA samples with masked answer-only labels."""

    def __init__(self, rows: List[Dict], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.examples = []
        for r in rows:
            prompt = f"{r['question']}\n\n{r['input']}\n\nAnswer:"
            p = tokenizer(prompt, add_special_tokens=True)
            a = tokenizer(" " + r["output"], add_special_tokens=False)
            ids = p["input_ids"] + a["input_ids"]
            labels = [-100] * len(p["input_ids"]) + a["input_ids"]
            if len(ids) > max_length:
                # Keep the answer; truncate the prompt head.
                keep = max_length - len(a["input_ids"])
                if keep <= 0:
                    ids = ids[:max_length]
                    labels = [-100] * max_length
                else:
                    ids = p["input_ids"][-keep:] + a["input_ids"]
                    labels = [-100] * keep + a["input_ids"]
            # Fixed-length batches are the framework contract (its collate
            # torch.stacks tensors; variable lengths fall back to lists).
            pad_len = max_length - len(ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            ids = ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            mask = [1] * (max_length - pad_len) + [0] * pad_len
            self.examples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "output_ids": torch.tensor(labels, dtype=torch.long),
                "output_text": r["output"],
                "id": r.get("id", str(len(self.examples))),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]
