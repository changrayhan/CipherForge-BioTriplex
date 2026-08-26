"""BioTriplex-QA dataset utilities for SLG-HE-PIR.

Loads the JSONL train/val/test splits and wraps them in PyTorch-ready Dataset
objects with a LlamaTokenizerWrapper.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# BioTriplex-QA relation extraction options (common relation types)
# These are the valid answer letters for multiple-choice questions
OPTIONS_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                   'm', 'n', 'o', 'p', 'q', 'r', 's', 't']


# --------------------------------------------------------------------------- #
#  Tokenizer wrapper                                                           #
# --------------------------------------------------------------------------- #

class LlamaTokenizerWrapper:
    """Thin wrapper around a HF tokenizer that produces dicts with pad-safe keys."""

    def __init__(self, model_path: str, max_length: int = 512):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        # Llama fast tokenizer sometimes misbehaves; fall back to slow if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, text: str, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=True,
        )

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


# --------------------------------------------------------------------------- #
#  Dataset class                                                              #
# --------------------------------------------------------------------------- #

class BioTriplexQADataset(Dataset):
    """PyTorch Dataset for BioTriplex-QA (NER-style entity extraction).

    Each sample has fields: id, input (text), question (prompt), output (answer).

    For training: formats as "question\n\ninput" → model → output
    The dataset returns dicts with input_ids, attention_mask, and output_text.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: LlamaTokenizerWrapper,
        max_length: int = 512,
        task: str = "train",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        question = sample.get("question", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        sample_id = sample.get("id", str(idx))

        # Format prompt: "question\n\ninput_text"
        full_text = f"{question}\n\n{input_text}"

        # Tokenize
        encoded = self.tokenizer.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "prompt": question,
            "input_text": input_text,
            "output_text": output_text,
            "id": sample_id,
        }

        # Optionally include label input IDs (for loss computation during training)
        if self.task in ("train", "val", "test"):
            label_text = output_text
            # Tokenize label (we'll use the same tokenizer, shift later in loss)
            label_encoded = self.tokenizer.tokenizer(
                label_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["labels"] = label_encoded["input_ids"].squeeze(0)

        return item


# --------------------------------------------------------------------------- #
#  Dataset loading                                                            #
# --------------------------------------------------------------------------- #

def load_biotriplex_dataset(
    data_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple:
    """Load BioTriplex-QA dataset from JSONL files.

    Args:
        data_dir: Path to directory containing train.jsonl, val.jsonl, test.jsonl
        train_ratio: Ratio of train split (if val/test not pre-split)
        seed: Random seed for shuffling

    Returns:
        (train_samples, val_samples, test_samples) each as list of dicts
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load all splits
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    if train_path.exists():
        train_samples = _load_jsonl(train_path)
        logger.info("Loaded train: %d samples from %s", len(train_samples), train_path)
    else:
        raise FileNotFoundError(f"train.jsonl not found in {data_dir}")

    if val_path.exists():
        val_samples = _load_jsonl(val_path)
        logger.info("Loaded val: %d samples from %s", len(val_samples), val_path)
    else:
        logger.warning("val.jsonl not found in %s, splitting from train", data_dir)
        val_samples = []

    if test_path.exists():
        test_samples = _load_jsonl(test_path)
        logger.info("Loaded test: %d samples from %s", len(test_samples), test_path)
    else:
        logger.warning("test.jsonl not found in %s, splitting from train", data_dir)
        test_samples = []

    # If val/test are empty, split from train
    if not val_samples or not test_samples:
        random.seed(seed)
        shuffled = train_samples.copy()
        random.shuffle(shuffled)

        if not val_samples:
            split_i = int(len(shuffled) * train_ratio)
            val_samples = shuffled[:split_i]
            train_samples = shuffled[split_i:]
            logger.info("Split val: %d samples from train", len(val_samples))

        if not test_samples and train_ratio < 1.0:
            # Further split the remaining train
            remaining = train_samples.copy()
            random.shuffle(remaining)
            test_ratio = (1.0 - train_ratio) / train_ratio if train_ratio > 0 else 0.1
            test_count = max(1, int(len(remaining) * test_ratio))
            test_samples = remaining[:test_count]
            train_samples = remaining[test_count:]
            logger.info("Split test: %d samples from remaining train", len(test_samples))

    return train_samples, val_samples, test_samples


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


# --------------------------------------------------------------------------- #
#  Answer parsing                                                               #
# --------------------------------------------------------------------------- #

def parse_answer_letter(text: str) -> str:
    """Parse a model's text output and extract the answer letter(s).

    For BioTriplex-QA, outputs are like "l)", "a)", "j), o)" etc.
    This function extracts the letter(s) from the text.

    Args:
        text: Raw model output text.

    Returns:
        The parsed answer letters (e.g., "l)" or "j), o)").
    """
    if not text:
        return ""

    text = str(text).strip()

    # Handle NER format: "Entities: (hyperuricemia, DISEASE)"
    if text.startswith("Entities:"):
        return text

    # Handle multiple choice format: look for patterns like "a)", "j), o)"
    import re
    # Match patterns like "a)", "ab)", "j), o)", "a), b), c)"
    matches = re.findall(r'[a-z]+\)', text)
    if matches:
        return ", ".join(matches)

    # Fallback: try to find any letter followed by )
    matches = re.findall(r'\b([a-z])\)', text, re.IGNORECASE)
    if matches:
        return ", ".join([f"{m})" for m in matches])

    return text


# --------------------------------------------------------------------------- #
#  Convenience alias (used by finetune.py)                                    #
# --------------------------------------------------------------------------- #

def load_dataset(data_dir: str, train_ratio: float = 0.9, seed: int = 42):
    """Alias for load_biotriplex_dataset for backwards compatibility."""
    return load_biotriplex_dataset(data_dir, train_ratio, seed)
