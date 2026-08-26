"""BioTriplex dataset adapters for SLG-HE-PIR.

These classes wrap the BioTriplex pre-processed text+sentences+ner document
format used by the original llama-rec repo and expose them in the same
dict-based interface expected by ``src.training.trainer.Trainer``.

Two tasks are supported (matching ``docs/BIOTRIPLEX_FINETUNE_README.md``):

* **Task A — Classification (GenRel QA)**:
  ``BioTriplexQADatasetClassification`` reads ``train/val/test_para.txt``,
  one JSON document per line. Each sentence that contains at least one
  (gene, disease) pair is expanded into a prompt asking the model to
  predict a single relation letter (a) ~ g)). The label is a 7-bit
  multi-label binary vector (one bit per ``GENERAL_RELATIONS`` class).

* **Task B — Generation (NER JSON)**:
  ``BioTriplexQADatasetGeneration`` reads ``train/val/test_shorter.txt``,
  one JSON sentence per line. Each sample asks the model to output a
  JSON list ``[{"span": "...", "entity_type": "GENE|DISEASE|RELATION"}, ...]``.

Both datasets emit the same dict schema required by the heterogeneous
protocol:

    {
        "input_ids":       Tensor[max_length],
        "attention_mask":  Tensor[max_length],
        "labels":          Tensor[max_length],    # same as input_ids, prompt tokens are masked at -100 by trainer
        "output_ids":      Tensor[max_length] or None,  # tokenized gold output (used for CE loss)
        "output_text":     str,                   # gold text (e.g. "l)" or JSON string)
        "doc_key":         str,                   # unique sample id
        "prompt":          str,                   # raw prompt text (for inference / debugging)
        "input_text":      str,                   # raw input (sentence) text
        "entities":        list,                  # task A: [[start,end,type], ...]   task B: gold entities
        "relation":        dict | None,           # task A only: {gene, disease, relation}
    }

The dataset classes also auto-write the gold JSONL files expected by
``baseline/classification_genrel/scripts/evaluate_metrics.py`` and
``baseline/generation_ner/scripts/evaluate_metrics.py`` so that the same
evaluator code paths can be reused:

* ``{data_path}/test_gold_general_qa.txt``   (task A)
* ``{data_path}/{split}_gold_ner.txt``        (task B)

The prompt format follows ``baseline/llama-rec/src/llama_recipes/datasets``
exactly:

* **Prompt prefix** = Llama-3 chat template with system prompt and few-shot examples
* **Prompt input** = "### Instruction:\n{INSTRUCTION}\n### Input:\n{input_text}"
* **Prompt suffix** = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
* **Gold output** = "\n### Response:\n{output}"
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  BioTriplex constants (mirrored from baseline/llama-rec/...)
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

# Inverse: general relation -> 0-based index
GENERAL_REL_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(GENERAL_RELATIONS)}

OPTION_LETTERS: List[str] = ["a", "b", "c", "d", "e", "f", "g"]

# Fine-grained relation -> coarse general relation mapping. Used to lift the
# baseline's fine-grained label into the 7-class coarse label space.
# Mirrors baseline/llama-rec/.../biotriplex_qakshot_dataset.py::GENERAL_REL
FINE_TO_GENERAL: Dict[str, str] = {
    # pathological
    "pathological role": "pathological",
    "causative activation": "pathological",
    "causative inhibition": "pathological",
    "causative mutation": "pathological",
    "associated mutation": "pathological",
    # modulatory
    "modulator decrease disease": "modulatory",
    "modulator increase disease": "modulatory",
    "genetic susceptibility": "modulatory",
    # expression change
    "increased expression": "expression change",
    "decreased expression": "expression change",
    "dysregulation": "expression change",
    # diagnosis
    "biomarker": "diagnosis",
    "diagnostic tool": "diagnosis",
    "epigenetic marker": "diagnosis",
    "prognostic indicator": "diagnosis",
    "positive prognostic marker": "diagnosis",
    "negative prognostic marker": "diagnosis",
    # therapy
    "therapy resistance": "therapy",
    "therapeutic target": "therapy",
    # negative / undefined (pass-through)
    "no relation": "no relation",
    "relation undefined": "relation undefined",
}

# NER entity types (task B)
ENTITY_TYPES: List[str] = ["GENE", "DISEASE", "RELATION"]

# NER system prompt + instruction (verbatim from baseline dataset)
NER_SYS_PROMPT = (
    'You are a helpful assistant that extracts the list of entities in the '
    'form of {"span":<entity_text>, "entity_type":<"GENE"|"DISEASE"|"RELATION">} '
    'json entries. If no triplets are found, please provide an empty list. '
)

NER_INSTRUCTION = (
    "**Extract Named Entities**: Identify and extract three types of entities "
    "in the same sequence as they appear in the text. The entity types are:\n"
    "  - **Gene**: A human gene name, symbol (e.g., *SLC02A1*, *PCSK5*) or synonym.\n"
    "  - **Human Disease**: A specific human disease or disorder name (e.g., "
    "*lung adenocarcinoma*, *coronary artery disease*).\n"
    "  - **Relation**: The relationship between the gene and the human disease "
    "(e.g., *associated with*, *causes*, *inhibits*).\n"
)


def _make_general_options() -> str:
    """Build the 'a) pathological, b) modulatory, ..., or g) relation undefined' string."""
    parts = [f"{chr(ord('a') + i)}) {r}" for i, r in enumerate(GENERAL_RELATIONS)]
    if len(parts) <= 1:
        return ", ".join(parts)
    head = ", ".join(parts[:-1])
    return head + ", or " + parts[-1]


GENERAL_OPTIONS: str = _make_general_options()


def _build_general_instruction(gene: str, disease: str) -> str:
    """Build the prompt instruction for task A.

    Mirrors ``biotriplex_qakshot_dataset::INSTRUCTION`` with
    ``general_relations=True, group_relations=False``.
    """
    relation_options = "\n".join(
        f" {chr(ord('a') + i)}) {r}" for i, r in enumerate(GENERAL_RELATIONS)
    )
    return (
        f"What is the relation between the gene {gene} and the disease {disease}?"
        f"{relation_options}"
        f"\nPlease select the correct option by answering {GENERAL_OPTIONS} and nothing else."
    )


def _gen_qa_sys_prompt() -> str:
    """System prompt for task A.

    Mirrors ``biotriplex_qakshot_dataset::SYS_PROMPT`` with
    ``general_relations=True, group_relations=False``.
    """
    return (
        f"You are a helpful assistant that answers questions about the relation "
        f"between genes and diseases by answering {GENERAL_OPTIONS} and nothing else."
    )


def _build_qa_fewshot_block(num_shots: int) -> str:
    """A minimal zero-shot-friendly few-shot block.

    We always return an empty block for ``num_shots=0`` (the README default);
    this function is kept here for future extension.
    """
    return ""


def _qa_input_to_prompt(
    input_text: str,
    triplet: Dict[str, str],
    general_relations: bool = True,
    num_shots: int = 0,
) -> Tuple[str, str, str]:
    """Build the (prefix, body, suffix) triple for task A.

    Returns the components of the prompt so the trainer can mask the prompt
    portion when computing the loss.
    """
    if not general_relations:
        # The README mandates general_relations=True; the fallback path is
        # only here for safety and matches baseline behaviour.
        raise NotImplementedError(
            "Only general_relations=True is supported in this SLG-HE-PIR "
            "BioTriplex integration (matches the paper)."
        )
    instr = _build_general_instruction(triplet["gene"], triplet["disease"])
    sys_prompt = _gen_qa_sys_prompt()
    fewshot_prompts = _build_qa_fewshot_block(num_shots)
    prompt_prefix = (
        f"<|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>"
        f"{fewshot_prompts}<|start_header_id|>user<|end_header_id|>"
        f"### Instruction:\n{instr}\n### Input:\n"
    )
    prompt_input = input_text + "\n\n"
    prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt_prefix, prompt_input, prompt_suffix


def _ner_input_to_prompt(input_text: str) -> Tuple[str, str, str]:
    """Build the (prefix, body, suffix) triple for task B (NER)."""
    prompt_prefix = (
        f"<|start_header_id|>system<|end_header_id|>{NER_SYS_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>"
        f"### Instruction:\n{NER_INSTRUCTION}\n### Input:\n"
    )
    prompt_input = input_text + "\n\n"
    prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt_prefix, prompt_input, prompt_suffix


# ---------------------------------------------------------------------------
#  Helpers shared between the two datasets
# ---------------------------------------------------------------------------
def _strip_leading_whitespace(sentence: str) -> Tuple[str, int]:
    stripped = sentence.lstrip()
    return stripped, len(sentence) - len(stripped)


def _correct_entity_char_index(
    entities: List[List[Any]],
    sentences: List[str],
    sentence_idx: int,
    num_leading_spaces: int,
    stripped_sentence: str,
) -> List[List[int]]:
    """Convert document-level entity char offsets to per-sentence char offsets.

    Mirrors ``BioTriplexNERDataset.correct_entity_char_index``. We omit the
    overlap resolution logic because the BioTriplex pre-processed data is
    expected to be clean (the baseline dataset's ``correct_overlap`` is a
    defensive layer against raw data corruption).
    """
    offset = sum(len(s) for s in sentences[:sentence_idx]) + num_leading_spaces
    corrected: List[List[int]] = []
    for entity in entities:
        # Entity may be [start, end, type] or [start_list, end_list, type]
        if isinstance(entity[0], list):
            for s, e in zip(entity[0], entity[1]):
                corrected.append([int(s) - offset, int(e) - offset, entity[2]])
        else:
            corrected.append([int(entity[0]) - offset, int(entity[1]) - offset, entity[2]])
    # Drop trailing whitespace on `end` indices so they match tokenizer offsets.
    for ent in corrected:
        while ent[1] > ent[0] and stripped_sentence[ent[1] - 1].isspace():
            ent[1] -= 1
    return corrected


def _correct_relation_char_index(
    raw_relations: List[Any],
    sentences: List[str],
    sentence_idx: int,
    num_leading_spaces: int,
    stripped_sentence: str,
    entities: List[List[int]],
    return_neg_relations: bool,
) -> List[Dict[str, Any]]:
    """Adjust each relation's gene/disease char offsets to per-sentence indices.

    Mirrors ``BioTriplexQADataset.correct_relation_char_index``.

    The on-disk relation schema (per the BioTriplex pre-processed data) is a
    list of 5-tuples::

        [gene_start, gene_end, disease_start, disease_end, relation_str]

    For multi-token entities, ``gene_start`` / ``disease_start`` may themselves
    be lists; in that case the relation is exploded into one entry per index
    combination (same as the baseline).
    """
    offset = sum(len(s) for s in sentences[:sentence_idx]) + num_leading_spaces
    corrected: List[List[Any]] = []
    for rel in raw_relations:
        # rel is a list of length 5; gene and disease offsets may be lists.
        if not isinstance(rel, (list, tuple)) or len(rel) != 5:
            continue
        gene_s, gene_e, dis_s, dis_e, relation = rel
        relation = str(relation).strip() if relation else ""

        if isinstance(gene_s, list):
            gene_iter = zip(gene_s, gene_e)
        else:
            gene_iter = [(gene_s, gene_e)]
        if isinstance(dis_s, list):
            dis_iter = zip(dis_s, dis_e)
        else:
            dis_iter = [(dis_s, dis_e)]

        # Cartesian product of gene offsets × disease offsets
        for g_start, g_end in gene_iter:
            for d_start, d_end in dis_iter:
                corrected.append(
                    [
                        int(g_start) - offset,
                        int(g_end) - offset,
                        int(d_start) - offset,
                        int(d_end) - offset,
                        relation,
                    ]
                )

    # Deduplicate (preserve order)
    seen = set()
    deduped: List[List[Any]] = []
    for r in corrected:
        key = tuple(r)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    corrected = deduped
    corrected.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    # Strip trailing whitespace on offsets so they match tokenizer offsets.
    for r in corrected:
        for idx in (1, 3):
            while r[idx] > r[idx - 1] and stripped_sentence[r[idx] - 1].isspace():
                r[idx] -= 1

    # Lift to {gene, disease, relation} dicts (skip empty strings).
    out: List[Dict[str, Any]] = []
    for r in corrected:
        gene_text = stripped_sentence[r[0] : r[1]].strip()
        disease_text = stripped_sentence[r[2] : r[3]].strip()
        if not gene_text or not disease_text:
            continue
        rel_label = r[4]
        if not return_neg_relations and rel_label.lower() in ("no relation", "relation undefined"):
            continue
        out.append(
            {
                "gene": gene_text,
                "disease": disease_text,
                "relation": rel_label,
            }
        )
    return out


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, one document per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tokenize_with_mask(
    tokenizer,
    prefix: str,
    body: str,
    suffix: str,
    gold_response: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Tokenize a (prefix, body, suffix, gold_response) and build labels with -100 masking.

    The trainer expects:
      * ``input_ids``: full sequence (prompt + response) truncated to ``max_length``
      * ``labels``: same shape; prompt portion is -100, response portion is the
        real token id.

    We concatenate using the tokenizer's plain ``encode`` so we can recover the
    boundary between prompt and response (mirrors baseline ``__getitem__``).
    """
    prompt_ids = tokenizer.encode(prefix + body + suffix, add_special_tokens=True)
    response_text = "\n### Response:\n" + gold_response
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    response_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + list(response_ids)

    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


# ---------------------------------------------------------------------------
#  Task A — Classification (GenRel QA)
# ---------------------------------------------------------------------------
class BioTriplexQADatasetClassification(Dataset):
    """Task A — GenRel 7-class relation QA.

    Reads ``{train,val,test}_para.txt`` (paragraph-level) and emits one sample
    per (sentence, relation) tuple — i.e. multiple samples per document if the
    document has multiple (gene, disease, relation) triplets.

    Args:
        data_dir: path to ``datasets/botriplex/Preprocessed BioTriplex/``
            (the directory containing ``train_para.txt`` etc.)
        tokenizer: HF tokenizer
        split: one of {"train", "val", "test"}
        max_length: max sequence length for tokenization
        return_neg_relations: whether to keep samples whose relation is
            "no relation" or "relation undefined" (matches baseline
            ``return_neg_relations`` flag, default False per README)
        general_relations: if True (the only supported mode), use the
            7-class coarse label space.
        write_gold: if True, write ``{data_dir}/test_gold_general_qa.txt`` so
            the baseline evaluator can consume it.
        seed: random seed for any shuffling (currently unused)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        split: str = "train",
        max_length: int = 4096,
        return_neg_relations: bool = False,
        general_relations: bool = True,
        write_gold: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not general_relations:
            raise ValueError(
                "general_relations=False is not supported; the README mandates the 7-class coarse label."
            )

        self.data_dir = data_dir.rstrip("/") + "/"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.return_neg_relations = return_neg_relations

        file_map = {
            "train": "train_para.txt",
            "val": "val_para.txt",
            "test": "test_para.txt",
        }
        if split not in file_map:
            raise ValueError(f"Invalid split: {split} (expected one of {list(file_map)})")
        docs = _read_jsonl(self.data_dir + file_map[split])

        # Expand each document into one sample per (sentence, relation).
        samples: List[Dict[str, Any]] = []
        for doc in docs:
            sentences = doc.get("sentences", [])
            ners = doc.get("ner", [])
            relations = doc.get("relations", [])
            for sent_idx, sentence in enumerate(sentences):
                stripped_sentence, n_lead = _strip_leading_whitespace(sentence)
                if not stripped_sentence.strip():
                    continue
                ents = _correct_entity_char_index(
                    ners[sent_idx] if sent_idx < len(ners) else [],
                    sentences,
                    sent_idx,
                    n_lead,
                    stripped_sentence,
                )
                rels = _correct_relation_char_index(
                    relations[sent_idx] if sent_idx < len(relations) else [],
                    sentences,
                    sent_idx,
                    n_lead,
                    stripped_sentence,
                    ents,
                    return_neg_relations=return_neg_relations,
                )
                for rel in rels:
                    coarse = FINE_TO_GENERAL.get(rel["relation"].lower().strip())
                    if coarse is None:
                        continue
                    idx = GENERAL_REL_TO_IDX[coarse]
                    samples.append(
                        {
                            "doc_key": (
                                f"{doc['doc_key']}_sentence_{sent_idx}_"
                                f"gene_{rel['gene']}_disease_{rel['disease']}_rel_{coarse.replace(' ', '_')}"
                            ),
                            "input": stripped_sentence,
                            "output": f"{OPTION_LETTERS[idx]})",
                            "relation": rel,
                            "entities": ents,
                            "coarse_relation": coarse,
                            "label_idx": idx,
                        }
                    )
        self.data = samples
        logger.info(
            "BioTriplexQADatasetClassification[%s]: %d samples from %s",
            split,
            len(self.data),
            self.data_dir,
        )

        # Persist gold JSONL so the baseline evaluator can run unchanged.
        if write_gold:
            gold_path = os.path.join(self.data_dir, f"{split}_gold_general_qa.txt")
            with open(gold_path, "w", encoding="utf-8") as f:
                for item in self.data:
                    f.write(json.dumps(item) + "\n")
            logger.info("Wrote gold file → %s", gold_path)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        triplet = {
            "gene": item["relation"]["gene"],
            "disease": item["relation"]["disease"],
            "relation": item["relation"]["relation"],
        }
        prefix, body, suffix = _qa_input_to_prompt(item["input"], triplet)
        input_ids, labels = _tokenize_with_mask(
            self.tokenizer,
            prefix,
            body,
            suffix,
            item["output"],
            max_length=self.max_length,
        )

        # Pad to max_length so the DataLoader can stack them.
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "output_ids": torch.tensor(labels, dtype=torch.long),
            "output_text": item["output"],
            "doc_key": item["doc_key"],
            "prompt": prefix + body + suffix,
            "input_text": item["input"],
            "entities": item["entities"],
            "relation": item["relation"],
            "label_idx": item["label_idx"],
        }


# ---------------------------------------------------------------------------
#  Task B — Generation (NER JSON)
# ---------------------------------------------------------------------------
def _entities_to_json(entities: List[List[int]], sentence: str) -> str:
    """Mirror of ``biotriplex_ner_dataset.entities_to_json``."""
    dicts = []
    for ent in entities:
        dicts.append({"span": sentence[ent[0] : ent[1]], "entity_type": ent[2]})
    return json.dumps(dicts)


class BioTriplexQADatasetGeneration(Dataset):
    """Task B — NER JSON generation.

    Reads ``{train,val,test}_shorter.txt`` (sentence-level) and emits one
    sample per sentence. The gold output is a JSON list
    ``[{"span": "...", "entity_type": "GENE|DISEASE|RELATION"}, ...]``
    (empty list if no entities).

    Args:
        data_dir, tokenizer, split, max_length: same as classification.
        write_gold: if True, write ``{split}_gold_ner.txt`` (mandatory for
            downstream ``evaluate_metrics.py`` consumption).
        seed: random seed for any shuffling.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        split: str = "train",
        max_length: int = 4096,
        write_gold: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir.rstrip("/") + "/"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        file_map = {
            "train": "train_shorter.txt",
            "val": "val_shorter.txt",
            "test": "test_shorter.txt",
        }
        if split not in file_map:
            raise ValueError(f"Invalid split: {split} (expected one of {list(file_map)})")
        docs = _read_jsonl(self.data_dir + file_map[split])

        samples: List[Dict[str, Any]] = []
        for doc in docs:
            sentences = doc.get("sentences", [])
            ners = doc.get("ner", [])
            for sent_idx, sentence in enumerate(sentences):
                stripped_sentence, n_lead = _strip_leading_whitespace(sentence)
                if not stripped_sentence.strip():
                    continue
                ents = _correct_entity_char_index(
                    ners[sent_idx] if sent_idx < len(ners) else [],
                    sentences,
                    sent_idx,
                    n_lead,
                    stripped_sentence,
                )
                output_json = _entities_to_json(ents, stripped_sentence)
                samples.append(
                    {
                        "doc_key": f"{doc['doc_key']}_sentence_{sent_idx}",
                        "input": stripped_sentence,
                        "output": output_json,
                        "entities": ents,
                    }
                )
        self.data = samples
        logger.info(
            "BioTriplexQADatasetGeneration[%s]: %d samples from %s",
            split,
            len(self.data),
            self.data_dir,
        )

        # NER gold file is mandatory for downstream evaluation.
        if write_gold:
            gold_path = os.path.join(self.data_dir, f"{split}_gold_ner.txt")
            with open(gold_path, "w", encoding="utf-8") as f:
                for item in self.data:
                    f.write(json.dumps(item) + "\n")
            logger.info("Wrote gold file → %s", gold_path)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        prefix, body, suffix = _ner_input_to_prompt(item["input"])
        input_ids, labels = _tokenize_with_mask(
            self.tokenizer,
            prefix,
            body,
            suffix,
            item["output"],
            max_length=self.max_length,
        )
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "output_ids": torch.tensor(labels, dtype=torch.long),
            "output_text": item["output"],
            "doc_key": item["doc_key"],
            "prompt": prefix + body + suffix,
            "input_text": item["input"],
            "entities": item["entities"],
            "relation": None,
            "label_idx": -1,
        }


# ---------------------------------------------------------------------------
#  Convenience loader
# ---------------------------------------------------------------------------
def build_biotriplex_dataset(
    task: str,
    data_dir: str,
    tokenizer,
    split: str,
    max_length: int = 4096,
    return_neg_relations: bool = False,
):
    """Factory matching the simple ``dataset_class(split)`` calling convention.

    Args:
        task: "classification" or "generation"
        data_dir: path to ``Preprocessed BioTriplex/``
        tokenizer: HF tokenizer
        split: one of {"train", "val", "test"}
        max_length: max sequence length
        return_neg_relations: classification only — see README.
    """
    task = task.lower()
    if task in ("classification", "genrel", "qa"):
        return BioTriplexQADatasetClassification(
            data_dir=data_dir,
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
            return_neg_relations=return_neg_relations,
            general_relations=True,
        )
    if task in ("generation", "ner", "json"):
        return BioTriplexQADatasetGeneration(
            data_dir=data_dir,
            tokenizer=tokenizer,
            split=split,
            max_length=max_length,
        )
    raise ValueError(f"Unknown task: {task} (expected 'classification' or 'generation')")
