#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task profiles for the CipherForge fine-tuning menu and coordinator.

Each profile describes a supported dataset/task so the RMS-PIR / Block-PIR
pipelines (and the plaintext baseline) can run without task-specific
hardcoding.  Profiles are referenced by ``task_type`` in the run config.
"""
from __future__ import annotations

from typing import Any, Dict

_BIOTRIPLEX_ROOT = "/root/CipherForge/CipherForge-ClinVar/single_process/data"

TASK_PROFILES: Dict[str, Dict[str, Any]] = {
    "clinvar": {
        "label": "ClinVar \u81f4\u75c5\u6027\u4e8c\u5206\u7c7b (Yes/No)",
        "data_dir": "/root/CipherForge/CipherForge-ClinVar/three_party/party_u/data/qa",
        "max_seq_length": 128,
        "dp_num_classes": 2,
        "class_outputs": ["Yes", "No"],
        "eval_mode": "binary",
        "answer_prefix": " ",
    },
    "biotriplex21": {
        "label": "BioTriplex 21\u7c7b\u7ec6\u7c92\u5ea6\u5173\u7cfb\u5206\u7c7b (a)~u))",
        "data_dir": f"{_BIOTRIPLEX_ROOT}/BioTriplex-21",
        "max_seq_length": 2048,
        "dp_num_classes": 21,
        "class_outputs": [f"{chr(ord('a') + i)})" for i in range(21)],
        "eval_mode": "multiclass",
        "answer_prefix": " ",
    },
    "biotriplex7": {
        "label": "BioTriplex 7\u7c7b\u7c97\u7c92\u5ea6\u5173\u7cfb\u5206\u7c7b (a)~g))",
        "data_dir": f"{_BIOTRIPLEX_ROOT}/BioTriplex",
        "max_seq_length": 2048,
        "dp_num_classes": 7,
        "class_outputs": [f"{chr(ord('a') + i)})" for i in range(7)],
        "eval_mode": "multiclass",
        "answer_prefix": " ",
    },
    "biotriplex21obal": {
        "label": "BioTriplex 21\u7c7b\u7ec6\u7c92\u5ea6\u5173\u7cfb\u5206\u7c7b (a)~u)) \u8fc7\u91c7\u6837\u7248",
        "data_dir": f"{_BIOTRIPLEX_ROOT}/BioTriplex-21-obal",
        "max_seq_length": 2048,
        "dp_num_classes": 21,
        "class_outputs": [f"{chr(ord('a') + i)})" for i in range(21)],
        "eval_mode": "multiclass",
        "answer_prefix": " ",
    },
    "biotriplex7obal": {
        "label": "BioTriplex 7\u7c7b\u7c97\u7c92\u5ea6\u5173\u7cfb\u5206\u7c7b (a)~g)) \u8fc7\u91c7\u6837\u7248",
        "data_dir": f"{_BIOTRIPLEX_ROOT}/BioTriplex-obal",
        "max_seq_length": 2048,
        "dp_num_classes": 7,
        "class_outputs": [f"{chr(ord('a') + i)})" for i in range(7)],
        "eval_mode": "multiclass",
        "answer_prefix": " ",
    },
}

DEFAULT_TASK = "clinvar"


def get_profile(task_type: str) -> Dict[str, Any]:
    """Return the profile dict for *task_type* (falls back to the default)."""
    return TASK_PROFILES.get(task_type, TASK_PROFILES[DEFAULT_TASK])
