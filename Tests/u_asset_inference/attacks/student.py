#!/usr/bin/env python3
"""Student trunk (M shard + LoRA) + linear head for U1c/U1d attacks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/root/CipherForge/CipherForge-ClinVar/three_party")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common import MODEL_PATH  # noqa: E402


def build_student(cfg, device="cuda"):
    from shared.model.model_splitting import (
        detect_model_spec,
        load_m_submodel_with_lora,
    )

    spec = detect_model_spec(MODEL_PATH, u_layers=int(cfg.get("u_layers", 11)))
    model = load_m_submodel_with_lora(
        spec=spec, model_path=MODEL_PATH, device=device,
        lora_rank=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        use_flash_attention=False,
        use_sage_attention=False,
        gradient_checkpointing_style="none",
    )
    for name, p in model.named_parameters():
        if "lora" not in name:
            p.requires_grad_(False)
    # 加载器强制开启逐层 checkpoint；关闭它，否则输入不 requires_grad 时
    # checkpoint 段内的 LoRA 梯度恒为 None（学生学不到东西）。
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    if hasattr(model, "layers"):
        for layer in model.layers:
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = False
    C = 2
    H = int(cfg.get("hidden_dim", 2048))
    head = nn.Parameter(torch.zeros(C, H, dtype=torch.bfloat16, device=device))
    return model, head


def student_z(model, head, h_u, positions):
    """h_u: (B,S,H) float32 cuda; positions: (n,) -> (n,C) logits."""
    dtype = next(model.parameters()).dtype  # bf16 (base weights)
    h_u = h_u.to(dtype)
    if model.training and not h_u.requires_grad:
        h_u.requires_grad_(True)
    H_M = model(h_u)
    h = H_M.reshape(-1, H_M.shape[-1])[positions]  # (n,H)
    return (h @ head.t()).float()


def freeze_lora_state_dict(model):
    """Snapshot of the student's LoRA parameters (composite BA per layer)."""
    comp = {}
    names = {}
    for name, p in model.named_parameters():
        if "lora" not in name:
            continue
        names.setdefault(name, p.detach().cpu().clone())
    for name, t in names.items():
        if name.endswith("lora_A"):
            continue
        if name.endswith("lora_B"):
            base = name[: -len("lora_B")]
            A = names.get(base + "lora_A")
            if A is not None:
                comp[base] = (t @ A).detach().cpu().float()
    return comp


def composite_vector(comp) -> torch.Tensor:
    parts = [comp[k].reshape(-1) for k in sorted(comp)]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)
