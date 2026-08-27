#!/usr/bin/env python3
"""Shared helpers for the T1 U-asset-inference experiment (runs on the server)."""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

THREE_PARTY = Path("/root/CipherForge/CipherForge-ClinVar/three_party")
if str(THREE_PARTY) not in sys.path:
    sys.path.insert(0, str(THREE_PARTY))

import numpy as np  # noqa: E402

from shared.remote_protocol import RemoteClient, _unb64  # noqa: E402

MODEL_PATH = os.environ.get(
    "CF_MODEL_PATH",
    "/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
    "snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
)


def _expand_env(s: str) -> str:
    def _sub(m):
        name, _, default = m.group(1).partition(":-")
        return os.environ.get(name, default)
    return re.sub(r"\$\{([^}]+)\}", _sub, s)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    out = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            v = _expand_env(v)
        elif isinstance(v, dict):
            v = {kk: (_expand_env(vv) if isinstance(vv, str) else vv) for kk, vv in v.items()}
        out[k] = v
    return out


def build_worker_config(cfg: Dict[str, Any], pir_mode: str) -> Dict[str, Any]:
    """Replicate coordinator/main.py worker-config construction."""
    from transformers import AutoTokenizer
    from coordinator.task_profiles import get_profile

    task_type = cfg.get("task_type", "clinvar")
    profile = get_profile(task_type)
    class_outputs = list(cfg.get("class_outputs") or profile["class_outputs"])
    eval_mode = cfg.get("eval_mode") or profile["eval_mode"]
    answer_prefix = profile.get("answer_prefix", " ")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    class_token_ids = []
    for out in class_outputs:
        ids = tokenizer(answer_prefix + out, add_special_tokens=False).input_ids
        if len(ids) < 2:
            raise RuntimeError(f"class output {out!r} tokenizes into <2 tokens: {ids}")
        class_token_ids.append(int(ids[1]))
    if len(set(class_token_ids)) != len(class_token_ids):
        raise RuntimeError(f"class answer tokens not unique: {class_token_ids}")
    answer_token_to_class = {str(tok): i for i, tok in enumerate(class_token_ids)}
    yes_id = class_token_ids[0] if eval_mode == "binary" else -1
    no_id = class_token_ids[1] if eval_mode == "binary" else -1
    wc = {
        "vocab_size": cfg["vocab_size"], "hidden_dim": cfg["hidden_dim"],
        "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
        "scale": cfg["scale"], "lam": cfg["lam"],
        "pir_block_size": cfg["pir_block_size"],
        "debug_grad": cfg.get("debug_grad", False),
        "yes_token_id": yes_id, "no_token_id": no_id,
        "task_type": task_type, "eval_mode": eval_mode,
        "class_outputs": class_outputs,
        "class_token_ids": class_token_ids,
        "answer_token_to_class": answer_token_to_class,
        "u_layers": cfg["u_layers"], "m_layers": cfg["m_layers"],
        "lora_r": cfg["lora_rank"], "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "gradient_clip_norm": 1.0, "warmup_steps": cfg["warmup_steps"],
        "lr_scheduler": "cosine_with_warmup",
        "batch_size": cfg["batch_size"],
        "max_epochs": cfg["max_epochs"],
        "max_seq_length": cfg["max_seq_length"],
        "n_train_samples": 0,
        "N_CRYPTO_U_WORKERS": cfg.get("N_CRYPTO_U_WORKERS", 1),
        "N_CRYPTO_M_WORKERS": cfg.get("N_CRYPTO_M_WORKERS", 1),
        "N_CRYPTO_S_WORKERS": cfg.get("N_CRYPTO_S_WORKERS", 1),
        "pir_dummy_weights": [],
        "pir_fake_ratio": cfg.get("pir_fake_ratio", 0.0),
        "http_timeout_s": cfg.get("http_timeout_s", 300),
        "pir_mode": pir_mode,
        "dp_enable": cfg.get("dp_enable", False),
        "dp_alpha": cfg.get("dp_alpha", 0.15),
        "dp_eta0": cfg.get("dp_eta0"),
        "dp_answer_beta": cfg.get("dp_answer_beta", 0.5),
        "dp_calibration_steps": cfg.get("dp_calibration_steps", 1),
        "dp_calibration_mode": cfg.get("dp_calibration_mode", False),
        "dp_num_classes": cfg.get("dp_num_classes", profile["dp_num_classes"]),
        "dp_clip_value": cfg.get("dp_clip_value"),
    }
    return wc


def init_nodes(cfg, pir_mode: str, run_dir: Path, trace: str = "t1_uai"):
    """Replicate the coordinator INIT handshake. Returns (clients, wc, u_info)."""
    u_url, m_url, s_url = cfg["nodes"]["U"], cfg["nodes"]["M"], cfg["nodes"]["S"]
    _to = float(cfg.get("http_timeout_s", 300))
    u_cli = RemoteClient(u_url, timeout=_to)
    m_cli = RemoteClient(m_url, timeout=_to)
    s_cli = RemoteClient(s_url, timeout=_to)
    for role, cli in (("U", u_cli), ("M", m_cli), ("S", s_cli)):
        h = cli.hello()
        assert h.get("ok") and h.get("role") == role, f"{role} hello failed: {h}"

    wc = build_worker_config(cfg, pir_mode)
    r = m_cli.action(trace, "INIT", 0, "bfv_keygen", {
        "vocab_size": cfg["vocab_size"], "hidden_dim": cfg["hidden_dim"],
        "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
        "scale": cfg["scale"],
    })
    assert r.get("ok"), r
    pk_pem_b64 = r["result"]["pk_pem_b64"]
    prg_seed = os.urandom(32)

    s_cli.action(trace, "INIT", 0, "init_runtime", {**wc, "m_url": m_url})
    r = s_cli.action(trace, "INIT", 0, "build_enc_db", {
        "pk_pem_b64": pk_pem_b64, "m_url": m_url,
    })
    assert r.get("ok"), r
    s_cli.action(trace, "INIT", 0, "pir_prg_setup", {
        "prg_seed_b64": base64.b64encode(prg_seed).decode(),
    })

    rms_params = {
        "rms_partition_size": cfg.get("rms_partition_size", 200),
        "rms_lam": cfg.get("rms_lam", 16),
        "rms_hints_dir": cfg.get("rms_hints_dir")
        or "/root/autodl-tmp/CipherForge-RMS/rms_hints",
        "rms_db_dir": cfg.get("rms_db_dir")
        or "/root/autodl-tmp/CipherForge-RMS/rms_db",
        "rms_db_download_chunk_mb": cfg.get("rms_db_download_chunk_mb", 32),
        "rms_min_coverage": cfg.get("rms_min_coverage"),
        "rms_offline_chunk": cfg.get("rms_offline_chunk", 64),
    }
    if pir_mode == "rms":
        rms_seed = bytes.fromhex(os.environ["CF_RMS_SEED_HEX"]) if os.environ.get("CF_RMS_SEED_HEX") else os.urandom(32)
        rms_params["rms_seed_hex"] = rms_seed.hex()
        (run_dir / "rms_seed.txt").write_text(rms_seed.hex() + "\n")

    r = u_cli.action(trace, "INIT", 0, "init_runtime", {
        "worker_config": wc,
        "pk_pem_b64": pk_pem_b64,
        "prg_seed_b64": base64.b64encode(prg_seed).decode(),
        "m_url": m_url,
        "s_url": s_url,
        "hf_model": cfg["hf_model"],
        "data_dir": cfg["data_dir"],
        "rms": rms_params,
    })
    assert r.get("ok"), r
    u_info = r["result"]
    wc["n_train_samples"] = int(u_info["n_train"])
    m_cli.action(trace, "INIT", 0, "init_runtime", {**wc, "s_url": s_url})
    return u_cli, m_cli, s_cli, wc, u_info


def query_val(m_cli, trace, step, B=4, S=8, H=2048, scale=0.5):
    """Active oracle: crafted H_U -> class logits via M.val_forward."""
    h = (np.random.randn(B, S, H).astype(np.float32) * scale)
    positions = list(range(B * S))
    r = m_cli.action(trace, "EVAL", int(step), "val_forward", {
        "H_U": base64.b64encode(np.ascontiguousarray(h).tobytes()).decode(),
        "H_U_shape": [B, S, H],
        "positions": positions,
    })
    if not r.get("ok"):
        raise RuntimeError(f"val_forward failed: {r.get('error')}")
    z = np.frombuffer(
        _unb64(r["result"]["logits_b64"]), dtype=np.float32,
    ).copy().reshape(*r["result"]["logits_shape"])
    return h, np.asarray(positions, dtype=np.int64), z
