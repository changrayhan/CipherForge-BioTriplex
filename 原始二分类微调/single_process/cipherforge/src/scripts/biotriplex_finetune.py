#!/usr/bin/env python3
"""biotriplex_finetune.py — SLG-HE-PIR BioTriplex three-party fine-tuning.

This script orchestrates the three-stage pipeline for the BioTriplex tasks
on top of SLG-HE-PIR's HeterogeneousProtocol (the privacy-preserving
LoRA fine-tuning runtime).

Tasks
-----
* ``--task_type classification`` — GenRel 7-class QA (6 epochs default).
* ``--task_type generation`` — NER JSON generation (10 epochs default).

Pipeline
--------
Stage 0 — one-time offline prep:
  * Build BFV encrypted lm_head DB (vocab_size × hidden_dim BFV ciphertexts)
  * Build S3PIR hint table
  Both reuse the existing ``src/scripts/build_encrypted_db.py`` and
  ``src/scripts/build_s3pir_hints.py`` drivers.

Stage 1 — three-party LoRA fine-tuning:
  * Load the BioTriplex dataset (``BioTriplexQADatasetClassification`` /
    ``BioTriplexQADatasetGeneration``).
  * Build BFV backend, drop ``sk_M``, then construct
    ``HeterogeneousProtocol``.
  * Run ``Trainer.train()`` for ``max_epochs``.

Stage 2 — evaluation (plaintext forward):
  * Load ``best_checkpoint.pt``.
  * Merge the LoRA adapter into the base model (or save it via
    ``PeftModel.save_pretrained`` and reload it cleanly for evaluation).
  * Run ``evaluate_biotriplex.py`` on the test split; this writes the
    final ``{genrel|ner}_<TS>_evaluate_metrics.json``.

Usage
-----
::

    python src/scripts/biotriplex_finetune.py \\
        --task_type classification \\
        --max_epochs 6 \\
        --data_path /path/to/Preprocessed\\ BioTriplex/ \\
        --output_dir /path/to/baseline/classification_genrel/checkpoints

All hyperparameters default to the values listed in
``docs/BIOTRIPLEX_FINETUNE_README.md``. Pass ``--bf16``, ``--use_fast_kernels``
etc. only when you want to override the README defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

# Bootstrap project root
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.biotriplex_dataset import build_biotriplex_dataset  # noqa: E402
from src.training.trainer import Trainer, TrainerConfig  # noqa: E402

logger = logging.getLogger("biotriplex_finetune")


# --------------------------------------------------------------------------- #
#  CLI defaults aligned with BIOTRIPLEX_FINETUNE_README.md
# --------------------------------------------------------------------------- #
TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "classification": {
        "max_epochs": 6,
        "weight_decay": 0.0,
        "learning_rate": 1e-4,
        "general_relations": True,
        "return_neg_relations": False,
        "upweight_minority_class": False,
        "num_of_shots": 0,
        "task_label": "GenRel QA (Classification)",
    },
    "generation": {
        "max_epochs": 10,
        "weight_decay": 0.2,
        "learning_rate": 1e-4,
        "general_relations": False,
        "return_neg_relations": False,
        "upweight_minority_class": False,
        "num_of_shots": 0,
        "task_label": "NER JSON (Generation)",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="SLG-HE-PIR BioTriplex three-party fine-tuning")
    # --- Task selection ---
    p.add_argument("--task_type", choices=["classification", "generation"], required=True)
    p.add_argument("--stage", choices=["0", "1", "2", "all"], default="all")

    # --- Paths ---
    p.add_argument("--data_path", required=True,
                   help="Path to datasets/botriplex/Preprocessed BioTriplex/")
    p.add_argument("--hf_model", default="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I")
    p.add_argument("--bfv_cache_dir", default="/root/autodl-tmp/slg-bfv-cache")
    p.add_argument("--output_dir", required=True,
                   help="Where to save LoRA checkpoints (also where logs/<TS>/ will live)")
    p.add_argument("--log_dir", default=None,
                   help="If omitted, uses ${output_dir}/logs")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint to evaluate (Stage 2). "
                        "Defaults to ${output_dir}/checkpoints/best_checkpoint.pt")
    p.add_argument("--adapter_dir", default=None,
                   help="Directory to write the PEFT adapter (Stage 2). "
                        "Defaults to ${output_dir}/adapter.")

    # --- BFV / protocol params (rarely overridden) ---
    p.add_argument("--vocab_size", type=int, default=128_256)
    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--poly_degree", type=int, default=4096)
    p.add_argument("--plain_bits", type=int, default=30)
    p.add_argument("--scale", type=int, default=10000)
    p.add_argument("--lam", type=int, default=80)
    p.add_argument("--u_layers", type=int, default=16,
                   help="Number of decoder layers in U shard (first u_layers). Remaining layers go to M. Default: 16 (half split)")
    p.add_argument("--m_layers", type=int, default=16,
                   help="Number of decoder layers in M shard (after u_layers). Default: 16 (half split)")

    # --- Stage 0 ---
    p.add_argument("--skip_db", action="store_true",
                   help="Stage 0: skip building the BFV Enc DB")
    p.add_argument("--skip_hints", action="store_true",
                   help="Stage 0: skip building the S3PIR hints table")

    # --- Stage 1 training params (README-aligned) ---
    p.add_argument("--max_epochs", type=int, default=None,
                   help="If omitted, uses the per-task README default (6 or 10).")
    p.add_argument("--batch_size", type=int, default=1,
                   help="The README mandates batch_size=1 with 'padding' strategy.")
    p.add_argument("--max_seq_length", type=int, default=10000,
                   help="README context_length=10000. Note the SLG-HE-PIR limit is "
                        "typically smaller (128 / 3072 / 10000) — large values will OOM.")
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", default="cosine_with_warmup")

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # --- GPU Memory Optimizations (v2.2) ---
    p.add_argument("--use_flash_attention", type=lambda s: s.lower() == "true",
                   default=True,
                   help="Use FlashAttention2 (O(N) memory). Default: True (recommended for long sequences)")
    p.add_argument("--use_sage_attention", type=lambda s: s.lower() == "true",
                   default=True,
                   help="Use SageAttention2++ (INT8) for reduced memory. Default: True (recommended for RTX 5090)")
    p.add_argument("--gradient_checkpointing_style", choices=["reentrant", "full"],
                   default="reentrant",
                   help="Gradient checkpointing style: reentrant (default, ~50%% memory savings) "
                        "or full (lowest memory, slower)")
    p.add_argument("--use_deepspeed_zero", type=lambda s: s.lower() == "true",
                   default=True,
                   help="Enable DeepSpeed ZeRO for optimizer state partitioning. Default: True (recommended)")
    p.add_argument("--zero_stage", type=int, choices=[1, 2, 3], default=1,
                   help="DeepSpeed ZeRO stage: 1=optimizer states, 2=+gradients, 3=+parameters. Default: 1 (single-GPU recommended)")

    # --- Trainer runtime ---
    p.add_argument("--use_chunked_pipeline", type=lambda s: s.lower() == "true",
                   default=True)
    p.add_argument("--chunk_tokens", type=int, default=4096)
    p.add_argument("--n_crypto_u_workers", type=int, default=8)
    p.add_argument("--n_crypto_m_workers", type=int, default=8)
    p.add_argument("--n_crypto_s_workers", type=int, default=1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=1)
    p.add_argument("--patience", type=int, default=999)
    p.add_argument("--do_test_eval", action="store_true")
    p.add_argument("--dump_attacks", action="store_true")
    p.add_argument("--max_train_steps", type=int, default=None,
                   help="If set, stop training after this many steps (testing/CI).")

    # --- Stage 2 evaluation ---
    p.add_argument("--eval_max_seq_length", type=int, default=4096,
                   help="Stage 2 inference input length (smaller than training context).")
    p.add_argument("--eval_max_samples", type=int, default=-1)
    p.add_argument("--save_metrics", action="store_true", default=True)

    # --- Stage 2 PEFT adapter save options ---
    p.add_argument("--save_peft_adapter", action="store_true", default=True)

    # --- Validation metric ---
    p.add_argument("--val_metric", default="val_entity_micro_f1")

    # --- dχ-privacy (see DP机制-迁移参考.md §3.7 / §4.1) ---
    p.add_argument("--dp_enable", action="store_true",
                   help="Turn on dχ-privacy noise on the U→M cut layer.")
    p.add_argument("--dp_alpha", type=float, default=0.15,
                   help="Relative noise ratio (α). Default: 0.15.")
    p.add_argument("--dp_eta0", type=float, default=None,
                   help="Override η₀. When set, skip calibration.")
    p.add_argument("--dp_clip_value", type=float, default=None,
                   help="Optional L∞ clip on the noise. Default: no clip.")
    p.add_argument("--dp_answer_beta", type=float, default=0.5,
                   help="Multiplier on answer positions. Default: 0.5.")
    p.add_argument("--dp_calibration_steps", type=int, default=1,
                   help="Number of clean batches to observe before locking η₀. Default: 1.")
    p.add_argument("--dp_calibration_mode", action="store_true",
                   help="Run the first N steps in calibration mode (no noise).")
    p.add_argument("--dp_dump_audit", action="store_true",
                   help="Write per-step audit records to log_dir/dp_audit.jsonl.")
    p.add_argument("--dp_num_classes", type=int, default=7,
                   help="Number of coarse classes for the LabelBasedCTI. Default: 7 (BioTriplex GenRel).")

    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str, stage: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"biotriplex_finetune_{stage}_{int(time.time())}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("Logging to %s", log_file)
    return logger


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Helpers for V matrix loading + key serialization (mirrors finetune.py)
# --------------------------------------------------------------------------- #
def _load_V_for_db(model_path: str, vocab_size: int, hidden_dim: int) -> np.ndarray:
    """Load the Llama lm_head weight matrix as a numpy array."""
    import torch as _torch
    from safetensors.torch import load_file
    snap = Path(model_path)
    idx_path = snap / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            index = json.load(f)
        lm_head_files = sorted({
            str(snap / fn) for k, fn in index["weight_map"].items() if "lm_head" in k
        })
    else:
        lm_head_files = sorted(snap.glob("*.safetensors"))
    V = None
    for sf in lm_head_files:
        sd = load_file(str(sf), device="cpu")
        for k, v in sd.items():
            if "lm_head" in k and "weight" in k:
                v_fp = v.float().numpy() if v.dtype != _torch.float32 else v.numpy()
                V = v_fp.astype(np.float64) if V is None else np.concatenate(
                    [V, v_fp.astype(np.float64)], axis=0
                )
        del sd
    if V is None:
        raise FileNotFoundError(f"lm_head.weight not found in {model_path}")
    return V


def _serialize_sk(bfv_backend) -> bytes:
    from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
    return _seal_to_bytes(bfv_backend._secret_key)


def _serialize_pk(bfv_backend) -> bytes:
    import pickle as _pickle
    return _pickle.dumps({"pk_bytes": bfv_backend.public_key_bytes})


# --------------------------------------------------------------------------- #
#  Stage 0
# --------------------------------------------------------------------------- #
def run_stage0(args, logger_: logging.Logger) -> Dict[str, Any]:
    logger_.info("=" * 60)
    logger_.info("STAGE 0: Offline Preparation")
    logger_.info("=" * 60)
    results: Dict[str, Any] = {}

    if not args.skip_db:
        logger_.info("[Stage0] Step 1: Building BFV Encrypted DB ...")
        from src.scripts.build_encrypted_db import build_encrypted_db
        db_result = build_encrypted_db(
            model_path=args.hf_model,
            cache_dir=args.bfv_cache_dir,
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            poly_degree=args.poly_degree,
            plain_bits=args.plain_bits,
            scale=args.scale,
            force=False,
        )
        results["encrypted_db"] = db_result
        size_gb = os.path.getsize(db_result["database_path"]) / 1e9 if os.path.exists(db_result["database_path"]) else 0
        logger_.info("[Stage0] Encrypted DB: %.2f GB in %.1fs", size_gb, db_result.get("total_time_s", 0))

    if not args.skip_hints:
        logger_.info("[Stage0] Step 2: Building S3PIR hints ...")
        from src.scripts.build_s3pir_hints import build_s3pir_hints
        hints_result = build_s3pir_hints(
            cache_dir=args.bfv_cache_dir,
            n_entries=args.vocab_size,
            lam=args.lam,
        )
        results["s3pir_hints"] = hints_result
        logger_.info("[Stage0] S3PIR hints: %s in %.1fs", hints_result.get("hints_dir", ""), hints_result.get("build_time_s", 0))

    return results


# --------------------------------------------------------------------------- #
#  Stage 1 — three-party LoRA fine-tuning
# --------------------------------------------------------------------------- #
def run_stage1(args, logger_: logging.Logger) -> Dict[str, Any]:
    logger_.info("=" * 60)
    logger_.info("STAGE 1: Three-Party LoRA Fine-Tuning (%s)", args.task_type.upper())
    logger_.info("=" * 60)

    set_seed(args.seed)

    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ---- Build BioTriplex dataset ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger_.info("[Stage1] Loading BioTriplex dataset (task=%s) ...", args.task_type)
    train_ds = build_biotriplex_dataset(
        task=args.task_type,
        data_dir=args.data_path,
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_seq_length,
        return_neg_relations=TASK_DEFAULTS[args.task_type]["return_neg_relations"],
    )
    val_ds = build_biotriplex_dataset(
        task=args.task_type,
        data_dir=args.data_path,
        tokenizer=tokenizer,
        split="val",
        max_length=args.max_seq_length,
        return_neg_relations=TASK_DEFAULTS[args.task_type]["return_neg_relations"],
    )
    test_ds = build_biotriplex_dataset(
        task=args.task_type,
        data_dir=args.data_path,
        tokenizer=tokenizer,
        split="test",
        max_length=args.max_seq_length,
        return_neg_relations=TASK_DEFAULTS[args.task_type]["return_neg_relations"],
    )
    logger_.info(
        "[Stage1] Datasets ready: train=%d val=%d test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # ---- BFV backend ----
    # IMPORTANT: When reusing cached encrypted DB (bfv_ct_db_*.bin), we MUST use
    # the SAME public key that was used to encrypt it. Otherwise CryptoMWorker
    # won't be able to decrypt the responses.
    pk_cache_path = os.path.join(args.bfv_cache_dir, "bfv_pk.bin")
    pk_path = pk_cache_path if os.path.exists(pk_cache_path) else None

    logger_.info("[Stage1] Building BFV backend (pk_path=%s)...", pk_path)
    from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
    bfv_backend = BFVPrivSelectV2Backend(
        n_entries=args.vocab_size,
        vec_dim=args.hidden_dim,
        shared_seed=os.urandom(32),
        cache_dir=args.bfv_cache_dir,
        poly_degree=args.poly_degree,
        plain_bits=args.plain_bits,
        scale=args.scale,
        pk_path=pk_path,
        force_new_keys=(pk_path is None),  # Generate new keys only if no cached pk
    )
    V = _load_V_for_db(args.hf_model, args.vocab_size, args.hidden_dim)
    bfv_backend.build_encrypted_database(V, force=False)
    bfv_backend.drop_encrypted_db()  # Workers hold their own copies; free ~16 GB from main process

    # ---- Key extraction before any worker pool forks ----
    sk_pem = _serialize_sk(bfv_backend)
    pk_pem = _serialize_pk(bfv_backend)
    bfv_backend._drop_secret_key()
    logger_.info("[Stage1] sk_M dropped from shared backend — only M will hold it")
    prg_seed = os.urandom(32)

    # ---- Hint table ----
    from src.core.s3pir_hints import HintTable
    hints_dir = os.path.join(args.bfv_cache_dir, "s3pir_hints")
    partition_size = 1 << ((args.vocab_size.bit_length() - 1) // 2)
    if os.path.exists(os.path.join(hints_dir, "hint_table.json")):
        hint_table = HintTable.from_cache_files(hints_dir)
    else:
        hint_table = HintTable(
            n_entries=args.vocab_size,
            partition_size=partition_size,
            lam=args.lam,
            cache_dir=hints_dir,
        )
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()

    # ---- Worker config (passed to HeterogeneousProtocol) ----
    worker_config = {
        "vocab_size": args.vocab_size,
        "hidden_dim": args.hidden_dim,
        "poly_degree": args.poly_degree,
        "plain_bits": args.plain_bits,
        "scale": args.scale,
        "bfv_cache_dir": args.bfv_cache_dir,
        "lam": args.lam,
        "lora_r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate if args.learning_rate is not None else TASK_DEFAULTS[args.task_type]["learning_rate"],
        "weight_decay": args.weight_decay if args.weight_decay is not None else TASK_DEFAULTS[args.task_type]["weight_decay"],
        "gradient_clip_norm": args.gradient_clip_norm,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler": args.lr_scheduler,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs if args.max_epochs is not None else TASK_DEFAULTS[args.task_type]["max_epochs"],
        "n_train_samples": len(train_ds),
        "dump_attack_intermediates": args.dump_attacks,
        "attack_dump_dir": os.path.join(log_dir, "attack_dumps"),
        "hf_model_path": args.hf_model,
        "u_layers": args.u_layers,
        "m_layers": args.m_layers,
        "N_CRYPTO_U_WORKERS": args.n_crypto_u_workers,
        "N_CRYPTO_M_WORKERS": args.n_crypto_m_workers,
        "N_CRYPTO_S_WORKERS": args.n_crypto_s_workers,
        "ENABLE_STEP_PROFILING": True,
        "LOG_DIR": log_dir,
        # --- GPU Memory Optimizations (v2.2) ---
        "use_flash_attention": args.use_flash_attention,
        "use_sage_attention": args.use_sage_attention,
        "gradient_checkpointing_style": args.gradient_checkpointing_style,
        "use_deepspeed_zero": args.use_deepspeed_zero,
        "zero_stage": args.zero_stage,
        # --- dχ-privacy (DP机制-迁移参考.md §4.2) ---
        "dp_enable": args.dp_enable,
        "dp_alpha": args.dp_alpha,
        "dp_eta0": args.dp_eta0,
        "dp_clip_value": args.dp_clip_value,
        "dp_answer_beta": args.dp_answer_beta,
        "dp_calibration_steps": args.dp_calibration_steps,
        "dp_calibration_mode": args.dp_calibration_mode,
        "dp_dump_audit": args.dp_dump_audit,
        "dp_num_classes": args.dp_num_classes,
    }

    # ---- Construct HeterogeneousProtocol ----
    logger_.info("[Stage1] Constructing HeterogeneousProtocol ...")
    from src.parties.heterogeneous_protocol import HeterogeneousProtocol
    protocol = HeterogeneousProtocol(
        u_submodel_path=args.hf_model,
        m_submodel_path=args.hf_model,
        s_lm_head_path=args.hf_model,
        bfv_backend=bfv_backend,
        hint_table=hint_table,
        bfv_sk_pem=sk_pem,
        bfv_pk_pem=pk_pem,
        prg_seed=prg_seed,
        config=worker_config,
    )

    # ---- Save PEFT adapter next to checkpoints (so Stage 2 can load it) ----
    adapter_dir = args.adapter_dir or os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    worker_config["PEFT_ADAPTER_DIR"] = adapter_dir

    # ---- Build TrainerConfig ----
    trainer_cfg = TrainerConfig(
        max_epochs=args.max_epochs if args.max_epochs is not None else TASK_DEFAULTS[args.task_type]["max_epochs"],
        patience=args.patience,
        train_ratio=0.9,
        seed=args.seed,
        val_metric=args.val_metric,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        log_dir=log_dir,
        dump_attacks=args.dump_attacks,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        USE_CHUNKED_PIPELINE=args.use_chunked_pipeline,
        CHUNK_TOKENS=args.chunk_tokens,
        do_test_eval=args.do_test_eval,
        task_type=args.task_type,
        ner_gold_path=(
            os.path.join(args.data_path, "val_gold_ner.txt")
            if args.task_type == "generation" else None
        ),
        # --- dχ-privacy knobs (see DP机制-迁移参考.md §4.5) ---
        dp_enable=args.dp_enable,
        dp_dump_audit=args.dp_dump_audit,
        dp_alpha=args.dp_alpha,
        dp_eta0=args.dp_eta0,
        dp_clip_value=args.dp_clip_value,
        dp_answer_beta=args.dp_answer_beta,
        dp_calibration_steps=args.dp_calibration_steps,
        dp_calibration_mode=args.dp_calibration_mode,
        dp_num_classes=args.dp_num_classes,
    )

    trainer = Trainer(
        config=trainer_cfg,
        ipc_protocol=protocol,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
    )

    logger_.info(
        "[Stage1] Starting training: epochs=%d, batch=%d, lr=%g, wd=%g",
        trainer_cfg.max_epochs, trainer_cfg.batch_size,
        worker_config["learning_rate"], worker_config["weight_decay"],
    )

    # Honour --max_train_steps for quick tests: monkey-patch the trainer's
    # epoch loop to bail out early. This is a CI/testing convenience and is a
    # no-op when --max_train_steps is not passed.
    if args.max_train_steps is not None:
        _patch_trainer_for_max_steps(trainer, int(args.max_train_steps), logger_)
    result = trainer.train()
    logger_.info(
        "[Stage1] Training done: best_metric=%.4f, steps=%d",
        result["best_metric"], result["total_steps"],
    )

    protocol.shutdown()

    # ---- Save PEFT adapter from M's LoRA state ----
    if args.save_peft_adapter:
        save_peft_adapter(protocol, args.hf_model, adapter_dir, logger_)

    return {
        "best_metric": result["best_metric"],
        "total_steps": result["total_steps"],
        "elapsed_s": result["elapsed_s"],
        "adapter_dir": adapter_dir,
        "metrics_path": result.get("metrics_path"),
    }


def save_peft_adapter(protocol, base_model_path: str, adapter_dir: str, logger_: logging.Logger) -> None:
    """Materialise the LoRA weights into a PEFT adapter directory.

    SLG-HE-PIR saves LoRA state in ``gather_checkpoints()['M']['lora_state']``
    as a flat dict (see ``src/training/evaluation.py::_inject_lora_manually``
    for the same key conventions). We rebuild a ``PeftModel`` from the base
    model using ``get_peft_model`` (not ``inject_adapter_in_model``, which
    saves the merged model instead of just the adapter), remap keys so they
    match the live PEFT module paths, then call ``save_pretrained`` to
    produce ``adapter_config.json`` + ``adapter_model.safetensors`` — the
    standard PEFT format expected by ``evaluate_biotriplex.py``.
    """
    import shutil
    from peft import LoraConfig, get_peft_model

    ckpts = protocol.gather_checkpoints()
    lora_state = ckpts.get("M", {}).get("lora_state", {})
    if not lora_state:
        logger_.warning("No lora_state in M checkpoint — skipping PEFT adapter save.")
        return

    # Build a transient base model on CPU, wrap with PEFT using get_peft_model
    # (not inject_adapter_in_model which doesn't register the adapter under
    # the standard name for save_pretrained).
    from transformers import AutoModelForCausalLM
    logger_.info("Building transient base model for adapter export ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    # The M shard uses the *custom* LoRA injection (``_LoRALinear``), so its
    # checkpoint keys look like ``layers.{local}.self_attn.q_proj.lora_A``
    # (local index 0 → global layer u_layers). Remap them onto the PEFT live
    # keys (``base_model.model.model.layers.{global}.self_attn.q_proj.lora_A.default.weight``)
    # so the exported adapter is a valid PEFT adapter for ``evaluate_auprc.py``.
    u_layers = None
    try:
        u_layers = int(getattr(protocol.party_m.model, "u_layers", None))
    except Exception:
        pass
    if u_layers is None:
        try:
            u_layers = int(protocol.party_m.spec.u_layers)
        except Exception:
            u_layers = 0

    live_keys = set(peft_model.state_dict().keys())
    canon_state = {}
    unmatched = []
    for k, v in lora_state.items():
        key = k[:-len(".weight")] if k.endswith(".weight") else k
        parts = key.split(".")
        if (
            len(parts) >= 4
            and parts[0] == "layers"
            and parts[1].isdigit()
            and parts[-1] in ("lora_A", "lora_B")
        ):
            local_idx = int(parts[1])
            global_idx = local_idx + u_layers
            proj_path = ".".join(parts[2:-1])
            cand = (
                f"base_model.model.model.layers.{global_idx}.{proj_path}."
                f"{parts[-1]}.default.weight"
            )
            if cand in live_keys:
                canon_state[cand] = v
                continue
            # Robust fallback: unique live key containing the same suffix.
            suffix = f"layers.{global_idx}.{proj_path}.{parts[-1]}"
            matches = [lk for lk in live_keys if suffix in lk]
            if matches:
                canon_state[matches[0]] = v
                continue
            unmatched.append(k)
        else:
            # Legacy layouts: try the raw key or the stripped form directly.
            if k in live_keys:
                canon_state[k] = v
            else:
                nk = k.replace(".lora_A.default.", ".lora_A.").replace(
                    ".lora_B.default.", ".lora_B."
                )
                if nk in live_keys:
                    canon_state[nk] = v
                else:
                    unmatched.append(k)

    missing, unexpected = peft_model.load_state_dict(canon_state, strict=False)
    if missing:
        missing_lora = [m for m in missing if "lora_" in m]
        if missing_lora:
            # With u_layers=11 on the M shard, the 154 missing keys are the
            # U-side layers 0..10 which were never trained (they correctly
            # stay at PEFT's init values). All trained M-side tensors were
            # loaded via canon_state above.
            logger_.info(
                "PEFT adapter: %d LoRA keys left at PEFT init (U-side "
                "layers, intentionally not trained); e.g. %s",
                len(missing_lora), missing_lora[:3],
            )
        logger_.info(
            "PEFT adapter state: loaded %d LoRA tensors (base-weight keys "
            "stay at the from_pretrained values, as expected for frozen base)",
            sum(1 for ck in canon_state if "lora_" in ck),
        )
    if unmatched:
        logger_.warning("Unmapped M LoRA keys: %s ... (%d total)",
                        unmatched[:5], len(unmatched))
    if unexpected:
        logger_.debug("Unexpected LoRA keys: %s ... (%d total)",
                       unexpected[:5], len(unexpected))

    if os.path.exists(adapter_dir):
        shutil.rmtree(adapter_dir)
    os.makedirs(adapter_dir, exist_ok=True)
    peft_model.save_pretrained(adapter_dir)
    logger_.info("PEFT adapter saved → %s", adapter_dir)


# --------------------------------------------------------------------------- #
#  Stage 2 — evaluation (delegates to evaluate_biotriplex.py)
# --------------------------------------------------------------------------- #
def run_stage2(args, logger_: logging.Logger) -> Dict[str, Any]:
    logger_.info("=" * 60)
    logger_.info("STAGE 2: Evaluation (%s)", args.task_type.upper())
    logger_.info("=" * 60)

    adapter_dir = args.adapter_dir or os.path.join(args.output_dir, "adapter")
    if not os.path.exists(adapter_dir) or not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"No PEFT adapter found at {adapter_dir}. Re-run Stage 1 first."
        )

    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Build a subprocess call to evaluate_biotriplex.py
    import subprocess
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "src" / "scripts" / "evaluate_biotriplex.py"),
        "--task_type", args.task_type,
        "--base_model", args.hf_model,
        "--adapter_dir", adapter_dir,
        "--data_path", args.data_path,
        "--split", "test",
        "--output_dir", log_dir,
        "--save_prefix", ("genrel_" if args.task_type == "classification" else "ner_"),
        "--max_seq_length", str(args.eval_max_seq_length),
    ]
    if args.eval_max_samples > 0:
        cmd.extend(["--max_eval_samples", str(args.eval_max_samples)])

    cmd.extend(["--log_file", os.path.join(log_dir, f"evaluate_{int(time.time())}.log")])

    logger_.info("[Stage2] Running: %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage 2 evaluator exited with code {proc.returncode}")
    logger_.info("[Stage2] Done in %.1fs", elapsed)

    # Locate the metrics file the evaluator just wrote
    pattern = (
        "genrel_*_evaluate_metrics.json" if args.task_type == "classification"
        else "ner_*_evaluate_metrics.json"
    )
    import glob
    candidates = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not candidates:
        raise FileNotFoundError(f"No metrics JSON matching {pattern} in {log_dir}")
    metrics_path = candidates[-1]
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {"metrics_path": metrics_path, "metrics": metrics}


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    setup_logging(log_dir, args.stage)
    logger.info("Task: %s, stage: %s", args.task_type, args.stage)

    # Apply per-task defaults if user didn't override
    if args.max_epochs is None:
        args.max_epochs = TASK_DEFAULTS[args.task_type]["max_epochs"]
    if args.learning_rate is None:
        args.learning_rate = TASK_DEFAULTS[args.task_type]["learning_rate"]
    if args.weight_decay is None:
        args.weight_decay = TASK_DEFAULTS[args.task_type]["weight_decay"]

    stages = ["0", "1", "2"] if args.stage == "all" else [args.stage]
    all_results: Dict[str, Any] = {}
    for s in stages:
        t0 = time.time()
        if s == "0":
            all_results["0"] = run_stage0(args, logger)
        elif s == "1":
            all_results["1"] = run_stage1(args, logger)
        elif s == "2":
            all_results["2"] = run_stage2(args, logger)
        logger.info("Stage %s complete in %.1fs", s, time.time() - t0)

    logger.info("=" * 60)
    logger.info("All stages complete!")
    # Print the final metrics to stdout (very useful for shell capture)
    if "2" in all_results and "metrics" in all_results["2"]:
        print()
        print("FINAL METRICS:")
        print(json.dumps(all_results["2"]["metrics"], indent=2))


def _patch_trainer_for_max_steps(trainer, max_steps: int, logger_) -> None:
    """Testing/CI hook: stop trainer.train() after ``max_steps`` global steps.

    Implemented by wrapping ``trainer._run_epoch`` so that it returns after
    ``max_steps`` accumulations. Original behavior is preserved when this
    function is not called. Subsequent epochs return immediately (the smoke
    run must not keep training past ``max_steps``), and a small validation
    subset is evaluated so the run actually reports real CE/accuracy instead
    of the all-zero defaults.
    """
    from torch.utils.data import DataLoader
    from src.training.trainer import make_string_safe_collate

    def patched(epoch: int):
        if trainer.global_step >= max_steps:
            logger_.info(
                "[max_train_steps] global_step=%d already >= %d — skipping epoch %d",
                trainer.global_step, max_steps, epoch,
            )
            return {
                "train_loss": 0.0,
                "train_steps": 0,
                "avg_step_time_ms": 0.0,
                "avg_gpu_mem_mb": 0.0,
                "elapsed_s": 0.0,
            }

        loader = DataLoader(
            trainer.train_ds,
            batch_size=trainer.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=make_string_safe_collate(),
        )
        total = 0
        loss_sum = 0.0
        ce_sum = 0.0
        ce_count = 0
        t0 = time.time()
        for batch in loader:
            if bool(trainer.config.USE_CHUNKED_PIPELINE):
                result = trainer.ipc.step_train_chunked(
                    batch, trainer.global_step,
                    chunk_tokens=int(trainer.config.CHUNK_TOKENS),
                )
            else:
                result = trainer.ipc.step_train(batch, trainer.global_step)
            trainer.global_step += 1
            total += 1
            loss_sum += float(result.loss)
            loss_ce = getattr(result, "loss_ce", None)
            if loss_ce is not None:
                ce_sum += float(loss_ce)
                ce_count += 1
            logger_.info(
                "[max_train_steps] step=%d loss=%.4f loss_ce=%s step_time_ms=%.1f",
                trainer.global_step, result.loss,
                "%.4f" % float(loss_ce) if loss_ce is not None else "n/a",
                result.step_time_ms,
            )
            if trainer.global_step >= max_steps:
                logger_.info("[max_train_steps] reached %d, stopping", max_steps)
                break

        metrics = {
            "train_loss": loss_sum / max(total, 1),
            "train_steps": total,
            "avg_step_time_ms": (time.time() - t0) * 1000 / max(total, 1),
            "avg_gpu_mem_mb": 0.0,
            "elapsed_s": time.time() - t0,
        }
        if ce_count:
            metrics["train_loss_ce"] = ce_sum / ce_count

        # Smoke validation on a subset: exercises the full val forward path
        # and yields real val_ce_loss / accuracy for the trainer log.
        old_val_ds = trainer.val_ds
        try:
            from torch.utils.data import Subset as _Subset
            n_val = min(len(trainer.val_ds), 512)
            trainer.val_ds = _Subset(trainer.val_ds, list(range(n_val)))
            val_metrics = trainer._run_val_epoch(epoch)
            metrics.update(val_metrics)
        except Exception:
            logger_.exception("[max_train_steps] smoke validation failed")
        finally:
            trainer.val_ds = old_val_ds

        return metrics

    trainer._run_epoch = patched


if __name__ == "__main__":
    main()
