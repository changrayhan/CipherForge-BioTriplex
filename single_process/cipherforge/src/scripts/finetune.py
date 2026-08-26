#!/usr/bin/env python3
"""
SLG-HE-PIR v2.0 — Main entry point (heterogeneous runtime).

This entry point wires up the **single-process heterogeneous runtime**:

    Main Process (HeterogeneousProtocol)
    ├── PartyU / PartyM / PartyS   (GPU, single CUDA context)
    ├── CryptoUWorker pool         (CPU, no CUDA)
    ├── CryptoMWorker pool         (CPU, holds sk_M)
    └── CryptoSWorker pool         (CPU, mmap DB + PRG share)

The legacy ``IPCProtocol`` (three-spawn-process) runtime is preserved at
``src.parties.legacy_ipc_stub`` for audit / multi-host preview, but is
**not** invoked here. See ``docs/PROJECT_DOCUMENTATION.md`` for details.

Usage:
  # Stage 0 (one-time offline):
    python finetune.py --stage 0

  # Stage 1 (online training):
    python finetune.py --stage 1

  # Stage 2 (test evaluation):
    python finetune.py --stage 2 --checkpoint checkpoints/best_checkpoint.pt

  # Full pipeline (Stage 0 → 1 → 2):
    python finetune.py --stage all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__version__ = "2.0.0"

logger = logging.getLogger("slg_he_pir")


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Paths
    hf_model: str = "/root/autodl-tmp/hf_cache/Llama-3-1-8B-I"
    bfv_cache_dir: str = "/root/autodl-tmp/slg-bfv-cache"
    data_dir: str = "/root/slg-v2.0/data/biotriplex_qa"
    project_root: str = "/root/autodl-tmp/SLG-HE-PIR"
    checkpoint_dir: str = "/root/autodl-tmp/SLG-HE-PIR/checkpoints"
    log_dir: str = "/root/autodl-tmp/SLG-HE-PIR/logs"

    # Model
    vocab_size: int = 128_256
    hidden_dim: int = 4096
    poly_degree: int = 4096
    plain_bits: int = 30
    u_layers: int = 0    # embeddings only on U
    m_layers: int = 32   # all 32 decoder layers + norm + LoRA on M

    # BFV
    scale: int = 10000

    # S3PIR
    lam: int = 80
    pir_block_size: int = 8      # real+PIR block size (real 1 + dummies)

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training
    batch_size: int = 4           # small for 32-layer M to avoid OOM
    max_seq_length: int = 128     # max sequence length (matches docs §6.1)
    max_epochs: int = 10
    patience: int = 999           # no early stopping by default
    learning_rate: float = 3.5e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    train_ratio: float = 0.9
    warmup_steps: int = 200
    lr_scheduler: str = "cosine_with_warmup"

    # GPU memory fractions (kept for backwards compat; the heterogeneous
    # runtime doesn't use per-process fractions but config-validation
    # tooling still reads them).
    gpu_fraction_u: float = 0.10
    gpu_fraction_m: float = 0.22
    gpu_fraction_s: float = 0.22

    # Validation
    val_metric: str = "val_entity_micro_f1"   # must match Trainer output key
    task_type: str = "classification"          # classification | generation | clinvar
    max_train_steps: int = 0                   # 0 = full epochs; >0 = smoke/CI
    resume: bool = False                       # resume Stage1 from last_checkpoint.pt

    # Flags
    seed: int = 42
    log_freq: int = 10
    save_freq: int = 1
    dump_attacks: bool = False
    do_test_eval: bool = False   # run _run_test_epoch() after training finishes

    # Heterogeneous runtime knobs
    N_CRYPTO_U_WORKERS: int = 8
    N_CRYPTO_M_WORKERS: int = 8
    N_CRYPTO_S_WORKERS: int = 1

    # Pipeline mode
    USE_CHUNKED_PIPELINE: bool = True
    CHUNK_TOKENS: int = 3072


def setup_logging(log_dir: str, stage: int) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"finetune_stage{stage}_{int(time.time())}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("Logging to %s", log_file)


def set_seed(seed: int) -> None:
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Stage 0: Offline preparation
# --------------------------------------------------------------------------- #
def run_stage0(cfg: Config, skip_db: bool = False, skip_hints: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info("STAGE 0: Offline Preparation")
    logger.info("=" * 60)

    results = {}

    if not skip_db:
        logger.info("[Stage0] Step 1: Building BFV encrypted database...")
        from src.scripts.build_encrypted_db import build_encrypted_db

        db_result = build_encrypted_db(
            model_path=cfg.hf_model,
            cache_dir=cfg.bfv_cache_dir,
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.hidden_dim,
            poly_degree=cfg.poly_degree,
            plain_bits=cfg.plain_bits,
            scale=cfg.scale,
            force=False,
        )
        results["encrypted_db"] = db_result
        db_size = os.path.getsize(db_result["database_path"]) / 1e9 if os.path.exists(db_result["database_path"]) else 0
        logger.info("[Stage0] Encrypted DB: %.2f GB in %.1fs",
                    db_size, db_result.get("total_time_s", 0))
    else:
        logger.info("[Stage0] Skipping encrypted DB build (using existing cache)")

    if not skip_hints:
        logger.info("[Stage0] Step 2: Building S3PIR hints...")
        from src.scripts.build_s3pir_hints import build_s3pir_hints

        hints_result = build_s3pir_hints(
            cache_dir=cfg.bfv_cache_dir,
            n_entries=cfg.vocab_size,
            lam=cfg.lam,
        )
        results["s3pir_hints"] = hints_result
        logger.info("[Stage0] S3PIR hints: %s in %.1fs",
                    hints_result.get("hints_dir", ""), hints_result.get("build_time_s", 0))
    else:
        logger.info("[Stage0] Skipping S3PIR hints build (using existing cache)")

    logger.info("[Stage0] Complete!")
    return results


# --------------------------------------------------------------------------- #
#  Stage 1: Online training
# --------------------------------------------------------------------------- #
def run_stage1(cfg: Config) -> dict:
    """Run Stage 1: online training with U/M/S via HeterogeneousProtocol."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Online Training (heterogeneous runtime)")
    logger.info("=" * 60)

    set_seed(cfg.seed)

    # ── Step 1: Load datasets ─────────────────────────────────────────────
    logger.info("[Stage1] Loading datasets...")
    if getattr(cfg, "task_type", "classification") == "clinvar":
        from transformers import AutoTokenizer as _AT
        from src.data.clinvar_dataset import load_clinvar_samples, ClinVarQADataset
        tokenizer = _AT.from_pretrained(cfg.hf_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_samples, val_samples, test_samples = load_clinvar_samples(cfg.data_dir)
        train_ds = ClinVarQADataset(train_samples, tokenizer, max_length=cfg.max_seq_length)
        val_ds = ClinVarQADataset(val_samples, tokenizer, max_length=cfg.max_seq_length)
        test_ds = ClinVarQADataset(test_samples, tokenizer, max_length=cfg.max_seq_length)
    else:
        from src.data.dataset import load_biotriplex_dataset
        train_samples, val_samples, test_samples = load_biotriplex_dataset(
            data_dir=cfg.data_dir,
            train_ratio=cfg.train_ratio,
            seed=cfg.seed,
        )
        from src.data.dataset import LlamaTokenizerWrapper, BioTriplexQADataset
        tokenizer = LlamaTokenizerWrapper(cfg.hf_model, max_length=cfg.max_seq_length)
        train_ds = BioTriplexQADataset(train_samples, tokenizer, max_length=cfg.max_seq_length, task="train")
        val_ds = BioTriplexQADataset(val_samples, tokenizer, max_length=cfg.max_seq_length, task="val")
        test_ds = BioTriplexQADataset(test_samples, tokenizer, max_length=cfg.max_seq_length, task="test")
    logger.info(
        "[Stage1] Datasets: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # ── Step 2: Build BFV backend (S-side, holds encrypted DB) ────────────
    # IMPORTANT: When reusing cached encrypted DB, we MUST use the SAME public key.
    # Otherwise CryptoMWorker won't be able to decrypt the responses.
    pk_cache_path = os.path.join(cfg.bfv_cache_dir, "bfv_pk.bin")
    pk_path = pk_cache_path if os.path.exists(pk_cache_path) else None
    force_new_keys = (pk_path is None)

    logger.info("[Stage1] Initializing BFV backend (pk_path=%s, force_new_keys=%s)...",
                pk_path, force_new_keys)
    from src.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend

    bfv_backend = BFVPrivSelectV2Backend(
        n_entries=cfg.vocab_size,
        vec_dim=cfg.hidden_dim,
        shared_seed=os.urandom(32),
        cache_dir=cfg.bfv_cache_dir,
        poly_degree=cfg.poly_degree,
        plain_bits=cfg.plain_bits,
        scale=cfg.scale,
        pk_path=pk_path,
        force_new_keys=force_new_keys,
    )
    # Reuse the persisted M-side secret key so the DB/pk/sk stay consistent
    # across runs (Stage 0 persists bfv_sk.bin; the backend otherwise
    # generates a fresh sk that cannot decrypt the cached DB).
    sk_cache = os.path.join(cfg.bfv_cache_dir, "bfv_sk.bin")
    if os.path.exists(sk_cache):
        from seal import Decryptor, SecretKey
        sk = SecretKey()
        sk.load(bfv_backend._context, sk_cache)
        bfv_backend._secret_key = sk
        bfv_backend._decryptor = Decryptor(bfv_backend._context, sk)
        logger.info("Loaded persisted sk_M from %s", sk_cache)

    V = _load_V_for_db(cfg)
    bfv_backend.build_encrypted_database(V, force=False)
    # The main process never serves PIR queries; workers hold their own copies.
    bfv_backend.drop_encrypted_db()

    # ── Step 3: Extract keys BEFORE dropping sk_M ─────────────────────────
    sk_pem = _serialize_sk(bfv_backend)
    pk_pem = _serialize_pk(bfv_backend)

    # ── Step 3b: Drop sk_M from backend BEFORE any worker pool forks ───────
    # This is the privacy boundary: only the M-side parties (PartyM and the
    # CryptoMWorker pool) ever get sk_M. The GPU Fusion process structurally
    # cannot decrypt anything because its bfv_backend has had sk_M wiped.
    bfv_backend._drop_secret_key()
    logger.info("[Stage1] sk_M dropped from shared backend — only M will hold it")
    prg_seed = os.urandom(32)

    # ── Step 4: Load hint table ───────────────────────────────────────────
    logger.info("[Stage1] Loading hint table...")
    from src.core.s3pir_hints import HintTable

    hints_dir = os.path.join(cfg.bfv_cache_dir, "s3pir_hints")
    partition_size = 1 << ((cfg.vocab_size.bit_length() - 1) // 2)
    hint_table = HintTable(
        n_entries=cfg.vocab_size,
        partition_size=partition_size,
        lam=cfg.lam,
        cache_dir=hints_dir,
    )
    if os.path.exists(os.path.join(hints_dir, "hint_table.json")):
        hint_table = HintTable.from_cache_files(hints_dir)
    else:
        hint_table.compute_main_hints_skeleton()
        hint_table.compute_backup_hints_skeleton()
        hint_table.to_cache_files()

    # ── Step 5: Build config dict for the runtime ─────────────────────────
    if getattr(cfg, "task_type", "classification") == "clinvar":
        yes_tok = tokenizer("Yes", add_special_tokens=False).input_ids
        no_tok = tokenizer("No", add_special_tokens=False).input_ids
        if len(yes_tok) != 1 or len(no_tok) != 1:
            raise RuntimeError(
                "Yes/No must be single tokens for the monitor/PRG pipeline"
            )
        yes_token_id, no_token_id = yes_tok[0], no_tok[0]
    else:
        yes_token_id, no_token_id = -1, -1
    worker_config = {
        "vocab_size": cfg.vocab_size,
        "hidden_dim": cfg.hidden_dim,
        "poly_degree": cfg.poly_degree,
        "plain_bits": cfg.plain_bits,
        "scale": cfg.scale,
        "bfv_cache_dir": cfg.bfv_cache_dir,
        "lam": cfg.lam,
        "pir_block_size": cfg.pir_block_size,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        # LoRA
        "lora_r": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        # Optimizer / scheduler
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "gradient_clip_norm": cfg.gradient_clip_norm,
        "warmup_steps": cfg.warmup_steps,
        "lr_scheduler": cfg.lr_scheduler,
        # Misc bookkeeping
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "n_train_samples": len(train_ds),
        "dump_attack_intermediates": cfg.dump_attacks,
        "attack_dump_dir": os.path.join(cfg.log_dir, "attack_dumps"),
        "hf_model_path": cfg.hf_model,
        "u_layers": cfg.u_layers,
        "m_layers": cfg.m_layers,
        # Worker pool sizes
        "N_CRYPTO_U_WORKERS": cfg.N_CRYPTO_U_WORKERS,
        "N_CRYPTO_M_WORKERS": cfg.N_CRYPTO_M_WORKERS,
        "N_CRYPTO_S_WORKERS": cfg.N_CRYPTO_S_WORKERS,
        # Profiling / logging
        "ENABLE_STEP_PROFILING": True,
        "LOG_DIR": cfg.log_dir,
    }

    # ── Step 6: Construct HeterogeneousProtocol (the only active runtime) ─
    logger.info(
        "[Stage1] Constructing HeterogeneousProtocol "
        "(single Fusion process + CPU Crypto Worker pools) ..."
    )
    from src.parties.heterogeneous_protocol import HeterogeneousProtocol

    protocol = HeterogeneousProtocol(
        u_submodel_path=cfg.hf_model,
        m_submodel_path=cfg.hf_model,
        s_lm_head_path=cfg.hf_model,
        bfv_backend=bfv_backend,
        hint_table=hint_table,
        bfv_sk_pem=sk_pem,
        bfv_pk_pem=pk_pem,
        prg_seed=prg_seed,
        config=worker_config,
    )

    # ── Step 7: Run training loop ─────────────────────────────────────────
    logger.info("[Stage1] Starting training loop...")
    from src.training.trainer import Trainer, TrainerConfig

    trainer_cfg = TrainerConfig(
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        train_ratio=cfg.train_ratio,
        seed=cfg.seed,
        val_metric=cfg.val_metric,
        save_freq=cfg.save_freq,
        log_freq=cfg.log_freq,
        checkpoint_dir=cfg.checkpoint_dir,
        log_dir=cfg.log_dir,
        dump_attacks=cfg.dump_attacks,
        batch_size=cfg.batch_size,                  # ← fix: was not propagated
        max_seq_length=cfg.max_seq_length,
        USE_CHUNKED_PIPELINE=cfg.USE_CHUNKED_PIPELINE,
        CHUNK_TOKENS=cfg.CHUNK_TOKENS,
        do_test_eval=cfg.do_test_eval,
        task_type=cfg.task_type,                    # ← ClinVar val branch needs this
    )

    tokenizer = train_ds.tokenizer
    trainer = Trainer(
        config=trainer_cfg,
        ipc_protocol=protocol,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
    )

    if getattr(cfg, "resume", False):
        resume_ckpt = os.path.join(cfg.checkpoint_dir, "last_checkpoint.pt")
        if os.path.exists(resume_ckpt):
            trainer.resume_from(resume_ckpt)
            logger.info("[Stage1] Resuming from %s", resume_ckpt)
        else:
            logger.warning(
                "[Stage1] --resume requested but %s missing — starting fresh",
                resume_ckpt,
            )

    if cfg.max_train_steps > 0:
        from src.scripts.biotriplex_finetune import _patch_trainer_for_max_steps
        _patch_trainer_for_max_steps(trainer, int(cfg.max_train_steps), logger)
        logger.info("Smoke mode: max_train_steps=%d", cfg.max_train_steps)

    results = trainer.train()
    logger.info("[Stage1] Training complete: best_metric=%.4f, steps=%d",
                results["best_metric"], results["total_steps"])

    protocol.shutdown()

    if getattr(cfg, "task_type", "classification") == "clinvar":
        from src.scripts.biotriplex_finetune import save_peft_adapter as _save_peft
        adapter_dir = os.path.join(cfg.checkpoint_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        _save_peft(protocol, cfg.hf_model, adapter_dir, logger)
        logger.info("[Stage1] PEFT adapter saved -> %s", adapter_dir)
        results["adapter_dir"] = adapter_dir

    return results


# --------------------------------------------------------------------------- #
#  Stage 2: Test evaluation
# --------------------------------------------------------------------------- #
def run_stage2(cfg: Config, checkpoint_path: str) -> dict:
    """Run Stage 2: test evaluation with merged LoRA model."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Test Evaluation")
    logger.info("=" * 60)

    if getattr(cfg, "task_type", "classification") == "clinvar":
        import subprocess
        adapter_dir = os.path.join(cfg.checkpoint_dir, "adapter")
        if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            raise FileNotFoundError(f"No PEFT adapter at {adapter_dir}; run Stage 1 first")
        eval_script = os.path.join(cfg.project_root, "clinvar_plain", "scripts", "evaluate_auprc.py")
        data_path = os.path.join(cfg.data_dir, "test.jsonl")
        out_path = os.path.join(cfg.log_dir, "clinvar_auprc.json")
        cmd = [
            sys.executable, "-s", eval_script,
            "--adapter", adapter_dir,
            "--data", data_path,
            "--out", out_path,
        ]
        logger.info("[Stage2] Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"evaluate_auprc.py exited with {proc.returncode}")
        import json as _json
        with open(out_path) as f:
            metrics = _json.load(f)
        return {"metrics_path": out_path, "metrics": metrics}

    from src.training.evaluation import evaluate_test_set, save_test_results
    from src.data.dataset import load_biotriplex_dataset, LlamaTokenizerWrapper, BioTriplexQADataset

    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint not found: %s", checkpoint_path)
        raise FileNotFoundError(checkpoint_path)

    _, _, test_samples = load_biotriplex_dataset(
        data_dir=cfg.data_dir,
        train_ratio=cfg.train_ratio,
        seed=cfg.seed,
    )
    tokenizer = LlamaTokenizerWrapper(cfg.hf_model, max_length=cfg.max_seq_length)
    test_ds = BioTriplexQADataset(test_samples, tokenizer, max_length=cfg.max_seq_length, task="test")

    results = evaluate_test_set(
        checkpoint_path=checkpoint_path,
        test_ds=test_ds,
        model_path=cfg.hf_model,
        max_new_tokens=128,
    )

    output_path = os.path.join(cfg.log_dir, "test_results.json")
    save_test_results(results, output_path)

    logger.info(
        "[Stage2] Test results: micro_f1=%.4f, precision=%.4f, recall=%.4f",
        results["entity_micro_f1"],
        results["entity_micro_precision"],
        results["entity_micro_recall"],
    )

    return results


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _load_V_for_db(cfg: Config) -> "np.ndarray":
    """Load V matrix for BFV encrypted DB building."""
    import torch
    from safetensors.torch import load_file

    snap_path = cfg.hf_model
    index_path = Path(snap_path) / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        wm = index["weight_map"]
        lm_head_files = sorted({
            str(Path(snap_path) / fn)
            for k, fn in wm.items() if "lm_head" in k
        })
    else:
        lm_head_files = sorted(Path(snap_path).glob("*.safetensors"))

    V = None
    for sf in lm_head_files:
        sd = load_file(str(sf), device="cpu")
        for k, v in sd.items():
            if "lm_head" in k and "weight" in k:
                v_fp = v.float().numpy() if v.dtype != torch.float32 else v.numpy()
                V = v_fp.astype(np.float64) if V is None else np.concatenate(
                    [V, v_fp.astype(np.float64)], axis=0
                )
        del sd

    if V is None:
        raise FileNotFoundError(f"lm_head.weight not found in {snap_path}")
    return V


def _serialize_sk(bfv_backend) -> bytes:
    """Serialize BFV secret key for distribution to M."""
    from src.core.bfv_privselect_v2_adapter import _seal_to_bytes
    return _seal_to_bytes(bfv_backend._secret_key)


def _serialize_pk(bfv_backend) -> bytes:
    """Serialize BFV public key for distribution to workers.

    Crypto workers expect a pickle of {"pk_bytes": raw_seal_bytes}.
    """
    import pickle as _pickle
    return _pickle.dumps({"pk_bytes": bfv_backend.public_key_bytes})


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="SLG-HE-PIR v2.0: Sublinear PIR Private Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stage", type=str, default="all",
                        choices=["0", "1", "2", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (for --stage 2)")
    parser.add_argument("--skip_db", action="store_true",
                        help="Skip encrypted DB build (use existing cache)")
    parser.add_argument("--skip_hints", action="store_true",
                        help="Skip S3PIR hints build (use existing cache)")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume Stage1 from last_checkpoint.pt")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dump_attacks", action="store_true")
    parser.add_argument("--log_freq", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file to override defaults")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override training batch size")
    parser.add_argument("--use_chunked_pipeline", type=lambda s: s.lower() == "true",
                        default=None,
                        help="Enable chunked pipeline path (default: True)")
    parser.add_argument("--chunk_tokens", type=int, default=None,
                        help="Chunk size in tokens for chunked path")
    parser.add_argument("--n_crypto_u_workers", type=int, default=None)
    parser.add_argument("--n_crypto_m_workers", type=int, default=None)
    parser.add_argument("--n_crypto_s_workers", type=int, default=None)
    parser.add_argument("--do_test_eval", action="store_true",
                        help="Run test evaluation after training finishes")
    args = parser.parse_args()

    # Load config
    cfg = Config()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.max_train_steps is not None:
        cfg.max_train_steps = args.max_train_steps
    if args.resume:
        cfg.resume = True
    if args.patience is not None:
        cfg.patience = args.patience
    if args.dump_attacks:
        cfg.dump_attacks = True
    if args.log_freq is not None:
        cfg.log_freq = args.log_freq
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.use_chunked_pipeline is not None:
        cfg.USE_CHUNKED_PIPELINE = args.use_chunked_pipeline
    if args.chunk_tokens is not None:
        cfg.CHUNK_TOKENS = args.chunk_tokens
    if args.n_crypto_u_workers is not None:
        cfg.N_CRYPTO_U_WORKERS = args.n_crypto_u_workers
    if args.n_crypto_m_workers is not None:
        cfg.N_CRYPTO_M_WORKERS = args.n_crypto_m_workers
    if args.n_crypto_s_workers is not None:
        cfg.N_CRYPTO_S_WORKERS = args.n_crypto_s_workers
    if args.do_test_eval:
        cfg.do_test_eval = True

    stage = args.stage
    if stage == "all":
        stages = ["0", "1", "2"]
    else:
        stages = [stage]

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    all_results = {}
    for s in stages:
        setup_logging(cfg.log_dir, s)
        logger.info("SLG-HE-PIR v%s starting stage %s", __version__, s)
        t0 = time.time()
        if s == "0":
            result = run_stage0(cfg, skip_db=args.skip_db, skip_hints=args.skip_hints)
        elif s == "1":
            result = run_stage1(cfg)
        elif s == "2":
            ckpt = args.checkpoint or os.path.join(cfg.checkpoint_dir, "best_checkpoint.pt")
            result = run_stage2(cfg, ckpt)
        all_results[s] = result
        logger.info("Stage %s complete in %.1fs", s, time.time() - t0)

    logger.info("=" * 60)
    logger.info("All stages complete!")
    logger.info("Results: %s", json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
