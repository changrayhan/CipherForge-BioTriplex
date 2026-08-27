"""
Stage 0 Step 1: Build encrypted V-matrix database.

This script implements the offline preprocessing phase for BFV-based PIR:
  1. Load lm_head.weight (V matrix) from a HuggingFace model
  2. Create BFVPrivSelectV2Backend to trigger KeyGenerator
  3. Encrypt V rows as BFV ciphertexts
  4. Output bfv_pk.bin, bfv_meta.json, bfv_keys.json

Usage:
    python -m src.scripts.build_encrypted_db \
        --model_path /root/autodl-tmp/hf_cache/Llama-3-1-8B-I \
        --cache_dir /root/autodl-tmp/slg-bfv-cache \
        [--vocab_size 128256] \
        [--hidden_dim 4096] \
        [--poly_degree 4096] \
        [--plain_bits 30] \
        [--scale 10000] \
        [--force]

Output files (in cache_dir):
    bfv_pk.bin           - Serialized BFV public key (bytes)
    bfv_meta.json        - Metadata (n_entries, vec_dim, poly_degree, etc.)
    bfv_keys.json        - Serialized keys metadata (for reference)
    bfv_ct_db_*.bin      - Encrypted V-matrix database (large file, ~1.9 GB)

Reference: docs/SLG_HE_PIR_SYSTEM_DOCUMENTATION.md §3.1.1 (Stage 0 Step 1)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.bfv_privselect_v2_adapter import (
    BFVPrivSelectV2Backend,
    MMAP_MAGIC,
    MMAP_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_V_matrix(model_path: str, vocab_size: int, hidden_dim: int) -> np.ndarray:
    """Load the V matrix (lm_head.weight) from a HuggingFace model.

    Args:
        model_path: Path to the HuggingFace model directory.
        vocab_size: Expected vocabulary size.
        hidden_dim: Expected hidden dimension.

    Returns:
        V matrix as float64 array of shape (vocab_size, hidden_dim).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info("Loading model from %s", model_path)
    t0 = time.time()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()

    # Extract lm_head weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        V = model.lm_head.weight.detach().cpu().numpy()
    elif hasattr(model, "model") and hasattr(model.model, "lm_head") and hasattr(model.model.lm_head, "weight"):
        V = model.model.lm_head.weight.detach().cpu().numpy()
    else:
        raise ValueError(
            f"Could not find lm_head.weight in model. "
            f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}"
        )

    elapsed = time.time() - t0
    logger.info(
        "Loaded V matrix: shape=%s, dtype=%s, loaded in %.1fs",
        V.shape, V.dtype, elapsed,
    )

    # Validate shape
    actual_vocab, actual_hidden = V.shape
    if actual_vocab != vocab_size:
        logger.warning(
            "Vocab size mismatch: V has %d rows, expected %d. Using V's actual size.",
            actual_vocab, vocab_size,
        )
        vocab_size = actual_vocab
    if actual_hidden != hidden_dim:
        logger.warning(
            "Hidden dim mismatch: V has %d cols, expected %d. Using V's actual size.",
            actual_hidden, hidden_dim,
        )
        hidden_dim = actual_hidden

    return V.astype(np.float64), vocab_size, hidden_dim


def build_encrypted_db(
    model_path: str,
    cache_dir: str,
    vocab_size: int = 128256,
    hidden_dim: int = 4096,
    poly_degree: int = 4096,
    plain_bits: int = 30,
    scale: float = 10000.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Stage 0 Step 1: Encrypt V matrix with BFV.

   按照文档 §3.1.1:
      1. 读取 lm_head.weight (V 矩阵)
      2. 创建 BFVPrivSelectV2Backend 触发 KeyGenerator
      3. 对 V 的每一行执行 encrypt
      4. 输出 bfv_pk.bin, bfv_meta.json, bfv_keys.json

    Args:
        model_path: Path to HuggingFace model.
        cache_dir: Directory for output files.
        vocab_size: Expected vocabulary size.
        hidden_dim: Expected hidden dimension.
        poly_degree: BFV polynomial modulus degree.
        plain_bits: BFV plaintext modulus bit width.
        scale: Fixed-point scaling factor.
        force: If True, rebuild even if cache exists.

    Returns:
        dict with build statistics and output paths.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Load V matrix
    V, actual_vocab, actual_hidden = load_V_matrix(model_path, vocab_size, hidden_dim)

    # Generate random shared seed for PRG (in production, this is shared between U and S)
    shared_seed = os.urandom(32)

    # Create BFV backend (triggers KeyGenerator inside)
    logger.info(
        "Initializing BFV context: poly_degree=%d, plain_bits=%d, scale=%.1f",
        poly_degree, plain_bits, scale,
    )
    t0 = time.time()

    backend = BFVPrivSelectV2Backend(
        n_entries=actual_vocab,
        vec_dim=actual_hidden,
        shared_seed=shared_seed,
        scale=scale,
        cache_dir=cache_dir,
        poly_degree=poly_degree,
        plain_bits=plain_bits,
    )

    init_time = time.time() - t0
    logger.info("BFV backend initialized in %.1fs", init_time)

    # Build encrypted database
    t0 = time.time()
    result = backend.build_encrypted_database(V, force=force)
    build_time = time.time() - t0

    # Save public key
    pk_path = os.path.join(cache_dir, "bfv_pk.bin")
    with open(pk_path, "wb") as f:
        f.write(backend.public_key_bytes)
    logger.info("Saved public key to %s (%.1f KB)", pk_path, os.path.getsize(pk_path) / 1e3)

    # Save metadata
    meta = {
        "n_entries": actual_vocab,
        "vec_dim": actual_hidden,
        "poly_degree": poly_degree,
        "plain_bits": plain_bits,
        "scale": scale,
        "pk_path": pk_path,
        "db_cache_path": os.path.join(
            cache_dir,
            f"bfv_ct_db_n{actual_vocab}_d{actual_hidden}_p{poly_degree}.bin"
        ),
        "build_time_s": result.get("build_time_s", build_time),
        "init_time_s": init_time,
        "from_cache": result.get("from_cache", False),
    }
    meta_path = os.path.join(cache_dir, "bfv_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # Save keys metadata (for reference, not for distribution)
    keys_meta = {
        "poly_degree": poly_degree,
        "plain_bits": plain_bits,
        "scale": scale,
        "note": "Secret key is held by CryptoMWorker pool, not saved to disk",
    }
    keys_path = os.path.join(cache_dir, "bfv_keys.json")
    with open(keys_path, "w") as f:
        json.dump(keys_meta, f, indent=2)
    logger.info("Saved keys metadata to %s", keys_path)

    total_time = init_time + result.get("build_time_s", build_time)
    logger.info(
        "=== Stage 0 Step 1 Complete ===\n"
        "  Entries encrypted: %d\n"
        "  Total time: %.1fs\n"
        "  Public key: %s\n"
        "  Database: %s",
        actual_vocab,
        total_time,
        pk_path,
        meta["db_cache_path"],
    )

    return {
        "public_key_path": pk_path,
        "metadata_path": meta_path,
        "database_path": meta["db_cache_path"],
        "n_entries": actual_vocab,
        "from_cache": result.get("from_cache", False),
        "total_time_s": total_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0 Step 1: Build encrypted V-matrix database (BFV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/hf_cache/Llama-3-1-8B-I",
        help="Path to HuggingFace model (default: /root/autodl-tmp/hf_cache/Llama-3-1-8B-I)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/slg-bfv-cache",
        help="Directory for encrypted DB cache (default: /root/autodl-tmp/slg-bfv-cache)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128256,
        help="Vocabulary size (default: 128256)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=4096,
        help="Hidden dimension (default: 4096)",
    )
    parser.add_argument(
        "--poly_degree",
        type=int,
        default=4096,
        help="BFV polynomial modulus degree (default: 4096)",
    )
    parser.add_argument(
        "--plain_bits",
        type=int,
        default=30,
        help="BFV plaintext modulus bit width (default: 30)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10000.0,
        help="Fixed-point scaling factor (default: 10000.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if cache exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = build_encrypted_db(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        poly_degree=args.poly_degree,
        plain_bits=args.plain_bits,
        scale=args.scale,
        force=args.force,
    )

    print("\nResult:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
