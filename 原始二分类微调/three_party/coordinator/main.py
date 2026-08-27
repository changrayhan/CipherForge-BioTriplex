#!/usr/bin/env python3
"""Coordinator — independent control plane (paper-faithful TriadFT topology).

The coordinator is NOT U.  It does not load datasets, does not run PartyU,
does not hold (x, y), and never relays protocol payloads.  U, M, and S are
three separate parties that exchange messages directly over HTTP:

    U -> M : H_U (trunk_forward), C_U (grad_reconstruct)
    M -> S : H_M (head_forward / val_head)
    S -> M : s_S (receive_share)
    U <-> S : PIR (fetch_rows / rms_parity / db_download)

The coordinator only:
  * issues high-level control commands (init / train_step / run_eval / ...),
  * collects scalar metrics and logs,
  * saves/loads M's LoRA checkpoints (trusted console),
  * exports the PEFT adapter and aggregates the final eval JSON.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from shared.remote_protocol import (  # noqa: E402
    RemoteClient,
    _tensordict_to_b64,
    _unb64,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("coordinator")


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


class AdapterSource:
    """Minimal protocol-shaped object so save_peft_adapter() can run."""

    def __init__(self, m_cli, trace_id, u_layers: int):
        self._m = m_cli
        self._trace = trace_id
        self.u_layers = int(u_layers)

    def gather_checkpoints(self) -> Dict[str, Any]:
        return {
            "U": {"party": "U"},
            "M": _fetch_m_checkpoint(self._m, self._trace),
            "S": {"party": "S"},
        }


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _fetch_m_checkpoint(m_cli, trace_id) -> Dict[str, Any]:
    """Pull M's LoRA/optimizer/scheduler state (trusted console only)."""
    r = m_cli.action(trace_id, "EVAL", 0, "gather_checkpoint", {})
    if not r.get("ok"):
        raise RuntimeError(f"gather_checkpoint failed: {r.get('error')}")
    res = r["result"]
    lora_state = {
        k: torch.load(io.BytesIO(_unb64(v["b64"])), map_location="cpu", weights_only=False)
        for k, v in (res.get("lora_state_b64") or {}).items()
    }
    optimizer_state = torch.load(
        io.BytesIO(_unb64(res["optimizer_state_b64"])), map_location="cpu",
        weights_only=False,
    ) if res.get("optimizer_state_b64") else {}
    scheduler_state = torch.load(
        io.BytesIO(_unb64(res["scheduler_state_b64"])), map_location="cpu",
        weights_only=False,
    ) if res.get("scheduler_state_b64") else {}
    return {
        "party": "M",
        "lora_state": lora_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
    }


def _save_checkpoint(
    path: str, epoch: int, global_step: int, best_metric, m_ckpt: Dict[str, Any],
    u_spec: Dict[str, Any], config: Dict[str, Any],
) -> None:
    torch.save({
        "epoch": epoch,
        "completed_epochs": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": epoch,
        "party_checkpoints": {
            "U": u_spec or {"party": "U"},
            "M": m_ckpt,
            "S": {"party": "S"},
        },
        "config": config,
    }, path)
    logger.info("checkpoint saved -> %s (epoch=%d step=%d)", path, epoch, global_step)


def _load_checkpoint(m_cli, trace_id, path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m_ckpt = (ckpt.get("party_checkpoints") or {}).get("M") or {}
    lora_state = m_ckpt.get("lora_state") or {}
    opt_state = m_ckpt.get("optimizer_state") or {}
    sch_state = m_ckpt.get("scheduler_state") or {}
    opt_buf = io.BytesIO()
    torch.save(opt_state, opt_buf)
    sch_buf = io.BytesIO()
    torch.save(sch_state, sch_buf)
    r = m_cli.action(trace_id, "EVAL", 0, "load_checkpoint", {
        "lora_state_b64": _tensordict_to_b64(lora_state),
        "optimizer_state_b64": _b64(opt_buf.getvalue()),
        "scheduler_state_b64": _b64(sch_buf.getvalue()),
    })
    if not r.get("ok"):
        raise RuntimeError(f"load_checkpoint failed: {r.get('error')}")
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_train_steps", type=int, default=0)
    ap.add_argument("--log_freq", type=int, default=10)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--pir_mode", default=None,
                    help="'block' (default) or 'rms' (backup RMS-PIR)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_train", action="store_true",
                    help="load checkpoint, export adapter and evaluate, no training")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    task_type = cfg.get("task_type", "clinvar")
    pir_mode = args.pir_mode or cfg.get("pir_mode", "block")
    if pir_mode not in ("block", "rms"):
        raise SystemExit(f"unsupported pir_mode: {pir_mode!r}")
    logger.info("task_type=%s pir_mode=%s", task_type, pir_mode)

    # ---- task profile + token-id mapping (tokenizer only, no datasets) ----
    from transformers import AutoTokenizer
    from coordinator.task_profiles import get_profile

    profile = get_profile(task_type)
    class_outputs = list(cfg.get("class_outputs") or profile["class_outputs"])
    eval_mode = cfg.get("eval_mode") or profile["eval_mode"]
    answer_prefix = profile.get("answer_prefix", " ")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    class_token_ids = []
    for _out in class_outputs:
        _ids = tokenizer(answer_prefix + _out, add_special_tokens=False).input_ids
        if len(_ids) < 2:
            raise RuntimeError(
                f"class output {_out!r} does not tokenize with prefix into >=2 tokens: {_ids}")
        class_token_ids.append(int(_ids[1]))
    if len(set(class_token_ids)) != len(class_token_ids):
        raise RuntimeError(f"class answer tokens are not unique: {class_token_ids}")
    answer_token_to_class = {str(tok): i for i, tok in enumerate(class_token_ids)}
    logger.info(
        "eval_mode=%s classes=%d class_token_ids=%s",
        eval_mode, len(class_token_ids), class_token_ids,
    )

    # ---- nodes ----
    u_url = cfg["nodes"]["U"]
    m_url = cfg["nodes"]["M"]
    s_url = cfg["nodes"]["S"]
    # RMS-PIR 首次初始化时 U 要下载 4.2GB 密文库，必须用配置的超时（默认 300s）
    _to = float(cfg.get("http_timeout_s", 300))
    u_cli, m_cli, s_cli = (
        RemoteClient(u_url, timeout=_to),
        RemoteClient(m_url, timeout=_to),
        RemoteClient(s_url, timeout=_to),
    )
    for role, cli in (("U", u_cli), ("M", m_cli), ("S", s_cli)):
        h = cli.hello()
        assert h.get("ok") and h.get("role") == role, f"{role} hello failed: {h}"
    logger.info("nodes ready: U=%s M=%s S=%s", u_url, m_url, s_url)

    trace_id = "task_%s_%s" % (time.strftime("%Y%m%d%H%M%S"), "3pty")
    yes_id = class_token_ids[0] if eval_mode == "binary" else -1
    no_id = class_token_ids[1] if eval_mode == "binary" else -1

    worker_config = {
        "vocab_size": cfg["vocab_size"], "hidden_dim": cfg["hidden_dim"],
        "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
        "scale": cfg["scale"], "lam": cfg["lam"], "pir_block_size": cfg["pir_block_size"],
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
        "batch_size": args.batch_size or cfg["batch_size"],
        "max_epochs": args.max_epochs or cfg["max_epochs"],
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

    # ---- INIT: M keygen -> S enc-db/seed -> U runtime -> M runtime ----
    r = m_cli.action(trace_id, "INIT", 0, "bfv_keygen", {
        "vocab_size": cfg["vocab_size"], "hidden_dim": cfg["hidden_dim"],
        "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
        "scale": cfg["scale"],
    })
    assert r.get("ok"), r
    pk_pem_b64 = r["result"]["pk_pem_b64"]
    logger.info("M pk_sha256=%s", r["result"].get("pk_sha256"))

    prg_seed = os.urandom(32)
    s_cli.action(trace_id, "INIT", 0, "init_runtime", {**worker_config, "m_url": m_url})
    r = s_cli.action(trace_id, "INIT", 0, "build_enc_db", {
        "pk_pem_b64": pk_pem_b64,
        "m_url": m_url,
    })
    assert r.get("ok"), r
    logger.info("S pk_sha256=%s", r["result"].get("pk_sha256"))
    s_cli.action(trace_id, "INIT", 0, "pir_prg_setup", {
        "prg_seed_b64": base64.b64encode(prg_seed).decode(),
    })
    logger.info("S: 密文库/PRG 就绪")

    rms_seed_hex = ""
    rms_params = {
        "rms_partition_size": cfg.get("rms_partition_size", 200),
        "rms_lam": cfg.get("rms_lam", 16),
        "rms_hints_dir": cfg.get("rms_hints_dir") or "/root/autodl-tmp/CipherForge-RMS/rms_hints",
        "rms_db_dir": cfg.get("rms_db_dir") or "/root/autodl-tmp/CipherForge-RMS/rms_db",
        "rms_db_download_chunk_mb": cfg.get("rms_db_download_chunk_mb", 32),
        "rms_min_coverage": cfg.get("rms_min_coverage"),
        "rms_offline_chunk": cfg.get("rms_offline_chunk", 64),
    }
    if pir_mode == "rms":
        rms_seed = bytes.fromhex(os.environ.get("CF_RMS_SEED_HEX", "")) if os.environ.get("CF_RMS_SEED_HEX") else os.urandom(32)
        rms_seed_hex = rms_seed.hex()
        try:
            (Path(cfg["log_dir"]) / "rms_seed.txt").write_text(rms_seed_hex + "\n")
        except Exception:
            pass
        rms_params["rms_seed_hex"] = rms_seed_hex
        logger.info("RMS seed (hex): %s", rms_seed_hex)

    r = u_cli.action(trace_id, "INIT", 0, "init_runtime", {
        "worker_config": worker_config,
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
    steps_per_epoch = int(u_info["steps_per_epoch"])
    worker_config["n_train_samples"] = int(u_info["n_train"])
    logger.info(
        "U ready: steps/epoch=%d n_train=%d n_val=%d n_test=%d",
        steps_per_epoch, u_info["n_train"], u_info["n_val"], u_info["n_test"],
    )

    m_cli.action(trace_id, "INIT", 0, "init_runtime", {**worker_config, "s_url": s_url})
    logger.info("INIT 完成：U/M/S 就绪，edges U->M / M->S / S->M / U<->S 直连")

    # ---- checkpoint resume ----
    last_ckpt = os.path.join(cfg["checkpoint_dir"], "last_checkpoint.pt")
    start_epoch = 0
    global_step = 0
    best_metric = None
    if args.resume and os.path.exists(last_ckpt):
        ck = _load_checkpoint(m_cli, trace_id, last_ckpt)
        start_epoch = int(ck.get("completed_epochs", 0))
        global_step = int(ck.get("global_step", 0))
        best_metric = ck.get("best_metric")
        logger.info("resumed: epoch=%d step=%d best=%s", start_epoch, global_step, best_metric)

    max_epochs = args.max_epochs or cfg["max_epochs"]
    val_key = "val_ce_loss" if eval_mode == "binary" else "val_accuracy"
    lower_better = eval_mode == "binary"

    if not args.skip_train:
        for epoch in range(start_epoch, int(max_epochs)):
            logger.info("=== epoch %d/%d (global_step=%d) ===", epoch + 1, max_epochs, global_step)
            t_epoch = time.time()
            for step in range(steps_per_epoch):
                t0 = time.time()
                res = u_cli.action(trace_id, "TRAIN", global_step, "train_step", {"step": global_step})
                if not res.get("ok"):
                    raise RuntimeError(f"train_step failed: {res.get('error')}")
                metrics = res.get("metrics") or {}
                result = res.get("result") or {}
                loss = metrics.get("loss", 0.0)
                if args.log_freq and step % args.log_freq == 0:
                    logger.info(
                        "step %d: loss=%.4f g_absmax=%.4g g_meanabs=%.4g ce=%s t=%.0fms",
                        global_step, loss,
                        metrics.get("g_absmax", -1), metrics.get("g_meanabs", -1),
                        result.get("monitor_ce"), (time.time() - t0) * 1000,
                    )
                global_step += 1
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break
            logger.info("epoch %d done in %.1fs (steps so far %d)", epoch + 1, time.time() - t_epoch, global_step)

    # ---- validation (S returns only class logits to U) ----
            max_val_batches = int(os.environ.get("CF_MAX_VAL_BATCHES", "0"))
            r = u_cli.action(trace_id, "EVAL", 0, "run_eval", {
                "kind": "val", "max_batches": max_val_batches,
            })
            if not r.get("ok"):
                raise RuntimeError(f"run_eval(val) failed: {r.get('error')}")
            val_metrics = r["result"] or {}
            cur = val_metrics.get(val_key)
            if cur is not None:
                if best_metric is None or (cur < best_metric if lower_better else cur > best_metric):
                    best_metric = float(cur)
            logger.info(
                "val: %s",
                {k: (round(v, 5) if isinstance(v, float) else v) for k, v in val_metrics.items()},
            )

            # ---- checkpoint (trusted console pulls M's LoRA state) ----
            try:
                u_spec = u_cli.action(trace_id, "EVAL", 0, "save_checkpoint", {}).get("result") or {}
                m_ckpt = _fetch_m_checkpoint(m_cli, trace_id)
                _save_checkpoint(
                    last_ckpt, epoch, global_step, best_metric, m_ckpt, u_spec, cfg,
                )
            except Exception as ckpt_e:  # noqa: BLE001
                logger.warning("checkpoint save failed (continuing): %s", ckpt_e)
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break
        logger.info("training complete: best_metric=%s steps=%d", best_metric, global_step)
    else:
        if not os.path.exists(last_ckpt):
            raise SystemExit(f"no checkpoint at {last_ckpt} for --skip_train")
        ck = _load_checkpoint(m_cli, trace_id, last_ckpt)
        logger.info("skip_train: loaded %s (epoch=%s step=%s)", last_ckpt, ck.get("epoch"), ck.get("global_step"))

    # ---- adapter export (PEFT, from M's LoRA state) ----
    os.makedirs(cfg["adapter_dir"], exist_ok=True)
    from shared.scripts.biotriplex_finetune import save_peft_adapter
    adapter_src = AdapterSource(m_cli, trace_id, u_layers=int(cfg["u_layers"]))
    save_peft_adapter(adapter_src, cfg["hf_model"], cfg["adapter_dir"], logger)
    logger.info("adapter exported -> %s", cfg["adapter_dir"])

    # ---- final test eval (live protocol, class logits only) ----
    max_test_batches = int(os.environ.get("CF_MAX_VAL_BATCHES", "0"))
    r = u_cli.action(trace_id, "EVAL", 0, "run_eval", {
        "kind": "test", "max_batches": max_test_batches,
    })
    if not r.get("ok"):
        raise RuntimeError(f"run_eval(test) failed: {r.get('error')}")
    test_metrics = r["result"] or {}
    test_metrics["name"] = f"{task_type}_cipherforge"
    out_path = os.path.join(cfg["log_dir"], f"{task_type}_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    if task_type == "clinvar":
        with open(os.path.join(cfg["log_dir"], "clinvar_auprc.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    logger.info("eval result -> %s", out_path)

    for cli in (u_cli, m_cli, s_cli):
        try:
            cli.action(trace_id, "EVAL", 0, "shutdown", {})
        except Exception:
            pass


if __name__ == "__main__":
    main()
