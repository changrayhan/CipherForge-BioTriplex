#!/usr/bin/env python3
"""U（协调者）进程：数据/标签只在本地；通过 HTTP 驱动 M/S 完成 CipherForge
三方隔离微调（真实块 PIR，S 标签无关）。"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from shared.remote_protocol import RemoteClient, RemoteProtocol  # noqa: E402
from shared.parties.party_u import PartyU  # noqa: E402
from shared.parties.crypto_workers.pool import CryptoWorkerPool  # noqa: E402
from shared.parties.crypto_workers.crypto_u import CryptoUWorker  # noqa: E402
from shared.core.s3pir_hints import HintTable  # noqa: E402

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

    # ---- 数据 + tokenizer（U 本地） ----
    from transformers import AutoTokenizer
    from shared.data.clinvar_dataset import load_clinvar_samples, ClinVarQADataset

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Task profile: class outputs -> answer token ids ----
    from coordinator.task_profiles import get_profile
    _task_type = cfg.get("task_type", "clinvar")
    _profile = get_profile(_task_type)
    class_outputs = list(cfg.get("class_outputs") or _profile["class_outputs"])
    eval_mode = cfg.get("eval_mode") or _profile["eval_mode"]
    _ans_prefix = _profile.get("answer_prefix", " ")
    class_token_ids = []
    for _out in class_outputs:
        _ids = tokenizer(_ans_prefix + _out, add_special_tokens=False).input_ids
        if len(_ids) < 2:
            raise RuntimeError(
                f"class output {_out!r} does not tokenize with prefix into >=2 tokens: {_ids}")
        class_token_ids.append(int(_ids[1]))
    if len(set(class_token_ids)) != len(class_token_ids):
        raise RuntimeError(f"class answer tokens are not unique: {class_token_ids}")
    answer_token_to_class = {str(tok): i for i, tok in enumerate(class_token_ids)}
    logger.info(
        "task_type=%s eval_mode=%s classes=%d class_token_ids=%s",
        _task_type, eval_mode, len(class_token_ids), class_token_ids,
    )

    train_samples, val_samples, test_samples = load_clinvar_samples(cfg["data_dir"])
    train_ds = ClinVarQADataset(train_samples, tokenizer, max_length=cfg["max_seq_length"])
    val_ds = ClinVarQADataset(val_samples, tokenizer, max_length=cfg["max_seq_length"])
    test_ds = ClinVarQADataset(test_samples, tokenizer, max_length=cfg["max_seq_length"])
    logger.info("datasets: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    # ---- 经验标签边际：dummy 采样分布对齐真实监督 token 分布 ----
    from collections import Counter
    label_counter: Counter = Counter()
    for ex in train_ds:
        lab = ex["output_ids"]
        for t in lab[lab != -100].tolist():
            label_counter[int(t)] += 1
    if not label_counter:
        raise RuntimeError("train labels empty — cannot derive PIR dummy distribution")
    _smooth = 1e-3
    _n_tok = len(label_counter)
    _total = sum(label_counter.values())
    pir_dummy_weights = [
        [t, (c + _smooth) / (_total + _smooth * _n_tok)]
        for t, c in label_counter.most_common()
    ]
    logger.info(
        "PIR dummy distribution aligned to label marginal: %s",
        [(t, round(w, 4)) for t, w in pir_dummy_weights],
    )

    yes_tok = tokenizer("Yes", add_special_tokens=False).input_ids
    no_tok = tokenizer("No", add_special_tokens=False).input_ids
    space_tok = tokenizer(" ", add_special_tokens=False).input_ids
    assert len(yes_tok) == 1 and len(no_tok) == 1
    assert len(space_tok) == 1

    pir_mode = args.pir_mode or cfg.get("pir_mode", "block")
    if pir_mode not in ("block", "rms"):
        raise SystemExit(f"unsupported pir_mode: {pir_mode!r}")
    logger.info("PIR mode: %s", pir_mode)

    m_url = cfg["nodes"]["M"]
    s_url = cfg["nodes"]["S"]
    m_cli, s_cli = RemoteClient(m_url), RemoteClient(s_url)
    for role, cli in (("M", m_cli), ("S", s_cli)):
        h = cli.hello()
        assert h.get("ok") and h.get("role") == role, f"{role} hello failed: {h}"
    logger.info("nodes ready: M=%s S=%s", m_url, s_url)

    trace_id = "task_%s_%s" % (time.strftime("%Y%m%d%H%M%S"), "3pty")

    # ---- worker_config（发给 M/S） ----
    worker_config = {
        "vocab_size": cfg["vocab_size"], "hidden_dim": cfg["hidden_dim"],
        "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
        "scale": cfg["scale"], "lam": cfg["lam"], "pir_block_size": cfg["pir_block_size"],
        "debug_grad": cfg.get("debug_grad", False),
        "yes_token_id": yes_tok[0], "no_token_id": no_tok[0],
        "task_type": _task_type,
        "eval_mode": eval_mode,
        "class_outputs": class_outputs,
        "class_token_ids": class_token_ids,
        "answer_token_to_class": answer_token_to_class,
        "u_layers": cfg["u_layers"], "m_layers": cfg["m_layers"],
        "lora_r": cfg["lora_rank"], "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "gradient_clip_norm": 1.0, "warmup_steps": cfg["warmup_steps"],
        "lr_scheduler": "cosine_with_warmup",
        "batch_size": cfg["batch_size"], "max_epochs": cfg["max_epochs"],
        "n_train_samples": len(train_ds),
        "N_CRYPTO_U_WORKERS": cfg.get("N_CRYPTO_U_WORKERS", 8),
        "N_CRYPTO_M_WORKERS": cfg.get("N_CRYPTO_M_WORKERS", 8),
        "N_CRYPTO_S_WORKERS": cfg.get("N_CRYPTO_S_WORKERS", 1),
        "pir_dummy_weights": pir_dummy_weights,
        "pir_fake_ratio": cfg.get("pir_fake_ratio", 0.0),
        "http_timeout_s": cfg.get("http_timeout_s", 300),
        "pir_mode": pir_mode,
        "dp_enable": cfg.get("dp_enable", False),
        "dp_alpha": cfg.get("dp_alpha", 0.15),
        "dp_eta0": cfg.get("dp_eta0"),
        "dp_answer_beta": cfg.get("dp_answer_beta", 0.5),
        "dp_calibration_steps": cfg.get("dp_calibration_steps", 1),
        "dp_calibration_mode": cfg.get("dp_calibration_mode", False),
        "dp_num_classes": cfg.get("dp_num_classes", _profile["dp_num_classes"]),
        "dp_clip_value": cfg.get("dp_clip_value"),
    }

    # ---- INIT：M 先出密钥，再初始化运行时；S 建库/种子 ----
    r = m_cli.action(trace_id, "INIT", 0, "bfv_keygen", {
        "vocab_size": cfg["vocab_size"], "hidden_dim": cfg["hidden_dim"],
        "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
        "scale": cfg["scale"],
    })
    assert r.get("ok"), r
    pk_pem_b64 = r["result"]["pk_pem_b64"]
    logger.info("M pk_sha256=%s", r["result"].get("pk_sha256"))
    m_cli.action(trace_id, "INIT", 0, "init_runtime", worker_config)
    s_cli.action(trace_id, "INIT", 0, "init_runtime", worker_config)
    r = s_cli.action(trace_id, "INIT", 0, "build_enc_db", {"pk_pem_b64": pk_pem_b64})
    assert r.get("ok"), r
    logger.info("S pk_sha256=%s", r["result"].get("pk_sha256"))
    prg_seed = os.urandom(32)
    r = s_cli.action(trace_id, "INIT", 0, "pir_prg_setup",
                     {"prg_seed_b64": base64.b64encode(prg_seed).decode()})
    assert r.get("ok"), r
    logger.info("INIT 完成：密钥/密文库/PRG 就绪")

    # ---- U 侧 PartyU + CryptoUWorker ----
    pk_bytes = base64.b64decode(pk_pem_b64)
    pk_pem_pickled = base64.b64encode(
        __import__("pickle").dumps({"pk_bytes": pk_bytes})
    ).decode()
    hints_dir = cfg.get("hints_dir")
    hint_table = None
    if hints_dir and (Path(hints_dir) / "hint_table.json").exists():
        hint_table = HintTable.from_cache_files(hints_dir)
    party_u = PartyU(
        model_path=cfg["hf_model"], bfv_pk_pem=pk_bytes, prg_seed=prg_seed,
        hint_table=hint_table, config=worker_config,
    )
    from shared.model.model_splitting import clear_safetensor_cache
    clear_safetensor_cache()
    from shared.core.bfv_privselect_v2_adapter import get_plain_modulus
    plain_modulus = get_plain_modulus(int(cfg["poly_degree"]), int(cfg["plain_bits"]))

    # ---- RMS-PIR v2 预备：U 作为 offline server 需要本地密文库 ----
    rms_db_path = ""
    rms_store = None
    if pir_mode == "rms":
        from shared.core.rms_pir import RMSHintParams, RMSHintStore

        rms_p = int(cfg.get("rms_partition_size", 200))
        rms_lam = int(cfg.get("rms_lam", 16))
        rms_params_obj = RMSHintParams(int(cfg["vocab_size"]), rms_p, rms_lam)
        worker_config["rms_params"] = rms_params_obj.to_dict()
        _rms_seed_env = os.environ.get("CF_RMS_SEED_HEX", "")
        rms_seed = bytes.fromhex(_rms_seed_env) if _rms_seed_env else os.urandom(32)
        logger.info("RMS seed (hex): %s", rms_seed.hex())
        try:
            (Path(cfg["log_dir"]) / "rms_seed.txt").write_text(rms_seed.hex() + "\n")
        except Exception as e:
            logger.warning("could not write rms_seed.txt: %s", e)
        rms_dir = cfg.get("rms_hints_dir") or "/root/autodl-tmp/CipherForge-RMS/rms_hints"
        rms_dir = os.path.abspath(rms_dir)
        rms_store = RMSHintStore(rms_seed, rms_params_obj, rms_dir)

        # 下载 S 的密文库（Enc(-V), pk_M 加密，U 不可解密；V 为公开权重）
        rms_db_dir = cfg.get("rms_db_dir") or "/root/autodl-tmp/CipherForge-RMS/rms_db"
        rms_db_dir = os.path.abspath(rms_db_dir)
        os.makedirs(rms_db_dir, exist_ok=True)
        rms_db_path = os.path.join(rms_db_dir, "bfv_ct_db_n32000_d2048_p4096.bin")
        if not os.path.exists(rms_db_path) or os.path.getsize(rms_db_path) < 1 << 20:
            chunk_mb = max(1, int(cfg.get("rms_db_download_chunk_mb", 32)))
            chunk_bytes = chunk_mb << 20
            t_db = time.time()
            offset = 0
            with open(rms_db_path, "wb") as fh:
                while True:
                    r = s_cli.action(trace_id, "INIT", 0, "db_download",
                                     {"offset": offset, "size": chunk_bytes})
                    assert r.get("ok"), r
                    data = base64.b64decode(r["result"]["data_b64"])
                    if not data:
                        break
                    fh.write(data)
                    offset += len(data)
                    if r["result"].get("eof"):
                        break
            logger.info(
                "RMS v2: encrypted DB downloaded to %s (%.2f GB) in %.1fs",
                rms_db_path, os.path.getsize(rms_db_path) / 1e9,
                time.time() - t_db,
            )
        else:
            logger.info("RMS v2: local encrypted DB cache hit: %s", rms_db_path)

    u_pool = CryptoWorkerPool(
        CryptoUWorker,
        n_workers=int(worker_config["N_CRYPTO_U_WORKERS"]),
        init_kwargs={
            "bfv_pk_pem": base64.b64decode(pk_pem_pickled), "prg_seed": prg_seed,
            "poly_degree": cfg["poly_degree"], "plain_bits": cfg["plain_bits"],
            "scale": cfg["scale"], "plain_modulus": plain_modulus,
            "rms_db_path": rms_db_path,
            "rms_n_entries": int(cfg["vocab_size"]) if rms_db_path else 0,
        },
    )
    party_u.crypto_u_pool = u_pool

    # ---- RMS-PIR v2: U 本地构建 offline hint pool（S 不可见）----
    if pir_mode == "rms":
        from concurrent.futures import ThreadPoolExecutor

        store = rms_store
        min_cov = int(cfg.get(
            "rms_min_coverage", int(cfg.get("batch_size", 16)) + 4
        ))
        # 关键修复：初始 hint 池必须覆盖训练中真实出现的全部标签 token。
        # 注意 tokenizer(" ") 得到 259（普通空格），而答案 " Yes"/" No" 编码为
        # [29871(▁), ...]——▁(29871) 才是 y_shift 后的真实标签且占比 50%。
        # 用标签边际（pir_dummy_weights）里的 token 作为 known_labels，
        # 否则 29871 零覆盖、训练一开始 hint 池即耗尽（RMS-PIR 发散根因）。
        known_labels = [int(t) for t, _ in pir_dummy_weights]
        if not known_labels:
            known_labels = [space_tok[0], yes_tok[0], no_tok[0]]
        req, topups = store.build_initial_pool(
            known_labels, min_coverage=min_cov,
        )
        chunk = int(cfg.get("rms_offline_chunk", 64))
        ids = sorted(req) + sorted(topups)
        t0 = time.time()

        def _fetch_chunk(ids_chunk):
            row_lists = [
                req[j] if j in req else topups[j]["row_list"] for j in ids_chunk
            ]
            out = u_pool.submit({"mode": "rms_local_parity",
                                 "row_lists": row_lists})
            return ids_chunk, out.get("parities") or []

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(_fetch_chunk, ids[s : s + chunk])
                for s in range(0, len(ids), chunk)
            ]
            for f in futures:
                ids_chunk, pars = f.result()
                for j, p in zip(ids_chunk, pars):
                    if j in topups:
                        t = topups[j]
                        store.add_hint(j, t["picked_rows"], t["y"], p)
                    else:
                        store.complete_hint(j, p)
        logger.info(
            "RMS v2 offline (U-local): %d hints (%d top-ups) built in %.1fs "
            "(hints dir=%s)",
            len(ids), len(topups), time.time() - t0, rms_dir,
        )
        party_u.rms_store = store

    # ---- Trainer + RemoteProtocol ----
    from shared.remote_protocol import RemoteProtocol as RP
    from shared.training.trainer import TrainerConfig
    from coordinator.remote_trainer import RemoteTrainer

    protocol = RP(party_u, m_url, s_url, worker_config, trace_id, prg_seed=prg_seed)
    protocol.u_layers = int(cfg["u_layers"])
    trainer_cfg = TrainerConfig(
        max_epochs=args.max_epochs or cfg["max_epochs"],
        patience=cfg["patience"], train_ratio=0.9, seed=cfg["seed"],
        val_metric=cfg["val_metric"], save_freq=1, log_freq=args.log_freq,
        checkpoint_dir=cfg["checkpoint_dir"], log_dir=cfg["log_dir"],
        dump_attacks=False, batch_size=args.batch_size or cfg["batch_size"],
        max_seq_length=cfg["max_seq_length"], USE_CHUNKED_PIPELINE=True,
        CHUNK_TOKENS=cfg.get("CHUNK_TOKENS", 128), do_test_eval=False,
        task_type=cfg.get("task_type", "clinvar"),
        # Sync dchi-privacy knobs so the trainer can fit the per-class CTI.
        dp_enable=bool(cfg.get("dp_enable", False)),
        dp_alpha=cfg.get("dp_alpha", 0.15),
        dp_eta0=cfg.get("dp_eta0"),
        dp_answer_beta=cfg.get("dp_answer_beta", 0.5),
        dp_num_classes=int(cfg.get("dp_num_classes", 2)),
        dp_calibration_steps=int(cfg.get("dp_calibration_steps", 1)),
        dp_calibration_mode=bool(cfg.get("dp_calibration_mode", False)),
        dp_clip_value=cfg.get("dp_clip_value"),
    )
    trainer = RemoteTrainer(
        config=trainer_cfg, ipc_protocol=protocol,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, tokenizer=tokenizer,
    )
    if args.resume:
        ck = os.path.join(cfg["checkpoint_dir"], "last_checkpoint.pt")
        if os.path.exists(ck):
            trainer.resume_from(ck)
    if args.max_train_steps > 0:
        from shared.scripts.biotriplex_finetune import _patch_trainer_for_max_steps
        _patch_trainer_for_max_steps(trainer, args.max_train_steps, logger)

    if args.skip_train:
        ck = os.path.join(cfg["checkpoint_dir"], "last_checkpoint.pt")
        if not os.path.exists(ck):
            raise SystemExit(f"no checkpoint at {ck} for --skip_train")
        trainer.resume_from(ck)
        results = {"best_metric": None, "total_steps": 0}
        logger.info("skip_train: loaded %s", ck)
    else:
        results = trainer.train()
        logger.info("training complete: best_metric=%s steps=%d",
                    results["best_metric"], results["total_steps"])
    protocol.shutdown()
    u_pool.close()

    # ---- adapter 导出 + 评测 ----
    from shared.scripts.biotriplex_finetune import save_peft_adapter
    os.makedirs(cfg["adapter_dir"], exist_ok=True)
    save_peft_adapter(protocol, cfg["hf_model"], cfg["adapter_dir"], logger)
    out_path = os.path.join(cfg["log_dir"], f"{_task_type}_eval.json")
    cmd = [
        sys.executable, "-s", cfg["eval_script"],
        "--adapter", cfg["adapter_dir"],
        "--data", os.path.join(cfg["data_dir"], "test.jsonl"),
        "--out", out_path,
        "--task_type", _task_type,
        "--class_outputs", json.dumps(class_outputs),
        "--max_seq_length", str(cfg["max_seq_length"]),
    ]
    logger.info("eval: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate_auprc.py exited with code {proc.returncode} — see "
            f"coordinator log for traceback."
        )
    if not os.path.exists(out_path):
        raise RuntimeError(
            f"evaluate_auprc.py claimed success but did not write {out_path}"
        )
    with open(out_path, encoding="utf-8") as f:
        logger.info("eval result: %s", json.dumps(json.load(f), ensure_ascii=False))

    # ---- before/after 评测汇总（/v1/eval/run 数据源，docs/02 §4）----
    fixtures_dir = ROOT / "data" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    before_path = fixtures_dir / f"{_task_type}-zeroshot-metrics.json"
    if not before_path.exists():
        logger.info("computing zero-shot baseline (微调前) ...")
        cmd_before = [
            sys.executable, "-s", cfg["eval_script"],
            "--model_id", cfg["hf_model"],
            "--data", os.path.join(cfg["data_dir"], "test.jsonl"),
            "--out", str(before_path),
            "--task_type", _task_type,
            "--class_outputs", json.dumps(class_outputs),
            "--max_seq_length", str(cfg["max_seq_length"]),
        ]
        proc_before = subprocess.run(cmd_before, check=False)
        if proc_before.returncode != 0:
            logger.warning("zero-shot baseline eval failed rc=%d", proc_before.returncode)
    with open(out_path, encoding="utf-8") as f:
        after = json.load(f)
    before = json.load(open(before_path, encoding="utf-8")) if before_path.exists() else {}
    summary_path = os.path.join(cfg["log_dir"], "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"before": before, "after": after}, f, ensure_ascii=False, indent=2)
    logger.info("eval_summary written: %s (before=%s after=%s)",
                summary_path, bool(before), bool(after))


if __name__ == "__main__":
    main()
