#!/usr/bin/env python3
"""Party U node — the real user party (paper-faithful TriadFT topology).

U owns (x, y), the bottom model, the PIR client and the masking worker pool.
It drives its own data through the paper's direct message edges:

    U -> M : H_U  (trunk_forward)
    U -> M : C_U  (grad_reconstruct)
    U <-> S : PIR (fetch_rows / rms_parity / db_download)

U NEVER receives the trunk output H_M, the plaintext share s_S, the
full-vocab logits, or M's LoRA weights.  The coordinator is a separate,
independent control plane that only issues high-level commands
(init_runtime / train_step / run_eval / shutdown) and never relays payloads.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.node_server import RoleHandler, eval_run_response, serve  # noqa: E402
from shared.remote_protocol import (  # noqa: E402
    RemoteClient,
    _RemoteSFetcher,
    _b64_bytes,
    _tensor_to_b64,
    _unb64,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("node-u")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64_bytes(s: str) -> bytes:
    return base64.b64decode(s)


class _CapturingClient:
    """Wrap a RemoteClient so every response U receives is audited (U1a).

    Enabled only when CF_U_CAPTURE_DIR is set (experiment harness). Records
    result keys, payload sizes, and flags for forbidden plaintext tensors:
    H_M / s_S / full-vocab logits / LoRA state.
    """

    def __init__(self, inner, peer: str, cap_dir: str):
        self._inner = inner
        self._peer = peer
        self._log_path = os.path.join(cap_dir, "u_responses.jsonl")

    @staticmethod
    def _size_of(v) -> int:
        if isinstance(v, (str, bytes, list, dict)):
            return len(v)
        return 0

    def action(self, trace_id, stage, step, action, params):
        r = self._inner.action(trace_id, stage, step, action, params)
        res = r.get("result") or {}
        keys = sorted(res.keys())
        flags = {
            "has_H_M": ("H_M" in res) or ("H_M_shape" in res),
            "has_s_shares": ("s_shares" in res) or ("s_shares_b64" in res),
            "has_lora_state": "lora_state_b64" in res,
            # 全词表 logits 按"词表维"判断：(B, 32000) 而非类别 (B, C)
            "has_full_logits": bool(
                "logits_b64" in res
                and len(res.get("logits_shape") or []) > 1
                and int(res["logits_shape"][1]) > 8
            ),
        }
        rec = {
            "ts": time.time(), "peer": self._peer, "stage": stage,
            "step": int(step), "action": action, "ok": bool(r.get("ok")),
            "result_keys": keys,
            "result_sizes": {k: self._size_of(res[k]) for k in keys},
            "flags": flags,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("capture write failed: %s", exc)
        return r

    def hello(self):
        return self._inner.hello()


class URole(RoleHandler):
    role = "U"
    assets = [
        "ClinVar/BioTriplex 标注数据 (x, y) — 仅本节点",
        "embed_tokens + 底部 Decoder（PartyU）",
        "PIR 客户端（real+dummy 查询块）",
        "PRG 掩码（与 S 共享种子，M 不可见）",
        "CryptoUWorker 池（仅 pk_M，无 sk_M）",
    ]

    def __init__(self, node_id: str, config: Dict[str, Any]) -> None:
        super().__init__(node_id, config)
        self.runtime: Dict[str, Any] = {}
        self._cap_dir = os.environ.get("CF_U_CAPTURE_DIR", "")

    # ------------------------------------------------------------------ #
    #  INIT
    # ------------------------------------------------------------------ #
    def act_init_runtime(self, session, params, stage, step, trace_id):
        """Build PartyU + data loaders + PIR client + masking pool.

        params:
          worker_config: full worker config (vocab/hidden/BFV/LoRA/DP/PIR ...)
          pk_pem_b64: M's BFV public key (base64)
          prg_seed_b64: shared U/S PRG seed (base64)
          m_url / s_url: peer node addresses (direct edges, paper-faithful)
          hf_model: base model path
          data_dir: directory with train/val/test jsonl
          rms: optional RMS-PIR v2 settings (hints/db dirs, seed, coverage)
        """
        wc: Dict[str, Any] = params.get("worker_config") or {}
        self.runtime["worker_config"] = wc
        self.runtime["trace_id"] = trace_id
        m_url = str(params.get("m_url") or "")
        s_url = str(params.get("s_url") or "")
        if not m_url or not s_url:
            raise RuntimeError("U node requires m_url and s_url (direct peer edges)")
        self.runtime["m_client"] = RemoteClient(m_url, timeout=float(wc.get("http_timeout_s", 300)))
        self.runtime["s_client"] = RemoteClient(s_url, timeout=float(wc.get("http_timeout_s", 300)))
        if self._cap_dir:
            os.makedirs(self._cap_dir, exist_ok=True)
            self.runtime["m_client"] = _CapturingClient(
                self.runtime["m_client"], "M", self._cap_dir)
            self.runtime["s_client"] = _CapturingClient(
                self.runtime["s_client"], "S", self._cap_dir)

        hf_model = params["hf_model"]
        data_dir = params["data_dir"]
        pk_bytes = _unb64_bytes(params["pk_pem_b64"])
        prg_seed = _unb64_bytes(params["prg_seed_b64"])
        pir_mode = wc.get("pir_mode", "block")

        # ---- tokenizer + datasets (labels stay in U) ----
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.runtime["tokenizer"] = tokenizer

        from shared.data.clinvar_dataset import ClinVarQADataset, load_clinvar_samples
        from shared.training.trainer import make_string_safe_collate
        from torch.utils.data import DataLoader

        train_samples, val_samples, test_samples = load_clinvar_samples(data_dir)
        batch_size = max(1, int(wc.get("batch_size", 16)))
        max_len = int(wc.get("max_seq_length", 128))
        train_ds = ClinVarQADataset(train_samples, tokenizer, max_length=max_len)
        val_ds = ClinVarQADataset(val_samples, tokenizer, max_length=max_len)
        test_ds = ClinVarQADataset(test_samples, tokenizer, max_length=max_len)
        self.runtime["train_loader"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=make_string_safe_collate(), drop_last=False,
        )
        self.runtime["val_loader"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=make_string_safe_collate(), drop_last=False,
        )
        self.runtime["test_loader"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=make_string_safe_collate(), drop_last=False,
        )
        self.runtime["train_iter"] = iter(self.runtime["train_loader"])
        self.runtime["epoch"] = 0
        logger.info(
            "U datasets: train=%d val=%d test=%d batch=%d",
            len(train_ds), len(val_ds), len(test_ds), batch_size,
        )

        # ---- PIR dummy distribution aligned to real label marginal ----
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
        wc["pir_dummy_weights"] = pir_dummy_weights
        logger.info(
            "PIR dummy distribution aligned to label marginal: %s",
            [(t, round(w, 4)) for t, w in pir_dummy_weights],
        )

        # ---- PartyU (bottom model) ----
        hints_dir = self.config.get("hints_dir", "")
        hint_table = None
        if hints_dir and (Path(hints_dir) / "hint_table.json").exists():
            from shared.core.s3pir_hints import HintTable
            hint_table = HintTable.from_cache_files(hints_dir)
        from shared.parties.party_u import PartyU
        party_u = PartyU(
            model_path=hf_model, bfv_pk_pem=pk_bytes, prg_seed=prg_seed,
            hint_table=hint_table, config=wc,
        )
        from shared.model.model_splitting import clear_safetensor_cache
        clear_safetensor_cache()

        # ---- CryptoUWorker pool ----
        from shared.core.bfv_privselect_v2_adapter import get_plain_modulus
        from shared.parties.crypto_workers.crypto_u import CryptoUWorker
        from shared.parties.crypto_workers.pool import CryptoWorkerPool

        plain_modulus = get_plain_modulus(
            int(wc.get("poly_degree", 4096)), int(wc.get("plain_bits", 30)),
        )
        pk_pem_pickled = base64.b64encode(
            pickle.dumps({"pk_bytes": pk_bytes})
        ).decode()

        # RMS-PIR v2: U is the offline server -> local encrypted DB copy
        rms_db_path = ""
        rms_params_obj = None
        if pir_mode == "rms":
            from shared.core.rms_pir import RMSHintParams
            rms_p = int((params.get("rms") or {}).get("rms_partition_size", 200))
            rms_lam = int((params.get("rms") or {}).get("rms_lam", 16))
            rms_params_obj = RMSHintParams(int(wc["vocab_size"]), rms_p, rms_lam)
            wc["rms_params"] = rms_params_obj.to_dict()
            rms_cfg = params.get("rms") or {}
            rms_dir = os.path.abspath(rms_cfg.get("rms_hints_dir")
                                     or "/root/autodl-tmp/CipherForge-RMS/rms_hints")
            os.makedirs(rms_dir, exist_ok=True)
            rms_db_dir = os.path.abspath(rms_cfg.get("rms_db_dir")
                                         or "/root/autodl-tmp/CipherForge-RMS/rms_db")
            os.makedirs(rms_db_dir, exist_ok=True)
            rms_db_path = os.path.join(
                rms_db_dir,
                f"bfv_ct_db_n{wc['vocab_size']}_d{wc['hidden_dim']}_p{wc.get('poly_degree', 4096)}.bin",
            )
            if not os.path.exists(rms_db_path) or os.path.getsize(rms_db_path) < 1 << 20:
                chunk_mb = max(1, int(rms_cfg.get("rms_db_download_chunk_mb", 32)))
                chunk_bytes = chunk_mb << 20
                t_db = time.time()
                offset = 0
                with open(rms_db_path, "wb") as fh:
                    while True:
                        r = self.runtime["s_client"].action(
                            trace_id, "INIT", 0, "db_download",
                            {"offset": offset, "size": chunk_bytes},
                        )
                        assert r.get("ok"), r
                        data = _unb64_bytes(r["result"]["data_b64"])
                        if not data:
                            break
                        fh.write(data)
                        offset += len(data)
                        if r["result"].get("eof"):
                            break
                logger.info(
                    "RMS v2: encrypted DB downloaded to %s (%.2f GB) in %.1fs",
                    rms_db_path, os.path.getsize(rms_db_path) / 1e9, time.time() - t_db,
                )
            else:
                logger.info("RMS v2: local encrypted DB cache hit: %s", rms_db_path)

        pool = CryptoWorkerPool(
            CryptoUWorker,
            n_workers=int(wc.get("N_CRYPTO_U_WORKERS", 1)),
            init_kwargs={
                "bfv_pk_pem": _unb64_bytes(pk_pem_pickled),
                "prg_seed": prg_seed,
                "poly_degree": int(wc.get("poly_degree", 4096)),
                "plain_bits": int(wc.get("plain_bits", 30)),
                "scale": int(wc.get("scale", 10000)),
                "plain_modulus": plain_modulus,
                "rms_db_path": rms_db_path,
                "rms_n_entries": int(wc["vocab_size"]) if rms_db_path else 0,
            },
        )
        party_u.crypto_u_pool = pool
        party_u._s_ref = _RemoteSFetcher(self.runtime["s_client"], trace_id)

        # ---- RMS-PIR v2 offline hint pool (S never sees it) ----
        if pir_mode == "rms":
            from concurrent.futures import ThreadPoolExecutor
            from shared.core.rms_pir import RMSHintStore

            rms_cfg = params.get("rms") or {}
            rms_seed = bytes.fromhex(rms_cfg.get("rms_seed_hex", "")) if rms_cfg.get("rms_seed_hex") else os.urandom(32)
            store = RMSHintStore(rms_seed, rms_params_obj, rms_dir)
            min_cov = int(rms_cfg.get(
                "rms_min_coverage", int(wc.get("batch_size", 16)) + 4,
            ))
            known_labels = [int(t) for t, _ in pir_dummy_weights]
            if not known_labels:
                yes_id = int(wc.get("yes_token_id", -1))
                no_id = int(wc.get("no_token_id", -1))
                space_id = int(tokenizer(" ", add_special_tokens=False).input_ids[0])
                known_labels = [space_id, yes_id, no_id]
            req, topups = store.build_initial_pool(known_labels, min_coverage=min_cov)
            chunk = int(rms_cfg.get("rms_offline_chunk", 64))
            ids = sorted(req) + sorted(topups)
            t0 = time.time()

            def _fetch_chunk(ids_chunk):
                row_lists = [
                    req[j] if j in req else topups[j]["row_list"] for j in ids_chunk
                ]
                out = pool.submit({"mode": "rms_local_parity", "row_lists": row_lists})
                return ids_chunk, out.get("parities") or []

            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [
                    ex.submit(_fetch_chunk, ids[s: s + chunk])
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
            party_u.rms_store = store
            logger.info(
                "RMS v2 offline (U-local): %d hints (%d top-ups) built in %.1fs",
                len(ids), len(topups), time.time() - t0,
            )

        self.runtime["party_u"] = party_u
        self.runtime["pool"] = pool
        return (
            [{"type": "task", "payload": {"msg": "U runtime 就绪"}}],
            {},
            {
                "steps_per_epoch": max(1, int(np.ceil(len(train_ds) / batch_size))),
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_test": len(test_ds),
                "yes_token_id": int(wc.get("yes_token_id", -1)),
                "no_token_id": int(wc.get("no_token_id", -1)),
            },
        )

    # ------------------------------------------------------------------ #
    #  LABEL PREP (U-local; mirrors the old coordinator semantics)
    # ------------------------------------------------------------------ #
    def _prepare_labels(self, batch: Dict) -> Dict:
        wc = self.runtime["worker_config"]
        lab = batch.get("output_ids") if isinstance(batch, dict) else None
        out = {
            "y_valid": [], "valid_indices": [], "valid_mask": None,
            "monitor_positions": None, "gold_yes": None, "gold_class_ids": None,
        }
        if lab is None or not isinstance(lab, torch.Tensor):
            return out
        lab = lab.to("cpu")
        B, S = lab.shape
        y_shift = torch.full_like(lab, -100)
        if S > 1:
            y_shift[:, :-1] = lab[:, 1:]
        valid = y_shift != -100
        valid_idx = valid.flatten().nonzero().flatten().tolist()
        y_valid = y_shift.flatten()[valid_idx].tolist()
        first = (lab != -100).long().argmax(dim=1)
        monitor_pos = (first - 1).clamp(min=0).tolist()
        yes_id = int(wc.get("yes_token_id", -1))
        gold_pos = (first + 1).clamp(max=S - 1)
        gold_tok = lab.gather(1, gold_pos.unsqueeze(1)).squeeze(1)
        gold_yes = (gold_tok == yes_id).float().tolist()
        cls_map = wc.get("answer_token_to_class") or {}
        gold_class = [int(cls_map.get(str(int(t)), -1)) for t in gold_tok.tolist()]
        out.update({
            "y_valid": y_valid,
            "valid_indices": valid_idx,
            "valid_mask": valid.numpy(),
            "monitor_positions": monitor_pos,
            "gold_yes": gold_yes,
            "gold_class_ids": gold_class,
        })
        return out

    @staticmethod
    def _binary_ce_from_p(gold_yes, p_yes) -> Optional[float]:
        if not gold_yes or not p_yes or len(gold_yes) != len(p_yes):
            return None
        import math
        total = 0.0
        for g, p in zip(gold_yes, p_yes):
            p = min(max(float(p), 1e-7), 1.0 - 1e-7)
            total += -(g * math.log(p) + (1.0 - g) * math.log(1.0 - p))
        return total / max(len(gold_yes), 1)

    def _next_train_batch(self):
        try:
            return next(self.runtime["train_iter"])
        except StopIteration:
            self.runtime["epoch"] += 1
            loader = self.runtime["train_loader"]
            self.runtime["train_iter"] = iter(loader)
            return next(self.runtime["train_iter"])

    # ------------------------------------------------------------------ #
    #  TRAIN STEP (paper edges: U->M H_U/C_U, U<->S PIR)
    # ------------------------------------------------------------------ #
    def act_train_step(self, session, params, stage, step, trace_id):
        t0 = time.time()
        wc = self.runtime["worker_config"]
        party_u = self.runtime["party_u"]
        m = self.runtime["m_client"]
        global_step = int(params["step"])
        batch = self._next_train_batch()

        labels_info = self._prepare_labels(batch)
        valid_indices = labels_info["valid_indices"]
        if not valid_indices:
            raise RuntimeError(f"step {global_step}: no valid answer tokens")
        y_valid = labels_info["y_valid"]
        gold_yes = labels_info["gold_yes"]
        monitor_positions = labels_info["monitor_positions"]

        # 1. U -> M : H_U (direct edge; M pushes H_M to S and asks for s_S)
        u_result = party_u.forward_train(batch)
        H_U = u_result["H_U"]
        attn = batch.get("attention_mask") if isinstance(batch, dict) else None
        r = m.action(
            trace_id, "FORWARD", global_step, "trunk_forward", {
                "H_U": _tensor_to_b64(H_U),
                "H_U_shape": list(H_U.shape),
                "attention_mask": attn.tolist() if attn is not None else None,
                "monitor_positions": monitor_positions,
                "valid_indices": valid_indices,
                "step": global_step,
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"trunk_forward failed: {r.get('error')}")
        p_yes = r["result"].get("p_yes")

        # 2. U <-> S : PIR query + local mask -> Enc(-V_y + r_t)
        block_size = int(wc.get("pir_block_size", 8))
        ct_list = party_u.pir_query_mask(
            party_u._s_ref, y_valid, valid_indices, global_step, block_size,
        )

        # 3. U -> M : C_U (direct edge)
        r = m.action(
            trace_id, "BACKWARD", global_step, "grad_reconstruct", {
                "cts": [_b64_bytes(c) for c in ct_list],
                "valid_indices": valid_indices,
                "expected_shape": [
                    int(batch["input_ids"].shape[0]),
                    int(batch["input_ids"].shape[1]),
                ],
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"grad_reconstruct failed: {r.get('error')}")

        # 4. M combines C_U + s_S and updates LoRA
        r = m.action(trace_id, "UPDATE", global_step, "lora_update", {})
        if not r.get("ok"):
            raise RuntimeError(f"lora_update failed: {r.get('error')}")
        metrics = r.get("metrics") or {}

        eval_mode = str(wc.get("eval_mode", "binary"))
        monitor_ce = self._binary_ce_from_p(gold_yes, p_yes) if eval_mode == "binary" else None
        if self._cap_dir:
            try:
                h_full = H_U.detach().cpu().half().numpy()
                np.savez(
                    os.path.join(self._cap_dir, f"train_{global_step:05d}.npz"),
                    h_u_full=h_full,
                    monitor_positions=np.asarray(
                        monitor_positions or [], dtype=np.int64),
                    p_yes=np.asarray(p_yes or [], dtype=np.float32),
                    gold_yes=np.asarray(gold_yes or [], dtype=np.float32),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("train capture failed: %s", exc)
        logger.info(
            "step %d: g_absmax=%.4g g_meanabs=%.4g loss=%.4g",
            global_step,
            metrics.get("g_absmax", -1), metrics.get("g_meanabs", -1),
            metrics.get("loss", 0.0),
        )
        return (
            [{"type": "message", "payload": {"edge": "U->M", "msg": "H_U / C_U 已直发 M"}}],
            metrics,
            {
                "step": global_step,
                "monitor_ce": monitor_ce,
                "step_time_ms": (time.time() - t0) * 1000,
            },
        )

    # ------------------------------------------------------------------ #
    #  EVAL (S returns only class logits — never full-vocab logits)
    # ------------------------------------------------------------------ #
    def act_run_eval(self, session, params, stage, step, trace_id):
        kind = str(params.get("kind", "val"))
        max_batches = int(params.get("max_batches") or 0)
        party_u = self.runtime["party_u"]
        m = self.runtime["m_client"]
        wc = self.runtime["worker_config"]
        loader = self.runtime["val_loader"] if kind == "val" else self.runtime["test_loader"]
        eval_mode = str(wc.get("eval_mode", "binary"))
        class_ids = [int(t) for t in (wc.get("class_token_ids") or [])]
        yes_id = int(wc.get("yes_token_id", -1))
        no_id = int(wc.get("no_token_id", -1))

        all_probs: List[float] = []
        all_gold: List[int] = []
        all_pred: List[int] = []
        all_gene: List[str] = []
        n_samples = 0

        for b_idx, batch in enumerate(loader):
            if max_batches > 0 and b_idx >= max_batches:
                break
            lab = batch["output_ids"].to("cpu")
            attn = batch.get("attention_mask")
            attn_cpu = attn if isinstance(attn, torch.Tensor) else torch.as_tensor(attn)
            attn_cpu = attn_cpu.to("cpu").long()
            first = (lab != -100).long().argmax(dim=1)
            score_pos = first.tolist()

            u_result = party_u.forward_val(batch)
            H_U = u_result["H_U"]
            r = m.action(
                trace_id, "EVAL", 0, "val_forward", {
                    "H_U": _tensor_to_b64(H_U),
                    "H_U_shape": list(H_U.shape),
                    "attention_mask": attn_cpu.tolist(),
                    "positions": score_pos,
                },
            )
            if not r.get("ok"):
                raise RuntimeError(f"val_forward failed: {r.get('error')}")
            res = r["result"]
            logits = np.frombuffer(_unb64_bytes(res["logits_b64"]), dtype=np.float32).copy()
            logits = torch.from_numpy(logits).view(*res["logits_shape"]).float()  # (B, C)
            if self._cap_dir:
                try:
                    h_full = H_U.detach().cpu().half().numpy()
                    np.savez(
                        os.path.join(self._cap_dir, f"eval_{kind}_{b_idx:04d}.npz"),
                        h_u_full=h_full,
                        score_pos=np.asarray(score_pos, dtype=np.int64),
                        z_cls=logits.numpy().astype(np.float32),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("eval capture failed: %s", exc)

            gold_pos = (first + 1).clamp(max=lab.size(1) - 1)
            gold_tok = lab.gather(1, gold_pos.unsqueeze(1)).squeeze(1)
            if eval_mode == "binary":
                probs = torch.softmax(logits, dim=-1)[:, 0].cpu().tolist()
                gold = [1 if int(g) == yes_id else 0 for g in gold_tok.tolist()]
                all_probs.extend(probs)
                all_gold.extend(gold)
                all_pred.extend([1 if p >= 0.5 else 0 for p in probs])
                n_samples += len(probs)
                meta = batch.get("meta") or []
                for m_ in meta:
                    all_gene.append(m_.get("gene", "") if isinstance(m_, dict) else "")
            else:
                pred = logits.argmax(dim=-1).cpu().tolist()
                cls_map = wc.get("answer_token_to_class") or {}
                gold = [int(cls_map.get(str(int(g)), -1)) for g in gold_tok.tolist()]
                all_pred.extend(pred)
                all_gold.extend(gold)
                n_samples += len(pred)

        if not any(all_gene):
            all_gene = [""] * len(all_probs)

        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

        if eval_mode == "binary":
            try:
                auprc = float(average_precision_score(all_gold, all_probs))
            except Exception:
                auprc = 0.0
            try:
                auc = float(roc_auc_score(all_gold, all_probs)) if len(set(all_gold)) > 1 else 0.5
            except Exception:
                auc = 0.5
            acc = float(accuracy_score(all_gold, all_pred)) if all_gold else 0.0
            ce_total = 0.0
            import math
            for g, p in zip(all_gold, all_probs):
                p = min(max(float(p), 1e-7), 1.0 - 1e-7)
                ce_total += -(g * math.log(p) + (1.0 - g) * math.log(1.0 - p))
            ce_loss = ce_total / max(len(all_gold), 1)
            metrics = {
                "val_ce_loss": ce_loss,
                "val_auprc": auprc,
                "val_auc": auc,
                "val_accuracy": acc,
                "val_samples": n_samples,
                "val_entity_micro_f1": acc,
                "val_letter_micro_f1": acc,
                "val_macro_f1": acc,
            }
        else:
            valid = [(p, g) for p, g in zip(all_pred, all_gold) if g >= 0]
            acc = float(accuracy_score([g for _, g in valid], [p for p, _ in valid])) if valid else 0.0
            try:
                macro_f1 = float(f1_score(
                    [g for _, g in valid], [p for p, _ in valid],
                    average="macro", zero_division=0,
                )) if valid else 0.0
            except Exception:
                macro_f1 = 0.0
            metrics = {
                "val_accuracy": acc,
                "val_macro_f1": macro_f1,
                "val_samples": n_samples,
                "val_entity_micro_f1": acc,
                "val_letter_micro_f1": acc,
                "val_ce_loss": 1.0 - acc,
            }
        logger.info("eval[%s] %s", kind, {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()})
        return (
            [{"type": "compute", "payload": {"msg": f"eval[{kind}] 完成"}}],
            {},
            {"kind": kind, **metrics},
        )

    # ------------------------------------------------------------------ #
    #  CHECKPOINT / SHUTDOWN
    # ------------------------------------------------------------------ #
    def act_save_checkpoint(self, session, params, stage, step, trace_id):
        party_u = self.runtime.get("party_u")
        return (
            [],
            {},
            party_u.save_checkpoint() if party_u is not None else {"party": "U"},
        )

    def act_shutdown(self, session, params, stage, step, trace_id):
        pool = self.runtime.get("pool")
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
        return ([{"type": "message", "payload": {"msg": "U 节点关闭"}}], {}, {})

    def act_pir_prg_setup(self, session, params, stage, step, trace_id):
        """Compat action (platform INIT flow): acknowledge the shared PRG seed.

        The seed is consumed by the coordinator's init_runtime call; this
        action only exists so the demo platform's INIT orchestration does not
        get a 501 from U.
        """
        if params.get("prg_seed_b64"):
            self.runtime["prg_seed_b64"] = params["prg_seed_b64"]
        return (
            [{"type": "pir", "payload": {"msg": "U: PRG 种子已登记（幂等确认）"}}],
            {},
            {},
        )


URole.actions = {
    "init_runtime": URole.act_init_runtime,
    "train_step": URole.act_train_step,
    "run_eval": URole.act_run_eval,
    "save_checkpoint": URole.act_save_checkpoint,
    "pir_prg_setup": URole.act_pir_prg_setup,
    "shutdown": URole.act_shutdown,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--node_id", default="srv-u-01")
    ap.add_argument("--model_path", default=os.environ.get("CF_MODEL_PATH", ""))
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--hints_dir", default="")
    ap.add_argument("--metrics_dir", default=str(ROOT / "coordinator" / "logs"))
    ap.add_argument("--presets_json", default=str(ROOT / "data" / "fixtures" / "eval-presets.json"))
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("CF_MODEL_PATH must be set (--model_path)")
    serve(
        URole(args.node_id, {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "hints_dir": args.hints_dir,
            "metrics_dir": args.metrics_dir,
            "presets_json": args.presets_json,
        }),
        args.host, args.port,
    )


if __name__ == "__main__":
    main()
