#!/usr/bin/env python3
"""S 节点服务进程：完整服务方逻辑（密文库、head_forward、PIR 行取回、
份额计算、评测）。标签值永不到达本节点；只接收监督位置与查询块。"""
import argparse
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shared.core.bfv_privselect_v2_adapter import get_plain_modulus  # noqa: E402
from shared.node_server import RoleHandler, eval_run_response, serve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("node-s")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


class SRole(RoleHandler):
    role = "S"
    assets = ["输出头 V 矩阵（lm_head）", "BFV 密文库 + S3PIR hints", "PIR 响应 + 份额 s_S（标签无关）"]
    actions = {}

    def __init__(self, node_id: str, config: Dict[str, Any]) -> None:
        super().__init__(node_id, config)
        self.runtime: Dict[str, Any] = {}

    def _backend(self):
        wc = self.runtime["worker_config"]
        if "backend" not in self.runtime:
            from shared.core.bfv_privselect_v2_adapter import BFVPrivSelectV2Backend
            self.runtime["backend"] = BFVPrivSelectV2Backend(
                n_entries=int(wc["vocab_size"]), vec_dim=int(wc["hidden_dim"]),
                shared_seed=os.urandom(32), cache_dir=self.config["db_dir"],
                poly_degree=int(wc["poly_degree"]), plain_bits=int(wc["plain_bits"]),
                scale=int(wc["scale"]),
            )
        return self.runtime["backend"]

    # ---- INIT -------------------------------------------------------------
    def act_init_runtime(self, session, params, stage, step, trace_id):
        self.runtime["worker_config"] = dict(params)
        return ([{"type": "task", "payload": {"msg": "S runtime 配置就绪"}}], {}, {})

    def act_build_enc_db(self, session, params, stage, step, trace_id):
        # 平台 INIT 流程可能不先调用 init_runtime：用动作参数/默认值自举
        wc = dict(self.runtime.get("worker_config") or {})
        for k, dflt in (
            ("vocab_size", 32000), ("hidden_dim", 2048), ("poly_degree", 4096),
            ("plain_bits", 30), ("scale", 10000), ("lam", 80),
        ):
            wc.setdefault(k, int(params.get(k, dflt)))
        self.runtime["worker_config"] = wc
        pk_pem = _unb64(params["pk_pem_b64"])
        backend = self._backend()
        backend.attach_public_key(pk_pem)
        # V 矩阵 = lm_head（S 分片）
        from shared.model.model_splitting import detect_model_spec, load_s_submodel
        spec = detect_model_spec(self.config["model_path"])
        s_model = load_s_submodel(spec=spec, model_path=self.config["model_path"], device="cpu")
        V = s_model.weight.detach().float().numpy().astype(np.float64)
        backend.build_encrypted_database(V, force=False)
        backend.drop_encrypted_db()
        self.runtime["pk_pem"] = pk_pem
        import hashlib
        return (
            [{"type": "crypto", "payload": {"msg": "密文库构建/缓存命中", "rows": int(wc["vocab_size"])}}],
            {},
            {"db_ready": True, "pk_sha256": hashlib.sha256(pk_pem).hexdigest()},
        )

    def act_pir_prg_setup(self, session, params, stage, step, trace_id):
        wc = self.runtime["worker_config"]
        prg_seed = _unb64(params["prg_seed_b64"])
        pk_pem = self.runtime.get("pk_pem")
        if pk_pem is None:
            raise RuntimeError("build_enc_db must run before pir_prg_setup")
        from shared.parties.party_s import PartyS
        from shared.parties.crypto_workers.pool import CryptoWorkerPool
        from shared.parties.crypto_workers.crypto_s import CryptoSWorker
        from shared.core.s3pir_hints import HintTable

        import pickle
        pk_pem_pickled = pickle.dumps({"pk_bytes": pk_pem})

        backend = self._backend()
        hints_dir = Path(self.config["db_dir"]) / "s3pir_hints"
        partition_size = 1 << ((int(wc["vocab_size"]).bit_length() - 1) // 2)
        hint_table = HintTable(
            n_entries=int(wc["vocab_size"]), partition_size=partition_size,
            lam=int(wc.get("lam", 80)), cache_dir=str(hints_dir),
        )
        if (hints_dir / "hint_table.json").exists():
            hint_table = HintTable.from_cache_files(str(hints_dir))
        else:
            hint_table.compute_main_hints_skeleton()
            hint_table.compute_backup_hints_skeleton()
            hint_table.to_cache_files()

        party_s = PartyS(
            lm_head_path=self.config["model_path"],
            bfv_pk_pem=pk_pem,
            prg_seed=prg_seed,
            bfv_backend=backend,
            hint_table=hint_table,
            config=wc,
        )
        pool = CryptoWorkerPool(
            CryptoSWorker,
            n_workers=int(wc.get("N_CRYPTO_S_WORKERS", 1)),
            init_kwargs={
                "bfv_pk_pem": pk_pem_pickled, "prg_seed": prg_seed,
                "bfv_cache_dir": self.config["db_dir"],
                "poly_degree": int(wc["poly_degree"]),
                "plain_bits": int(wc["plain_bits"]),
                "scale": int(wc["scale"]),
                "plain_modulus": get_plain_modulus(
                    int(wc["poly_degree"]), int(wc["plain_bits"])
                ),
                "n_entries": int(wc["vocab_size"]),
                "vec_dim": int(wc["hidden_dim"]),
                "partition_size": partition_size,
                "lam": int(wc.get("lam", 80)),
                "skip_enc_db": wc.get("pir_mode") == "rms",
            },
        )
        party_s.crypto_s_pool = pool
        self.runtime["party_s"] = party_s
        self.runtime["pool"] = pool
        self.runtime["prg_seed"] = prg_seed
        from shared.model.model_splitting import clear_safetensor_cache
        clear_safetensor_cache()
        return ([{"type": "pir", "payload": {"msg": "PRG 种子已就绪（仅 U/S 共享）"}}], {}, {})

    def _V_f32(self):
        import logging as _log
        _log.getLogger().setLevel(logging.INFO)
        if "V_f32" not in self.runtime:
            # Always keep V on GPU for fast matmul. PartyS logs which device it
            # chose at startup so we can verify the assumption here.
            target_device = str(self.runtime["party_s"].V_weight.device)
            self.runtime["V_f32"] = (
                self.runtime["party_s"].V_weight.detach().float().to(target_device)
            )
        _log.info(f"[_V_f32] returning device={self.runtime['V_f32'].device}")
        return self.runtime["V_f32"]

    def _V_cpu(self):
        """CPU copy of V (safe for share_compute which needs CPU numpy output)."""
        V = self._V_f32()
        return V.cpu()

    def _logits_at(self, H_M_f32, positions):
        """z = H_M[pos] @ V^T for the given flat positions (small n).

        H_M comes from base64 decode (CPU). We move it to GPU to match V's
        device, do the matmul there, then return CPU for base64 transport.
        """
        V = self._V_f32()            # V lives on GPU (lm_head device)
        device = V.device             # GPU device
        H_gpu = H_M_f32.to(device)  # H_M is CPU; move to GPU for matmul
        sel = torch.as_tensor(positions, dtype=torch.long, device=device)
        H_flat = H_gpu.reshape(-1, H_gpu.shape[-1])
        z = H_flat[sel] @ V.t()     # (n, V) on GPU
        return z.cpu()               # back to CPU for b64 transport

    def _monitor_p_yes_from_h(self, H_M_f32, monitor_positions):
        B = H_M_f32.shape[0]
        if not monitor_positions or len(monitor_positions) != B:
            return None
        try:
            party_s = self.runtime["party_s"]
            if not hasattr(party_s, "_tokenizer"):
                from transformers import AutoTokenizer
                party_s._tokenizer = AutoTokenizer.from_pretrained(
                    party_s.spec.model_path, trust_remote_code=True, use_fast=True,
                )
                if party_s._tokenizer.pad_token is None:
                    party_s._tokenizer.pad_token = party_s._tokenizer.eos_token
            yes_id = party_s._tokenizer("Yes", add_special_tokens=False).input_ids[0]
            no_id = party_s._tokenizer("No", add_special_tokens=False).input_ids[0]
            z = self._logits_at(H_M_f32, monitor_positions)
            sc = torch.stack([z[:, yes_id], z[:, no_id]], dim=1).float()
            return torch.softmax(sc, dim=1)[:, 0].cpu().tolist()
        except Exception:
            return None

    # ---- FORWARD -----------------------------------------------------------
    def act_head_forward(self, session, params, stage, step, trace_id):
        H_M = torch.from_numpy(
            np.frombuffer(_unb64(params["H_M"]), dtype=np.float32).copy()
        ).view(*params["H_M_shape"])
        session["state"]["H_M_f32"] = H_M
        monitor_p_yes = self._monitor_p_yes_from_h(
            H_M, params.get("monitor_positions")
        )
        return (
            [{"type": "compute", "payload": {"msg": "head_forward: H_M 已缓存"}}],
            {},
            {"monitor_p_yes": monitor_p_yes},
        )

    # ---- BACKWARD ----------------------------------------------------------
    def act_fetch_rows(self, session, params, stage, step, trace_id):
        party_s = self.runtime["party_s"]
        r = party_s.pir_fetch_dispatch([int(i) for i in params["indices"]])
        rows_b64 = [_b64(r[idx]) for idx in params["indices"]]
        return (
            [{"type": "pir", "payload": {"edge": "S->U", "bytes": sum(len(x) for x in r.values()),
                                          "msg": f"PIR 块返回 {len(rows_b64)} 行"}}],
            {},
            {"rows": rows_b64},
        )

    def act_share_compute(self, session, params, stage, step, trace_id):
        H_M = session["state"].get("H_M_f32")
        if H_M is None:
            raise RuntimeError("head_forward must run before share_compute")
        positions = [int(i) for i in params["positions"]]
        z = self._logits_at(H_M, positions).float()
        probs = torch.softmax(z, dim=-1)
        # Use CPU V for share_compute since output must be numpy.
        V_cpu = self._V_cpu()
        a_all = (probs @ V_cpu).numpy().astype("float32")
        result = self.runtime["party_s"].crypto_s_pool.submit({
            "mode": "make_shares",
            "a_t_list": [a_all[i] for i in range(len(positions))],
            "t_flats": positions,
            "step": int(params.get("step", step)),
        })
        shares = np.asarray(result["s_shares"], dtype=np.int64)
        return (
            [{"type": "message", "payload": {"edge": "S->M", "bytes": shares.nbytes,
                                             "msg": "份额已发送"}}],
            {},
            {"s_shares_b64": _b64(shares.tobytes()), "n": len(positions)},
        )

    def act_rms_parity(self, session, params, stage, step, trace_id):
        """RMS-PIR v2 online parity oracle.

        S holds the PLAINTEXT V (lm_head).  For each row list it returns
        Enc(-Σ V_i) computed as: per-row fixed-point ints (round(V_i·scale)),
        summed in integers, then encoded + encrypted once with pk_M.  This is
        bit-exact with the ciphertext-sum over the encrypted DB (Σ round ==
        round per row, summed), so U's recovery math is unchanged.

        S only sees the two online subsets (real+dummy, permuted); it never
        sees hint subsets or replenishment halves (those are computed by U
        from U's local encrypted DB copy), which is what gives RMS-PIR its
        multi-query privacy.
        """
        row_lists = [[int(i) for i in lst] for lst in params.get("row_lists", [])]
        from seal import BatchEncoder, Ciphertext
        from shared.core.bfv_privselect_v2_adapter import (
            _seal_to_bytes,
            encode_vector_as_ints,
        )
        backend = self._backend()
        batch = BatchEncoder(backend._context)
        # 缓存 CPU 副本：每步只做一次 GPU->CPU 转换，避免 2.6GB 大拷贝 × N 次
        if "V_np" not in self.runtime:
            V = self._V_f32()  # (32000, 2048) float32 on S's device
            if V.is_cuda:
                V = V.cpu()
            self.runtime["V_np"] = V.detach().numpy().astype(np.float64)
        V_np = self.runtime["V_np"]
        scale = int(backend.scale)
        parities: List[str] = []
        for rows in row_lists:
            valid = [int(i) for i in rows if 0 <= int(i) < V_np.shape[0]]
            if not valid:
                ints = np.zeros(backend.poly_degree, dtype=np.int64)
            else:
                # Per-row rounding first, then integer sum — bit-exact with the
                # ciphertext-sum over Enc(-V) rows stored in the encrypted DB.
                int_sum = -np.round(V_np[valid] * scale).astype(np.int64).sum(axis=0)
                ints = np.zeros(backend.poly_degree, dtype=np.int64)
                ints[: int_sum.shape[0]] = int_sum
            pt = batch.encode(ints)
            ct = Ciphertext()
            backend._encryptor.encrypt(pt, ct)
            parities.append(_b64(_seal_to_bytes(ct)))
        return (
            [{"type": "pir", "payload": {
                "edge": "S->U", "bytes": sum(len(b) for b in parities),
                "msg": f"RMS parity ×{len(row_lists)}"}}],
            {},
            {"parities": parities},
        )

    def act_db_download(self, session, params, stage, step, trace_id):
        """Stream the encrypted DB to U (RMS-PIR v2: U is the offline server).

        Chunked base64 over the existing JSON envelope: params
        ``{"offset": int, "size": int}`` → ``{"data_b64", "offset", "n", "eof"}``.
        The DB is Enc(-V) under pk_M; U cannot decrypt it, and V is a public
        lm_head anyway — no new secret leaves S.
        """
        import os
        db_path = Path(self.config["db_dir"]) / "bfv_ct_db_n32000_d2048_p4096.bin"
        if not db_path.exists():
            raise RuntimeError(f"encrypted DB not found: {db_path}")
        fsize = db_path.stat().st_size
        offset = max(0, int(params.get("offset", 0)))
        size = max(1, min(int(params.get("size", 1 << 26)), fsize - offset))
        with open(db_path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(size)
        return (
            [{"type": "data", "payload": {"msg": f"DB chunk {offset}+{len(data)}"}}],
            {},
            {
                "data_b64": _b64(data),
                "offset": offset,
                "n": len(data),
                "file_size": fsize,
                "eof": offset + len(data) >= fsize,
            },
        )

    # ---- EVAL --------------------------------------------------------------
    def act_val(self, session, params, stage, step, trace_id):
        """Run ClinVar validation forward at the SAME last-answer position as
        baseline ``evaluate_auprc.predict_probs`` and return the FULL vocab logits.

        Contract (matches ``single_process/baseline/scripts/evaluate_auprc.py``):
            U passes ``input_ids`` and ``attention_mask`` for each sample
            (already tokenized on U with the same prompt template). For every
            sample we compute ``last_pos = attention_mask.sum(dim=1) - 1`` (last
            non-pad position), then ``logits[last_pos] @ V.T`` ∈ ℝ^{vocab}.

        U receives the per-sample full-vocab logits and runs
            softmax([logits[yes_id], logits[no_id]]) → P(Yes)
        to mirror baseline exactly.

        ``positions`` (legacy 2-token argmax API) is still supported for
        backwards compatibility — only used for token-accuracy debug logging.
        """
        import numpy as np
        H_M = torch.from_numpy(
            np.frombuffer(_unb64(params["H_M"]), dtype=np.float32).copy()
        ).view(*params["H_M_shape"])
        attn = params.get("attention_mask")  # [[B,S]] or None
        monitor_positions = params.get("monitor_positions")  # list[int] of length B (legacy)
        positions_legacy = params.get("positions")  # [[B,2]] or None

        # Scoring positions. The coordinator passes the ▁ prefix position
        # (``first``): training's y_shift makes logits[first] target the
        # Yes/No token at first+1, so val must score there to match. When
        # absent, fall back to the last non-pad position (baseline format).
        if monitor_positions is not None:
            last_pos = torch.tensor(monitor_positions, dtype=torch.long)
        elif attn is not None:
            attn_t = torch.tensor(attn, dtype=torch.long)
            last_pos = attn_t.sum(dim=1) - 1  # (B,)
        else:
            # Fallback: last sequence index per sample (assumes no padding).
            B = H_M.shape[0]
            last_pos = torch.full((B,), H_M.shape[1] - 1, dtype=torch.long)

        # 1) Compute full vocab logits at last_pos: (B, V)
        # H_M comes from base64 (CPU); move to GPU to match V, then return CPU.
        V = self._V_f32()            # V lives on GPU
        device = V.device
        H_gpu = H_M.to(device)
        # 逐样本取 last_pos 位置的隐状态。不能把 per-sample 的 last_pos
        # 直接当扁平行号用（H_flat[last_pos] 会取到第 0 个样本的对应 token，
        # 导致 B>1 时验证 logits 串位、等长样本概率完全相同）。
        row_idx = torch.arange(H_gpu.shape[0], device=device)
        logits_last = H_gpu[row_idx, last_pos] @ V.t()  # (B, V) on GPU
        # Match baseline dtype: cast to float32 for transport.
        logits_last = logits_last.detach().cpu().float().contiguous()
        logits_b64 = _b64(np.ascontiguousarray(logits_last.numpy()).tobytes())

        # 2) Legacy argmax at the 2-token answer positions (for debugging only).
        argmax_legacy: List[List[int]] = []
        if positions_legacy is not None:
            flat = [p for row in positions_legacy for p in row]
            z_legacy = self._logits_at(H_M, flat)
            z_legacy2 = z_legacy.view(len(positions_legacy), 2, -1)
            argmax_legacy = z_legacy2.argmax(dim=-1).cpu().tolist()

        # 3) Per-sample P(Yes) at last_pos using Yes/No softmax (same path as
        #    baseline). Returned for the live-loss monitor (mirrors baseline CE).
        p_yes = self._monitor_p_yes_at_logits(logits_last)

        return (
            [{"type": "compute", "payload": {"msg": "val forward 完成 (full logits)"}}],
            {},
            {
                # New (full-logits) API used by the fixed remote_val:
                "logits_b64": logits_b64,
                "logits_shape": list(logits_last.shape),
                "last_positions": last_pos.cpu().tolist(),
                "p_yes": p_yes,
                # Legacy (kept for compatibility):
                "argmax": argmax_legacy,
            },
        )

    def _monitor_p_yes_at_logits(self, logits_last: torch.Tensor) -> Optional[List[float]]:
        """Compute per-sample softmax([logits[yes], logits[no]]) → P(Yes).

        Uses S's cached tokenizer so the Yes/No token IDs match U's tokenizer.
        """
        try:
            party_s = self.runtime["party_s"]
            if not hasattr(party_s, "_tokenizer"):
                from transformers import AutoTokenizer
                party_s._tokenizer = AutoTokenizer.from_pretrained(
                    party_s.spec.model_path, trust_remote_code=True, use_fast=True,
                )
                if party_s._tokenizer.pad_token is None:
                    party_s._tokenizer.pad_token = party_s._tokenizer.eos_token
            yes_id = party_s._tokenizer("Yes", add_special_tokens=False).input_ids[0]
            no_id = party_s._tokenizer("No", add_special_tokens=False).input_ids[0]
            sc = torch.stack([logits_last[:, yes_id], logits_last[:, no_id]], dim=1).float()
            return torch.softmax(sc, dim=1)[:, 0].cpu().tolist()
        except Exception as exc:
            logger.warning("_monitor_p_yes_at_logits failed: %s", exc)
            return None

    def act_shutdown(self, session, params, stage, step, trace_id):
        try:
            self.runtime.get("pool") and self.runtime["pool"].close()
        except Exception:
            pass
        return ([{"type": "message", "payload": {"msg": "S 节点关闭"}}], {}, {})

    # ---- L1 fallback（docs/02 §3 / docs/04 §2.3）----
    def fallback_pretrained(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """S 节点：登记检查点，将后续 /v1/eval/run 的 after 侧绑定到该检查点产物。"""
        t0 = time.time()
        checkpoint_id = str(payload.get("checkpoint_id") or self.config.get("checkpoint_id", ""))
        self.runtime["fallback_checkpoint_id"] = checkpoint_id
        self.runtime["fallback_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        load_ms = int((time.time() - t0) * 1000)
        logger.info("S fallback bound to checkpoint: %s", checkpoint_id)
        return {"ok": True, "checkpoint": checkpoint_id, "loaded": True, "load_ms": load_ms}

    # ---- EVAL（docs/02 §4：acc / auprc / macro_f1 必测）----
    def eval_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return eval_run_response(
            str(payload.get("test_id", "")),
            self.config.get("metrics_dir", ""),
            self.config.get("presets_json", ""),
            self.runtime.get("fallback_checkpoint_id", ""),
        )


SRole.actions = {
    "init_runtime": SRole.act_init_runtime,
    "build_enc_db": SRole.act_build_enc_db,
    "pir_prg_setup": SRole.act_pir_prg_setup,
    "head_forward": SRole.act_head_forward,
    "fetch_rows": SRole.act_fetch_rows,
    "share_compute": SRole.act_share_compute,
    "rms_parity": SRole.act_rms_parity,
    "db_download": SRole.act_db_download,
    "val": SRole.act_val,
    "shutdown": SRole.act_shutdown,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9003)
    ap.add_argument("--node_id", default="srv-s-01")
    ap.add_argument("--db_dir", default=str(Path(__file__).resolve().parent / "db"))
    ap.add_argument("--model_path", default=os.environ.get("CF_MODEL_PATH", ""))
    ap.add_argument("--device", default=os.environ.get("CF_PARTY_DEVICE", ""))
    ap.add_argument("--metrics_dir", default=str(ROOT / "coordinator" / "logs"))
    ap.add_argument("--presets_json", default=str(ROOT / "data" / "fixtures" / "eval-presets.json"))
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("CF_MODEL_PATH must be set (--model_path)")
    if args.device:
        os.environ["CF_PARTY_DEVICE"] = args.device
    serve(
        SRole(args.node_id, {
            "db_dir": args.db_dir,
            "model_path": args.model_path,
            "metrics_dir": args.metrics_dir,
            "presets_json": args.presets_json,
        }),
        args.host, args.port,
    )


if __name__ == "__main__":
    main()
