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
        wc = dict(params)
        # 论文拓扑：S -> M 直连（s_S 份额直推 M）
        self.runtime["m_url"] = str(wc.pop("m_url", "") or "")
        self.runtime["worker_config"] = wc
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
        if params.get("m_url"):
            self.runtime["m_url"] = str(params.get("m_url"))
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
        # ---- T1 experiment capture (env-gated, evaluation ground truth) ----
        cap = os.environ.get("CF_S_CAPTURE_DIR", "")
        if cap:
            import hashlib as _hashlib
            os.makedirs(cap, exist_ok=True)
            V = party_s.V_weight.detach().cpu().float()
            class_ids = [int(t) for t in (wc.get("class_token_ids") or [])]
            if not class_ids:
                yes_id = int(wc.get("yes_token_id", -1))
                no_id = int(wc.get("no_token_id", -1))
                class_ids = [yes_id, no_id]
            v_rows = V[class_ids].numpy().astype(np.float32)  # (C, H)
            np.savez(
                os.path.join(cap, "v_rows.npz"),
                v_rows=v_rows,
                class_token_ids=np.asarray(class_ids, dtype=np.int64),
                v_hash=_hashlib.sha256(V.numpy().tobytes()).hexdigest(),
            )
        return ([{"type": "pir", "payload": {"msg": "PRG 种子已就绪（仅 U/S 共享）"}}], {}, {})

    def _m_client(self):
        if "m_client" not in self.runtime:
            m_url = self.runtime.get("m_url") or ""
            if not m_url:
                raise RuntimeError("S node requires m_url (M peer) for direct S->M edges")
            from shared.remote_protocol import RemoteClient
            self.runtime["m_client"] = RemoteClient(
                m_url,
                timeout=float(self.runtime["worker_config"].get("http_timeout_s", 300)),
            )
        return self.runtime["m_client"]

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
        # 论文拓扑：s_S = scale·a_t - r_t 由 S 直推 M（receive_share），
        # 绝不经过 U / coordinator。
        r = self._m_client().action(
            trace_id, "RECONSTRUCT", int(params.get("step", step)),
            "receive_share",
            {"s_shares_b64": _b64(shares.tobytes()), "n": len(positions)},
        )
        if not r.get("ok"):
            raise RuntimeError(f"M receive_share failed: {r.get('error')}")
        return (
            [{"type": "message", "payload": {"edge": "S->M", "bytes": shares.nbytes,
                                             "msg": "份额已发送"}}],
            {},
            {"pushed": True, "n": len(positions)},
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
    def act_val_head(self, session, params, stage, step, trace_id):
        """Validation-time output-head forward: return ONLY the class logits.

        Paper-faithful: U never receives full-vocab logits (which would allow
        linear recovery of V).  S projects H_M at the scoring positions onto
        the task's class token ids and returns a (B, C) float32 tensor.
        """
        import numpy as np
        H_M = torch.from_numpy(
            np.frombuffer(_unb64(params["H_M"]), dtype=np.float32).copy()
        ).view(*params["H_M_shape"])
        positions = params.get("positions")
        if not positions:
            raise RuntimeError("val_head requires positions (score positions)")
        wc = self.runtime["worker_config"]
        class_ids = [int(t) for t in (wc.get("class_token_ids") or [])]
        if not class_ids:
            yes_id = int(wc.get("yes_token_id", -1))
            no_id = int(wc.get("no_token_id", -1))
            class_ids = [yes_id, no_id]
        z = self._logits_at(H_M, [int(i) for i in positions]).float()  # (B, V)
        z_cls = z[:, class_ids].contiguous()                            # (B, C)
        logits_b64 = _b64(np.ascontiguousarray(z_cls.numpy()).tobytes())
        return (
            [{"type": "compute", "payload": {"msg": "val_head: class logits only"}}],
            {},
            {
                "logits_b64": logits_b64,
                "logits_shape": list(z_cls.shape),
                "class_token_ids": class_ids,
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
    "val_head": SRole.act_val_head,
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
