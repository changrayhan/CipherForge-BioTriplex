#!/usr/bin/env python3
"""M 节点服务进程：完整模型方逻辑（BFV 密钥、trunk_forward、梯度重建、LoRA 更新、
checkpoint 汇聚/加载）。私钥 sk_M 只在本进程内存中，永不出域。"""
import argparse
import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from shared.node_server import RoleHandler, eval_run_response, serve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("node-m")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


class MRole(RoleHandler):
    role = "M"
    assets = ["主干 Decoder（层11-21）+ LoRA", "BFV 私钥 sk_M（不出域）", "梯度重建 + LoRA 更新"]
    actions = {}

    def __init__(self, node_id: str, config: Dict[str, Any]) -> None:
        super().__init__(node_id, config)
        self.runtime: Dict[str, Any] = {}

    # ---- INIT -------------------------------------------------------------
    def act_bfv_keygen(self, session, params, stage, step, trace_id):
        keys_dir = Path(self.config["keys_dir"])
        keys_dir.mkdir(parents=True, exist_ok=True)
        sk_path, pk_path = keys_dir / "bfv_sk.bin", keys_dir / "bfv_pk.bin"
        p = params or {}
        wc = self.runtime.get("worker_config") or {}
        if sk_path.exists() and pk_path.exists():
            sk_pem = sk_path.read_bytes()
            pk_pem = pk_path.read_bytes()
            ev = [{"type": "crypto", "payload": {"msg": "BFV 密钥缓存命中"}}]
        else:
            from shared.core.bfv_privselect_v2_adapter import (
                BFVPrivSelectV2Backend, _seal_to_bytes,
            )
            backend = BFVPrivSelectV2Backend(
                n_entries=int(p.get("vocab_size", wc.get("vocab_size", 32000))),
                vec_dim=int(p.get("hidden_dim", wc.get("hidden_dim", 2048))),
                shared_seed=os.urandom(32), cache_dir=str(keys_dir),
                poly_degree=int(p.get("poly_degree", wc.get("poly_degree", 4096))),
                plain_bits=int(p.get("plain_bits", wc.get("plain_bits", 30))),
                scale=int(p.get("scale", wc.get("scale", 10000))),
            )
            sk_pem = _seal_to_bytes(backend._secret_key)
            pk_pem = backend.public_key_bytes
            sk_path.write_bytes(sk_pem)
            pk_path.write_bytes(pk_pem)
            os.chmod(sk_path, 0o600)
            ev = [{"type": "crypto", "payload": {
                "msg": "BFV.KeyGen 完成",
                "poly_degree": int(p.get("poly_degree", wc.get("poly_degree", 4096))),
            }}]
        self.runtime["sk_pem"] = sk_pem
        self.runtime["pk_pem"] = pk_pem
        import hashlib
        return (ev, {}, {
            "pk_pem_b64": _b64(pk_pem),
            "pk_sha256": hashlib.sha256(pk_pem).hexdigest(),
        })

    def act_init_runtime(self, session, params, stage, step, trace_id):
        wc = params
        self.runtime["worker_config"] = wc
        # 论文拓扑：M -> S 直连（H_M / share_compute 控制），S -> M 直连（s_S）
        self.runtime["s_url"] = str(params.get("s_url") or wc.get("s_url") or "")
        model_path = self.config["model_path"]
        sk_pem = self.runtime.get("sk_pem")
        pk_pem = self.runtime.get("pk_pem")
        if sk_pem is None or pk_pem is None:
            keys_dir = Path(self.config["keys_dir"])
            sk_pem = (keys_dir / "bfv_sk.bin").read_bytes()
            pk_pem = (keys_dir / "bfv_pk.bin").read_bytes()
            self.runtime["sk_pem"] = sk_pem
            self.runtime["pk_pem"] = pk_pem

        from shared.parties.party_m import PartyM
        from shared.parties.crypto_workers.pool import CryptoWorkerPool
        from shared.parties.crypto_workers.crypto_m import CryptoMWorker

        import pickle
        pk_pem_pickled = pickle.dumps({"pk_bytes": pk_pem})

        party_m = PartyM(model_path, sk_pem, pk_pem, wc)
        pool = CryptoWorkerPool(
            CryptoMWorker,
            n_workers=int(wc.get("N_CRYPTO_M_WORKERS", 8)),
            init_kwargs={
                "bfv_sk_pem": sk_pem, "bfv_pk_pem": pk_pem_pickled,
                "poly_degree": int(wc["poly_degree"]),
                "plain_bits": int(wc["plain_bits"]),
                "scale": int(wc["scale"]),
                "vec_dim": int(wc["hidden_dim"]),
            },
        )
        party_m.crypto_m_pool = pool
        self.runtime["party_m"] = party_m
        self.runtime["pool"] = pool
        # L1 fallback 在 init_runtime 之前到达时，先把 LoRA 状态暂存，这里统一应用
        if self.runtime.get("pending_fallback"):
            party_m.model.load_state_dict(self.runtime["pending_fallback"], strict=False)
            logger.info("pending fallback LoRA applied at init_runtime")
            self.runtime.pop("pending_fallback", None)
        from shared.model.model_splitting import clear_safetensor_cache
        clear_safetensor_cache()
        # 密钥自检：M 自己的 pk 加密 → sk 解密应还原
        try:
            import numpy as np
            from seal import BatchEncoder, Ciphertext, Plaintext
            enc = BatchEncoder(party_m.bfv_backend._context)
            vec = np.zeros(int(wc["poly_degree"]), dtype=np.int64)
            vec[:3] = [1, 2, 3]
            pt0 = enc.encode(vec)
            ct0 = Ciphertext()
            party_m.bfv_backend._encryptor.encrypt(pt0, ct0)
            pt1 = Plaintext()
            party_m.bfv_backend._decryptor.decrypt(ct0, pt1)
            logger.info("M key self-check decode=%s", list(enc.decode(pt1))[:3])
        except Exception as exc:
            logger.warning("M key self-check failed: %s", exc)
        return ([{"type": "task", "payload": {"msg": "M runtime 就绪"}}], {}, {})

    def _s_client(self):
        if "s_client" not in self.runtime:
            s_url = self.runtime.get("s_url") or ""
            if not s_url:
                raise RuntimeError("M node requires s_url (S peer) for direct M->S edges")
            from shared.remote_protocol import RemoteClient
            self.runtime["s_client"] = RemoteClient(
                s_url,
                timeout=float(self.runtime["worker_config"].get("http_timeout_s", 300)),
            )
        return self.runtime["s_client"]

    # ---- FORWARD -----------------------------------------------------------
    def act_trunk_forward(self, session, params, stage, step, trace_id):
        import numpy as np
        party_m = self.runtime["party_m"]
        if torch.cuda.is_available():
            rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1048576
            logger.info(
                "M step=%d RSS=%.2fGB gpu_alloc=%.0fMB gpu_reserved=%.0fMB",
                step, rss, torch.cuda.memory_allocated() / 1048576,
                torch.cuda.memory_reserved() / 1048576,
            )
        H_U = torch.from_numpy(
            np.frombuffer(_unb64(params["H_U"]), dtype=np.float32).copy()
        ).view(*params["H_U_shape"]).to(torch.bfloat16).to(party_m.device)
        # 关键：reentrant checkpoint 要求输入 requires_grad=True，否则整段
        # 自动求导被跳过，LoRA 梯度恒为 None（模型冻结）。U 侧隐状态经网络
        # 传输后是叶子张量，需显式打开梯度。
        H_U = H_U.requires_grad_(True)
        attn = None
        if params.get("attention_mask"):
            attn = torch.tensor(params["attention_mask"], dtype=torch.long, device=party_m.device)
        m_result = party_m.forward(H_U, attention_mask=attn)
        H_M = m_result["H_M"]
        if torch.cuda.is_available():
            logger.info("M gpu_mem after fwd: %.0fMB", torch.cuda.memory_allocated() / 1048576)
        # 先建 step_state：S 的 share_compute 会在本请求尚未返回时并发直推
        # s_S 到 receive_share，必须保证它写入的是同一个 dict（否则会被覆盖）。
        session["state"]["step_state"] = {
            "step": step, "p_yes": None,
            "cts": None, "valid_indices": None, "expected_shape": None,
            "s_shares": None,
        }
        # 论文拓扑：H_M 由 M 直发 S（head_forward），绝不再返回给 U。
        # S 计算 monitor p_yes 后，M 再请求 S 计算并直推 s_S = a_t - r_t 给 M。
        h_bytes = np.ascontiguousarray(H_M.detach().cpu().float().numpy()).tobytes()
        s_cli = self._s_client()
        r1 = s_cli.action(trace_id, "FORWARD", step, "head_forward", {
            "H_M": _b64(h_bytes),
            "H_M_shape": list(H_M.shape),
            "monitor_positions": params.get("monitor_positions"),
        })
        if not r1.get("ok"):
            raise RuntimeError(f"S head_forward failed: {r1.get('error')}")
        p_yes = r1["result"].get("monitor_p_yes")
        positions = [int(i) for i in (params.get("valid_indices") or [])]
        r2 = s_cli.action(trace_id, "BACKWARD", step, "share_compute", {
            "positions": positions,
            "step": step,
        })
        if not r2.get("ok"):
            raise RuntimeError(f"S share_compute failed: {r2.get('error')}")
        # 保存本步状态：等 U 的 C_U 和 S 的 s_S 都到齐后由 lora_update 组合
        session["state"]["step_state"]["p_yes"] = p_yes
        return (
            [{"type": "message", "payload": {"edge": "M->S", "bytes": len(h_bytes), "msg": "H_M 直发 S"}}],
            {},
            {"p_yes": p_yes},
        )

    def act_val_forward(self, session, params, stage, step, trace_id):
        import numpy as np
        party_m = self.runtime["party_m"]
        H_U = torch.from_numpy(
            np.frombuffer(_unb64(params["H_U"]), dtype=np.float32).copy()
        ).view(*params["H_U_shape"]).to(torch.bfloat16).to(party_m.device)
        attn = None
        if params.get("attention_mask"):
            attn = torch.tensor(params["attention_mask"], dtype=torch.long, device=party_m.device)
        # 验证必须在 eval 模式下进行：训练模式会让 LoRA dropout 生效，
        # 导致远程验证指标系统性低估。
        was_training = party_m.model.training
        party_m.model.eval()
        try:
            with torch.no_grad():
                # 验证路径直接走 model.forward（不走 reentrant checkpoint，
                # 避免 no_grad 下 checkpoint 语义问题），也不污染训练缓存。
                H_M = party_m.model.forward(H_U.to(party_m.device))
        finally:
            if was_training:
                party_m.model.train()
        party_m._last_H_U = None
        party_m._last_H_M = None
        party_m._last_attention_mask = None
        # 论文拓扑：M -> S 直发 H_M 做输出头前向；S 只返回类别 logits（B, C），
        # 绝不返回全词表 logits，也不返回 H_M。
        h_bytes = np.ascontiguousarray(H_M.detach().cpu().float().numpy()).tobytes()
        s_cli = self._s_client()
        r = s_cli.action(trace_id, "EVAL", 0, "val_head", {
            "H_M": _b64(h_bytes),
            "H_M_shape": list(H_M.shape),
            "positions": params.get("positions"),
        })
        if not r.get("ok"):
            raise RuntimeError(f"S val_head failed: {r.get('error')}")
        return (
            [{"type": "compute", "payload": {"msg": "val forward (class logits only)"}}], {},
            r["result"],
        )

    # ---- BACKWARD / RECONSTRUCT / UPDATE ------------------------------------
    def act_grad_reconstruct(self, session, params, stage, step, trace_id):
        """Receive C_U = Enc(-V_y + r_t) directly from U (paper edge U->M)."""
        st = session["state"].setdefault("step_state", {})
        st["cts"] = [_unb64(c) for c in params["cts"]]
        st["valid_indices"] = params.get("valid_indices")
        st["expected_shape"] = params.get("expected_shape")
        return (
            [{"type": "crypto", "payload": {"msg": f"梯度重建参数就绪: {len(params['cts'])} cts"}}],
            {},
            {"n_cts": len(params["cts"])},
        )

    def act_receive_share(self, session, params, stage, step, trace_id):
        """Receive s_S = scale·a_t - r_t directly from S (paper edge S->M)."""
        import numpy as np
        wc = self.runtime["worker_config"]
        s_shares_raw = _unb64(params["s_shares_b64"])
        n = int(params.get("n", 0))
        dim = int(wc["hidden_dim"])
        arr = np.frombuffer(s_shares_raw, dtype=np.int64).reshape(n, dim)
        st = session["state"].setdefault("step_state", {})
        st["s_shares"] = arr.tolist()
        return (
            [{"type": "message", "payload": {"edge": "S->M", "bytes": len(s_shares_raw), "msg": "s_S 直收"}}],
            {},
            {"n_shares": n},
        )

    def act_debug_decrypt(self, session, params, stage, step, trace_id):
        party_m = self.runtime["party_m"]
        cts = [_unb64(c) for c in params["cts"]]
        res = party_m.crypto_m_pool.submit({
            "ct_list": cts,
            "scale": int(self.runtime["worker_config"]["scale"]),
            "vec_dim": int(self.runtime["worker_config"]["hidden_dim"]),
        })
        dec = res["decrypted"]
        return ({}, {}, {
            "rows": [dec[i][:8].tolist() for i in range(len(cts))],
        })

    def act_lora_update(self, session, params, stage, step, trace_id):
        party_m = self.runtime["party_m"]
        st = session["state"].get("step_state") or {}
        if not st.get("cts"):
            raise RuntimeError("grad_reconstruct (C_U from U) must arrive before lora_update")
        if not st.get("s_shares"):
            raise RuntimeError("receive_share (s_S from S) must arrive before lora_update")
        ack = party_m.backward_and_update({
            "ct_from_U": st["cts"],
            "s_share": st["s_shares"],
            "valid_indices": st.get("valid_indices"),
            "expected_shape": tuple(st["expected_shape"]) if st.get("expected_shape") else None,
            "step": step,
        })
        session["state"].pop("step_state", None)
        # ---- T1 experiment capture (env-gated, evaluation ground truth) ----
        cap = os.environ.get("CF_M_CAPTURE_DIR", "")
        if cap:
            import json as _json
            os.makedirs(cap, exist_ok=True)
            lora_state = {}
            for name, p in party_m.model.named_parameters():
                if p.requires_grad:
                    lora_state[name] = p.detach().cpu().clone()
            torch.save(lora_state, os.path.join(cap, f"w_step_{step:05d}.pt"))
            with open(os.path.join(cap, "meta.jsonl"), "a", encoding="utf-8") as f:
                f.write(_json.dumps({
                    "step": int(step),
                    "loss": ack.get("loss"),
                    "g_absmax": ack.get("g_absmax"),
                    "g_meanabs": ack.get("g_meanabs"),
                }) + "\n")
        return (
            [{"type": "monitor", "payload": {"msg": "LoRA 更新完成", "loss": ack.get("loss")}}],
            {
                "loss": ack.get("loss", 0.0),
                "gpu_mem_mb": ack.get("gpu_mem_mb", 0.0),
                "step_ms": step,
                "g_absmax": ack.get("g_absmax", -1.0),
                "g_meanabs": ack.get("g_meanabs", -1.0),
            },
            {},
        )

    # ---- CHECKPOINT ----------------------------------------------------------
    def act_gather_checkpoint(self, session, params, stage, step, trace_id):
        party_m = self.runtime["party_m"]
        ck = party_m.save_checkpoint()
        lora_b64 = {
            k: {
                "b64": _b64((lambda buf: (torch.save(v, buf), buf.getvalue())[1])(io.BytesIO())),
                "dtype": str(v.dtype), "shape": list(v.shape),
            }
            for k, v in ck.get("lora_state", {}).items()
        }
        opt_buf = io.BytesIO()
        torch.save(ck.get("optimizer_state", {}), opt_buf)
        sch_buf = io.BytesIO()
        torch.save(ck.get("scheduler_state", {}), sch_buf)
        return ({}, {}, {
            "party": "M",
            "lora_state_b64": lora_b64,
            "optimizer_state_b64": _b64(opt_buf.getvalue()),
            "scheduler_state_b64": _b64(sch_buf.getvalue()),
        })

    def act_load_checkpoint(self, session, params, stage, step, trace_id):
        party_m = self.runtime["party_m"]
        lora_state = {
            k: torch.load(io.BytesIO(_unb64(v["b64"])), map_location="cpu", weights_only=False)
            for k, v in (params.get("lora_state_b64") or {}).items()
        }
        if lora_state:
            party_m.model.load_state_dict(lora_state, strict=False)
        if params.get("optimizer_state_b64"):
            opt = torch.load(io.BytesIO(_unb64(params["optimizer_state_b64"])), map_location="cpu", weights_only=False)
            party_m.optimizer.load_state_dict(opt)
        if params.get("scheduler_state_b64") and getattr(party_m, "lr_scheduler", None) is not None:
            sch = torch.load(io.BytesIO(_unb64(params["scheduler_state_b64"])), map_location="cpu", weights_only=False)
            party_m.lr_scheduler.load_state_dict(sch)
        return ([{"type": "task", "payload": {"msg": "LoRA 检查点已加载"}}], {}, {"loaded": True})

    def act_shutdown(self, session, params, stage, step, trace_id):
        try:
            self.runtime.get("pool") and self.runtime["pool"].close()
        except Exception:
            pass
        return ([{"type": "message", "payload": {"msg": "M 节点关闭"}}], {}, {})

    # ---- L1 fallback（docs/02 §3 / docs/04 §2.3）----
    def fallback_pretrained(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """M 节点：将 LoRA 参数替换为预微调检查点值（真实加载）。"""
        t0 = time.time()
        checkpoint_id = str(payload.get("checkpoint_id") or self.config.get("checkpoint_id", ""))
        ck_path = Path(
            self.config.get("fallback_checkpoint")
            or (Path(__file__).resolve().parent / "checkpoints" / "best_checkpoint.pt")
        )
        if not ck_path.exists():
            return self._error(500, f"fallback checkpoint not found: {ck_path}", False)
        try:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            return self._error(500, f"fallback checkpoint load failed: {exc}", False)
        m_ck = (ck.get("party_checkpoints") or {}).get("M") or {}
        lora_state = m_ck.get("lora_state") or {}
        if not lora_state:
            return self._error(500, "checkpoint has no party_checkpoints.M.lora_state", False)
        party_m = self.runtime.get("party_m")
        if party_m is not None:
            party_m.model.load_state_dict(lora_state, strict=False)
            opt = m_ck.get("optimizer_state")
            if opt:
                try:
                    party_m.optimizer.load_state_dict(opt)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fallback optimizer load skipped: %s", exc)
            sched = m_ck.get("scheduler_state")
            if sched and getattr(party_m, "lr_scheduler", None) is not None:
                try:
                    party_m.lr_scheduler.load_state_dict(sched)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fallback scheduler load skipped: %s", exc)
        else:
            self.runtime["pending_fallback"] = lora_state
        self.runtime["fallback_checkpoint_id"] = checkpoint_id
        load_ms = int((time.time() - t0) * 1000)
        logger.info("fallback pretrained loaded: %s (%d ms)", checkpoint_id, load_ms)
        return {"ok": True, "checkpoint": checkpoint_id, "loaded": True, "load_ms": load_ms}

    # ---- EVAL（docs/02 §4）----
    def eval_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return eval_run_response(
            str(payload.get("test_id", "")),
            self.config.get("metrics_dir", ""),
            self.config.get("presets_json", ""),
            self.runtime.get("fallback_checkpoint_id", ""),
        )


MRole.actions = {
    "bfv_keygen": MRole.act_bfv_keygen,
    "init_runtime": MRole.act_init_runtime,
    "trunk_forward": MRole.act_trunk_forward,
    "val_forward": MRole.act_val_forward,
    "grad_reconstruct": MRole.act_grad_reconstruct,
    "receive_share": MRole.act_receive_share,
    "debug_decrypt": MRole.act_debug_decrypt,
    "lora_update": MRole.act_lora_update,
    "gather_checkpoint": MRole.act_gather_checkpoint,
    "load_checkpoint": MRole.act_load_checkpoint,
    "shutdown": MRole.act_shutdown,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9002)
    ap.add_argument("--node_id", default="srv-m-01")
    ap.add_argument("--keys_dir", default=str(Path(__file__).resolve().parent / "keys"))
    ap.add_argument("--model_path", default=os.environ.get("CF_MODEL_PATH", ""))
    ap.add_argument("--metrics_dir", default=str(ROOT / "coordinator" / "logs"))
    ap.add_argument("--presets_json", default=str(ROOT / "data" / "fixtures" / "eval-presets.json"))
    ap.add_argument(
        "--fallback_checkpoint",
        default=str(Path(__file__).resolve().parent / "checkpoints" / "best_checkpoint.pt"),
    )
    args = ap.parse_args()
    if not args.model_path:
        raise SystemExit("CF_MODEL_PATH must be set (--model_path)")
    serve(
        MRole(args.node_id, {
            "keys_dir": args.keys_dir,
            "model_path": args.model_path,
            "metrics_dir": args.metrics_dir,
            "presets_json": args.presets_json,
            "fallback_checkpoint": args.fallback_checkpoint,
        }),
        args.host, args.port,
    )


if __name__ == "__main__":
    main()
