"""RemoteProtocol：U（协调者）侧驱动，通过 HTTP 调用 M/S 节点完成训练。

与单机版 ``HeterogeneousProtocol`` 暴露相同的 Trainer 接口：
``step_train_chunked / step_train / remote_val / gather_checkpoints /
load_checkpoints / shutdown``。标签值只在本进程（U）内使用，S 只收到
监督位置与 PIR 查询块。
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .parties.wire import StepResult

logger = logging.getLogger("remote_protocol")


def _b64_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _tensor_to_b64(t: torch.Tensor) -> str:
    return _b64_bytes(np.ascontiguousarray(t.detach().cpu().float().numpy()).tobytes())


def _tensor_from_b64(s: str, shape=None, device: Optional[str] = None) -> torch.Tensor:
    arr = np.frombuffer(_unb64(s), dtype=np.float32)
    return torch.from_numpy(arr.copy()).view(shape if shape is not None else (-1,)).to(device)


def _tensordict_to_b64(d: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        buf = __import__("io").BytesIO()
        torch.save(v, buf)
        out[k] = {"b64": _b64_bytes(buf.getvalue()), "dtype": str(v.dtype), "shape": list(v.shape)}
    return out


def _tensordict_from_b64(d: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        buf = _unb64(v["b64"])
        out[k] = torch.load(
            __import__("io").BytesIO(buf), map_location="cpu", weights_only=False,
        )
    return out


class RemoteClient:
    """HTTP/JSON client for a node (/v1/action)."""

    def __init__(self, url: str, timeout: float = 90.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def action(self, trace_id: str, stage: str, step: int, action: str,
               params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "protocol_version": "1.0",
            "trace_id": trace_id,
            "stage": stage,
            "step": step,
            "action": action,
            "params": params or {},
        }
        req = urllib.request.Request(
            self.url + "/v1/action",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def hello(self) -> Dict[str, Any]:
        with urllib.request.urlopen(self.url + "/v1/hello", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


class _RemoteSFetcher:
    """Adapter so PartyU.pir_query_mask can fetch PIR blocks from remote S."""

    def __init__(self, s_client: RemoteClient, trace_id: str) -> None:
        self._s = s_client
        self._trace = trace_id

    def pir_fetch_dispatch(self, indices: List[int], step: int = 0) -> Dict[int, bytes]:
        # Deduplicate the request: a block may legally contain repeated row
        # indices (weighted dummy sampling with replacement).  Fetching each
        # unique row once keeps the S->U payload proportional to the *unique*
        # rows instead of the raw position count (block=64 would otherwise
        # multiply bandwidth by the duplicate factor).
        unique = list(dict.fromkeys(int(i) for i in indices))
        r = self._s.action(self._trace, "BACKWARD", int(step), "fetch_rows", {"indices": unique})
        if not r.get("ok"):
            raise RuntimeError(f"fetch_rows failed: {r.get('error')}")
        if "result" not in r:
            raise RuntimeError(f"fetch_rows returned no result (cached?): {r}")
        rows_b64 = r["result"].get("rows") or []
        if len(rows_b64) != len(unique):
            raise RuntimeError(f"fetch_rows returned {len(rows_b64)} rows for {len(unique)} indices")
        return {idx: _unb64(b) for idx, b in zip(unique, rows_b64)}

    def action(self, action: str, params: Dict[str, Any],
               stage: str = "BACKWARD", step: int = 0) -> Dict[str, Any]:
        """Generic S action passthrough (used by the RMS-PIR parity oracle)."""
        r = self._s.action(self._trace, stage, int(step), action, params or {})
        if not r.get("ok"):
            raise RuntimeError(f"{action} failed: {r.get('error')}")
        return r.get("result") or {}


class RemoteProtocol:
    """Trainer-facing remote driver (runs on the U/coordinator process)."""

    def __init__(
        self,
        party_u,
        m_url: str,
        s_url: str,
        config: Dict[str, Any],
        trace_id: str,
        prg_seed: bytes = b"",
    ) -> None:
        self.party_u = party_u
        self.config = config
        self.trace_id = trace_id
        self.prg_seed = prg_seed
        timeout = float(config.get("http_timeout_s", 300.0))
        self.m = RemoteClient(m_url, timeout=timeout)
        self.s = RemoteClient(s_url, timeout=timeout)
        # PartyU.pir_query_mask calls ``_s_ref.pir_fetch_dispatch``
        self.party_u._s_ref = _RemoteSFetcher(self.s, trace_id)

    # ------------------------------------------------------------------ #
    #  Label handling (U-side only)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _binary_ce_from_p(gold_yes, p_yes) -> Optional[float]:
        if not gold_yes or not p_yes or len(gold_yes) != len(p_yes):
            return None
        total = 0.0
        for g, p in zip(gold_yes, p_yes):
            p = min(max(float(p), 1e-7), 1.0 - 1e-7)
            total += -(g * math.log(p) + (1.0 - g) * math.log(1.0 - p))
        return total / max(len(gold_yes), 1)

    def _prepare_labels(self, batch: Dict) -> Dict:
        lab = batch.get("output_ids") if isinstance(batch, dict) else None
        out = {
            "y_valid": [], "valid_indices": [], "valid_mask": None,
            "monitor_positions": None, "gold_yes": None,
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
        yes_id = int(self.config.get("yes_token_id", -1))
        gold_pos = (first + 1).clamp(max=S - 1)
        gold_tok = lab.gather(1, gold_pos.unsqueeze(1)).squeeze(1)
        gold_yes = (gold_tok == yes_id).float().tolist()
        cls_map = self.config.get("answer_token_to_class") or {}
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

    # ------------------------------------------------------------------ #
    #  Training step
    # ------------------------------------------------------------------ #
    def step_train(self, batch: Dict, global_step: int) -> StepResult:
        return self.step_train_chunked(batch, global_step, chunk_tokens=2**30)

    def step_train_chunked(
        self, batch: Dict, global_step: int, chunk_tokens: int = 3072,
    ) -> StepResult:
        t0 = time.time()
        if global_step % 50 == 0:
            try:
                rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1048576
            except Exception:
                rss = -1
            gpu = 0
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1048576
            logger.info("coordinator RSS=%.2fGB GPU=%.0fMB (step %d)", rss, gpu, global_step)
        labels_info = self._prepare_labels(batch)
        valid_indices = labels_info["valid_indices"]
        if not valid_indices:
            raise RuntimeError(f"step {global_step}: no valid answer tokens")
        y_valid = labels_info["y_valid"]
        valid_mask = labels_info["valid_mask"]
        gold_yes = labels_info["gold_yes"]
        monitor_positions = labels_info["monitor_positions"]

        # ---- U forward (local) ----
        u_result = self.party_u.forward_train(batch)
        H_U = u_result["H_U"]

        # ---- M trunk_forward ----
        attn = batch.get("attention_mask") if isinstance(batch, dict) else None
        r = self.m.action(
            self.trace_id, "FORWARD", global_step, "trunk_forward", {
                "H_U": _tensor_to_b64(H_U),
                "H_U_shape": list(H_U.shape),
                "attention_mask": attn.tolist() if attn is not None else None,
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"trunk_forward failed: {r.get('error')}")
        H_M = _tensor_from_b64(r["result"]["H_M"], r["result"].get("H_M_shape"))

        # ---- S head_forward (label-free, cache a_t) ----
        r = self.s.action(
            self.trace_id, "FORWARD", global_step, "head_forward", {
                "H_M": _tensor_to_b64(H_M),
                "H_M_shape": list(H_M.shape),
                "monitor_positions": monitor_positions,
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"head_forward failed: {r.get('error')}")
        p_yes = r["result"].get("monitor_p_yes")

        # ---- U real block PIR + mask ----
        block_size = int(self.config.get("pir_block_size", 8))
        ct_list = self.party_u.pir_query_mask(
            self.party_u._s_ref, y_valid, valid_indices, global_step, block_size,
        )

        # ---- S share_compute ----
        r = self.s.action(
            self.trace_id, "BACKWARD", global_step, "share_compute",
            {"positions": valid_indices, "step": global_step},
        )
        if not r.get("ok"):
            raise RuntimeError(f"share_compute failed: {r.get('error')}")
        s_shares_raw = _unb64(r["result"]["s_shares_b64"])
        s_shares = np.frombuffer(s_shares_raw, dtype=np.int64).reshape(-1, 2048).tolist()

        # ---- M reconstruct + update ----
        r = self.m.action(
            self.trace_id, "RECONSTRUCT", global_step, "grad_reconstruct", {
                "cts": [_b64_bytes(c) for c in ct_list],
                "s_shares": s_shares,
                "valid_indices": valid_indices,
                "expected_shape": [
                    int(batch["input_ids"].shape[0]),
                    int(batch["input_ids"].shape[1]),
                ],
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"grad_reconstruct failed: {r.get('error')}")
        r = self.m.action(self.trace_id, "UPDATE", global_step, "lora_update", {})
        if not r.get("ok"):
            raise RuntimeError(f"lora_update failed: {r.get('error')}")

        if str(self.config.get("eval_mode", "binary")) == "binary":
            monitor_ce = self._binary_ce_from_p(gold_yes, p_yes)
        else:
            monitor_ce = None
        metrics = r.get("metrics") or {}
        logger.info(
            "step %d: g_absmax=%.4g g_meanabs=%.4g",
            global_step, metrics.get("g_absmax", -1), metrics.get("g_meanabs", -1),
        )
        return StepResult(
            step=global_step,
            loss=float(metrics.get("loss", 0.0)),
            gpu_mem_mb=float(metrics.get("gpu_mem_mb", 0.0)),
            step_time_ms=(time.time() - t0) * 1000,
            attack_dumps={},
            n_chunks=1,
            dp_audit={},
            loss_ce=monitor_ce,
        )

    # ------------------------------------------------------------------ #
    #  Validation (ClinVar: AUPRC/AUC/acc/per-gene, baseline-aligned)
    # ------------------------------------------------------------------ #
    # Forward pass for validation. Mirrors ``baseline/evaluate_auprc.py``:
    #   last_pos = attention_mask.sum(dim=1) - 1
    #   logits = H_M[last_pos] @ V.T          (full vocab)
    #   P(Yes) = softmax([logits[yes], logits[no]])
    # U receives the full vocab logits and computes every metric the same way
    # the plaintext baseline does. PIR mode (block / RMS) does NOT affect the
    # forward path — both go through S's cleartext V matrix at val time, just
    # like during training the gradient PIR is independent of the val forward.
    def remote_val(
        self,
        batch: Dict,
        *,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        lab = batch["output_ids"].to("cpu")
        attn = batch.get("attention_mask")
        if attn is None:
            raise RuntimeError("remote_val requires attention_mask in batch")
        attn_t = attn if isinstance(attn, torch.Tensor) else torch.as_tensor(attn)
        attn_cpu = attn_t.to("cpu").long()

        # Gold Yes/No is at ``first + 1`` (▁ + Yes/No). The model is trained
        # (y_shift semantics) to predict Yes/No AT the ▁ position ``first``
        # (logits[first] targets lab[first+1]), so the scoring position must
        # be ``first`` — scoring at ``last_pos`` (= first+1) is off by one and
        # produces inverted/random AUPRC.
        first = (lab != -100).long().argmax(dim=1)          # (B,) ▁ position
        score_pos = first                                   # (B,)

        # ---- U forward (local, same as training) ----
        u_result = self.party_u.forward_val(batch)
        H_U = u_result["H_U"]
        r = self.m.action(
            self.trace_id, "EVAL", 0, "val_forward", {
                "H_U": _tensor_to_b64(H_U),
                "H_U_shape": list(H_U.shape),
                "attention_mask": attn_cpu.tolist(),
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"val_forward failed: {r.get('error')}")
        H_M = _tensor_from_b64(r["result"]["H_M"], r["result"].get("H_M_shape"))
        # ---- S head logits (full vocab at last_pos) ----
        r = self.s.action(
            self.trace_id, "EVAL", 0, "val", {
                "H_M": _tensor_to_b64(H_M),
                "H_M_shape": list(H_M.shape),
                "attention_mask": attn_cpu.tolist(),
                "monitor_positions": score_pos.tolist(),
                # Legacy keys for backward-compat (ignored by the new S contract):
                "positions": None,
            },
        )
        if not r.get("ok"):
            raise RuntimeError(f"S val failed: {r.get('error')}")
        res = r["result"]
        logits_b64 = res.get("logits_b64")
        if logits_b64 is None:
            raise RuntimeError("S val did not return logits_b64 — please restart S")
        shape = res.get("logits_shape") or [attn_cpu.size(0), int(self.config.get("vocab_size", 32000))]
        logits_last = torch.from_numpy(
            np.frombuffer(_unb64(logits_b64), dtype=np.float32).copy()
        ).view(*shape).float()                              # (B, V)

        # ---- U: baseline-identical softmax P(Yes) + AUPRC/AUC/per-gene ----
        yes_id = int(self.config.get("yes_token_id", -1))
        no_id = int(self.config.get("no_token_id", -1))
        scores = torch.stack([logits_last[:, yes_id], logits_last[:, no_id]], dim=1).float()
        probs = torch.softmax(scores, dim=1)[:, 0].cpu().tolist()
        preds = [1 if p >= 0.5 else 0 for p in probs]

        # Gold Yes/No flags (parse from output_ids at the supervised position).
        # We use the same convention as baseline: output is " Yes" or " No"
        # so the LAST non-(-100) token is the Yes/No id (the FIRST one is the
        # SentencePiece ▁ prefix token: 29871). Training's _prepare_labels
        # uses ``first + 1`` for the same reason — match it here.
        first = (lab != -100).long().argmax(dim=1)
        gold_pos = (first + 1).clamp(max=lab.size(1) - 1)
        gold_yn = lab.gather(1, gold_pos.unsqueeze(1)).squeeze(1)

        # ---- U: multiclass branch (BioTriplex 7/21-class letter QA) ----
        if str(self.config.get("eval_mode", "binary")) == "multiclass":
            class_token_ids = [int(t) for t in (self.config.get("class_token_ids") or [])]
            if class_token_ids:
                cls_map = self.config.get("answer_token_to_class") or {}
                class_logits = logits_last[:, class_token_ids].float()  # (B, C)
                pred_class = class_logits.argmax(dim=1).tolist()
                gold_class = [int(cls_map.get(str(int(g)), -1)) for g in gold_yn.tolist()]
                return {
                    "class_logits": class_logits.cpu().tolist(),
                    "pred_class_ids": pred_class,
                    "gold_class_ids": gold_class,
                    "class_token_ids": class_token_ids,
                    "n": len(pred_class),
                }
        gold = [1 if int(g) == yes_id else 0 for g in gold_yn]

        # Per-sample outputs (for the caller to assemble AUPRC/per-gene later).
        out: Dict[str, Any] = {
            "probs": probs,
            "preds": preds,
            "gold": gold,
            "yes_id": yes_id,
            "no_id": no_id,
            "n": len(probs),
        }
        if not return_probs:
            return out
        return out

    def step_val(self, batch: Dict, global_step: int) -> Dict:
        # Compatibility stub: Trainer._run_val_clinvar is overridden by
        # RemoteTrainer to use remote_val(); this path is unreachable in
        # practice.
        return {"predictions": [], "labels": [], "logits": None, "labels_tensor": None}

    # ------------------------------------------------------------------ #
    #  Checkpoints
    # ------------------------------------------------------------------ #
    def gather_checkpoints(self) -> Dict:
        r = self.m.action(self.trace_id, "EVAL", 0, "gather_checkpoint", {})
        if not r.get("ok"):
            raise RuntimeError(f"gather_checkpoint failed: {r.get('error')}")
        m_ckpt = {
            "party": "M",
            "lora_state": _tensordict_from_b64(r["result"].get("lora_state_b64") or {}),
            "optimizer_state": torch.load(
                __import__("io").BytesIO(_unb64(r["result"]["optimizer_state_b64"])),
                map_location="cpu", weights_only=False,
            ) if r["result"].get("optimizer_state_b64") else {},
            "scheduler_state": torch.load(
                __import__("io").BytesIO(_unb64(r["result"]["scheduler_state_b64"])),
                map_location="cpu", weights_only=False,
            ) if r["result"].get("scheduler_state_b64") else {},
        }
        return {
            "U": self.party_u.save_checkpoint(),
            "M": m_ckpt,
            "S": {"party": "S"},
        }

    def load_checkpoints(self, checkpoint_dir: str, ckpt_path: str = None) -> None:
        if ckpt_path is None:
            ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        if not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m_ckpt = (ckpt.get("party_checkpoints") or {}).get("M") or {}
        lora_state = m_ckpt.get("lora_state") or {}
        opt_state = m_ckpt.get("optimizer_state") or {}
        sch_state = m_ckpt.get("scheduler_state") or {}
        opt_buf = __import__("io").BytesIO()
        torch.save(opt_state, opt_buf)
        sch_buf = __import__("io").BytesIO()
        torch.save(sch_state, sch_buf)
        self.m.action(self.trace_id, "EVAL", 0, "load_checkpoint", {
            "lora_state_b64": _tensordict_to_b64(lora_state),
            "optimizer_state_b64": _b64_bytes(opt_buf.getvalue()),
            "scheduler_state_b64": _b64_bytes(sch_buf.getvalue()),
        })

    def shutdown(self) -> None:
        for cli in (self.m, self.s):
            try:
                cli.action(self.trace_id, "EVAL", 0, "shutdown", {})
            except Exception:
                pass
