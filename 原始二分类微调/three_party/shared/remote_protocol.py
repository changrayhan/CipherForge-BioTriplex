"""Slim HTTP client layer for the paper-faithful three-party runtime.

After the framework refactor the coordinator is an *independent control plane*:
U, M, and S are three separate parties that exchange protocol payloads directly
over HTTP, exactly as described in the TriadFT paper:

    U -> M : H_U (trunk_forward), C_U (grad_reconstruct)
    M -> S : H_M (head_forward / val_head) + share_compute control
    S -> M : s_S (receive_share)
    U <-> S: PIR (fetch_rows / rms_parity / db_download)

This module only provides the HTTP client plus tensor (de)serialization
helpers.  No H_M or s_S ever flows through the coordinator, and the
coordinator never receives a full-vocab logits tensor.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.request
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger("remote_protocol")


def _b64_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _tensor_to_b64(t: torch.Tensor) -> str:
    return _b64_bytes(np.ascontiguousarray(t.detach().cpu().float().numpy()).tobytes())


def _tensor_from_b64(s: str, shape=None, device: str | None = None) -> torch.Tensor:
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
        # indices (weighted dummy sampling with replacement).
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
