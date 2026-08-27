"""Zero-dependency HTTP node server for the three-party CipherForge runtime.

Implements the node-side contract from docs/02-节点接口规范.md:
  GET  /v1/hello                 — 握手与能力协商
  POST /v1/action                — 执行协议动作（唯一计算入口）
  POST /v1/fallback/pretrained   — 加载预微调检查点（L1）
  POST /v1/eval/run              — 评测接口

All responses use the unified envelope ``{ok, events, metrics, result}``.
Idempotency: ``trace_id + stage + step + action`` — a retried action returns
the cached result instead of re-executing (docs/05 §6).
"""
from __future__ import annotations

import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("node")

PROTOCOL_VERSION = "1.0"

REQUIRED_EVAL_TESTS = ("acc", "auprc", "macro_f1")


def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def eval_run_response(
    test_id: str,
    metrics_dir: str = "",
    presets_path: str = "",
    checkpoint_id: str = "",
) -> Dict[str, Any]:
    """Build the docs/02 §4 /v1/eval/run response.

    Real metrics first: ``metrics_dir/clinvar_auprc.json`` (after) and
    ``metrics_dir/clinvar_auprc_before.json`` or the bundled zero-shot fixture
    (before).  Non-required test_ids fall back to presets
    (``data/fixtures/eval-presets.json``).
    """
    after = _read_json_file(os.path.join(metrics_dir, "clinvar_auprc.json")) if metrics_dir else None
    before = _read_json_file(os.path.join(metrics_dir, "clinvar_auprc_before.json")) if metrics_dir else None
    if before is None and presets_path:
        fixtures_dir = os.path.dirname(os.path.abspath(presets_path))
        before = _read_json_file(os.path.join(fixtures_dir, "clinvar-zeroshot-metrics.json"))

    mapping = {
        "acc": ("accuracy@0.5", "准确率（acc）", "%"),
        "auprc": ("auprc", "AUPRC", "%"),
        "macro_f1": ("macro_f1", "Macro-F1", "%"),
    }
    if test_id in mapping and after is not None:
        key, label, unit = mapping[test_id]
        a_raw = after.get(key)
        if a_raw is None:
            return {
                "ok": False,
                "error": {
                    "code": 404,
                    "message": f"metric {key!r} missing in {after.get('name', 'metrics file')}",
                    "retryable": False,
                },
            }

        def _pct(v: Any) -> Optional[float]:
            try:
                return round(float(v) * 100.0, 2)
            except (TypeError, ValueError):
                return None

        a = _pct(a_raw)
        b_raw = before.get(key) if before else None
        b = _pct(b_raw) if b_raw is not None else None
        metrics: List[Dict[str, Any]] = []
        if b is not None:
            metrics.append({"key": "before", "label": "微调前", "value": b, "unit": unit})
        metrics.append({"key": "after", "label": "微调后", "value": a, "unit": unit})
        if b is not None:
            conclusion = f"{label}：微调前 {b}{unit} → 微调后 {a}{unit}，提升 +{a - b:.1f}pp"
        else:
            conclusion = f"{label}：微调后 {a}{unit}"
        result: Dict[str, Any] = {"status": "DONE", "metrics": metrics, "conclusion": conclusion}
        if checkpoint_id:
            result["checkpoint_id"] = checkpoint_id
        return {"ok": True, "result": result}

    if presets_path:
        presets = _read_json_file(presets_path) or {}
        tests = presets.get("tests") if isinstance(presets, dict) else None
        preset = (tests or {}).get(test_id) if isinstance(tests, dict) else None
        if preset:
            return {
                "ok": True,
                "result": {
                    "status": preset.get("status", "DONE"),
                    "metrics": preset.get("metrics", []),
                    "conclusion": preset.get("conclusion", "预设数据回放"),
                },
            }
    return {"ok": False, "error": {"code": 404, "message": f"unknown test_id: {test_id}", "retryable": False}}


class SessionStore:
    """Per-trace_id session state: current step, completed actions, caches."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get(self, trace_id: str) -> Dict[str, Any]:
        if trace_id not in self._sessions:
            self._sessions[trace_id] = {
                "trace_id": trace_id,
                "step": 0,
                "completed": {},   # (stage,step,action) -> result
                "state": {},       # role-specific caches (H_U/H_M/a_t...)
            }
        return self._sessions[trace_id]

    def set_step(self, trace_id: str, step: int) -> None:
        self.get(trace_id)["step"] = int(step)


class RoleHandler:
    """Base class for U/M/S role logic.

    Subclasses implement ``actions``: a dict ``action_id -> callable(session,
    params, stage, step, trace_id) -> (events, metrics, result)``.
    The framework wraps the return values into the protocol envelope and
    enforces idempotency + 脱敏 (nodes must never emit raw labels/keys/seed).
    """

    role = "?"
    assets: List[str] = []
    capabilities: List[str] = ["action", "fallback", "eval"]
    actions: Dict[str, Callable[..., Any]] = {}
    # Data-returning actions are not cached for idempotency (their response IS
    # the payload; a cached marker would break the caller on retry).
    no_cache_actions = {
        "trunk_forward", "val_forward", "head_forward",
        "fetch_rows", "share_compute", "rms_parity", "db_download",
        "val", "val_head", "receive_share", "gather_checkpoint",
        "run_eval",
    }

    def __init__(self, node_id: str, config: Dict[str, Any]) -> None:
        self.node_id = node_id
        self.config = config
        self.sessions = SessionStore()

    # ---- protocol endpoints -------------------------------------------------
    def hello(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "node_id": self.node_id,
            "role": self.role,
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": self.capabilities,
            "assets": self.assets,
            "status": "idle",
        }

    def action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        protocol_version = payload.get("protocol_version")
        if protocol_version and protocol_version.split(".")[0] != PROTOCOL_VERSION.split(".")[0]:
            return self._error(426, f"protocol major mismatch: {protocol_version}", False)
        trace_id = str(payload.get("trace_id", ""))
        stage = str(payload.get("stage", ""))
        step = int(payload.get("step", 0))
        action = str(payload.get("action", ""))
        params = payload.get("params") or {}
        if not trace_id:
            return self._error(400, "trace_id is required", True)
        session = self.sessions.get(trace_id)
        self.sessions.set_step(trace_id, step)

        idem_key = (stage, step, action)
        if idem_key in session["completed"]:
            return session["completed"][idem_key]

        handler = self.actions.get(action)
        if handler is None:
            return self._error(501, f"action not implemented: {action}", False)
        try:
            events, metrics, result = handler(self, session, params, stage, step, trace_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("action %s failed", action)
            return self._error(500, f"{action} failed: {exc}", True)

        resp = {
            "ok": True,
            "events": sanitize_events(events),
            "metrics": metrics or {},
            "result": sanitize_result(result),
        }
        if action not in self.no_cache_actions:
            # 幂等缓存只存轻量标记，不缓存完整响应（否则大数据动作会让节点
            # RSS 线性增长直至 OOM）。
            session["completed"][idem_key] = {"ok": True, "cached": True}
        return resp

    def fallback_pretrained(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._error(501, "fallback not implemented on this node", False)

    def eval_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._error(501, "eval not implemented on this node", False)

    @staticmethod
    def _error(code: int, message: str, retryable: bool) -> Dict[str, Any]:
        return {"ok": False, "error": {"code": code, "message": message, "retryable": retryable}}


def sanitize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """脱敏铁律（docs/03 §6）：节点上报只允许 ID/哈希/尺寸/耗时/误差/统计。"""
    blocked = {"raw_text", "label", "sk_M", "prg_seed", "mask_vector", "head_row", "weights"}
    out = []
    for ev in events or []:
        e = dict(ev)
        p = dict(e.get("payload") or {})
        for k in list(p.keys()):
            if k.lower() in blocked:
                p[k] = "***"
        e["payload"] = p
        out.append(e)
    return out


def sanitize_result(result: Any) -> Any:
    if isinstance(result, dict):
        blocked = {"raw_text", "label", "sk_M", "prg_seed", "mask_vector", "head_row", "weights"}
        return {k: ("***" if k.lower() in blocked else sanitize_result(v)) for k, v in result.items()}
    if isinstance(result, list):
        return [sanitize_result(x) for x in result]
    return result


class PartyRequestHandler(BaseHTTPRequestHandler):
    handler: RoleHandler = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _send(self, obj: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/hello":
            self._send(self.handler.hello())
            return
        self._send(self.handler._error(404, f"not found: {self.path}", False), 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        payload = self._read_json()
        if path == "/v1/action":
            t0 = time.time()
            resp = self.handler.action(payload)
            logger.info(
                "[%s] action=%s stage=%s step=%s ok=%s dt=%.1fs",
                self.handler.role, payload.get("action"), payload.get("stage"),
                payload.get("step"), resp.get("ok"), time.time() - t0,
            )
            self._send(resp)
        elif path == "/v1/fallback/pretrained":
            self._send(self.handler.fallback_pretrained(payload))
        elif path == "/v1/eval/run":
            self._send(self.handler.eval_run(payload))
        else:
            self._send(self.handler._error(404, f"not found: {self.path}", False), 404)


def serve(role_handler: RoleHandler, host: str, port: int) -> None:
    PartyRequestHandler.handler = role_handler
    server = ThreadingHTTPServer((host, port), PartyRequestHandler)
    logger.info(
        "[%s] node %s listening on http://%s:%d (protocol %s)",
        role_handler.role, role_handler.node_id, host, port, PROTOCOL_VERSION,
    )
    server.serve_forever()
