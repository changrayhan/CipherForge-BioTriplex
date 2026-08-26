#!/usr/bin/env python3
"""U 节点服务进程（数据方/协调者侧 PartyU 逻辑）。

实现 docs/02 节点接口规范：
  - GET  /v1/hello：握手与能力协商
  - POST /v1/action：pir_prg_setup（RMS-PIR v2 中 U 为 offline server，种子由
    coordinator 生成后在此登记；U 侧其余密码学动作运行在 coordinator 进程内）
  - POST /v1/fallback/pretrained：L1 兜底幂等确认（数据侧无需变更）
  - POST /v1/eval/run：读取 coordinator 产出的评测 JSON，返回 acc/auprc/macro_f1
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.node_server import RoleHandler, eval_run_response, serve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class URole(RoleHandler):
    role = "U"
    assets = [
        "ClinVar 标注数据（仅本节点）",
        "embed_tokens + 底部 Decoder",
        "PIR 客户端（real+dummy 查询块）",
        "PRG 种子 σ（与 S 共享）",
    ]

    def __init__(self, node_id: str, config: Dict[str, Any]) -> None:
        super().__init__(node_id, config)
        self.fallback_state: Dict[str, Any] = {}

    # ---- INIT -------------------------------------------------------------
    def act_pir_prg_setup(self, session, params, stage, step, trace_id):
        prg_seed_b64 = (params or {}).get("prg_seed_b64", "")
        session["state"]["prg_seed_b64"] = prg_seed_b64
        return (
            [{"type": "pir", "payload": {"msg": "PIR PRG 种子已登记（U 侧确认）"}}],
            {},
            {"registered": True},
        )

    # ---- L1 fallback ------------------------------------------------------
    def fallback_pretrained(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """U 节点幂等确认：数据侧无需变更，仅记录会话切换（docs/04 §2.3）。"""
        t0 = time.time()
        checkpoint_id = str(payload.get("checkpoint_id") or "unknown")
        self.fallback_state["checkpoint_id"] = checkpoint_id
        self.fallback_state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        load_ms = int((time.time() - t0) * 1000)
        return {"ok": True, "checkpoint": checkpoint_id, "loaded": True, "load_ms": load_ms}

    # ---- EVAL -------------------------------------------------------------
    def eval_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return eval_run_response(
            str(payload.get("test_id", "")),
            self.config.get("metrics_dir", ""),
            self.config.get("presets_json", ""),
            self.fallback_state.get("checkpoint_id", ""),
        )


URole.actions = {
    "pir_prg_setup": URole.act_pir_prg_setup,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--node_id", default="srv-u-01")
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--metrics_dir", default=str(ROOT / "coordinator" / "logs"))
    ap.add_argument("--presets_json", default=str(ROOT / "data" / "fixtures" / "eval-presets.json"))
    args = ap.parse_args()
    serve(
        URole(args.node_id, {
            "data_dir": args.data_dir,
            "metrics_dir": args.metrics_dir,
            "presets_json": args.presets_json,
        }),
        args.host, args.port,
    )


if __name__ == "__main__":
    main()
