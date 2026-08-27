#!/usr/bin/env python3
"""U1a — protocol message-surface audit.

Verifies that U's captured responses never contain H_M, s_S, full-vocab
logits, or LoRA state (paper-faithful topology).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    log_path = run_dir / "captures" / "u" / "u_responses.jsonl"
    records = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    by = defaultdict(lambda: {
        "n": 0,
        "flags": {"has_H_M": False, "has_s_shares": False,
                  "has_lora_state": False, "has_full_logits": False},
        "keys": set(),
    })
    violations = []
    for rec in records:
        key = (rec.get("peer"), rec.get("action"))
        b = by[key]
        b["n"] += 1
        for k, v in (rec.get("flags") or {}).items():
            b["flags"][k] = b["flags"][k] or bool(v)
        b["keys"].update(rec.get("result_keys") or [])
        if any((rec.get("flags") or {}).values()):
            violations.append(rec)

    table = []
    for (peer, action), b in sorted(by.items()):
        row = {
            "peer": peer, "action": action, "n": b["n"],
            "result_keys": sorted(b["keys"]),
            "flags": {k: bool(v) for k, v in b["flags"].items()},
        }
        table.append(row)
        print(f"{peer:>2} {action:<20} n={b['n']:<4} "
              f"H_M={row['flags']['has_H_M']} s_S={row['flags']['has_s_shares']} "
              f"full_logits={row['flags']['has_full_logits']} "
              f"lora={row['flags']['has_lora_state']}")

    verdict = "PASS" if not violations else "FAIL"
    out = {
        "u1a_audit": {
            "total_records": len(records),
            "by_action": table,
            "n_violations": len(violations),
            "violations": violations[:20],
            "verdict": verdict,
        }
    }
    (run_dir / "u1a_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nU1a verdict: {verdict} (records={len(records)}, violations={len(violations)})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
