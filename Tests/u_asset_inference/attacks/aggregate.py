#!/usr/bin/env python3
"""Aggregate the four U1 sub-step results into report.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    def load(name):
        p = run_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    a1 = load("u1a_audit.json").get("u1a_audit", {})
    a2 = load("u1b_ciphertext.json").get("u1b_ciphertext", {})
    a3 = load("u1c_recover_v.json").get("u1c_recover_v", {})
    a4 = load("u1d_recover_weights.json").get("u1d_recover_weights", {})

    lines = [
        "# T1 实验报告（U 资产推断）",
        "",
        f"- run_dir: `{run_dir.name}`",
        f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## U1a 协议信息面审计",
        "",
        f"- 记录数: {a1.get('total_records')}，违规数: {a1.get('n_violations')}，"
        f"判定: **{a1.get('verdict')}**",
        "",
        "| peer | action | n | H_M | s_S | full_logits | lora |",
        "|---|---|---|:---:|:---:|:---:|:---:|",
    ]
    for row in a1.get("by_action", []):
        f = row["flags"]
        lines.append(
            f"| {row['peer']} | {row['action']} | {row['n']} | "
            f"{f['has_H_M']} | {f['has_s_shares']} | "
            f"{f['has_full_logits']} | {f['has_lora_state']} |")

    lines += [
        "",
        "## U1b 密文通道对照",
        "",
        f"- 行数: {a2.get('n_rows')}，样本数: {a2.get('n_samples')}",
        f"- 同行 L1 距离均值: {a2.get('same_row_l1_mean')}，异行: {a2.get('diff_row_l1_mean')}，"
        f"差距: {a2.get('same_vs_diff_gap')}",
        f"- 最近邻行链接恢复率: {a2.get('link_recovery_rate')}（机会水平: {a2.get('chance_link_rate')}）",
        f"- 判定: **{a2.get('verdict')}**",
        "",
        "## U1c 复合 oracle 恢复 V",
        "",
        f"- 样本: {a3.get('n_samples')} 查询（train {a3.get('n_train_queries')} / "
        f"heldout {a3.get('n_heldout_queries')}）",
        f"- 盲拟合: RE_F={a3.get('blind', {}).get('re_f')}，"
        f"行余弦={a3.get('blind', {}).get('row_cosine')}",
        "",
        "| 查询数 | RE_F(V) | 行余弦 | 功能一致率 | 功能相关 |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in a3.get("joint_fit_curve", []):
        lines.append(
            f"| {c['queries']} | {c['re_f_v']:.4f} | {c['row_cosine']:.4f} | "
            f"{c['func_agree']:.4f} | {c['func_corr']:.4f} |")
    lines += [
        f"- 判定: **{a3.get('verdict')}**",
        "",
        "## U1d 复合 oracle 恢复 M 每步权重",
        "",
        f"- 快照步: {a4.get('snapshot_steps')}",
        f"- 平均 RE_F(ΔW): {a4.get('mean_re_f')}，平均方向余弦: "
        f"{a4.get('mean_direction_cosine')}，轨迹余弦: {a4.get('trajectory_cosine')}",
        f"- 判定: **{a4.get('verdict')}**",
        "",
        "| from | to | RE_F(ΔW) | 方向余弦 |",
        "|---|---:|---:|---:|",
    ]
    for p in a4.get("per_step", []):
        lines.append(f"| {p['from']} | {p['to']} | {p['re_f']:.4f} | {p['direction_cosine']:.4f} |")

    lines += ["", "## 汇总", "",
              f"- U1a: **{a1.get('verdict')}** | U1b: **{a2.get('verdict')}** | "
              f"U1c: **{a3.get('verdict')}** | U1d: **{a4.get('verdict')}**"]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-12:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
