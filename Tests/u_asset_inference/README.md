# T1 U 资产推断攻击（u_asset_inference）

依据《测试交接文档-U资产推断-重构后.md》第 6 节实现。测试项 T1 的四个子步：

| 子步 | 内容 | 脚本 | 判定 |
|---|---|---|---|
| U1a | 协议信息面审计（U 收到的消息白名单） | `attacks/u1a_audit.py` | 不含 H_M/s_S/全词表 logits/lora_state 即 PASS |
| U1b | 密文通道对照（BFV 密文无信号） | `attacks/u1b_ciphertext.py` | 同行/异行距离不可分、行链接恢复≈机会 |
| U1c | 复合 oracle 恢复 V | `attacks/u1c_recover_v.py` | RE_F(V)、行余弦、功能一致率 |
| U1d | 复合 oracle 恢复 M 每步权重 | `attacks/u1d_recover_weights.py` | 每步 RE_F(ΔW)、方向余弦、轨迹余弦 |

## 运行

```bash
cd /root/CipherForge/Tests/u_asset_inference
bash run_all.sh                       # 默认 block PIR、100 步
PIR_MODE=rms N_STEPS=100 bash run_all.sh
```

环境变量：`PIR_MODE`（block/rms）、`N_STEPS`、`Q_SNAP`（每快照步主动查询数）、
`MAX_SNAP_STEP`、`Q_CORPUS`（U1c 语料查询数）、`RUN_DIR`（默认自动生成）。

## 采集

- U 节点在 `CF_U_CAPTURE_DIR` 下记录：`u_responses.jsonl`（每条响应的键/大小/违规标志）、
  `train_*.npz`（H_U 全量 + p_yes）、`eval_val_*.npz`（H_U 全量 + 类别 logits）；
- M 节点在 `CF_M_CAPTURE_DIR` 下记录每步 LoRA 真值 `w_step_*.pt`（仅评估用）；
- S 节点在 `CF_S_CAPTURE_DIR` 下记录 V 类行真值 `v_rows.npz`（仅评估用）；
- driver 额外采集：固定索引的 PIR 密文（U1b）、主动 oracle 快照（U1d）、
  批量主动语料（U1c）。

> 真值（V 行、W_M(t)）只用于评估端，不进入 U 节点。

## 输出

`results/run_<ts>/`：`u1{a,b,c,d}_*.json`、`report.md`、各进程日志。
