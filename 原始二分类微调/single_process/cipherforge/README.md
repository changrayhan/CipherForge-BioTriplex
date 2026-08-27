# CipherForge 隐私保护微调（单机版）

在单机（RTX 4060）上以“U/M/S 三方 + BFV 同态加密 + S3PIR”的方式完成
ClinVar 致病性二分类 LoRA 微调。本目录是**单进程融合运行时**：三个 Party
在同一进程内（共享 GPU），加密原语在 `spawn` 子进程池中执行，密钥材料按
职责隔离（sk 只进 M 侧，PRG seed 只进 U/S 侧）。

## 流程

```text
Stage 0（一次性）  Stage 1（训练）        Stage 2（评估）
密钥+密文库+hints  → 3 epochs LoRA 微调  → 测试集 AUPRC
```

每步数据流：

```text
U: embed + 层0-10 → H_U ──► M: 层11-21 + LoRA → H_M ──► S: lm_head
S: 取 Enc(−V_y) + 生成 s_share=a_t−r_t ──(密文)──► U: 同态加 r_t
U ──Enc(−V_y+r_t)──► M: 解密 + s_share → (a_t−V_y)·scale → 反向传播更新 LoRA
```

## 运行

```bash
cipherforge/scripts/run_stage0.sh   # 密钥 + 4.2GB 密文库 + hints（约 1 分钟）
cipherforge/scripts/run_smoke.sh    # 10 步冒烟（含 adapter 导出）
cipherforge/scripts/run_epoch.sh 0  # 全量 3 epochs（每段约 16 分钟）
cipherforge/scripts/run_epoch.sh 1
cipherforge/scripts/run_epoch.sh 2
cipherforge/scripts/run_stage2.sh   # AUPRC
```

## 目录

```text
cipherforge/
├── src/                  # 最小代码子集（parties/model/core/data/training/scripts）
├── scripts/              # Linux 运行脚本
├── tools/                # 真实 SEAL 往返验证、模型路径解析、AUPRC 评估
└── checkpoints/          # 运行产物（不入库）：bfv_cache、clinvar_ckpts
```

## 密码学参数

BFV：N=4096、plain_bits=30、scale=10000；密文库 32000 行 × 2048 维；
S3PIR hint：179 分区、lam=80。

## 验证

`tools/rt_bfv_roundtrip_test.py` 使用真实 SEAL + 持久化密钥 + 真实密文库做
加解密往返，将重建梯度与明文 `(a_t − V_y)` 对比（预期误差 ≤ 1e-3）。
