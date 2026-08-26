# 明文 LoRA 基准（TinyLlama × ClinVar）

标准 `transformers.Trainer` + `peft` LoRA 微调，作为隐私保护管线的对照。

## 运行

```bash
MAX_STEPS=30 baseline/run_baseline.sh    # 冒烟：30 步后评估
baseline/run_baseline.sh                 # 全量：3 epochs（~30 分钟，RTX 4060）
```

产物：

- `baseline/outputs/clinvar_tinylama_plain_128/` — PEFT adapter + trainer 状态。
- `baseline/outputs/metrics.json` — 测试集 AUPRC/AUC/accuracy。

## 脚本

| 脚本 | 作用 |
|---|---|
| `scripts/finetune_plain.py` | 训练入口（参数：`--max_steps`、`--batch_size`、`--lr` 等） |
| `scripts/evaluate_auprc.py` | 测试集评估（最后一位置 Yes/No logits → AUPRC） |
| `scripts/build_clinvar_qa.py` | 从 ClinVar 构建 QA 三元组 |
| `scripts/download_model.py` | 下载 TinyLlama（支持 `HF_ENDPOINT` 镜像） |
| `scripts/download_clinvar.py` | 下载 ClinVar 原始数据 |

## 预期结果

全量训练后测试集 AUPRC ≈ **0.792**（详见 `docs/results.md`）。
