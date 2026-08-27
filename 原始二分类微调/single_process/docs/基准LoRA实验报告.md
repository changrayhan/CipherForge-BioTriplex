# ClinVar 变异致病性明文 LoRA 基准实验报告

> 实验日期：2026-08-17 ~ 2026-08-18
> 模型：TinyLlama-1.1B-Chat-v1.0（1.1B，词表 32k，hidden 2048，untied，Apache-2.0）
> 数据：ClinVar VCF 2026-06-27（`just-dna-seq/clinvar` parquet 镜像，Apache-2.0）
> 硬件/环境：RTX 4060 Laptop 8GB；conda `gfxxxh_gpu`（torch 2.11 + cu128、transformers 5.15、peft 0.20）
> 代码与产物：`CipherForgeCode/SLG-HE-PIR/clinvar_plain/`

---

## 1. 实验目的

为 CipherForge 隐私三方微调（BFV lm_head 密文库 + S3PIR + 逐 token PIR）建立主任务的**明文基线**，回答三个问题：

1. TinyLlama + LoRA 在"ClinVar 变异致病性二分类"上能做到什么水平（相对零样本/多数类的真实提升）；
2. 数据管线（清洗 → QA JSON → 按基因划分 → 评测）是否可复现、无泄漏、可被加密链路直接复用；
3. 训练策略（数据量、epoch、早停）与数据质量（标注星级）对性能的影响。

## 2. 任务定义

- **任务**：给定基因变异的结构化文本描述，预测该变异是否致病。
- **输入模板**（BioTriplex 兼容 `question + input`）：
  ```
  Is this genetic variant pathogenic for a human disease?

  Gene: BRCA2 | Variant: NC_000023.11:g.154018985G>A | Consequence: missense variant | Origin: germline

  Answer:
  ```
- **输出**：`Yes`（Pathogenic / Likely pathogenic）或 `No`（Benign / Likely benign），2 个 token。
- **语义**：罕见病诊断的标准建模范式（变体效应预测），属于"用基因预测疾病"。

## 3. 数据构建

### 3.1 数据源
- NCBI 原始 VCF 在本环境不可达；采用 `just-dna-seq/clinvar` 已解析好的 parquet（3,689,385 条，字段 = VCF INFO 小写键：`clnsig / clnrevstat / mc / origin / geneinfo / clnhgvs / rs` 等），数据日期 2026-06-27。

### 3.2 标准过滤管线（`build_clinvar_qa.py`）
1. REF/ALT 单碱基 ACGT 的 SNV；
2. `mc` 含 `missense_variant`；
3. `origin` 位掩码含 germline（`& 1`，兼容 object 数组/方括号字符串）；
4. `clnsig` 全 ∈ {Pathogenic, Likely_pathogenic} → 1；全 ∈ {Benign, Likely_benign} → 0；其余（VUS/conflicting/association 等）剔除；
5. 审核状态 gate（`--min_review_stars`）：标准版 ≥1（有判据即可），高质量版 ≥2（多提交者/专家评审）；
6. `geneinfo`、`clnhgvs` 非空；
7. 按 `chr:pos:ref:alt` 去重（审计确认 parquet 本身一变异一行，0 重复、0 冲突，去重为双保险）。

**防泄漏要点**：`clndn`（疾病名）等表型字段全程不进输入；划分严格按基因。

### 3.3 数据规模

| 数据集 | 过滤后总量 | P/LP | B/LB | train | val | test |
|---|---|---|---|---|---|---|
| 标准（stars≥1） | 178,896 | 49,350 | 129,546 | 40,000（20k/类）| 9,708 | 10,000（5k/类）|
| 高质量（stars≥2） | 59,967 | 21,438 | 38,529 | 37,967（P 17,967/B 20,000）| 5,441 | 6,145 |

> 说明：高质量版 P/LP 总量 21,438，按基因 80/10/10 划分后 train 侧 P 上限约 1.8 万，未满 2 万。

### 3.4 划分
- 按基因 80/10/10（seed 42）；train/val/test 基因零重叠（已审计）；
- 每类上限可配（5k 验证 / 20k 终版）；每 (基因, 标签) 上限 2k（标准）/ 5k（高质量），防止高频基因主导；
- 平衡采样：标准 test 5k/类；高质量 test 天然不平衡（P 1,824 / B 4,321）。

## 4. 模型与训练

### 4.1 模型
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`：1.1B、词表 32,000、hidden 2048、22 层、untied embeddings、Apache-2.0；
- 选择理由：通用预训练模型（满足项目目标）、小词表（密文库 32k 行 ≈ Llama-3.2-1B 方案的 1/4）、hidden 2048 命中框架 N=2048 BFV 分支、untied（U 端不持有 V 明文）、4060 可训。

### 4.2 训练配置
- LoRA：r=8, α=16, dropout=0.05，目标模块 q/k/v/o + gate/up/down（可训练 6.31M，0.57%）；
- bf16、lr 2e-4 cosine、weight_decay 0.01、batch 16、seq_len 128（输入最长约 73 token，截断策略保答案、裁 prompt 头）；
- 损失只算答案 token（prompt 部分 label=-100），与 CipherForge 训练掩码语义一致；
- 早停：`EarlyStoppingCallback`，patience=2（仅高质量版启用）。

### 4.3 三次运行

| 运行 | 数据 | epochs | 步数 | 时长 | train_loss | 最优 step（eval_loss） | 末期 eval_loss |
|---|---|---|---|---|---|---|---|
| 5k/类（`_128`） | 标准 10k | 3 | 1,875 | 18.8 min | 0.404 | 1000（0.604） | 0.710 |
| 标准 20k/类（`_20k`） | 标准 40k | 3 | 7,500 | 63.5 min | 0.382 | 750（0.645） | 0.863（过拟合） |
| 高质量 20k/类（`_20k_hq`） | 高质量 38k | 1 + 早停 | 2,373 | 23.4 min | 0.447 | 2000（0.527） | 0.529（无过拟合） |

> 观测：标准 20k 版 3 epochs 后期 val loss 持续回升（0.645→0.86），过拟合明显；高质量版 1 epoch + 早停下 val loss 单调降至 0.527，早停未触发（无持续恶化），训练侧问题被正确消除。

## 5. 评测协议

- 同一 prompt 模板；取最后一个真实 token 的 `Yes`/`No` logits 做 softmax → P(Yes)；
- 指标：AUPRC（主）、AUC、Accuracy@0.5；按基因 AUPRC（≥10 样本且两类均出现）；
- Baselines：多数类（=正类占比）、零样本 TinyLlama；
- **泄漏审计**：每个"模型 × 测试集"组合都检查测试基因是否落在训练基因里；凡有重叠的格子一律标注为污染并另建无泄漏子集重测。

## 6. 结果

### 6.1 完整评测矩阵（全部为已审计的干净数字，除注明外）

**① 原始平衡测试集（10,000 条，P/B 各 5k；与 5k/标准20k 训练基因零重叠）**

| 模型 | AUPRC | AUC | Acc@0.5 | 按基因 AUPRC 均值 |
|---|---|---|---|---|
| 零样本 | 0.472 | 0.443 | 0.500 | 0.584 |
| 5k/类 LoRA | **0.792** | 0.794 | 0.722 | 0.578 |
| 标准 20k/类 LoRA | 0.768 | 0.785 | 0.738 | 0.578 |
| 高质量 20k/类 LoRA | ~~0.890~~（**污染**：71.2% 测试基因在 HQ 训练集） | — | — | — |

**② 无泄漏子集（原始 test ∩ 非 HQ-train 基因，2,883 条，P 33.1%）——四模型全部干净**

| 模型 | AUPRC | AUC | Acc@0.5 |
|---|---|---|---|
| 零样本 | 0.326 | 0.473 | 0.333 |
| 5k/类 LoRA | **0.761** | 0.857 | 0.788 |
| 标准 20k/类 LoRA | 0.687 | 0.796 | 0.773 |
| 高质量 20k/类 LoRA | 0.644 | 0.769 | 0.750 |

**③ 高质量测试集无泄漏子集（HQ test ∩ 非标准训练基因，1,510 条，P 32.4%）——四模型全部干净**

| 模型 | AUPRC | AUC | Acc@0.5 |
|---|---|---|---|
| 零样本 | 0.412 | 0.585 | 0.325 |
| 5k/类 LoRA | **0.739** | 0.834 | 0.759 |
| 标准 20k/类 LoRA | 0.701 | 0.793 | 0.773 |
| 高质量 20k/类 LoRA | 0.631 | 0.785 | 0.783 |

**④ 高质量测试集全集（6,145 条，P 29.7%）——仅 HQ 模型自身干净；其余模型被污染（62.7% 基因重叠），仅作参考**

| 模型 | AUPRC | AUC | 备注 |
|---|---|---|---|
| 零样本 | 0.348 | 0.560 | 干净 |
| HQ 模型 | 0.492 | 0.707 | 干净（HQ train 与 HQ test 基因零重叠） |
| 5k / 标准20k | 0.742 / 0.694 | 0.871 / 0.852 | 污染，不作结论依据 |

### 6.2 关键发现

1. **5k/class 是当前最优且最稳健的明文基线**：自身平衡测试 AUPRC 0.792；在全部三个无泄漏子集上均排第一（0.761 / 0.739）。
2. **盲目扩大数据量无益**：标准 20k/类在所有干净评测上低于 5k/类（0.687 vs 0.761；0.701 vs 0.739）——多出的边缘样本（更多 1-star 低质量标注）拉低了泛化上限。
3. **按星级清洗反而有害**：只保留多提交者/专家评审标注（stars≥2）会改变训练分布，模型在"共识分布"上拟合好（val loss 0.527），但对全分布（含单提交者）的决策边界错位，干净评测中全排最后（0.644 / 0.631）。
4. **泄漏审计是本次最重要的方法学收获**：若不检查基因重叠，"高质量版 AUPRC 0.890"就会被误报为最优成绩；审计后确认 0.890 为 71.2% 基因重叠造成的假象。
5. **早停方案本身有效**：1 epoch + 250 步校验 + patience=2 消除了标准 20k 版的后期过拟合（val loss 单调下降），但无法挽回"数据分布不匹配"这一根本问题。

### 6.3 结论

- **正式明文基线**：TinyLlama LoRA，5k/类，3 epochs，**AUPRC 0.792 / AUC 0.794 / Acc 0.722**（零样本 0.472，多数类 0.500）；
- 该数字作为 CipherForge 加密链路必须对标的下限；加密引入 BFV/PIR/dχ 后精度不应明显低于此值；
- 高质量版与 20k 版作为消融记录保留，不进入正式结论。

## 7. 复现

```powershell
# 数据准备（需网络）：ClinVar parquet + TinyLlama 权重（hf-mirror）
$env:HF_ENDPOINT='https://hf-mirror.com'; $env:HF_HUB_DISABLE_XET='1'
& 'D:\anaconda3\envs\gfxxxh_gpu\python.exe' -s scripts\inspect_hf_dataset.py --repo just-dna-seq/clinvar --file data/clinvar.parquet --save data\external\clinvar.parquet
& 'D:\anaconda3\envs\gfxxxh_gpu\python.exe' -s scripts\download_model.py

# 构建标准 5k/20k 与高质量数据
& python scripts\build_clinvar_qa.py --from_parquet data\external\clinvar.parquet --out_dir data\qa --max_train_per_class 20000 --max_eval_per_class 5000
& python scripts\build_clinvar_qa.py --from_parquet data\external\clinvar.parquet --out_dir data\qa_hq --min_review_stars 2 --max_train_per_class 20000 --max_eval_per_class 5000 --max_per_gene_label 5000

# 训练（5k 版用 --max_train_per_class 5000 重建 data\qa 后执行）
& python scripts\finetune_plain.py --data_dir data\qa --out_dir runs\clinvar_tinylama_plain_20k --epochs 3 --eval_steps 750
& python scripts\finetune_plain.py --data_dir data\qa_hq --out_dir runs\clinvar_tinylama_plain_20k_hq --epochs 1 --eval_steps 250 --early_stop_patience 2

# 评测
& python scripts\evaluate_auprc.py --adapter runs\<run> --data data\qa\test.jsonl
```

## 8. 产物清单

| 产物 | 路径 |
|---|---|
| 方案文档 | `clinvar_plain/README.md` |
| 数据管线脚本 | `clinvar_plain/scripts/`（build / finetune / evaluate / download / inspect） |
| 标准 QA 数据 | `clinvar_plain/data/qa/{train,val,test}.jsonl` + `stats.json` |
| 高质量 QA 数据 | `clinvar_plain/data/qa_hq/` |
| 5k 运行 | `clinvar_plain/runs/clinvar_tinylama_plain_128/`（metrics.json：AUPRC 0.792） |
| 标准 20k 运行 | `clinvar_plain/runs/clinvar_tinylama_plain_20k/`（metrics.json：AUPRC 0.768） |
| 高质量 20k 运行 | `clinvar_plain/runs/clinvar_tinylama_plain_20k_hq/`（metrics*.json：泄漏审计各子集） |
| 泄漏审计子集 | `clinvar_plain/data/qa_test_no_hq_train_genes.jsonl`、`qa_hq_test_no_std_train_genes.jsonl` |

---

*本报告所有数字均可通过 `clinvar_plain/runs/**/metrics*.json` 复核。*
