# CipherForge-ClinVar

ClinVar 变异致病性二分类任务的 **隐私保护微调** 完整实现：以
`TinyLlama-1.1B-Chat-v1.0` 为底座、5k/类 QA 数据（train 10,000 / val 9,708 /
test 10,000，按基因切分），在 CipherForge（BFV 同态加密 + S3PIR 真 PIR）框架下
做 LoRA 微调，并与明文 LoRA 基线逐指标对照。

仓库包含**两种互不隶属的实现模式**，代码、数据、文档各自独立，严禁跨模式复用：

| 目录 | 说明 | 架构 |
|---|---|---|
| [`single_process/`](single_process/README.md) | 单进程融合版（已完整验证） | U/M/S 在同一 Python 进程内做加密原语级隔离 |
| [`three_party/`](three_party/README.md) | 三进程隔离版（U/M/S 独立进程 + 独立目录 + HTTP/JSON 通信） | 进程级 + 目录级隔离，可平滑迁移到三台主机 |

## 任务与数据

- 任务：给定 ClinVar 变异描述（question + 变异文本），预测致病性二元标签
  （`Yes` = 致病 / `No` = 良性）
- 输入模态：结构化变体文本（QA JSON），词表 32,000，最大序列 128 token
- 数据构建：`single_process/baseline/scripts/build_clinvar_qa.py`
  （原始 ClinVar 数据不随仓库分发，QA 三元组已内置在 `data/qa/`）

## 环境

- Linux / WSL Ubuntu 24.04，RTX 4060 8GB 或同级别单卡
- 两个模式各自提供 `requirements.txt` / `environment.yml`
- 关键版本：torch 2.10.0+cu130、transformers 5.2.0、peft 0.18.1、seal-python 4.1.2.1
- 模型权重经 hf-mirror 下载（不入 Git，见各模式 README）

## 结果一览（RTX 4060，测试集 10,000 条）

| 指标 | 明文 LoRA 基线 | single_process（单进程 CipherForge） | three_party（三进程 CipherForge） |
|---|---:|---:|---:|
| AUPRC | 0.7922 | **0.8033** | **0.7976** |
| AUC | 0.7936 | **0.8081** | 0.7887 |
| accuracy@0.5 | 0.722 | 0.6666 | 0.5541 |
| per-gene 平均 AUPRC | 0.5784 | **0.5961** | **0.6248** |

> 单进程版在真 PIR（block=8）下实现；per-gene 平均 AUPRC 0.5961。
> 详细口径见各模式 `docs/` 下的实验报告。

## 隐私口径

- 标签值永不出用户端（U）；模型方（M）与服务方（S）只接触密文、监督位置与
  PIR 查询块，拿不到明文标签。
- S 通过 S3PIR 返回整块查询（real+dummy，block=8），U 用私有置换取行；
  S 对真实查询行的猜中率为 1/8。
- BFV 加解密在 CPU（SEAL worker），模型训练/微调/评估在 GPU。

## 快速入口

- 单进程版：`single_process/README.md`（Stage0 → 冒烟 → 3 epochs → Stage2 评测）
- 三进程版：`three_party/README.md`（Stage0 → run_full.sh 3 epochs → --skip_train 评测）

## 目录说明

```text
CipherForge-ClinVar/
├── README.md            # 本文件
├── single_process/      # 单进程融合版（独立仓库结构）
└── three_party/         # 三进程隔离版（独立仓库结构）
```
