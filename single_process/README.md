# ClinVar 变异致病性二分类微调（TinyLlama-1.1B）

本仓库包含 **两条可复现的微调管线**，任务均为 ClinVar 变异致病性二分类
（`Yes` = 致病 / `No` = 良性），模型为 `TinyLlama-1.1B-Chat-v1.0`，数据为
5k/类（train 10,000 / val 9,708 / test 10,000，按基因切分）：

1. **`baseline/` — 明文 LoRA 基准**：标准 HuggingFace + PEFT 微调。
2. **`cipherforge/` — CipherForge 隐私保护微调**：三方（U/M/S）BFV 同态加密
   + S3PIR 管线，明文数据不出用户侧、模型侧拿不到密文对应的明文标签。

> 三进程隔离版（`cipherforge-three-party`）是独立仓库/目录，见交接文档。

## 结果（RTX 4060，测试集 10,000 条）

| 指标 | 明文基线 | CipherForge（单机融合版） |
|---|---:|---:|
| AUPRC | 0.7922 | **0.8033** |
| AUC | 0.7936 | **0.8081** |
| accuracy@0.5 | 0.722 | 0.6666 |
| per-gene 平均 AUPRC | 0.5784 | **0.5961** |

详见 [`docs/results.md`](docs/results.md)。

## 目录结构

```text
clinvar-submit/
├── baseline/            # 明文 LoRA 基准（脚本 + 运行入口）
├── cipherforge/         # CipherForge 单机版（src 子集 + 运行脚本 + 工具）
├── configs/             # CipherForge 配置文件（路径用 ${VAR} 占位）
├── data/qa/             # 5k/类 QA 数据（train/val/test.jsonl + splits/stats）
├── docs/                # 环境搭建、结果记录
├── environment.yml      # conda 环境定义
└── requirements.txt     # pip 依赖（含 SEAL）
```

## 快速开始（Linux / WSL Ubuntu 24.04）

### 1. 安装依赖

```bash
conda env create -f environment.yml      # 或手动建环境后 pip install -r requirements.txt
conda activate clinvar-ft
nvidia-smi                               # 确认 GPU 可用（RTX 4060 8GB 即可）
```

### 2. 下载模型（hf-mirror）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python baseline/scripts/download_model.py
export CF_MODEL_PATH=$(python cipherforge/tools/resolve_model_path.py)
```

模型权重（约 2.1GB）只进 HF 缓存，**不提交到 Git**。

### 3. CipherForge 隐私保护微调

```bash
export REPO_ROOT=$(pwd)                  # 仓库根目录
cipherforge/scripts/run_stage0.sh        # 一次性：BFV 密钥 + 4.2GB 密文库 + hints
cipherforge/scripts/run_smoke.sh         # 冒烟：10 步 + 小验证 + adapter 导出
cipherforge/scripts/run_epoch.sh 0       # 全量训练：3 × (625 步) ≈ 3 × 16 分钟
cipherforge/scripts/run_epoch.sh 1
cipherforge/scripts/run_epoch.sh 2
cipherforge/scripts/run_stage2.sh        # 测试集 AUPRC
```

### 4. 明文 LoRA 基准

```bash
MAX_STEPS=30 baseline/run_baseline.sh    # 冒烟（30 步）
baseline/run_baseline.sh                 # 全量（3 epochs，约 30 分钟）
```

## 数据

`data/qa/` 已包含微调用的 QA 三元组（question / input / output），由
`baseline/scripts/build_clinvar_qa.py` 从 ClinVar 数据构建（原始数据见该脚本与
`download_clinvar.py`，不入库）。统计见 `data/qa/stats.json`。

## 隐私保护说明

CipherForge 管线中：用户侧 U 持有明文输入与标签；模型侧 M 只拿到
`Enc(−V_y + r_t)` 密文与 `s_share`；服务侧 S 持有 lm_head 与密文库，**全程
不接触标签值**——U 通过真实块 PIR（real+dummy、块大小 8、私有置换取行）
取回密文行，S 对每个查询猜中真实行的概率仅 1/8；S 只获知监督位置（答案在
哪），不获知任何 Yes/No 标签。BFV 参数 N=4096 / plain_bits=30 / scale=10000。
本仓库是**单机融合运行版**（进程内 Party + 子进程加密 worker），进程级三方
隔离版见独立目录 `cipherforge-three-party`。
