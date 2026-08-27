# CipherForge（原 SLG-HE-PIR）仓库结构与代码分析

> 分析方式：GitHub 网页只读浏览 + GitHub REST API（只读查询），未下载仓库。
> 数据来源：`main` 分支（2026-08-06 最后推送），GitHub API 文件树共 4529 个条目（4093 个文件 / 436 个目录）。

## 1. 仓库概况

| 项目 | 内容 |
|---|---|
| 仓库 | [changrayhan/CipherForge](https://github.com/changrayhan/CipherForge) |
| 定位 | 隐私保护大模型 LoRA 微调框架与实现（三方不互信场景） |
| 曾用名 | SLG-HE-PIR（仓库内全部代码仍在此目录名下） |
| 语言 | Python（202 个 `.py`，约 1.8 MB） |
| License | MIT |
| 创建 / 最近推送 | 2026-07-27 / 2026-08-06 |
| 默认分支 | main |
| Git 体积 | 约 90 MB（解包后文件总大小约 285 MB，主要为测试数据） |

一句话概括：在**单台 GPU 主机**上模拟 **U（用户）/ M（模型）/ S（服务器）三参与方**协作，对 Llama-3.1-8B-Instruct 做隐私保护 LoRA 微调。核心思路是用 **BFV 同态加密 + S3PIR 私有信息检索 + PRG 共享掩码** 让 LoRA 训练在各方不暴露私有数据 `(x, y)` 与 lm_head 矩阵 `V` 的前提下收敛；另有 **dχ 差分隐私**（多元 Laplace 噪声）加在 U→M 切分层上。

## 2. 目录结构总览（真实文件树，depth ≤ 2）

仓库根只有 3 项：`SLG-README.md`（主 README）、`.gitignore`、`SLG-HE-PIR/`（全部代码）。注意：README 中提到的 `S3PIR/`、`checkpoints/`、`logs/`、`SLG-attack-test/`、`tests/env/` 在 main 分支**已不存在或已迁移**（详见 §7）。

```
CipherForge/
├── SLG-README.md                  # 项目主 README（含目录说明与逐文件作用表）
└── SLG-HE-PIR/
    ├── configs/                   # 配置 dataclass（2 个 .py）
    ├── src/                       # 框架核心源代码（73 个 .py）
    │   ├── core/                  # 密码学与隐私原语层（BFV / S3PIR / dχ）
    │   ├── model/                 # 模型切分与加载（model_splitting.py）
    │   ├── parties/               # 三方协议、运行时、通信
    │   │   └── crypto_workers/    # U/M/S 三个 CPU 密码学进程池
    │   ├── data/                  # 数据集与 prompt 适配
    │   ├── training/              # 训练循环、指标、checkpoint、评测
    │   ├── scripts/               # CLI 入口（finetune / evaluate / build_*）
    │   ├── attacks/               # 协议级攻击与安全审计（18 个攻击脚本）
    │   │   └── test_doubles/      # 攻击测试替身（mock party / bus / recorder）
    │   ├── audit/                 # 离线隐私审计
    │   └── utils/                 # DeepSpeed / metrics / statistics
    ├── scripts/                   # 顶层生产脚本与功能测试（21 个 .py）
    │   ├── biotriplex_*.sh        # 任务 A/B 一键启动脚本（3 个）
    │   └── function-tests/        # 18 个功能/正确性/性能测试脚本
    ├── tests/                     # 单元与集成测试（14 个 .py）
    │   ├── dp-tests/              # dχ/DP 机制单元测试（8 个 + conftest + README）
    │   └── data-analysis/         # 一次性数据分析脚本
    ├── docs/                      # 系统文档 / 使用文档 / 流程图 SVG
    ├── datasets/                  # BioTriplex 语料 + TREC-QC（744 项）
    │   ├── botriplex/             # 原始 XML 语料 + 预处理文本（约 700 项）
    │   ├── botriplex_classification/   # 7 类 GenRel QA 划分
    │   ├── botriplex_generation/       # NER 生成任务划分
    │   └── trec-qc/               # 6 类粗粒度分类数据
    ├── baseline/                  # 明文 LoRA 微调基线（复现 BioTriplex 论文）
    │   ├── classification_genrel/ # 任务 A 产物与脚本
    │   ├── generation_ner/        # 任务 B 产物与脚本
    │   └── docs/                  # 基线测试报告与图表
    ├── test-data/                 # 测试数据与运行产物（3590 项，占仓库绝大部分）
    │   ├── AttackTest/            # 攻击测试（套件源码 + 数据 + TEST_REPORT.md）
    │   ├── PrecisionTest/         # 精度消融测试（4.2 GB 级产物）
    │   ├── BioTriplex1BTestData/  # 105 runs 大规模对照实验（Llama-3.2-1B）
    │   ├── PerformanceTest/       # 性能基准数据
    │   └── AATestArchive/         # 历史实验归档
    └── papers/                    # 参考论文（BioTriplex.pdf 等 2 份）
```

## 3. 代码组成统计

| 文件类型 | 数量 | 体积 | 分布 |
|---|---:|---:|---|
| `.py` Python | 202 | 1.8 MB | src 73、test-data 85、scripts 21、tests 14、baseline 7、configs 2 |
| `.sh` Shell | 67 | 0.2 MB | test-data 51、baseline 10、scripts 3、tests 3 |
| `.md` 文档 | 44 | 0.6 MB | 文档/报告/README |
| `.json` | 2540 | 32 MB | 几乎全部是 test-data 里的运行产物与指标 |
| `.npy` | 218 | 165 MB | test-data 激活/梯度矩阵等中间数据 |
| `.xml` | 604 | 5.8 MB | datasets 的 BioTriplex 原始语料 |
| `.txt` | 39 | 54 MB | 数据集文本（train/val/test、gold 文件） |
| `.log` | 243 | 9.5 MB | 训练/测试日志 |
| `.jsonl` | 54 | 5.0 MB | 指标流、数据集 |
| `.pt` / `.safetensors` | 31 / 4 | <0.1 MB | 测试期 checkpoint 片段 |
| `.pdf` / `.png` / `.svg` | 2 / 30 / 1 | ~10 MB | 论文、报告图表、系统流程图 |

结论：**真正的产品代码集中在 `SLG-HE-PIR/src/`（73 个 .py）+ 顶层 `scripts/`（21 个）+ `tests/`（14 个）**，约 108 个脚本；仓库体积的绝大部分是测试数据、运行日志和数据集。

## 4. 核心架构与功能分析

### 4.1 三参与方 + CPU 密码学进程池（HeterogeneousProtocol）

物理上所有代码跑在**同一台主机的同一进程**（Fusion），但密码学边界用**进程边界**实现：只有专门的 CPU 子进程持有解密私钥 `sk_M`，主进程在启动后立刻 `_drop_secret_key()`，结构上不持有私钥。

```
主进程（Fusion 驱动，单 CUDA context）
┌──────────────────────────────────────────────────────────────┐
│  PartyU (GPU) ──H_U──▶ PartyM (GPU) ──H_M──▶ PartyS (GPU)    │
│  embed/前16层    (每步H_U传引用)   decoder 16..32 + LoRA       │  lm_head V + Enc DB
│        │                │                     │               │
│        ▼                ▼                     ▼               │
│  CryptoUWorker    CryptoMWorker         CryptoSWorker          │
│  (CPU 池, 8)      (CPU 池, 8)           (CPU 池, 1)           │
│  +pk_M +PRG       +sk_M（唯一持有）      +pk +PRG +mmap密文库    │
└──────────────────────────────────────────────────────────────┘
```

| 参与方 | 职责 | 看到什么 | 隐私边界 |
|---|---|---|---|
| U（用户） | 持有私有 `(x,y)` 与 embed/前半模型；在 U→M 切分层注入 dχ 噪声 | 公钥、PRG seed、密文 | 不持有 `sk_M`，看不到 `V` 明文 |
| M（模型） | 后半 decoder + LoRA；解密掩码密文并反传，只更新 M 侧 LoRA | 解密后的 `masked_arr`、`s_share` | 看不到 `x,y` 明文与 `V` |
| S（服务器） | 持有冻结的 `V=lm_head.weight`；GPU 算 logits 与 `a_t=softmax·V`；mmap 读取加密行 | `H_M`、logits、`y_t` | 看不到输入与标签明文 |

### 4.2 密码学原语（src/core/）

| 文件 | 功能 |
|---|---|
| `bfv_privselect_v2_adapter.py`（1004 行） | BFV 主实现：SEAL 上下文（poly_degree=4096、plain_bits=30、scale=10000）、定点编码、密钥管理、将 lm_head 逐行加密为密文数据库、密文序列化、PRG 掩码份额 |
| `s3pir_hints.py` | S3PIR HintTable：分区、主/备 hint 骨架、查询索引构造、JSON 缓存 |
| `dchi_privacy.py`（720 行） | dχ 差分隐私：多元 Laplace 噪声采样（高斯方向 × Gamma 半径）、激活范数 EMA 校准（η₀ = d/(α·A)）、标签条件 CTI、`H15Privatizer` 门面 |
| `key_remapping.py` | 修复 PEFT/LoRA checkpoint 键名不一致与权重方向 |
| `protocol_he_pir.py` | 历史 RSA-KEM + AES-GCM 协议封装（仅审计用，不在当前主链） |

### 4.3 三方协议层（src/parties/）

| 文件 | 功能 |
|---|---|
| `heterogeneous_protocol.py`（945 行） | **当前主运行时**：装配 U/M/S + 三个 CryptoWorker 池；向 Trainer 暴露 `step_train(_chunked)/step_val/gather_checkpoints/shutdown`；支持 DP 校准模式 |
| `party_u.py` / `party_m.py` / `party_s.py` | 三方各自的前向、私密选择（privselect）、反传与更新 |
| `transport.py` / `wire.py` | 统一 MessageBus 协议、进程内/跨进程队列总线；`StepResult` 与 `StepProfiler` 数据结构 |
| `fusion_protocol.py` | 早期同进程直调版本（已被 HeterogeneousProtocol 取代） |
| `ipc_protocol.py` / `legacy_ipc_stub.py` | 历史三独立进程版本，保留作多主机预演/兼容 |

`crypto_workers/`（隐私关键）：
- `crypto_u.py`：对 S 返回的 `Enc(-V_y)` 同态加入与 S 共享 PRG 的掩码 `R_t` → `Enc(-V_y + R_t)`；
- `crypto_m.py`：**唯一持有 sk_M 的进程**，批量解密得 `masked_arr`；
- `crypto_s.py`：生成 `s_share = a_t − R_t`，并从 mmap 密文库读 `Enc(-V_y)`；
- `pool.py`：长生命周期 spawn 进程池。

### 4.4 模型 / 数据 / 训练 / 入口

| 模块 | 功能 |
|---|---|
| `model/model_splitting.py` | 将 Llama 切成 U/M/S 三份，safetensors 按需加载，支持 FlashAttention2/SageAttention、梯度检查点 |
| `data/biotriplex_dataset.py` | BioTriplex 分类（GenRel 7 类）与 NER 生成数据、Llama chat prompt、gold 文件 |
| `data/dataset.py` | 通用 JSONL 加载器与答案/实体解析 |
| `training/trainer.py`（1199 行） | 协议无关的高层训练循环：epoch/step、指标、早停、checkpoint、测试与 dχ 审计 |
| `training/checkpoint.py` / `evaluation.py` / `biotriplex_metrics.py` | U/M/S 联合 checkpoint 管理；明文评估（合并 LoRA 后 HF 生成）；分类多类 F1/AUC 与 NER 实体级指标 |
| `src/scripts/biotriplex_finetune.py`（866 行） | **BioTriplex 总入口**：任务分类(6 epoch)/生成 NER(10 epoch)/trec-qc(5 epoch)，三阶段 all |
| `src/scripts/finetune.py` | 通用/旧格式入口（内置 Config，支持 JSON override） |
| `src/scripts/build_encrypted_db.py` | Stage 0 Step 1：加密 lm_head 每行 → 密文库 + 公钥 + 元数据 |
| `src/scripts/build_s3pir_hints.py` | Stage 0 Step 2：构建 S3PIR 分区/hint 骨架（parity 目前是简化骨架） |
| `configs/llama_biotriplex_he_pir.py` | 全量配置 dataclass：BFV/LoRA/训练超参/worker 数等 |

### 4.5 攻击与审计（src/attacks/、src/audit/）

针对"诚实但好奇"三方设计了 18 个攻击/审计脚本：
- **L 系列（标签/梯度推断）**：L1A 掩码份额分离、L1C DLG 梯度反演、L2 cutgrad 聚类标签推断、L3B PIR 字节流泄漏、L4A0/L4A/L4B 激活反演、L5 S 方输入恢复、L6 长期训练隐私退化；
- **M 系列（模型推断/成员推断）**：M1 蒸馏提取、M2 结构探测（Jacobian 秩/MMD）、M3 LoRA 内部审计、M4 成员推断、M5 V 矩阵推断；
- **P 系列（原语安全）**：P1 BFV 参数/噪声预算审计、P2 PRG 随机性审计、P3 PIR 查询不可区分性（明确报告 Design-2 中 S 知道 `y_t` 的问题）、P4 系统级风险汇总；
- `test_doubles/`：mock party / 恶意中间人 bus / 消息录制，用于在不跑真实 GPU 训练的情况下做攻击实验；
- `audit/lia_h15_audit.py`：读取训练期 `dp_audit.jsonl`，汇总 dχ 激活率、校准次数、η 与噪声范数，输出 Markdown 报告。

### 4.6 测试与测试数据

| 套件 | 内容 |
|---|---|
| `tests/` | 单元/集成测试：party_s 分类、PRG 向量化、3-step 分类、8 个 DP 机制测试（calibrator/CTI/dchi sampler/H15 privatizer/协议 smoke/TREC-QC e2e 等） |
| `scripts/function-tests/` | 18 个功能测试：2-epoch、10-step smoke、chunk 与 flat 路径等价、异构协议正确性、e2e 数学验证、性能基准、step profiler 等 |
| `test-data/AttackTest/` | 攻击测试套件（独立子项目，含 L1/L2/M1/M2 四大攻击面）与 27 组 DP 参数网格消融 |
| `test-data/PrecisionTest/` | 精度消融：6 种量化变体 × 3 seed 等 105 runs 规模对照，产出 `QUANT_ABLATION_REPORT.md`、`CLS_PRECISION_COMPARISON_REPORT.md` |
| `test-data/BioTriplex1BTestData/` | Llama-3.2-1B 上 4 phase × 35 configs × 3 seeds = 105 runs 大规模实验 |
| `test-data/PerformanceTest/` | 每步耗时 profile、通信/离线准备开销估算 |
| `test-data/AATestArchive/` | 历史失败实验归档 + 废弃的 TREC-QC 测试 |

## 5. 运行流程（三阶段）

```
Stage 0（离线，一次）     Stage 1（在线训练）          Stage 2（明文评估）
build_encrypted_db.py ──▶ biotriplex_finetune.py ──▶ evaluate_biotriplex.py
  → BFV 密文库/公钥        → 三方协议 + Trainer          → 合并 LoRA → HF 生成
build_s3pir_hints.py        → U/M/S + 3 个 worker 池      → 指标 JSON
  → hint_table.json
```

单步训练链（README + 代码双重确认）：
`PartyU.forward_train` → `PartyM.forward` → `PartyS.process_logits_dispatch`（GPU 算 logits/softmax·V/argmax）→ `CryptoSWorker`（PRG 掩码 + mmap 读加密行）→ `PartyU.privselect` → `CryptoUWorker`（密文加掩码）→ `PartyM.backward_and_update`（`CryptoMWorker` 解密 → `masked_arr + s_share = a_t − V_y`）→ LoRA 更新。

顶层 Shell：`scripts/biotriplex_run_all.sh` 串联任务 A（分类 6 epoch）→ 任务 B（NER 10 epoch）；另有单任务脚本。

## 6. 技术栈

| 层 | 选型 |
|---|---|
| 基础模型 | Llama-3.1-8B-Instruct（vocab 128256, hidden 4096, 32 层） |
| 微调 | LoRA r=8 / α=16 / dropout=0.05，注入 Q/K/V/O + Gate/Up/Down 7 个投影 |
| 同态加密 | BFV（seal-python / TenSEAL，N=4096, plain_bits=30, scale=10000） |
| PIR | S3PIR（hint 骨架 + 密文 mmap 库；实现采用 Design-2 单行直取） |
| 差分隐私 | dχ 多元 Laplace + 标签条件 CTI + H15 私有化器（默认关闭，`--dp_enable`） |
| 注意力/优化 | FlashAttention2、SageAttention2++(INT8)/3(FP4)、梯度检查点、DeepSpeed ZeRO 1/2/3、Chunked Pipeline |

## 7. 观察与注意点

1. **README 部分过时**：`SLG-README.md` 中的 `S3PIR/`、`checkpoints/`、`logs/`、`SLG-attack-test/`、`tests/env/` 等目录在 main 分支已不存在/迁移（攻击套件实际位于 `test-data/AttackTest/scripts/SLG-attack-test/`）；`docs/` 也只有 3 个文件而非 16 个。
2. **仓库含大量测试产物**：2540 个 JSON、218 个 NPY、243 个日志，占仓库体积绝大部分；若只关心源码，`src/ + configs/ + scripts/ + tests/` 才是核心（约 110 个脚本，2.5 MB）。
3. **硬编码路径**：默认路径大量指向 `/root/autodl-tmp/...`（原开发机），换机器需通过 CLI 参数覆盖；仓库根没有统一的 `requirements.txt`（依赖散见于文档：torch/transformers/peft/safetensors/seal-python/tenseal 等）。
4. **已知协议局限（代码内自述）**：PIR 采用 Design-2 时 S 直接知道 `y_t`（argmax token id），文档明确记录该隐私权衡；S3PIR parity 计算目前是简化骨架；文档记录早期 NER 数据与 QA 字母评测指标错位导致 `val_* = 0` 的问题（后续版本已针对 BioTriplex 修正评测）。
5. **定位**：README 明示这不是生产系统，而是隐私路径下 LoRA 能收敛的工程验证；多主机部署接口（IPCProtocol/LegacyIPCStub）已预留但非默认路径。
