# CipherForge-ClinVar 三进程隔离版（three_party）

在 **三个独立进程 + 三个独立目录** 的隔离架构下，完成 ClinVar 变异致病性二分类的
隐私保护微调（TinyLlama-1.1B + LoRA + BFV 同态加密 + S3PIR）。与
`single_process/`（单进程融合版）完全独立、互不隶属。

- 任务：ClinVar 变异致病性二分类（`Yes` = 致病 / `No` = 良性）
- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`（22 层，U/M 各 11 层）
- 数据：5k/类 QA（train 10,000 / val 9,708 / test 10,000，按基因切分互不重叠）
- 协议：U（协调者）↔ M/S 经 HTTP/JSON（TCP 环回，三主机部署时换 IP）

---

## 1. 三方与协调者

| 角色 | 目录 | 职责 | 运行设备 |
|---|---|---|---|
| U（用户/数据方） | `party_u/` | 持有明文样本与标签、嵌入 + 前 11 层、发 PIR 查询、DP 加噪 | GPU |
| M（模型方） | `party_m/` | 后 11 层 + LoRA、梯度更新、BFV 密钥（sk 私钥）、checkpoint | GPU + CPU（SEAL worker） |
| S（服务方） | `party_s/` | lm_head(V 矩阵)、BFV 密文库、S3PIR hints、监督位置 logits | CPU（BFV worker 亦 CPU） |
| 协调者 | `coordinator/` | 训练/验证/导出/评测编排（独立于三方之外） | CPU + GPU（U 侧） |

`shared/` 是三方共享的只读 Python 库（节点框架、协议、BFV/S3PIR 后端、模型切分、
训练器）；每个 party 目录只包含自身节点入口与其私有的数据/密钥/产物。

## 2. 隐私口径（真 PIR，v2 默认）

- 标签**值**永不出 U：U 只发送“监督位置 + 真/伪 PIR 查询块（real+dummy）”。
- 默认**块 PIR v2**：block=64、dummy 按真实标签边际采样（29871/3869/1939 =
  0.5/0.25/0.25）、真假查询块 8:2、按唯一索引去重取行；S 无法分辨真实查询行，
  频率攻击恢复率从 v1 的 100% 降至 ~40%（≈1/3 随机水平，见测试报告 §6.3）。
- S 获知监督位置（哪些 token 位置有监督），但**不获知标签值**；U 的样本与标签不泄露。
- BFV 密文库（`bfv_ct_db_*.bin`）由公钥加密，私钥仅在 M 端，M 与 S 均拿不到对方明文。
- **差分隐私（dχ-privacy，默认开启）**：`shared/core/dchi_privacy.py` 在 U 端
  `H_U` 交给 M 之前注入多元拉普拉斯噪声（`dp_eta0=900`、答案位置 β=0.5 放大），
  为 U→M 切面增加 DP 层；见 [真PIR隐私保护测试报告.md](docs/真PIR隐私保护测试报告.md) §7。
- **备用 PIR 模式：RMS-PIR v2**（`pir_mode: "rms"`，论文 Ren-Mughees-Sun CCS'24
  两服务器变体）：U 一次性下载密文库加密副本（`party_u/db/`）承担 offline
  角色（本地构建 hints 与补充半区），S 持明文 V 只应答在线子集（明文聚合+一次
  加密）——S 从不见 hint 状态，**单查询与多查询隐私均按论文成立**。
  详见 [04-备用方案与降级预案.md](docs/04-备用方案与降级预案.md)。

详见 [docs/真PIR隐私保护测试报告.md](docs/真PIR隐私保护测试报告.md) 与
[docs/块PIR与S3PIR与RMS-PIR方案对比分析.md](docs/块PIR与S3PIR与RMS-PIR方案对比分析.md)。

## 3. 目录结构

```text
cipherforge-three-party/
├── coordinator/          # 独立协调者：main.py、three_party_config.json、remote_trainer.py、evaluate_auprc.py
├── party_u/              # U 节点：main_u.py + data/qa/（明文样本）
├── party_m/              # M 节点：main_m.py + keys/ + checkpoints/
├── party_s/              # S 节点：main_s.py + db/（BFV 密文库 + hints）
├── shared/               # 三方只读共享库：node_server / remote_protocol / parties / core(BFV,S3PIR,RMS-PIR) / model / training / data / scripts
├── platform/             # Node.js 演示平台骨架（零依赖，SSE + 探活 + 会话）
├── scripts/              # run_full.sh（完整训练）、run_smoke.sh、launch_full_bg.sh（后台启动）
└── docs/                 # 架构、接口、通信协议、实验报告等
```

## 4. 环境与依赖

- Linux / WSL Ubuntu 24.04，RTX 4060 8GB 即可（M 峰值 ≈1.0GB、U ≈1.4GB）
- conda 环境：`conda env create -f environment.yml`（或 `pip install -r requirements.txt`）
- 关键版本：torch 2.10.0+cu130、transformers 5.2.0、peft 0.18.1、seal-python 4.1.2.1
- BFV：N=4096、plain_bits=30、scale=10000、lam=80；词表 32,000 × 隐层 2,048

### 下载模型（hf-mirror，约 2.1GB，不入 Git）

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir /path/to/tinyllama-snapshot
export CF_MODEL_PATH=/path/to/tinyllama-snapshot
```

## 5. 运行流程

### Stage 0 —— 一次性：BFV 密钥 + 密文库 + hints（可复用，默认跳过）

首次运行会自动生成密钥并构建密文库（约 4.2GB，耗时约 1 小时）；密钥与密文库
保存在 `party_m/keys/` 与 `party_s/db/`，**默认复用，不再重建**。

### Stage 1 —— 三进程全量微调（GPU 训练 / CPU 加解密）

```bash
export CF_MODEL_PATH=/path/to/tinyllama-snapshot
export PYTHON=$(which python)
export S_DEVICE=cpu
bash scripts/run_full.sh --max_epochs 3 --batch_size 16 --log_freq 10
```

- 后台运行：`bash scripts/launch_full_bg.sh --max_epochs 3 --batch_size 16 --log_freq 10`
- 冒烟验证：`bash scripts/run_smoke.sh`（少量 step + 小验证）
- 断点续训：`bash scripts/run_full.sh --resume ...`

训练参数与明文 LoRA 基准完全一致：lr=2e-4、warmup=100、cosine、
LoRA r=8/α=16/dropout=0.05、batch=16、max_seq=128、3 epochs（625 步/epoch，约 20 分钟/epoch）。

### PIR 模式切换

- 默认（块 PIR v2）：`coordinator/three_party_config.json` 中
  `pir_block_size: 64`、`pir_fake_ratio: 0.25`（真:假 = 8:2）、
  `pir_dummy_weights` 由 coordinator 启动时按训练集标签边际自动计算。
- RMS-PIR 备用模式：`--config coordinator/three_party_config_rms.json`
  （或 `--pir_mode rms`），参数 `rms_partition_size: 200`、`rms_lam: 16`、
  `rms_min_coverage: 20`（启动时对每个标签确定性补足 hint 池，避免批内断供）、
  `rms_db_dir: ${REPO_ROOT}/party_u/db`（U 本地密文库副本，自动下载并缓存）、
  `rms_db_download_chunk_mb: 32`。
- 块 PIR 的频率攻击模拟：`scripts/pir_frequency_attack_sim.py`；
  RMS-PIR 真实 SEAL 往返验证：`scripts/rms_roundtrip_verify.py`。

### Stage 2 —— 导出 LoRA adapter + 测试集 AUPRC

```bash
bash scripts/run_full.sh --skip_train   # 加载 last_checkpoint，导出 adapter 并评测
```

结果写入 `coordinator/logs/clinvar_auprc.json`。

## 6. 结果（RTX 4060，测试集 10,000 条，pos_rate=0.5）

> 完整数据见 `coordinator/logs/clinvar_auprc.json` 与
> [docs/明文与CipherForge隐私保护微调实验报告.md](docs/明文与CipherForge隐私保护微调实验报告.md)。

| 指标 | 明文 LoRA 基线（single_process） | CipherForge 三进程版 |
|---|---:|---:|
| AUPRC | 0.7922 | **0.7976** |
| AUC | 0.7936 | 0.7887 |
| accuracy@0.5 | 0.722 | 0.5541 |
| per-gene 平均 AUPRC | 0.5784 | **0.6248**（135 基因） |

三进程版与明文基线、单进程版（AUPRC 0.8033）使用完全相同的 5k/类数据与超参数；
AUPRC / per-gene 排序类主指标持平或更优，AUC 差距 ≈ 0.005，属加密传输浮点量化
噪声范围（±0.01）。accuracy@0.5 对概率校准敏感，与单进程版同样偏低，不作为
公平对比依据。

## 7. 已解决的关键工程问题

- **密钥绑定**：S 重建公钥后必须同步重建 Encryptor（`attach_public_key()`），否则密文库
  用新 pk 加密而 M 用旧 sk 解密，全是垃圾。
- **requires_grad**：M 从网络收到 `H_U` 后需 `H_U.requires_grad_(True)`，否则
  reentrant checkpoint 禁用自动求导、LoRA 梯度全为 None（模型冻结）。
- **导出偏移**：`save_peft_adapter` 通过 `protocol.u_layers` 把 M 侧局部层 0-10
  映射回全局层 11-21（PEFT 键名兼容 `.default` 与不带 `.default` 两种命名）。
- **内存**：8GB 单卡下 U/M 各 1 个 crypto worker；S 全程 CPU。

## 8. 文档索引

- [HANDOVER.md](HANDOVER.md) —— 三方隔离版交接文档
- [docs/00-交付清单与部署指南.md](docs/00-交付清单与部署指南.md)
- [docs/01-平台架构说明.md](docs/01-平台架构说明.md)
- [docs/02-节点接口规范.md](docs/02-节点接口规范.md)
- [docs/03-通信协议与消息格式.md](docs/03-通信协议与消息格式.md)
- [docs/真PIR隐私保护测试报告.md](docs/真PIR隐私保护测试报告.md)
- [docs/明文与CipherForge隐私保护微调实验报告.md](docs/明文与CipherForge隐私保护微调实验报告.md)
