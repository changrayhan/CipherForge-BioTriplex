# CipherForge 替代方案筛选：基因→疾病预测（小模型 + 小数据，RTX 4060 可微调）

> 日期：2026-08-17
> 背景：CipherForge（原 SLG-HE-PIR）主基线为 BioTriplex（Llama-3.1-8B + 100 篇全文语料），依赖过大。
> 目标：在"用基因预测疾病"这一语义下（而非文献关系抽取），寻找近两年（2024-08 ~ 2026-08）顶刊/顶会支撑的、模型更小、数据更小、可在 RTX 4060（8GB）上完整微调的方案。

---

## 一、筛选口径（按用户澄清）

用户明确："基因疾病预测 = 用基因来预测疾病"。即：

- **输入**：基因 / 变异 / 基因型层面的特征（变异位点、HGVS、基因符号、蛋白或 DNA 序列、体细胞突变谱等）
- **输出**：疾病相关结论（该变异是否致病、该基因关联哪些疾病、该肿瘤属于哪类癌症等）
- **排除**：BioRED / BioTriplex 这类"从文献文本中抽取基因-疾病关系"的任务（虽然都叫 gene-disease，但不是预测）

在此基础上同时满足：模型小（≤1B 量级或更小）、数据小（几百到几万条）、可公开获取、近两年有顶刊/顶会锚点、RTX 4060 上可跑通 CipherForge 三阶段隐私微调流程（或仅需小幅适配）。

---

## 二、候选方案总览

| # | 任务（输入→输出） | 数据集（规模） | 候选模型 | 顶刊/顶会锚点 | CipherForge 兼容度 | RTX 4060 |
|---|-------------------|----------------|----------|---------------|--------------------|----------|
| 1 | 变异→致病性（是否致病/关联疾病） | ClinVar 子集（1k~30k 条）；DYNA CM 658 / ARM 489 | Llama-3.2-1B-Instruct（原生兼容） | DYNA（Nat Mach Intell 2025）、GPN-MSA（Nat Biotechnol 2025）、popEVE（Nat Genet 2025）、AlphaMissense（Nature 2023）、PrimateAI-3D（Science 2023） | 高（Llama 因果 LM，作者已预留支持） | ✅ LoRA 宽裕；全参需 8bit 优化器 |
| 2 | 基因→关联疾病（QA） | OMIM morbid map（约 1.5~2 万对）、ClinGen（约 2k 条）、GeneTuring GDA 模块（100 QA） | Llama-3.2-1B-Instruct | GeneTuring（Brief Bioinform 2025）、LLM 文献基因-疾病关联预测（Brief Bioinform 2025）、GP-GPT（preprint，参照） | 高（同 QA 格式） | ✅ |
| 3 | 蛋白/DNA 序列变异→致病性（病种特异） | DYNA Zenodo 公开数据（CM 658 / ARM 489 / MFASS 非编码） | ESM-2 650M/150M/35M、NT-50M | DYNA（Nat Mach Intell 2025） | 低（encoder 无 lm_head，需适配切分/分类头） | ✅（35M/150M 全参均可） |
| 4 | 体细胞突变谱→癌症类型 | TCGA MAF（约 1 万肿瘤 / 33 型，可子集到 1k~5k） | Llama-3.2-1B 或小 transformer | 锚点弱（DeepGraphMut 为 bioRxiv 预印本；D3NS 为 Cancers 2024） | 高（QA 格式） | ✅ |
| 5 | 单细胞基因表达→疾病状态 | GEO 小数据集（如心肌病队列） | Geneformer（15M）、scGPT（~100M） | Geneformer（Nature 2023）、scGPT（Nat Methods 2024） | 低（BERT 式编码器 + 分类头） | ✅ 但不推荐（表达层面，非"基因型"） |

---

## 三、首选方案：ClinVar 变异致病性预测 + Llama-3.2-1B-Instruct

### 3.1 为什么选它

1. **语义最贴合**："给定基因/变异，预测其是否导致疾病"是基因组学中"用基因预测疾病"的标准建模方式，也是罕见病诊断、变异解读的核心步骤（GPN-MSA 论文开篇即称 variant deleteriousness 预测是 rare disease diagnosis 的关键）。
2. **近两年顶刊锚点密集**：
   - DYNA —— Nature Machine Intelligence 2025（7(4):661-671），病种特异变异致病性微调，基座 ESM-1b/ESM-2 650M，数据仅数百条；
   - GPN-MSA —— Nature Biotechnology 2025（43(12):1960-1965），全基因组变异效应预测，在 ClinVar/COSMIC/OMIM 基准上超越 NT、HyenaDNA、CADD、ESM-1b，训练仅 3.5 小时/4×A100；
   - popEVE —— Nature Genetics 2025，蛋白质组规模变异致病性；
   - AlphaMissense（Nature 2023）与 PrimateAI-3D（Science 2023）为奠基性参照（略早于两年窗口）。
3. **数据可做到很小**：DYNA 证明约 600 条病种特异变异即可完成有效微调（用预训练蛋白 LM 基座）；若用通用 ClinVar 平衡子集，1 万~3 万条已远超常见 VEP 微调规模。
4. **模型兼容性最好**：CipherForge 的 `model_splitting.py` 原生支持 Llama 与 GPT-2 因果 LM；仓库代码已显式预留 Llama-3.2-1B（见 §3.4）。

### 3.2 数据构造（公开、小）

**主数据源：ClinVar**

- NCBI FTP：`https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`
- 2024-06 版约含 289.7 万条临床相关变异（Bioinformatics Advances 2025 统计）。
- 构造步骤：
  1. 过滤：germline + missense/frameshift 等明确后果 + 临床意义仅为 P/LP 或 B/LB（剔除 VUS 与矛盾记录）；
  2. 按变异（chr:pos:ref:alt 或 VariationID）去重，避免同一变异多条提交造成的泄漏；
  3. 平衡采样（如每类 5k~20k），并按**基因**划分 train/val/test（防止同基因变异跨集泄漏）；
  4. 转成 BioTriplex QA JSON（字段 id / input / question / output，见 `src/data/dataset.py`）。

**QA 模板示例**（与 `BioTriplexQADataset` 直接兼容）：

```text
question: Is this genetic variant pathogenic for a human disease?
input:    Gene: BRCA2 | Variant: c.5946delT (p.Ser1982ArgfsTer22)
         | Consequence: frameshift | HGVS: NC_000013.11:g.32316468del
output:   Yes
```

也可做成疾病名输出（"Which disease is this variant associated with?" → "breast-ovarian cancer, familial"），此时评估方式与 TREC-QC/BioTriplex 的分类评估一致。

**备用小数据源：DYNA 公开数据（Zenodo）**

- 记录：`https://zenodo.org/records/12116074`（数据）、`https://zenodo.org/records/13397296`（论文版 v2）
- 心肌病（CM）：总计 658 条错义变异（致病 356 / 良性 302；训练 238/202）
- 心律失常（ARM）：总计 489 条（致病 252 / 良性 237；训练 168/158）
- 另有 MFASS 非编码剪接数据集
- 注意：作者声明 CM/ARM 子集噪声较大，仅用于测试泛化；完整复用时建议自行清洗后再作训练集。

### 3.3 模型与显存预算（RTX 4060 8GB）

**Llama-3.2-1B-Instruct 参数**：约 12.3 亿参数，hidden=2048、16 层、词表 128,256、tied embeddings；bf16 权重约 2.4GB。

| 方案 | 显存估算 | 结论 |
|------|----------|------|
| LoRA r=8（q/k/v/o/gate/up/down） | 权重 2.4GB + LoRA 可训练参数约 300 万（优化器状态可忽略）+ 梯度检查点激活约 1~2GB（seq≤512, batch=1）≈ **<6GB** | ✅ 安全，推荐 |
| 全参数 bf16 + AdamW | 2.4（权重）+ 2.4（梯度）+ 4.9（m/v）≈ 9.7GB，超 8GB | ❌ 直接不行 |
| 全参数 bf16 + 8bit AdamW（bitsandbytes）或 Adafactor + 梯度检查点 + seq≤512 + batch=1 | 权重/梯度约 4.8GB + 优化器约 1.2~2GB + 激活 1~2GB ≈ **7~8GB** | ⚠️ 紧张但可行（建议配合梯度累积） |

建议：以 **LoRA 走完三阶段隐私微调流程**作为"完整微调"的默认口径；若坚持全参数，用 8bit 优化器并把序列长度压到 256~512。

### 3.4 CipherForge 落地改动（几乎零改造）

对照仓库代码（已在线核实行号）：

1. `src/scripts/biotriplex_finetune.py`
   - `--hf_model Llama-3.2-1B-Instruct`
   - `--hidden_dim` 不传即可自动推断：`_resolve_auto_hidden_dim()`（L256-293）会从 config.json 读出 hidden_size=2048，代码注释已明确以 Llama-3.2-1B 为例（L139-146、L261-267）；
   - `--max_seq_length` 从默认 10000 改为 256~512（变异任务文本很短）；`--eval_max_seq_length` 从 4096 改为 512；
   - `--vocab_size 128256`。
2. Stage 1 加密库：`_load_V_for_db()` 已处理 tied embeddings（L351：`lm_head.weight = embed_tokens.weight`）；`bfv_privselect_v2_adapter.py:463` 注明 "V row dim can be smaller (2048 for Llama-3.2-1B)"，2048 维恰好装入 poly_degree=4096 的一个密文（无需拆行）。
3. 数据文件：将上述 QA JSON 放入 data 目录即可，`BioTriplexQADataset` 按 id/input/question/output 读取。
4. 运行前先修复《CipherForge_代码优化分析.md》中的阻断性缺陷，至少包括：
   - `party_s.py:258` 负索引 PIR（gold_ids 含 -100 未过滤）；
   - Stage 1 缓存公钥/新私钥不匹配（`bfv_privselect_v2_adapter.py:666-685`）；
   - attention_mask 未传入 decoder 层（`model_splitting.py:399,498`）。

### 3.5 隐私协议成本对比（8B 基线 vs 1B 方案）

隐私协议开销主要来自 lm_head 的 BFV 密文库与逐 token 的 PIR/解密，二者都与 hidden_dim 直接相关：

| 项目 | BioTriplex（8B，hidden 4096） | 本方案（1B，hidden 2048） | 说明 |
|------|------------------------------|---------------------------|------|
| 每行密文数 | 2 个（poly_degree=4096 装不下 4096 维，需拆行） | 1 个（2048 维恰好装入） | 密文库体积约减半（约 16GB → 8GB 量级） |
| 备选参数集 | — | 代码已有 N=2048 分支（`bfv_privselect_v2_adapter.py:113`，coeff [36,14]），2048 slots 无 padding | 密文库可再减半（约 4GB） |
| 每 token PIR/解密 | 取 2 行 + 2 次解密 + 拼接 | 取 1 行 + 1 次解密 | 逐 token HE 成本约减半 |
| dχ 噪声采样（`dchi_privacy.py`） | 按 hidden 维逐标量，4096 次/样本 | 2048 次 | 减半 |
| PRG 掩码（`bfv_privselect_v2_adapter.py:248-286`） | 4096 次 SHA-256/token | 2048 次 | 减半 |
| 序列长度 | 默认 10000（全文），pad 到 max_length | 256~512（变异文本） | pad token 仍跑完整 PIR/解密，短序列收益巨大 |

结论：切到 Llama-3.2-1B + 短文本后，隐私协议的**逐 token 加密开销约减半、总计算量（含序列长度）降低一个数量级以上**，RTX 4060 上可复现完整三阶段流程。

---

## 四、备选方案

### 4.1 基因→关联疾病 QA（最字面的"基因预测疾病"）

- 数据：OMIM morbid map（约 1.5~2 万基因-疾病对；需 OMIM 注册下载）、ClinGen gene-disease validity（约 2k 条，公开）、或 Monarch Initiative 整合表；
- 格式："Which diseases are associated with gene X?" → 疾病列表；
- 锚点：GeneTuring 的 gene-disease association 模块（Briefings in Bioinformatics 2025, 26(5):bbaf492，16 个基因组学 QA 模块之一）；Briefings in Bioinformatics 2025 的 "A large language model framework for literature-based disease-gene association prediction"（26(1):bbaf070）；GP-GPT（arXiv:2409.09825，基因-表型映射微调，但语料 300 万+词条，与"小数据"不符，仅作参照）；
- 注意：数据量小且本质是知识召回，模型容易变成记忆任务；必须按基因划分测试集，并建议与 RAG 对比（见 §5）。

### 4.2 DYNA 式病种特异致病性（ESM 微调，需适配框架）

- 与首选方案同任务族，但基座为蛋白语言模型（ESM-1b/ESM-2 650M，或更小的 35M/150M），数据极小（CM 658 / ARM 489 条）；
- CipherForge 兼容性低：ESM 是 encoder、无 lm_head，U/M/S 切分与"按词表行加密"的 S3PIR 协议需要适配（把分类头当作 lm_head 加密，或新增分类头协议）；
- 若允许改框架，35M/150M 全参数微调在 4060 上非常轻松，是"最省资源"的路线。

### 4.3 TCGA 体细胞突变→癌症类型

- 数据：GDC 开放访问 MAF（约 1 万肿瘤 / 33 种癌症，可子集 1k~5k），输入为肿瘤的突变基因集，输出癌症类型；
- 是"用基因（突变谱）预测疾病（癌症）"的直接建模；QA 格式可复用 Llama-3.2-1B；
- 缺点：近两年顶刊/顶会锚点弱（DeepGraphMut 为 bioRxiv 预印本、D3NS 为 Cancers 2024），适合作为自建任务而非顶刊背书方案。

### 4.4 单细胞基因表达→疾病（不建议）

- Geneformer（Nature 2023，15M 参数）与 scGPT（Nature Methods 2024，~100M）可做疾病状态分类，模型极小；
- 但输入是"基因表达"而非"基因/变异本身"，且均为 BERT 式编码器+分类头，框架适配成本高；
- 仅当用户本意是"基因表达预测疾病"时再考虑。

---

## 五、风险与注意事项

1. **LLM 对变异注释几乎没有先验知识**：Bioinformatics Advances 2025（5(1):vbaf019）实测 GPT-4o 直接回答"变异→基因/疾病"准确率 <2%；小样本 SFT 主要是注入/记忆知识。若目标是"事实召回"型应用，该文结论是 RAG 优于 SFT；若目标是"隐私保护微调流程 + 分类器"（CipherForge 的定位），SFT 合适。
2. **ClinVar 标签偏倚**：P/LP 与 B/LB 存在确证偏倚（ACMG BA1/PP3 标准相互引用），GPN-MSA 建议用 gnomAD 常见变异作为良性对照；至少应按基因分拆并报告 AUPRC 而非仅 Accuracy。
3. **极小数据过拟合**：数百条（DYNA 规模）时模型容易记住训练集；用预训练基座（如 ESM 或已指令微调的 Llama-3.2-1B-Instruct）+ LoRA + 按基因 holdout 可缓解。
4. **DYNA 数据噪声**：作者声明 CM/ARM 子集仅用于测试泛化，直接当训练集需清洗。
5. **密文库是一次性构建成本**：Stage 1 需对 128,256 行逐行 BFV 加密，单机仍可能耗时数小时~数天（受 CPU 性能影响），与数据规模无关，属框架固有成本。
6. **"完整微调"口径**：8GB 卡上全参数 bf16+AdamW 放不下，需 LoRA 或 8bit 优化器；若用户预期"全参数"，需确认可接受 8bit/Adafactor。

---

## 六、已确认口径（用户答复）

1. **粒度**：排除 PRS 式"个体基因型→疾病风险"（数据通常很大且需申请）；聚焦变异级（首选方案）或基因级（备选 4.1）。
2. **完整微调**：走完 CipherForge 三阶段隐私微调流程即可，采用 LoRA。
3. **模型范围**：不限定 Llama 系，允许为 ESM/DNA 模型适配框架（DYNA 式小模型路线可作为第二路线）。
4. **终极期待**：三台 RTX 4060 笔记本分别承担 U / M / S 三方，协同微调（可行性分析见 §7）。

---

## 七、三台 RTX 4060 笔记本协同微调可行性

### 7.1 现状：仓库已经为三机部署预留了架构

- `src/parties/transport.py` 定义了统一的 `MessageBus` 抽象（`send(peer, tag, payload, step)` / `recv(peer, tag, step, timeout)`），并给出两个实现：`InProcessBus`（单进程，当前激活运行时用）与 `QueueBus`（多进程 `mp.Queue`，供 LegacyIPCStub 使用）。协议层不感知具体传输。
- `src/parties/legacy_ipc_stub.py` 的模块文档明确写着：*"Multi-host deployment preview. When deploying to three physical hosts, swap the in-process bus for an RPC bus and reuse LegacyIPCStub's surface unchanged."*——即作者把"三台物理机部署"设计为"换一个 RPC 版 bus"，协议接口不动。
- 该模块已包含三个独立 worker 入口 `_worker_U_entry` / `_worker_M_entry` / `_worker_S_entry`，各自在独立进程中构造 PartyU / PartyM / PartyS，天然对应三台机器。
- 结论：三机化是仓库预期的部署方式，不是偏离设计；工作量集中在"实现网络版 MessageBus + 张量序列化 + 修复 LegacyIPCStub 的坏路径"，协议逻辑不需要重写。

### 7.2 每台机器的角色与资源预算（Llama-3.2-1B + LoRA，seq 256~512）

| 机器 | 角色 | 持有内容（示例切分：16 层 U/M 各 8 层） | 显存（4060 8GB） | 内存 | 网络职责 |
|------|------|----------------------------------------|------------------|------|----------|
| U（用户） | PartyU + CryptoUWorker | embedding + 前 8 层（约 1.2GB bf16）+ LoRA-U + PRG 种子/掩码 | 约 3~4GB | 小 | 与 M 交换 H_U 与梯度 |
| M（模型） | PartyM + CryptoMWorker | 后 8 层 + norm（约 1.2GB）+ LoRA-M + 优化器（小）+ sk_M（解密密钥） | 约 4~5GB | 小 | 与 U、S 双向 |
| S（服务器） | PartyS + CryptoSWorker | lm_head V 的 BFV 密文库（N=2048 时约 4GB；N=4096 时约 8~16GB，放内存/磁盘 mmap） | <1~2GB | 4~16GB | 每 token 回传密文行 |

- 切分层数可按各机显存实测调整；BFV 解密在 CPU 上做，S 机的 4060 显存几乎闲置，瓶颈是内存与 CPU。
- 结论：**三台 4060 笔记本跑 1B 方案在硬件上可行**（每机 GPU 占用 <6GB）。8B 基线即使拆三机也放不下（16GB 总显存不足 8B 权重+激活，且逐 token PIR 计算量太大）。

### 7.3 隐私边界反而更清晰

- 当前单机实现靠 `_drop_secret_key()` + 进程/Worker 隔离来保证 sk_M 不越界（`heterogeneous_protocol.py` 顶部注释自述）；三机部署把 U/M/S 放到不同物理机后，sk_M 只存在于 M 机、PRG 种子只存在于 U/S 机、密文库只存在于 S 机，物理隔离天然成立，更利于审计。

### 7.4 通信量与瓶颈（估算）

| 链路 | 每步数据量（batch=1, seq=512, hidden=2048, bf16） |
|------|--------------------------------------------------|
| U ↔ M | H_U 与梯度各约 4MB，合计约 8MB |
| M ↔ S（S3PIR） | 每 token 回传 1 个 BFV 密文（约 65~130KB），512 token 合计约 33~66MB |
| 每步合计 | 约 40~75MB |

- 千兆局域网（有效 ~100MB/s）：每步网络约 0.5~1s，可接受；WiFi（10~30MB/s）：每步 2~7s，若 1 万步则纯网络开销约 6~20 小时，偏慢但能跑。
- 降本手段：seq 用 256（变异文本足够）→ 通信与 PIR 计算直接减半；batch=1 + 梯度累积；后续可做 PIR 请求批量/流水线（当前协议为逐 token 阻塞设计，属未来优化）。
- 真正的大头是 S 机 BFV/PIR 的 CPU 计算与内存带宽，其次才是网络；1B 方案把两者都降到 RTX 4060 笔记本可承受的范围。

### 7.5 需要完成的工作清单（工程范围）

1. 实现 `NetworkBus`（ZeroMQ / gRPC / 自定义 socket 均可），满足 `MessageBus` 接口：peer 路由、step 过期消息清理、超时重连；定义张量序列化（torch 张量 → bytes）。
2. 修复 LegacyIPCStub 的坏路径：`step_val` 调用了不存在的 API（`legacy_ipc_stub.py:542` 注释自证），连同《CipherForge_代码优化分析.md》中的其他阻断缺陷一起修。
3. 把 QueueBus 的 queue 对替换为 NetworkBus 端点，配置三机 IP/端口与角色。
4. Stage 1 密文库在 S 机构建/加载（一次性、CPU 密集）；U/M 机各自加载自己的模型分片。
5. 容错与同步：每步是阻塞协议，对网络抖动敏感，需超时、断线重连、checkpoint 聚合（`gather_checkpoints` 接口已存在）。
6. 先用一台机器 + localhost RPC 冒烟，再上三台真机联调。

### 7.6 结论

- 三台 RTX 4060 笔记本协同微调**资源上可行**（1B + LoRA），且仓库架构本就为三机部署预留（RPC bus 替换 + 三个 worker 入口）。
- 主要成本是工程实现与联调（预计数周级），不是硬件。
- 若想更宽松：ESM-2 150M/35M（用户已允许非 Llama）资源占用更低，但需先做 encoder/分类头适配；建议先跑通 1B Llama 三机版，再视需要做第二路线。

---

## 八、参考资料

1. DYNA: Zhan H, Moore JH, Zhang Z. A disease-specific language model for variant pathogenicity in cardiac and regulatory genomics. Nature Machine Intelligence 7(4):661-671 (2025). DOI 10.1038/s42256-025-01016-8. 数据: Zenodo 12116074 / 13397296; 代码: github.com/zhanglab-aim/DYNA
2. GPN-MSA: Benegas G, et al. A DNA language model based on multispecies alignment predicts the effects of genome-wide variants. Nature Biotechnology 43(12):1960-1965 (2025). DOI 10.1038/s41587-024-02511-w; 预印本 bioRxiv 2023.10.10.561776
3. popEVE: Proteome-wide model for human disease genetics. Nature Genetics (2025). DOI 10.1038/s41588-025-02400-1
4. AlphaMissense: Cheng J, et al. Nature 629, 749-756 (2023)（奠基性参照）
5. PrimateAI-3D: Gao Z, et al. Science 381(6660) (2023). DOI 10.1126/science.abn8197（奠基性参照）
6. BEND: Marin FI, et al. BEND: Benchmarking DNA Language Models on Biologically Meaningful Tasks. ICLR 2024. github.com/frederikkemarin/BEND
7. GeneTuring: Hou W, et al. Benchmarking large language models for genomic knowledge with GeneTuring. Briefings in Bioinformatics 26(5):bbaf492 (2025). DOI 10.1093/bib/bbaf492
8. Li P-H, et al. A large language model framework for literature-based disease-gene association prediction. Briefings in Bioinformatics 26(1):bbaf070 (2025)
9. ClinVar-BERT（预印本）: From Text to Translation: Using Language Models to Prioritize Variants for Clinical Review. medRxiv 2024.12.31.24319792
10. Boosting GPT models for genomics analysis: RAG and fine-tuning. Bioinformatics Advances 5(1):vbaf019 (2025). PMC11842050
11. GP-GPT（预印本）: Lyu Y, et al. arXiv:2409.09825
12. CipherForge 仓库（只读核验）: github.com/changrayhan/CipherForge —— `src/scripts/biotriplex_finetune.py`、`src/model/model_splitting.py`、`src/core/bfv_privselect_v2_adapter.py`、`src/data/dataset.py`
