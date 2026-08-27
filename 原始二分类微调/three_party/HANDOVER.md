# CipherForge 三进程隔离版 — 交接文档

> 状态：已实现并通过全量验证（2026-08-19）。本文档为设计/交接底稿，
> 实现细节以 README.md 与 docs/ 为准；协调者已独立为 coordinator/（原 party_u/coordinator.py 路径已废弃）。按本文档可直接开工；预计工作量集中在传输层与
> 协调者（占 70%），Party 本体与密码学核心全部复用单机版。

## 1. 目标

把 ClinVar 致病性二分类的 CipherForge 隐私保护微调，从“单进程融合运行”
改造为 **三个独立操作系统进程**（对应 U / M / S），满足：

- **进程隔离**：三方互不 import、互不访问对方对象；只通过 TCP/IPC 交换
  规定好的消息。
- **目录隔离**：三个目录 `party_u/`、`party_m/`、`party_s/` 各自只放本方的
  代码与数据；共享的只读库放 `shared/`。
- **数据隔离**：明文 QA 只存在于 U；模型分片/私钥/checkpoint 只在 M；
  V 矩阵/密文库/hints 只在 S。
- **可迁移**：传输层用 TCP，将来三台主机（每台 RTX 4060）部署时只改 IP/端口。
- **功能正确**：本机先实现 + 冒烟，loss_ce 下降、验证指标正常、adapter 可导出，
  与单机版（AUPRC 0.7967）对照差距在噪声范围内。

## 2. 总体架构

采用 **星型拓扑：U 进程兼任协调者**（数据所有者在 U，符合隐私语义）。

```text
┌─────────────────────────────┐
│ party_u 进程（协调者）        │
│  ├─ 训练数据 data/qa/*.jsonl │
│  ├─ Trainer + RemoteProtocol │
│  ├─ PartyU（embed+层0-10）    │
│  └─ TCP 客户端（连 M 和 S）   │
└──────┬──────────────┬───────┘
       │ H_U / 密文   │ H_M / gold
       ▼              ▼
┌──────────────┐  ┌──────────────┐
│ party_m 进程  │  │ party_s 进程  │
│ PartyM+LoRA  │  │ PartyS(V)    │
│ sk/checkpoint│  │ 密文库+hints │
│ CryptoMWorker│  │ CryptoSWorker│
└──────────────┘  └──────────────┘
```

每步训练的数据流（与单机版数学完全一致）：

1. U：本地 `PartyU.forward_train(batch)` → `H_U`；
2. U→M：`FORWARD_M(H_U, attention_mask)` → `H_M`；
3. U→S：`PROCESS_S(H_M, gold_ids, step)` → `{s3pir_responses, s_shares,
   valid_mask, valid_indices, gold_ce}`；
4. U：本地 `PartyU.privselect_and_recover_dispatch(responses)`（CryptoUWorker
   同态加 `r_t`）→ `ct_list`；
5. U→M：`DECRYPT_UPDATE(ct_list, s_shares, valid_mask, valid_indices,
   expected_shape, step)` → `{loss, gpu_mem}`；
6. U 本地 `party_u.step_optimizer()` + 记录指标。

验证/测试步：U→M `FORWARD_M`，U→S `VALID(H_M, attention_mask, gold_ids)`
→ `{predictions, labels_letters, logits, labels_tensor}`，U 汇总指标。

## 3. 复用清单（来自单机版 `clinvar-submit/cipherforge/src`）

直接复用（原样拷贝）：

| 文件 | 说明 |
|---|---|
| `shared/parties/party_u.py` | 仅保留 forward_train / forward_val / privselect / step_optimizer 相关 |
| `shared/parties/party_m.py` | M 分片 + LoRA + 解密/更新（去掉 CryptoMWorker 绑定逻辑，改由进程内持有） |
| `shared/parties/party_s.py` | S 分片 + logits/a_t/PIR 选择 |
| `shared/parties/crypto_workers/*` | CryptoU/M/S worker + pool（在各自进程内跑） |
| `shared/model/model_splitting.py` | 分片加载 + 自定义 LoRA（含 dropout、norm 冻结） |
| `shared/core/bfv_privselect_v2_adapter.py` | BFV 后端、密钥、密文库、PRG |
| `shared/core/s3pir_hints.py` | hint 表 |
| `shared/core/key_remapping.py` | adapter 导出键映射 |
| `shared/data/clinvar_dataset.py` | QA 数据集（next-token 标签） |
| `shared/training/trainer.py` | Trainer（含 ClinVar 验证分支、断点续跑） |
| `shared/scripts/biotriplex_finetune.py` | `save_peft_adapter`、`_patch_trainer_for_max_steps`（可裁剪） |
| `shared/scripts/stage0_build_db.py` | 密钥 + 密文库 + 验证 |

需要改造/新增：

| 文件 | 说明 |
|---|---|
| `shared/transport.py`（新） | 帧协议 + TCP client/server（见 §5） |
| `party_u/coordinator.py`（新） | Trainer 驱动 + `RemoteProtocol`（实现 `step_train_chunked` / `step_val` / `gather_checkpoints` / `load_checkpoints` / `shutdown`） |
| `party_m/main_m.py`（新） | M 服务进程：TCP server + PartyM + CryptoMWorker pool |
| `party_s/main_s.py`（新） | S 服务进程：TCP server + PartyS + CryptoSWorker pool |
| `scripts/run_smoke.sh`（新） | 启动三进程 + 跑 50 步 + 断言 |
| `docs/protocol.md`（新） | 消息/帧格式定稿 |

## 4. 三目录与数据边界（强制约束）

```text
cipherforge-three-party/
├── shared/          # 三方共用只读库（唯一允许被 import 的“他人”代码）
├── party_u/
│   ├── main_u.py（协调者，包含 Trainer + RemoteProtocol）
│   └── data/        # 只有 QA jsonl + splits/stats；无模型权重、无 sk、无密文库
├── party_m/
│   ├── main_m.py
│   ├── keys/        # bfv_sk.bin（唯一持有私钥的进程目录）、bfv_pk.bin
│   ├── checkpoints/ # LoRA + optimizer + scheduler 状态
│   └── model/       # M 分片缓存（由 model_splitting 加载共享模型）
├── party_s/
│   ├── main_s.py
│   ├── db/          # bfv_ct_db_*.bin、s3pir_hints/（只 S 持有密文库）
│   └── model/       # lm_head 分片（V 矩阵）
├── scripts/         # build_stage0.sh / run_smoke.sh
└── docs/            # architecture.md / protocol.md / deployment.md / smoke_test.md
```

**验收约束**：

- `party_u/` 内不得出现 `bfv_sk.bin`、`bfv_ct_db_*.bin`、`lm_head` 分片；
- `party_m/` 内不得出现 QA jsonl、密文库、PRG seed；
- `party_s/` 内不得出现 QA jsonl、sk、LoRA checkpoint；
- 冒烟测试加一项 import 审计：每个 main 进程 import 的模块白名单（U 可 import
  shared+自身；M 不可 import `party_u`/`party_s` 的模块，反之亦然）。

## 5. IPC 协议设计（草案，实现时定稿到 `docs/protocol.md`）

### 5.1 传输与帧

- TCP，端口规划：M=40001，S=40002（本机）；三主机部署时改 IP。
- 帧：`[4B big-endian length][msgpack header][可选 raw tensor bytes]`
  - header：`{type, msg_id, step, shapes: {name: [shape, dtype]}, meta}`
  - 张量：numpy `.tobytes()`（bf16 保持位级一致），大消息（H_U/H_M/ct_list）
    单帧传输，不再分包（单帧最大 ~10MB，TCP 自动分片）。
- 心跳：空闲 >1s 发 `PING`；超时 30s 判定断线，训练终止并报错。
- 关闭：`SHUTDOWN` 双向确认。

### 5.2 消息目录（U=协调者客户端）

| 方向 | type | 载荷 | 返回 |
|---|---|---|---|
| U→M | `HELLO` | 协议版本、BFV 参数 | `{ok, version}` |
| U→S | `HELLO` | 协议版本、BFV 参数、`prg_seed`（仅 U/S 共享） | `{ok, version}` |
| U→M | `FORWARD_M` | `H_U`(bf16), `attention_mask` | `{H_M}` |
| U→S | `PROCESS_S` | `H_M`, `gold_ids`(int64, 含 -100), `step` | `{s3pir_responses[], s_shares[], valid_mask, valid_indices, gold_ce, n_pir}` |
| U→M | `DECRYPT_UPDATE` | `ct_list[]`, `s_shares[]`, `valid_mask`, `valid_indices`, `expected_shape`, `step` | `{loss, gpu_mem_mb}` |
| U→S | `VALID` | `H_M`, `attention_mask`, `gold_ids?` | `{predictions, labels, labels_tensor, logits}` |
| U→M | `GATHER_CHECKPOINT` | — | `{lora_state, optimizer_state, scheduler_state}` |
| U→M | `LOAD_CHECKPOINT` | `{lora_state, optimizer_state, scheduler_state}` | `{ok}` |
| U→M/S | `SHUTDOWN` | — | `{ok}` |

> 关键点：**gold_ids 只发给 S**（半诚实假设，S 需要标签选密文行）；**M 永远
> 收不到 gold_ids/明文输入**；**U/S 之间共享 prg_seed，M 永远拿不到**。

### 5.3 张量序列化约定

- 一律 `np.ascontiguousarray(t).astype(dtype)` → `tobytes()`；header 记录
  shape/dtype/endianness；接收端 `np.frombuffer(...).reshape(shape)`。
- `H_U`/`H_M`：`bfloat16`，(B,S,2048)。
- 密文/响应：bytes 数组（长度前缀列表）。
- `gold_ids`：int64，(B,S)，-100 为 ignore。

## 6. 实现阶段（建议顺序）

| 阶段 | 任务 | 产出/验收 |
|---|---|---|
| P0 | 拷贝 `shared/`（§3 清单），`python -m compileall` | 共享库可 import |
| P1 | `shared/transport.py`：帧编解码 + TCP client/server + 单测 | 三进程互发 H_U/密文字节级一致；`tests/test_transport.py` 通过 |
| P2 | `party_s/main_s.py` + `party_m/main_m.py` 服务骨架 | HELLO/心跳/SHUTDOWN 可通 |
| P3 | M 服务：`FORWARD_M`、`DECRYPT_UPDATE`；S 服务：`PROCESS_S`、`VALID` | 单步往返可用（先用合成张量） |
| P4 | `party_u/coordinator.py`：`RemoteProtocol` 接入 Trainer | 与单机版同 seed 下 loss_ce 曲线形状一致 |
| P5 | `scripts/run_smoke.sh` + 冒烟断言 | 见 §7 |
| P6 | 文档：protocol/architecture/deployment/smoke | 三主机可按文档部署 |

## 7. 本机冒烟方案与验收

单机三进程（一卡 4060 8GB）：

```bash
scripts/run_smoke.sh   # 启动 party_s + party_m，再启动 party_u（协调者），50 步
```

显存预案（三进程三个 CUDA 上下文，实测可能超 8GB）：

1. 冒烟参数 `batch=2, max_seq_len=64`；
2. 若仍 OOM：`party_s` 以 `--device cpu` 启动（S 只有 lm_head 与 matmul，慢但正确）；
3. 再不行：`CUDA_VISIBLE_DEVICES=""` 让 M 也走 CPU（仅冒烟，验证正确性）。

验收标准：

- 三进程握手成功，50 步完成无断线；
- `loss_ce` 从 ~4.7 下降（50 步内明显下降，参考单机版曲线）；
- 小验证集 token 准确率 > 80%（batch2 下可用 val 子集）；
- `GATHER_CHECKPOINT` 能取回 154 个 LoRA 张量并导出 PEFT adapter；
- import 审计通过（§4 约束）；
- （可选）对照单机版跑 3 epochs，AUPRC 差距 ≤ 0.02。

## 8. 迁移到三台 RTX 4060 主机

1. 三台装 Ubuntu 24.04 + 同一 conda 环境（依赖同单机版）；
2. M 主机：`scripts/build_stage0.sh` 生成密钥；`bfv_pk.bin` 分发给 U/S；
   S 主机用 pk + lm_head 构建 4.2GB 密文库；
3. prg_seed：U 生成后经 **TLS 信道**发给 S（三主机部署必须加 TLS，见 §10）；
4. 每台只拷贝自己目录 + `shared/` + 模型分片；
5. `party_m`/`party_s` 分别以 `--host <IP> --port <PORT>` 启动；
   `party_u/coordinator.py` 以 `--m_host --s_host` 启动；
6. 千兆有线网络，预计步时 1.3~1.8s（WiFi 2~3s，仅演示用）。

## 9. 风险与待决策项

- **S 看到标签**：设计内半诚实假设；若要求 S 也不知标签，需 OT/FHE 级改造
  （研究级），本期不做。
- **H_U/H_M 明文过网**：U→M、U→S 传的是隐状态（设计内的暴露面），报告需说明。
- **断点续跑**：LoRA/优化器在 M，数据迭代位置在 U，PRG 计数器按 (step,t_flat)
  派生、三方都从 checkpoint 的 global_step 恢复即可；`GATHER_CHECKPOINT`/
  `LOAD_CHECKPOINT` 已覆盖。
- **确定性**：bf16 位级一致，但每次运行 PRG seed 不同，指标允许小波动。
- **单卡 OOM**：冒烟用小 batch，正式三主机部署无此问题。
- 待决策：是否加 TLS（三主机必须）；是否把 `gold_ids` 对 M 做完整性保护；
  端口/协议版本号管理。

## 10. 参考资料

- 单机版仓库：`clinvar-submit/`（可复用源码、配置、数据、README）。
- 单机版最终结果：明文 AUPRC 0.7922 / CipherForge 0.7967。
- 参考实现：`clinvar-submit/cipherforge/src/parties/heterogeneous_protocol.py`
  （进程内数据流）、`legacy_ipc_stub.py`（三进程 IPC 旧骨架，仅参考不直接复用）。
- BFV 参数：N=4096 / plain_bits=30 / scale=10000；密文库 32000×2048；
  S3PIR：179 分区、lam=80。

---

## 11. 更新附记（2026-08-19：真 PIR v2 + RMS-PIR 备用方案）

本文档第 1-10 节为早期设计底稿；当前实现以 README.md 与 docs/ 为准，关键更新：

1. **传输层已定稿为 HTTP/JSON**（`shared/node_server.py` + `shared/remote_protocol.py`，
   而非早期草案的 TCP/msgpack），三方目录隔离与“协调者独立于 party_u/”已落地。
2. **标签保护已升级为真块 PIR v2**（旧设计 §9 “S 看到标签（半诚实假设）”已废弃）：
   U 只发“监督位置 + 查询块索引集合”；dummy 按真实标签边际采样、block=64、
   真假查询块 8:2、按唯一索引去重取行。
3. **新增 RMS-PIR 备用模式并升级为 v2 职责划分**（`pir_mode: "rms"`，论文
   Ren-Mughees-Sun CCS'24 两服务器变体）：`shared/core/rms_pir.py`（hint 池/
   子集/补充）+ U 端本地密文库副本（`party_u/db/`，offline 角色）+ S 端
   `rms_parity` 明文聚合应答（online 角色）；S 从不见 hint 状态，多查询隐私
   按论文成立；真实 SEAL 往返误差 0.000000，三进程冒烟通过。
4. 本机冒烟/全量运行均在 WSL（Ubuntu 24.04）完成；GitHub 归档仓库
   `CipherForge-ClinVar` 的 `three_party/` 目录与本文档所在工作树保持同步。
5. **差分隐私模块已迁入并默认开启**（2026-08-19）：
   `shared/core/dchi_privacy.py`（参考实现 `CipherForgeCode/SLG-HE-PIR`），
   U 端 `H_U` 出域前注入 dχ-隐私噪声；做了 ClinVar 二分类标签映射与噪声采样
   向量化两处适配；冒烟通过（详见真PIR测试报告 §7）。

详见：
- [docs/块PIR与S3PIR与RMS-PIR方案对比分析.md](docs/块PIR与S3PIR与RMS-PIR方案对比分析.md)
- [docs/真PIR隐私保护测试报告.md](docs/真PIR隐私保护测试报告.md)
- [docs/04-备用方案与降级预案.md](docs/04-备用方案与降级预案.md)
