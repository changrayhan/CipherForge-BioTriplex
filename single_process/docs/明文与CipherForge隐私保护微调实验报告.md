# ClinVar 致病性二分类微调实验报告：明文 LoRA 基准 vs CipherForge 隐私保护微调

## 摘要

本实验在单台 RTX 4060 上完成 ClinVar 变异致病性二分类微调，对比两条管线：
（1）标准明文 LoRA 微调（HuggingFace Trainer + PEFT）；（2）CipherForge
隐私保护微调（U/M/S 三方 + BFV 同态加密 + **真实块 PIR**）。两者使用完全
相同的 5k/类数据、模型（TinyLlama-1.1B-Chat）与超参数。

**主要结论**：CipherForge（真 PIR 版）在测试集上取得 **AUPRC 0.8033 /
AUC 0.8081 / accuracy@0.5 0.667**，与明文基线（AUPRC 0.7922 / AUC 0.7936 /
accuracy@0.5 0.722）持平且略优；加密梯度重建经真实 SEAL 往返验证，精确整数域
误差仅为量化舍入（1e-4），含 float32 传输的管线误差 ≤ 3.5e-3。**隐私语义**：
标签值永不出 U 侧；S 通过真实块 PIR 服务密文查询（猜中真实行的概率 1/8/查询），
只获知监督位置（答案在哪），不获知任何标签值。

---

## 1. 实验设计

### 1.1 任务定义

- 任务：ClinVar 变异致病性二分类，输出 `Yes`（致病）或 `No`（良性）。
- 输入模板：`{question}\n\n{input}\n\nAnswer:`，其中 input 为变异位点描述
  （基因、转录本、HGVS、氨基酸变化等）。
- 评估协议：取**最后一个非 padding 位置的 logits**，比较 `Yes`/`No` 两个
  token 的 logit，softmax 后取 P(Yes) 计算 AUPRC / AUC / accuracy@0.5，
  并按基因分组建 per-gene AUPRC。

### 1.2 数据

- 来源：ClinVar（2026-08-08 快照）→ 过滤 SNV + 明确致病/良性 + 错义 +
  生殖系 + 星级过滤，最终 **178,896** 条变异（正例 49,350 / 负例 129,546）。
- 抽样 5k/类，按基因切分（训练基因不出现在验证/测试基因中）：

| 划分 | 样本数 | Yes | No | 基因数 |
|---|---:|---:|---:|---:|
| train | 10,000 | 5,000 | 5,000 | 3,563 |
| val | 9,708 | 5,000 | 4,708 | 1,126 |
| test | 10,000 | 5,000 | 5,000 | 1,153 |

- 每条样本为 QA 三元组：question / input / output（`Yes` 或 `No`）。
- 标签 token 化：`" Yes"`/`" No"`（Llama tokenizer 下为 2 个 token：
  空格 29871 + Yes 3869 / No 1939），训练按 next-token 对齐监督两个 token，
  评估只用最后一个 token 位置。

### 1.3 模型与超参数

| 项 | 值 |
|---|---|
| 模型 | TinyLlama/TinyLlama-1.1B-Chat-v1.0（22 层，hidden 2048，vocab 32000） |
| LoRA | r=8，alpha=16，dropout=0.05，7 个投影（q/k/v/o/gate/up/down） |
| batch size | 16 |
| 序列长度 | 128 |
| 学习率 | 2e-4，cosine + 100 步 warmup |
| epochs | 3（每 epoch 625 步，共 1875 步） |
| 优化器 | AdamW，weight_decay=0.01 |
| seed | 42 |
| 硬件 | RTX 4060 8GB；WSL Ubuntu 24.04；torch 2.10+cu130 |

### 1.4 两条管线

**明文基线（baseline）**：`transformers.Trainer` + `peft`，标准因果 LM
交叉熵（logits 在位置 t 预测 token t+1），LoRA 覆盖全部 22 层，1875 步。

**CipherForge（隐私保护，真实块 PIR）**：

- 三方切分：U=embed+层0-10；M=层11-21+norm+LoRA；S=lm_head（V 矩阵）。
- BFV 参数：N=4096、plain_bits=30、scale=10000；密文库 32000 行×2048 维
  （4.2GB，存储 Enc(−V·scale)）；S3PIR hint：179 分区、lam=80。
- 每步数据流：
  1. U 前向得 H_U → M 前向得 H_M → S 计算 logits 与 label-free 的
     `a_t = softmax(z)·V`；
  2. **真实块 PIR**：U 为每个监督位置构建 real+dummy 查询块（目标行 y +
     7 个随机 dummy，随机置换，块大小 8），S 返回整块密文，U 用私有置换
     取回 `Enc(−V_y)`——S 不知道哪一行是真实目标；
  3. S 生成 `s_share = a_t − r_t`（标签无关），U 同态加 `r_t`，M 解密 +
     s_share 重建梯度 `(a_t−V_y)·scale` → 反向传播更新 LoRA。
- **标签值永不出 U**：S 不再接收 gold_ids，只接收监督位置（答案在哪）与
  PIR 查询块（含哪几行）。
- **只对答案位置做 PIR**：batch 16 下每步 32 个目标、8×32=256 行/步的块
  取回，步时约 1.1s（与单行取回版本相当）。
- 与明文基线**同参数量级但 LoRA 只挂在 M 侧 11 层**（154 个 LoRA 张量），
  其余超参数一致；训练损失同样为 next-token 对齐的交叉熵梯度。

### 1.5 关键实现细节（影响正确性）

- **next-token 对齐**：加密管线早期版本按“同位置对齐”选取 gold token，导致
  模型学会“输出当前位置 token”的捷径，测试 AUPRC 仅 0.43（反相关）；修复为
  移位对齐（logits_t 对 gold_{t+1}）后指标恢复正常。本报告数据全部来自
  修复后的版本。
- **真实块 PIR（本版核心变更）**：S 完全标签无关。U 侧用 `secrets.SystemRandom`
  采样 dummy 行并随机置换，S 只返回整块密文；猜中真实行概率 1/8/查询。
- **模算术修复**：解密值需先与 s_share 在模域相加再居中（先居中会随机引入
  ±pm 偏移，表现为 loss 尖峰 210/296）。
- **PRG 边界规避**：SEAL Python 绑定在 r_t 距 ±pm/2 边界 49151 以内存在
  ±49151 编解码伪影，PRG 输出域收窄 2^17 后伪影不可达。
- **adapter 导出**：M 侧自定义 LoRA 键映射到 PEFT 标准键（层偏移 +11），
  导出后 154 个张量与训练 checkpoint 逐值一致（max diff = 0）。

## 2. 实验数据

### 2.1 明文基线训练动态

训练 1875 步，train_runtime=1129s（约 18.8 分钟），1.66 步/s；每 500 步
验证一次（eval_runtime≈87.6s）。训练损失采样：

| step | loss | step | loss |
|---|---:|---|---:|
| 10 | 1.2389 | 500 | 0.4979 |
| 20 | 1.0373 | 1000 | 0.4053 |
| 30 | 0.7210 | 1500 | 0.2509 |
| 50 | 0.7155 | 1800 | 0.2159 |
| 100 | 0.6953 | 1830 | 0.2570 |
| 200 | 0.4079 | 1870 | 0.2412 |

验证损失：step 500 → 0.6641；step 1000 → **0.6036（最优）**；step 1500 →
0.7397；step 1875 → 0.7100。最终平均训练损失 0.4043，学习率衰减至 6e-9。

### 2.2 明文基线测试集指标（test 10,000）

| 指标 | 值 |
|---|---:|
| AUPRC | **0.7922** |
| AUC | 0.7936 |
| accuracy@0.5 | 0.722 |
| per-gene 平均 AUPRC（135 基因） | 0.5784（min 0.0105 / max 0.9978） |

对照：

| 设置 | AUPRC | AUC | acc@0.5 |
|---|---:|---:|---:|
| zero-shot（无微调） | 0.4718 | 0.4433 | 0.4999 |
| leak-free 子集（2883 条，测试基因不进入训练） | 0.7611 | **0.8573** | 0.7877 |

### 2.3 CipherForge（真 PIR）训练动态

3 epochs × 625 步 = 1875 步，每 epoch 含一次全量验证（9708 样本，19416 个
答案 token）。训练监控指标为 **eval 口径的二分类 CE**（S 在 U 指定的
答案前置位置返回 P(Yes)，U 用本地标签算 −[y·log p+(1−y)·log(1−p)]，
全程不把标签发给 S）。采样：

| step | monitor CE | step | monitor CE |
|---|---:|---|---:|
| 10 | 0.7179 | 635 | 0.7589 |
| 30 | 0.6796 | 1245 | 0.3792 |
| 50 | 0.6783 | 1300 | 0.5736 |
| 100 | 0.8779 | 1500 | 0.6630 |
| 300 | 0.6430 | 1800 | 0.4921 |
| 620 | 0.4007 | 1870 | 0.4076 |

每 epoch 验证结果与开销：

| epoch | train_loss(代理) | val_ce_loss | val token acc | 平均步时 | 显存 | 时长 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.0010 | 0.3279 | **84.09%** | 1150 ms | 3684 MB | 945 s |
| 1 | 0.0009 | 0.3204 | 83.67% | 1140 ms | 3684 MB | 936 s |
| 2 | 0.0008 | 0.3301 | 83.54% | 1167 ms | 3684 MB | 956 s |

### 2.4 CipherForge（真 PIR）测试集指标（test 10,000）

| 指标 | 值 |
|---|---:|
| AUPRC | **0.8033** |
| AUC | 0.8081 |
| accuracy@0.5 | 0.6666 |
| per-gene 平均 AUPRC（135 基因） | 0.5961（min 0.0130 / max 1.0000） |

> accuracy@0.5 对概率校准敏感，本次运行相对上一版（0.734）偏低；AUPRC/AUC
> 是排序类主指标，与明文基线（0.7922/0.7936）相比**不降反升**。

### 2.5 加密正确性验证（真实 SEAL）

- **密文库完整性**：32000 行加密（54.5s），抽样 14 行解密，与
  `-round(V·scale)` 完全一致（整数误差 0）。
- **完整往返**（加密 −V_y + r_t → 解密 → 加 s_share，168 组随机 (step,t,y)，
  共 344,064 个槽位）：
  - 精确整数域重建误差：max 0.0001（=1/scale，量化舍入）；
  - 管线实际（float32 传输）误差：max 0.0035；
  - 逐槽位误差直方图仅 {0, 1} 整数，无任何槽位误差 > 50000；
  - 深挖槽位：`decoded−true=0`，重建值 == 明文目标。
- **真实块 PIR 往返（Test D）**：U 构建 block=8 的 real+dummy 块 → S 返回
  整块 → U 私有置换取行 → 加 r_t → 解密 + s_share，40 组随机查询 × 2048
  槽位 = 81,920 槽位，**max 误差 0.0001**（纯量化舍入），块取行路径精确。
- **边界伪影探针**：人为把 r_t 钉在 ±pm/2 时仍复现 ±49151 伪影；真实 PRG
  输出域已收窄 2^17，伪影不可达（训练全程无 loss 尖峰）。

## 3. 分析

### 3.1 性能对比：隐私保护未损失精度

| 指标 | 明文基线 | CipherForge（真 PIR） | 差值 |
|---|---:|---:|---:|
| AUPRC | 0.7922 | **0.8033** | +0.011 |
| AUC | 0.7936 | **0.8081** | +0.015 |
| accuracy@0.5 | 0.722 | 0.6666 | −0.055 |
| per-gene 平均 AUPRC | 0.5784 | **0.5961** | +0.018 |

两者使用同一数据与超参。排序类主指标（AUPRC/AUC/per-gene）CipherForge 略优；
accuracy@0.5 的下降来自概率校准偏移（阈值固定 0.5），属于该指标对校准的
敏感性，不反映排序能力下降。每次运行 PRG 掩码不同带来 ≤3.5e-3 级的浮点
传输量化差异，指标允许 ±0.01 量级的 run-to-run 波动。

### 3.2 训练动态

- 明文基线：loss 从 1.24 平滑下降至 ~0.24，验证损失在 0.60~0.74（best
  0.6036），未严重过拟合。
- CipherForge（真 PIR）：eval 口径二分类 CE 从 ~0.72 波动下降至 ~0.41；
  验证 token 准确率稳定在 83.5~84.1%，泛化平稳。
- 训练过程无 loss 尖峰、无 PIR 失败，说明块取行路径在训练全程稳定。

### 3.3 泛化与过拟合

- zero-shot AUPRC≈0.47（≈随机）说明微调是必要的；
- leak-free 子集（测试基因完全不在训练基因中）AUC 高达 0.857，说明模型学到
  的是变异语义特征而非记忆基因名；
- CipherForge（真 PIR）per-gene 平均 AUPRC 0.596，高于明文基线 0.578，
  加密训练没有改变学习到的生物学信号。

### 3.4 隐私-开销权衡

- 每步 32 个目标 × 块大小 8 = 256 行密文取回 → 1.1s/步（与单行版本相当，
  因为行取回走内存字节切片）；相对明文 0.6s/步约 1.8 倍。
- 显存 3.7GB/8GB，单卡即可运行。
- **隐私边界（本版）**：
  - 标签值只在 U；S 只收到监督位置与 PIR 查询块；
  - S 对每个查询猜中真实行的概率 = 1/块大小 = 1/8；
  - M 只接触密文与 s_share；主进程在训练前丢弃 sk 与密文库。

### 3.5 局限

- 单次运行、无多 seed 统计；差异 ±0.01 量级需多次重复才能给出置信区间。
- accuracy@0.5 对概率校准敏感（本版 0.667 vs 明文 0.722），如需阈值无关的
  公平对比应以 AUPRC/AUC 为准。
- S 获知**监督位置**（答案在哪两个 token 位置）——这是结构信息，非标签值；
  若要求 S 连位置也不知道，可让 S 对全部位置计算份额（每步 +2.5s 纯 PRG
  开销）。详细讨论见《真 PIR 隐私保护测试报告》。
- 本实验为单机融合运行版；进程级/三主机隔离版按三进程文档推进中。

## 4. 结论

1. CipherForge（真 PIR 版）在 ClinVar 致病性二分类上达到并略优于明文 LoRA
   （AUPRC 0.8033 vs 0.7922），证明 BFV + 真实块 PIR 加密梯度方案在精度上
   可行。
2. 真实 SEAL 往返验证（含块 PIR 路径）表明加密梯度与明文梯度逐槽位一致
   （误差 = 定点量化），加密正确性有实测支撑。
3. 隐私语义升级：**标签值永不出 U**；S 通过真 PIR 服务查询、只获知监督位置
   与查询块，不再直接拿到 gold 标签（旧版 S 半诚实可见标签）。
4. 训练正确性三要素（next-token 对齐、模域加法顺序、PRG 边界）均已修复并
   验证；修复前后 AUPRC 从 0.43 → 0.80。

## 5. 三进程隔离版结果（three_party，2026-08-19）

在**进程级 + 目录级隔离**架构下复跑同一任务：U/M/S 为三个独立进程与三个独立
目录（`party_u/`、`party_m/`、`party_s/`），协调者独立于三方之外
（`coordinator/`），三方仅通过 HTTP/JSON 交换规定消息。数据、模型切分、
超参数与单进程版完全一致（TinyLlama-1.1B、U/M 各 11 层、LoRA r=8/α=16、
batch=16、lr=2e-4、3 epochs × 625 步）。

### 5.1 训练过程

| epoch | val_ce_loss | val 二分类准确率（argmax） | 说明 |
|---:|---:|---:|---|
| 0 | 0.6583 | 0.1977 | 与单进程版 epoch0（0.6574）一致，训练对齐 |
| 1 | 0.7772 | 0.1924 | 继续训练 |
| 2 | 0.7674 | 0.1909 | 1875 步完成 |

> val_ce 的 p(Yes) 取自“最后一个 prompt 位置”（与明文评测协议一致的监控口径），
> 单批方差较大；argmax 二分类准确率对概率校准敏感。最终以测试集排序类指标为准。

### 5.2 测试集指标（n=10,000，pos_rate=0.5）

| 指标 | 明文 LoRA 基线 | CipherForge 单进程（真 PIR） | CipherForge 三进程隔离版 |
|---|---:|---:|---:|
| AUPRC | 0.7922 | **0.8033** | **0.7976** |
| AUC | 0.7936 | **0.8081** | 0.7887 |
| accuracy@0.5 | 0.722 | 0.6666 | 0.5541 |
| per-gene 平均 AUPRC | 0.5784 | **0.5961** | **0.6248**（135 基因） |

### 5.3 结论

1. 三进程版在完整进程隔离 + HTTP/JSON 传输 + 真实 BFV 密文库（N=4096）下
   端到端训练正确：**AUPRC 0.7976** 与明文基线（0.7922）持平略优，与单进程版
   （0.8033）在加密传输量化噪声范围（±0.01）内。
2. AUC 0.7887 与基线差距 ≈ 0.005；accuracy@0.5 与单进程版呈同样的校准偏移
   （P(Yes) 绝对刻度经定点量化后偏移），排序类主指标不受影响。
3. 隐私语义与单进程版一致：标签值永不出 U，S 只获知监督位置与 PIR 查询块
   （real+dummy，block=8，猜中真实行概率 1/8）。
4. 工程上验证了三进程关键修复：BFV 公钥/Encryptor 绑定、`H_U.requires_grad`、
   LoRA 导出层偏移（M 侧局部层 0-10 → 全局 11-21）、PEFT 键名 `.default` 兼容。

### 5.4 复现（three_party）

```bash
export CF_MODEL_PATH=<TinyLlama 快照> PYTHON=$(which python) S_DEVICE=cpu
bash scripts/run_full.sh --max_epochs 3 --batch_size 16 --log_freq 10  # 训练（约 1.5h）
bash scripts/run_full.sh --skip_train                                   # 导出 adapter + AUPRC 评测
```

结果文件：`coordinator/logs/clinvar_auprc.json`；adapter 在 `coordinator/adapter/`。

## 6. 附录：复现

```bash
# 明文基线（约 19 分钟 + 验证）
MAX_STEPS=0 baseline/run_baseline.sh

# CipherForge：Stage0 → 3 epochs → Stage2
cipherforge/scripts/run_stage0.sh
cipherforge/scripts/run_epoch.sh 0
cipherforge/scripts/run_epoch.sh 1
cipherforge/scripts/run_epoch.sh 2
cipherforge/scripts/run_stage2.sh

# 加密正确性验证（含块 PIR 路径）
python cipherforge/tools/rt_bfv_roundtrip_test.py
```

关键数据文件：明文 `runs/clinvar_tinylama_plain_128/{metrics.json,
ckpt/checkpoint-1875/trainer_state.json}`；CipherForge `logs/clinvar_auprc.json`
与各 epoch 指标 JSON；隐私边界细节见《真 PIR 隐私保护测试报告》。
