# 02 · 交接文档：RMS-PIR 变体调试进度与已排除方向

> **交接对象**：继续处理 RMS-PIR 变体"不学习"问题的开发者 / AI 助手。
> **后端位置**：远程服务器 `three_party/`（U/M/S 节点 + coordinator + 演枢台平台）。
> **编写日期**：2026-08-21
> **当前状态**：RMS-PIR **不再发散**（3 个确定性 bug 已修复），但**完整一个 epoch 后仍不学习**（val_AUPRC≈0.51，Block≈0.74）。根因未完全定位，已排除大量方向（见第 4 节）。

---

## 1. 背景与目标

三进程隐私微调包含三种模式：明文 LoRA 基准、三进程 RMS-PIR、三进程 Block-PIR。
现象：**Block-PIR 可正常微调**（3 epoch 后 test AUPRC≈0.80、val AUPRC≈0.74），而 **RMS-PIR 不学习**（完整 1 epoch 后 val_AUPRC≈0.51、test AUPRC≈0.53，接近随机 0.5）。

目标：定位并修复 RMS-PIR 不学习的根因，使其达到与 Block 相当的微调效果。

---

## 2. 已完成的修复（请保留，勿回退）

以下修改已在服务器生效并验证，属于**正式修复**：

| 文件 | 修改 | 验证结果 |
|---|---|---|
| `coordinator/main.py` | RMS 初始 hint 池的 `known_labels` 改用训练标签边际的真实 token（`pir_dummy_weights`），而非 `tokenizer(" ")[0]` | 修复后 **0 次 pop_hint 报错**（此前训练开始即连续报错，▁=29871 占标签 50% 却零覆盖） |
| `shared/parties/party_u.py` | `_auto_replenish`：补货 hint 的 parity 改为对 `picked_half + [bad_y]` 求本地新鲜 parity（补齐 Enc(-V_y)），并递增 `store.next_j` | 消除了"恢复值为 0"的梯度污染与 hint ID 碰撞 |
| `shared/parties/crypto_workers/crypto_u.py` | `_rms_recover_and_mask_v2`：补货 hint 的 `new_parity` 改为**直接从本地密文库求 `picked_half + [y]` 的新鲜 parity**（不再 `recovered + half_ct`） | 消除了补货 hint 噪声随代数线性累积；修复后 loss_ce 稳定在 0.7–1.5、g_absmax 稳定在 ~2（此前发散到 loss_ce=10、g_absmax=47） |
| `shared/parties/crypto_workers/crypto_u.py` | `rms_query_mask` 的本地 parity 按 token 分片提交多个 worker 并行 | 性能 |
| `party_s/main_s.py` | `act_rms_parity` 缓存 V 的 CPU 副本（`V_np`），避免每步 2.6GB GPU→CPU 大拷贝 | 性能 |
| `coordinator/three_party_config_rms.json` / `_rms_dp.json` | `N_CRYPTO_U_WORKERS` 1 → 4 | 性能 |
| `scripts/run_finetune_menu.py` 等 | U 节点服务 + 平台启动、skip_train、保活、macro_f1 展示等（见 01 号接口文档） | 平台/接口验收通过 |
| `coordinator/evaluate_auprc.py` | 新增 `macro_f1` | 三模式统一输出 |

**性能结论**：RMS 单步耗时 3.7s → 2.3–2.8s（Block 约 1.65s）。

---

## 3. 当前症状（复现路径）

一键复现（全 epoch，约 30–35 分钟）：
```bash
cd /root/CipherForge/CipherForge-ClinVar/three_party
export CF_MODE=2 CF_EPOCHS=1 CF_BATCH_SIZE=16 CF_MAX_STEPS=0 \
  CF_OUT_ROOT=/tmp/rms-repro CF_KEEP_SERVICES=0
bash scripts/run_finetune_menu.sh
```

结果（`logs/training_metrics_*.json` 与 `test_metrics.json`）：
- epoch 0：`train_loss≈0.026`（梯度范数，稳定不降）、`val_AUPRC≈0.51`、`val_acc≈0.51`
- 对照 Block（`CF_MODE=3`）：`val_AUPRC≈0.74`、`val_acc≈0.66`，train_loss 收敛到 0.001
- RMS 的 `loss_ce`（监控 CE）稳定在 0.7–1.5（不再爆炸），g_absmax 稳定 ~2（不再发散）

结论：**不再发散，但不学习**——梯度稳定却无学习信号。

---

## 4. 已排除的方向（确保正确，附证据）

> 以下每一项都经过代码审查 + 实验验证，可以放心排除。

### 4.1 初始 hint 池 token 不匹配（已修复，不再视为根因）
- 现象：`tokenizer(" ")`=259，而标签中实际是 ▁=29871（来自 `" Yes"`→[29871,3869]），29871 占标签 50%，初始池对其零覆盖 → 训练一开始 `pop_hint(29871) pool empty`。
- 处理：已修复（见第 2 节）；当前运行 **0 次报错**。

### 4.2 `_auto_replenish` 补货 parity 缺 Enc(-V_y)（已修复）
- 现象：池空时合成的 hint 只含半区 parity，之后用它恢复得到 0 → 梯度污染。
- 处理：已修复；该路径当前 0 触发。

### 4.3 补货 hint 噪声随代数累积（已修复，发散根因之一）
- 现象：旧实现 `new_parity = recovered + half_ct` 继承旧 hint 噪声，几十代后超出 BFV 解密预算 → M 解密退化 → loss_ce 爆炸到 10、g_absmax 涨到 47。
- 处理：补货 parity 改为本地新鲜求和（噪声有界）；修复后指标稳定。

### 4.4 RMS 恢复 Enc(-V_y) 的数学精确性（确认正确）
用正确密钥（crypto 目录 `m_keys/bfv_sk.bin`，注意**不是** `party_m/keys/` 的旧密钥）做了解密级验证：
- 单行密文库行 vs 明文期望：误差 **0**
- 初始 hint parity（`hint_half_rows` 构造，本地 81 行密文和）：**0/120 错误**
- 明文级恢复 `H − q` vs 直接密文库行：误差 **0**（y 为 extra 或 rows_a 行值均验证）
- 密文级恢复 `H_ct − q_ct` + 掩码后解密：**0/40 错误，worst=0**
- S 端 V（GPU bf16）与建库 V（CPU bf16）：`max abs diff = 0.0`
- U 本地密文库副本（`rms_db/…bin`）与 S 建库（`enc_db/…bin`）：**sha256 完全一致**
- 掩码一致性：U 与 S 前 8 槽 r_t 完全一致（差异 ~30–70 = 真实梯度 a_t·scale）
- 噪声预算：1/10/30/60/81 行密文求和解密均精确

→ **加密/恢复/掩码/解密环节本身是正确的**。

### 4.5 DP 配置差异（排除）
- 对比 RMS 与 Block 实际运行配置：`pir_mode` 之外字段完全相同（`dp_enable=true, dp_alpha=0.03, dp_eta0=1500, dp_answer_beta=0.5` 等）。

### 4.6 评测口径错位（独立问题，不是不学习的根因）
- `evaluate_auprc.py` 在"最后一个非 pad 位置"打分，而训练/验证在 **▁(first) 位置**打分，导致所有模式官方测试指标系统性偏差（Block 正确位置 test acc≈0.72 而非 0.50）。
- 这是另一个待修复问题（见第 6 节），但与"RMS 不学习"无关——RMS 在正确位置打分同样随机（first-pos AUPRC≈0.50）。

---

## 5. 关键实验与当前最接近的线索

### 5.1 隔离实验（决定性发现）
在 `crypto_u._rms_recover_and_mask_v2` 中把掩码来源改为 **`use_fresh=True`（直接用 U 本地密文库对 `[y]` 求 Enc(-V_y)，跳过 hint 恢复）**：
- 32/32 token 的 M 端重建残差全部 ≤2220（此前 ~半数 token 出现 9000–20000 的残差）
- `g_absmax=0.222、g_meanabs=0.000278` —— **与 Block 完全一致**

→ **artifact 必然来自 hint 恢复路径（`hint_parity − q_parity`）在生产环境中的某个环节**，而隔离测试（第 4.4 节）显示该数学路径正确。

### 5.2 尚未解释的差异点
- 生产环境 M 端重建在**随机槽位**（如 1771、991、1709、795 等，随 token 而异）出现 9000–20000 int 的残差，约半数 token。
- 同一槽位在不同 token 上反复出现（如 1771 出现在 tf=69 与 tf=70），暗示与 **V 矩阵特定列** 或 **hint 文件/seed 特定组合** 相关。
- 生产 hint 由随机 seed 生成且 **seed 未持久化**，无法事后复现同一批 hint——这是排查的最大障碍。

### 5.3 进行中/被中断的实验（供接续）
为区分"U/S 掩码不一致"还是"恢复值错误"，已在代码中加入文件日志：
- `crypto_u.py`：向 `/tmp/urt.log` 写 `r_t[0:8]` 与 `r_t[1760:1780]`
- `crypto_s.py`：向 `/tmp/srt.log` 写 `s_share[0:8]` 与 `s_share[1760:1780]`

接续方法：恢复第 7 节的临时插桩后跑 1 步（`CF_MODE=2 CF_MAX_STEPS=1`），对比 artifact 槽位上 U 的 r_t 与 S 的 −s_share 是否一致。

---

## 6. 待办事项（按优先级）

1. **完成 artifact 定位**：对比 URT/SRT 在 artifact 槽位的值 → 若掩码不一致则查 PRG 参数/seed；若一致则查恢复值（需要持久化 seed 复现生产 hint）。
2. **候选修复**：`use_fresh=True`（掩码直接取本地密文库 Enc(-V_y)）已验证可行且与 Block 完全一致；但这改变了 RMS-PIR 的"在线恢复"语义，需与论文安全叙事对齐后再决定是否采用。
3. **修复评测口径**：`evaluate_auprc.py` 改为在 ▁(first) 位置打分（与训练/验证一致），并重跑三模式对比。
4. **还原临时插桩**（见第 7 节）并清理 `/tmp` 下的调试脚本。
5. 全量验证：RMS 与 Block 各 3 epoch 对比（正确口径下）。

---

## 7. 临时改动清单（**必须还原**，勿作为正式修复提交）

> 这些是为了诊断加入的临时插桩/开关，问题解决后必须移除。

| 文件 | 临时内容 | 处理 |
|---|---|---|
| `shared/parties/party_m.py` | `TMPDBG` / `TMPDBG2` 日志（step 0 打印 masked/share/diff 与每 token 最大残差） | 删除 |
| `shared/parties/crypto_workers/crypto_u.py` | `use_fresh` 分支（实验开关，当前 `False`）；向 `/tmp/urt.log` 写掩码 | 删除（并行分片与新鲜补货 parity 保留） |
| `shared/parties/crypto_workers/crypto_s.py` | 向 `/tmp/srt.log` 写份额 | 删除 |
| `shared/parties/party_u.py` | items 中的 `"use_fresh": False` 字段 | 删除 |
| `shared/training/trainer.py` | `train_loader` 的 `shuffle=False`（为同批对比临时改的） | **改回 `shuffle=True`** |

服务器 `/tmp` 下的调试脚本（可清理）：`score_pos_test.py`、`rms_verify*.py`、`hint_verify*.py`、`hint_dbg*.py`、`noise_probe.py`、`mask_probe.py`、`rename_doc.py` 等。

---

## 8. 关键路径与产物

| 路径 | 说明 |
|---|---|
| `three_party/` | 全部代码 |
| `coordinator/three_party_config_rms.json` | RMS 配置（`N_CRYPTO_U_WORKERS=4`） |
| `/root/autodl-tmp/CipherForge-final-test/crypto/three-party-rms-pir/` | 加密产物（m_keys / enc_db / rms_db / rms_hints） |
| `/root/CipherForge/final-test-data/` | 全量三模式运行产物 |
| `/tmp/rms-*` | 近期诊断运行产物（rms-fix2、rms-epoch1-fullval、samebatch、rms-maskcmp* 等） |
| `docs/07-接口规范-当前实现.md` | 当前后端接口规范（面向前端/AI 助手） |

---

## 9. 给接续者的建议

1. **先读本文档第 4 节**，不要重复排查已排除的方向（尤其是"恢复数学不正确"——已用解密级实验排除）。
2. 排查时**务必持久化 RMS seed**（`coordinator/main.py` 中 `rms_seed = os.urandom(32)` 可改为可配置/记录），否则无法复现同一批 hint。
3. 优先完成 5.3 的掩码对比实验；若掩码一致，则把注意力放在**生产 hint parity 文件**（`rms_hints/hint_*.ct`）与其对应 row_list 的一致性上（seed 持久化后可验证）。
4. 若时间紧张，`use_fresh=True` 是经过验证的可用回退方案（效果与 Block 一致），但需评估对 RMS-PIR 安全叙事的影响。
