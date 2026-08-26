# CipherForge（SLG-HE-PIR）代码优化分析

> 分析方式：GitHub 只读浏览 + API 逐个读取源码文件，全部结论均以代码为证（含文件与行号）。
> 范围：`main` 分支 `SLG-HE-PIR/src/` 34 个核心文件 + 关键测试（约 14,000 行）。
> 原则：只列出**已用代码验证**的优化项；影响幅度无法离线实测的项已标注"需运行时确认"。

---

## 一、最重要的性能问题（优先级最高）

### 1. CryptoWorker 进程池"并行"实际是串行（8 个 worker 只有一个干活）

**位置**：`src/parties/crypto_workers/pool.py:74-87`（`submit`），调用点 `party_u.py:279`、`party_m.py:376`、`party_s.py:277`。

**证据**：
```python
async_result = self._pool.apply_async(
    base.run_worker_request, (self.worker_cls_path, payload)
)
return async_result.get()
```
`apply_async` 只派发**一个任务**给**一个 worker**；而每个请求的 payload 是整批 token（`s3pir_responses` / `ct_list`），`handle_request` 内部是单进程顺序循环。全仓库没有任何把 token 列表拆分到多 worker 的逻辑（`submit_async` 定义了但从未被调用；`privselect_and_recover_parallel` 也只是转发到 `submit`）。

**后果**：`N_CRYPTO_U_WORKERS=8 / N_CRYPTO_M_WORKERS=8` 实际只用了 1 个进程；U 加掩码、M 解密这两个 CPU 密集阶段被白白串行化，每步等待时间就是"单进程处理全部 token"的时间。

**建议（正确且必要）**：在 `submit` 里把 payload 中的 token 列表按 `n_workers` 切分，用 `Pool.starmap`/`map` 并行处理，再按原顺序合并结果；或在 `step_train_chunked` 中改用 `submit_async` 让 U 的各 chunk 与 M 的解密重叠。代码语义不变（每 token 输出顺序与输入一致），可复用现有 `handle_request`。

**预期收益**：U/M 密码学阶段约 3-8 倍加速（受 SEAL 多进程扩展性限制）。

### 2. PRG 掩码生成逐元素 SHA-256（每 token 4096 次哈希，U/S 各算一遍）

**位置**：`src/core/bfv_privselect_v2_adapter.py:248-286`（`_prf_block` / `generate_mask_ints`）。

**证据**：`_prf_block` 对 `n=poly_degree=4096` 个元素逐个 `hashlib.sha256(prefix + chunk).digest()`，且只取 `digest[:8]`（32 字节只用 8 字节）。每个训练 token 在 U worker 和 S worker 各生成一次完整 4096 维掩码 → 每步 = `B×S × 4096 × 2` 次 SHA-256。仓库自带测试 `tests/test_prg_vectorization.py:74-84` 测得的单次 4096 维生成约 1-5 ms。

**建议（正确且必要，双方代码同步即可）**：
1. 把 32 字节 digest 切成 4 个 8 字节块 → 哈希次数降到 1/4；
2. 或改用更快的 `hashlib.blake2b`（同样按 `seed‖step‖t‖i` 计数器模式，保持可复现）。
同时更新 `tests/test_prg_vectorization.py` 里的 reference 实现。U/S 用同一份代码，安全性（PRF 计数器模式）不变。

**预期收益**：掩码生成耗时降到 1/4 至 1/8，U/S 两个阶段同时受益。

### 3. dχ 噪声采样逐元素 Python 循环（每 token 4096 次 GPU 调用）

**位置**：`src/core/dchi_privacy.py:93-103`（`DChiNoiseGenerator.sample`），调用点 `dchi_privacy.py:638-653`（`H15Privatizer.__call__` 按位置 m 循环）。

**证据**：
```python
r = 0.0
for _ in range(self.d):          # d = 4096
    x = torch.empty(1, ...)
    x.exponential_(lambd=eta, generator=self.gen)
    r += float(x.item())
```
每次采样产生 4096 次 `torch.empty(1).exponential_()` + 4096 次 GPU→CPU `.item()` 同步；`__call__` 里对序列每个位置调一次 `sample`。

**建议（正确且必要）**：向量化为 `r = torch.empty(d, ...).exponential_(lambd=eta, generator=gen).sum()`。d 个 i.i.d. Exp(η) 之和 ≡ Gamma(d, η)，分布完全一致，且 `exponential_` 支持 generator，可复现性不变（测试 `tests/dp-tests/test_dchi_sampler.py` 可继续通过）。进一步可把 `__call__` 的位置循环也批量化为一次采样 (S,d)。

**预期收益**：DP 路径的噪声采样从"每位置 4096 次调用"降到 1 次；仅在 `--dp_enable` 时生效（默认关闭）。

### 4. 每个密文一次临时文件往返（SEAL 序列化开销）

**位置**：`src/core/bfv_privselect_v2_adapter.py:150-198`（`_seal_tmpfile` / `_seal_ciphertext_to_bytes` / `_seal_ciphertext_from_bytes`），热路径 `crypto_u.py:107-165`、`crypto_m.py:111-154`。

**证据**：每个 token 至少 3 次 `mkstemp → write → SEAL load/save → read → unlink`（U 加载密文、U 保存加掩码密文、M 加载密文），路径在 /dev/shm 也只是减少磁盘 I/O，系统调用开销仍在。

**建议（正确）**：worker 进程是单线程顺序处理，可在 `init_state` 里创建**一个固定临时文件路径**复用（写前 truncate），把每次 mkstemp/unlink 变成 write+load，串行安全。

**预期收益**：U/M 每 token 的 3 次文件创建/删除全部消除；与第 1、2 项叠加后每步密码学阶段显著缩短。

### 5. 验证/测试阶段 logits 重复计算 + 把隐藏态误当 logits 传入

**位置**：`src/parties/heterogeneous_protocol.py:557-565`（`step_val`）、`:686-694`（`step_test`）；`party_s.py:366-378`（`generate_predictions`）。

**证据**：
```python
# step_val
logits_cpu = self.party_s.compute_logits_for_eval(H_M)   # 第一次 H_M @ V^T
s_pred = self.party_s.generate_predictions(H_M, ...)      # H_M 作为纯张量传入
```
`generate_predictions` 对纯张量的分支是 `logits = H_M_or_logits`（不做 matmul），只有 `{"H_M": ...}` 字典形式才会重算 logits。于是：
- 分类场景下 `_classify_from_logits` 会对 4096 维 hidden 张量做 `index_select(opt_ids)`（`party_s.py:435-480`）——opt 词元 id 超过 4096 时越界崩溃；不超界时得到无意义分数；
- 同时 `H_M@V^T` 在 `compute_logits_for_eval` 和（若改用字典形式）`generate_predictions` 各算一遍。

**建议（正确且必要）**：`step_val`/`step_test` 把 `compute_logits_for_eval` 的结果直接传给 `generate_predictions`（其签名本来就接受 logits），或传 `{"H_M": H_M}` 并去掉重复的 `compute_logits_for_eval`。这既修 bug 又省一半验证 matmul。

### 6. 训练步内多次 `torch.cuda.empty_cache()`

**位置**：`party_m.py:459`（每步 backward 后）、`party_s.py:169,191,208,224`（`compute_a_t_gpu` 入口与回退路径）。

**证据**：`empty_cache` 会清空 CUDA caching allocator 的缓存块，下一步再重新分配；在固定显存模式下这是已知反模式，逐步调用带来额外延迟和碎片化。

**建议（正确，幅度需运行时确认）**：移除常规路径的逐步 `empty_cache`，仅在 OOM 回退分支保留；靠梯度检查点 + chunked pipeline 控制峰值。

### 7. 旋转位置编码每次 forward 重算

**位置**：`src/model/model_splitting.py:373-386`（U shard）、`:445-457`（M shard）。

**证据**：`forward` 内每次执行 `torch.outer(position_ids, inv_freq)` + `cos/sin`，而位置是连续 `0..S-1`。

**建议（正确，低收益）**：按 seq_len 缓存 `cos/sin`，或预计算到 max_seq_length 后切片。

---

## 二、正确性问题（训练有效性受影响，"必要"修复）

### 8. `output_ids` 中的 -100 被当作 PIR 索引 → 负索引取错密文行

**位置**：`party_s.py:257-262`（gold 分支）、`crypto_s.py:133-160`（`handle_request` 用 `y_t` 取行）、`bfv_privselect_v2_adapter.py:576-578`（`get_encrypted_row`）。

**证据链**：
1. 新数据集 `biotriplex_dataset.py:517-559,650-685` 的 `output_ids` = `labels`（prompt 与 pad 全部为 -100）；
2. `heterogeneous_protocol.py:335-344/445-454` 把它作为 `gold_ids` 传入 S；
3. `party_s.py:258` `y_gold_flat = gold_ids.flatten()` **没有过滤 -100**；
4. `crypto_s.py` 用 `y_t=-100` 调 `get_encrypted_row(-100)` → Python 列表负索引 → 取到 `_ct_list[-100]`（真实存在的**错误**行）。

**后果**：训练时 prompt/pad 位置拿到 `a_t - V[错误行]` 的梯度，只有少量 response token 的梯度是正确的；`--stage all` 全程生效（非偶发）。

**建议（必要）**：gold 分支只对 `labels != -100` 的位置用 gold id，其余位置回退 `argmax`（`y_all`）。

### 9. `attention_mask` 在 U/M 前向中完全被丢弃（padding 参与注意力）

**位置**：`party_u.py:246-259`（`_u_forward` 调 `self.model.forward(input_ids)`，mask 未传）、`party_m.py:285-341`（`_m_forward` 同样丢弃）、`model_splitting.py:399,498`（层调用 `attention_mask=None`）。

**证据**：数据集把每个样本 pad 到 `max_length`（`biotriplex_dataset.py:517-559`），但 U/M shard 所有 decoder 层都收到 `attention_mask=None`；S 端还会对 pad 位置计算 logits/梯度（与第 8 条叠加）。

**建议（必要）**：把 `attention_mask` 从 `party_u/party_m` 传进 shard，并转换成 Llama 层需要的 4D mask 或使用 `position_ids` + SDPA 的 padding 语义。

### 10. 复用缓存密文库时密钥不匹配（sk 无法从 pk 恢复）

**位置**：`bfv_privselect_v2_adapter.py:666-685`（`pk_path` 分支）；`biotriplex_finetune.py` `run_stage1`、`finetune.py:run_stage1`。

**证据**：
```python
if pk_path and not force_new_keys:
    self._public_key = PublicKey(); self._public_key.load(...)   # 旧公钥
    self._keygen = KeyGenerator(self._context)
    self._secret_key = self._keygen.secret_key()                 # 新私钥，与旧 pk 不匹配！
```
Stage 0 从未保存 sk（`bfv_keys.json` 只含元数据）；Stage 1 复用缓存时生成全新 sk 交给 CryptoMWorker，无法解密旧 DB 的密文。文档 §6.4.6 已记录此风险。

**建议（必要）**：Stage 0 持久化 sk（受保护文件，权限 0600），后端支持 `sk_path` 加载；或每次 `force_new_keys=True` 强制重建 DB；或在同一进程内复用同一个 backend 实例。

---

## 三、内存优化

### 11. 16 GB safetensor 权重缓存从不释放

**位置**：`model_splitting.py:125-170`（`_SAFETENSOR_CACHE`、`clear_safetensor_cache`），调用点 `party_u.py:67-88`、`party_m.py:58-81`。

**证据**：`_get_shared_weights` 把全部权重（Llama-8B bf16 ≈ 16 GB）缓存到模块级 dict；shard 构造时 `load_state_dict` 已把权重复制进模型参数，但 `clear_safetensor_cache` **全仓库无人调用**（grep 验证）。

**建议（正确且必要）**：U/M shard 加载完成后调用 `clear_safetensor_cache()`，训练期 CPU 内存减约 16 GB。

### 12. 密文库"mmap"名不副实，且 `_load_cache_mmap` 与磁盘格式不兼容

**位置**：`bfv_privselect_v2_adapter.py:425-443`（`_load_cache_mmap`）、`:503-573`（`_save_to_file`/`_load_from_file`）、`:576-580`（`get_encrypted_row`）。

**证据**：
- `_save_to_file` 写入的是 `4 字节长度前缀 + 密文` 的变长格式；`_load_cache_mmap` 却用 `avg_ct_size = data_size // n` 等长切块读取 → 逐行错位；
- 实际 CryptoSWorker 走的是 `_load_from_file`，把全部 ~16 GB 密文读成 Python `bytes` 列表（`_mmap` 属性从不使用）。README 声称的 mmap 零拷贝并未实现。

**建议（正确且必要，影响大）**：实现真正的 mmap + 行偏移索引（文件格式已含 4 字节前缀，可一次遍历建索引）；S worker 启动时间和内存占用同时改善，`get_encrypted_row` 变成零拷贝切片。`_load_cache_mmap` 或修复或删除（它只被 `legacy_ipc_stub.py:265` 调用）。

### 13. Stage 0 加载整个模型（fp32）只为取 lm_head

**位置**：`src/scripts/build_encrypted_db.py:62-100`（`load_V_matrix`）。

**证据**：`AutoModelForCausalLM.from_pretrained(..., device_map="cpu", torch_dtype=torch.float32)` 全量加载 Llama-8B（fp32 约 32 GB 内存），只为提取 `lm_head.weight`。

**建议（正确且必要）**：改用 `model.safetensors.index.json` 只读含 `lm_head` 的 shard（`biotriplex_finetune.py:317-349` 的 `_load_V_for_db` 已示范正确做法）。

---

## 四、数据/流水线效率

### 14. 全量 padding + 全位置密文运算

**位置**：`biotriplex_dataset.py:517-559,650-685`（pad 到 `max_length`）、`party_s.py:231-289`（对所有 B×S 位置算 a_t/y_t）、`crypto_s/crypto_u/crypto_m`（逐位置 PIR/掩码/解密）。

**证据**：每个样本 pad 到固定 `max_length`（默认 128，CLI 可到 10000）；训练时 pad 位置同样执行 PRG、BFV 加掩码、解密与梯度注入。实际医学句子往往远短于 max_length。

**建议（正确，收益取决于数据长度分布）**：训练用动态 batch 内 padding（取 batch 最大长度）而非固定 max_length；配合第 8/9 条让 pad 位置跳过 PIR 并给零梯度。可将密码学工作量降低数倍。

### 15. `PartyM.backward_and_update` 逐 token numpy 循环

**位置**：`party_m.py:380-394`。

**证据**：对每个 token 单独做 `np.asarray(s_share[:vec_dim])` + `np.round` + `np.where` + 加法；s_share 也是 `List[List[int]]`，可在循环外一次性堆成 `(n, vec_dim)` 数组整体向量化。

**建议（正确）**：全部改为一两条向量化 numpy 语句，逻辑等价。

### 16. `party_s.process_logits_dispatch` 把 a_t 拆成逐行 Python list 传输

**位置**：`party_s.py:283-289`。

**证据**：`"a_t_list": [a_all_cpu[i] for i in range(n_tokens)]`——把 (n,4096) 数组拆成 n 个 numpy 行 + `y_t_list` 拆成 n 个 Python int，再 pickle 给 worker。

**建议（正确）**：直接传 `a_all_cpu`（单个 2D ndarray）与 `y_all_cpu`（int64 数组），worker 内按行迭代；减少 pickle 开销与内存碎片。

---

## 五、维护性与一致性（低优先级但已验证）

### 17. `build_s3pir_hints.py` 的 hint_table.json 写到错误目录

**位置**：`build_s3pir_hints.py:51-160` 用 `HintTable(cache_dir=cache_dir)`；`s3pir_hints.py:188-210` 的 `to_cache_files` 写 `{cache_dir}/hint_table.json`；而运行时（`finetune.py`、`crypto_s.py:100-106`）查找的是 `{cache_dir}/s3pir_hints/hint_table.json`。独立运行 Stage 0 Step 2 后运行时找不到表（Design-2 下会静默降级为 `hint_table=None`，不报错）。

### 18. `legacy_ipc_stub.py` 使用已不存在的 API（一旦运行即崩溃）

**位置**：`legacy_ipc_stub.py:257-265` 调用 `BFVEncryptedDatabase(ctx=..., public_key=..., cache_path=...)` 与 `bfv_backend_local._cache_path`、`public_key`——当前类签名是 `(context, encryptor, evaluator, n_entries, poly_degree, scale, plain_bits, data_path, *, load_ct_list)`，且后端没有 `_cache_path`/`public_key` 属性。该文件标称"多主机预演"，但已与当前 API 脱节；`transport.py` 的 `InProcessBus`/`QueueBus` 全仓库无使用。

### 19. `PartyU`/`PartyM` 各自构造完整 BFV 后端（含 KeyGenerator 生成新密钥）

**位置**：`party_u.py:89-118`、`party_m.py:82-108`。仅为持有 pk/sk 就触发 SEAL 上下文 + 密钥生成；可提供轻量构造（直接加载序列化密钥）。

### 20. DeepSpeed 默认开启但无环境变量时静默降级

**位置**：`party_m.py:169-260`（`_setup_deepspeed_zero`）；CLI 默认 `--use_deepspeed_zero true`。`dist.init_process_group(backend="nccl")` 在单机无 `MASTER_ADDR/RANK/WORLD_SIZE` 时抛 RuntimeError 被捕获 → 打日志后继续用普通 AdamW。行为正确但"宣称开启、实际可能未启用"，建议默认关闭或显式失败。

---

## 六、结论与建议实施顺序

| 优先级 | 事项 | 类型 | 关键文件 |
|---|---|---|---|
| P0 | #8 gold_ids 负索引 | 正确性 | party_s.py / crypto_s.py |
| P0 | #10 密钥不匹配 | 正确性 | bfv_privselect_v2_adapter.py / biotriplex_finetune.py |
| P0 | #9 attention_mask 丢弃 | 正确性 | model_splitting.py / party_u.py / party_m.py |
| P0 | #1 worker 池并行失效 | 性能 | pool.py / heterogeneous_protocol.py |
| P1 | #5 step_val 传参错误+重复 logits | 正确性+性能 | heterogeneous_protocol.py / party_s.py |
| P1 | #2 PRG 逐元素 SHA-256 | 性能 | bfv_privselect_v2_adapter.py |
| P1 | #3 dχ 采样循环 | 性能 | dchi_privacy.py |
| P1 | #12 密文库 mmap | 内存 | bfv_privselect_v2_adapter.py |
| P1 | #11 safetensor 缓存不释放 | 内存 | model_splitting.py |
| P1 | #13 Stage 0 全模型加载 | 内存/时间 | build_encrypted_db.py |
| P2 | #4 临时文件往返、#6 empty_cache、#7 旋转编码、#14 padding、#15/#16 numpy/list 打包 | 性能 | 见上文 |
| P3 | #17-#20 | 维护性 | 见上文 |

说明：#1-#7、#11-#16、#19、#20 为性能/内存/维护优化；#8-#10、#5、#17 为正确性问题（其中 #5、#8、#10 直接影响训练/评估结果，建议优先修复）。
