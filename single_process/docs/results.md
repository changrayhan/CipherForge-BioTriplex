# 结果记录（真 PIR 版）

## 任务与数据

- 任务：ClinVar 变异致病性二分类（`Yes`=致病，`No`=良性）。
- 模型：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，LoRA r=8 / alpha=16 / dropout=0.05，
  batch=16，lr=2e-4，warmup=100，3 epochs，max_seq_len=128。
- 数据：5k/类，train 10,000 / val 9,708 / test 10,000（按基因切分，互不重叠）。

## 测试集指标（n=10,000，pos_rate=0.5）

| 指标 | 明文基线 | CipherForge（真 PIR） |
|---|---:|---:|
| AUPRC | 0.7922 | **0.8033** |
| AUC | 0.7936 | **0.8081** |
| accuracy@0.5 | 0.722 | 0.6666 |
| per-gene 平均 AUPRC | 0.5784 | **0.5961** |

> accuracy@0.5 对概率校准敏感；排序类主指标（AUPRC/AUC/per-gene）为
> 公平对比依据。每次运行 PRG 掩码不同带来 ≤3.5e-3 浮点传输量化差异，
> 指标允许 ±0.01 量级波动。

## 训练过程要点

- 明文基线：HF Trainer，1875 步，train loss 1.24→~0.24。
- CipherForge（真 PIR）：S 标签无关；U 对 32 个监督位置构建 block=8 的
  real+dummy 查询块，S 返回整块、U 私有置换取行；监控 CE 0.72→0.41，
  验证 token 准确率 83.5~84.1%，步时 ~1.15s、显存 3.7GB。
- 加密正确性（真实 SEAL）：
  - 密文库 32000 行解密与明文逐槽位一致（误差 0）；
  - 完整往返 344,064 槽位：精确域误差 1e-4、float32 传输 ≤3.5e-3；
  - 块 PIR 往返 40 组 × 2048 槽位：max 误差 1e-4。

## 复现命令

见根目录 README「快速开始」。隐私边界细节见《真 PIR 隐私保护测试报告》。
