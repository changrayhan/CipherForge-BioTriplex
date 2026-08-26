# BioTriplex-21 微调对比：明文 LoRA vs RMS-PIR（TinyLlama-1.1B-Chat-v1.0）

数据: single_process/data/BioTriplex-21（train 801 / val 149 / test 230，21 类）
配置: 3 epochs, batch 4, max_seq 2048, LoRA r=8/alpha=16/dropout=0.05, lr=2e-4
RMS-PIR: dp_enable=true, dp_alpha=0.03, dp_num_classes=21, RMS partition=200/lam=16

## 零样本基线
accuracy=0.0783 (18/230), macro-F1=0.0081, weighted-F1=0.0148

## 明文 LoRA 基准（每 epoch 测试集精度）
| epoch | eval_loss | accuracy | macro-F1 | weighted-F1 | 正确/230 |
|---|---|---|---|---|---|
| 1 | 0.9362 | 0.2609 | 0.0368 | 0.1398 | 60 |
| 2 | 0.8375 | 0.2870 | 0.0891 | 0.2187 | 66 |
| 3 | 0.8787 | 0.3174 | 0.1002 | 0.2359 | 73 |
| 最终(best-eval) | 0.8375 | 0.2826 | 0.0885 | 0.2163 | 65 |

## RMS-PIR 变体（每 epoch 验证集指标；最终为测试集）
| epoch | val_ce_loss | val micro-F1 | val macro-F1 | val weighted-F1 |
|---|---|---|---|---|
| 0 | 2.7412 | 0.2483 | 0.0303 | 0.1158 |
| 1 | 2.4237 | 0.2148 | 0.0765 | 0.1696 |
| 2 | 2.6180 | 0.2550 | 0.0734 | 0.1822 |
| 最终测试 | - | accuracy=0.2478 | macro-F1=0.0842 | weighted-F1=0.1865 |

## 结论
- 两种方法都远超零样本（accuracy 0.078 -> 0.25~0.32）。
- 明文 LoRA 最终测试 accuracy 0.2826（最佳 epoch 0.3174）高于 RMS-PIR 0.2478，
  相对下降约 12%——这是 DP 噪声（alpha=0.03）与隐私保护管线的预期代价。
- 明文每 epoch 测试精度单调上升；RMS 每 epoch 验证 micro-F1 波动（val 仅 149 条、
  每 epoch 打乱），但 macro/weighted-F1 与 val_ce_loss 总体改善。
