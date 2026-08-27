# CipherForge-ClinVar —— 原始二分类微调

本目录为 CipherForge 的 **ClinVar 变异致病性二分类（原始二分类）** 版本，
与仓库根目录的 BioTriplex 7/21 类细粒度版本相互独立。

- 任务：ClinVar 变异致病性二分类（Yes/No）
- 模型：TinyLlama-1.1B-Chat-v1.0，U/M/S 三方切分 + LoRA（r=8、α=16）
- 协议：重构后四进程拓扑（U/M/S 直连 + 独立主控台），BFV + 块/RMS-PIR + dχ-DP
- 目录：
  - `single_process/`：单进程融合版（明文基线 / 张量级等价管线）
  - `three_party/`：三进程（四进程）隔离版，含 U/M/S 节点与独立主控台
  - `three_party/docs/10-重构说明…`：主控台独立与论文拓扑重构说明
  - `Tests/u_asset_inference/`（仓库根）：T1 U 资产推断攻击测试套件

> 本目录仅含源码、配置、文档与 ClinVar QA 数据；BFV 密文库、模型检查点、
> 训练日志、BFV 私钥等生成物/敏感文件不入库（见各层 .gitignore）。
