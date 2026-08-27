# 环境搭建

## 目标环境

- Linux（Ubuntu 24.04 / WSL2），单机 NVIDIA RTX 4060（8GB）即可。
- Python 3.11 + conda（推荐）。

## 创建环境

```bash
conda env create -f environment.yml
conda activate clinvar-ft
```

关键依赖（与开发验证环境一致）：

| 包 | 版本 |
|---|---|
| torch | 2.10.0+cu130 |
| transformers | 5.2.0 |
| peft | 0.18.1 |
| seal-python（导入名 `seal`） | 4.1.2.1 |
| numpy / pandas / pyarrow / scikit-learn | 最新兼容版 |

> `seal-python` 若需源码编译，请先安装 `cmake` 与 `g++`：
> `sudo apt install -y cmake g++`。
> 若 PyTorch 下载慢，可改用 `--extra-index-url https://download.pytorch.org/whl/cu121`
> 并把 requirements.txt 中的 `torch==2.10.0+cu130` 换成 `torch==2.10.0+cu121`。

## 模型下载（hf-mirror）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python baseline/scripts/download_model.py
# 输出形如：
#   export CF_MODEL_PATH=/home/<user>/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<hash>
export CF_MODEL_PATH=$(python cipherforge/tools/resolve_model_path.py)
```

模型仓库：[hf-mirror.com/TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://hf-mirror.com/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## 磁盘与显存

- 模型缓存：约 2.1GB。
- CipherForge BFV 缓存（`cipherforge/checkpoints/bfv_cache/`）：约 4.2GB
  （含密文库 `bfv_ct_db_n32000_d2048_p4096.bin`、密钥、hints）。
- 训练显存：batch=16 时约 4GB，8GB 的 4060 足够。

## 路径约定

所有脚本通过环境变量定位路径：

| 变量 | 含义 |
|---|---|
| `REPO_ROOT` | 仓库根目录（脚本自动推导，也可手动 export） |
| `CF_MODEL_PATH` | TinyLlama 快照目录（必须设置） |
| `HF_HOME` | HuggingFace 缓存根目录（默认 `~/.cache/huggingface`） |
| `HF_ENDPOINT` | 下载镜像（国内设 `https://hf-mirror.com`） |
| `PYTHON` | Python 解释器路径（默认 `python`，未激活 conda 环境时设为 `$CONDA_PREFIX/bin/python`） |

配置文件中 `${VAR}` 形式的值会在加载时被环境变量展开。
