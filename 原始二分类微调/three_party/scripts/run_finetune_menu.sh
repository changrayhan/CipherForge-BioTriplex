#!/usr/bin/env bash
# 三模式微调交互式启动器（薄包装）
# 模式 2（三进程 RMS-PIR 变体）逻辑在 run_finetune_menu.py 中：
#   启动 U/M/S 三节点服务 + 演枢台平台(:8600)，运行 coordinator 训练，
#   完成后默认保持服务运行以便联调（fallback / eval 接口见 docs/02、docs/04）。
# 可用环境变量：CF_MODE / CF_EPOCHS / CF_BATCH_SIZE / CF_MAX_STEPS /
#              CF_SKIP_TRAIN=1（跳过训练加载预微调检查点）/ CF_KEEP_SERVICES=0（跑完自动清理）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHON="${PYTHON:-/root/miniconda3/bin/python3.12}"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
exec "$PYTHON" -u -s "$SCRIPT_DIR/run_finetune_menu.py" "$@"
