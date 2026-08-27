#!/bin/bash
# 在后台启动三进程全量微调（与明文 LoRA 参数一致），日志写到 /root/three_party_full_run.log。
# 用法：CF_MODEL_PATH=<snapshot> PYTHON=<python> S_DEVICE=cpu bash scripts/launch_full_bg.sh [--max_epochs 3]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON="${PYTHON:-python}"
export CF_MODEL_PATH
export PYTHON
export S_DEVICE="${S_DEVICE:-cpu}"
export REPO_ROOT="$ROOT"
export HF_HUB_OFFLINE=1
LOG=/root/three_party_full_run.log
rm -f "$LOG"
setsid nohup bash "$ROOT/scripts/run_full.sh" "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched pid=$! log=$LOG"
