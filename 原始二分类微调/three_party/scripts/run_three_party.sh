#!/bin/bash
# 三进程全量微调入口（供后台/前台调用）。
# 环境变量：CF_MODEL_PATH（必填）、PYTHON（默认 python）、S_DEVICE（默认 cpu）。
set -e
ROOT=/root/cipherforge-three-party
cd "$ROOT"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export REPO_ROOT="$ROOT"
export S_DEVICE="${S_DEVICE:-cpu}"
export CF_MODEL_PATH
exec bash "$ROOT/scripts/run_full.sh" "$@"
