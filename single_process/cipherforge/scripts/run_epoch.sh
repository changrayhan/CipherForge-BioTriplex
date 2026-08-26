#!/bin/bash
# Run one epoch of the full Stage 1 training in the foreground.
# Usage: run_epoch.sh 0|1|2   (0 = fresh, 1/2 = resume from last checkpoint)
set -e
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT/cipherforge"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
PYTHON="${PYTHON:-python}"
if [ -z "$CF_MODEL_PATH" ]; then
  echo "CF_MODEL_PATH is not set (see README)"
  exit 1
fi
EPOCH="$1"
MAXEP=$((EPOCH + 1))
mkdir -p "$REPO_ROOT/cipherforge/checkpoints/clinvar_ckpts/logs"
if [ "$EPOCH" = "0" ]; then
  "$PYTHON" -u -s "$REPO_ROOT/cipherforge/src/scripts/finetune.py" \
    --config "$REPO_ROOT/cipherforge/configs/clinvar_tinylama_local.json" \
    --stage 1 --log_freq 10 --max_epochs 1
else
  "$PYTHON" -u -s "$REPO_ROOT/cipherforge/src/scripts/finetune.py" \
    --config "$REPO_ROOT/cipherforge/configs/clinvar_tinylama_local.json" \
    --stage 1 --log_freq 10 --max_epochs "$MAXEP" --resume
fi
echo "epoch_${EPOCH}_rc=$?"
