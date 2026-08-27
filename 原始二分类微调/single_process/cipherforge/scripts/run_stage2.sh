#!/bin/bash
# Stage 2: evaluate the trained adapter on the test set (AUPRC/AUC/acc).
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
"$PYTHON" -u -s "$REPO_ROOT/cipherforge/src/scripts/finetune.py" \
  --config "$REPO_ROOT/cipherforge/configs/clinvar_tinylama_local.json" \
  --stage 2 --checkpoint "$REPO_ROOT/cipherforge/checkpoints/clinvar_ckpts/checkpoints/best_checkpoint.pt"
