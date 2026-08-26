#!/bin/bash
# Stage 0 (one-time): generate M-side BFV keypair, build the encrypted lm_head
# DB (N=4096, ~4.2 GB) and the S3PIR hint table under cipherforge/checkpoints/.
set -e
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT/cipherforge"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
PYTHON="${PYTHON:-python}"
if [ -z "$CF_MODEL_PATH" ]; then
  echo "CF_MODEL_PATH is not set. Run:"
  echo "  export CF_MODEL_PATH=\$(python $REPO_ROOT/cipherforge/tools/resolve_model_path.py)"
  exit 1
fi
mkdir -p "$REPO_ROOT/cipherforge/checkpoints"
"$PYTHON" -u -s "$REPO_ROOT/cipherforge/scripts/stage0_build_db.py" \
  --model_path "$CF_MODEL_PATH" \
  --cache_dir "$REPO_ROOT/cipherforge/checkpoints/bfv_cache" \
  --vocab_size 32000 \
  --hidden_dim 2048 \
  --poly_degree 4096 \
  --plain_bits 30 \
  --scale 10000 \
  --verify_rows 8
