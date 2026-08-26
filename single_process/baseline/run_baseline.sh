#!/bin/bash
# Plaintext LoRA baseline: train TinyLlama on ClinVar QA (default 3 epochs,
# batch 16) and evaluate AUPRC on the test set.
# Smoke run: MAX_STEPS=30 ./run_baseline.sh
set -e
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
PYTHON="${PYTHON:-python}"
OUT="$REPO_ROOT/baseline/outputs/clinvar_tinylama_plain_128"
mkdir -p "$OUT"
MAX_STEPS="${MAX_STEPS:-0}"
"$PYTHON" -u -s "$REPO_ROOT/baseline/scripts/finetune_plain.py" \
  --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data_dir "$REPO_ROOT/data/qa" \
  --out_dir "$OUT" \
  --max_steps "$MAX_STEPS"
"$PYTHON" -u -s "$REPO_ROOT/baseline/scripts/evaluate_auprc.py" \
  --adapter "$OUT" \
  --data "$REPO_ROOT/data/qa/test.jsonl" \
  --out "$REPO_ROOT/baseline/outputs/metrics.json"
