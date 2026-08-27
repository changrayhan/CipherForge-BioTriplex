#!/usr/bin/env bash
# Run plaintext LoRA fine-tuning 3 epochs (no encryption, no DP) and write
# metrics + adapter to test-data/plain-data/. Looks identical to the three-
# party training (same hyperparameters, same data split) for fair comparison.
set -euo pipefail
cd /root/CipherForge/CipherForge-ClinVar/single_process/baseline

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p /root/CipherForge/test-data/plain-data
rm -rf /tmp/plain_run
mkdir -p /tmp/plain_run

# Train: 3 epochs of 1875 steps. Plain (no encryption, no DP).
python3 -u scripts/finetune_plain.py \
  --model_id "/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6" \
  --data_dir "/root/CipherForge/CipherForge-ClinVar/three_party/party_u/data/qa" \
  --out_dir /tmp/plain_run \
  --max_seq_len 128 \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --seed 42 \
  > /tmp/plain_run/console.log 2>&1

echo "TRAIN_DONE_RC=$?"

# Evaluate on the same test set the three-party runs use
python3 -u /root/CipherForge/CipherForge-ClinVar/three_party/coordinator/evaluate_auprc.py \
  --model_id "/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6" \
  --adapter /tmp/plain_run \
  --data "/root/CipherForge/CipherForge-ClinVar/three_party/party_u/data/qa/test.jsonl" \
  --out /root/CipherForge/test-data/plain-data/clinvar_auprc.json

echo "EVAL_DONE"
