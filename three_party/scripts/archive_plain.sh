#!/usr/bin/env bash
# Archive plain baseline run into test-data/plain-data/
set -euo pipefail

DST=/root/CipherForge/test-data/plain-data
mkdir -p "$DST/adapter" "$DST/logs"

# Copy adapter files
cp -a /tmp/plain_run/adapter_config.json      "$DST/adapter/"
cp -a /tmp/plain_run/adapter_model.safetensors "$DST/adapter/"
cp -a /tmp/plain_run/tokenizer.json            "$DST/adapter/"
cp -a /tmp/plain_run/tokenizer_config.json     "$DST/adapter/"
cp -a /tmp/plain_run/chat_template.jinja       "$DST/adapter/" 2>/dev/null || true
cp -a /tmp/plain_run/README.md                 "$DST/adapter/" 2>/dev/null || true
cp -a /tmp/plain_run/train_args.json           "$DST/"

# Copy training log
cp -a /tmp/plain_run/train.log                 "$DST/logs/"
cp -a /tmp/plain_run/console.log               "$DST/logs/" 2>/dev/null || true

# Copy trainer_state.json (the loss curve) from the final checkpoint
cp -a /tmp/plain_run/ckpt/checkpoint-1875/trainer_state.json "$DST/logs/"

# Extract loss curve into a tidy JSON
python3 - <<'PY'
import json, os
state = json.load(open("/root/CipherForge/test-data/plain-data/logs/trainer_state.json"))
curve = []
for h in state.get("log_history", []):
    e = {"step": h.get("step"), "epoch": h.get("epoch")}
    if "loss" in h:
        e["loss"] = h["loss"]
    if "eval_loss" in h:
        e["eval_loss"] = h["eval_loss"]
    if "learning_rate" in h:
        e["lr"] = h["learning_rate"]
    curve.append(e)
out = {
    "global_step": state.get("global_step"),
    "epoch": state.get("epoch"),
    "best_metric": state.get("best_metric"),
    "curve": curve,
}
with open("/root/CipherForge/test-data/plain-data/loss_curve.json", "w") as f:
    json.dump(out, f, indent=2)
print("loss curve rows:", len(curve))
PY

# Manifest
python3 - <<'PY'
import json, os, time
mtime = lambda p: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(os.path.getmtime(p)))
m = {
    "task": "plaintext LoRA fine-tuning of TinyLlama-1.1B-Chat on ClinVar QA (3 epochs)",
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "data_dir": "/root/CipherForge/CipherForge-ClinVar/three_party/party_u/data/qa",
    "test_jsonl": "/root/CipherForge/CipherForge-ClinVar/three_party/party_u/data/qa/test.jsonl",
    "out_dir_runtime": "/tmp/plain_run",
    "adapter_dir": "/root/CipherForge/test-data/plain-data/adapter",
    "hyperparams": {
        "epochs": 3.0,
        "batch_size": 16,
        "grad_accum": 1,
        "lr": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_seq_len": 128,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "warmup_pct": 0.1,
        "bf16": True,
        "seed": 42,
    },
    "training": {
        "global_step": 1875,
        "epoch": 3.0,
        "train_loss_final": 0.2524,
        "train_runtime_s": 531.3,
        "train_samples_per_s": 56.47,
    },
    "encryption": "none",
    "differential_privacy": "none",
    "model_splitting": "none (full model on one process)",
    "files": {
        "metrics": "clinvar_auprc.json",
        "loss_curve": "loss_curve.json",
        "train_log": "logs/train.log",
        "train_args": "train_args.json",
        "adapter_dir": "adapter/",
    },
    "archived_at": mtime("/root/CipherForge/test-data/plain-data/clinvar_auprc.json"),
}
with open("/root/CipherForge/test-data/plain-data/manifest.json", "w") as f:
    json.dump(m, f, indent=2, ensure_ascii=False)
print("manifest written")
PY

# SUMMARY
python3 - <<'PY'
import json
import os
m = json.load(open("/root/CipherForge/test-data/plain-data/manifest.json"))
r = json.load(open("/root/CipherForge/test-data/plain-data/clinvar_auprc.json"))
s = {
    "task": "plaintext LoRA fine-tuning (3 epochs, no encryption, no DP)",
    "model_splitting": m["model_splitting"],
    "encryption": m["encryption"],
    "differential_privacy": m["differential_privacy"],
    "test_set": m["test_jsonl"],
    "n_test": r["n"],
    "pos_rate": r["pos_rate"],
    "auprc": r["auprc"],
    "auc": r["auc"],
    "accuracy@0.5": r["accuracy@0.5"],
    "per_gene_mean_auprc": r["per_gene"]["mean_auprc"],
    "training": m["training"],
    "adapter_path": m["adapter_dir"],
    "data_split_note": "uses party_u/data/qa/ — the SAME test set the three-party runs evaluate on",
}
with open("/root/CipherForge/test-data/plain-data/SUMMARY.json", "w") as f:
    json.dump(s, f, indent=2, ensure_ascii=False)
print(json.dumps(s, indent=2))
PY

echo ""
echo "=== test-data/plain-data/ ==="
ls -la /root/CipherForge/test-data/plain-data/
