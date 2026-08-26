#!/bin/bash
# Three-party smoke test: start S/M nodes, then the U coordinator runs a few
# training steps + a small validation + adapter export (no final eval).
# Usage: CF_MODEL_PATH=<snapshot> PYTHON=... bash scripts/run_smoke.sh
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT"
export REPO_ROOT="$ROOT"
export HF_HUB_OFFLINE=1
export CF_MODEL_PATH
if [ -z "$CF_MODEL_PATH" ]; then
  echo "CF_MODEL_PATH is not set"
  exit 1
fi

cleanup() {
  kill "$S_PID" "$M_PID" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$ROOT/coordinator/logs"
"$PYTHON" -u -s "$ROOT/party_s/main_s.py" --port 9003 --db_dir "$ROOT/party_s/db" --device "${S_DEVICE:-cpu}" > "$ROOT/coordinator/logs/party_s.log" 2>&1 &
S_PID=$!
"$PYTHON" -u -s "$ROOT/party_m/main_m.py" --port 9002 --keys_dir "$ROOT/party_m/keys" > "$ROOT/coordinator/logs/party_m.log" 2>&1 &
M_PID=$!

sleep 3
"$PYTHON" -u -s "$ROOT/coordinator/main.py" --config "$ROOT/coordinator/three_party_config.json" \
  --max_train_steps 120 --batch_size 4 --log_freq 10
echo "smoke_rc=$?"
