#!/bin/bash
# 三进程完整微调：启动 S/M 节点，然后 U 协调者跑训练 + 导出 + 评测。
# 用法：CF_MODEL_PATH=... PYTHON=... bash scripts/run_full.sh [--max_train_steps N]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export REPO_ROOT="$ROOT"
if [ -z "$CF_MODEL_PATH" ]; then
  echo "CF_MODEL_PATH is not set"
  exit 1
fi
mkdir -p "$ROOT/coordinator/logs"

cleanup() {
  kill "$U_PID" "$S_PID" "$M_PID" 2>/dev/null || true
}
trap cleanup EXIT

"$PYTHON" -u -s "$ROOT/party_u/main_u.py" --port 9001 --model_path "$CF_MODEL_PATH" --data_dir "$ROOT/party_u/data" > "$ROOT/coordinator/logs/party_u.log" 2>&1 &
U_PID=$!
"$PYTHON" -u -s "$ROOT/party_s/main_s.py" --port 9003 --db_dir "$ROOT/party_s/db" --device "${S_DEVICE:-cpu}" > "$ROOT/coordinator/logs/party_s.log" 2>&1 &
S_PID=$!
"$PYTHON" -u -s "$ROOT/party_m/main_m.py" --port 9002 --keys_dir "$ROOT/party_m/keys" > "$ROOT/coordinator/logs/party_m.log" 2>&1 &
M_PID=$!

sleep 5
"$PYTHON" -u -s "$ROOT/coordinator/main.py" --config "$ROOT/coordinator/three_party_config.json" "$@"
rc=$?
echo "coordinator_rc=$rc"
