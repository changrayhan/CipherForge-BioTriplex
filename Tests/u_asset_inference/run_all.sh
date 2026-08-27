#!/bin/bash
# T1 U-asset-inference: start nodes with capture hooks, run driver + 4 attacks.
# Env: PIR_MODE=block|rms  N_STEPS  Q_CORPUS  Q_SNAP  SNAP_EVERY  MAX_SNAP_STEP
set -u
TESTS=/root/CipherForge/Tests/u_asset_inference
TP=/root/CipherForge/CipherForge-ClinVar/three_party
PY="${PYTHON:-/root/miniconda3/bin/python3}"
export CF_MODEL_PATH="${CF_MODEL_PATH:-/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
export PYTHONPATH="$TESTS:$TP"
export HF_HUB_OFFLINE=1
PIR_MODE="${PIR_MODE:-block}"
N_STEPS="${N_STEPS:-20}"
SNAP_EVERY="${SNAP_EVERY:-5}"
Q_SNAP="${Q_SNAP:-20}"
MAX_SNAP_STEP="${MAX_SNAP_STEP:-20}"
Q_CORPUS="${Q_CORPUS:-300}"
RUN_DIR="${RUN_DIR:-$TESTS/results/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR/captures/u" "$RUN_DIR/captures/m" "$RUN_DIR/captures/s"

echo "=== T1 run: $RUN_DIR (pir=$PIR_MODE) ==="
pkill -9 -f "party_u/main_u.py" 2>/dev/null
pkill -9 -f "party_s/main_s.py" 2>/dev/null
pkill -9 -f "party_m/main_m.py" 2>/dev/null
pkill -9 -f "coordinator/main.py" 2>/dev/null
sleep 2

CF_U_CAPTURE_DIR="$RUN_DIR/captures/u" setsid nohup "$PY" -u -s "$TP/party_u/main_u.py" \
    --port 9001 --model_path "$CF_MODEL_PATH" --data_dir "$TP/party_u/data" \
    > "$RUN_DIR/party_u.log" 2>&1 &
CF_S_CAPTURE_DIR="$RUN_DIR/captures/s" setsid nohup "$PY" -u -s "$TP/party_s/main_s.py" \
    --port 9003 --db_dir "$TP/party_s/db" --device cpu \
    > "$RUN_DIR/party_s.log" 2>&1 &
CF_M_CAPTURE_DIR="$RUN_DIR/captures/m" setsid nohup "$PY" -u -s "$TP/party_m/main_m.py" \
    --port 9002 --keys_dir "$TP/party_m/keys" --model_path "$CF_MODEL_PATH" \
    > "$RUN_DIR/party_m.log" 2>&1 &

"$PY" - "$RUN_DIR" <<'PYEOF'
import socket, sys, time
run = sys.argv[1]
for port in (9001, 9002, 9003):
    ok = False
    for _ in range(240):
        s = socket.socket(); ok = s.connect_ex(("127.0.0.1", port)) == 0; s.close()
        if ok: break
        time.sleep(1)
    print("port", port, "ready" if ok else "TIMEOUT", flush=True)
    if not ok: sys.exit(1)
PYEOF

echo "=== driver ==="
PIR_MODE="$PIR_MODE" RUN_DIR="$RUN_DIR" \
N_STEPS="$N_STEPS" SNAP_EVERY="$SNAP_EVERY" \
Q_SNAP="$Q_SNAP" MAX_SNAP_STEP="$MAX_SNAP_STEP" \
Q_CORPUS="$Q_CORPUS" \
"$PY" -u -s "$TESTS/driver.py" > "$RUN_DIR/driver.log" 2>&1
echo "driver_rc=$?"

pkill -9 -f "party_u/main_u.py" 2>/dev/null
pkill -9 -f "party_s/main_s.py" 2>/dev/null
pkill -9 -f "party_m/main_m.py" 2>/dev/null
sleep 2

echo "=== U1a audit ==="
"$PY" -u -s "$TESTS/attacks/u1a_audit.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1a.log"
echo "=== U1b ciphertext ==="
"$PY" -u -s "$TESTS/attacks/u1b_ciphertext.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1b.log"
echo "=== U1c recover V ==="
"$PY" -u -s "$TESTS/attacks/u1c_recover_v.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1c.log"
echo "=== U1d recover weights ==="
"$PY" -u -s "$TESTS/attacks/u1d_recover_weights.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1d.log"
echo "=== aggregate ==="
"$PY" -u -s "$TESTS/attacks/aggregate.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/aggregate.log"
echo "=== DONE: $RUN_DIR ==="
