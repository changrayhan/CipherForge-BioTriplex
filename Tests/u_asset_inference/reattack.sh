#!/bin/bash
# Re-run the four U1 attacks on an existing run dir (training data intact).
set -u
TESTS=/root/CipherForge/Tests/u_asset_inference
PY=/root/miniconda3/bin/python3
export PYTHONPATH="$TESTS:/root/CipherForge/CipherForge-ClinVar/three_party"
RUN_DIR="${1:-$TESTS/results/run_20260827_115137}"

echo "=== U1a ==="
"$PY" -u -s "$TESTS/attacks/u1a_audit.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1a.log"
echo "=== U1b ==="
"$PY" -u -s "$TESTS/attacks/u1b_ciphertext.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1b.log"
echo "=== U1c ==="
"$PY" -u -s "$TESTS/attacks/u1c_recover_v.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1c.log"
echo "=== U1d ==="
"$PY" -u -s "$TESTS/attacks/u1d_recover_weights.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/u1d.log"
echo "=== aggregate ==="
"$PY" -u -s "$TESTS/attacks/aggregate.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/aggregate.log"
echo "DONE $RUN_DIR"
