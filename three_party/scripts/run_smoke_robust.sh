#!/bin/bash
# 健壮启动：等待 S/M 端口就绪后再启动 Coordinator
# 支持环境变量：
#   CONFIG           coordinator 配置 json（默认 coordinator/three_party_config.json）
#   MAX_STEPS        训练步数（默认 1875 = 10000/16*3）
#   BATCH_SIZE       batch（默认 16，与明文一致）
#   RESUME=1         透传 --resume
#   SKIP_TRAIN=1     透传 --skip_train（仅 eval）
#   OUT_DIR          日志/产物输出根目录（默认 /root/CipherForge/test-data）
#   SCENARIO         子目录名，如 block_dp / rms_dp（默认 block_dp）
#   PIR_MODE         block|rms（默认走 CONFIG）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export CF_MODEL_PATH="${CF_MODEL_PATH:-/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
export PYTHONPATH="$ROOT"
export REPO_ROOT="$ROOT"
export HF_HUB_OFFLINE=1

# ---- Memory tuning (glibc arena / pymalloc / thread fan-out) ----
# 25-core machine → glibc default would be 200 arenas × ~64 MB each = 12.8 GB
# resident even when 99% of the memory is "free" inside freelists. Clamp it.
export MALLOC_ARENA_MAX=2
# Bypass pymalloc so Python's free() returns memory to the (now tiny) glibc
# arena instead of stacking on top of pymalloc's freelist.
export PYTHONMALLOC=malloc
# Tokenizers / OpenMP thread storms are also a hidden memory driver.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-coordinator/three_party_config.json}"
MAX_STEPS="${MAX_STEPS:-1875}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SCENARIO="${SCENARIO:-block_dp}"
OUT_DIR="${OUT_DIR:-/root/CipherForge/test-data}"

LOG_DIR="$OUT_DIR/logs/$SCENARIO"
mkdir -p "$LOG_DIR"

# 杀掉可能残留的 S/M 端口占用，避免段间重启冲突
fuser -k 9002/tcp 9003/tcp 2>/dev/null || true
sleep 2

# Coordinator 输出与 metrics 默认走 CONFIG 路径；启动器额外复制到 OUT_DIR
SRC_COORD_LOG="$ROOT/coordinator/logs/coordinator.log"
SRC_PARTY_M_LOG="$ROOT/coordinator/logs/party_m.log"
SRC_PARTY_S_LOG="$ROOT/coordinator/logs/party_s.log"
mkdir -p "$ROOT/coordinator/logs"
rm -f "$ROOT/coordinator/logs/coordinator.log"
rm -f "$ROOT/coordinator/logs/party_m.log"
rm -f "$ROOT/coordinator/logs/party_s.log"
rm -f "$ROOT/coordinator/logs/epoch_metrics.jsonl"
rm -f "$ROOT/coordinator/logs/training_metrics_"*.json
rm -f "$ROOT/coordinator/adapter/adapter_config.json"
rm -f "$ROOT/coordinator/adapter/adapter_model.safetensors"

echo "=== 配置: $CONFIG | scenario=$SCENARIO | max_train_steps=$MAX_STEPS | batch_size=$BATCH_SIZE | resume=${RESUME:-0} ==="
echo "=== OUT_DIR=$OUT_DIR  LOG_DIR=$LOG_DIR ==="

cleanup() {
    echo "=== 清理 S/M 进程 ==="
    kill "$S_PID" "$M_PID" 2>/dev/null || true
}
trap cleanup EXIT

# 准备 coordinator 透传参数
COORD_ARGS=(
    --config "$CONFIG"
    --max_train_steps "$MAX_STEPS"
    --batch_size "$BATCH_SIZE"
    --log_freq 10
)
if [ "${RESUME:-0}" = "1" ]; then
    COORD_ARGS+=(--resume)
fi
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    COORD_ARGS+=(--skip_train)
fi
if [ -n "${PIR_MODE:-}" ]; then
    COORD_ARGS+=(--pir_mode "$PIR_MODE")
fi

T_START=$(date +%s)
echo "=== 启动 Party S ==="
"$PYTHON" -u -s "$ROOT/party_s/main_s.py" --port 9003 --db_dir "$ROOT/party_s/db" --device cpu \
    > "$LOG_DIR/party_s.log" 2>&1 &
S_PID=$!
echo "S PID: $S_PID"

echo "=== 启动 Party M ==="
"$PYTHON" -u -s "$ROOT/party_m/main_m.py" --port 9002 --keys_dir "$ROOT/party_m/keys" \
    --model_path "$CF_MODEL_PATH" > "$LOG_DIR/party_m.log" 2>&1 &
M_PID=$!
echo "M PID: $M_PID"

echo "=== 等待 S 端口就绪 ==="
for i in $(seq 1 30); do
    if fuser -s 9003/tcp 2>/dev/null; then
        echo "S 端口就绪 (${i}s)"
        break
    fi
    sleep 1
done

echo "=== 等待 M 端口就绪 ==="
for i in $(seq 1 90); do
    if fuser -s 9002/tcp 2>/dev/null; then
        echo "M 端口就绪 (${i}s)"
        break
    fi
    sleep 1
done

echo "=== 启动 Coordinator ==="
"$PYTHON" -u -s "$ROOT/coordinator/main.py" "${COORD_ARGS[@]}" \
    > "$LOG_DIR/coordinator.log" 2>&1
rc=$?
T_END=$(date +%s)
echo "Coordinator exit code: $rc (wall=$((T_END-T_START))s)"

# 同步 copy 日志到 test-data，便于后续聚合
cp -f "$ROOT/coordinator/logs/coordinator.log" "$LOG_DIR/coordinator.log" 2>/dev/null || true
cp -f "$ROOT/coordinator/logs/party_m.log"   "$LOG_DIR/party_m.log"   2>/dev/null || true
cp -f "$ROOT/coordinator/logs/party_s.log"   "$LOG_DIR/party_s.log"   2>/dev/null || true

# 复制 adapter 到 test-data
if [ -d "$ROOT/coordinator/adapter" ]; then
    rm -rf "$OUT_DIR/adapters/${SCENARIO}_adapter"
    cp -r "$ROOT/coordinator/adapter" "$OUT_DIR/adapters/${SCENARIO}_adapter"
    echo "adapter copied to $OUT_DIR/adapters/${SCENARIO}_adapter"
fi

# 解析 coordinator.log 输出阶段时长（如果存在的话）
PHASES_JSON="$OUT_DIR/time_breakdown/${SCENARIO}_phases.json"
mkdir -p "$(dirname "$PHASES_JSON")"
"$PYTHON" - "$LOG_DIR/coordinator.log" "$PHASES_JSON" <<'PYEOF'
import json, sys, re
from pathlib import Path
log = Path(sys.argv[1]).read_text(errors="replace") if Path(sys.argv[1]).exists() else ""
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
def first(pat):
    m = re.search(pat, log)
    return m.group(1) if m else None
def last(pat):
    matches = list(re.finditer(pat, log))
    return matches[-1].group(1) if matches else None
def count(pat):
    return len(re.findall(pat, log))
step_times = [float(x) for x in re.findall(r"step_time_ms[=:]\s*([0-9.]+)", log)]
result = {
    "scenario": out.parent.parent.name + "/" + out.stem.replace("_phases",""),
    "wall_total_s": None,
    "n_steps_logged": len(step_times),
    "step_time_mean_ms": (sum(step_times)/len(step_times)) if step_times else None,
    "step_time_total_ms": (sum(step_times)) if step_times else None,
    "first_step_ts": first(r"step=\s*(\d+)"),
    "last_step_ts": last(r"step=\s*(\d+)"),
    "t1_s_ready": first(r"S 端口就绪 \((\d+)s\)"),
    "t1_m_ready": first(r"M 端口就绪 \((\d+)s\)"),
    "adapter_saved": bool(re.search(r"adapter saved|adapter exported", log, re.I)),
    "eval_done": bool(re.search(r"evaluation (complete|done)|metrics\.json.*written|test AUPRC", log, re.I)),
    "log_path": str(out.with_name(out.stem + ".log").with_suffix('')),
}
try:
    result["scenario"] = out.stem
except Exception:
    pass
out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"[phases] -> {out}")
PYEOF
echo "[phases] written"

exit $rc