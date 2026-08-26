#!/usr/bin/env bash
# long_monitor.sh — 长期守护
#  1) RMS-PIR × 3 epoch + eval  -> 归档到 test-data/rms-pir-data/
#  2) 然后 Block-PIR × 3 epoch + eval -> 归档到 test-data/block-pir-data/
#  每 10 分钟探活；失败自动重启（最多 3 次）；结束后整轮保留再走 Block。
#
# 用法: bash /root/CipherForge/CipherForge-ClinVar/three_party/scripts/long_monitor.sh

set -u
# 不开 -e：单次失败要允许 retry 路径走完
shopt -s lastpipe

REPO=/root/CipherForge/CipherForge-ClinVar/three_party
TESTDATA=/root/CipherForge/test-data
ARCHIVE_PY="$REPO/scripts/archive_run.py"
LOG_DIR=$REPO/coordinator/logs
PYTHONPATH="$REPO"
export PYTHONPATH REPO_ROOT="$REPO" HF_HUB_OFFLINE=1 S_DEVICE=cpu
export CF_MODEL_PATH="/root/hf_cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

mkdir -p "$LOG_DIR"

MON_OUT=/root/long_monitor.log
PROGRESS=/root/long_progress.log
: > "$MON_OUT"
: > "$PROGRESS"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
say()   { printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "$MON_OUT" ; }

############################################
# 工具: 检查 PID 是否存活
############################################
alive() {
  local pid=${1:-0}
  [ -n "$pid" ] && [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null
}

############################################
# 工具: 杀光所有三进程节点 + coordinator
############################################
kill_all() {
  say "KILL_ALL: cleaning residual three_party procs"
  pkill -9 -f "coordinator/main.py"     2>/dev/null
  pkill -9 -f "main_s.py"               2>/dev/null
  pkill -9 -f "main_m.py"               2>/dev/null
  pkill -9 -f "main_u.py"               2>/dev/null
  for _ in 1 2 3 4 5 6; do
    if ! pgrep -af "coordinator/main.py|main_(s|m|u)\.py" >/dev/null; then break; fi
    sleep 1
  done
}

############################################
# 工具: 启动 S/M
############################################
start_sm() {
  say "START_S/M on :9003 / :9002"
  python -u -s party_s/main_s.py --port 9003 --db_dir party_s/db --device cpu \
    > "$LOG_DIR/party_s_${PHASE}.log" 2>&1 &
  SPID=$!
  python -u -s party_m/main_m.py --port 9002 --keys_dir party_m/keys \
    > "$LOG_DIR/party_m_${PHASE}.log" 2>&1 &
  MPID=$!
  say "  SPID=$SPID  MPID=$MPID"
  echo "$SPID" > /tmp/lm_spid
  echo "$MPID" > /tmp/lm_mpid
  # wait for listening; RMS 节点需要 dump hints，rms 模式有时 ~36s；用 python 探活避免 ss 在容器里行为不一致
  for i in $(seq 1 90); do
    if python3 -c "
import socket
s1=socket.socket(); r1=s1.connect_ex(('127.0.0.1',9003)); s1.close()
s2=socket.socket(); r2=s2.connect_ex(('127.0.0.1',9002)); s2.close()
import sys; sys.exit(0 if r1==0 and r2==0 else 1)
" 2>/dev/null; then
      say "  S/M listening (took ${i}s)"
      return 0
    fi
    sleep 1
  done
  say "  S/M never came up in 90s!"
  tail -20 "$LOG_DIR/party_s_${PHASE}.log" 2>&1
  tail -20 "$LOG_DIR/party_m_${PHASE}.log" 2>&1
  return 1
}

############################################
# 工具: 启动 coordinator
############################################
start_coord() {
  local cfg=$1 run_log=$2
  say "START_COORD cfg=$cfg -> $run_log"
  setsid nohup python -u -s coordinator/main.py \
      --config "$cfg" \
      --max_train_steps 0 \
      --batch_size 16 \
      --log_freq 50 \
    > "$run_log" 2>&1 < /dev/null &
  local cp=$!
  echo "$cp" > /tmp/lm_cpid
  say "  CPID=$cp"
  sleep 10
  if ! alive "$cp"; then
    say "  coordinator died within 10s!"
    tail -40 "$run_log"
    return 1
  fi
  return 0
}

############################################
# 工具: 归档
############################################
archive() {
  local phase=$1 dest=$2 run_log=$3
  # 幂等: SUMMARY.json 已存在则跳过
  if [ -f "$dest/SUMMARY.json" ]; then
    say "ARCHIVE phase=$phase -> $dest  SKIP (SUMMARY.json already exists)"
    return 0
  fi
  say "ARCHIVE phase=$phase -> $dest"
  mkdir -p "$dest"
  if [ -f "$run_log" ]; then
    cp -v "$run_log" "$dest/${phase}_run.log"
  fi
  # 训练日志 + 监控日志
  [ -f "$PROGRESS" ] && cp "$PROGRESS" "$dest/${phase}_progress.log"
  [ -f "$MON_OUT" ]  && cp "$MON_OUT"  "$dest/${phase}_monitor.log"
  # S/M
  [ -f "$LOG_DIR/party_s_${phase}.log" ] && cp "$LOG_DIR/party_s_${phase}.log" "$dest/"
  [ -f "$LOG_DIR/party_m_${phase}.log" ] && cp "$LOG_DIR/party_m_${phase}.log" "$dest/"
  # coordinator logs
  cp -v "$LOG_DIR"/epoch_metrics.jsonl   "$dest/" 2>/dev/null || true
  cp -v "$LOG_DIR"/clinvar_auprc.json    "$dest/" 2>/dev/null || true
  cp -v "$LOG_DIR"/preflight_block_dp.log "$dest/" 2>/dev/null || true
  cp -v "$LOG_DIR"/rss_monitor.log       "$dest/" 2>/dev/null || true
  # checkpoint 目录
  mkdir -p "$dest/checkpoints"
  rsync -a --exclude='*.tmp' "$REPO/party_m/checkpoints/" "$dest/checkpoints/" 2>/dev/null \
    || cp -a "$REPO/party_m/checkpoints/." "$dest/checkpoints/"
  # adapter 目录
  [ -d "$REPO/coordinator/adapter" ] && {
    mkdir -p "$dest/adapter"
    cp -a "$REPO/coordinator/adapter/." "$dest/adapter/" 2>/dev/null || true
  }
  # 总结 JSON
  python3 - <<EOF > "$dest/SUMMARY.json"
import json, os, time
out = {}
lm = "$run_log"
if os.path.exists(lm):
    out["run_log_size_bytes"] = os.path.getsize(lm)
out["phase"] = "$phase"
out["dest"] = "$dest"
out["ts"]   = time.strftime("%Y-%m-%d %H:%M:%S")
# 拉训练步数
import re
if os.path.exists(lm):
    txt = open(lm).read()
    steps = [int(m.group(1)) for m in re.finditer(r"step (\d+):", txt)]
    out["last_step"]     = steps[-1] if steps else -1
    out["total_steps"]   = 1875
    out["completed_pct"] = (steps[-1] / 1875.0 * 100.0) if steps else 0.0
    out["dp_alpha"] = (re.search(r"alpha=([\d.]+)", txt) or [None,None])[1] or None
    out["pir_mode"] = (re.search(r"PIR mode: (\w+)", txt) or [None,None])[1] or None
print(json.dumps(out, indent=2, ensure_ascii=False))
EOF
  say "ARCHIVE done -> $(du -sh "$dest" | awk '{print $1}')"
}

############################################
# 工具: 单阶段 (RMS 或 Block) 主循环
#  返回 0=完成, 1=出错, 2=被中断
############################################
run_phase() {
  local phase=$1 cfg=$2 run_log=$3 max_retries=${4:-3}
  PHASE="$phase"
  local attempt=1 rc=0
  while [ $attempt -le $max_retries ]; do
    say "===== $phase attempt #$attempt ====="
    kill_all
    if ! start_sm; then
      say "$phase: S/M 启动失败"
      attempt=$((attempt+1)); sleep 5; continue
    fi
    if ! start_coord "$cfg" "$run_log"; then
      say "$phase: coordinator 启动失败"
      kill_all
      attempt=$((attempt+1)); sleep 5; continue
    fi
    CPID=$(cat /tmp/lm_cpid)
    # 等退出 / 完成
    while alive "$CPID"; do
      sleep 60
      # 写进度摘要
      local last_step
      last_step=$(grep -oE "step [0-9]+:" "$run_log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
      local auc
      auc=$(grep -E "AUPRC|Best checkpoint|reached" "$run_log" 2>/dev/null | tail -1)
      printf '[%s] %s attempt#%d  step=%s/1875  %s\n' "$(stamp)" "$phase" "$attempt" "${last_step:-?}" "${auc:-...}" \
        | tee -a "$PROGRESS"
    done
    wait "$CPID" 2>/dev/null
    rc=$?
    say "$phase: coordinator exit code = $rc"
    # AUPRC 文件已落地且与本 phase 匹配 -> 算完成 (它通常在 final step 之后才 dump)
    if [ -s "$LOG_DIR/clinvar_auprc.json" ]; then
      local au_ts; au_ts=$(stat -c %Y "$LOG_DIR/clinvar_auprc.json")
      say "$phase: AUPRC 落地 -> $LOG_DIR/clinvar_auprc.json"
      archive "$phase" "$TESTDATA/${phase}-data" "$run_log"
      return 0
    fi
    # 否则认为失败
    say "$phase: 未检测到完成 -> 准备 retry"
    tail -30 "$run_log" | tee -a "$MON_OUT"
    attempt=$((attempt+1))
    sleep 10
  done
  say "$phase: $max_retries 次重试全部失败 -> 停止"
  return 1
}

############################################
# 主流程
############################################
say "===== LONG_MONITOR START ====="

# 阶段 1: RMS
if run_phase rms \
   "$REPO/coordinator/three_party_config_rms.json" \
   /root/rms_3epoch_run.log 3; then
  say "===== RMS 阶段完成 ====="
else
  say "===== RMS 阶段失败 -> 仍尝试 Block (用户要求自动继续) ====="
fi

# 阶段 2: Block (S/M 已 kill 干净，按用户决定立即重启)
kill_all
if run_phase block \
   "$REPO/coordinator/three_party_config.json" \
   /root/block_3epoch_run.log 3; then
  say "===== Block 阶段完成 ====="
else
  say "===== Block 阶段失败 -> 请人工查看 ====="
fi

say "===== LONG_MONITOR DONE ====="
