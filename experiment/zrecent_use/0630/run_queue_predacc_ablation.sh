#!/usr/bin/env bash
# 预测器消融队列（无 β，WT_PREDICT_OUTPUT_SCALE=1.0，仅 GWF，无 UniBoost）：
# ours -> µ-Serve -> TRAIL++。每项在 8 卡空闲后才启动。无 timeout 硬编码（按 AGENTS.md 铁律）。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_LOG="${QUEUE_LOG:-$HERE/queue_run_predacc_ablation.log}"

# name|driver|result_dir
QUEUE=(
  "ours_nobeta_ablation|run_predacc_ours_nobeta.sh|$HERE/wl_predacc_ours_nobeta_o1600"
  "muserve_ablation|run_predacc_muserve.sh|$HERE/wl_predacc_muserve_o1600"
  "trailpp_ablation|run_predacc_trailpp.sh|$HERE/wl_predacc_trailpp_o1600"
)

log() { echo "[ablation-queue $(date -Is)] $*" >> "$QUEUE_LOG"; }

free_8gpus() {
  local n maxused ports
  n=$(ps -eo cmd 2>/dev/null | grep -c '[t]est_evaluation_scheme4_rr_compare')
  maxused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  ports=$(ss -ltn 2>/dev/null | grep -cE ':220[0-9][0-9]|:28002')
  (( n == 0 && maxused < 2000 && ports == 0 ))
}

run_item() {
  local name=$1 driver=$2
  log "===== 启动 [$name] ($driver) ====="
  setsid bash "$HERE/$driver" >> "$QUEUE_LOG" 2>&1 &
  local pgid=$!
  while kill -0 "$pgid" 2>/dev/null; do
    sleep 60
  done
  wait "$pgid"; local rc=$?
  log "[$name] 结束 rc=$rc"
  return "$rc"
}

log "===== 消融队列启动，共 ${#QUEUE[@]} 项 ====="

for item in "${QUEUE[@]}"; do
  IFS='|' read -r name driver dir <<< "$item"
  log "----- 等待 8 卡空闲以运行 [$name] -----"
  while true; do
    if free_8gpus; then
      sleep 30
      if free_8gpus; then break; fi
    fi
    sleep 30
  done
  run_item "$name" "$driver"
  rc=$?
  log "[$name] 退出码 rc=$rc"
  sleep 60   # 排空端口/显存
done

log "===== 消融队列全部结束 ====="
