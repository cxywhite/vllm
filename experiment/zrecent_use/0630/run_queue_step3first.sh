#!/usr/bin/env bash
# 编排：8 卡空闲 -> step3 优先 -> 依次跑其余 4p4d 待办（跳过 1p1d 的 2 卡项）。
# 不做看门狗自动杀进程（按用户要求改为人工定期检查）；仅保留超时 + 残留清理兜底。
# 卡死判据（供人工检查）：prefill 日志出现 "Dropping stale deferred KV tensor" 且
# 之后所有活跃实例 "Avg generation throughput" 持续 0。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_LOG="${QUEUE_LOG:-$HERE/queue_run_step3first.log}"

# name|driver|timeout_minutes|result_dir
QUEUE=(
  "step3_realpred|run_step3_realpred.sh|300|$HERE/wl_step3_realpred"
  "gammaada_stability|run_gammaada_stability.sh|90|$HERE/wl_kv1p6_gammaada_r2"
  "step2_amax220|run_step2_amax220.sh|90|$HERE/wl_kv1p6_gwf_amax220_r2"
  "real_lmsys_kv1p6like|run_real_kv1p6like_lmsys.sh|170|$HERE/wl_real_lmsys_kv1p6like_ab"
  "real_sharegpt_kv1p6like|run_real_kv1p6like_sharegpt.sh|170|$HERE/wl_real_sharegpt_kv1p6like_ab"
  "4p4d_x1p8|run_4p4d_x1p8_ab.sh|170|$HERE/wl_kv1p8_ab"
)

log() { echo "[queue $(date -Is)] $*" >> "$QUEUE_LOG"; }

free_8gpus() {
  local n maxused ports
  n=$(ps -eo cmd 2>/dev/null | grep -c '[t]est_evaluation_scheme4_rr_compare')
  maxused=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  ports=$(ss -ltn 2>/dev/null | grep -cE ':220[0-9][0-9]|:28002')
  (( n == 0 && maxused < 2000 && ports == 0 ))
}

# 只杀本项进程树：进程组 TERM -> KILL，再用本实验特征兜底清理
cleanup_tree() {
  local pgid=$1
  log "[cleanup] 终止进程组 -$pgid"
  kill -TERM -- -"$pgid" 2>/dev/null
  sleep 10
  kill -KILL -- -"$pgid" 2>/dev/null
  sleep 3
  local pids
  pids=$(ps -eo pid,cmd 2>/dev/null | grep -E 'vllm serve|test_disagg|EngineCore|APIServer|benchmark_serving|test_evaluation' | grep -v grep | awk '{print $1}')
  if [[ -n "$pids" ]]; then
    log "[cleanup] 兜底清理残留: $(echo $pids | tr '\n' ' ')"
    echo "$pids" | xargs -r kill -9 2>/dev/null
  fi
}

run_item() {
  local name=$1 driver=$2 timeout_min=$3 dir=$4
  log "===== 启动 [$name] ($driver) timeout=${timeout_min}min result_dir=$dir ====="

  setsid bash "$HERE/$driver" >> "$QUEUE_LOG" 2>&1 &
  local pgid=$!
  local start_s; start_s=$(date +%s)

  while kill -0 "$pgid" 2>/dev/null; do
    local now_s el
    now_s=$(date +%s); el=$(( now_s - start_s ))
    if (( el >= timeout_min * 60 )); then
      log "[止损] [$name] 超时 ${el}s，清理"
      cleanup_tree "$pgid"
      return 124
    fi
    sleep 60
  done
  wait "$pgid"; local rc=$?
  log "[$name] 进程结束 rc=$rc (0=成功,124=超时)"
  return "$rc"
}

log "===== 编排启动（step3 优先），队列 ${#QUEUE[@]} 项 ====="

for item in "${QUEUE[@]}"; do
  IFS='|' read -r name driver timeout_min dir <<< "$item"

  log "----- 等待 8 卡空闲以运行 [$name] ($driver) -----"
  while true; do
    if free_8gpus; then
      sleep 30
      if free_8gpus; then break; fi
    fi
    sleep 30
  done

  run_item "$name" "$driver" "$timeout_min" "$dir"
  rc=$?
  if (( rc != 0 )); then
    log "[注意] [$name] 退出码 rc=$rc"
  fi
  sleep 60   # 排空端口/显存
done

log "===== 队列全部结束 ====="
