#!/usr/bin/env bash
# watcher：等 lmsys 两个臂跑完后，停掉旧队列（sharegpt 不启动），释放 8 卡给其他 AI 使用。
# 只在检测到 lmsys 结束（日志出现 real_lmsys_kv1p6like 进程结束）后才动作，不碰正在跑的 lmsys。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_PID="${QUEUE_PID:-16208}"
LOG="$HERE/queue_run_step3first.log"
WLOG="$HERE/pause_after_lmsys.log"

wlog() { echo "[pause $(date -Is)] $*" >> "$WLOG"; }

wlog "启动，盯旧队列 pid=$QUEUE_PID，等 lmsys 跑完即停"

while true; do
  if grep -q "real_lmsys_kv1p6like.*进程结束" "$LOG" 2>/dev/null; then
    wlog "检测到 lmsys 已结束，停队列 $QUEUE_PID"
    kill -TERM "$QUEUE_PID" 2>/dev/null
    sleep 5
    kill -KILL "$QUEUE_PID" 2>/dev/null
    # 兜底清理残留（确保 8 卡彻底释放给其他 AI）
    pids=$(ps -eo pid,cmd 2>/dev/null | grep -E 'vllm serve|EngineCore|APIServer|test_disagg|benchmark_serving|test_evaluation' | grep -v grep | awk '{print $1}')
    if [[ -n "$pids" ]]; then
      wlog "清理残留进程: $(echo $pids | tr '\n' ' ')"
      echo "$pids" | xargs -r kill -TERM 2>/dev/null
      sleep 8
      echo "$pids" | xargs -r kill -9 2>/dev/null
    fi
    wlog "队列已停，8 卡应已释放（剩余 sharegpt→4p4d→redo 待用户说恢复再跑）"
    exit 0
  fi
  if ! kill -0 "$QUEUE_PID" 2>/dev/null; then
    wlog "旧队列进程已不在（可能已结束/崩溃），不动作"
    exit 0
  fi
  sleep 10
done
