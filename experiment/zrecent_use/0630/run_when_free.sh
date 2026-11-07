#!/usr/bin/env bash
# Wait for a free A/B slot, then run the dense_short 3-rep confirmation, with retries.
#
# "Free" means: no other eval process, every GPU < 2 GiB, and none of our ports
# (22xxx / 28002) listening -- held for two consecutive checks (40 s), so we do
# not race another launcher that is mid-restart.
#
# Memory note (experiment_log.md pitfall #14): the default P2p pool is 32 GB per
# instance (256 GB for 8 instances = 68 % of a 377 GB host); with a co-tenant
# that alone caused a global OOM that killed our engines. We run with 8 GB.
#
# Usage:  bash run_when_free.sh [max_attempts] [log]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTEMPTS="${1:-4}"
LOG="${2:-/root/myshare/ab_logs/confirm_dense_short_3rep.log}"

free_now() {
  local n used ports
  n=$(pgrep -fc 'test_evaluation_scheme4_rr_compare' 2>/dev/null || echo 0)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  ports=$(ss -ltn 2>/dev/null | grep -cE ':220[0-9][0-9]|:28002')
  (( n == 0 && used < 2000 && ports == 0 ))
}

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "[grab] attempt $attempt/$ATTEMPTS: waiting for a free slot ($(date -Is))"
  got=0
  for _ in $(seq 1 180); do          # up to 3 h per attempt
    if free_now; then
      sleep 20
      if free_now; then got=1; break; fi
    fi
    sleep 20
  done
  if (( got == 0 )); then
    echo "[grab] no free slot within 3 h, giving up"
    exit 1
  fi

  echo "[grab] launching at $(date -Is)"
  VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
  MODEL_TYPE=qwen QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct \
  PREFILL_TENSOR_POOL_MEMORY=8 DECODE_TENSOR_POOL_MEMORY=8 \
  WT_SIMPLIFY_PREDICT=1 WT_NATIVE_LOG_INTERVAL_MS=1000 \
  COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf" PD_RATIO_LIST="4:4" TPOT_MS_LIST="30" \
  RUN_REPEAT_PER_CONFIG=3 TRACE_NAME=bimodal_burst DATASET_BASE_DIR="$HERE/admission_ab_base" \
  BASE_RESULT_DIR="$HERE/admission_ab_confirm3c" \
    bash "$HERE/run_ab_with_hw.sh" >> "$LOG" 2>&1
  rc=$?
  echo "[grab] attempt $attempt finished rc=$rc at $(date -Is)"
  if (( rc == 0 )); then
    echo "[grab] DONE"
    tail -25 "$LOG"
    exit 0
  fi
  echo "[grab] failed (rc=$rc); last log lines:"
  tail -8 "$LOG" | cut -c1-160
  sleep 60    # let ports/GPU memory drain before the next attempt
done
echo "[grab] all attempts failed"
exit 1
