#!/usr/bin/env bash
# Rate sweep for the admission A/B: same 200-token bursty profile, three arrival
# densities.  Iteration mode (1 rep per variant) per AGENTS.md §7.
#
#   rate600 : 23.8 req/s cluster (~30/s per decode instance inside a burst)
#   rate900 : 35.6 req/s
#   dense_short (already run twice, 43.6 req/s) is the third point.
#
# Usage:  bash sweep_rates.sh rate600 rate900
# Every run is wrapped by run_ab_with_hw.sh, so each gets hw_metrics_*.csv and a
# pre-flight GPU check (refuses to start on a busy box / a stolen slot).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
export MODEL_TYPE=qwen
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export WT_SIMPLIFY_PREDICT=1 WT_NATIVE_LOG_INTERVAL_MS=1000
export COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf"
export PD_RATIO_LIST="4:4" TPOT_MS_LIST="30" RUN_REPEAT_PER_CONFIG=1
export DATASET_BASE_DIR="$HERE/admission_ab_base"

for name in "$@"; do
  echo "########## sweep: $name  $(date -Is)"
  TRACE_NAME="$name" BASE_RESULT_DIR="$HERE/admission_ab_sweep/$name" \
    bash "$HERE/run_ab_with_hw.sh"
  rc=$?
  echo "########## sweep: $name done rc=$rc $(date -Is)"
  if [[ $rc -ne 0 ]]; then
    echo "[sweep] stopping at $name (rc=$rc) — inspect $HERE/admission_ab_sweep/$name" >&2
    exit $rc
  fi
done
echo "[sweep] all done"
