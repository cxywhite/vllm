#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="${SCRIPT_DIR}/run_missing_4p4d_qwen_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }
main() {
    log "=== 4P4D RR QWEN ONLY (skip llama already done) ==="
    env \
        DATASET_BASE_DIR="/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets" \
        TRACE_LIST="strategyA_aggressive_densed_4p4d" \
        MODEL_TYPE="qwen" \
        PD_RATIO_LIST="4:4" \
        TPOT_MS_LIST="30" \
        COMPARE_EXPERIMENT_VARIANTS="roundrobin" \
        bash "${SCRIPT_DIR}/test_evaluation_strategyA.sh" 2>&1 | tee -a "$LOGFILE"
    log "=== DONE ==="
}
main "$@"
