#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/test_evaluation_strategyA.sh"
LOGFILE="${SCRIPT_DIR}/run_missing_4p4d_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

main() {
    log "============================================================"
    log "4P4D MISSING CONFIGS ONLY (RR only, all models)"
    log "Start: $(date)"
    log "============================================================"

    # trace=strategyA_aggressive_densed_4p4d, PD=4:4, all models, RR only
    # Existing SA results (SKIP): llama+lmsys, llama+mysharegpt, qwen+lmsys, qwen+mysharegpt
    # Missing RR: llama+lmsys, llama+mysharegpt, qwen+lmsys, qwen+mysharegpt  (4 experiments)
    env \
        DATASET_BASE_DIR="/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets" \
        TRACE_LIST="strategyA_aggressive_densed_4p4d" \
        MODEL_TYPE="all" \
        PD_RATIO_LIST="4:4" \
        TPOT_MS_LIST="30" \
        COMPARE_EXPERIMENT_VARIANTS="roundrobin" \
        bash "$RUN_SCRIPT" 2>&1 | tee -a "$LOGFILE"

    log "============================================================"
    log "ALL DONE. End: $(date)"
    log "============================================================"
}

main "$@"
