#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Run missing tpot=30ms derived_datasets experiments
# Batch 1: PD=2:6, trace=strategyA_aggressive_2p6d, llama only, RR+SA  (4 exps)
# Batch 2: PD=4:4, trace=strategyA_aggressive_densed_4p4d, all, RR only (4 exps)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/test_evaluation_strategyA.sh"
LOGFILE="${SCRIPT_DIR}/run_missing_derived_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

run_batch() {
    local label="$1"
    shift
    log "============================================================"
    log "START BATCH: ${label}"
    log "ENV: $*"
    log "============================================================"

    env "$@" bash "$RUN_SCRIPT" 2>&1 | tee -a "$LOGFILE"
    local rc=${PIPESTATUS[0]}

    if [[ "$rc" -eq 0 ]]; then
        log "BATCH COMPLETE: ${label} (exit=0)"
    else
        log "BATCH FAILED: ${label} (exit=${rc})"
    fi
    return "$rc"
}

main() {
    log "============================================================"
    log "DERIVED DATASETS MISSING CONFIGS RUNNER"
    log "Start time: $(date)"
    log "============================================================"

    # --- Batch 1: PD=2:6, llama only, RR+SA ---
    # Existing from old traces (SKIP qwen):
    #   qwen+lmsys: RR+SA (old qwen_strategyA_aggressive)
    #   qwen+mysharegpt: RR (old qwen_mysharegpt_aggressive_only) + SA (old qwen_sharegpt_aggressive)
    # Missing: llama+lmsys (RR+SA), llama+mysharegpt (RR+SA)
    run_batch "B1_pd2p6d_llama" \
        DATASET_BASE_DIR="/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets" \
        TRACE_LIST="strategyA_aggressive_2p6d" \
        MODEL_TYPE="llama" \
        PD_RATIO_LIST="2:6" \
        TPOT_MS_LIST="30" \
        COMPARE_EXPERIMENT_VARIANTS="roundrobin scheme4_gwf" \
        RUN_BATCH_TIMESTAMP="$(date +%Y%m%d_%H%M%S)" || log "WARNING: Batch 1 had errors"

    log "Pausing 30s between batches..."
    sleep 30

    # --- Batch 2: PD=4:4, all models, RR only ---
    # Existing from old trace strategyA_aggressive_densed (SKIP SA):
    #   llama+lmsys: SA, llama+mysharegpt: SA
    #   qwen+lmsys: SA, qwen+mysharegpt: SA
    # Missing: all 4 RR variants
    run_batch "B2_pd4p4d_all_RRonly" \
        DATASET_BASE_DIR="/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets" \
        TRACE_LIST="strategyA_aggressive_densed_4p4d" \
        MODEL_TYPE="all" \
        PD_RATIO_LIST="4:4" \
        TPOT_MS_LIST="30" \
        COMPARE_EXPERIMENT_VARIANTS="roundrobin" \
        RUN_BATCH_TIMESTAMP="$(date +%Y%m%d_%H%M%S)" || log "WARNING: Batch 2 had errors"

    log "============================================================"
    log "ALL BATCHES COMPLETE. End: $(date)"
    log "============================================================"
}

main "$@"
