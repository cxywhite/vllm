#!/usr/bin/env bash
# -------------------------------------------------------------------
# Breakpoint resume helper — only handles skip list.
# All other params come from the main script's own defaults.
# Usage: ./resume_evaluation.sh
#        BENCH_MAX_CONCURRENCY=512 ./resume_evaluation.sh  (optional overrides)
# -------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/test_evaluation_scheme4_rr_compare copy.sh"

RESULT_0630="/root/.cache/huggingface/hub/wt_predictor/0630/result_0630"
TMP_RESULT_0630="/root/.cache/huggingface/hub/wt_predictor/0630/tmp_result_0630"
EXPERIMENT_RESULT_2="${SCRIPT_DIR}/experiment_result_2"

# ── Collect skip list from three result directories ──
SKIP_LIST=""

for base_dir in "$RESULT_0630" "$TMP_RESULT_0630" "$EXPERIMENT_RESULT_2"; do
    if [[ -d "$base_dir" ]]; then
        for dir in "$base_dir"/*/; do
            dir="${dir%/}"
            bn="$(basename "$dir")"
            case "$bn" in
                *.md|*.csv|*.html|*.txt) continue ;;
            esac
            if [[ -z "$SKIP_LIST" ]]; then
                SKIP_LIST="$bn"
            else
                SKIP_LIST="${SKIP_LIST},${bn}"
            fi
        done
    fi
done

COUNT=$(echo "$SKIP_LIST" | tr ',' '\n' | wc -l)
echo "=== Breakpoint Resume ==="
echo "Skipping $COUNT existing result directories from:"
echo "  1. result_0630/"
echo "  2. tmp_result_0630/"
echo "  3. experiment_result_2/"
echo ""

# ── Only export the skip list; everything else is inherited ──
export SKIP_EXISTING_RESULT_DIRS="$SKIP_LIST"

echo "Launching: $EVAL_SCRIPT"
echo "Log: nohup_output.log"
echo ""

cd "$SCRIPT_DIR"
nohup bash "$EVAL_SCRIPT" > nohup_output.log 2>&1 &
PID=$!
echo "Started with PID=$PID"
echo "Monitor: tail -f ${SCRIPT_DIR}/nohup_output.log"
