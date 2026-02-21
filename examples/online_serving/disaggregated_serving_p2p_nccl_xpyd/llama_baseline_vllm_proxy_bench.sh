set -euo pipefail

# ---------------------------
# USER CONFIG
# ---------------------------

USE_CUSTOM_PROXY="false"
TEST_ABLATION_P2D="false"
IGNORE="true"
REPRODUCE_BASELINE="false"
SAVE_SAMPLE="true"
REPRODUCE_BASELINE_CSV_PATH="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_116/benchmark_np4000_rr20_mt3000_20260116_195621_1p4d_test_baseline/dataset_result/test_results_20260116_201101.csv"

BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-lmsyschat}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output}
SAVE_OUTPUT=${SAVE_OUTPUT:-True}

VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

# Benchmark Params
BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}

MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./wt_experiment/experiment_paper/tmp/baseline_212}

PROXY_PORT=${PROXY_PORT:-28002}
BENCH_PORT=${BENCH_PORT:-22007}


VLLM_SEED=${VLLM_SEED:-42}

BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_baseline.py}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT=${BENCH_GOODPUT:-ttft:1000 tpot:50}
CACULATE_GOODPUT=${CACULATE_GOODPUT:-tpot:50}
# ==========================================
# SWEEP CONFIGURATION
# ==========================================
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"2000"}
BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"4,8,12,16,20"}
BENCH_MAX_TOKENS_LIST=${BENCH_MAX_TOKENS_LIST:-"8192"}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}
# NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"2000"}
# BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"8"}
# BENCH_MAX_TOKENS_LIST=${BENCH_MAX_TOKENS_LIST:-"3000"}



# Misc
if [ "$USE_CUSTOM_PROXY" = "true" ]; then
    PROXY_SCRIPT=${PROXY_SCRIPT:-test_disagg_proxy.py}
elif [ "$TEST_ABLATION_P2D" = "true" ]; then
    PROXY_SCRIPT=${PROXY_SCRIPT:-test_disagg_proxy_p2d.py}
else
    PROXY_SCRIPT=${PROXY_SCRIPT:-disagg_proxy_p2p_nccl_xpyd.py}
fi

# ---------------------------
# End USER CONFIG
# ---------------------------
cd "$(dirname "${BASH_SOURCE[0]}")"
PIDS=()
PGIDS=()

setup_directories() {
    local num_prompts=$1
    local req_rate=$2
    local max_tokens=$3
    local timestamp=$4
    
    BENCHMARK_DIR="${BASE_RESULT_DIR}/benchmark_np${num_prompts}_rr${req_rate}_mt${max_tokens}_${timestamp}_2p4d"
    mkdir -p "$BENCHMARK_DIR"
    
    CONFIG_DIR="${BENCHMARK_DIR}/config"
    LOG_DIR="${BENCHMARK_DIR}/log"
    RESULT_DIR="${BENCHMARK_DIR}/dataset_result"
    
    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
    
    echo "Created benchmark directories:"
    echo "  Base: $BENCHMARK_DIR"
}

check_required_files() {
    local files=("$PROXY_SCRIPT")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_num_gpus() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found"
        exit 1
    fi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "Need at least 2 GPUs. Found $num_gpus."
        exit 1
    fi
}

ensure_python_library_installed() {
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 not installed."
        exit 1
    fi
}

dump_config_json() {
    local current_prompts=$1
    local current_rate=$2
    local current_tokens=$3
    local ts=$4

    cfgfile="${CONFIG_DIR}/config_${ts}.json"
    cat > "$cfgfile" <<EOF
{
"generated_at": "$(date --iso-8601=seconds)",
"model": "${MODEL}",
"timeout_seconds": ${TIMEOUT_SECONDS},
"proxy_port": ${PROXY_PORT},
"seed": ${VLLM_SEED},
"benchmark": {
    "bench_script": "${BENCH_SCRIPT}",
    "output_path": "${RESULT_DIR}",
    "bench_port": ${BENCH_PORT},
    "endpoint": "/v1/completions",
    "dataset_name": "${BENCH_DATASET_NAME}",
    "dataset_path": "${BENCH_DATASET_PATH}",
    "max_concurrency": ${BENCH_MAX_CONCURRENCY},
    "request_rate": "${current_rate}",
    "maxtokenscustom": ${current_tokens},
    "temperature": ${BENCH_TEMPERATURE},
    "top_p": ${BENCH_TOP_P},
    "top_k": ${BENCH_TOP_K},
    "repetition_penalty": ${BENCH_REPETITION_PENALTY},
    "goodput": "${BENCH_GOODPUT}",
    "num_prompts": ${current_prompts},
    "timestamp": "${ts}"
}
}
EOF
    echo "Wrote config JSON to: $cfgfile"
}

wait_for_port_free() {
    local port=$1
    local timeout=${2:-10}
    local start=$(date +%s)
    while true; do
        if ! ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q .; then return 0; fi
        now=$(date +%s)
        if (( now - start >= timeout )); then return 1; fi
        sleep 0.5
    done
}

stop_servers() {
    if [[ ${#PGIDS[@]} -eq 0 && ${#PIDS[@]} -eq 0 ]]; then return; fi
    echo "Stopping servers..."
    for pg in "${PGIDS[@]}"; do
        if [[ -n "$pg" ]]; then kill -TERM -"$pg" 2>/dev/null || true; fi
    done
    sleep 3
    for pg in "${PGIDS[@]}"; do
        if pgrep -g "$pg" >/dev/null 2>&1; then kill -KILL -"$pg" 2>/dev/null || true; fi
    done
    pkill -9 -f "$PROXY_SCRIPT" >/dev/null 2>&1 || true
    PIDS=()
    PGIDS=()
}

cleanup() {
    echo "Cleaning up..."
    stop_servers
    ports_to_check=()
    ports_to_check+=("${PROXY_PORT}")
    for port in "${ports_to_check[@]}"; do
        if wait_for_port_free "$port" 10; then :; else echo "Port ${port} stuck?"; fi
    done
    exit 0
}

wait_for_server() {
    local port=$1
    local timeout_seconds=$TIMEOUT_SECONDS
    local start_time=$(date +%s)
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s "localhost:${port}" > /dev/null 2>&1; then return 0; fi
        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then echo "Timeout $port"; return 1; fi
        sleep 1
    done
}

plot_decode_load() {
    local ts=$1
    local proxy_log="${LOG_DIR}/proxy_${ts}.log"
    local plot_script="$(dirname "${BASH_SOURCE[0]}")/plot_decode_load2.py"

    if [[ ! -f "$plot_script" ]]; then
        echo "[WARN] plot_decode_load2.py not found, skip plotting."
        return 0
    fi

    if [[ ! -f "$proxy_log" ]]; then
        echo "[WARN] Proxy log not found: $proxy_log, skip plotting."
        return 0
    fi

    echo "Generating decode load plot from proxy log..."
    echo "  Log: $proxy_log"

    python3 "$plot_script" \
        --log_path "$proxy_log" \
        > "${LOG_DIR}/plot_${ts}.log" 2>&1 || {
            echo "[WARN] Plotting failed, see ${LOG_DIR}/plot_${ts}.log"
        }
}

start_proxy() {
    PIDS=()
    PGIDS=()
    local timestamp=$1
    echo "Launching proxy..."
    
    setsid env PROXY_PORT="${PROXY_PORT}" BENCH_PORT="${BENCH_PORT}" VLLM_DTYPE="${VLLM_DTYPE}" MODEL_CONFIG_PATH="${MODEL}/config.json" TPOT="${CACULATE_GOODPUT}" bash -c "exec python3 \"$PROXY_SCRIPT\"" &> "${LOG_DIR}/proxy_${timestamp}.log" &
    proxy_pid=$!
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    PIDS+=("$proxy_pid"); PGIDS+=("$proxy_pgid")
}

trap cleanup INT TERM EXIT

main() {
    check_required_files
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart
    
    # Parse lists
    IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"
    IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$BENCH_REQUEST_RATE_LIST"
    IFS=',' read -ra MAX_TOKENS_ARRAY <<< "$BENCH_MAX_TOKENS_LIST"
    
    # 3-Layer Nested Loop: Prompts -> Request Rate -> Max Tokens
    for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
      for req_rate in "${REQUEST_RATE_ARRAY[@]}"; do
        for max_tokens in "${MAX_TOKENS_ARRAY[@]}"; do
            timestamp=$(date +%Y%m%d_%H%M%S)
            
            setup_directories "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"
            
            echo "========================================"
            echo "Run: Prompts=${num_prompts}, Rate=${req_rate}, MaxTokens=${max_tokens}"
            echo "Dir: ${BENCHMARK_DIR}"
            echo "========================================"
            
            dump_config_json "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"
            
            start_proxy "$timestamp"
            
            log_file="${LOG_DIR}/bench_np${num_prompts}_rr${req_rate}_mt${max_tokens}.log"
            
            # ==========================
            # Benchmark Command (Multi-line)
            # ==========================
            CMD="python3 \"$BENCH_SCRIPT\" \
                --backend openai-chat \
                --port \"${BENCH_PORT}\" \
                --endpoint '/v1/chat/completions' \
                --model \"${BENCH_MODEL}\" \
                --dataset-name ${BENCH_DATASET_NAME} \
                --dataset-path ${BENCH_DATASET_PATH} \
                --save-output ${SAVE_OUTPUT} \
                --out-path \"${RESULT_DIR}\" \
                --maxtokenscustom ${max_tokens} \
                --seed ${VLLM_SEED} \
                --num-prompts ${num_prompts} \
                --max-concurrency 1024 \
                --temperature ${BENCH_TEMPERATURE} \
                --top-p ${BENCH_TOP_P} \
                --top-k ${BENCH_TOP_K} \
                --repetition-penalty ${BENCH_REPETITION_PENALTY} \
                --request-rate ${req_rate} \
                --goodput ${BENCH_GOODPUT}"
            
            if [ "$IGNORE" = "true" ]; then CMD="$CMD --ignore-eos"; fi
            if [ "$TEST_ABLATION_P2D" = "true" ]; then CMD="$CMD --ablation-p2d"; fi
            
            if [ "$REPRODUCE_BASELINE" = "true" ]; then
                CMD="$CMD --reproduce-baseline"
                CMD="$CMD --baseline-csv-path ${REPRODUCE_BASELINE_CSV_PATH}"
                CMD="$CMD --ignore-eos"
            fi
            if [ "$SAVE_SAMPLE" = "true" ]; then
                CMD="$CMD --save-sample"
            fi            
            
            echo "Starting Benchmark Client..."
            setsid env BENCHMARK_INSTANCE_ID="benchmark-${num_prompts}" bash -c "exec $CMD" > "$log_file" 2>&1 &
            client_pid=$!
            wait "$client_pid"
            echo "Benchmark finished."
            plot_decode_load "$timestamp"
            
            # Verify proxy
            for pid in "${PIDS[@]}"; do
                if ! ps -p "$pid" > /dev/null 2>&1; then
                    echo "ERROR: Proxy died."; stop_servers; cleanup; exit 1
                fi
            done
            
            stop_servers
            
            # Wait for ports to clear
            all_ports=("${PROXY_PORT}")
            for p in "${all_ports[@]}"; do
                wait_for_port_free "$p" 15
            done
            
            echo "Sleeping ${SLEEP_BETWEEN_RUNS}s..."
            sleep ${SLEEP_BETWEEN_RUNS}
            
        done # End Max Tokens Loop
      done # End Request Rate Loop
    done # End Prompts Loop
    
    echo "All benchmarks finished."
    exit 0
}

main "$@"
