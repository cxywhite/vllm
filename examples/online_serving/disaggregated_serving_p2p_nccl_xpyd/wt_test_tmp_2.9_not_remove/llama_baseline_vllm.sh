set -euo pipefail

# ---------------------------
# USER CONFIG
# ---------------------------

# 定义是否使用pastfuture scheduler的变量
USE_PASTFUTURE_SCHEDULER="false"
USE_ACTIVATION_PREDICTOR="false"
USE_CUSTOM_PROXY="false"
USE_AIMD_SCHEDULER="false"
TEST_ABLATION_P2D="false"
IGNORE="false"
REPRODUCE_BASELINE="false"
SAVE_SAMPLE="true"
REPRODUCE_BASELINE_CSV_PATH="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_116/benchmark_np4000_rr20_mt3000_20260116_195621_1p4d_test_baseline/dataset_result/test_results_20260116_201101.csv"
TEST_MODEL="llama"  # 用于端到端测试的模型名称标识

BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-mysharegpt}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a}
SAVE_OUTPUT=${SAVE_OUTPUT:-True}

VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

# Benchmark Params
BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}

MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result/experiment_paper/tmp/baseline_122}

PROXY_PORT=${PROXY_PORT:-28002}
BENCH_PORT=${BENCH_PORT:-22007}


PREFILL_GPUS=${PREFILL_GPUS:-0,1}
PREFILL_PORTS=${PREFILL_PORTS:-22001,22002}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-22011,22012}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-8}

DECODE_GPUS=${DECODE_GPUS:-2,3,4,5,6,7}
DECODE_PORTS=${DECODE_PORTS:-22020,22021,22022,22023,22024,22025}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-22032,22033,22034,22035,22036,22037}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}
DECODE_TENSOR_POOL_MEMORY=${DECODE_TENSOR_POOL_MEMORY:-16}

KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e1}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
VLLM_SEED=${VLLM_SEED:-42}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}

BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_baseline.py}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT=${BENCH_GOODPUT:-ttft:1000 tpot:50}
CACULATE_GOODPUT=${CACULATE_GOODPUT:-tpot:50}
# ==========================================
# SWEEP CONFIGURATION
# ==========================================
MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS:-1}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-8192}
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"1000"}
BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"8,12,18,24,30"}
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
    
    BENCHMARK_DIR="${BASE_RESULT_DIR}/benchmark_np${num_prompts}_rr${req_rate}_mt${max_tokens}_${timestamp}_2p6d"
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

array_to_json() {
    local s="$1"
    if [[ -z "$s" ]]; then printf '[]'; return; fi
    IFS=',' read -ra _arr <<< "$s"
    printf '['
    local first=1
    for v in "${_arr[@]}"; do
        v="$(echo "$v" | sed -e 's/^\s*//' -e 's/\s*$//')"
        if [[ $first -eq 1 ]]; then printf '"%s"' "$v"; first=0; else printf ',"%s"' "$v"; fi
    done
    printf ']'
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
"prefill_gpus": $(array_to_json "$PREFILL_GPUS"),
"decode_gpus": $(array_to_json "$DECODE_GPUS"),
"prefill_ports": $(array_to_json "$PREFILL_PORTS"),
"decode_ports": $(array_to_json "$DECODE_PORTS"),
"prefill_kv_ports": $(array_to_json "$PREFILL_KV_PORTS"),
"decode_kv_ports": $(array_to_json "$DECODE_KV_PORTS"),
"prefill_gpu_memory_utilization": ${PREFILL_GPU_MEMORY_UTILIZATION},
"decode_gpu_memory_utilization": ${DECODE_GPU_MEMORY_UTILIZATION},
"kv_transfer_config": {
    "kv_connector": "${KV_CONNECTOR}",
    "producer_buffer": "${KV_PRODUCER_BUFFER}",
    "consumer_buffer": "${KV_CONSUMER_BUFFER}",
    "nccl_num_channels": ${KV_NCCL_CHANNELS},
    "send_type": "${KV_SEND_TYPE}",
    "prefill_tensor_pool_memory": ${PREFILL_TENSOR_POOL_MEMORY},
    "decode_tensor_pool_memory": ${DECODE_TENSOR_POOL_MEMORY}
},
"vllm_options": {
    "enforce_eager": $( [[ "$VLLM_ENFORCE_EAGER" -eq 1 ]] && echo true || echo false ),
    "seed": ${VLLM_SEED},
    "dtype": "${VLLM_DTYPE}",
    "max_model_len": ${VLLM_MAX_MODEL_LEN},
    "prefill_max_num_batched_tokens": ${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS},
    "decode_max_num_batched_tokens": ${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS},
    "max_num_seqs": ${VLLM_MAX_NUM_SEQS}
},
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
    pkill -9 -f vllm >/dev/null 2>&1 || true
    PIDS=()
    PGIDS=()
}

cleanup() {
    echo "Cleaning up..."
    stop_servers
    ports_to_check=()
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    ports_to_check+=("${PROXY_PORT}")
    for p in "${PREFILL_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
    for p in "${DECODE_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
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

start_servers() {
    PIDS=()
    PGIDS=()
    local timestamp=$1
    echo "Launching servers..."
    
    # Proxy
    setsid env PROXY_PORT="${PROXY_PORT}" BENCH_PORT="${BENCH_PORT}" VLLM_DTYPE="${VLLM_DTYPE}" MODEL_CONFIG_PATH="${MODEL}/config.json" TPOT="${CACULATE_GOODPUT}" bash -c "exec python3 \"$PROXY_SCRIPT\"" &> "${LOG_DIR}/proxy_${timestamp}.log" &
    proxy_pid=$!
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    PIDS+=("$proxy_pid"); PGIDS+=("$proxy_pgid")
    
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"
    
    # ==========================
    # Launch Prefill Instances
    # ==========================
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id=${PREFILL_GPU_ARRAY[$i]}
        port=${PREFILL_PORT_ARRAY[$i]:-$((20002 + i))}
        kv_port=${PREFILL_KV_PORT_ARRAY[$i]:-$((21001 + i))}
        
        # 1. 基础命令
        CMD="vllm serve \"$MODEL\" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port \"$port\" \
            --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} \
            --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --test-model ${TEST_MODEL} \
            --gpu-memory-utilization ${PREFILL_GPU_MEMORY_UTILIZATION}"

        # 2. 条件追加参数
        if [ "$USE_ACTIVATION_PREDICTOR" = "true" ]; then 
            CMD="$CMD --activation-predict"
        fi

        # 3. KV Transfer Config (拆分写，方便阅读)
        # 注意：JSON 内部用了转义的双引号
        KV_JSON_CFG='{'
        KV_JSON_CFG+='"kv_connector":"'"${KV_CONNECTOR}"'",'
        KV_JSON_CFG+='"kv_role":"kv_producer",'
        KV_JSON_CFG+='"kv_buffer_size":"'"${KV_PRODUCER_BUFFER}"'",'
        KV_JSON_CFG+='"kv_port":"'"$kv_port"'",'
        KV_JSON_CFG+='"kv_connector_extra_config":{'
        KV_JSON_CFG+='"proxy_ip":"0.0.0.0",'
        KV_JSON_CFG+='"proxy_port":"'"${PROXY_PORT}"'",'
        KV_JSON_CFG+='"http_port":"'"$port"'",'
        KV_JSON_CFG+='"send_type":"'"${KV_SEND_TYPE}"'",'
        KV_JSON_CFG+='"mem_pool_size_gb":"'"${PREFILL_TENSOR_POOL_MEMORY}"'",'
        KV_JSON_CFG+='"nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"'
        KV_JSON_CFG+='}}'

        CMD="$CMD --kv-transfer-config '$KV_JSON_CFG'"

        setsid env CUDA_VISIBLE_DEVICES="$gpu_id" VLLM_USE_V1=1 bash -c "exec $CMD" > "${LOG_DIR}/prefill${i}_${timestamp}.log" 2>&1 &
        pid=$!; pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid"); PGIDS+=("$pgid")
    done
    
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    IFS=',' read -ra DECODE_KV_PORT_ARRAY <<< "$DECODE_KV_PORTS"
    
    # ==========================
    # Launch Decode Instances
    # ==========================
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id=${DECODE_GPU_ARRAY[$i]}
        port=${DECODE_PORT_ARRAY[$i]:-$((20003 + i))}
        kv_port=${DECODE_KV_PORT_ARRAY[$i]:-$((22001 + i))}
        
        # 1. 基础命令
        CMD="vllm serve \"$MODEL\" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port \"$port\" \
            --tensor-parallel-size ${DECODE_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} \
            --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --gpu-memory-utilization ${DECODE_GPU_MEMORY_UTILIZATION} \
            --test-model ${TEST_MODEL} \
            --swap-space 0"

        # 2. 条件追加参数
        if [ "$USE_PASTFUTURE_SCHEDULER" = "true" ]; then CMD="$CMD --pastfuture-scheduler"; fi
        if [ "$USE_AIMD_SCHEDULER" = "true" ]; then CMD="$CMD --aimd-scheduler"; fi
        if [ "$TEST_ABLATION_P2D" = "true" ]; then CMD="$CMD --test-p2d"; fi
        
        # 3. KV Transfer Config (Consumer)
        KV_JSON_CFG='{'
        KV_JSON_CFG+='"kv_connector":"'"${KV_CONNECTOR}"'",'
        KV_JSON_CFG+='"kv_role":"kv_consumer",'
        KV_JSON_CFG+='"kv_buffer_size":"'"${KV_CONSUMER_BUFFER}"'",'
        KV_JSON_CFG+='"kv_port":"'"$kv_port"'",'
        KV_JSON_CFG+='"kv_connector_extra_config":{'
        KV_JSON_CFG+='"proxy_ip":"0.0.0.0",'
        KV_JSON_CFG+='"proxy_port":"'"${PROXY_PORT}"'",'
        KV_JSON_CFG+='"http_port":"'"$port"'",'
        KV_JSON_CFG+='"send_type":"'"${KV_SEND_TYPE}"'",'
        KV_JSON_CFG+='"mem_pool_size_gb":"'"${DECODE_TENSOR_POOL_MEMORY}"'",'
        KV_JSON_CFG+='"nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"'
        KV_JSON_CFG+='}}'

        CMD="$CMD --kv-transfer-config '$KV_JSON_CFG'"

        setsid env DECODE_INSTANCE_ID="decode-${i}" VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "exec $CMD" > "${LOG_DIR}/decode${i}_${timestamp}.log" 2>&1 &
        pid=$!; pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid"); PGIDS+=("$pgid")
    done
    
    echo "Waiting for servers..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server $port"
            stop_servers; return 1
        fi
    done
    return 0
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
            if [ "$num_prompts" -eq 4000 ] && [ "$req_rate" -eq 4 ]; then
                echo "Skipping invalid config: Prompts=${num_prompts}, Rate=${req_rate}"
                continue
            fi
            if [ "$num_prompts" -eq 2000 ] && [ "$req_rate" -eq 6 ] && [ "$max_tokens" -eq 3000 ]; then
                echo "Skipping invalid config: Prompts=${num_prompts}, MaxTokens=${max_tokens}"
                continue
            fi
            if [ "$num_prompts" -eq 2000 ] && [ "$req_rate" -eq 6 ] && [ "$max_tokens" -eq 4000 ]; then
                echo "Skipping invalid config: Prompts=${num_prompts}, MaxTokens=${max_tokens}"
                continue
            fi
            timestamp=$(date +%Y%m%d_%H%M%S)
            
            setup_directories "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"
            
            echo "========================================"
            echo "Run: Prompts=${num_prompts}, Rate=${req_rate}, MaxTokens=${max_tokens}"
            echo "Dir: ${BENCHMARK_DIR}"
            echo "========================================"
            
            dump_config_json "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"
            
            if ! start_servers "$timestamp"; then
                echo "Failed to start servers."
                cleanup; exit 1
            fi
            
            log_file="${LOG_DIR}/bench_np${num_prompts}_rr${req_rate}_mt${max_tokens}.log"
            
            # ==========================
            # Benchmark Command (Multi-line)
            # ==========================
            CMD="python3 \"$BENCH_SCRIPT\" \
                --backend vllm \
                --port \"${BENCH_PORT}\" \
                --endpoint '/v1/completions' \
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
                --goodput ${BENCH_GOODPUT} \
                --min-prompt-tokens ${MIN_PROMPT_TOKENS} \
                --max-prompt-tokens ${MAX_PROMPT_TOKENS}"
            
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
            
            # Verify servers
            for pid in "${PIDS[@]}"; do
                if ! ps -p "$pid" > /dev/null 2>&1; then
                    echo "ERROR: Server died."; stop_servers; cleanup; exit 1
                fi
            done
            
            stop_servers
            
            # Wait for ports to clear
            all_ports=("${PROXY_PORT}" "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}")
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