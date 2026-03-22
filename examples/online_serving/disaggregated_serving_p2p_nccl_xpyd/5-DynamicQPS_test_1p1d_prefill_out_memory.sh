set -euo pipefail

# ---------------------------
# USER CONFIG - SIMPLIFIED FOR 1P1D PREFILL OOM BOUNDARY TEST (PURE RANDOM DATASET)
# ---------------------------

# 强制 1 prefill + 1 decode（GPU 配置完全保持原风格）
PREFILL_GPUS=${PREFILL_GPUS:-6}
PREFILL_PORTS=${PREFILL_PORTS:-22001}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-22011}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-8}

DECODE_GPUS=${DECODE_GPUS:-7}
DECODE_PORTS=${DECODE_PORTS:-22020}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-22032}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}
DECODE_TENSOR_POOL_MEMORY=${DECODE_TENSOR_POOL_MEMORY:-16}

# 基础配置
MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
VLLM_SEED=${VLLM_SEED:-42}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}

KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e9}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

# Benchmark 配置（使用内置 random dataset）
BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_load_balance.py}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_PORT=${BENCH_PORT:-22007}
BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}
SAVE_OUTPUT=${SAVE_OUTPUT:-True}
IGNORE=${IGNORE:-true}
BENCH_GOODPUT=${BENCH_GOODPUT:-ttft:1000 tpot:50}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-1}   # 输出长度固定（prefill OOM 测试无需长输出）

# OOM 测试扫参（batchsize × inputlength）
BATCH_SIZE_LIST=${BATCH_SIZE_LIST:-"64,128,256,512,1024"}
INPUT_LENGTH_LIST=${INPUT_LENGTH_LIST:-"8180"}

# 结果目录
BASE_RESULT_DIR=${BASE_RESULT_DIR:-/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/tmp/baseline_316/prefill_oom_test_1p1d_random}

# Proxy（默认 disagg proxy）
PROXY_SCRIPT=disagg_proxy_p2p_nccl_xpyd.py
PROXY_PORT=${PROXY_PORT:-28002}

# Misc
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}

# ---------------------------
# End USER CONFIG
# ---------------------------

cd "$(dirname "${BASH_SOURCE[0]}")"
PIDS=()
PGIDS=()

setup_directories() {
    local batch_size=$1
    local input_length=$2
    local timestamp=$3
    
    BENCHMARK_DIR="${BASE_RESULT_DIR}/test_b${batch_size}_in${input_length}_${timestamp}_1p1d_random"
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
    local current_batch=$1
    local current_input=$2
    local current_maxtokens=$3
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
    "endpoint": "/v1/chat/completions",
    "dataset_name": "random",
    "dataset_path": "synthetic_random_exact",
    "max_concurrency": ${current_batch},
    "request_rate": "inf",
    "maxtokenscustom": ${current_maxtokens},
    "temperature": ${BENCH_TEMPERATURE},
    "top_p": ${BENCH_TOP_P},
    "top_k": ${BENCH_TOP_K},
    "repetition_penalty": ${BENCH_REPETITION_PENALTY},
    "goodput": "${BENCH_GOODPUT}",
    "num_prompts": ${current_batch},
    "batch_size": ${current_batch},
    "input_length": ${current_input},
    "random_range_ratio": 0.0,
    "random_prefix_len": 0,
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

start_servers() {
    PIDS=()
    PGIDS=()
    local timestamp=$1
    echo "Launching 1 Prefill + 1 Decode servers..."
    
    # Proxy
    setsid env PROXY_PORT="${PROXY_PORT}" BENCH_PORT="${BENCH_PORT}" VLLM_DTYPE="${VLLM_DTYPE}" MODEL_CONFIG_PATH="${MODEL}/config.json" bash -c "exec python3 \"$PROXY_SCRIPT\"" &> "${LOG_DIR}/proxy_${timestamp}.log" &
    proxy_pid=$!
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    PIDS+=("$proxy_pid"); PGIDS+=("$proxy_pgid")
    
    # Prefill
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"
    
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id=${PREFILL_GPU_ARRAY[$i]}
        port=${PREFILL_PORT_ARRAY[$i]}
        kv_port=${PREFILL_KV_PORT_ARRAY[$i]}
        
        KV_JSON_CFG='{"kv_connector":"'"${KV_CONNECTOR}"'","kv_role":"kv_producer","kv_buffer_size":"'"${KV_PRODUCER_BUFFER}"'","kv_port":"'"$kv_port"'","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"'"${PROXY_PORT}"'","http_port":"'"$port"'","send_type":"'"${KV_SEND_TYPE}"'","mem_pool_size_gb":"'"${PREFILL_TENSOR_POOL_MEMORY}"'","nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"}}'
        
        CMD="vllm serve \"${MODEL}\" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port \"$port\" \
            --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} \
            --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --gpu-memory-utilization ${PREFILL_GPU_MEMORY_UTILIZATION} \
            --kv-transfer-config '$KV_JSON_CFG'"

        setsid env CUDA_VISIBLE_DEVICES="$gpu_id" VLLM_USE_V1=1 bash -c "exec $CMD" > "${LOG_DIR}/prefill_${timestamp}.log" 2>&1 &
        pid=$!; pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid"); PGIDS+=("$pgid")
    done
    
    # Decode
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    IFS=',' read -ra DECODE_KV_PORT_ARRAY <<< "$DECODE_KV_PORTS"
    
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id=${DECODE_GPU_ARRAY[$i]}
        port=${DECODE_PORT_ARRAY[$i]}
        kv_port=${DECODE_KV_PORT_ARRAY[$i]}
        
        KV_JSON_CFG='{"kv_connector":"'"${KV_CONNECTOR}"'","kv_role":"kv_consumer","kv_buffer_size":"'"${KV_CONSUMER_BUFFER}"'","kv_port":"'"$kv_port"'","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"'"${PROXY_PORT}"'","http_port":"'"$port"'","send_type":"'"${KV_SEND_TYPE}"'","mem_pool_size_gb":"'"${DECODE_TENSOR_POOL_MEMORY}"'","nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"}}'
        
        CMD="vllm serve \"${MODEL}\" \
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
            --swap-space 0 \
            --kv-transfer-config '$KV_JSON_CFG'"

        setsid env DECODE_INSTANCE_ID="decode-0" VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "exec $CMD" > "${LOG_DIR}/decode_${timestamp}.log" 2>&1 &
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
    ensure_python_library_installed vllm
    ensure_python_library_installed quart
    
    # Parse sweep lists
    IFS=',' read -ra BATCH_SIZE_ARRAY <<< "$BATCH_SIZE_LIST"
    IFS=',' read -ra INPUT_LENGTH_ARRAY <<< "$INPUT_LENGTH_LIST"
    
    # 双层循环：batchsize × inputlength（使用 benchmark 内置 random dataset）
    for batch_size in "${BATCH_SIZE_ARRAY[@]}"; do
      for input_length in "${INPUT_LENGTH_ARRAY[@]}"; do
            timestamp=$(date +%Y%m%d_%H%M%S)
            
            setup_directories "$batch_size" "$input_length" "$timestamp"
            
            echo "========================================"
            echo "Run: BatchSize=${batch_size}, InputLength=${input_length} (random exact, prefix=0, rate=inf)"
            echo "Dir: ${BENCHMARK_DIR}"
            echo "========================================"
            
            BENCH_DATASET_PATH="synthetic_random_exact"   # 仅用于 config JSON 记录
            
            dump_config_json "$batch_size" "$input_length" "${OUTPUT_TOKENS}" "$timestamp"
            
            if ! start_servers "$timestamp"; then
                echo "Failed to start servers."
                cleanup; exit 1
            fi
            
            log_file="${LOG_DIR}/bench_b${batch_size}_i${input_length}.log"
            
            # Benchmark 命令（直接使用内置 random dataset + 精确参数）
            CMD="python3 \"$BENCH_SCRIPT\" \
                --backend openai-chat \
                --port \"${BENCH_PORT}\" \
                --endpoint '/v1/chat/completions' \
                --model \"${BENCH_MODEL}\" \
                --dataset-name random \
                --save-output ${SAVE_OUTPUT} \
                --out-path \"${RESULT_DIR}\" \
                --seed ${VLLM_SEED} \
                --num-prompts ${batch_size} \
                --max-concurrency ${batch_size} \
                --temperature ${BENCH_TEMPERATURE} \
                --top-p ${BENCH_TOP_P} \
                --top-k ${BENCH_TOP_K} \
                --repetition-penalty ${BENCH_REPETITION_PENALTY} \
                --request-rate inf \
                --goodput ${BENCH_GOODPUT} \
                --random-input-len ${input_length} \
                --random-output-len ${OUTPUT_TOKENS} \
                --random-range-ratio 0.0 \
                --random-prefix-len 0"
            
            if [ "$IGNORE" = "true" ]; then CMD="$CMD --ignore-eos"; fi
            
            echo "Starting Benchmark Client (batch=${batch_size}, input=${input_length} exact tokens, all sent simultaneously)..."
            setsid env BENCHMARK_INSTANCE_ID="benchmark-oomtest" bash -c "exec $CMD" > "$log_file" 2>&1 &
            client_pid=$!
            wait "$client_pid"
            echo "Benchmark finished (check prefill log for OOM)."
            
            # 验证服务器状态
            servers_alive=true
            for pid in "${PIDS[@]}"; do
                if ! ps -p "$pid" > /dev/null 2>&1; then
                    echo "WARN: Server died (likely Prefill OOM) - continuing to next test..."
                    servers_alive=false
                fi
            done
            
            stop_servers
            
            # 清理端口
            all_ports=("${PROXY_PORT}" "${PREFILL_PORTS}" "${DECODE_PORTS}")
            for p in "${all_ports[@]}"; do
                wait_for_port_free "$p" 15
            done
            
            echo "Sleeping ${SLEEP_BETWEEN_RUNS}s before next test..."
            sleep ${SLEEP_BETWEEN_RUNS}
            
        done
    done
    
    echo "All OOM boundary tests finished."
    exit 0
}

main "$@"