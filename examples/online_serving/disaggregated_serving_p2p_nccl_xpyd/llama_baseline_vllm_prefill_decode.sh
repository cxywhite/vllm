set -euo pipefail

# ---------------------------
# USER CONFIG
# ---------------------------

# 定义是否使用pastfuture scheduler的变量
USE_PASTFUTURE_SCHEDULER="false"
USE_ACTIVATION_PREDICTOR="false"
USE_AIMD_SCHEDULER="false"
TEST_ABLATION_P2D="false"
TEST_MODEL="llama"  # 用于端到端测试的模型名称标识

VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./wt_experiment/experiment_paper/tmp/baseline_212}

PROXY_PORT=${PROXY_PORT:-28002}


PREFILL_GPUS=${PREFILL_GPUS:-0,1}
PREFILL_PORTS=${PREFILL_PORTS:-22001,22002}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-22011,22012}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-8}

DECODE_GPUS=${DECODE_GPUS:-2,3,4,5}
DECODE_PORTS=${DECODE_PORTS:-22020,22021,22022,22023}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-22032,22033,22034,22035}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}
DECODE_TENSOR_POOL_MEMORY=${DECODE_TENSOR_POOL_MEMORY:-16}

KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e9}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
VLLM_SEED=${VLLM_SEED:-42}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}

# ---------------------------
# End USER CONFIG
# ---------------------------
cd "$(dirname "${BASH_SOURCE[0]}")"
PIDS=()
PGIDS=()

setup_directories() {
    local timestamp=$1
    
    BENCHMARK_DIR="${BASE_RESULT_DIR}/prefill_decode_${timestamp}"
    mkdir -p "$BENCHMARK_DIR"
    
    CONFIG_DIR="${BENCHMARK_DIR}/config"
    LOG_DIR="${BENCHMARK_DIR}/log"
    RESULT_DIR="${BENCHMARK_DIR}/dataset_result"
    
    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
    
    echo "Created benchmark directories:"
    echo "  Base: $BENCHMARK_DIR"
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
    echo "Launching servers..."

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
    check_num_gpus
    ensure_python_library_installed vllm
    timestamp=$(date +%Y%m%d_%H%M%S)

    setup_directories "$timestamp"

    if ! start_servers "$timestamp"; then
        echo "Failed to start servers."
        cleanup; exit 1
    fi

    echo "Servers started. Press Ctrl+C to stop."
    wait
}

main "$@"
