#!/bin/bash
set -euo pipefail

# ==========================================
# NEW CONFIGURATION FOR DIRECTORY ITERATION
# ==========================================
# 设置包含 benchmark 结果的根目录。如果设置了这个路径，脚本将忽略下面的 LIST 设置，
# 转而遍历该目录下的子目录进行复现测试。
# BASELINE_ROOT_DIR="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_500_baseline"
BASELINE_ROOT_DIR=""
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-mysharegpt}
MODEL=${MODEL:-/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-40960}
QWEN_VLLM="false" #只有BENCH_DATASET_NAME为qwen才起作用 true决定用模型最大上下文长度计算max-tokens false用baseline数据的输出作为max-tokens
TEST_MODEL="qwen"  # 用于端到端测试的模型名称标识
SAVE_SAMPLE="true"
# BASELINE_ROOT_DIR=""  # 如果为空，则回退到原有的列表循环模式

# ==========================================
# ORIGINAL CONFIGURATION
# ==========================================
USE_PASTFUTURE_SCHEDULER="false"  # 或者 "false"
USE_ACTIVATION_PREDICTOR="false"  # 或者 "false"
TEST_ALL="false"  # 是否运行端到端实验
USE_AIMD_SCHEDULER="false"
TEST_ABLATION_P2D="false"  # 是否测试请求感知调度的消融实验
TEST_ABLATION_DECODE="false"  # 是否测试解码器调度的消融实验
IGNORE="true"  # 是否忽略EOS标记

# 注意：如果启用了 BASELINE_ROOT_DIR，BASELINE_CSV_PATH 将会被自动覆盖
BASELINE_CSV_PATH="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_119/benchmark_np1000_rr4_mt8192_20260119_150213_2p4d/dataset_result/test_results_20260119_151551.csv"



BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a}
SAVE_OUTPUT=${SAVE_OUTPUT:-True}
#这里设置为float32以匹配预测器
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}  # float16 or bfloat16
# DECODE_METRICS_PORTS=${DECODE_METRICS_PORTS:-9400,9401,9402,9403,9404,9405,8406,9407}
BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}
# Model and general
 #/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct   # local path or HF id
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
# Base directory for all benchmark results
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result/experiment_paper/tmp/qwen_baseline_auto_126_2p4d}
# Proxy
PROXY_PORT=${PROXY_PORT:-29002}
BENCH_PORT=${BENCH_PORT:-25007}

# 这里的默认值仅在 BASELINE_ROOT_DIR 为空时生效，否则会被动态覆盖
PREFILL_GPUS=${PREFILL_GPUS:-2,3}
PREFILL_PORTS=${PREFILL_PORTS:-25001,25002}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-25010,25011}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-8}

# 这里的默认值仅在 BASELINE_ROOT_DIR 为空时生效，否则会被动态覆盖
DECODE_GPUS=${DECODE_GPUS:-4,5,6,7}
DECODE_PORTS=${DECODE_PORTS:-25020,25021,25022,25023}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-25030,25031,25032,25033}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}
DECODE_TENSOR_POOL_MEMORY=${DECODE_TENSOR_POOL_MEMORY:-16}

KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e1}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}
# vLLM options
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}   # 1=true, 0=false
VLLM_SEED=${VLLM_SEED:-42}


DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}
# Benchmark script and parameters
BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_baseline.py}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}

BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT=${BENCH_GOODPUT:-ttft:1000 tpot:50}
CACULATE_GOODPUT=${CACULATE_GOODPUT:-tpot:50}
# ==========================================
# SWEEP CONFIGURATION (Used only if BASELINE_ROOT_DIR is empty)
# ==========================================
MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS:-1}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-32768}
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"500"}
BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"4,8,12,16,20"}
BENCH_MAX_TOKENS_LIST=${BENCH_MAX_TOKENS_LIST:-"32768"}

SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}
# Misc
if [ "$TEST_ALL" = "true" ]; then
    PROXY_SCRIPT=${PROXY_SCRIPT:-test_disagg_proxy.py}
elif [ "$TEST_ABLATION_DECODE" = "true" ]; then
    PROXY_SCRIPT=${PROXY_SCRIPT:-test_disagg_proxy_decode.py}
elif [ "$TEST_ABLATION_P2D" = "true" ]; then
    PROXY_SCRIPT=${PROXY_SCRIPT:-test_disagg_proxy_p2d.py}
else
    PROXY_SCRIPT=${PROXY_SCRIPT:-disagg_proxy_p2p_nccl_xpyd.py}
fi
# ---------------------------
# End USER CONFIG
# ---------------------------
cd "$(dirname "${BASH_SOURCE[0]}")"
# Tracks per-run process pids and process group ids
PIDS=()     # individual PIDs for checking / logging
PGIDS=()    # process group IDs (negative kills will use these)

# Setup directories for current benchmark run
setup_directories() {
    local num_prompts=$1
    local req_rate=$2
    local max_tokens=$3
    local timestamp=$4
    
    BENCHMARK_DIR="${BASE_RESULT_DIR}/benchmark_np${num_prompts}_rr${req_rate}_mt${max_tokens}_${timestamp}_2p4d"
    mkdir -p "$BENCHMARK_DIR"
    
    # Create subdirectories
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

# Convert a comma-separated string into a JSON array string
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
"test_model": "${TEST_MODEL}",
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

# helper: wait until port is free (no LISTEN) or timeout (secs)
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
    set +e

    local self_pgid
    self_pgid=$(ps -o pgid= $$ | tr -d ' ')

    echo "Stopping servers (self PGID=${self_pgid})..."

    # ---- [1] 优先停止 Proxy（最关键）----
    if [[ -n "${PROXY_PGID:-}" && "$PROXY_PGID" != "$self_pgid" ]]; then
        echo "Stopping proxy PGID=${PROXY_PGID}"
        kill -TERM -"$PROXY_PGID" 2>/dev/null || true
        sleep 1
    fi

    # ---- [2] 停止其他实例（prefill / decode）----
    for pg in "${PGIDS[@]}"; do
        [[ -z "$pg" ]] && continue
        [[ "$pg" == "$self_pgid" ]] && continue
        [[ "$pg" == "$PROXY_PGID" ]] && continue

        kill -TERM -"$pg" 2>/dev/null || true
    done

    sleep 3

    # ---- [3] 强杀兜底 ----
    if [[ -n "${PROXY_PGID:-}" && "$PROXY_PGID" != "$self_pgid" ]]; then
        kill -KILL -"$PROXY_PGID" 2>/dev/null || true
    fi

    for pg in "${PGIDS[@]}"; do
        [[ -z "$pg" ]] && continue
        [[ "$pg" == "$self_pgid" ]] && continue
        [[ "$pg" == "$PROXY_PGID" ]] && continue

        kill -KILL -"$pg" 2>/dev/null || true
    done

    PIDS=()
    PGIDS=()
    unset PROXY_PGID

    set -e
}


cleanup() {
    echo "Cleaning up..."
    stop_servers
}

trap 'cleanup; exit 1' INT TERM
# stop_servers() {
#     if [[ ${#PGIDS[@]} -eq 0 && ${#PIDS[@]} -eq 0 ]]; then return; fi
#     echo "Stopping servers..."
#     for pg in "${PGIDS[@]}"; do
#         if [[ -n "$pg" ]]; then kill -TERM -"$pg" 2>/dev/null || true; fi
#     done
#     sleep 3
#     for pg in "${PGIDS[@]}"; do
#         if pgrep -g "$pg" >/dev/null 2>&1; then kill -KILL -"$pg" 2>/dev/null || true; fi
#     done
#     pkill -9 -f "$PROXY_SCRIPT" >/dev/null 2>&1 || true
#     pkill -9 -f vllm >/dev/null 2>&1 || true
#     PIDS=()
#     PGIDS=()
# }

# cleanup() {
#     echo "Cleaning up..."
#     stop_servers
#     ports_to_check=()
#     IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
#     IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
#     ports_to_check+=("${PROXY_PORT}")
#     for p in "${PREFILL_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
#     for p in "${DECODE_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
#     for port in "${ports_to_check[@]}"; do
#         if wait_for_port_free "$port" 10; then :; else echo "Port ${port} stuck?"; fi
#     done
#     # 注意：在 main 函数循环中，不要使用 exit 0，否则会中断循环
#     # 这里如果是trap调用的，需要退出
# }

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
    # PROXY_PGID="$proxy_pgid"
    PROXY_PGID="$proxy_pgid"
    export PROXY_PGID

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
            --test-model ${TEST_MODEL} \
            --gpu-memory-utilization ${DECODE_GPU_MEMORY_UTILIZATION} \
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

# ==========================================
# HELPER: RUN SINGLE BENCHMARK ITERATION
# ==========================================
run_one_benchmark() {
    local num_prompts=$1
    local req_rate=$2
    local max_tokens=$3
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    setup_directories "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"
    
    echo "========================================"
    echo "Run: Prompts=${num_prompts}, Rate=${req_rate}, MaxTokens=${max_tokens}"
    echo "Dir: ${BENCHMARK_DIR}"
    echo "Baseline CSV: ${BASELINE_CSV_PATH}"
    echo "Prefill GPUs: ${PREFILL_GPUS}"
    echo "Decode GPUs:  ${DECODE_GPUS}"
    echo "========================================"
    dump_config_json "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"
    
    if ! start_servers "$timestamp"; then
        echo "Failed to start servers."
        cleanup # 这里 cleanup 会 exit 0，如果是在 loop 中使用，需要注意 cleanup 行为
        # 因为 trap 的存在，这里调用 stop_servers 即可，不需要 exit
        stop_servers
        return 1
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
        --min-prompt-tokens ${MIN_PROMPT_TOKENS} \
        --max-prompt-tokens ${MAX_PROMPT_TOKENS}"
    
    # 条件参数
    if [ "$IGNORE" = "true" ]; then
        CMD="$CMD --ignore-eos"
        echo "ignore eos"
    fi
    if [ "$TEST_ABLATION_P2D" = "true" ]; then
        CMD="$CMD --ablation-p2d"
        CMD="$CMD --baseline-csv-path ${BASELINE_CSV_PATH}"
    fi
    if [ "$TEST_ABLATION_DECODE" = "true" ]; then
        CMD="$CMD --ablation-decode"
        CMD="$CMD --baseline-csv-path ${BASELINE_CSV_PATH}"
    fi
    if [ "$TEST_ALL" = "true" ]; then
        CMD="$CMD --all"
        CMD="$CMD --baseline-csv-path ${BASELINE_CSV_PATH}"
    fi
    if [ "$SAVE_SAMPLE" = "true" ]; then
        CMD="$CMD --save-sample"
    fi
    if [ "$QWEN_VLLM" = "true" ]; then
        CMD="$CMD --qwen-vllm"
        CMD="$CMD --baseline-csv-path ${BASELINE_CSV_PATH}"
    fi
    # if [ "$QWEN_VLLM" = "false" ]; then
    #     CMD="$CMD --baseline-csv-path ${BASELINE_CSV_PATH}"
    # fi
    CMD="$CMD --test-model ${TEST_MODEL}"
    CMD="$CMD --goodput ${BENCH_GOODPUT}"
    
    echo "Starting Benchmark Client..."
    setsid env BENCHMARK_INSTANCE_ID="benchmark-${num_prompts}" bash -c "exec $CMD" > "$log_file" 2>&1 &
    client_pid=$!
    wait "$client_pid"
    echo "Benchmark finished."
    plot_decode_load "$timestamp"
    
    # Verify servers still running (optional)
    for pid in "${PIDS[@]}"; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
                echo "ERROR: Server died."; stop_servers; return 1
        fi
    done
    
    # stop servers for this run
    stop_servers
    
    # Wait for ports to be free
    all_ports=("${PROXY_PORT}" "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}")
    for p in "${all_ports[@]}"; do
            wait_for_port_free "$p" 15
    done
    
    echo "Sleeping ${SLEEP_BETWEEN_RUNS}s..."
    sleep ${SLEEP_BETWEEN_RUNS}
}

trap 'cleanup; exit 1' INT TERM
# 注意：移除了 EXIT trap 自动调用 cleanup，因为我们在脚本中有两套逻辑，避免混乱退出。
# 手动在错误或结束时清理。

# ==========================================
# RESOURCE ALLOCATOR HELPER
# ==========================================
generate_resources() {
    local n_prefill=$1
    local n_decode=$2
    
    
    
    local p_gpus_list=()
    local d_gpus_list=()
    local p_ports_list=()
    local p_kv_ports_list=()
    local d_ports_list=()
    local d_kv_ports_list=()
    # GPU改成从2开始分配，保留0,1给系统和其他任务
    # Generate Prefill Configs
    for ((i=0; i<n_prefill; i++)); do
        local gpu_idx=$((2 + i))
        p_gpus_list+=("$gpu_idx")
        p_ports_list+=("$((25000 + i + 1))") # 25001...
        p_kv_ports_list+=("$((25010 + i + 1))") # 25011...
    done
    # Generate Decode Configs
    for ((i=0; i<n_decode; i++)); do
        local gpu_idx=$((2 + n_prefill + i))
        d_gpus_list+=("$gpu_idx")
        d_ports_list+=("$((25020 + i + 1))") # 25021...
        d_kv_ports_list+=("$((25030 + i + 1))") # 25031...
    done
    
    # 假设 GPU 从 0 开始连续分配
    # Prefill: 0 到 n_prefill-1
    # Decode: n_prefill 到 n_prefill + n_decode - 1
    # Generate Prefill Configs
    # for ((i=2; i<n_prefill; i++)); do
    #     p_gpus_list+=("$i")
    #     p_ports_list+=("$((25000 + i + 1))") # 25001...
    #     p_kv_ports_list+=("$((25010 + i + 1))") # 25011...
    # done
    
    # # Generate Decode Configs
    # for ((i=2; i<n_decode; i++)); do
    #     local gpu_idx=$((n_prefill + i))
    #     d_gpus_list+=("$gpu_idx")
    #     d_ports_list+=("$((25020 + i + 1))") # 25021...
    #     d_kv_ports_list+=("$((25030 + i + 1))") # 25031...
    # done
    
    # Join arrays with commas
    PREFILL_GPUS=$(IFS=,; echo "${p_gpus_list[*]}")
    PREFILL_PORTS=$(IFS=,; echo "${p_ports_list[*]}")
    PREFILL_KV_PORTS=$(IFS=,; echo "${p_kv_ports_list[*]}")
    
    DECODE_GPUS=$(IFS=,; echo "${d_gpus_list[*]}")
    DECODE_PORTS=$(IFS=,; echo "${d_ports_list[*]}")
    DECODE_KV_PORTS=$(IFS=,; echo "${d_kv_ports_list[*]}")
    
    export PREFILL_GPUS PREFILL_PORTS PREFILL_KV_PORTS
    export DECODE_GPUS DECODE_PORTS DECODE_KV_PORTS
}

main() {
    check_required_files
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart
    
    # 判断是否运行目录遍历模式
    if [ -n "${BASELINE_ROOT_DIR:-}" ] && [ -d "$BASELINE_ROOT_DIR" ]; then
        echo "========================================================"
        echo "Running in AUTOMATED DIRECTORY TRAVERSAL MODE"
        echo "Root Dir: $BASELINE_ROOT_DIR"
        echo "========================================================"
        
        # 遍历目录
        for dir_path in "$BASELINE_ROOT_DIR"/benchmark_np*_rr*_mt*_2p*d; do
            if [ ! -d "$dir_path" ]; then continue; fi
            
            dir_name=$(basename "$dir_path")
            echo "Processing: $dir_name"
            
            # 使用 Regex 解析参数
            # 格式: benchmark_np1000_rr4_mt8192_20260119_150213_2p4d
            # 正则: np([0-9]+) ... rr([0-9.]+) ... mt([0-9]+) ... ([0-9]+)p([0-9]+)d
            if [[ $dir_name =~ benchmark_np([0-9]+)_rr([0-9.]+)_mt([0-9]+)_.*_([0-9]+)p([0-9]+)d ]]; then
                # 测试500 prompt
                np="500"
                # np="${BASH_REMATCH[1]}"
                rr="${BASH_REMATCH[2]}"
                mt="32768"
                p_count="${BASH_REMATCH[4]}"
                d_count="${BASH_REMATCH[5]}"
                
                # # 判断req_rate如果是12或者16就跳过
                # if [ "$rr" = "12" ] || [ "$rr" = "16" ]; then
                #     echo "[INFO] Skipping req_rate=$rr as per configuration."
                #     continue
                # fi
                # if [ "$rr" = "12" ] || [ "$rr" = "16" ]; then
                #     echo "[INFO] Skipping req_rate=$rr as per configuration."
                #     continue
                # fi
                echo "Parsed -> Prompts: $np, Rate: $rr, MaxTokens: $mt, P: $p_count, D: $d_count"
                
                # 1. 查找 Baseline CSV
                # 查找 dataset_result 下的 test_results_*.csv
                # sort -r 保证取到最新的（如果有多个），或者 head -n 1
                found_csv=$(find "$dir_path/dataset_result" -name "test_results_*.csv" | head -n 1)
                
                if [ -z "$found_csv" ]; then
                    echo "[WARN] No csv found in $dir_path/dataset_result, skipping..."
                    continue
                fi
                
                # 设置全局变量
                export BASELINE_CSV_PATH="$found_csv"
                
                # 2. 动态设置 GPU 和 端口
                generate_resources "$p_count" "$d_count"
                
                # 3. 运行测试
                run_one_benchmark "$np" "$rr" "$mt"
                
            else
                echo "[WARN] Skipping $dir_name - regex did not match format."
            fi
        done
        
    else
        # ==========================================
        # ORIGINAL LOOP LOGIC
        # ==========================================
        echo "Running in ORIGINAL LIST CONFIG MODE"
        
        IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"
        IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$BENCH_REQUEST_RATE_LIST"
        IFS=',' read -ra MAX_TOKENS_ARRAY <<< "$BENCH_MAX_TOKENS_LIST"
        
        # 3-Layer Nested Loop
        for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
          for req_rate in "${REQUEST_RATE_ARRAY[@]}"; do
            for max_tokens in "${MAX_TOKENS_ARRAY[@]}"; do
                run_one_benchmark "$num_prompts" "$req_rate" "$max_tokens"
            done
          done
        done
    fi
    
    echo "All benchmarks finished."
    exit 0
}

main "$@"