#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Disaggregated vLLM evaluation script
# Iterates over CSV datasets under evaluation_dataset and benchmarks each one
# using benchmark_serving_trace.py.
# ----------------------------------------------------------------------------

####################
# USER CONFIG
####################

# Which subtree to run:
#   llama -> evaluation_dataset/llama/**/*.csv
#   qwen  -> evaluation_dataset/qwen/**/*.csv
#   all   -> evaluation_dataset/**/*.csv
MODEL_TYPE=${MODEL_TYPE:-qwen}

TEST_MODEL=${TEST_MODEL:-qwen}
# MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
MODEL=${MODEL:-/root/.cache/huggingface/hub/Qwen-2.5-7b-Instruct}
# Relative to this script directory, or absolute path.
DATASET_BASE_DIR=${DATASET_BASE_DIR:-../../dataset/evaluation_dataset}

TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# Base output directory. Per-run subdirectories are created automatically.
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result}

# User-specified conflict resolutions
PROXY_PORT=${PROXY_PORT:-28002}
BENCH_PORT=${BENCH_PORT:-22007}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

# Keep current 1p1d defaults unless explicitly overridden.
PREFILL_GPUS=${PREFILL_GPUS:-0}
PREFILL_PORTS=${PREFILL_PORTS:-20002}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-20010}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-32}

DECODE_GPUS=${DECODE_GPUS:-1,2,3}
DECODE_PORTS=${DECODE_PORTS:-20020,20021,20022}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-20030,20031,20032}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}
DECODE_TENSOR_POOL_MEMORY=${DECODE_TENSOR_POOL_MEMORY:-32}

KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e9}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
VLLM_SEED=${VLLM_SEED:-42}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-}
DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}

BENCH_SCRIPT=${BENCH_SCRIPT:-../../../../../../benchmarks/benchmark_serving_trace.py}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_BACKEND=${BENCH_BACKEND:-openai-chat}
BENCH_ENDPOINT=${BENCH_ENDPOINT:-/v1/chat/completions}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-trace}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT="${BENCH_GOODPUT:-ttft:1000 tpot:50}"

BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}

SAVE_OUTPUT=${SAVE_OUTPUT:-True}
SAVE_SAMPLE=${SAVE_SAMPLE:-true}
BENCH_IGNORE_EOS=${BENCH_IGNORE_EOS:-ignore-eos}

USE_TRACE_TIMESTAMPS=${USE_TRACE_TIMESTAMPS:-true}
BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"8,6"}
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"1000"}
BENCH_MAX_TOKENS_LIST=${BENCH_MAX_TOKENS_LIST:-"8192"}

RUN_REPEAT_PER_CONFIG=${RUN_REPEAT_PER_CONFIG:-1}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}

CACULATE_GOODPUT="${CACULATE_GOODPUT:-tpot:50}"
INSTANCE_DOWN_TIMEOUT_SECONDS=${INSTANCE_DOWN_TIMEOUT_SECONDS:-120}
INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS=${INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS:-5}

PROXY_BASE_DIR=${PROXY_BASE_DIR:-/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd}
PROXY_SCRIPT=${PROXY_SCRIPT:-${PROXY_BASE_DIR}/disagg_proxy_p2p_nccl_xpyd.py}
PLOT_SCRIPT=${PLOT_SCRIPT:-${PROXY_BASE_DIR}/plot_decode_load2.py}

####################
# END USER CONFIG
####################

cd "$(dirname "${BASH_SOURCE[0]}")"
DATASET_BASE_DIR="$(realpath "$DATASET_BASE_DIR")"

PIDS=()
PGIDS=()
PREFILL_PIDS=()
DECODE_PIDS=()

CURRENT_BENCH_DATASET_PATH=""
CURRENT_BENCH_DATASET_LABEL=""

BENCHMARK_DIR=""
CONFIG_DIR=""
LOG_DIR=""
RESULT_DIR=""
MODEL_MT_TAG_FOR_NAME=""

_CLEANED_UP=0

is_true() {
    local v="${1:-}"
    [[ "${v,,}" == "true" ]]
}

sanitize_name() {
    local raw_name="$1"
    raw_name="${raw_name// /_}"
    raw_name="$(printf '%s' "$raw_name" | sed 's/[^[:alnum:]_.-]/_/g')"
    printf '%s' "$raw_name"
}

apply_model_dependent_defaults() {
    local model_lc
    model_lc="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"

    if [[ -z "$VLLM_MAX_MODEL_LEN" ]]; then
        if [[ "$model_lc" == *"llama-3-8b-instruct"* ]] || [[ "$model_lc" == *"meta-llama-3-8b-instruct"* ]]; then
            VLLM_MAX_MODEL_LEN=8192
        elif [[ "$model_lc" == *"qwen2.5-7b-instruct"* ]]; then
            VLLM_MAX_MODEL_LEN=32768
        else
            VLLM_MAX_MODEL_LEN=32768
        fi
    fi

    if [[ -z "$PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS" ]]; then
        if [[ "$model_lc" == *"llama-3-8b-instruct"* ]] || [[ "$model_lc" == *"meta-llama-3-8b-instruct"* ]]; then
            PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768
        elif [[ "$model_lc" == *"qwen2.5-7b-instruct"* ]]; then
            PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=40960
        else
            PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768
        fi
    fi
}

resolve_model_mt_tag_for_name() {
    local type_lc model_lc test_model_lc
    type_lc="$(echo "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')"
    model_lc="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
    test_model_lc="$(echo "$TEST_MODEL" | tr '[:upper:]' '[:lower:]')"

    if [[ "$type_lc" == "llama" ]] || [[ "$model_lc" == *"llama"* ]] || [[ "$test_model_lc" == *"llama"* ]]; then
        MODEL_MT_TAG_FOR_NAME="8192"
    elif [[ "$type_lc" == "qwen" ]] || [[ "$model_lc" == *"qwen"* ]] || [[ "$test_model_lc" == *"qwen"* ]]; then
        MODEL_MT_TAG_FOR_NAME="32768"
    else
        # Fallback to current runtime max model len when model family is unclear.
        MODEL_MT_TAG_FOR_NAME="${VLLM_MAX_MODEL_LEN}"
    fi
}

sync_tpot_between_proxy_and_benchmark() {
    local tpot_from_bench=""
    local bench_tokens="${BENCH_GOODPUT//,/ }"
    local item

    for item in $bench_tokens; do
        if [[ "$item" == tpot:* ]]; then
            tpot_from_bench="${item#tpot:}"
            break
        fi
    done

    if [[ -z "$tpot_from_bench" ]]; then
        echo "ERROR: BENCH_GOODPUT must include 'tpot:<value>', current: ${BENCH_GOODPUT}"
        exit 1
    fi

    local synced_tpot="tpot:${tpot_from_bench}"
    if [[ "$CACULATE_GOODPUT" != "$synced_tpot" ]]; then
        echo "Sync CACULATE_GOODPUT to BENCH_GOODPUT tpot: ${CACULATE_GOODPUT} -> ${synced_tpot}"
    fi
    CACULATE_GOODPUT="$synced_tpot"
}

array_to_json() {
    local s="$1"
    if [[ -z "$s" ]]; then
        printf '[]'
        return
    fi
    IFS=',' read -ra _arr <<< "$s"
    printf '['
    local first=1
    for v in "${_arr[@]}"; do
        v="$(echo "$v" | sed -e 's/^\s*//' -e 's/\s*$//')"
        if [[ $first -eq 1 ]]; then
            printf '"%s"' "$v"
            first=0
        else
            printf ',"%s"' "$v"
        fi
    done
    printf ']'
}

resolve_dataset_paths() {
    local root="$DATASET_BASE_DIR"
    local query_dir
    case "$MODEL_TYPE" in
        llama) query_dir="${root}/llama" ;;
        qwen) query_dir="${root}/qwen" ;;
        all|*) query_dir="${root}" ;;
    esac

    if [[ ! -d "$query_dir" ]]; then
        echo "ERROR: dataset directory not found: $query_dir"
        exit 1
    fi

    # Use NUL-delimited paths to safely handle any special characters.
    mapfile -d '' -t DATASET_PATH_ARRAY < <(find "$query_dir" -type f -name '*.csv' -print0 | sort -z)
    if [[ ${#DATASET_PATH_ARRAY[@]} -eq 0 ]]; then
        echo "ERROR: no CSV datasets found under: $query_dir"
        exit 1
    fi
}

set_current_dataset() {
    local dataset_path="$1"
    local dataset_file_name
    dataset_file_name="$(basename "$dataset_path")"

    CURRENT_BENCH_DATASET_PATH="$dataset_path"
    CURRENT_BENCH_DATASET_LABEL="$(sanitize_name "${dataset_file_name%.csv}")"
}

setup_directories() {
    local req_rate="$1"
    local max_tokens="$2"
    local timestamp="$3"
    local repeat_idx="$4"

    BENCHMARK_DIR="${BASE_RESULT_DIR}/benchmark_ds${CURRENT_BENCH_DATASET_LABEL}_rr${req_rate}_mt${MODEL_MT_TAG_FOR_NAME}_${timestamp}_rep${repeat_idx}"
    CONFIG_DIR="${BENCHMARK_DIR}/config"
    LOG_DIR="${BENCHMARK_DIR}/log"
    RESULT_DIR="${BENCHMARK_DIR}/dataset_result"

    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
}

check_required_files() {
    if [[ ! -f "$PROXY_SCRIPT" ]]; then
        echo "ERROR: proxy script not found: $PROXY_SCRIPT"
        exit 1
    fi
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        echo "ERROR: benchmark script not found: $BENCH_SCRIPT"
        exit 1
    fi
    if [[ ! -d "$DATASET_BASE_DIR" ]]; then
        echo "ERROR: dataset base dir not found: $DATASET_BASE_DIR"
        exit 1
    fi
}

check_num_gpus() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found"
        exit 1
    fi
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$num_gpus" -lt 2 ]]; then
        echo "ERROR: need at least 2 GPUs, found $num_gpus"
        exit 1
    fi
}

ensure_python_library_installed() {
    if ! python3 -c "import $1" >/dev/null 2>&1; then
        echo "ERROR: python library '$1' is not installed"
        exit 1
    fi
}

dump_config_json() {
    local req_rate="$1"
    local max_tokens="$2"
    local repeat_idx="$3"
    local ts="$4"
    local cfgfile="${CONFIG_DIR}/config_${ts}.json"

    cat > "$cfgfile" <<EOF
{
  "generated_at": "$(date --iso-8601=seconds)",
  "model_type": "${MODEL_TYPE}",
  "model": "${MODEL}",
  "test_model": "${TEST_MODEL}",
  "dataset_path": "${CURRENT_BENCH_DATASET_PATH}",
  "dataset_label": "${CURRENT_BENCH_DATASET_LABEL}",
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "proxy_port": ${PROXY_PORT},
  "bench_port": ${BENCH_PORT},
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
    "backend": "${BENCH_BACKEND}",
    "endpoint": "${BENCH_ENDPOINT}",
    "dataset_name": "${BENCH_DATASET_NAME}",
    "request_rate": "${req_rate}",
    "maxtokenscustom": ${max_tokens},
    "repeat_idx": ${repeat_idx},
    "save_output": "${SAVE_OUTPUT}",
    "save_sample": "${SAVE_SAMPLE}",
    "max_concurrency": ${BENCH_MAX_CONCURRENCY},
    "temperature": ${BENCH_TEMPERATURE},
    "top_p": ${BENCH_TOP_P},
    "top_k": ${BENCH_TOP_K},
    "repetition_penalty": ${BENCH_REPETITION_PENALTY},
    "goodput": "${BENCH_GOODPUT}"
  }
}
EOF
}

wait_for_server() {
    local port="$1"
    local start_time
    start_time=$(date +%s)

    echo "Waiting for server on port $port ..."
    while true; do
        if curl -s "localhost:${port}" >/dev/null 2>&1; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start_time >= TIMEOUT_SECONDS )); then
            echo "ERROR: timeout waiting for server on port $port"
            return 1
        fi
        sleep 1
    done
}

wait_for_port_free() {
    local port="$1"
    local timeout="${2:-10}"
    local start
    start=$(date +%s)

    while true; do
        if ! ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q .; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start >= timeout )); then
            return 1
        fi
        sleep 0.5
    done
}

stop_servers() {
    if [[ ${#PGIDS[@]} -eq 0 && ${#PIDS[@]} -eq 0 ]]; then
        return
    fi

    echo "Stopping servers..."
    for pg in "${PGIDS[@]}"; do
        [[ -n "$pg" ]] && kill -TERM -"$pg" 2>/dev/null || true
    done

    sleep 3

    for pg in "${PGIDS[@]}"; do
        pgrep -g "$pg" >/dev/null 2>&1 && kill -KILL -"$pg" 2>/dev/null || true
    done

    pkill -9 -f "$PROXY_SCRIPT" >/dev/null 2>&1 || true
    pkill -9 -f "vllm serve" >/dev/null 2>&1 || true

    PIDS=()
    PGIDS=()
    PREFILL_PIDS=()
    DECODE_PIDS=()
}

cleanup() {
    if [[ $_CLEANED_UP -eq 1 ]]; then
        return
    fi
    _CLEANED_UP=1

    stop_servers

    local ports_to_check=()
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    ports_to_check+=("${PROXY_PORT}")
    for p in "${PREFILL_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
    for p in "${DECODE_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
    for port in "${ports_to_check[@]}"; do
        wait_for_port_free "$port" 10 || true
    done
}

plot_decode_load() {
    local ts="$1"
    local proxy_log="${LOG_DIR}/proxy_${ts}.log"
    [[ -f "$PLOT_SCRIPT" ]] || return 0
    [[ -f "$proxy_log" ]] || return 0

    python3 "$PLOT_SCRIPT" --log_path "$proxy_log" > "${LOG_DIR}/plot_${ts}.log" 2>&1 || true
}

start_servers() {
    local timestamp="$1"
    PIDS=()
    PGIDS=()
    PREFILL_PIDS=()
    DECODE_PIDS=()

    echo "Launching servers..."

    setsid env \
        PROXY_PORT="${PROXY_PORT}" \
        BENCH_PORT="${BENCH_PORT}" \
        VLLM_DTYPE="${VLLM_DTYPE}" \
        MODEL_CONFIG_PATH="${MODEL}/config.json" \
        TPOT="${CACULATE_GOODPUT}" \
        bash -c "exec python3 \"$PROXY_SCRIPT\"" \
        &> "${LOG_DIR}/proxy_${timestamp}.log" &

    local proxy_pid=$!
    local proxy_pgid
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    PIDS+=("$proxy_pid")
    PGIDS+=("$proxy_pgid")

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"

    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]:-$((20002 + i))}
        local kv_port=${PREFILL_KV_PORT_ARRAY[$i]:-$((21001 + i))}

        local cmd
        cmd="vllm serve \"$MODEL\" \
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

        local kv_json_cfg='{'
        kv_json_cfg+='"kv_connector":"'"${KV_CONNECTOR}"'",'
        kv_json_cfg+='"kv_role":"kv_producer",'
        kv_json_cfg+='"kv_buffer_size":"'"${KV_PRODUCER_BUFFER}"'",'
        kv_json_cfg+='"kv_port":"'"$kv_port"'",'
        kv_json_cfg+='"kv_connector_extra_config":{'
        kv_json_cfg+='"proxy_ip":"0.0.0.0",'
        kv_json_cfg+='"proxy_port":"'"${PROXY_PORT}"'",'
        kv_json_cfg+='"http_port":"'"$port"'",'
        kv_json_cfg+='"send_type":"'"${KV_SEND_TYPE}"'",'
        kv_json_cfg+='"mem_pool_size_gb":"'"${PREFILL_TENSOR_POOL_MEMORY}"'",'
        kv_json_cfg+='"nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"'
        kv_json_cfg+='}}'

        cmd="$cmd --kv-transfer-config '$kv_json_cfg'"

        setsid env CUDA_VISIBLE_DEVICES="$gpu_id" VLLM_USE_V1=1 bash -c "exec $cmd" > "${LOG_DIR}/prefill${i}_${timestamp}.log" 2>&1 &
        local pid=$!
        local pgid
        pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid")
        PGIDS+=("$pgid")
        PREFILL_PIDS+=("$pid")
    done

    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    IFS=',' read -ra DECODE_KV_PORT_ARRAY <<< "$DECODE_KV_PORTS"

    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]:-$((20003 + i))}
        local kv_port=${DECODE_KV_PORT_ARRAY[$i]:-$((22001 + i))}

        local cmd
        cmd="vllm serve \"$MODEL\" \
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

        local kv_json_cfg='{'
        kv_json_cfg+='"kv_connector":"'"${KV_CONNECTOR}"'",'
        kv_json_cfg+='"kv_role":"kv_consumer",'
        kv_json_cfg+='"kv_buffer_size":"'"${KV_CONSUMER_BUFFER}"'",'
        kv_json_cfg+='"kv_port":"'"$kv_port"'",'
        kv_json_cfg+='"kv_connector_extra_config":{'
        kv_json_cfg+='"proxy_ip":"0.0.0.0",'
        kv_json_cfg+='"proxy_port":"'"${PROXY_PORT}"'",'
        kv_json_cfg+='"http_port":"'"$port"'",'
        kv_json_cfg+='"send_type":"'"${KV_SEND_TYPE}"'",'
        kv_json_cfg+='"mem_pool_size_gb":"'"${DECODE_TENSOR_POOL_MEMORY}"'",'
        kv_json_cfg+='"nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"'
        kv_json_cfg+='}}'

        cmd="$cmd --kv-transfer-config '$kv_json_cfg'"

        setsid env DECODE_INSTANCE_ID="decode-${i}" VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "exec $cmd" > "${LOG_DIR}/decode${i}_${timestamp}.log" 2>&1 &
        local pid=$!
        local pgid
        pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid")
        PGIDS+=("$pgid")
        DECODE_PIDS+=("$pid")
    done

    echo "Waiting for servers..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$port"; then
            echo "ERROR: failed to start server on port $port"
            stop_servers
            return 1
        fi
    done
    return 0
}

run_benchmark_with_health_guard() {
    local client_pid="$1"
    local timeout_seconds="${INSTANCE_DOWN_TIMEOUT_SECONDS}"
    local check_interval="${INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS}"
    declare -A down_since=()

    while ps -p "$client_pid" >/dev/null 2>&1; do
        local now
        now=$(date +%s)

        for i in "${!PREFILL_PIDS[@]}"; do
            local pid="${PREFILL_PIDS[$i]}"
            local key="prefill-${i}"
            if ! ps -p "$pid" >/dev/null 2>&1; then
                if [[ -z "${down_since[$key]:-}" ]]; then
                    down_since[$key]="$now"
                    echo "[WARN] ${key} down, start countdown"
                elif (( now - down_since[$key] >= timeout_seconds )); then
                    echo "[ERROR] ${key} down for ${timeout_seconds}s"
                    return 2
                fi
            else
                unset 'down_since[$key]'
            fi
        done

        for i in "${!DECODE_PIDS[@]}"; do
            local pid="${DECODE_PIDS[$i]}"
            local key="decode-${i}"
            if ! ps -p "$pid" >/dev/null 2>&1; then
                if [[ -z "${down_since[$key]:-}" ]]; then
                    down_since[$key]="$now"
                    echo "[WARN] ${key} down, start countdown"
                elif (( now - down_since[$key] >= timeout_seconds )); then
                    echo "[ERROR] ${key} down for ${timeout_seconds}s"
                    return 2
                fi
            else
                unset 'down_since[$key]'
            fi
        done

        sleep "$check_interval"
    done

    wait "$client_pid"
}

run_one_benchmark() {
    local req_rate="$1"
    local max_tokens="$2"
    local repeat_idx="$3"
    local timestamp="$4"
    local log_file="${LOG_DIR}/bench_rr${req_rate}_mt${MODEL_MT_TAG_FOR_NAME}_rep${repeat_idx}.log"

    local bench_cmd=(
        python3 "$BENCH_SCRIPT"
        --backend "$BENCH_BACKEND"
        --port "$BENCH_PORT"
        --endpoint "$BENCH_ENDPOINT"
        --model "$BENCH_MODEL"
        --dataset-name "$BENCH_DATASET_NAME"
        --dataset-path "$CURRENT_BENCH_DATASET_PATH"
        --save-output "$SAVE_OUTPUT"
        --out-path "$RESULT_DIR"
        --maxtokenscustom "$max_tokens"
        --seed "$VLLM_SEED"
        --max-concurrency "$BENCH_MAX_CONCURRENCY"
        --temperature "$BENCH_TEMPERATURE"
        --top-p "$BENCH_TOP_P"
        --top-k "$BENCH_TOP_K"
        --repetition-penalty "$BENCH_REPETITION_PENALTY"
    )

    local goodput_arr=()
    if [[ -n "$BENCH_GOODPUT" ]]; then
        read -r -a goodput_arr <<< "$BENCH_GOODPUT"
        bench_cmd+=(--goodput "${goodput_arr[@]}")
    fi

    if ! is_true "$USE_TRACE_TIMESTAMPS"; then
        bench_cmd+=(--request-rate "$req_rate")
    fi

    if [[ "$BENCH_DATASET_NAME" != "trace" && "$BENCH_DATASET_NAME" != "lmsyschat" ]]; then
        IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"
        bench_cmd+=(--num-prompts "${NUM_PROMPTS_ARRAY[0]}")
    fi

    if [[ -n "$BENCH_IGNORE_EOS" ]]; then
        bench_cmd+=(--ignore-eos)
    fi
    if is_true "$SAVE_SAMPLE"; then
        bench_cmd+=(--save-sample)
    fi

    echo "Command: ${bench_cmd[*]}"

    setsid env BENCHMARK_INSTANCE_ID="benchmark-${CURRENT_BENCH_DATASET_LABEL}" bash -c "exec ${bench_cmd[*]}" > "$log_file" 2>&1 &
    local client_pid=$!

    local bench_exit_code=0
    if run_benchmark_with_health_guard "$client_pid"; then
        bench_exit_code=0
    else
        bench_exit_code=$?
    fi

    if [[ "$bench_exit_code" -eq 2 ]]; then
        echo "Dataset ${CURRENT_BENCH_DATASET_PATH}: instance down >= ${INSTANCE_DOWN_TIMEOUT_SECONDS}s"
        kill -TERM "$client_pid" >/dev/null 2>&1 || true
        sleep 2
        kill -KILL "$client_pid" >/dev/null 2>&1 || true
        return 2
    fi

    if [[ "$bench_exit_code" -ne 0 ]]; then
        echo "ERROR: benchmark failed with exit code ${bench_exit_code}"
        return "$bench_exit_code"
    fi

    plot_decode_load "$timestamp"
    return 0
}

trap cleanup INT TERM EXIT

main() {
    # Disable -e for the entire batch run so a single dataset failure never
    # aborts the remaining datasets. Individual error paths use explicit checks.
    set +e

    apply_model_dependent_defaults
    resolve_model_mt_tag_for_name
    sync_tpot_between_proxy_and_benchmark
    check_required_files
    check_num_gpus

    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    resolve_dataset_paths

    echo "Found ${#DATASET_PATH_ARRAY[@]} dataset(s)."
    for i in "${!DATASET_PATH_ARRAY[@]}"; do
        echo "  [$((i + 1))/${#DATASET_PATH_ARRAY[@]}] ${DATASET_PATH_ARRAY[$i]}"
    done
    echo "MODEL=${MODEL}"
    echo "VLLM_DTYPE=${VLLM_DTYPE}"
    echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
    echo "MODEL_MT_TAG_FOR_NAME=${MODEL_MT_TAG_FOR_NAME}"
    echo "PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS}"
    echo "DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS}"

    IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$BENCH_REQUEST_RATE_LIST"
    IFS=',' read -ra MAX_TOKENS_ARRAY <<< "$BENCH_MAX_TOKENS_LIST"
    if is_true "$USE_TRACE_TIMESTAMPS"; then
        REQUEST_RATE_ARRAY=("trace")
    fi

    local failed=0
    local total_datasets="${#DATASET_PATH_ARRAY[@]}"

    for dataset_idx in "${!DATASET_PATH_ARRAY[@]}"; do
        local dataset_path="${DATASET_PATH_ARRAY[$dataset_idx]}"
        set_current_dataset "$dataset_path"
        local dataset_should_skip=false

        echo "============================================================"
        echo "Dataset [$((dataset_idx + 1))/${total_datasets}]: ${CURRENT_BENCH_DATASET_PATH}"
        echo "============================================================"

        for req_rate in "${REQUEST_RATE_ARRAY[@]}"; do
            [[ "$dataset_should_skip" == true ]] && break
            for max_tokens in "${MAX_TOKENS_ARRAY[@]}"; do
                [[ "$dataset_should_skip" == true ]] && break
                for repeat_idx in $(seq 1 "$RUN_REPEAT_PER_CONFIG"); do
                    [[ "$dataset_should_skip" == true ]] && break

                    local timestamp
                    timestamp=$(date +%Y%m%d_%H%M%S)

                    setup_directories "$req_rate" "$max_tokens" "$timestamp" "$repeat_idx"
                    dump_config_json "$req_rate" "$max_tokens" "$repeat_idx" "$timestamp"

                    echo "Run: dataset=${CURRENT_BENCH_DATASET_LABEL}, rate=${req_rate}, max_tokens=${max_tokens}, repeat=${repeat_idx}/${RUN_REPEAT_PER_CONFIG}"

                    if ! start_servers "$timestamp"; then
                        echo "ERROR: failed to start servers"
                        failed=$((failed + 1))
                        stop_servers
                        continue
                    fi

                    if run_one_benchmark "$req_rate" "$max_tokens" "$repeat_idx" "$timestamp"; then
                        echo "Benchmark finished successfully"
                    else
                        local code=$?
                        if [[ "$code" -eq 2 ]]; then
                            dataset_should_skip=true
                            failed=$((failed + 1))
                        else
                            failed=$((failed + 1))
                        fi
                    fi

                    stop_servers

                    local all_ports=()
                    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
                    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
                    all_ports+=("${PROXY_PORT}")
                    all_ports+=("${PREFILL_PORT_ARRAY[@]}")
                    all_ports+=("${DECODE_PORT_ARRAY[@]}")
                    for p in "${all_ports[@]}"; do
                        wait_for_port_free "$p" 15 || true
                    done

                    sleep "$SLEEP_BETWEEN_RUNS"
                done
            done
        done
    done

    echo "============================================================"
    echo "Evaluation finished"
    echo "Failed runs: $failed"
    echo "Results root: $(realpath "$BASE_RESULT_DIR")"
    echo "============================================================"
}

main "$@"
