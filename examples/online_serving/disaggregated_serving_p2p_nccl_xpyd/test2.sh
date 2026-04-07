#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1p1d: 1 Prefill server + 3 Decode servers (current default)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

set -e

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30002}
QUART_DEBUG=${QUART_DEBUG:-0}
LOG_NO_COLOR=${LOG_NO_COLOR:-1}

# Default 1P1D configuration for A/B comparison with launch_nixl_disagg.sh
PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-20103}
DECODE_PORTS=${DECODE_PORTS:-20105}

# Model/runtime configuration aligned with launch_nixl_disagg.sh
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
SEED=${SEED:-1024}

# UCX configuration (kept for parity in comparison runs)
UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
UCX_TLS=${UCX_TLS:-tcp,cuda_copy}

# Benchmark configuration aligned with launch_nixl_disagg.sh
# Note: disagg_proxy_p2p_nccl_xpyd.py listens on HTTP port 10001 by design.
BENCH_PORT=${BENCH_PORT:-10002}
BENCH_SEED=${BENCH_SEED:-$(date +%s)}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-1}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-1}
BENCH_RANDOM_INPUT_LENS=${BENCH_RANDOM_INPUT_LENS:-4000,6000}
BENCH_RANDOM_OUTPUT_LENS=${BENCH_RANDOM_OUTPUT_LENS:-50,100,200,500,1000,2000,4000,6000}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-1000}
BENCH_BURSTINESS=${BENCH_BURSTINESS:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_REQUEST_RATES=${BENCH_REQUEST_RATES:-1,2,4,8,100}
BENCH_GOODPUT_TTFT_MS=${BENCH_GOODPUT_TTFT_MS:-500}
BENCH_GOODPUT_TPOT_MS=${BENCH_GOODPUT_TPOT_MS:-50}
# Set to 0 to skip vllm bench's initial single-prompt ready check.
BENCH_READY_CHECK_TIMEOUT_SEC=${BENCH_READY_CHECK_TIMEOUT_SEC:-0}
BENCH_SCRIPT=${BENCH_SCRIPT:-../../../benchmarks/benchmark_serving_baseline.py}

# GPU watchdog: if both prefill and decode groups stay at 0 util too long,
# current combo is marked failed, services are restarted, and next combo runs.
GPU_IDLE_TIMEOUT_SECONDS=${GPU_IDLE_TIMEOUT_SECONDS:-120}
GPU_IDLE_CHECK_INTERVAL_SECONDS=${GPU_IDLE_CHECK_INTERVAL_SECONDS:-30}
ENABLE_GPU_IDLE_WATCHDOG=${ENABLE_GPU_IDLE_WATCHDOG:-1}
# Keep disabled by default to avoid cross-run accidental kills when multiple
# test scripts run concurrently on the same host.
ENABLE_GLOBAL_PKILL_FALLBACK=${ENABLE_GLOBAL_PKILL_FALLBACK:-0}

# Logs (aligned with 5-D style directory layout)
BASE_RESULT_DIR=${BASE_RESULT_DIR:-$(dirname "${BASH_SOURCE[0]}")/wt_experiment/test_runs}
LOG_SUFFIX=${LOG_SUFFIX:-run2_g0g1}

BENCH_INPUT_TAG=$(echo "$BENCH_RANDOM_INPUT_LENS" | sed -E 's/[ ,\/]+/-/g; s/^-+//; s/-+$//; s/-+/-/g; s/\./p/g')
BENCH_OUTPUT_TAG=$(echo "$BENCH_RANDOM_OUTPUT_LENS" | sed -E 's/[ ,\/]+/-/g; s/^-+//; s/-+$//; s/-+/-/g; s/\./p/g')
BENCH_RATE_TAG=$(echo "$BENCH_REQUEST_RATES" | sed -E 's/[ ,\/]+/-/g; s/^-+//; s/-+$//; s/-+/-/g; s/\./p/g')

[ -z "$BENCH_INPUT_TAG" ] && BENCH_INPUT_TAG="na"
[ -z "$BENCH_OUTPUT_TAG" ] && BENCH_OUTPUT_TAG="na"
[ -z "$BENCH_RATE_TAG" ] && BENCH_RATE_TAG="na"

RUNTIME_LOG_SUFFIX="in${BENCH_INPUT_TAG}_out${BENCH_OUTPUT_TAG}_rr${BENCH_RATE_TAG}"
if [ -n "$LOG_SUFFIX" ]; then
    RUNTIME_LOG_SUFFIX="${LOG_SUFFIX}_${RUNTIME_LOG_SUFFIX}"
fi

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
BENCHMARK_DIR=${BENCHMARK_DIR:-${BASE_RESULT_DIR}/benchmark_${RUN_TIMESTAMP}_${RUNTIME_LOG_SUFFIX}_1p1d}
CONFIG_DIR=${CONFIG_DIR:-${BENCHMARK_DIR}/config}
LOG_DIR=${LOG_DIR:-${BENCHMARK_DIR}/log}
RESULT_DIR=${RESULT_DIR:-${BENCHMARK_DIR}/dataset_result}

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"

PROXY_LOG="$LOG_DIR/proxy_${RUN_TIMESTAMP}.log"
PID_TRACK_FILE=${PID_TRACK_FILE:-$LOG_DIR/pids_${RUN_TIMESTAMP}.txt}
RUN_CONFIG_JSON="$CONFIG_DIR/config_${RUN_TIMESTAMP}.json"
: > "$PID_TRACK_FILE"

# KV transfer / memory pool configuration
# mem_pool_size_gb controls TensorMemoryPool(max_block_size=int(gb*1024**3))
PREFILL_MEM_POOL_SIZE_GB=${PREFILL_MEM_POOL_SIZE_GB:-32}
DECODE_MEM_POOL_SIZE_GB=${DECODE_MEM_POOL_SIZE_GB:-32}
PREFILL_KV_BUFFER_SIZE=${PREFILL_KV_BUFFER_SIZE:-1e1}
DECODE_KV_BUFFER_SIZE=${DECODE_KV_BUFFER_SIZE:-8e9}
PREFILL_KV_PORT_BASE=${PREFILL_KV_PORT_BASE:-21101}
DECODE_KV_PORT_BASE=${DECODE_KV_PORT_BASE:-22101}

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Prefill Mem Pool (GB): $PREFILL_MEM_POOL_SIZE_GB, KV Buffer: $PREFILL_KV_BUFFER_SIZE"
echo "  Decode Mem Pool (GB): $DECODE_MEM_POOL_SIZE_GB, KV Buffer: $DECODE_KV_BUFFER_SIZE"
echo "  Prefill KV Port Base: $PREFILL_KV_PORT_BASE"
echo "  Decode KV Port Base: $DECODE_KV_PORT_BASE"
echo "  Proxy Port: $PROXY_PORT"
echo "  Benchmark Port: $BENCH_PORT"
echo "  Benchmark Script: $BENCH_SCRIPT"
echo "  Quart Debug: $QUART_DEBUG"
echo "  Disable Color Logs: $LOG_NO_COLOR"
echo "  Log Suffix: $RUNTIME_LOG_SUFFIX"
echo "  Run Timestamp: $RUN_TIMESTAMP"
echo "  Benchmark Dir: $BENCHMARK_DIR"
echo "  Config Dir: $CONFIG_DIR"
echo "  PID Track File: $PID_TRACK_FILE"
echo "  Log Dir: $LOG_DIR"
echo "  Result Dir: $RESULT_DIR"
echo "  GPU Idle Watchdog: $ENABLE_GPU_IDLE_WATCHDOG"
echo "  GPU Idle Timeout (s): $GPU_IDLE_TIMEOUT_SECONDS"
echo "  GPU Idle Check Interval (s): $GPU_IDLE_CHECK_INTERVAL_SECONDS"
echo "  Global pkill fallback: $ENABLE_GLOBAL_PKILL_FALLBACK"
echo "  Bench Ready Check Timeout (s): $BENCH_READY_CHECK_TIMEOUT_SEC"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()
PROXY_PID=""
CURRENT_BOTTLENECK="unknown"

record_pid() {
    local pid="$1"
    PIDS+=("$pid")
    if [ -n "$PID_TRACK_FILE" ]; then
        echo "$pid" >> "$PID_TRACK_FILE"
    fi
}

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("disagg_proxy_p2p_nccl_xpyd.py" "$BENCH_SCRIPT")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Continue without token (local/cached model only)."
        return 0
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN format looks invalid (should start with hf_)."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

dump_run_config_json() {
    cat > "$RUN_CONFIG_JSON" <<EOF
{
    "generated_at": "$(date --iso-8601=seconds)",
    "model": "$MODEL",
    "seed": "$SEED",
    "proxy_port": "$PROXY_PORT",
    "bench_port": "$BENCH_PORT",
    "prefill_gpus": "$PREFILL_GPUS",
    "decode_gpus": "$DECODE_GPUS",
    "prefill_ports": "$PREFILL_PORTS",
    "decode_ports": "$DECODE_PORTS",
    "max_model_len": "$MAX_MODEL_LEN",
    "max_num_batched_tokens": "$MAX_NUM_BATCHED_TOKENS",
    "max_num_seqs": "$MAX_NUM_SEQS",
    "gpu_memory_utilization": "$GPU_MEMORY_UTILIZATION",
    "prefill_mem_pool_size_gb": "$PREFILL_MEM_POOL_SIZE_GB",
    "decode_mem_pool_size_gb": "$DECODE_MEM_POOL_SIZE_GB",
    "prefill_kv_buffer_size": "$PREFILL_KV_BUFFER_SIZE",
    "decode_kv_buffer_size": "$DECODE_KV_BUFFER_SIZE",
    "bench_random_input_lens": "$BENCH_RANDOM_INPUT_LENS",
    "bench_random_output_lens": "$BENCH_RANDOM_OUTPUT_LENS",
    "bench_request_rates": "$BENCH_REQUEST_RATES",
    "bench_num_prompts": "$BENCH_NUM_PROMPTS",
    "bench_burstiness": "$BENCH_BURSTINESS",
    "bench_ready_check_timeout_sec": "$BENCH_READY_CHECK_TIMEOUT_SEC",
    "bench_goodput_ttft_ms": "$BENCH_GOODPUT_TTFT_MS",
    "bench_goodput_tpot_ms": "$BENCH_GOODPUT_TPOT_MS"
}
EOF
}

is_truthy() {
    local raw="${1:-}"
    local v
    v=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "y" ]]
}

kill_process_tree() {
    local root_pid="$1"
    local signal="${2:-TERM}"
    local children=""
    local child

    if [[ -z "$root_pid" ]] || ! kill -0 "$root_pid" 2>/dev/null; then
        return 0
    fi

    children=$(pgrep -P "$root_pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child" "$signal"
    done

    kill "-$signal" "$root_pid" 2>/dev/null || true
}

stop_all_services() {
    local tracked_pids=("${PIDS[@]}")

    for pid in "${tracked_pids[@]}"; do
        kill_process_tree "$pid" TERM
    done

    local deadline=$((SECONDS + 10))
    while [ $SECONDS -lt $deadline ]; do
        local alive=0
        for pid in "${tracked_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        if [ "$alive" -eq 0 ]; then
            break
        fi
        sleep 1
    done

    for pid in "${tracked_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill_process_tree "$pid" KILL
        fi
    done

    if is_truthy "$ENABLE_GLOBAL_PKILL_FALLBACK"; then
        echo "[WARN] ENABLE_GLOBAL_PKILL_FALLBACK=1, running global pkill cleanup"
        pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py" 2>/dev/null || true
        pkill -9 -f "vllm serve" 2>/dev/null || true
    fi

    PIDS=()
    PROXY_PID=""
}

cleanup() {
    local exit_code=${1:-0}
    echo "Stopping everything..."
    trap - INT TERM
    stop_all_services
    echo "Cleanup complete."
    exit "$exit_code"
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
        if curl -s "localhost:${port}/v1/models" > /dev/null 2>&1; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

wait_for_proxy_http() {
    local timeout_seconds=${1:-30}
    local start_time=$(date +%s)

    while true; do
        if curl -s "http://127.0.0.1:${BENCH_PORT}/" > /dev/null 2>&1; then
            return 0
        fi

        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            return 1
        fi

        sleep 1
    done
}

wait_for_proxy_registration() {
    local timeout_seconds=${1:-45}
    local start_time=$(date +%s)
    local prefill_ports=()
    local decode_ports=()

    IFS=',' read -ra prefill_ports <<< "$PREFILL_PORTS"
    IFS=',' read -ra decode_ports <<< "$DECODE_PORTS"

    while true; do
        local has_prefill=0
        local has_decode=0
        local port

        for port in "${prefill_ports[@]}"; do
            if grep -qE "Add \\[HTTP:[^,]*:${port}," "$PROXY_LOG" 2>/dev/null; then
                has_prefill=1
                break
            fi
        done

        for port in "${decode_ports[@]}"; do
            if grep -qE "Add \\[HTTP:[^,]*:${port}," "$PROXY_LOG" 2>/dev/null; then
                has_decode=1
                break
            fi
        done

        if [[ "$has_prefill" -eq 1 && "$has_decode" -eq 1 ]]; then
            return 0
        fi

        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            return 1
        fi

        sleep 1
    done
}

stop_proxy_server() {
    if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        kill "$PROXY_PID" 2>/dev/null || true
        local deadline=$((SECONDS + 10))
        while [ $SECONDS -lt $deadline ]; do
            if ! kill -0 "$PROXY_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$PROXY_PID" 2>/dev/null; then
            kill -9 "$PROXY_PID" 2>/dev/null || true
        fi
    fi
    PROXY_PID=""
}

start_proxy_server() {
    local tag="$1"
    PROXY_LOG="$LOG_DIR/proxy_${RUN_TIMESTAMP}_${tag}.log"
    local proxy_model_config_path="${MODEL}/config.json"
    local proxy_tpot="tpot:${BENCH_GOODPUT_TPOT_MS}"
    local proxy_dtype="bfloat16"

    echo "Starting proxy server on port $PROXY_PORT (tag=$tag)..."
    echo "Proxy self-check env: MODEL_CONFIG_PATH=$proxy_model_config_path TPOT=$proxy_tpot VLLM_DTYPE=$proxy_dtype"
    PROXY_PORT=$PROXY_PORT BENCH_PORT=$BENCH_PORT QUART_DEBUG=$QUART_DEBUG MODEL_CONFIG_PATH=$proxy_model_config_path TPOT=$proxy_tpot VLLM_DTYPE=$proxy_dtype python3 disagg_proxy_p2p_nccl_xpyd.py \
            > "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!
    record_pid "$PROXY_PID"

    if ! wait_for_proxy_http 30; then
        echo "Failed to start proxy HTTP service, log: $PROXY_LOG"
        return 1
    fi

    if ! wait_for_proxy_registration 45; then
        echo "[WARN] Proxy registration timeout, continue anyway. Log: $PROXY_LOG"
    fi

    return 0
}

restart_proxy_server() {
    local tag="$1"
    stop_proxy_server
    start_proxy_server "$tag"
}

get_gpu_utilization() {
    local gpu_id="$1"
    local util
    util=$(nvidia-smi --id="$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]')
    if [[ "$util" =~ ^[0-9]+$ ]]; then
        echo "$util"
        return 0
    fi
    return 1
}

is_gpu_group_all_zero() {
    local gpu_csv="$1"
    local arr=()
    local gpu_id
    local seen=0
    IFS=',' read -ra arr <<< "$gpu_csv"
    for gpu_id in "${arr[@]}"; do
        gpu_id=$(echo "$gpu_id" | xargs)
        [ -z "$gpu_id" ] && continue
        seen=1
        local util
        if ! util=$(get_gpu_utilization "$gpu_id"); then
            return 1
        fi
        if (( util > 0 )); then
            return 1
        fi
    done
    if (( seen == 0 )); then
        return 1
    fi
    return 0
}

launch_model_servers() {
    local tag="$1"

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((PREFILL_KV_PORT_BASE + i))
        local prefill_log="$LOG_DIR/prefill$((i+1))_${RUN_TIMESTAMP}_${tag}.log"

        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=$UCX_TLS CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size 1 \
        --seed $SEED \
        --dtype bfloat16 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs $MAX_NUM_SEQS \
        --trust-remote-code \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"$PREFILL_KV_BUFFER_SIZE\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"$PREFILL_MEM_POOL_SIZE_GB\",\"p2p_hostname\":\"0.0.0.0\"}}" > "$prefill_log" 2>&1 &
        record_pid "$!"
    done

    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((DECODE_KV_PORT_BASE + i))
        local decode_log="$LOG_DIR/decode$((i+1))_${RUN_TIMESTAMP}_${tag}.log"

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=$UCX_TLS VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $MODEL \
        --enforce-eager \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size 1 \
        --seed $SEED \
        --dtype bfloat16 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens 10000 \
        --max-num-seqs $MAX_NUM_SEQS \
        --trust-remote-code \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"$DECODE_KV_BUFFER_SIZE\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"$DECODE_MEM_POOL_SIZE_GB\",\"p2p_hostname\":\"0.0.0.0\"}}" > "$decode_log" 2>&1 &
        record_pid "$!"
    done

    echo ""
    echo "Waiting for all model servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$port"; then
            echo "Failed to start server on port $port"
            return 1
        fi
    done
    return 0
}

restart_all_services_and_recover() {
    local tag="$1"
    echo "[WATCHDOG] Restarting proxy/prefill/decode before next combo (tag=$tag)"
    stop_all_services
    if ! launch_model_servers "recover_${tag}"; then
        return 1
    fi
    if ! start_proxy_server "recover_${tag}"; then
        return 1
    fi
    return 0
}

run_benchmark_with_watchdog() {
    local client_pid="$1"
    local combo_tag="$2"
    local zero_since=0

    while kill -0 "$client_pid" 2>/dev/null; do
        if is_truthy "$ENABLE_GPU_IDLE_WATCHDOG"; then
            if is_gpu_group_all_zero "$PREFILL_GPUS" && is_gpu_group_all_zero "$DECODE_GPUS"; then
                local now
                now=$(date +%s)
                if (( zero_since == 0 )); then
                    zero_since=$now
                    echo "[WATCHDOG] combo=$combo_tag prefill/decode all gpu util=0, start timer"
                else
                    local elapsed=$((now - zero_since))
                    if (( elapsed >= GPU_IDLE_TIMEOUT_SECONDS )); then
                        echo "[WATCHDOG][TIMEOUT] combo=$combo_tag zero-util elapsed=${elapsed}s >= ${GPU_IDLE_TIMEOUT_SECONDS}s"
                        kill_process_tree "$client_pid" TERM
                        wait "$client_pid" 2>/dev/null || true
                        return 124
                    fi
                fi
            else
                if (( zero_since > 0 )); then
                    echo "[WATCHDOG] combo=$combo_tag gpu util recovered, reset zero timer"
                fi
                zero_since=0
            fi
        fi
        sleep "$GPU_IDLE_CHECK_INTERVAL_SECONDS"
    done

    wait "$client_pid"
    return $?
}

plot_decode_load() {
    local tag="$1"
    local plot_script="$(dirname "${BASH_SOURCE[0]}")/plot_decode_load2.py"
    local plot_log="$LOG_DIR/plot_${RUN_TIMESTAMP}_${tag}.log"

    if [[ ! -f "$plot_script" ]]; then
        echo "[WARN] plot_decode_load2.py not found: $plot_script"
        return 0
    fi

    if [[ ! -f "$PROXY_LOG" ]]; then
        echo "[WARN] Proxy log not found for plotting: $PROXY_LOG"
        return 0
    fi

    python3 "$plot_script" --log_path "$PROXY_LOG" > "$plot_log" 2>&1 || {
        echo "[WARN] Plotting failed, see $plot_log"
    }
}

analyze_proxy_bottleneck() {
    local log_path="$1"
    local combo_tag="$2"
    local parsed
    parsed=$(python3 - "$log_path" <<'PY'
import re
import sys

path = sys.argv[1]
pat = re.compile(r"\[WT\]\[MONITOR\].*?compute=([0-9.]+).*?memory=([0-9.]+).*?capacity=([0-9.]+)")
counts = {"compute": 0, "memory": 0, "capacity": 0}
order = ["compute", "memory", "capacity"]
total = 0
zero_triplet_skipped = 0

try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            c, m_, cap = map(float, m.groups())

            # Do not count periods where all three dimensions are zero.
            if c == 0.0 and m_ == 0.0 and cap == 0.0:
                zero_triplet_skipped += 1
                continue

            vals = {"compute": c, "memory": m_, "capacity": cap}
            top = max(order, key=lambda k: vals[k])
            counts[top] += 1
            total += 1
except FileNotFoundError:
    print("NO_LOG|0|0|0|0")
    raise SystemExit(0)

if total == 0:
    print("NO_VALID_MONITOR_DATA|0|0|0|0")
    raise SystemExit(0)

ratios = {k: counts[k] / total for k in counts}
top_metric = max(order, key=lambda k: ratios[k])
top_ratio = ratios[top_metric]
if top_ratio > 0.5:
    label = f"{top_metric}({top_ratio*100:.1f}%)"
else:
    label = f"NO_DOMINANT(top={top_metric},{top_ratio*100:.1f}%)"

print(f"{label}|{total}|{counts['compute']}|{counts['memory']}|{counts['capacity']}")
PY
)

    local label total cnt_c cnt_m cnt_cap
    IFS='|' read -r label total cnt_c cnt_m cnt_cap <<< "$parsed"
    CURRENT_BOTTLENECK="$label"
    echo "bottleneck combo=$combo_tag result=$label samples=$total compute_top=$cnt_c memory_top=$cnt_m capacity_top=$cnt_cap log=$log_path"
}

main() {
    if [ "$LOG_NO_COLOR" = "1" ]; then
        export NO_COLOR=1
        export CLICOLOR=0
        export FORCE_COLOR=0
    fi

    check_required_files
    check_hf_token
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    dump_run_config_json

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - ${LOG_DIR}/prefill*_$(printf '%s' "$RUN_TIMESTAMP")_*.log: Prefill server logs"
    echo "  - ${LOG_DIR}/decode*_$(printf '%s' "$RUN_TIMESTAMP")_*.log: Decode server logs"
    echo "  - ${LOG_DIR}/proxy_${RUN_TIMESTAMP}_*.log: Proxy logs (one per benchmark config)"
    echo "  - $RUN_CONFIG_JSON: Run config snapshot"

    echo ""
    if ! launch_model_servers "bootstrap"; then
        cleanup 1
    fi

    if ! start_proxy_server "bootstrap"; then
        cleanup 1
    fi

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    local input_lens_raw="${BENCH_RANDOM_INPUT_LENS//,/ }"
    local output_lens_raw="${BENCH_RANDOM_OUTPUT_LENS//,/ }"
    local request_rates_raw="${BENCH_REQUEST_RATES//,/ }"
    local input_lens=()
    local output_lens=()
    local request_rates=()
    read -r -a input_lens <<< "$input_lens_raw"
    read -r -a output_lens <<< "$output_lens_raw"
    read -r -a request_rates <<< "$request_rates_raw"

    if [ "${#input_lens[@]}" -eq 0 ] || [ "${#output_lens[@]}" -eq 0 ] || [ "${#request_rates[@]}" -eq 0 ]; then
        echo "BENCH_RANDOM_INPUT_LENS / BENCH_RANDOM_OUTPUT_LENS / BENCH_REQUEST_RATES is empty."
        cleanup 1
    fi

    echo "Benchmark configuration:"
    echo "  model: $MODEL"
    echo "  port: $BENCH_PORT"
    echo "  random_input_lens: ${input_lens[*]}"
    echo "  random_output_lens: ${output_lens[*]}"
    echo "  request_rates: ${request_rates[*]}"
    echo "  goodput_ttft_ms: $BENCH_GOODPUT_TTFT_MS"
    echo "  goodput_tpot_ms: $BENCH_GOODPUT_TPOT_MS"
    echo "  num_prompts: $BENCH_NUM_PROMPTS"
    echo "  burstiness: $BENCH_BURSTINESS"
    echo ""

    local total_runs=0
    local success_runs=0
    local failed_runs=0
    local final_exit_code=0
    local summary_log="$RESULT_DIR/benchmark_summary_${RUN_TIMESTAMP}.log"
    local -a success_configs=()
    local -a failed_configs=()
    : > "$summary_log"

    for input_len in "${input_lens[@]}"; do
        for output_len in "${output_lens[@]}"; do
            for request_rate in "${request_rates[@]}"; do
                # 如果input_len+output_len>=8192则跳过这个组合
                if [ $((input_len + output_len)) -ge 8192 ]; then
                    echo "Skipping combo: in${input_len}_out${output_len}_rr${rate_tag}"
                    continue
                fi
                # [wt] qps=100 仅在总长度不超过 1000 时测试。
                if [ "$request_rate" = "100" ] && [ $((input_len + output_len)) -gt 1000 ]; then
                    echo "Skipping qps=100 combo: in${input_len}_out${output_len} (sum>1000)"
                    continue
                fi
                total_runs=$((total_runs + 1))
                local rate_tag="${request_rate//./p}"
                local combo_tag="in${input_len}_out${output_len}_rr${rate_tag}"
                local combo_log="$LOG_DIR/bench_${combo_tag}_${RUN_TIMESTAMP}.log"
                local combo_proxy_log

                if ! restart_proxy_server "$combo_tag"; then
                    echo "Failed to restart proxy for combo: $combo_tag"
                    cleanup 1
                fi
                combo_proxy_log="$PROXY_LOG"

                echo ""
                echo "[Benchmark $total_runs] random_input_len=$input_len, random_output_len=$output_len, request_rate=$request_rate"
                echo "Log: $combo_log"
                echo "Proxy Log: $PROXY_LOG"

                python3 "$BENCH_SCRIPT" \
                    --host 127.0.0.1 \
                    --port $BENCH_PORT \
                    --seed $BENCH_SEED \
                    --model "$MODEL" \
                    --backend openai-chat \
                    --endpoint /v1/chat/completions \
                    --dataset-name random \
                    --random-input-len "$input_len" \
                    --random-output-len "$output_len" \
                    --num-prompts $BENCH_NUM_PROMPTS \
                    --burstiness $BENCH_BURSTINESS \
                    --request-rate "$request_rate" \
                    --goodput "ttft:$BENCH_GOODPUT_TTFT_MS" "tpot:$BENCH_GOODPUT_TPOT_MS" \
                    --ignore-eos \
                    > "$combo_log" 2>&1 &

                local client_pid=$!
                local bench_exit_code
                set +e
                run_benchmark_with_watchdog "$client_pid" "$combo_tag"
                bench_exit_code=$?
                set -e

                if (( bench_exit_code == 124 )); then
                    failed_runs=$((failed_runs + 1))
                    final_exit_code=1
                    failed_configs+=("$combo_tag(watchdog_timeout)")
                    analyze_proxy_bottleneck "$combo_proxy_log" "$combo_tag" | tee -a "$summary_log" "$combo_log"
                    echo "input_len=$input_len, output_len=$output_len, request_rate=$request_rate, status=watchdog_timeout, bottleneck=$CURRENT_BOTTLENECK, log=$combo_log, proxy_log=$combo_proxy_log" | tee -a "$summary_log"
                    if ! restart_all_services_and_recover "$combo_tag"; then
                        echo "[ERROR] Failed to recover services after watchdog timeout"
                        cleanup 1
                    fi
                    continue
                fi

                if [ "$bench_exit_code" -eq 0 ]; then
                    success_runs=$((success_runs + 1))
                    success_configs+=("$combo_tag")
                else
                    failed_runs=$((failed_runs + 1))
                    final_exit_code=1
                    failed_configs+=("$combo_tag(exit_$bench_exit_code)")
                fi

                analyze_proxy_bottleneck "$combo_proxy_log" "$combo_tag" | tee -a "$summary_log" "$combo_log"
                echo "input_len=$input_len, output_len=$output_len, request_rate=$request_rate, ttft_ms=$BENCH_GOODPUT_TTFT_MS, tpot_ms=$BENCH_GOODPUT_TPOT_MS, exit_code=$bench_exit_code, bottleneck=$CURRENT_BOTTLENECK, log=$combo_log, proxy_log=$combo_proxy_log" | tee -a "$summary_log"
                plot_decode_load "$combo_tag"
            done
        done
    done

    {
        echo ""
        echo "success_configs_count=${#success_configs[@]}"
        for cfg in "${success_configs[@]}"; do
            echo "success_config=$cfg"
        done
        echo "failed_configs_count=${#failed_configs[@]}"
        for cfg in "${failed_configs[@]}"; do
            echo "failed_config=$cfg"
        done
    } | tee -a "$summary_log"

    echo ""
    echo "Benchmark finished. total=$total_runs success=$success_runs failed=$failed_runs"
    echo "Benchmark summary: $summary_log"
    echo ""

    echo "Benchmarking done. Cleaning up..."
    cleanup "$final_exit_code"
}

main
