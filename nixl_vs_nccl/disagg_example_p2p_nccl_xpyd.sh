#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1P3D: 1 Prefill server + 3 Decode servers (current default)
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
PROXY_PORT=${PROXY_PORT:-30001}
QUART_DEBUG=${QUART_DEBUG:-0}
LOG_NO_COLOR=${LOG_NO_COLOR:-1}

# Default 1P1D configuration for A/B comparison with launch_nixl_disagg.sh
PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-20003}
DECODE_PORTS=${DECODE_PORTS:-20005}

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
BENCH_PORT=${BENCH_PORT:-10001}
BENCH_SEED=${BENCH_SEED:-$(date +%s)}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-1}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-1}
BENCH_RANDOM_INPUT_LENS=${BENCH_RANDOM_INPUT_LENS:-4000}
BENCH_RANDOM_OUTPUT_LENS=${BENCH_RANDOM_OUTPUT_LENS:-512,1024,4000}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-500}
BENCH_BURSTINESS=${BENCH_BURSTINESS:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_REQUEST_RATES=${BENCH_REQUEST_RATES:-1,4}
BENCH_GOODPUT_TTFT_MS=${BENCH_GOODPUT_TTFT_MS:-500}
BENCH_GOODPUT_TPOT_MS=${BENCH_GOODPUT_TPOT_MS:-50}

# Logs
LOG_DIR=${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/logs}
LOG_SUFFIX=${LOG_SUFFIX:-}

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
LOG_SUFFIX_FMT="_${RUNTIME_LOG_SUFFIX}"

mkdir -p "$LOG_DIR"
PROXY_LOG="$LOG_DIR/proxy${LOG_SUFFIX_FMT}.log"
PID_TRACK_FILE=${PID_TRACK_FILE:-$LOG_DIR/pids${LOG_SUFFIX_FMT}.txt}
: > "$PID_TRACK_FILE"

# KV transfer / memory pool configuration
# mem_pool_size_gb controls TensorMemoryPool(max_block_size=int(gb*1024**3))
PREFILL_MEM_POOL_SIZE_GB=${PREFILL_MEM_POOL_SIZE_GB:-16}
DECODE_MEM_POOL_SIZE_GB=${DECODE_MEM_POOL_SIZE_GB:-16}
PREFILL_KV_BUFFER_SIZE=${PREFILL_KV_BUFFER_SIZE:-1e1}
DECODE_KV_BUFFER_SIZE=${DECODE_KV_BUFFER_SIZE:-8e9}

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Prefill Mem Pool (GB): $PREFILL_MEM_POOL_SIZE_GB, KV Buffer: $PREFILL_KV_BUFFER_SIZE"
echo "  Decode Mem Pool (GB): $DECODE_MEM_POOL_SIZE_GB, KV Buffer: $DECODE_KV_BUFFER_SIZE"
echo "  Proxy Port: $PROXY_PORT"
echo "  Benchmark Port: $BENCH_PORT"
echo "  Quart Debug: $QUART_DEBUG"
echo "  Disable Color Logs: $LOG_NO_COLOR"
echo "  Log Suffix: $RUNTIME_LOG_SUFFIX"
echo "  PID Track File: $PID_TRACK_FILE"
echo "  Log Dir: $LOG_DIR"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

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
    local files=("disagg_proxy_p2p_nccl_xpyd.py")
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

cleanup() {
    local exit_code=${1:-0}
    echo "Stopping everything..."
    trap - INT TERM

    local tracked_pids=("${PIDS[@]}")
    if [ -n "$PID_TRACK_FILE" ] && [ -f "$PID_TRACK_FILE" ]; then
        while IFS= read -r pid; do
            if [ -n "$pid" ]; then
                tracked_pids+=("$pid")
            fi
        done < "$PID_TRACK_FILE"
    fi

    for pid in "${tracked_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
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
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
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

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*${LOG_SUFFIX_FMT}.log: Prefill server logs"
    echo "  - decode*${LOG_SUFFIX_FMT}.log: Decode server logs"
    echo "  - proxy${LOG_SUFFIX_FMT}.log: Proxy server log"

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    QUART_DEBUG=$QUART_DEBUG python3 disagg_proxy_p2p_nccl_xpyd.py \
        > "$PROXY_LOG" 2>&1 &
    record_pid "$!"

    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))

        # echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        # CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL \
        # --enforce-eager \
        # --host 127.0.0.1 \  # [wt]
        # --port $port \
        # --tensor-parallel-size 1 \
        # --seed 1024 \
        # --dtype float16 \
        # --max-model-len 8192 \
        # --max-num-batched-tokens 32768 \
        # --max-num-seqs 1024 \
        # --trust-remote-code \
        # --gpu-memory-utilization 0.8 \
        # --kv-transfer-config \
        # "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"$PREFILL_KV_BUFFER_SIZE\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"16\"}}" > prefill$((i+1)).log 2>&1 &  # [wt]
        # PIDS+=($!)
        # [wt] loopback bind for prefill server
        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        local prefill_log="$LOG_DIR/prefill$((i+1))${LOG_SUFFIX_FMT}.log"
        UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=$UCX_TLS CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL \
        --enforce-eager \
        --host 127.0.0.1 \
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
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"http_port\":\"20003\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"$PREFILL_MEM_POOL_SIZE_GB\",\"p2p_hostname\":\"127.0.0.1\"}}" > "$prefill_log" 2>&1 &  # [wt]
        record_pid "$!"
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))

        # # Hard-coded decode example (commented out)
        # CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve $MODEL \
        # --enforce-eager \
        # --host 127.0.0.1 \  # [wt]
        # --port 20005 \
        # --tensor-parallel-size 1 \
        # --seed 1024 \
        # --dtype float16 \
        # --max-model-len 8192 \
        # --max-num-batched-tokens 10000 \
        # --max-num-seqs 1024 \
        # --trust-remote-code \
        # --gpu-memory-utilization 0.8 \
        # --kv-transfer-config \
        # "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"22001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"30001\",\"http_port\":\"20005\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"16\"}}" > decode$((i+1)).log 2>&1 &  # [wt]
        # PIDS+=($!)

        # [wt] loopback bind for decode server
        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        local decode_log="$LOG_DIR/decode$((i+1))${LOG_SUFFIX_FMT}.log"
        UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=$UCX_TLS VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $MODEL \
        --enforce-eager \
        --host 127.0.0.1 \
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
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"$DECODE_KV_BUFFER_SIZE\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"$DECODE_MEM_POOL_SIZE_GB\",\"p2p_hostname\":\"127.0.0.1\"}}" > "$decode_log" 2>&1 &  # [wt]
        record_pid "$!"
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done

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
    local benchmark_ts
    benchmark_ts=$(date +%Y%m%d_%H%M%S)
    local summary_log="$LOG_DIR/benchmark_summary_${benchmark_ts}.log"
    : > "$summary_log"

    for input_len in "${input_lens[@]}"; do
        for output_len in "${output_lens[@]}"; do
            for request_rate in "${request_rates[@]}"; do
                # 如果input_len==1024且output_len<4096则跳过这个组合
                if [ "$input_len" -eq 1024 ] && [ "$output_len" -lt 4096 ]; then
                    continue
                fi
                total_runs=$((total_runs + 1))
                local rate_tag="${request_rate//./p}"
                local combo_tag="in${input_len}_out${output_len}_rr${rate_tag}"
                local combo_log="$LOG_DIR/benchmark_${combo_tag}.log"

                echo ""
                echo "[Benchmark $total_runs] random_input_len=$input_len, random_output_len=$output_len, request_rate=$request_rate"
                echo "Log: $combo_log"

                vllm bench serve \
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
                    2>&1 | tee "$combo_log"

                local bench_exit_code=${PIPESTATUS[0]}
                if [ "$bench_exit_code" -eq 0 ]; then
                    success_runs=$((success_runs + 1))
                else
                    failed_runs=$((failed_runs + 1))
                    final_exit_code=1
                fi

                echo "input_len=$input_len, output_len=$output_len, request_rate=$request_rate, ttft_ms=$BENCH_GOODPUT_TTFT_MS, tpot_ms=$BENCH_GOODPUT_TPOT_MS, exit_code=$bench_exit_code, log=$combo_log" | tee -a "$summary_log"
            done
        done
    done

    echo ""
    echo "Benchmark finished. total=$total_runs success=$success_runs failed=$failed_runs"
    echo "Benchmark summary: $summary_log"
    echo ""

    echo "Benchmarking done. Cleaning up..."
    cleanup "$final_exit_code"
}

main
