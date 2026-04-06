#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Disaggregated vLLM startup script (with batchsize sweep)
# All runtime-configurable parameters are defined in the header below.
# Edit the variables in the "USER CONFIG" section only — the rest of the
# script reads those values and starts the proxy/prefill/decode servers and
# runs the benchmark in a configurable loop over batch sizes.
# ----------------------------------------------------------------------------

####################
# USER CONFIG (edit these)
####################

# Model and general
MODEL=${MODEL:-/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct}   # local path or HF id
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# Logging / output
CONFIG_DIR=${CONFIG_DIR:-./experiment_result/config}
LOG_DIR=${LOG_DIR:-./experiment_result/log}
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

# Proxy
PROXY_PORT=${PROXY_PORT:-20001}

# Prefill instances: comma-separated lists (GPU ids, ports, kv-ports optional)
# Example: PREFILL_GPUS="0,1" PREFILL_PORTS="20002,20003" PREFILL_KV_PORTS="20004,20005"
PREFILL_GPUS=${PREFILL_GPUS:-6}
PREFILL_PORTS=${PREFILL_PORTS:-20002}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-20010}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}  # single value applied to all prefill instances
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}

# Decode instances: comma-separated lists (GPU ids, ports, kv-ports optional)
DECODE_GPUS=${DECODE_GPUS:-7}
DECODE_PORTS=${DECODE_PORTS:-20020}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-20030}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.3}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}

# KV transfer template (can be tuned)
KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e1}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-2e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

# vLLM options (will be recorded to config JSON)
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}   # 1=true, 0=false
VLLM_SEED=${VLLM_SEED:-42}
VLLM_DTYPE=${VLLM_DTYPE:-float16}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-512}

# Benchmark script and parameters
BENCH_SCRIPT=${BENCH_SCRIPT:-../../../benchmarks/benchmark_serving_batchsize.py}
BENCH_PORT=${BENCH_PORT:-20006}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-random}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-100}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-512}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-512}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-1024}
BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.7}
BENCH_TOP_P=${BENCH_TOP_P:-0.8}
BENCH_TOP_K=${BENCH_TOP_K:-20}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.05}
BENCH_GOODPUT=${BENCH_GOODPUT:-tpot:50}
BENCH_IGNORE_EOS=${BENCH_IGNORE_EOS:-ignore_eos}

# Batchsize sweep (comma-separated list)
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"1,4,8,16,32,64,128,256,280,300,320,360,400,440,480,512"}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}

# Misc
# Path to the proxy python script expected in same dir as this script
PROXY_SCRIPT=${PROXY_SCRIPT:-disagg_proxy_p2p_nccl_xpyd.py}

# ---------------------------
# End USER CONFIG
# ---------------------------

# Internal state
PIDS=()

cd "$(dirname "${BASH_SOURCE[0]}")"

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
        echo "nvidia-smi not found in PATH. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill. Found $num_gpus."
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

# Convert a comma-separated string into a JSON array string
array_to_json() {
    local s="$1"
    # empty -> empty array
    if [[ -z "$s" ]]; then
        printf '[]'
        return
    fi
    IFS=',' read -ra _arr <<< "$s"
    printf '['
    local first=1
    for v in "${_arr[@]}"; do
        # trim whitespace
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

# Dump config JSON based on current variables
dump_config_json() {
    now_ts=$(date +%Y%m%d_%H%M%S)
    cfgfile="${CONFIG_DIR}/config_${now_ts}.json"

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
    "send_type": "${KV_SEND_TYPE}"
  },
  "vllm_options": {
    "enforce_eager": $( [[ "$VLLM_ENFORCE_EAGER" -eq 1 ]] && echo true || echo false ),
    "seed": ${VLLM_SEED},
    "dtype": "${VLLM_DTYPE}",
    "max_model_len": ${VLLM_MAX_MODEL_LEN},
    "max_num_batched_tokens": ${VLLM_MAX_NUM_BATCHED_TOKENS},
    "max_num_seqs": ${VLLM_MAX_NUM_SEQS}
  },
  "benchmark": {
    "bench_script": "${BENCH_SCRIPT}",
    "bench_port": ${BENCH_PORT},
    "endpoint": "/v1/completions",
    "dataset_name": "${BENCH_DATASET_NAME}",
    "num_prompts": ${BENCH_NUM_PROMPTS},
    "max_concurrency": ${BENCH_MAX_CONCURRENCY},
    "request_rate": "${BENCH_REQUEST_RATE}",
    "random_input_len": ${BENCH_RANDOM_INPUT_LEN},
    "random_output_len": ${BENCH_RANDOM_OUTPUT_LEN},
    "temperature": ${BENCH_TEMPERATURE},
    "top_p": ${BENCH_TOP_P},
    "top_k": ${BENCH_TOP_K},
    "repetition_penalty": ${BENCH_REPETITION_PENALTY},
    "goodput": "${BENCH_GOODPUT}",
    "ignore_eos": "${BENCH_IGNORE_EOS}",
    "num_prompts_list": $(array_to_json "$NUM_PROMPTS_LIST")
  },
  "notes": "Config generated from environment / header settings"
}
EOF

    echo "Wrote config JSON to: $cfgfile"
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM
    pkill -9 -f "$PROXY_SCRIPT" || true
    kill -- -$$ 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}" > /dev/null 2>&1; then
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
    check_required_files
    check_num_gpus

    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    dump_config_json

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching disaggregated serving components..."
    echo "Logs: ${LOG_DIR}/*.log"

    # Start proxy
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 "$PROXY_SCRIPT" &> "${LOG_DIR}/proxy.log" &
    PIDS+=("$!")

    # Parse comma-separated lists into arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    IFS=',' read -ra DECODE_KV_PORT_ARRAY <<< "$DECODE_KV_PORTS"

    # Launch Prefill servers
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id=${PREFILL_GPU_ARRAY[$i]}
        port=${PREFILL_PORT_ARRAY[$i]:-$((20002 + i))}
        kv_port=${PREFILL_KV_PORT_ARRAY[$i]:-$((21001 + i))}

        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve "$MODEL" \
        --enforce-eager \
        --host 0.0.0.0 \
        --port "$port" \
        --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
        --seed ${VLLM_SEED} \
        --dtype ${VLLM_DTYPE} \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS} \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --gpu-memory-utilization ${PREFILL_GPU_MEMORY_UTILIZATION} \
        --kv-transfer-config \
        "{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"${KV_PRODUCER_BUFFER}\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"$port\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"${KV_NCCL_CHANNELS}\"}}" \
        > "${LOG_DIR}/prefill$((i+1)).log" 2>&1 &
        PIDS+=("$!")
    done

    # Launch Decode servers
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id=${DECODE_GPU_ARRAY[$i]}
        port=${DECODE_PORT_ARRAY[$i]:-$((20003 + i))}
        kv_port=${DECODE_KV_PORT_ARRAY[$i]:-$((22001 + i))}

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve "$MODEL" \
        --enforce-eager \
        --host 0.0.0.0 \
        --port "$port" \
        --tensor-parallel-size ${DECODE_TENSOR_PARALLEL_SIZE} \
        --seed ${VLLM_SEED} \
        --dtype ${VLLM_DTYPE} \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --max-num-batched-tokens ${VLLM_MAX_NUM_BATCHED_TOKENS} \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --gpu-memory-utilization ${DECODE_GPU_MEMORY_UTILIZATION} \
        --swap-space 0 \
        --kv-transfer-config \
        "{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"${KV_CONSUMER_BUFFER}\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"$port\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"${KV_NCCL_CHANNELS}\"}}" \
        > "${LOG_DIR}/decode$((i+1)).log" 2>&1 &
        PIDS+=("$!")
    done

    # Wait for servers
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done

    echo "All servers are up. Starting benchmark loop..."

    # Prepare NUM_PROMPTS array
    IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"

    # Run benchmark loop
    for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="${LOG_DIR}/benchmark_np${num_prompts}_${timestamp}.log"

        echo "========================================"
        echo "Starting benchmark: num-prompts=${num_prompts}, max-concurrency=${num_prompts}"
        echo "Log file: ${log_file}"
        echo "========================================"

        # Launch benchmark in background and wait (keeps behavior similar to earlier script)
        python3 "$BENCH_SCRIPT" \
            --backend vllm \
            --port ${BENCH_PORT} \
            --endpoint '/v1/completions' \
            --model "${BENCH_MODEL}" \
            --dataset-name ${BENCH_DATASET_NAME} \
            --num-prompts ${num_prompts} \
            --max-concurrency ${num_prompts} \
            --request-rate ${BENCH_REQUEST_RATE} \
            --random-input-len ${BENCH_RANDOM_INPUT_LEN} \
            --random-output-len ${BENCH_RANDOM_OUTPUT_LEN} \
            --temperature ${BENCH_TEMPERATURE} \
            --top-p ${BENCH_TOP_P} \
            --top-k ${BENCH_TOP_K} \
            --repetition-penalty ${BENCH_REPETITION_PENALTY} \
            --goodput ${BENCH_GOODPUT} \
            --save-input-requests \
            ${BENCH_IGNORE_EOS:+--ignore-eos} > "$log_file" 2>&1 &

        client_pid=$!
        echo "Benchmark client started with PID: $client_pid"

        # Wait for benchmark to finish
        wait "$client_pid"
        client_exit_code=$?

        echo "Benchmark completed for num-prompts=${num_prompts} (exit code: ${client_exit_code})"
        echo "Log file: ${log_file}"

        # Verify servers still running
        for pid in "${PIDS[@]}"; do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                echo "ERROR: a server process (PID $pid) is no longer running. Check logs in ${LOG_DIR}."
                tail -n 200 "${LOG_DIR}/proxy.log" || true
                cleanup
                exit 1
            fi
        done

        echo "Waiting ${SLEEP_BETWEEN_RUNS}s before next run..."
        sleep ${SLEEP_BETWEEN_RUNS}
    done

    echo "Benchmark loop finished. Cleaning up..."

    cleanup
}

main "$@"