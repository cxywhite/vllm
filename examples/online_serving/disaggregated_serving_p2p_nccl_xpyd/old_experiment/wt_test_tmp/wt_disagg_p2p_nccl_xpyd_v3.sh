# nohup python /root/vllm/benchmarks/benchmark_serving_baseline.py \
#   --backend vllm \
#   --base-url http://127.0.0.1:8007 \
#   --endpoint '/v1/completions' \
#   --model /root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct \
#   --dataset-name custom \
#   --dataset-path /root/predict-schedule/baseline_experiment/pastfuture/dataset/lmsys-50k-filtered \
#   --maxtokenscustom 8192 \
#   --seed 42 \
#   --num-prompts 500 \
#   --max-concurrency 64 \
#   --request-rate 1 \
#   --temperature 0.7 \
#   --top-p 0.8 \
#   --top-k 20 \
#   --repetition-penalty 1.05 \
#   --save-output True \
#   --out-path /root/predict/predict_experiment/dataset_result/benchmarkserving_result \
#   > dataset_result/log/clientnew.log 2>&1 &

#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Disaggregated vLLM startup script (with batchsize sweep)
# Modified: ensure Ctrl+C / SIGTERM cleanly kills servers and frees ports.
# ----------------------------------------------------------------------------

####################
# USER CONFIG (edit these)
####################
# 定义是否使用pastfuture scheduler的变量，可以根据需要设置为"true"或"false"
# export https_proxy=http://172.18.164.110:7890
# export http_proxy=http://172.18.164.110:7890
# all_proxy=socks5://172.18.164.110:7890
USE_PASTFUTURE_SCHEDULER="false"  # 或者 "false"
# USE_ACTIVATION_PREDICTOR="false"  # 或者 "false"
PREDICTOR_SCRIPT="/root/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py"

# Model and general
MODEL=${MODEL:-/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct}   # local path or HF id
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# Logging / output
CONFIG_DIR=${CONFIG_DIR:-./experiment_result/config}
LOG_DIR=${LOG_DIR:-./experiment_result/log}
RESULT_DIR=${RESULT_DIR:-./experiment_result/dataset_result}
mkdir -p "$CONFIG_DIR" "$LOG_DIR"   "$RESULT_DIR"

# Proxy
PROXY_PORT=${PROXY_PORT:-28001}

# Prefill instances: comma-separated lists (GPU ids, ports, kv-ports optional)
PREFILL_GPUS=${PREFILL_GPUS:-6}
PREFILL_PORTS=${PREFILL_PORTS:-22002}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-22010}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}

# Decode instances: comma-separated lists (GPU ids, ports, kv-ports optional)
DECODE_GPUS=${DECODE_GPUS:-7}
DECODE_PORTS=${DECODE_PORTS:-22020}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-22030}
DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.7}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}

# KV transfer template (can be tuned)
KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e1}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

# vLLM options (will be recorded to config JSON)
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}   # 1=true, 0=false
VLLM_SEED=${VLLM_SEED:-42}
VLLM_DTYPE=${VLLM_DTYPE:-float16}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-1024}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}

# Benchmark script and parameters
BENCH_SCRIPT=${BENCH_SCRIPT:-../../../benchmarks/benchmark_serving_baseline.py}
BENCH_PORT=${BENCH_PORT:-22006}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-custom}
# BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/predict-schedule/baseline_experiment/pastfuture/dataset/lmsys-50k-filtered}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/predict-schedule/baseline_experiment/pastfuture/dataset/lmsys-50k-filtered-with-sample}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-100}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
# BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-512}
# BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-1024}
BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.7}
BENCH_TOP_P=${BENCH_TOP_P:-0.8}
BENCH_TOP_K=${BENCH_TOP_K:-20}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.05}
BENCH_GOODPUT=${BENCH_GOODPUT:-tpot:50}
BENCH_IGNORE_EOS=${BENCH_IGNORE_EOS:-ignore_eos}
SAVE_OUTPUT=${SAVE_OUTPUT:-True}
# Batchsize sweep (comma-separated list)
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"128,256,280,300,320,360,400,440,480,512"}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}

# Misc
PROXY_SCRIPT=${PROXY_SCRIPT:-disagg_proxy_p2p_nccl_xpyd.py}

# ---------------------------
# End USER CONFIG
# ---------------------------

cd "$(dirname "${BASH_SOURCE[0]}")"

# Tracks per-run process pids and process group ids
PIDS=()     # individual PIDs for checking / logging
PGIDS=()    # process group IDs (negative kills will use these)

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
    "num_prompts": ${BENCH_NUM_PROMPTS},
    "max_concurrency": ${BENCH_MAX_CONCURRENCY},
    "request_rate": "${BENCH_REQUEST_RATE}",
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

# helper: wait until port is free (no LISTEN) or timeout (secs)
wait_for_port_free() {
    local port=$1
    local timeout=${2:-10}
    local start=$(date +%s)
    while true; do
        if ! ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q .; then
            return 0
        fi
        now=$(date +%s)
        if (( now - start >= timeout )); then
            return 1
        fi
        sleep 0.5
    done
}

# stop_servers: try graceful group-terminate for all PGIDS, then force kill if needed.
# ---------- 替换：stop_servers ----------
stop_servers() {
    if [[ ${#PGIDS[@]} -eq 0 && ${#PIDS[@]} -eq 0 ]]; then
        return
    fi

    echo "Stopping servers: PGIDS = ${PGIDS[*]} (and PIDs = ${PIDS[*]})"

    # 1) send TERM to process groups (negative PGID)
    for pg in "${PGIDS[@]}"; do
        if [[ -n "$pg" ]]; then
            echo " -> TERM to pgid $pg"
            kill -TERM -"$pg" 2>/dev/null || true
        fi
    done

    # 2) wait small grace period
    sleep 3

    # 3) force kill remaining process groups
    for pg in "${PGIDS[@]}"; do
        if pgrep -g "$pg" >/dev/null 2>&1; then
            echo " -> KILL pgid $pg"
            kill -KILL -"$pg" 2>/dev/null || true
        fi
    done

    # 4) explicit kill any remaining tracked PIDs (fallback)
    for pid in "${PIDS[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            echo " -> explicit KILL pid $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    # 5) final fallback: pattern kill
    pkill -9 -f "$PROXY_SCRIPT" >/dev/null 2>&1 || true
    pkill -9 -f vllm >/dev/null 2>&1 || true

    # 6) clear arrays
    PIDS=()
    PGIDS=()
}
# ---------- end stop_servers ----------


# cleanup (trap handler) - will be called on SIGINT/SIGTERM/EXIT
cleanup() {
    echo "Received termination signal. Cleaning up..."
    # Stop any servers launched for this run
    stop_servers

    # Wait for important ports to free (proxy + prefill + decode lists)
    # Build a list of ports to check
    ports_to_check=()
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    ports_to_check+=("${PROXY_PORT}")
    for p in "${PREFILL_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
    for p in "${DECODE_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done

    for port in "${ports_to_check[@]}"; do
        echo "Waiting for port ${port} to be released..."
        if wait_for_port_free "$port" 10; then
            echo "Port ${port} free."
        else
            echo "Port ${port} may still be in use after timeout."
        fi
    done
    echo "rm wt_handle files..."
    rm /root/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_handle/*
    
    echo "Cleanup finished. Exiting."
    exit 0
}

# Wait for server on localhost:port to be ready (HTTP)
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

# start_servers: launches proxy + prefill + decode for this iteration and
# records both PIDs and PGIDs (process group ids).
# ---------- 替换：start_servers ----------
start_servers() {
    PIDS=()
    PGIDS=()

    echo "Launching disaggregated serving components for this run..."
    echo "Logs: ${LOG_DIR}/*.log"

    # Start proxy with setsid + bash -c 'exec ...' so $! is the real process PID
    echo "Starting proxy server on port $PROXY_PORT..."
    setsid bash -c "exec python3 \"$PROXY_SCRIPT\"" &> "${LOG_DIR}/proxy_${timestamp}.log" &
    proxy_pid=$!
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    echo "  proxy pid=$proxy_pid pgid=$proxy_pgid"
    PIDS+=("$proxy_pid")
    PGIDS+=("$proxy_pgid")
    # 
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id=${PREFILL_GPU_ARRAY[$i]}
        # ----- 启动 PredictorWorker -----
        echo "Starting PredictorWorker..."
        PREDICTOR_LOG="${LOG_DIR}/predictor_${timestamp}.log"

        setsid env CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "exec python3 -u \"$PREDICTOR_SCRIPT\"" > "$PREDICTOR_LOG" 2>&1 &
        predictor_pid=$!
        predictor_pgid=$(ps -o pgid= -p "$predictor_pid" | tr -d ' ')
        echo " predictor pid=$predictor_pid pgid=$predictor_pgid (log: $PREDICTOR_LOG)"
        # ----- PredictorWorker 启动完成 -----

        PIDS+=("$predictor_pid")
        PGIDS+=("$predictor_pgid")
    done
    # Launch Prefill servers
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id=${PREFILL_GPU_ARRAY[$i]}
        port=${PREFILL_PORT_ARRAY[$i]:-$((20002 + i))}
        kv_port=${PREFILL_KV_PORT_ARRAY[$i]:-$((21001 + i))}

        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"

        # 构建命令
        CMD="vllm serve \"$MODEL\" \
            --enforce-eager --host 0.0.0.0 --port \"$port\" \
            --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --gpu-memory-utilization ${PREFILL_GPU_MEMORY_UTILIZATION}"
            
        # # 条件添加 --activation_predictor 参数
        # if [ "$USE_ACTIVATION_PREDICTOR" = "true" ]; then
        #     CMD="$CMD --activation-predict"
        # fi
        
        # 添加 KV 转移配置
        CMD="$CMD --kv-transfer-config \
            '{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"${KV_PRODUCER_BUFFER}\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"$port\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"${KV_NCCL_CHANNELS}\"}}'"
        # 启动服务
        setsid env CUDA_VISIBLE_DEVICES="$gpu_id" VLLM_USE_V1=1 bash -c "exec $CMD" \
            > "${LOG_DIR}/prefill${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
        pid=$!
        pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        echo "    prefill pid=$pid pgid=$pgid"
        PIDS+=("$pid")
        PGIDS+=("$pgid")
        # # ----- 启动 PredictorWorker -----
        # echo "Starting PredictorWorker..."
        # PREDICTOR_LOG="${LOG_DIR}/predictor_${timestamp}.log"

        # setsid env CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "exec python3 -u \"$PREDICTOR_SCRIPT\"" > "$PREDICTOR_LOG" 2>&1 &
        # predictor_pid=$!
        # predictor_pgid=$(ps -o pgid= -p "$predictor_pid" | tr -d ' ')
        # echo "predictor pid=$predictor_pid pgid=$predictor_pgid (log: $PREDICTOR_LOG)"
        # # ----- PredictorWorker 启动完成 -----

        # PIDS+=("$predictor_pid")
        # PGIDS+=("$predictor_pgid")
    done
    
    
    

    # Launch Decode servers
    # Decode servers
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id=${DECODE_GPU_ARRAY[$i]}
        port=${DECODE_PORT_ARRAY[$i]:-$((20003 + i))}
        kv_port=${DECODE_KV_PORT_ARRAY[$i]:-$((22001 + i))}

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        # 构建命令
        CMD="vllm serve \"$MODEL\" \
            --enforce-eager --host 0.0.0.0 --port \"$port\" \
            --tensor-parallel-size ${DECODE_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --gpu-memory-utilization ${DECODE_GPU_MEMORY_UTILIZATION} \
            --swap-space 0"

        # 条件添加 --pastfuture-scheduler 参数
        if [ "$USE_PASTFUTURE_SCHEDULER" = "true" ]; then
            CMD="$CMD --pastfuture-scheduler"
        fi
        # 条件添加 --activation_predictor 参数
        # if [ "$USE_ACTIVATION_PREDICTOR" = "true" ]; then
        #     CMD="$CMD --activation-predict"
        # fi
        # 添加 KV 转移配置
        CMD="$CMD --kv-transfer-config \
            '{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"${KV_CONSUMER_BUFFER}\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"$port\",\"send_type\":\"${KV_SEND_TYPE}\",\"nccl_num_channels\":\"${KV_NCCL_CHANNELS}\"}}'"

        # 启动服务
        setsid env VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="$gpu_id" bash -c "exec $CMD" \
            > "${LOG_DIR}/decode${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
        pid=$!
        pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        echo "    decode pid=$pid pgid=$pgid"
        PIDS+=("$pid")
        PGIDS+=("$pgid")
    done

    # Wait for servers to become ready
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            stop_servers
            return 1
        fi
    done
    echo "All servers are up for this run."
    return 0
}
# ---------- end start_servers ----------


# Parse comma-separated lists into arrays (done once)
IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"
IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
IFS=',' read -ra DECODE_KV_PORT_ARRAY <<< "$DECODE_KV_PORTS"

# Set traps for signals (SIGINT from Ctrl+C, SIGTERM, EXIT)
trap cleanup INT TERM EXIT

main() {
    check_required_files
    check_num_gpus

    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    dump_config_json

    # Prepare NUM_PROMPTS array
    IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"

    for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
        timestamp=$(date +%Y%m%d_%H%M%S)
        echo "========================================"
        echo "Run timestamp=${timestamp}  num-prompts=${num_prompts}"
        echo "========================================"

        # start servers for this run
        if ! start_servers; then
            echo "Failed to start servers for num_prompts=${num_prompts}. Check logs in ${LOG_DIR}."
            cleanup
            exit 1
        fi
        echo "Waiting 60 seconds before starting benchmark..."
        sleep 60
        # Launch benchmark in background and wait
        log_file="${LOG_DIR}/benchmark_np${num_prompts}_${timestamp}.log"
        echo "Starting benchmark: num-prompts=${num_prompts}, max-concurrency=${num_prompts}"
        setsid bash -c "exec python3 \"$BENCH_SCRIPT\" \
                --backend vllm --port ${BENCH_PORT} --endpoint '/v1/completions' \
                --model \"${BENCH_MODEL}\" --dataset-name ${BENCH_DATASET_NAME} \
                --dataset-path ${BENCH_DATASET_PATH} \
                --save-output ${SAVE_OUTPUT} --out-path \"${RESULT_DIR}\" \
                --maxtokenscustom ${VLLM_MAX_MODEL_LEN} --seed ${VLLM_SEED} \
                --num-prompts ${num_prompts} --max-concurrency ${num_prompts} \
                --request-rate ${BENCH_REQUEST_RATE} \
                --temperature ${BENCH_TEMPERATURE} \
                --top-p ${BENCH_TOP_P} --top-k ${BENCH_TOP_K} --repetition-penalty ${BENCH_REPETITION_PENALTY} \
                --goodput ${BENCH_GOODPUT}  ${BENCH_IGNORE_EOS:+--ignore-eos}" > "$log_file" 2>&1 &
        client_pid=$!
        client_pgid=$(ps -o pgid= -p "$client_pid" | tr -d ' ')
        echo "Benchmark client started pid=$client_pid pgid=$client_pgid"

        # Wait for benchmark to finish
        wait "$client_pid"
        client_exit_code=$?

        echo "Benchmark completed for num-prompts=${num_prompts} (exit code: ${client_exit_code})"
        echo "Log file: ${log_file}"

        # Verify servers still running (optional)
        for pid in "${PIDS[@]}"; do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                echo "ERROR: a server process (PID $pid) is no longer running. Check logs in ${LOG_DIR}."
                stop_servers
                cleanup
                exit 1
            fi
        done

        # stop servers for this run
        stop_servers

        # ---------- 在这里加入：等待端口释放 ----------
        # 等待 proxy / prefill / decode 的监听端口真正释放，避免下一轮 start bind 失败
        all_ports=("${PROXY_PORT}" "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}")
        for p in "${all_ports[@]}"; do
            echo "Waiting for port $p to be free..."
            if ! wait_for_port_free "$p" 15; then
                echo "Warning: port $p still in use after timeout"
            else
                echo "Port $p is free."
            fi
        done
        # ---------- 等待端口释放结束 ----------

        echo "Waiting ${SLEEP_BETWEEN_RUNS}s before next run..."
        sleep ${SLEEP_BETWEEN_RUNS}
    done

    echo "Benchmark loop finished."
    exit 0
}


main "$@"
