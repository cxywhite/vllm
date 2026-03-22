set -euo pipefail

# ---------------------------
# USER CONFIG
# ---------------------------

USE_PASTFUTURE_SCHEDULER="false"
USE_ACTIVATION_PREDICTOR="false"
USE_CUSTOM_PROXY="false"
USE_AIMD_SCHEDULER="false"
TEST_ABLATION_P2D="false"
TEST_OPTIMAL="false"
TEST_MAX_OUTPUTLEN="false"
IGNORE="false"
REPRODUCE_BASELINE="false"
SAVE_SAMPLE="true"
REPRODUCE_BASELINE_CSV_PATH="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_116/benchmark_np4000_rr20_mt3000_20260116_195621_1p4d_test_baseline/dataset_result/test_results_20260116_201101.csv"
TEST_MODEL="llama"

BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-lmsyschat}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/3-Mean_and_sigma/dataset/llama_mean_and_sigma_merged_50_with_stats.csv}
SAVE_OUTPUT=${SAVE_OUTPUT:-True}

VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}

MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./wt_experiment/experiment_paper/tmp/baseline_307/1p3d_llama_lmsyschat_mean_and_sigma}

PROXY_PORT=${PROXY_PORT:-28002}
BENCH_PORT=${BENCH_PORT:-22007}

PREFILL_GPUS=${PREFILL_GPUS:-4}
PREFILL_PORTS=${PREFILL_PORTS:-22001}
PREFILL_KV_PORTS=${PREFILL_KV_PORTS:-22011}
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-8}

DECODE_GPUS=${DECODE_GPUS:-5,6,7}
DECODE_PORTS=${DECODE_PORTS:-22020,22021,22022}
DECODE_KV_PORTS=${DECODE_KV_PORTS:-22032,22033,22034}
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

BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_mean_and_sigma.py}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT=${BENCH_GOODPUT:-ttft:1000 tpot:50}
CACULATE_GOODPUT=${CACULATE_GOODPUT:-tpot:50}

NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"1000"}
BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"4,14,12,10,8,6"}
BENCH_MAX_TOKENS_LIST=${BENCH_MAX_TOKENS_LIST:-"8192"}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}
RUN_REPEAT_PER_CONFIG=${RUN_REPEAT_PER_CONFIG:-3}
PROXY_SCRIPT_OVERRIDE=${PROXY_SCRIPT_OVERRIDE:-}
ONLY_RUN_RR=${ONLY_RUN_RR:-true}

is_true() {
    local v="${1:-}"
    [[ "${v,,}" == "true" ]]
}

update_proxy_script() {
    case "${RUN_METHOD:-mean_pred_sched}" in
        optimal)
            PROXY_SCRIPT="test_disagg_proxy_p2d_mean_and_sigma.py"
            ;;
        rr)
            PROXY_SCRIPT="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py"
            ;;
        *)
            PROXY_SCRIPT="disagg_proxy_p2d_mean_and_sigma.py"
            ;;
    esac
    # if [ -n "$PROXY_SCRIPT_OVERRIDE" ]; then
    #     PROXY_SCRIPT="$PROXY_SCRIPT_OVERRIDE"
    # elif [ "$USE_CUSTOM_PROXY" = "true" ]; then
    #     PROXY_SCRIPT="test_disagg_proxy.py"
    # elif is_true "$TEST_OPTIMAL"; then
    #     PROXY_SCRIPT="test_disagg_proxy_p2d_loadbalance.py"
    # elif [ "$TEST_ABLATION_P2D" = "true" ]; then
    #     PROXY_SCRIPT="test_disagg_proxy_p2d.py"
    # else
    #     PROXY_SCRIPT="disagg_proxy_p2p_nccl_xpyd.py"
    # fi
}

cd "$(dirname "${BASH_SOURCE[0]}")"
PIDS=()
PGIDS=()
LAST_BENCHMARK_DIR=""
LAST_RESULT_DIR=""
CURRENT_MEAN_PRED_SCHED_RESULTS_CSV=""
RUN_METHOD="mean_pred_sched"
MEAN_PRED_SCHED_RESULT_ROOT="${BASE_RESULT_DIR}/mean_pred_sched"
OPTIMAL_RESULT_ROOT="${BASE_RESULT_DIR}/optimal"
RR_RESULT_ROOT="${BASE_RESULT_DIR}/rr"

setup_directories() {
    local num_prompts=$1
    local req_rate=$2
    local max_tokens=$3
    local timestamp=$4
    local dir_suffix=$5
    local repeat_idx=$6
    local run_root_dir

    if [[ "$dir_suffix" == *"optimal"* ]]; then
        run_root_dir="$OPTIMAL_RESULT_ROOT"
    elif [[ "$dir_suffix" == *"mean_pred_sched"* ]]; then
        run_root_dir="$MEAN_PRED_SCHED_RESULT_ROOT"
    else
        run_root_dir="$RR_RESULT_ROOT"
    fi

    mkdir -p "$run_root_dir"

    BENCHMARK_DIR="${run_root_dir}/benchmark_np${num_prompts}_rr${req_rate}_mt${max_tokens}_${timestamp}_rep${repeat_idx}_${dir_suffix}"
    mkdir -p "$BENCHMARK_DIR"

    CONFIG_DIR="${BENCHMARK_DIR}/config"
    LOG_DIR="${BENCHMARK_DIR}/log"
    RESULT_DIR="${BENCHMARK_DIR}/dataset_result"

    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
    echo "Created benchmark directories: $BENCHMARK_DIR"
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
    if ! python3 -c "import $1" >/dev/null 2>&1; then
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
}

find_latest_test_results_csv() {
    local result_dir=$1
    local latest=""
    shopt -s nullglob
    local files=("${result_dir}"/test_results_*.csv)
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
        return 1
    fi
    latest=$(ls -1t "${files[@]}" | head -n 1)
    printf '%s\n' "$latest"
}

find_latest_mean_pred_sched_benchmark_dir() {
    local num_prompts=$1
    local req_rate=$2
    local max_tokens=$3
    local repeat_idx=$4
    local pattern="${MEAN_PRED_SCHED_RESULT_ROOT}/benchmark_np${num_prompts}_rr${req_rate}_mt${max_tokens}_*_rep${repeat_idx}_1p3d_mean_pred_sched"

    shopt -s nullglob
    local dirs=( $pattern )
    shopt -u nullglob

    if [[ ${#dirs[@]} -eq 0 ]]; then
        return 1
    fi

    # Directory names embed timestamps as YYYYMMDD_HHMMSS, so lexical sort gives newest.
    printf '%s\n' "${dirs[@]}" | sort | tail -n 1
}

build_optimal_dataset() {
    local source_dataset=$1
    local mean_pred_sched_csv=$2
    local output_dir=$3
    local ts=$4
    local output_dataset="${output_dir}/optimal_dataset_${ts}.csv"

    python3 - "$source_dataset" "$mean_pred_sched_csv" "$output_dataset" <<'PY'
import sys
import pandas as pd

source_path, mean_pred_sched_path, output_path = sys.argv[1:4]
base_df = pd.read_csv(source_path)
mean_pred_sched_df = pd.read_csv(mean_pred_sched_path)

required_base = {"req_id", "output_tokens"}
required_mean_pred_sched = {"req_id", "output_tokens"}
if not required_base.issubset(base_df.columns):
    missing = sorted(required_base - set(base_df.columns))
    raise ValueError(f"Source dataset missing required columns: {missing}")
if not required_mean_pred_sched.issubset(mean_pred_sched_df.columns):
    missing = sorted(required_mean_pred_sched - set(mean_pred_sched_df.columns))
    raise ValueError(f"mean_pred_sched result CSV missing required columns: {missing}")

mean_pred_sched_map = mean_pred_sched_df[["req_id", "output_tokens"]].dropna(subset=["req_id"]).drop_duplicates(subset=["req_id"], keep="last")
mean_pred_sched_map["req_id"] = pd.to_numeric(mean_pred_sched_map["req_id"], errors="coerce")
base_df["req_id"] = pd.to_numeric(base_df["req_id"], errors="coerce")

merged = base_df.merge(
    mean_pred_sched_map.rename(columns={"output_tokens": "mean_pred_sched_output_tokens"}),
    on="req_id",
    how="left",
)
merged["output_tokens"] = merged["mean_pred_sched_output_tokens"].combine_first(merged["output_tokens"])
merged = merged.drop(columns=["mean_pred_sched_output_tokens"])
merged.to_csv(output_path, index=False)
print(output_path)
PY
}

run_single_benchmark() {
    local num_prompts=$1
    local req_rate=$2
    local max_tokens=$3
    local repeat_idx=$4
    local run_method=$5
    local dataset_path=$6

    RUN_METHOD="$run_method"

    if [[ "$RUN_METHOD" == "optimal" ]]; then
        export TEST_OPTIMAL="true"
        export TEST_MAX_OUTPUTLEN="false"
        DIR_SUFFIX="1p3d_optimal"
    elif [[ "$RUN_METHOD" == "rr" ]]; then
        export TEST_OPTIMAL="false"
        export TEST_MAX_OUTPUTLEN="false"
        DIR_SUFFIX="1p3d_rr"
    else
        export TEST_OPTIMAL="false"
        export TEST_MAX_OUTPUTLEN="true"
        DIR_SUFFIX="1p3d_mean_pred_sched"
    fi

    update_proxy_script
    check_required_files

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)

    setup_directories "$num_prompts" "$req_rate" "$max_tokens" "$timestamp" "$DIR_SUFFIX" "$repeat_idx"

    local effective_dataset_path="$dataset_path"
    if [[ ( "$RUN_METHOD" == "optimal" || "$RUN_METHOD" == "rr" ) && ( "$dataset_path" == "__AUTO_OPTIMAL__" || "$dataset_path" == "__AUTO_FROM_MEAN_PRED_SCHED__" ) ]]; then
        local mean_pred_sched_benchmark_dir
        local mean_pred_sched_result_dir
        local mean_pred_sched_results_csv

        if ! mean_pred_sched_benchmark_dir=$(find_latest_mean_pred_sched_benchmark_dir "$num_prompts" "$req_rate" "$max_tokens" "$repeat_idx"); then
            echo "Cannot find matching mean_pred_sched directory in ${MEAN_PRED_SCHED_RESULT_ROOT} for np=${num_prompts}, rr=${req_rate}, mt=${max_tokens}, rep=${repeat_idx}."
            return 1
        fi

        mean_pred_sched_result_dir="${mean_pred_sched_benchmark_dir}/dataset_result"
        if ! mean_pred_sched_results_csv=$(find_latest_test_results_csv "$mean_pred_sched_result_dir"); then
            echo "No mean_pred_sched test_results_*.csv found in ${mean_pred_sched_result_dir}."
            return 1
        fi

        CURRENT_MEAN_PRED_SCHED_RESULTS_CSV="$mean_pred_sched_results_csv"
        effective_dataset_path=$(build_optimal_dataset "$BENCH_DATASET_PATH" "$mean_pred_sched_results_csv" "$BENCHMARK_DIR" "$timestamp")
        echo "Generated dataset from mean_pred_sched results: ${effective_dataset_path}"
    fi

    echo "========================================"
    echo "Run: METHOD=${RUN_METHOD}, TEST_OPTIMAL=${TEST_OPTIMAL}, Repeat=${repeat_idx}/${RUN_REPEAT_PER_CONFIG}, Prompts=${num_prompts}, Rate=${req_rate}, MaxTokens=${max_tokens}"
    echo "Dir: ${BENCHMARK_DIR}"
    echo "Proxy: ${PROXY_SCRIPT}"
    echo "Dataset: ${effective_dataset_path}"
    echo "IGNORE=${IGNORE} (false means benchmark command will not add --ignore-eos)"
    echo "========================================"

    local old_dataset_path="$BENCH_DATASET_PATH"
    local effective_ignore="$IGNORE"
    BENCH_DATASET_PATH="$effective_dataset_path"
    dump_config_json "$num_prompts" "$req_rate" "$max_tokens" "$timestamp"

    if ! start_servers "$timestamp"; then
        echo "Failed to start servers."
        BENCH_DATASET_PATH="$old_dataset_path"
        return 1
    fi

    local log_file="${LOG_DIR}/bench_np${num_prompts}_rr${req_rate}_mt${max_tokens}_rep${repeat_idx}.log"

    CMD="python3 \"$BENCH_SCRIPT\" \
        --backend openai-chat \
        --port \"${BENCH_PORT}\" \
        --endpoint '/v1/chat/completions' \
        --model \"${BENCH_MODEL}\" \
        --dataset-name ${BENCH_DATASET_NAME} \
        --dataset-path ${effective_dataset_path} \
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
        --test-model ${TEST_MODEL} \
        --goodput ${BENCH_GOODPUT}"

    # Optimal and rr runs always benchmark with --ignore-eos.
    if [[ "$RUN_METHOD" == "optimal" || "$RUN_METHOD" == "rr" ]]; then
        effective_ignore="true"
    fi

    [ "$effective_ignore" = "true" ] && CMD="$CMD --ignore-eos"
    [ "$TEST_ABLATION_P2D" = "true" ] && CMD="$CMD --ablation-p2d"

    if [ "$REPRODUCE_BASELINE" = "true" ]; then
        CMD="$CMD --reproduce-baseline"
        CMD="$CMD --baseline-csv-path ${REPRODUCE_BASELINE_CSV_PATH}"
        CMD="$CMD --ignore-eos"
    fi
    [ "$SAVE_SAMPLE" = "true" ] && CMD="$CMD --save-sample"
    [ "$TEST_MAX_OUTPUTLEN" = "true" ] && CMD="$CMD --test-max-outputlen"
    # [ "$TEST_OPTIMAL" = "true" ] && CMD="$CMD --test-optimal"
    setsid env BENCHMARK_INSTANCE_ID="benchmark-${num_prompts}" bash -c "exec $CMD" > "$log_file" 2>&1 &
    local client_pid=$!
    wait "$client_pid"
    echo "Benchmark finished."

    plot_decode_load "$timestamp"

    for pid in "${PIDS[@]}"; do
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "ERROR: Server died."
            stop_servers
            BENCH_DATASET_PATH="$old_dataset_path"
            return 1
        fi
    done

    stop_servers

    local all_ports=("${PROXY_PORT}")
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    all_ports+=("${PREFILL_PORT_ARRAY[@]}")
    all_ports+=("${DECODE_PORT_ARRAY[@]}")

    for p in "${all_ports[@]}"; do
        wait_for_port_free "$p" 15
    done

    LAST_BENCHMARK_DIR="$BENCHMARK_DIR"
    LAST_RESULT_DIR="$RESULT_DIR"
    BENCH_DATASET_PATH="$old_dataset_path"

    echo "Sleeping ${SLEEP_BETWEEN_RUNS}s..."
    sleep "${SLEEP_BETWEEN_RUNS}"
    return 0
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
        [[ -n "$pg" ]] && kill -TERM -"$pg" 2>/dev/null || true
    done
    sleep 3
    for pg in "${PGIDS[@]}"; do
        pgrep -g "$pg" >/dev/null 2>&1 && kill -KILL -"$pg" 2>/dev/null || true
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
        wait_for_port_free "$port" 10 || echo "Port ${port} stuck?"
    done
    exit 0
}

wait_for_server() {
    local port=$1
    local timeout_seconds=$TIMEOUT_SECONDS
    local start_time=$(date +%s)
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s "localhost:${port}" >/dev/null 2>&1; then return 0; fi
        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then echo "Timeout $port"; return 1; fi
        sleep 1
    done
}

plot_decode_load() {
    local ts=$1
    local proxy_log="${LOG_DIR}/proxy_${ts}.log"
    local plot_script="$(dirname "${BASH_SOURCE[0]}")/plot_decode_load2.py"

    [[ -f "$plot_script" ]] || { echo "[WARN] plot_decode_load2.py not found"; return 0; }
    [[ -f "$proxy_log" ]] || { echo "[WARN] Proxy log not found: $proxy_log"; return 0; }

    python3 "$plot_script" --log_path "$proxy_log" > "${LOG_DIR}/plot_${ts}.log" 2>&1 || {
        echo "[WARN] Plotting failed, see ${LOG_DIR}/plot_${ts}.log"
    }
}

start_servers() {
    PIDS=()
    PGIDS=()
    local timestamp=$1
    echo "Launching servers..."

    setsid env PROXY_PORT="${PROXY_PORT}" BENCH_PORT="${BENCH_PORT}" VLLM_DTYPE="${VLLM_DTYPE}" MODEL_CONFIG_PATH="${MODEL}/config.json" TPOT="${CACULATE_GOODPUT}" bash -c "exec python3 \"$PROXY_SCRIPT\"" &> "${LOG_DIR}/proxy_${timestamp}.log" &
    proxy_pid=$!
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    PIDS+=("$proxy_pid"); PGIDS+=("$proxy_pgid")

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"

    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        gpu_id=${PREFILL_GPU_ARRAY[$i]}
        port=${PREFILL_PORT_ARRAY[$i]:-$((20002 + i))}
        kv_port=${PREFILL_KV_PORT_ARRAY[$i]:-$((21001 + i))}

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

        [ "$USE_ACTIVATION_PREDICTOR" = "true" ] && CMD="$CMD --activation-predict"

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

    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        gpu_id=${DECODE_GPU_ARRAY[$i]}
        port=${DECODE_PORT_ARRAY[$i]:-$((20003 + i))}
        kv_port=${DECODE_KV_PORT_ARRAY[$i]:-$((22001 + i))}

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

        [ "$USE_PASTFUTURE_SCHEDULER" = "true" ] && CMD="$CMD --pastfuture-scheduler"
        [ "$USE_AIMD_SCHEDULER" = "true" ] && CMD="$CMD --aimd-scheduler"
        [ "$TEST_ABLATION_P2D" = "true" ] && CMD="$CMD --test-p2d"
        is_true "$TEST_OPTIMAL" && CMD="$CMD --test-optimal"
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
        if ! wait_for_server "$port"; then
            echo "Failed to start server $port"
            stop_servers
            return 1
        fi
    done
    return 0
}

trap cleanup INT TERM EXIT

main() {
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    mkdir -p "$MEAN_PRED_SCHED_RESULT_ROOT" "$OPTIMAL_RESULT_ROOT" "$RR_RESULT_ROOT"

    IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"
    IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$BENCH_REQUEST_RATE_LIST"
    IFS=',' read -ra MAX_TOKENS_ARRAY <<< "$BENCH_MAX_TOKENS_LIST"

    for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
        for req_rate in "${REQUEST_RATE_ARRAY[@]}"; do
            for max_tokens in "${MAX_TOKENS_ARRAY[@]}"; do
                if [[ "$max_tokens" != "8192" ]]; then
                    echo "Skipping max_tokens=${max_tokens}. This llama script requires mt=8192."
                    continue
                fi

                for repeat_idx in $(seq 1 "$RUN_REPEAT_PER_CONFIG"); do
                    CURRENT_MEAN_PRED_SCHED_RESULTS_CSV=""

                    if is_true "$ONLY_RUN_RR"; then
                        echo "ONLY_RUN_RR=true, skipping mean_pred_sched and optimal."
                    else
                        if ! run_single_benchmark "$num_prompts" "$req_rate" "$max_tokens" "$repeat_idx" "mean_pred_sched" "$BENCH_DATASET_PATH"; then
                            echo "mean_pred_sched benchmark failed."
                            cleanup
                            exit 1
                        fi

                        if ! run_single_benchmark "$num_prompts" "$req_rate" "$max_tokens" "$repeat_idx" "optimal" "__AUTO_FROM_MEAN_PRED_SCHED__"; then
                            echo "Optimal benchmark failed."
                            cleanup
                            exit 1
                        fi
                    fi

                    if ! run_single_benchmark "$num_prompts" "$req_rate" "$max_tokens" "$repeat_idx" "rr" "__AUTO_FROM_MEAN_PRED_SCHED__"; then
                        echo "RR benchmark failed."
                        cleanup
                        exit 1
                    fi
                done
            done
        done
    done

    echo "All benchmarks finished."
    exit 0
}

main "$@"