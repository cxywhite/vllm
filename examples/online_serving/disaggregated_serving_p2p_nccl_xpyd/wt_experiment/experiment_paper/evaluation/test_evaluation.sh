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
#   llama -> llama model + llama dataset + all configured PD/TPOT combos
#   qwen  -> qwen model + qwen dataset + all configured PD/TPOT combos
#   all   -> llama and qwen both
MODEL_TYPE=${MODEL_TYPE:-all}

LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/.cache/huggingface/hub/Qwen-2.5-7b-Instruct}

# Active model fields are populated at runtime from MODEL_TYPE.
TEST_MODEL=""
MODEL=""
# Relative to this script directory, or absolute path.
DATASET_BASE_DIR=${DATASET_BASE_DIR:-../../dataset/evaluation_dataset}

TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# Base output directory. Per-run subdirectories are created automatically.
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result}

# User-specified conflict resolutions
PROXY_PORT=${PROXY_PORT:-28002}
BENCH_PORT=${BENCH_PORT:-22007}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

# Shared runtime knobs across all PD ratios.
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-32}

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
BENCH_MODEL=${BENCH_MODEL:-}
BENCH_BACKEND=${BENCH_BACKEND:-openai-chat}
BENCH_ENDPOINT=${BENCH_ENDPOINT:-/v1/chat/completions}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-trace}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT_TTFT_MS=${BENCH_GOODPUT_TTFT_MS:-1000}

BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}

SAVE_OUTPUT=${SAVE_OUTPUT:-True}
SAVE_SAMPLE=${SAVE_SAMPLE:-true}
BENCH_IGNORE_EOS=${BENCH_IGNORE_EOS:-ignore-eos}

USE_TRACE_TIMESTAMPS=${USE_TRACE_TIMESTAMPS:-true}
PD_RATIO_LIST=${PD_RATIO_LIST:-"2:6,4:4"}
TPOT_MS_LIST=${TPOT_MS_LIST:-"30,50"}

# PD profile mappings (override these to match your machine layout).
PD_1P3D_PREFILL_GPUS=${PD_1P3D_PREFILL_GPUS:-0}
PD_1P3D_PREFILL_PORTS=${PD_1P3D_PREFILL_PORTS:-22001}
PD_1P3D_PREFILL_KV_PORTS=${PD_1P3D_PREFILL_KV_PORTS:-22011}
PD_1P3D_DECODE_GPUS=${PD_1P3D_DECODE_GPUS:-1,2,3}
PD_1P3D_DECODE_PORTS=${PD_1P3D_DECODE_PORTS:-22020,22021,22022}
PD_1P3D_DECODE_KV_PORTS=${PD_1P3D_DECODE_KV_PORTS:-22032,22033,22034}

PD_2P6D_PREFILL_GPUS=${PD_2P6D_PREFILL_GPUS:-0,1}
PD_2P6D_PREFILL_PORTS=${PD_2P6D_PREFILL_PORTS:-22001,22002}
PD_2P6D_PREFILL_KV_PORTS=${PD_2P6D_PREFILL_KV_PORTS:-22011,22012}
PD_2P6D_DECODE_GPUS=${PD_2P6D_DECODE_GPUS:-2,3,4,5,6,7}
PD_2P6D_DECODE_PORTS=${PD_2P6D_DECODE_PORTS:-22020,22021,22022,22023,22024,22025}
PD_2P6D_DECODE_KV_PORTS=${PD_2P6D_DECODE_KV_PORTS:-22032,22033,22034,22035,22036,22037}

PD_4P4D_PREFILL_GPUS=${PD_4P4D_PREFILL_GPUS:-0,1,2,3}
PD_4P4D_PREFILL_PORTS=${PD_4P4D_PREFILL_PORTS:-22001,22002,22003,22004}
PD_4P4D_PREFILL_KV_PORTS=${PD_4P4D_PREFILL_KV_PORTS:-22011,22012,22013,22014}
PD_4P4D_DECODE_GPUS=${PD_4P4D_DECODE_GPUS:-4,5,6,7}
PD_4P4D_DECODE_PORTS=${PD_4P4D_DECODE_PORTS:-22020,22021,22022,22023}
PD_4P4D_DECODE_KV_PORTS=${PD_4P4D_DECODE_KV_PORTS:-22032,22033,22034,22035}

# Active runtime assignment is selected by configure_pd_ratio().
# Initialize with 1:3 profile so cleanup paths always have valid values.
PREFILL_GPUS="$PD_1P3D_PREFILL_GPUS"
PREFILL_PORTS="$PD_1P3D_PREFILL_PORTS"
PREFILL_KV_PORTS="$PD_1P3D_PREFILL_KV_PORTS"
DECODE_GPUS="$PD_1P3D_DECODE_GPUS"
DECODE_PORTS="$PD_1P3D_DECODE_PORTS"
DECODE_KV_PORTS="$PD_1P3D_DECODE_KV_PORTS"

RUN_REPEAT_PER_CONFIG=${RUN_REPEAT_PER_CONFIG:-1}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}

BENCH_GOODPUT=""
CACULATE_GOODPUT=""
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
CURRENT_BENCH_DATASET_TAG=""

BENCHMARK_DIR=""
CONFIG_DIR=""
LOG_DIR=""
RESULT_DIR=""
MODEL_MT_TAG_FOR_NAME=""
CURRENT_MODEL_NAME=""
CURRENT_DATASET_ROOT=""
CURRENT_PD_RATIO=""
CURRENT_PD_RATIO_TAG=""
CURRENT_TPOT_MS=""
CURRENT_DATASET_SIZE=""
CURRENT_BENCH_MAX_TOKENS=""
CURRENT_BENCH_MODEL=""
RUN_BATCH_TIMESTAMP=${RUN_BATCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
FAILED_CONFIG_LOG=""
SUMMARY_CSV=""

USER_VLLM_MAX_MODEL_LEN_OVERRIDE=${VLLM_MAX_MODEL_LEN:-}
USER_PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-}

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
    type_lc="$(echo "$CURRENT_MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
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

join_by_comma() {
    local result=""
    local item
    for item in "$@"; do
        if [[ -z "$result" ]]; then
            result="$item"
        else
            result+=",$item"
        fi
    done
    printf '%s' "$result"
}

generate_comma_range() {
    local start="$1"
    local count="$2"
    local values=()
    local index
    for ((index = 0; index < count; index++)); do
        values+=("$((start + index))")
    done
    join_by_comma "${values[@]}"
}

generate_port_list() {
    local base_port="$1"
    local count="$2"
    generate_comma_range "$base_port" "$count"
}

get_selected_model_names() {
    case "${MODEL_TYPE,,}" in
        llama)
            printf '%s\n' "llama"
            ;;
        qwen)
            printf '%s\n' "qwen"
            ;;
        all)
            printf '%s\n%s\n' "llama" "qwen"
            ;;
        *)
            echo "ERROR: unsupported MODEL_TYPE=${MODEL_TYPE}, expected llama/qwen/all"
            return 1
            ;;
    esac
}

configure_active_model() {
    local model_name="$1"

    CURRENT_MODEL_NAME="$model_name"
    case "$model_name" in
        llama)
            MODEL="$LLAMA_MODEL_PATH"
            TEST_MODEL="llama"
            CURRENT_DATASET_ROOT="${DATASET_BASE_DIR}/llama"
            ;;
        qwen)
            MODEL="$QWEN_MODEL_PATH"
            TEST_MODEL="qwen"
            CURRENT_DATASET_ROOT="${DATASET_BASE_DIR}/qwen"
            ;;
        *)
            echo "ERROR: unsupported model_name=${model_name}"
            return 1
            ;;
    esac

    VLLM_MAX_MODEL_LEN="$USER_VLLM_MAX_MODEL_LEN_OVERRIDE"
    PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS="$USER_PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE"
    apply_model_dependent_defaults
    resolve_model_mt_tag_for_name

    CURRENT_BENCH_MAX_TOKENS="$MODEL_MT_TAG_FOR_NAME"
    if [[ -n "$BENCH_MODEL" ]]; then
        CURRENT_BENCH_MODEL="$BENCH_MODEL"
    else
        CURRENT_BENCH_MODEL="$MODEL"
    fi
}

configure_pd_ratio() {
    local pd_ratio="$1"
    CURRENT_PD_RATIO="$pd_ratio"
    CURRENT_PD_RATIO_TAG="$(printf '%s' "$pd_ratio" | sed -E 's/^([0-9]+):([0-9]+)$/\1p\2d/')"

    case "$pd_ratio" in
        1:3)
            PREFILL_GPUS="$PD_1P3D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_1P3D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_1P3D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_1P3D_DECODE_GPUS"
            DECODE_PORTS="$PD_1P3D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_1P3D_DECODE_KV_PORTS"
            ;;
        2:6)
            PREFILL_GPUS="$PD_2P6D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_2P6D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_2P6D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_2P6D_DECODE_GPUS"
            DECODE_PORTS="$PD_2P6D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_2P6D_DECODE_KV_PORTS"
            ;;
        4:4)
            PREFILL_GPUS="$PD_4P4D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_4P4D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_4P4D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_4P4D_DECODE_GPUS"
            DECODE_PORTS="$PD_4P4D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_4P4D_DECODE_KV_PORTS"
            ;;
        *)
            echo "ERROR: unsupported PD ratio=${pd_ratio}, expected 1:3 / 2:6 / 4:4"
            return 1
            ;;
    esac
}

configure_tpot() {
    local tpot_ms="$1"
    CURRENT_TPOT_MS="$tpot_ms"
    BENCH_GOODPUT="ttft:${BENCH_GOODPUT_TTFT_MS} tpot:${tpot_ms}"
    sync_tpot_between_proxy_and_benchmark
}

resolve_dataset_paths() {
    local query_dir="$CURRENT_DATASET_ROOT"

    if [[ ! -d "$query_dir" ]]; then
        echo "ERROR: dataset directory not found: $query_dir"
        exit 1
    fi

    mapfile -d '' -t DATASET_PATH_ARRAY < <(find "$query_dir" -type f -name '*.csv' -print0 | sort -z)
    if [[ ${#DATASET_PATH_ARRAY[@]} -eq 0 ]]; then
        echo "ERROR: no CSV datasets found under: $query_dir"
        exit 1
    fi
}

log_failed_config() {
    local reason="$1"
    mkdir -p "$BASE_RESULT_DIR"
    if [[ -z "$FAILED_CONFIG_LOG" ]]; then
        FAILED_CONFIG_LOG="${BASE_RESULT_DIR}/failed_configs_${RUN_BATCH_TIMESTAMP}.log"
    fi
    printf '%s | model=%s | dataset=%s | pd=%s | tpot=%sms | reason=%s\n' \
        "$(date --iso-8601=seconds)" \
        "$CURRENT_MODEL_NAME" \
        "$CURRENT_BENCH_DATASET_PATH" \
        "$CURRENT_PD_RATIO" \
        "$CURRENT_TPOT_MS" \
        "$reason" >> "$FAILED_CONFIG_LOG"
}

csv_escape() {
    local value="${1:-}"
    value="${value//\"/\"\"}"
    printf '"%s"' "$value"
}

init_summary_csv() {
    if [[ -z "$SUMMARY_CSV" ]]; then
        SUMMARY_CSV="${BASE_RESULT_DIR}/summary_${RUN_BATCH_TIMESTAMP}.csv"
    fi
    if [[ ! -f "$SUMMARY_CSV" ]]; then
        printf '%s\n' 'timestamp,status,model,dataset_path,pd_ratio,tpot_ms,repeat_idx,result_dir,failure_reason' > "$SUMMARY_CSV"
    fi
}

append_summary_csv() {
    local status="$1"
    local repeat_idx="$2"
    local result_dir="$3"
    local reason="$4"
    init_summary_csv
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "$(date --iso-8601=seconds)")" \
        "$(csv_escape "$status")" \
        "$(csv_escape "$CURRENT_MODEL_NAME")" \
        "$(csv_escape "$CURRENT_BENCH_DATASET_PATH")" \
        "$(csv_escape "$CURRENT_PD_RATIO")" \
        "$(csv_escape "${CURRENT_TPOT_MS}")" \
        "$(csv_escape "$repeat_idx")" \
        "$(csv_escape "$result_dir")" \
        "$(csv_escape "$reason")" >> "$SUMMARY_CSV"
}

check_selected_gpus_idle() {
    local all_gpu_ids=()
    local gpu_index memory_used gpu_util
    local status_output
    local selected_gpu_lookup="," 

    IFS=',' read -ra _prefill_gpu_array <<< "$PREFILL_GPUS"
    IFS=',' read -ra _decode_gpu_array <<< "$DECODE_GPUS"
    all_gpu_ids=("${_prefill_gpu_array[@]}" "${_decode_gpu_array[@]}")

    for gpu_index in "${all_gpu_ids[@]}"; do
        selected_gpu_lookup+="${gpu_index},"
    done

    status_output=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    while IFS=',' read -r gpu_index memory_used gpu_util; do
        gpu_index="$(echo "$gpu_index" | xargs)"
        memory_used="$(echo "$memory_used" | xargs)"
        gpu_util="$(echo "$gpu_util" | xargs)"

        if [[ "$selected_gpu_lookup" != *",${gpu_index},"* ]]; then
            continue
        fi

        if [[ "$memory_used" != "0" || "$gpu_util" != "0" ]]; then
            echo "GPU busy: gpu=${gpu_index}, memory_used_mib=${memory_used}, util_percent=${gpu_util}"
            log_failed_config "gpu_not_idle(gpu=${gpu_index},memory=${memory_used},util=${gpu_util})"
            return 1
        fi
    done <<< "$status_output"

    return 0
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

count_csv_records() {
    local dataset_path="$1"
    python3 - "$dataset_path" <<'PY'
import csv
import sys

path = sys.argv[1]
try:
    csv.field_size_limit(sys.maxsize)
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = sum(1 for _ in csv.reader(f))
    print(max(rows - 1, 0))
except Exception:
    print("na")
PY
}

set_current_dataset() {
    local dataset_path="$1"
    local dataset_file_name
    local dataset_name_no_ext
    local dataset_name_norm
    local parsed_dataset_size
    local filename_dataset_size
    dataset_file_name="$(basename "$dataset_path")"
    dataset_name_no_ext="${dataset_file_name%.csv}"
    dataset_name_norm="$(echo "$dataset_name_no_ext" | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g')"

    CURRENT_BENCH_DATASET_PATH="$dataset_path"
    CURRENT_BENCH_DATASET_LABEL="$(sanitize_name "$dataset_name_no_ext")"

    case "$dataset_name_norm" in
        *lmsys-chat*|*lmsys*)
            CURRENT_BENCH_DATASET_TAG="${CURRENT_MODEL_NAME}-lmsys-chat"
            ;;
        *mysharegpt*|*sharegpt*)
            CURRENT_BENCH_DATASET_TAG="${CURRENT_MODEL_NAME}-mysharegpt"
            ;;
        *)
            CURRENT_BENCH_DATASET_TAG="${CURRENT_MODEL_NAME}-$(echo "$dataset_name_norm" | sed 's/[^a-z0-9-]/-/g')"
            ;;
    esac

    filename_dataset_size=""
    if [[ "$dataset_file_name" =~ (^|[_-])n([0-9]+)([_-]|\.) ]]; then
        filename_dataset_size="${BASH_REMATCH[2]}"
    fi

    # Prefer real CSV record count (header excluded), not raw line count.
    parsed_dataset_size="$(count_csv_records "$dataset_path")"
    if [[ "$parsed_dataset_size" =~ ^[0-9]+$ ]]; then
        CURRENT_DATASET_SIZE="$parsed_dataset_size"
    elif [[ -n "$filename_dataset_size" ]]; then
        CURRENT_DATASET_SIZE="$filename_dataset_size"
    else
        CURRENT_DATASET_SIZE="na"
    fi
}

setup_directories() {
    local timestamp="$1"
    local repeat_idx="$2"

    BENCHMARK_DIR="${BASE_RESULT_DIR}/${CURRENT_MODEL_NAME}_${CURRENT_BENCH_DATASET_TAG}_n${CURRENT_DATASET_SIZE}_pd${CURRENT_PD_RATIO_TAG}_tpot${CURRENT_TPOT_MS}ms_rep${repeat_idx}_${timestamp}"
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
    if [[ "$num_gpus" -lt 8 ]]; then
        echo "ERROR: need at least 8 GPUs for configured PD ratios 1:3, 2:6, 4:4, found $num_gpus"
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
        local repeat_idx="$1"
        local ts="$2"
    local cfgfile="${CONFIG_DIR}/config_${ts}.json"

    cat > "$cfgfile" <<EOF
{
  "generated_at": "$(date --iso-8601=seconds)",
    "selected_model_type": "${MODEL_TYPE}",
    "current_model_name": "${CURRENT_MODEL_NAME}",
  "model": "${MODEL}",
  "test_model": "${TEST_MODEL}",
  "dataset_path": "${CURRENT_BENCH_DATASET_PATH}",
  "dataset_label": "${CURRENT_BENCH_DATASET_LABEL}",
    "pd_ratio": "${CURRENT_PD_RATIO}",
    "tpot_ms": ${CURRENT_TPOT_MS},
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
        "bench_model": "${CURRENT_BENCH_MODEL}",
    "backend": "${BENCH_BACKEND}",
    "endpoint": "${BENCH_ENDPOINT}",
    "dataset_name": "${BENCH_DATASET_NAME}",
        "request_rate": "trace",
        "maxtokenscustom": ${CURRENT_BENCH_MAX_TOKENS},
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
    local repeat_idx="$1"
    local timestamp="$2"
    local log_file="${LOG_DIR}/bench_trace_pd${CURRENT_PD_RATIO_TAG}_tpot${CURRENT_TPOT_MS}ms_mt${MODEL_MT_TAG_FOR_NAME}_rep${repeat_idx}.log"

    local bench_cmd=(
        python3 "$BENCH_SCRIPT"
        --backend "$BENCH_BACKEND"
        --port "$BENCH_PORT"
        --endpoint "$BENCH_ENDPOINT"
        --model "$CURRENT_BENCH_MODEL"
        --dataset-name "$BENCH_DATASET_NAME"
        --dataset-path "$CURRENT_BENCH_DATASET_PATH"
        --save-output "$SAVE_OUTPUT"
        --out-path "$RESULT_DIR"
        --maxtokenscustom "$CURRENT_BENCH_MAX_TOKENS"
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
        log_failed_config "benchmark_exit_code=${bench_exit_code}"
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

    check_required_files
    check_num_gpus

    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    mkdir -p "$BASE_RESULT_DIR"
    FAILED_CONFIG_LOG="${BASE_RESULT_DIR}/failed_configs_${RUN_BATCH_TIMESTAMP}.log"
    SUMMARY_CSV="${BASE_RESULT_DIR}/summary_${RUN_BATCH_TIMESTAMP}.csv"
    init_summary_csv

    mapfile -t SELECTED_MODEL_ARRAY < <(get_selected_model_names)
    IFS=',' read -ra PD_RATIO_ARRAY <<< "$PD_RATIO_LIST"
    IFS=',' read -ra TPOT_ARRAY <<< "$TPOT_MS_LIST"

    local failed=0
    local total_datasets=0
    local model_name pd_ratio tpot_ms dataset_idx dataset_path repeat_idx timestamp code

    for model_name in "${SELECTED_MODEL_ARRAY[@]}"; do
        if ! configure_active_model "$model_name"; then
            failed=$((failed + 1))
            continue
        fi

        resolve_dataset_paths
        total_datasets="${#DATASET_PATH_ARRAY[@]}"

        echo "============================================================"
        echo "Model=${CURRENT_MODEL_NAME}"
        echo "Dataset root=${CURRENT_DATASET_ROOT}"
        echo "Found ${total_datasets} dataset(s)."
        for dataset_idx in "${!DATASET_PATH_ARRAY[@]}"; do
            echo "  [$((dataset_idx + 1))/${total_datasets}] ${DATASET_PATH_ARRAY[$dataset_idx]}"
        done
        echo "MODEL=${MODEL}"
        echo "VLLM_DTYPE=${VLLM_DTYPE}"
        echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
        echo "MODEL_MT_TAG_FOR_NAME=${MODEL_MT_TAG_FOR_NAME}"
        echo "PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS}"
        echo "DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS}"
        echo "PD_RATIO_LIST=${PD_RATIO_LIST}"
        echo "TPOT_MS_LIST=${TPOT_MS_LIST}"
        echo "============================================================"

        for pd_ratio in "${PD_RATIO_ARRAY[@]}"; do
            pd_ratio="$(echo "$pd_ratio" | xargs)"
            [[ -z "$pd_ratio" ]] && continue
            if ! configure_pd_ratio "$pd_ratio"; then
                failed=$((failed + 1))
                continue
            fi

            for tpot_ms in "${TPOT_ARRAY[@]}"; do
                tpot_ms="$(echo "$tpot_ms" | xargs)"
                [[ -z "$tpot_ms" ]] && continue
                configure_tpot "$tpot_ms"

                for dataset_idx in "${!DATASET_PATH_ARRAY[@]}"; do
                    dataset_path="${DATASET_PATH_ARRAY[$dataset_idx]}"
                    set_current_dataset "$dataset_path"

                    echo "============================================================"
                    echo "Model=${CURRENT_MODEL_NAME} Dataset [$((dataset_idx + 1))/${total_datasets}]: ${CURRENT_BENCH_DATASET_PATH}"
                    echo "PD=${CURRENT_PD_RATIO} TPOT=${CURRENT_TPOT_MS}ms"
                    echo "============================================================"

                    for repeat_idx in $(seq 1 "$RUN_REPEAT_PER_CONFIG"); do
                        timestamp=$(date +%Y%m%d_%H%M%S)

                        if ! check_selected_gpus_idle; then
                            echo "SKIP: GPUs are not idle for model=${CURRENT_MODEL_NAME}, pd=${CURRENT_PD_RATIO}, tpot=${CURRENT_TPOT_MS}ms, dataset=${CURRENT_BENCH_DATASET_PATH}"
                            failed=$((failed + 1))
                            append_summary_csv "skipped_gpu_busy" "$repeat_idx" "" "gpu_not_idle"
                            sleep "$SLEEP_BETWEEN_RUNS"
                            continue
                        fi

                        setup_directories "$timestamp" "$repeat_idx"
                        dump_config_json "$repeat_idx" "$timestamp"

                        echo "Run: model=${CURRENT_MODEL_NAME}, dataset=${CURRENT_BENCH_DATASET_LABEL}, pd=${CURRENT_PD_RATIO}, tpot=${CURRENT_TPOT_MS}ms, repeat=${repeat_idx}/${RUN_REPEAT_PER_CONFIG}"

                        if ! start_servers "$timestamp"; then
                            echo "ERROR: failed to start servers"
                            log_failed_config "start_servers_failed"
                            failed=$((failed + 1))
                            append_summary_csv "start_failed" "$repeat_idx" "$BENCHMARK_DIR" "start_servers_failed"
                            stop_servers
                            continue
                        fi

                        if run_one_benchmark "$repeat_idx" "$timestamp"; then
                            echo "Benchmark finished successfully"
                            append_summary_csv "success" "$repeat_idx" "$BENCHMARK_DIR" ""
                        else
                            code=$?
                            failed=$((failed + 1))
                            if [[ "$code" -eq 2 ]]; then
                                echo "Benchmark stopped due to instance health failure"
                                append_summary_csv "benchmark_failed" "$repeat_idx" "$BENCHMARK_DIR" "instance_health_failure"
                            else
                                append_summary_csv "benchmark_failed" "$repeat_idx" "$BENCHMARK_DIR" "benchmark_exit_code=${code}"
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
    done

    echo "============================================================"
    echo "Evaluation finished"
    echo "Failed runs: $failed"
    echo "Results root: $(realpath "$BASE_RESULT_DIR")"
    echo "Failed config log: $FAILED_CONFIG_LOG"
    echo "Summary CSV: $SUMMARY_CSV"
    echo "============================================================"
}

main "$@"
