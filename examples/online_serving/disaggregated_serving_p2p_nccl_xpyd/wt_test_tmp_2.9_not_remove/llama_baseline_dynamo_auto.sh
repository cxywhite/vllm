#!/usr/bin/env bash
set -euo pipefail

# ==============================
# User Configurable Parameters
# ==============================

# [新功能] 设置包含历史 benchmark 结果的根目录
# 如果设置了这个路径，脚本将遍历该目录下的子目录进行复现测试
BASELINE_ROOT_DIR="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_121_2p6d"

# 数据集相关
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-mysharegpt}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a}

# 输出选项
SAVE_OUTPUT=${SAVE_OUTPUT:-True}
REPRODUCE_BASELINE=${REPRODUCE_BASELINE:-true}

# 模型与后端
MODEL=${MODEL:-/home/dynamo/share/models/Meta-Llama-3-8B-Instruct}
BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BACKEND=${BACKEND:-vllm}
TOKENIZER=${TOKENIZER:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
# 端口配置
BENCH_PORT=${BENCH_PORT:-8000}
ENDPOINT=${ENDPOINT:-'/v1/completions'}

# 随机种子
VLLM_SEED=${VLLM_SEED:-42}

# 默认参数 (如果文件名解析失败或未启用目录遍历时使用)
DEFAULT_NUM_PROMPTS=${NUM_PROMPTS:-1000}
DEFAULT_REQUEST_RATE=${REQUEST_RATE:-4}
DEFAULT_MAX_TOKENS=${MAX_TOKENS:-8192}

MAX_CONCURRENCY=${MAX_CONCURRENCY:-1024}
MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS:-1}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-8192}

# 采样参数
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-50}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

# Goodput / TTFT 测试参数
GOODPUT=${GOODPUT:-ttft:1000 tpot:50}

# 日志与输出目录
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result/experiment_paper/tmp/baseline_dynamo_123_2p6d}
BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_baseline.py}

# 运行间隔 (秒)
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}

# ==============================
# Function: Run Single Benchmark
# ==============================
run_benchmark_step() {
    local np=$1
    local rr=$2
    local mt=$3
    local csv_path=$4
    local dir_name=$5 # 原始目录名，用于日志

    echo "----------------------------------------------------------------"
    echo "Starting Benchmark Iteration"
    echo "Source Dir: $dir_name"
    echo "Params: NP=$np, RR=$rr, MT=$mt"
    echo "CSV: $csv_path"
    echo "----------------------------------------------------------------"

    # 生成本次运行的 Timestamp
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # 构建当前运行的目录结构
    local BENCHMARK_DIR="${BASE_RESULT_DIR}/benchmark_np${np}_rr${rr}_mt${mt}_${TIMESTAMP}"
    local CONFIG_DIR="${BENCHMARK_DIR}/config"
    local LOG_DIR="${BENCHMARK_DIR}/log"
    local RESULT_DIR="${BENCHMARK_DIR}/dataset_result"

    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
    echo "Created directories: $BENCHMARK_DIR"

    local LOG_FILE="${LOG_DIR}/bench_np${np}_rr${rr}_mt${mt}.log"

    # 构建命令
    local CMD="python3 $BENCH_SCRIPT \
        --backend ${BACKEND} \
        --port ${BENCH_PORT} \
        --endpoint ${ENDPOINT} \
        --model ${BENCH_MODEL} \
        --tokenizer ${TOKENIZER} \
        --dataset-name ${BENCH_DATASET_NAME} \
        --dataset-path ${BENCH_DATASET_PATH} \
        --save-output ${SAVE_OUTPUT} \
        --out-path ${RESULT_DIR} \
        --num-prompts ${np} \
        --max-concurrency ${MAX_CONCURRENCY} \
        --request-rate ${rr} \
        --maxtokenscustom ${mt} \
        --seed ${VLLM_SEED} \
        --temperature ${TEMPERATURE} \
        --top-p ${TOP_P} \
        --top-k ${TOP_K} \
        --repetition-penalty ${REPETITION_PENALTY} \
        --goodput ${GOODPUT} \
        --min-prompt-tokens ${MIN_PROMPT_TOKENS} \
        --max-prompt-tokens ${MAX_PROMPT_TOKENS}"

    if [ "$REPRODUCE_BASELINE" = "true" ]; then
        CMD="$CMD --reproduce-baseline"
        CMD="$CMD --baseline-csv-path ${csv_path}"
        CMD="$CMD --ignore-eos"
    fi

    echo "Running command..."
    # 记录命令到日志
    echo "$CMD" > "$LOG_FILE"
    
    # 执行
    # 使用 set +e 暂时允许失败，以免单个测试失败中断整个循环
    set +e
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "Benchmark finished successfully."
    else
        echo "Benchmark failed with exit code $exit_code. Check log: $LOG_FILE"
    fi

    echo "Sleeping for ${SLEEP_BETWEEN_RUNS} seconds..."
    sleep ${SLEEP_BETWEEN_RUNS}
}

# ==============================
# Main Logic: Directory Traversal
# ==============================

if [ -z "${BASELINE_ROOT_DIR:-}" ]; then
    echo "Error: BASELINE_ROOT_DIR is not set. Please specify the root directory containing benchmarks."
    exit 1
fi

if [ ! -d "$BASELINE_ROOT_DIR" ]; then
    echo "Error: Directory $BASELINE_ROOT_DIR does not exist."
    exit 1
fi

echo "Scanning directory: $BASELINE_ROOT_DIR"

# 遍历目录
for dir_path in "$BASELINE_ROOT_DIR"/benchmark_np*_rr*_mt*; do
    if [ ! -d "$dir_path" ]; then
        continue
    fi

    dir_name=$(basename "$dir_path")
    
    # 1. 使用正则解析参数
    # 格式: benchmark_np1000_rr4_mt8192_...
    if [[ $dir_name =~ benchmark_np([0-9]+)_rr([0-9.]+)_mt([0-9]+) ]]; then
        current_np="${BASH_REMATCH[1]}"
        current_rr="${BASH_REMATCH[2]}"
        current_mt="${BASH_REMATCH[3]}"
        
        # 2. 查找对应的 CSV 文件
        # 假设 csv 在 dataset_result 目录下，且名字以 test_results_ 开头
        csv_search_path="$dir_path/dataset_result"
        found_csv=$(find "$csv_search_path" -maxdepth 1 -name "test_results_*.csv" | head -n 1)

        if [ -n "$found_csv" ]; then
            # 3. 执行测试
            run_benchmark_step "$current_np" "$current_rr" "$current_mt" "$found_csv" "$dir_name"
        else
            echo "[WARN] No CSV file found in $csv_search_path. Skipping $dir_name."
        fi
    else
        echo "[WARN] Could not parse parameters from directory name: $dir_name. Skipping."
    fi
done

echo "All benchmarks completed."