#!/usr/bin/env bash
set -euo pipefail

# ==============================
# User Configurable Parameters
# ==============================

# 数据集相关
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-mysharegpt}
MODEL=${MODEL:-/home/dynamo/share/models/Qwen2.5-7B-Instruct}
TEST_MODEL="qwen"  # 用于端到端测试的模型名称标识
SAVE_SAMPLE="true"
IGNORE="true"  # 是否忽略EOS标记

BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_dataset/sharegpt_preprocessed_prompts_with_output_tokens.csv}

# 输出选项
SAVE_OUTPUT=${SAVE_OUTPUT:-True}

BENCH_MODEL=${BENCH_MODEL:-$MODEL}
BACKEND=${BACKEND:-vllm}
TOKENIZER=${TOKENIZER:-/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct}
# 端口配置
BENCH_PORT=${BENCH_PORT:-8000}
ENDPOINT=${ENDPOINT:-'/v1/completions'}

# 随机种子
VLLM_SEED=${VLLM_SEED:-42}

# 默认参数 (如果文件名解析失败或未启用目录遍历时使用)
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"500"}
BENCH_REQUEST_RATE_LIST=${BENCH_REQUEST_RATE_LIST:-"8,12,18,24,30"}
BENCH_MAX_TOKENS_LIST=${BENCH_MAX_TOKENS_LIST:-"32768"}

MAX_CONCURRENCY=${MAX_CONCURRENCY:-1024}
MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS:-1}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-32768}

# 采样参数
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-50}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

# Goodput / TTFT 测试参数
GOODPUT=${GOODPUT:-ttft:1000 tpot:50}

# 日志与输出目录
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result/experiment_paper/tmp/qwen_baseline_dynamo_atuo_128_2p6d}
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

    echo "----------------------------------------------------------------"
    echo "Starting Benchmark Iteration"
    echo "Params: NP=$np, RR=$rr, MT=$mt"
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

    if [ "$IGNORE" = "true" ]; then
        CMD="$CMD --ignore-eos"
        echo "ignore eos"
    fi
    if [ "$SAVE_SAMPLE" = "true" ]; then
        CMD="$CMD --save-sample"
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
    IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"
    IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$BENCH_REQUEST_RATE_LIST"
    IFS=',' read -ra MAX_TOKENS_ARRAY <<< "$BENCH_MAX_TOKENS_LIST"
    for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
      for req_rate in "${REQUEST_RATE_ARRAY[@]}"; do
        for max_tokens in "${MAX_TOKENS_ARRAY[@]}"; do
            run_benchmark_step "$num_prompts" "$req_rate" "$max_tokens"
    done
    done
    done

echo "All benchmarks completed."