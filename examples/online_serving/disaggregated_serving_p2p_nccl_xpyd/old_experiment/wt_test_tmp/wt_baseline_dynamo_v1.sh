#!/usr/bin/env bash
set -euo pipefail

# ==============================
# User Configurable Parameters
# ==============================

# 数据集相关
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-mysharegpt}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a}

# 输出选项
SAVE_OUTPUT=${SAVE_OUTPUT:-True}
REPRODUCE_BASELINE=${REPRODUCE_BASELINE:-true}
REPRODUCE_BASELINE_CSV_PATH=${REPRODUCE_BASELINE_CSV_PATH:-"/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_119/benchmark_np1000_rr20_mt8192_20260119_155336_2p4d/dataset_result/test_results_20260119_160536.csv"}

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

# 请求控制
NUM_PROMPTS=${NUM_PROMPTS:-1000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1024}
REQUEST_RATE=${REQUEST_RATE:-20}

# 输出长度控制
MAX_TOKENS=${MAX_TOKENS:-8192}
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
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result/tmp/baseline_dynamo}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCHMARK_DIR="${BASE_RESULT_DIR}/benchmark_np${NUM_PROMPTS}_rr${REQUEST_RATE}_mt${MAX_TOKENS}_${TIMESTAMP}"
CONFIG_DIR="${BENCHMARK_DIR}/config"
LOG_DIR="${BENCHMARK_DIR}/log"
RESULT_DIR="${BENCHMARK_DIR}/dataset_result"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
echo "Benchmark directories created:"
echo "  Base: $BENCHMARK_DIR"

# 日志文件
LOG_FILE="${LOG_DIR}/bench_np${NUM_PROMPTS}_rr${REQUEST_RATE}_mt${MAX_TOKENS}.log"

# ==============================
# Construct Benchmark Command
# ==============================
BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/vllm/benchmarks/benchmark_serving_baseline.py}
CMD="python3 $BENCH_SCRIPT \
    --backend ${BACKEND} \
    --port ${BENCH_PORT} \
    --endpoint '${ENDPOINT}' \
    --model ${BENCH_MODEL} \
    --tokenizer ${TOKENIZER} \
    --dataset-name ${BENCH_DATASET_NAME} \
    --dataset-path ${BENCH_DATASET_PATH} \
    --save-output ${SAVE_OUTPUT} \
    --out-path ${RESULT_DIR} \
    --num-prompts ${NUM_PROMPTS} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --request-rate ${REQUEST_RATE} \
    --maxtokenscustom ${MAX_TOKENS} \
    --seed ${VLLM_SEED} \
    --temperature ${TEMPERATURE} \
    --top-p ${TOP_P} \
    --top-k ${TOP_K} \
    --repetition-penalty ${REPETITION_PENALTY} \
    --goodput ${GOODPUT} \
    --min-prompt-tokens ${MIN_PROMPT_TOKENS} \
    --max-prompt-tokens ${MAX_PROMPT_TOKENS}"

# ==============================
# Optional Flags
# ==============================

if [ "$REPRODUCE_BASELINE" = "true" ]; then
    CMD="$CMD --reproduce-baseline"
    CMD="$CMD --baseline-csv-path ${REPRODUCE_BASELINE_CSV_PATH}"
    CMD="$CMD --ignore-eos"
fi

# ==============================
# Run Benchmark
# ==============================

echo "Running benchmark with command:"
echo "$CMD"
echo "Logs will be saved to $LOG_FILE"

# 输出同时打印到日志文件
eval "$CMD" 2>&1 | tee "$LOG_FILE"
