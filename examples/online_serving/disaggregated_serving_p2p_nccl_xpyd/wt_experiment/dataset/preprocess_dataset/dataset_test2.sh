#!/bin/bash
# nohup ./qwen_dataset.sh > ./logs/serve/sh.log 2>&1 &
# =============================================================================
# 配置参数
# =============================================================================
# MODEL_PATH="/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct"   # 已不需要
PROXY_SCRIPT="proxy2.py"
LOG_DIR="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/logs"
PROXY_PORT=9003
# VLLM_PORTS=(10080 10081 10082 10083 10084 10085 10086 10087)   # 已不需要
# VLLM_GPUS=(0 1 2 3 4 5 6 7)                                     # 已不需要

# 路径配置
DATASET_DIR="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/mysharegpt/split"
BENCHMARK_SCRIPT="/root/predict-schedule/vllm/benchmarks/benchmark_serving_dataset.py"
OUTPUT_ROOT="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/mysharegpt/processed_output"

# 【修改点】跳过索引列表：例如 "0 2" 表示跳过 part_0 和 part_2
SKIP_INDICES=""

# =============================================================================
# 初始化与清理
# =============================================================================
ulimit -n 65535
PIDS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR/serve" "$LOG_DIR/benchmark" "$LOG_DIR/proxy"

cleanup() {
    echo ""
    echo "跳过清理，避免影响其他任务。"
    exit 0
}

trap cleanup INT TERM ERR

# =============================================================================
# 1. 只启动 Proxy（如果仍然需要使用 proxy2.py）
# =============================================================================

echo "启动自定义 Proxy..."
proxy_timestamp=$(date +%Y%m%d_%H%M%S)
proxy_log="$LOG_DIR/proxy/proxy_${PROXY_PORT}_${proxy_timestamp}.log"
echo "Proxy log: $proxy_log"

python3 "$PROXY_SCRIPT" > "$proxy_log" 2>&1 &
PIDS+=($!)

# 检查 Proxy 是否能正常响应（最基本连通性）
wait_for_proxy_ready() {
    local port=$1
    local name=$2
    local timeout=180
    local start_time=$(date +%s)
    
    echo "等待 $name (端口 $port) 就绪..."
    while true; do
        if curl -s -m 3 "localhost:${port}/v1/chat/completions" > /dev/null; then
            echo "$name 端口 $port 已响应"
            return 0
        fi

        local now=$(date +%s)
        if (( now - start_time >= timeout )); then
            echo "错误：$name 启动超时，请检查 proxy 日志: $proxy_log"
            return 1
        fi
        
        sleep 3
    done
}

if ! wait_for_proxy_ready "$PROXY_PORT" "Proxy-Server"; then
    echo "Proxy 无法正常工作，请检查 proxy2.py 日志。"
    cleanup
fi

echo "Proxy 已就绪，开始 Benchmark。"
echo "-----------------------------------------------------------------------"

# =============================================================================
# 2. 循环处理数据集文件
# =============================================================================
mkdir -p "$OUTPUT_ROOT"

files=$(ls "$DATASET_DIR"/mysharegpt_preprocessed_prompts_part_*.csv 2>/dev/null | sort -V)

for file_path in $files; do
    file_name=$(basename "$file_path")
    
    index=$(echo "$file_name" | grep -oP 'part_\K\d+')
    
    should_skip=false
    for skip_idx in $SKIP_INDICES; do
        if [ "$index" == "$skip_idx" ]; then
            should_skip=true
            break
        fi
    done

    if [ "$should_skip" = true ]; then
        echo "跳过数据集 (索引 $index): $file_name"
        continue
    fi

    echo "正在测试数据集: $file_name (索引: $index)"
    bench_timestamp=$(date +%Y%m%d_%H%M%S)
    bench_log="$LOG_DIR/benchmark/bench_${index}_${bench_timestamp}.log"
    echo "Benchmark log: $bench_log"
    
    python3 "$BENCHMARK_SCRIPT" \
        --backend openai-chat \
        --model "/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct" \
        --endpoint /v1/chat/completions \
        --host 127.0.0.1 \
        --port "$PROXY_PORT" \
        --dataset-name mysharegpt \
        --dataset-path "$file_path" \
        --max-concurrency 1024 \
        --request-rate 2 \
        --save-output \
        --goodput ttft:1000 tpot:50 \
        --out-path "${OUTPUT_ROOT}/${file_name%.csv}" > "$bench_log" 2>&1

    echo "完成测试: $file_name"
    echo "-----------------------------------------------------------------------"
    sleep 5
done

echo "所有测试任务已完成！"
cleanup