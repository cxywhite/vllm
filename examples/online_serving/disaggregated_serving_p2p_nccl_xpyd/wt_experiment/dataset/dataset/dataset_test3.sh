#!/bin/bash
# nohup ./qwen_dataset.sh > ./logs/serve/sh.log 2>&1 &
# =============================================================================
# 配置参数
# =============================================================================
MODEL_PATH="/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct"   # 已不需要
PROXY_SCRIPT="proxy3.py"
LOG_DIR="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/logs"
PROXY_PORT=9006
VLLM_PORTS=(10080 10081 10082 10083 10084 10085)
VLLM_GPUS=(0 1 2 3 4 5)
MAX_BATCHED_TOKENS=40960
MAX_MODEL_LEN=32768

# 路径配置
DATASET_DIR="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/mysharegpt/split"
BENCHMARK_SCRIPT="/root/predict-schedule/vllm/benchmarks/benchmark_serving_dataset.py"
OUTPUT_ROOT="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/mysharegpt/processed_output"

# 【修改点2】跳过索引列表：例如 "0 2" 表示跳过 part_0 和 part_2
SKIP_INDICES=""

# =============================================================================
# 初始化与清理
# =============================================================================
ulimit -n 65535
PIDS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$LOG_DIR/serve" "$LOG_DIR/benchmark" "$LOG_DIR/proxy"

SERVE_INDEX="0"
files=$(ls "$DATASET_DIR"/mysharegpt_preprocessed_prompts_part_*.csv 2>/dev/null | sort -V)
if [ -n "$files" ]; then
    first_file=$(echo "$files" | head -n 1)
    first_name=$(basename "$first_file")
    first_index=$(echo "$first_name" | grep -oP 'part_\K\d+')
    if [ -n "$first_index" ]; then
        SERVE_INDEX="$first_index"
    fi
fi

cleanup() {
    echo ""
    # echo "清理环境中，正在停止所有进程..."
    # for pid in "${PIDS[@]}"; do
    #     kill -9 "$pid" 2>/dev/null
    # done
    # fuser -k ${PROXY_PORT}/tcp 2>/dev/null
    # for port in "${VLLM_PORTS[@]}"; do
    #     fuser -k ${port}/tcp 2>/dev/null
    # done
    echo "清理完成。"
    exit 0
}

trap cleanup INT TERM ERR

# 【修改点1】增强型就绪检查：不仅检查端口，还发送一个真实的极短请求尝试生成
wait_for_server_ready() {
    local port=$1
    local name=$2
    local timeout=600
    local start_time=$(date +%s)
    
    echo "等待 $name (端口 $port) 彻底就绪并验证推理能力..."
    while true; do
        if curl -s "localhost:${port}/v1/chat/completions" > /dev/null; then
            echo "$name 端口 $port 已响应，进行推理验证..."
            return 0
        fi
        

        # 3. 超时检查
        local now=$(date +%s)
        if (( now - start_time >= timeout )); then
            echo "错误：$name 启动超时或推理失败，请检查服务端日志 "
            return 1
        fi
        
        # 4. 每 5 秒轮询一次，避免刷屏
        sleep 5
    done
}

# =============================================================================
# 1. 启动服务实例
# =============================================================================

# 启动 vLLM 后端
for i in "${!VLLM_GPUS[@]}"; do
    gpu_id=${VLLM_GPUS[$i]}
    port=${VLLM_PORTS[$i]}
    echo "正在 GPU $gpu_id 上启动 vLLM (端口: $port)..."
    serve_timestamp=$(date +%Y%m%d_%H%M%S)
    serve_log="$LOG_DIR/serve/serve_${port}_${SERVE_INDEX}_${serve_timestamp}.log"
    echo "Serve log: $serve_log"
    
    CUDA_VISIBLE_DEVICES=$gpu_id vllm serve "$MODEL_PATH" \
        --tokenizer "$MODEL_PATH" \
        --port "$port" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.9 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
        --host 0.0.0.0 > "$serve_log" 2>&1 &
    PIDS+=($!)
done

# 先确保后端 vLLM 能够正常工作
for port in "${VLLM_PORTS[@]}"; do
    if ! wait_for_server_ready "$port" "vLLM-Backend"; then
        cleanup
    fi
done

# 后端 OK 后再启动 Proxy
echo "所有后端已就绪，启动自定义 Proxy..."
proxy_timestamp=$(date +%Y%m%d_%H%M%S)
proxy_log="$LOG_DIR/proxy/proxy_${PROXY_PORT}_${proxy_timestamp}.log"
echo "Proxy log: $proxy_log"
python3 "$PROXY_SCRIPT" > "$proxy_log" 2>&1 &
PIDS+=($!)

# 检查 Proxy 是否能连通后端
if ! wait_for_server_ready "$PROXY_PORT" "Proxy-Server"; then
    echo "Proxy 无法正常转发请求，请检查 proxy.py 日志。"
    cleanup
fi

echo "所有服务彻底就绪，开始 Benchmark。"
echo "-----------------------------------------------------------------------"

# =============================================================================
# 2. 循环处理数据集文件
# =============================================================================
mkdir -p "$OUTPUT_ROOT"

for file_path in $files; do
    file_name=$(basename "$file_path")
    
    # 【修改点2实现】提取索引：从 "qwen_remain_dataset_part_0.csv" 提取 "0"
    # 逻辑：删除非数字字符，或者根据固定格式提取
    index=$(echo "$file_name" | grep -oP 'part_\K\d+')
    
    # 检查索引是否在跳过列表中
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
        --model "$MODEL_PATH" \
        --endpoint /v1/chat/completions \
        --host 127.0.0.1 \
        --port "$PROXY_PORT" \
        --dataset-name mysharegpt \
        --dataset-path "$file_path" \
        --max-concurrency 1024 \
        --request-rate 9 \
        --save-output \
        --goodput ttft:1000 tpot:50 \
        --out-path "${OUTPUT_ROOT}/${file_name%.csv}" > "$bench_log" 2>&1

    echo "完成测试: $file_name"
    echo "-----------------------------------------------------------------------"
    sleep 5
done

echo "所有测试任务已完成！"
cleanup