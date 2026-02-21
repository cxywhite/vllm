#!/bin/bash

# ================= 配置区域 =================

# 1. 基础路径配置
DATASET_NAME="custom"
DATASET_ROOT="/root/myshare/predict_project/act_predictor_train/qwen_dataset/qwen_remain_dataset"
MODEL_PATH="/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct"
BENCHMARK_SCRIPT="/root/predict-schedule/vllm/benchmarks/benchmark_serving_predict_dataset.py"
OUTPUT_DIR="./predict_dataset"

# 2. 压测参数
COUNT_PER_RUN=4
REPEAT_M=2
GPU_UTIL=0.9
MAX_MODEL_LEN=32768
SAVE_OUTPUT=true  # 确保变量已定义

# 3. GPU 与 任务分配
declare -A GPU_TASKS
GPU_TASKS[0]="0 100 200 300 400 500 600 700 800 900"
GPU_TASKS[1]="1000 1100 1200 1300 1400 1500 1600 1700 1800 1900"

# ================= 目录初始化 =================

# 创建主输出目录及其子目录
SERVE_LOG_DIR="${OUTPUT_DIR}/serve_log"
BENCH_LOG_DIR="${OUTPUT_DIR}/bench"

mkdir -p "$SERVE_LOG_DIR"
mkdir -p "$BENCH_LOG_DIR"

# 检查脚本是否存在
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "错误: 找不到脚本 $BENCHMARK_SCRIPT"
    exit 1
fi

# ================= 函数定义 =================

run_gpu_worker() {
    local gpu_id=$1
    local task_indices=$2
    local port=$((9000 + gpu_id))
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # 修改 1: Serve Log 路径
    local log_file="${SERVE_LOG_DIR}/gpu${gpu_id}_server_${timestamp}.log"
    
    echo "[GPU $gpu_id] 启动 vLLM 服务，端口: $port..."
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    vllm serve "$MODEL_PATH" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port $port \
            --tensor-parallel-size 1 \
            --seed 42 \
            --dtype bfloat16 \
            --max-model-len 32768 \
            --max-num-batched-tokens 40960 \
            --max-num-seqs 1024 \
            --gpu-memory-utilization $GPU_UTIL \
            --swap-space 8 \
            > "$log_file" 2>&1 &
    
    SERVER_PID=$!
    echo "[GPU $gpu_id] vLLM PID: $SERVER_PID. 等待服务就绪..."

    # 健康检查
    max_retries=20
    ready=1
    for ((i=0; i<max_retries; i++)); do
        # if curl -s http://localhost:$port/v1/completions > /dev/null; then
        #     ready=1
        #     break
        # fi
        sleep 3
    done

    if [ $ready -eq 0 ]; then
        echo "[GPU $gpu_id] 错误: vLLM 服务启动超时，查看日志: $log_file"
        kill $SERVER_PID
        return
    fi

    echo "[GPU $gpu_id] 服务已就绪，开始执行任务队列."

    # 循环执行任务
    for start_idx in $task_indices; do
        local current_ts=$(date +%Y%m%d_%H%M%S)
        echo "[GPU $gpu_id] ===> 开始测试: startidx=$start_idx, count=$COUNT_PER_RUN"
        
        # 修改 2: Bench Log 路径
        local bench_file="${BENCH_LOG_DIR}/gpu${gpu_id}_bench_idx${start_idx}_cnt${COUNT_PER_RUN}_${current_ts}.log"
        
        # 修改 3: 动态命名输出路径 (假设 Python 脚本内部会根据此 prefix 生成 csv/hf 文件)
        # 我们把 startidx 和 count 传入 out_path 参数
        local output_prefix="${OUTPUT_DIR}/result_gpu${gpu_id}_idx${start_idx}_cnt${COUNT_PER_RUN}_${current_ts}"

        python "$BENCHMARK_SCRIPT" \
            --backend vllm \
            --port "$port" \
            --endpoint "/v1/completions" \
            --model "$MODEL_PATH" \
            --dataset-name "$DATASET_NAME" \
            --dataset-path "$DATASET_ROOT" \
            --startidx "$start_idx" \
            --count "$COUNT_PER_RUN" \
            --m "$REPEAT_M" \
            --out-path "$output_prefix" \
            --save-output True \
            --save-sample \
            --maxtokenscustom 10 \
            --seed 42 \
            --max-concurrency 1024 \
            --request-rate 4 \
            > "$bench_file" 2>&1
        
        status=$?
        if [ $status -eq 0 ]; then
            echo "[GPU $gpu_id] startidx=$start_idx 完成。"
        else
            echo "[GPU $gpu_id] startidx=$start_idx 失败 (Exit Code: $status)。"
        fi
    done

    echo "[GPU $gpu_id] 所有任务完成，关闭 vLLM (PID $SERVER_PID)..."
    kill $SERVER_PID
}

# ================= 主循环 =================

echo "开始多 GPU 并行测试..."
echo "任务列表: ${!GPU_TASKS[@]}"

pids=""
for gpu_id in "${!GPU_TASKS[@]}"; do
    tasks="${GPU_TASKS[$gpu_id]}"
    run_gpu_worker "$gpu_id" "$tasks" &
    pids="$pids $!"
done

wait $pids
echo "所有 GPU 任务已全部结束。"