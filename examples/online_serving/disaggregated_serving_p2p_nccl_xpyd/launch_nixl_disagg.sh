#!/bin/bash

#=============================================================================
# NixlConnector disaggregated serving 启动脚本
# 架构: 1 Prefill + 1 Decode (1P1D)
# 对齐 disagg_example_p2p_nccl_xpyd.sh 的参数配置
#=============================================================================
# 配置说明：
#   - MODEL: 模型路径，默认使用本地缓存模型
#   - PREFILL_GPU: Prefill 实例 GPU ID（默认 6）
#   - DECODE_GPU: Decode 实例 GPU ID（默认 7）
#   - PROXY_PORT: Proxy 端口（默认 30001，与原脚本对齐）
#
# 示例：
#   bash launch_nixl_disagg.sh
#=============================================================================

set -e

# =============================================================================
# 配置（对齐原 P2pNccl 脚本）
# =============================================================================
# 默认使用本地缓存模型（与原脚本一致）
MODEL=${MODEL:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}

# GPU 配置（与原脚本的 PREFILL_GPUS=6, DECODE_GPUS=7 对齐）
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}

# 端口配置
PREFILL_PORT=${PREFILL_PORT:-20003}
DECODE_PORT=${DECODE_PORT:-20005}
PROXY_PORT=${PROXY_PORT:-30001}

# NIXL Side Channel 端口（避免与 kv_port 冲突）
PREFILL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL_PORT:-5559}
DECODE_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL_PORT:-5659}

# 超时
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# 模型参数（与原脚本对齐）
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
SEED=${SEED:-1024}

# UCX 配置
UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
UCX_TLS=${UCX_TLS:-tcp,cuda_copy}

# Benchmark 配置
BENCH_PORT=${BENCH_PORT:-30001}
BENCH_SEED=${BENCH_SEED:-$(date +%s)}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-256}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-256}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-1000}
BENCH_BURSTINESS=${BENCH_BURSTINESS:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-0.5}

# 日志
LOG_DIR=${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/logs}
LOG_NO_COLOR=${LOG_NO_COLOR:-1}
mkdir -p "$LOG_DIR"

PIDS=()

# =============================================================================
# 前置检查
# =============================================================================
check_num_gpus() {
    local num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "❌ 需要至少 2 块 GPU"
        exit 1
    fi
    echo "✅ 检测到 $num_gpus 块 GPU"
}

check_nixl_installed() {
    if ! python3 -c "from nixl._api import nixl_agent" &> /dev/null; then
        echo "❌ nixl 未安装，请先运行 setup_nixl_env.sh"
        exit 1
    fi
    echo "✅ nixl 已安装"
}

check_proxy_script() {
    # 优先使用远程服务器上的路径
    local proxy_paths=(
        "$(dirname "${BASH_SOURCE[0]}")/toy_proxy_server.py"
        "/root/predict-schedule/baseline_experiment/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/toy_proxy_server.py"
    )
    for path in "${proxy_paths[@]}"; do
        if [ -f "$path" ]; then
            PROXY_SCRIPT_PATH="$path"
            break
        fi
    done
    if [ -z "$PROXY_SCRIPT_PATH" ]; then
        echo "❌ toy_proxy_server.py 未找到"
        echo "   请确认以下路径存在："
        echo "   - $(dirname "${BASH_SOURCE[0]}")/toy_proxy_server.py"
        echo "   - /root/predict-schedule/baseline_experiment/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/toy_proxy_server.py"
        exit 1
    fi
    echo "✅ Proxy 脚本: $PROXY_SCRIPT_PATH"
}

check_model_exists() {
    if [ ! -d "$MODEL" ] && [ ! -f "$MODEL/config.json" ]; then
        echo "⚠️ 模型路径不存在: $MODEL"
        echo "   请设置 MODEL 环境变量指向有效模型"
        # 不退出，允许远程模型下载场景
    else
        echo "✅ 模型路径: $MODEL"
    fi
}

# =============================================================================
# 清理函数
# =============================================================================
cleanup() {
    echo ""
    echo "🛑 停止所有服务..."
    trap - INT TERM
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    pkill -9 -f "toy_proxy_server.py" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    echo "✅ 清理完成"
    exit 0
}

# =============================================================================
# 等待服务就绪
# =============================================================================
wait_for_server() {
    local port=$1
    local name=$2
    local timeout=$TIMEOUT_SECONDS
    local start_time=$(date +%s)

    echo "⏳ 等待 $name 就绪 (端口 $port)..."

    while true; do
        if curl -s "localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "✅ $name 已就绪 (端口 $port)"
            return 0
        fi

        local now=$(date +%s)
        if (( now - start_time >= timeout )); then
            echo "❌ $name 启动超时 (端口 $port)"
            return 1
        fi
        sleep 2
    done
}

# =============================================================================
# 主流程
# =============================================================================
main() {
    if [ "$LOG_NO_COLOR" = "1" ]; then
        export NO_COLOR=1
        export CLICOLOR=0
        export FORCE_COLOR=0
    fi

    echo "============================================"
    echo "  NixlConnector Disaggregated Serving"
    echo "============================================"
    echo ""
    echo "配置（对齐 P2pNccl 原始参数）："
    echo "  模型: $MODEL"
    echo "  Prefill: GPU $PREFILL_GPU, 端口 $PREFILL_PORT, SideChannel $PREFILL_SIDE_CHANNEL_PORT"
    echo "  Decode:  GPU $DECODE_GPU, 端口 $DECODE_PORT, SideChannel $DECODE_SIDE_CHANNEL_PORT"
    echo "  Proxy:   端口 $PROXY_PORT"
    echo "  max_model_len: $MAX_MODEL_LEN"
    echo "  max_num_batched_tokens: $MAX_NUM_BATCHED_TOKENS"
    echo "  max_num_seqs: $MAX_NUM_SEQS"
    echo "  gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
    echo "  日志: $LOG_DIR"
    echo ""

    # 前置检查
    check_num_gpus
    check_nixl_installed
    check_proxy_script
    check_model_exists

    trap cleanup INT
    trap cleanup TERM

    # -----------------------------------------------------------------------------
    # 启动 Proxy 服务器
    # -----------------------------------------------------------------------------
    echo ""
    echo "[1/4] 启动 Proxy 服务器 (端口 $PROXY_PORT)..."
    QUART_DEBUG=0 python3 "$PROXY_SCRIPT_PATH" \
        --port $PROXY_PORT \
        --prefiller-hosts localhost \
        --prefiller-ports $PREFILL_PORT \
        --decoder-hosts localhost \
        --decoder-ports $DECODE_PORT \
        > "$LOG_DIR/proxy.log" 2>&1 &
    PIDS+=($!)
    echo "✅ Proxy 已启动 (PID: ${PIDS[-1]})"

    # -----------------------------------------------------------------------------
    # 启动 Prefill 实例
    # -----------------------------------------------------------------------------
    echo ""
    echo "[2/4] 启动 Prefill 实例 (GPU $PREFILL_GPU)..."

    UCX_NET_DEVICES=$UCX_NET_DEVICES \
    UCX_TLS=$UCX_TLS \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_CHANNEL_PORT \
    VLLM_USE_V1=1 \
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    vllm serve $MODEL \
        --enforce-eager \
        --host 127.0.0.1 \
        --port $PREFILL_PORT \
        --tensor-parallel-size 1 \
        --seed $SEED \
        --dtype bfloat16 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs $MAX_NUM_SEQS \
        --trust-remote-code \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backends":["UCX"]}}' \
        > "$LOG_DIR/prefill.log" 2>&1 &
    PIDS+=($!)
    echo "✅ Prefill 已启动 (PID: ${PIDS[-1]})"

    # -----------------------------------------------------------------------------
    # 启动 Decode 实例
    # -----------------------------------------------------------------------------
    echo ""
    echo "[3/4] 启动 Decode 实例 (GPU $DECODE_GPU)..."

    UCX_NET_DEVICES=$UCX_NET_DEVICES \
    UCX_TLS=$UCX_TLS \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_CHANNEL_PORT \
    VLLM_USE_V1=1 \
    CUDA_VISIBLE_DEVICES=$DECODE_GPU \
    vllm serve $MODEL \
        --enforce-eager \
        --host 127.0.0.1 \
        --port $DECODE_PORT \
        --tensor-parallel-size 1 \
        --seed $SEED \
        --dtype bfloat16 \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens 10000 \
        --max-num-seqs $MAX_NUM_SEQS \
        --trust-remote-code \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backends":["UCX"]}}' \
        > "$LOG_DIR/decode.log" 2>&1 &
    PIDS+=($!)
    echo "✅ Decode 已启动 (PID: ${PIDS[-1]})"

    # -----------------------------------------------------------------------------
    # 等待所有服务就绪
    # -----------------------------------------------------------------------------
    echo ""
    echo "[4/4] 等待服务就绪..."

    if ! wait_for_server $PREFILL_PORT "Prefill"; then
        echo "❌ Prefill 启动失败，查看日志: $LOG_DIR/prefill.log"
        cleanup
    fi

    if ! wait_for_server $DECODE_PORT "Decode"; then
        echo "❌ Decode 启动失败，查看日志: $LOG_DIR/decode.log"
        cleanup
    fi

    if ! wait_for_server $PROXY_PORT "Proxy"; then
        echo "❌ Proxy 启动失败，查看日志: $LOG_DIR/proxy.log"
        cleanup
    fi

    # -----------------------------------------------------------------------------
    # 运行 Benchmark
    # -----------------------------------------------------------------------------
    echo ""
    echo "============================================"
    echo "  ✅ 所有服务已就绪！"
    echo "============================================"
    echo ""
    echo "服务地址："
    echo "  Proxy:   http://localhost:$PROXY_PORT"
    echo "  Prefill: http://localhost:$PREFILL_PORT"
    echo "  Decode:  http://localhost:$DECODE_PORT"
    echo ""
    echo "日志文件："
    echo "  $LOG_DIR/prefill.log"
    echo "  $LOG_DIR/decode.log"
    echo "  $LOG_DIR/proxy.log"
    echo ""

    # Benchmark 参数
    echo "--------------------------------------------"
    echo "Benchmark 配置："
    echo "  模型: $MODEL"
    echo "  端口: $BENCH_PORT"
    echo "  random_input_len: $BENCH_RANDOM_INPUT_LEN"
    echo "  random_output_len: $BENCH_RANDOM_OUTPUT_LEN"
    echo "  num_prompts: $BENCH_NUM_PROMPTS"
    echo "  burstiness: $BENCH_BURSTINESS"
    echo "  request_rate: $BENCH_REQUEST_RATE"
    echo "--------------------------------------------"
    echo ""

    echo "启动 Benchmark..."
    vllm bench serve \
        --host 127.0.0.1 \
        --port $BENCH_PORT \
        --seed $BENCH_SEED \
        --model "$MODEL" \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --dataset-name random \
        --random-input-len $BENCH_RANDOM_INPUT_LEN \
        --random-output-len $BENCH_RANDOM_OUTPUT_LEN \
        --num-prompts $BENCH_NUM_PROMPTS \
        --burstiness $BENCH_BURSTINESS \
        --request-rate $BENCH_REQUEST_RATE \
        --ignore-eos \
        2>&1 | tee "$LOG_DIR/benchmark.log"

    BENCH_EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    echo "Benchmark 结束，退出码: $BENCH_EXIT_CODE"
    echo "Benchmark 日志: $LOG_DIR/benchmark.log"
    echo ""

    echo "清理服务..."
    cleanup
}

main
