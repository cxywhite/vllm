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
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-4000}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-128}
# 支持多组组合测试（逗号或空格分隔），默认兼容单值变量
BENCH_RANDOM_INPUT_LENS=${BENCH_RANDOM_INPUT_LENS:-4000}
BENCH_RANDOM_OUTPUT_LENS=${BENCH_RANDOM_OUTPUT_LENS:-1024,4000}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-500}
BENCH_BURSTINESS=${BENCH_BURSTINESS:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-inf}
BENCH_REQUEST_RATES=${BENCH_REQUEST_RATES:-1,4,8}
BENCH_GOODPUT_TTFT_MS=${BENCH_GOODPUT_TTFT_MS:-500}
BENCH_GOODPUT_TPOT_MS=${BENCH_GOODPUT_TPOT_MS:-50}

# 日志
LOG_DIR=${LOG_DIR:-$(dirname "${BASH_SOURCE[0]}")/logs_nixl_gpu_0.8}
LOG_NO_COLOR=${LOG_NO_COLOR:-1}
LOG_SUFFIX=${LOG_SUFFIX:-}

mkdir -p "$LOG_DIR"
PREFILL_LOG=""
DECODE_LOG=""
PROXY_LOG=""
PID_TRACK_FILE=""

PIDS=()

normalize_tag_value() {
    local value="$1"
    value=$(echo "$value" | sed -E 's/[ ,\/]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')
    value=${value//./p}
    if [ -z "$value" ]; then
        value="na"
    fi
    echo "$value"
}

record_pid() {
    local pid="$1"
    PIDS+=("$pid")
    if [ -n "$PID_TRACK_FILE" ]; then
        echo "$pid" >> "$PID_TRACK_FILE"
    fi
}

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
    local exit_code=${1:-0}
    echo ""
    echo "🛑 停止所有服务..."
    trap - INT TERM

    local tracked_pids=("${PIDS[@]}")
    if [ -n "$PID_TRACK_FILE" ] && [ -f "$PID_TRACK_FILE" ]; then
        while IFS= read -r pid; do
            if [ -n "$pid" ]; then
                tracked_pids+=("$pid")
            fi
        done < "$PID_TRACK_FILE"
    fi

    for pid in "${tracked_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    local deadline=$((SECONDS + 10))
    while [ $SECONDS -lt $deadline ]; do
        local alive=0
        for pid in "${tracked_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        if [ "$alive" -eq 0 ]; then
            break
        fi
        sleep 1
    done

    for pid in "${tracked_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    pkill -9 -f "toy_proxy_server.py" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    echo "✅ 清理完成"
    exit "$exit_code"
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

    local bench_input_tag
    local bench_output_tag
    local bench_rate_tag
    local runtime_log_suffix
    bench_input_tag=$(normalize_tag_value "$BENCH_RANDOM_INPUT_LENS")
    bench_output_tag=$(normalize_tag_value "$BENCH_RANDOM_OUTPUT_LENS")
    bench_rate_tag=$(normalize_tag_value "$BENCH_REQUEST_RATES")
    runtime_log_suffix="in${bench_input_tag}_out${bench_output_tag}_rr${bench_rate_tag}"
    if [ -n "$LOG_SUFFIX" ]; then
        runtime_log_suffix="${LOG_SUFFIX}_${runtime_log_suffix}"
    fi

    PREFILL_LOG="$LOG_DIR/prefill_${runtime_log_suffix}.log"
    DECODE_LOG="$LOG_DIR/decode_${runtime_log_suffix}.log"
    PROXY_LOG="$LOG_DIR/proxy_${runtime_log_suffix}.log"
    PID_TRACK_FILE="${PID_TRACK_FILE:-$LOG_DIR/pids_${runtime_log_suffix}.txt}"
    : > "$PID_TRACK_FILE"

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
    echo "  log_suffix: $runtime_log_suffix"
    echo "  pid_track_file: $PID_TRACK_FILE"
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
        > "$PROXY_LOG" 2>&1 &
    record_pid "$!"
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
        > "$PREFILL_LOG" 2>&1 &
    record_pid "$!"
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
        > "$DECODE_LOG" 2>&1 &
    record_pid "$!"
    echo "✅ Decode 已启动 (PID: ${PIDS[-1]})"

    # -----------------------------------------------------------------------------
    # 等待所有服务就绪
    # -----------------------------------------------------------------------------
    echo ""
    echo "[4/4] 等待服务就绪..."

    if ! wait_for_server $PREFILL_PORT "Prefill"; then
        echo "❌ Prefill 启动失败，查看日志: $PREFILL_LOG"
        cleanup 1
    fi

    if ! wait_for_server $DECODE_PORT "Decode"; then
        echo "❌ Decode 启动失败，查看日志: $DECODE_LOG"
        cleanup 1
    fi

    if ! wait_for_server $PROXY_PORT "Proxy"; then
        echo "❌ Proxy 启动失败，查看日志: $PROXY_LOG"
        cleanup 1
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
    echo "  $PREFILL_LOG"
    echo "  $DECODE_LOG"
    echo "  $PROXY_LOG"
    echo ""

    local input_lens_raw="${BENCH_RANDOM_INPUT_LENS//,/ }"
    local output_lens_raw="${BENCH_RANDOM_OUTPUT_LENS//,/ }"
    local request_rates_raw="${BENCH_REQUEST_RATES//,/ }"
    local input_lens=()
    local output_lens=()
    local request_rates=()
    read -r -a input_lens <<< "$input_lens_raw"
    read -r -a output_lens <<< "$output_lens_raw"
    read -r -a request_rates <<< "$request_rates_raw"

    if [ "${#input_lens[@]}" -eq 0 ] || [ "${#output_lens[@]}" -eq 0 ] || [ "${#request_rates[@]}" -eq 0 ]; then
        echo "❌ BENCH_RANDOM_INPUT_LENS / BENCH_RANDOM_OUTPUT_LENS / BENCH_REQUEST_RATES 配置为空"
        cleanup 1
    fi

    # Benchmark 参数
    echo "--------------------------------------------"
    echo "Benchmark 配置："
    echo "  模型: $MODEL"
    echo "  端口: $BENCH_PORT"
    echo "  random_input_lens: ${input_lens[*]}"
    echo "  random_output_lens: ${output_lens[*]}"
    echo "  request_rates: ${request_rates[*]}"
    echo "  goodput_ttft_ms: $BENCH_GOODPUT_TTFT_MS"
    echo "  goodput_tpot_ms: $BENCH_GOODPUT_TPOT_MS"
    echo "  num_prompts: $BENCH_NUM_PROMPTS"
    echo "  burstiness: $BENCH_BURSTINESS"
    echo "--------------------------------------------"
    echo ""

    echo "启动 Benchmark 组合测试..."
    local total_runs=0
    local success_runs=0
    local failed_runs=0
    local final_exit_code=0
    local benchmark_ts
    benchmark_ts=$(date +%Y%m%d_%H%M%S)
    local summary_log="$LOG_DIR/benchmark_summary_${benchmark_ts}.log"
    : > "$summary_log"

    for input_len in "${input_lens[@]}"; do
        for output_len in "${output_lens[@]}"; do
            for request_rate in "${request_rates[@]}"; do
                
                total_runs=$((total_runs + 1))
                local rate_tag="${request_rate//./p}"
                local combo_tag="in${input_len}_out${output_len}_rr${rate_tag}"
                local combo_log="$LOG_DIR/benchmark_${combo_tag}.log"

                echo ""
                echo "[Benchmark $total_runs] random_input_len=$input_len, random_output_len=$output_len, request_rate=$request_rate"
                echo "日志: $combo_log"

                vllm bench serve \
                    --host 127.0.0.1 \
                    --port $BENCH_PORT \
                    --seed $BENCH_SEED \
                    --model "$MODEL" \
                    --backend openai-chat \
                    --endpoint /v1/chat/completions \
                    --dataset-name random \
                    --random-input-len "$input_len" \
                    --random-output-len "$output_len" \
                    --num-prompts $BENCH_NUM_PROMPTS \
                    --burstiness $BENCH_BURSTINESS \
                    --request-rate "$request_rate" \
                    --goodput "ttft:$BENCH_GOODPUT_TTFT_MS" "tpot:$BENCH_GOODPUT_TPOT_MS" \
                    --ignore-eos \
                    2>&1 | tee "$combo_log"

                local bench_exit_code=${PIPESTATUS[0]}
                if [ "$bench_exit_code" -eq 0 ]; then
                    success_runs=$((success_runs + 1))
                else
                    failed_runs=$((failed_runs + 1))
                    final_exit_code=1
                fi

                echo "input_len=$input_len, output_len=$output_len, request_rate=$request_rate, ttft_ms=$BENCH_GOODPUT_TTFT_MS, tpot_ms=$BENCH_GOODPUT_TPOT_MS, exit_code=$bench_exit_code, log=$combo_log" | tee -a "$summary_log"
            done
        done
    done

    echo ""
    echo "Benchmark 结束，总组合数: $total_runs，成功: $success_runs，失败: $failed_runs"
    echo "Benchmark 汇总日志: $summary_log"
    echo ""

    echo "清理服务..."
    cleanup "$final_exit_code"
}

main
