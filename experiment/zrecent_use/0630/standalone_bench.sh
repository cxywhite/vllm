#!/usr/bin/env bash
# -------------------------------------------------------------------
# Standalone benchmark: llama + 2P:6D + 30ms TPOT + 600s qwen trace
# Prefill & decode instances start in parallel per role.
# Cleanup uses SIGTERM first so CUDA can release GPU memory.
# Usage: bash standalone_bench.sh [roundrobin|scheme4_gwf]
# -------------------------------------------------------------------
set -euo pipefail
METHOD="${1:-roundrobin}"

MODEL="/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct"
DATASET="${DATASET:-/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset/qwen/llama_lmsys_qwentrace_4080_600s.csv}"
BENCH_SCRIPT="/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/benchmark_serving_trace.py"
RESULT_DIR="${RESULT_DIR:-./standalone_result}"
PROXY_PORT=28002; BENCH_PORT=22007
PREFILL_PORTS=(22001 22002); PREFILL_KV_PORTS=(22011 22012)
DECODE_PORTS=(22020 22021 22022 22023 22024 22025); DECODE_KV_PORTS=(22032 22033 22034 22035 22036 22037)

# Track PIDs for graceful teardown
ALL_PIDS=()

cleanup() {
    echo "Cleaning up (SIGTERM)..."
    for pid in "${ALL_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    pkill -TERM -f "vllm serve" 2>/dev/null || true
    pkill -TERM -f "disagg_proxy" 2>/dev/null || true
    pkill -TERM -f "test_disagg_proxy" 2>/dev/null || true
    sleep 5
    echo "Cleaning up (SIGKILL fallback)..."
    for pid in "${ALL_PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    pkill -KILL -f "disagg_proxy" 2>/dev/null || true
    pkill -KILL -f "test_disagg_proxy" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT
mkdir -p "$RESULT_DIR"

# Wait for a TCP port to become responsive
wait_for_port() {
    local port=$1 label=$2 timeout=${3:-1800}
    for _ in $(seq 1 "$timeout"); do
        curl -s localhost:"$port" >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

case "$METHOD" in
    roundrobin)
        PROXY_SCRIPT="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py"
        export WT_PREDICT_ROUTE_SCHEME=roundrobin WT_PREDICT_ANALYSIS_ENABLE=0
        PREFILL_EXTRA=""; DECODE_EXTRA=""
        ;;
    scheme4_gwf)
        PROXY_SCRIPT="/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py"
        export WT_PREDICT_ROUTE_SCHEME=scheme4 WT_PREDICT_ANALYSIS_ENABLE=1
        PREFILL_EXTRA="--activation-predict"; DECODE_EXTRA="--geometric-window-scheduler --test-p2d"
        ;;
    *) echo "ERROR: unknown method $METHOD"; exit 1 ;;
esac

echo "=== Method: $METHOD ==="

# ── 1. Proxy ──
echo "[1/4] Starting proxy..."
setsid env PYTHONPATH="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd:${PYTHONPATH:-}" \
    PROXY_PORT=$PROXY_PORT BENCH_PORT=$BENCH_PORT VLLM_DTYPE=bfloat16 \
    MODEL_CONFIG_PATH=$MODEL/config.json TPOT=tpot:30 \
    bash -c "exec python3 $PROXY_SCRIPT" > "$RESULT_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
ALL_PIDS+=($PROXY_PID)
echo "  Proxy PID=$PROXY_PID"

# ── 2. Prefill instances (GPU 0,1) — fire and forget ──
echo "[2/4] Launching prefill instances..."
for i in 0 1; do
    port=${PREFILL_PORTS[$i]}; kv_port=${PREFILL_KV_PORTS[$i]}; gpu=$i
    setsid env CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$((30000+port)) VLLM_USE_V1=1 \
        vllm serve "$MODEL" --enforce-eager --host 0.0.0.0 --port $port \
        --tensor-parallel-size 1 --seed 42 --dtype bfloat16 \
        --max-model-len 8192 --max-num-batched-tokens 32768 --max-num-seqs 1024 \
        --test-model llama --gpu-memory-utilization 0.8 $PREFILL_EXTRA \
        --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"mem_pool_size_gb\":\"32\",\"nccl_num_channels\":\"8\"}}" \
        > "$RESULT_DIR/prefill${i}.log" 2>&1 &
    ALL_PIDS+=($!)
    echo "  prefill$i PID=${ALL_PIDS[-1]} (GPU=$gpu, port=$port)"
done

# ── 3. Decode instances (GPU 2-7) — fire and forget ──
echo "[3/4] Launching decode instances..."
for i in 0 1 2 3 4 5; do
    port=${DECODE_PORTS[$i]}; kv_port=${DECODE_KV_PORTS[$i]}; gpu=$((2+i))
    setsid env CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$((40000+port)) VLLM_USE_V1=1 \
        vllm serve "$MODEL" --enforce-eager --host 0.0.0.0 --port $port \
        --tensor-parallel-size 1 --seed 42 --dtype bfloat16 \
        --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 1024 \
        --test-model llama --gpu-memory-utilization 0.8 --swap-space 0 $DECODE_EXTRA \
        --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"mem_pool_size_gb\":\"32\",\"nccl_num_channels\":\"8\"}}" \
        > "$RESULT_DIR/decode${i}.log" 2>&1 &
    ALL_PIDS+=($!)
    echo "  decode$i PID=${ALL_PIDS[-1]} (GPU=$gpu, port=$port)"
done

# ── 4. Wait for all servers ──
echo "[4/4] Waiting for all servers..."
# Verify proxy first (fast)
sleep 5
ps -p $PROXY_PID >/dev/null 2>&1 || { echo "ERROR: proxy died"; cat "$RESULT_DIR/proxy.log"; exit 1; }
echo "  Proxy ready ✅"
# Then wait for all prefill + decode ports
for i in 0 1; do
    port=${PREFILL_PORTS[$i]}
    echo "  Waiting for prefill$i (port $port)..."
    wait_for_port "$port" "prefill$i" || { echo "  prefill$i timeout ❌"; exit 1; }
    echo "  prefill$i ready ✅"
done
for i in 0 1 2 3 4 5; do
    port=${DECODE_PORTS[$i]}
    echo "  Waiting for decode$i (port $port)..."
    wait_for_port "$port" "decode$i" || { echo "  decode$i timeout ❌"; exit 1; }
    echo "  decode$i ready ✅"
done

# ── 4. Benchmark ──
echo "=== All ready. Running benchmark... ==="
python3 "$BENCH_SCRIPT" \
    --backend openai-chat --port "$BENCH_PORT" --endpoint /v1/chat/completions \
    --model "$MODEL" --dataset-name trace --dataset-path "$DATASET" \
    --save-output True --out-path "$RESULT_DIR" --maxtokenscustom 8192 --seed 42 \
    --max-concurrency 1024 --temperature 0.8 --top-p 0.7 --top-k 50 --repetition-penalty 1.0 \
    --goodput ttft:1000 tpot:30 --ignore-eos --save-sample 2>&1 | tee "$RESULT_DIR/benchmark.log"

echo "=== Done. Results in $RESULT_DIR/ ==="
