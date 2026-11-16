#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 1P1D 三策略对比实验: Aggressive vs Predict+GWF vs Conservative
# Llama-3-8B, 1 Prefill GPU + 1 Decode GPU
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 固定配置 ---
MODEL_PATH="/root/myshare/models/Meta-Llama-3-8B-Instruct"
DATASET_BASE_DIR="/root/myshare/dataset"
MODEL_TYPE="llama"
PD_RATIO="1:1"
TPOT_MS="30"
MAX_MODEL_LEN=8192

# 1P1D GPU 分配
PREFILL_GPU=2
DECODE_GPU=3

# 数据集：从 predictor 预测最准的 500 条
DATASETS=(
    "${SCRIPT_DIR}/experiment_1p1d_compare/llama_1p1d_400.csv"
)
REQUEST_RATE=inf
DECODE_MAX_BATCHED_TOKENS=8192
DECODE_MAX_NUM_SEQS=256
BENCH_GOODPUT="ttft:1000 tpot:30"

# --- 实验策略定义 ---
# 格式: name|route_scheme|use_predict|use_gwf|test_p2d|conservative|extra_env
STRATEGIES=(
    # Default — roundrobin proxy + vLLM 默认调度器 (no predict/gwf/p2d)
    "default|roundrobin|false|false|false|false|"
    # GWF — scheme4 proxy + GWF 准入控制
    "gwf|scheme4|true|true|true|false|"
    # Conservative — scheme4 proxy + GWF + conservative predict
    "conservative|scheme4|true|true|true|true|WT_CONSERVATIVE_PREDICT=1 WT_CONSERVATIVE_MAX_MODEL_LEN=${MAX_MODEL_LEN}"
)

# --- 实验参数 ---
TIMEOUT_SECONDS=2400
BASE_RESULT_DIR="./experiment_1p1d_compare"
RUN_BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${BASE_RESULT_DIR}/run_${RUN_BATCH_TIMESTAMP}.log"
mkdir -p "$BASE_RESULT_DIR"

# Proxy & Benchmark
PROXY_SCRIPT="/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py"
BENCH_SCRIPT="/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/benchmark_serving_trace.py"
PROXY_PORT=28002
BENCH_PORT=22007
PREFILL_PORT=22001
DECODE_PORT=22020
PREFILL_KV_PORT=22011
DECODE_KV_PORT=22032
DIST_PREFILL_PORT=52001
DIST_DECODE_PORT=62020

VLLM_ENFORCE_EAGER=1
VLLM_SEED=42
VLLM_DTYPE=bfloat16
GPU_MEM_UTIL=0.8
TENSOR_PARALLEL_SIZE=1
TENSOR_POOL_MEMORY=32

KV_CONNECTOR=P2pNcclConnector
KV_PRODUCER_BUFFER=1e9
KV_CONSUMER_BUFFER=8e9
KV_NCCL_CHANNELS=8
KV_SEND_TYPE=PUT_ASYNC

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_sep() { log "============================================================"; }

# --- 工具函数 ---
stop_all() {
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "test_disagg_proxy" 2>/dev/null || true
    pkill -9 -f "disagg_proxy" 2>/dev/null || true
    pkill -9 -f "benchmark_serving_trace" 2>/dev/null || true
    for port in ${PROXY_PORT} ${BENCH_PORT} ${PREFILL_PORT} ${DECODE_PORT}; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    sleep 3
}

wait_for_server() {
    local port="$1" label="$2"
    local start_time=$(date +%s)
    log "  Waiting for ${label} on port ${port}..."
    while true; do
        if curl -s "localhost:${port}" >/dev/null 2>&1; then
            return 0
        fi
        if (( $(date +%s) - start_time > 600 )); then
            log "  ERROR: timeout waiting for ${label} on port ${port}"
            return 1
        fi
        sleep 2
    done
}

# --- 主流程 ---
main() {
    log_sep
    log "1P1D Three-Strategy Experiment — Llama-3-8B"
    log "  Model:     ${MODEL_PATH}"
    log "  Datasets:  ${#DATASETS[@]} files in mooncake/"
    log "  Strategies: ${#STRATEGIES[@]} (aggressive / predict_gwf / conservative)"
    log "  PD Ratio:  ${PD_RATIO} (GPU${PREFILL_GPU} prefill + GPU${DECODE_GPU} decode)"
    log_sep

    local total=0 success=0

    for ds_path in "${DATASETS[@]}"; do
        local ds_name="$(basename "${ds_path%.csv}")"

        for strat in "${STRATEGIES[@]}"; do
            IFS='|' read -r s_name s_route s_predict s_gwf s_p2d s_cons s_extra <<< "$strat"
            total=$((total + 1))

            local ts=$(date +%Y%m%d_%H%M%S)
            local run_dir="${BASE_RESULT_DIR}/${MODEL_TYPE}_${s_name}_${ds_name}_pd1p1d_${ts}"

            log_sep
            log "[${total}/$(( ${#DATASETS[@]} * ${#STRATEGIES[@]} ))] Strategy=${s_name} Dataset=${ds_name}"
            log "  route=${s_route} predict=${s_predict} gwf=${s_gwf} p2d=${s_p2d} conservative=${s_cons}"

            mkdir -p "${run_dir}/log" "${run_dir}/dataset_result"

            # --- Kill stale processes ---
            stop_all

            # --- Launch Proxy ---
            local proxy_env="PYTHONPATH=/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd:${PYTHONPATH:-} PROXY_PORT=${PROXY_PORT} BENCH_PORT=${BENCH_PORT} VLLM_DTYPE=${VLLM_DTYPE} MODEL_CONFIG_PATH=${MODEL_PATH}/config.json TPOT=tpot:${TPOT_MS} WT_PREDICT_ROUTE_SCHEME=${s_route} WT_PREDICT_CHAIN_VERBOSE=0 WT_ROUTE_DECISION_LOG_ENABLED=0 WT_PREDICT_ANALYSIS_ENABLE=0 WT_MONITOR_LOG_INTERVAL_MS=5000 ${s_extra}"
            log "  Proxy env: ${proxy_env}"
            setsid env ${proxy_env} bash -c "exec python3 ${PROXY_SCRIPT}" > "${run_dir}/log/proxy_${ts}.log" 2>&1 &
            local proxy_pid=$!
            sleep 3
            if ! ps -p ${proxy_pid} >/dev/null 2>&1; then
                log "  ERROR: proxy died immediately"
                cat "${run_dir}/log/proxy_${ts}.log" | tail -20
                stop_all; continue
            fi

            # --- Launch Prefill ---
            local kv_cfg_prefill="{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"${KV_PRODUCER_BUFFER}\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"${PREFILL_PORT}\",\"send_type\":\"${KV_SEND_TYPE}\",\"mem_pool_size_gb\":\"${TENSOR_POOL_MEMORY}\",\"nccl_num_channels\":\"${KV_NCCL_CHANNELS}\"}}"

            local prefill_cmd="vllm serve \"${MODEL_PATH}\" --enforce-eager --host 0.0.0.0 --port ${PREFILL_PORT} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --seed ${VLLM_SEED} --dtype ${VLLM_DTYPE} --max-model-len ${MAX_MODEL_LEN} --max-num-batched-tokens 32768 --max-num-seqs 1024 --test-model llama --gpu-memory-utilization ${GPU_MEM_UTIL} --kv-transfer-config '${kv_cfg_prefill}'"

            if [[ "${s_predict}" == "true" ]] || [[ "${s_p2d}" == "true" ]]; then
                prefill_cmd="${prefill_cmd} --activation-predict"
            fi

            setsid env CUDA_VISIBLE_DEVICES=${PREFILL_GPU} MASTER_ADDR=127.0.0.1 MASTER_PORT=${DIST_PREFILL_PORT} VLLM_USE_V1=1 bash -c "exec ${prefill_cmd}" > "${run_dir}/log/prefill_${ts}.log" 2>&1 &
            local prefill_pid=$!

            # --- Launch Decode ---
            local kv_cfg_decode="{\"kv_connector\":\"${KV_CONNECTOR}\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"${KV_CONSUMER_BUFFER}\",\"kv_port\":\"${DECODE_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"${DECODE_PORT}\",\"send_type\":\"${KV_SEND_TYPE}\",\"mem_pool_size_gb\":\"${TENSOR_POOL_MEMORY}\",\"nccl_num_channels\":\"${KV_NCCL_CHANNELS}\"}}"

            local decode_cmd="vllm serve \"${MODEL_PATH}\" --enforce-eager --host 0.0.0.0 --port ${DECODE_PORT} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --seed ${VLLM_SEED} --dtype ${VLLM_DTYPE} --max-model-len ${MAX_MODEL_LEN} --max-num-batched-tokens ${DECODE_MAX_BATCHED_TOKENS} --max-num-seqs ${DECODE_MAX_NUM_SEQS} --gpu-memory-utilization ${GPU_MEM_UTIL} --test-model llama --swap-space 0 --kv-transfer-config '${kv_cfg_decode}'"

            if [[ "${s_gwf}" == "true" ]]; then
                decode_cmd="${decode_cmd} --geometric-window-scheduler"
            fi
            if [[ "${s_p2d}" == "true" ]]; then
                decode_cmd="${decode_cmd} --test-p2d"
            fi

            setsid env CUDA_VISIBLE_DEVICES=${DECODE_GPU} MASTER_ADDR=127.0.0.1 MASTER_PORT=${DIST_DECODE_PORT} VLLM_USE_V1=1 bash -c "exec ${decode_cmd}" > "${run_dir}/log/decode_${ts}.log" 2>&1 &
            local decode_pid=$!

            # --- Wait for servers ---
            if ! wait_for_server ${PREFILL_PORT} "prefill"; then stop_all; continue; fi
            if ! wait_for_server ${DECODE_PORT} "decode"; then stop_all; continue; fi
            log "  All servers ready."

            # --- Run Benchmark ---
            local bench_cmd="python3 ${BENCH_SCRIPT} --backend openai-chat --port ${BENCH_PORT} --endpoint /v1/chat/completions --model ${MODEL_PATH} --dataset-name trace --dataset-path ${ds_path} --save-output True --out-path ${run_dir}/dataset_result --maxtokenscustom ${MAX_MODEL_LEN} --seed ${VLLM_SEED} --max-concurrency 1024 --temperature 0.8 --top-p 0.7 --top-k 50 --repetition-penalty 1.0 --goodput ${BENCH_GOODPUT} --ignore-eos --save-sample --request-rate ${REQUEST_RATE}"
            log "  Benchmark: ${bench_cmd}"
            if env PYTHONPATH=.:${PYTHONPATH:-} bash -c "exec ${bench_cmd}" > "${run_dir}/log/bench_trace_${ts}.log" 2>&1; then
                log "  BENCHMARK PASSED ✅"
                success=$((success + 1))
            else
                log "  BENCHMARK FAILED ❌ (exit code: $?)"
                tail -20 "${run_dir}/log/bench_trace_${ts}.log" | while read line; do log "    ${line}"; done
            fi

            stop_all
        done
    done

    log_sep
    log "DONE: ${success}/${total} passed"
    log_sep
}

main "$@"
