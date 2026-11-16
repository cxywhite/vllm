#!/bin/bash
# 补跑 Dynamo 缺失 2p6d (7个), 异常不中断, 自动跑完全部
PROJECT_DIR="/home/dynamo/dynamo"
LAUNCH_DIR="${PROJECT_DIR}/examples/backends/vllm/launch"
RESULT_BASE="${PROJECT_DIR}/experiment_result"
T=$(date +%Y%m%d_%H%M%S)
EVAL="/home/dynamo/share/huggingface-cache/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset"
M="${EVAL}/mooncake"
Q="${EVAL}/derived_datasets/strategyA_aggressive_2p6d"
ML="/home/dynamo/myshare/models/Meta-Llama-3-8B-Instruct"
MQ="/home/dynamo/myshare/models/Qwen-2.5-7b-Instruct"

cleanup(){ pkill -f dynamo.frontend 2>/dev/null; pkill -f dynamo.vllm 2>/dev/null; pkill -f nats-server 2>/dev/null; pkill -f etcd 2>/dev/null; sleep 3; }
trap cleanup EXIT

wait_ready(){
  for((i=0;i<60;i++)); do
    curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"' && { sleep 5; return 0; }
    sleep 5
  done
  echo "TIMEOUT: Dynamo 未就绪"; return 1
}

run(){
  local model="$1" ds="$2" dir="$3" name; name=$(basename "$ds" .csv)
  echo "--- [${CUR}/7] ${name} --- $(date +%H:%M:%S)"
  [[ -f "$ds" ]] || { echo "SKIP: 数据集不存在"; FAILED+=("$name"); return; }
  mkdir -p "$dir"
  cd "${PROJECT_DIR}/benchmarks" || { echo "SKIP: cd 失败"; FAILED+=("$name"); return; }
  if python benchmark_serving_trace.py --model "$model" --backend openai-chat \
    --endpoint /v1/chat/completions --dataset-name trace --dataset-path "$ds" \
    --max-concurrency 1024 --temperature 0.8 --top-p 0.7 --top-k 50 --repetition-penalty 1.0 \
    --ignore-eos --goodput ttft:1000 tpot:30 --save-result \
    --result-dir "$dir" --save-output True --out-path "$dir" \
    2>&1 | tee "$dir/benchmark.log"; then
    echo "OK: $(date +%H:%M:%S)"
  else
    echo "FAILED: $(date +%H:%M:%S)"; FAILED+=("$name")
  fi
}

# 格式: model|dataset_path|launch_script|result_dir_name
CONFIGS=(
  "$ML|${M}/llama_lmsys_mooncake_1790_new.csv|llama_disagg_router_2p6d.sh|llama_llama-lmsys-chat-mooncake_n1790_pd2p6d_tpot30ms_tracemooncake_rep1_dynamo"
  "$ML|${M}/llama_mysharegpt_mooncake_1790_new.csv|llama_disagg_router_2p6d.sh|llama_llama-mysharegpt-chat-mooncake_n1790_pd2p6d_tpot30ms_tracemooncake_rep1_dynamo"
  "$ML|${Q}/llama_lmsys_qwentrace_4080_600s_strategyA.csv|llama_disagg_router_2p6d.sh|llama_llama-lmsys-chat-qwen_strategyA_n4080_pd2p6d_tpot30ms_traceqwen_strategyA_rep1_dynamo"
  "$ML|${Q}/llama_mysharegpt_qwentrace_4080_600s_strategyA.csv|llama_disagg_router_2p6d.sh|llama_llama-mysharegpt-chat-qwen_strategyA_n4080_pd2p6d_tpot30ms_traceqwen_strategyA_rep1_dynamo"
  "$MQ|${M}/qwen_lmsys_mooncake_1790_new.csv|qwen_disagg_router_2p6d.sh|qwen_qwen-lmsys-chat-mooncake_n1790_pd2p6d_tpot30ms_tracemooncake_rep1_dynamo"
  "$MQ|${M}/qwen_mysharegpt_mooncake_1790_new.csv|qwen_disagg_router_2p6d.sh|qwen_qwen-mysharegpt-chat-mooncake_n1790_pd2p6d_tpot30ms_tracemooncake_rep1_dynamo"
  "$MQ|${Q}/qwen_lmsys_qwentrace_4080_600s_strategyA.csv|qwen_disagg_router_2p6d.sh|qwen_qwen-lmsys-chat-qwen_strategyA_n4080_pd2p6d_tpot30ms_traceqwen_strategyA_rep1_dynamo"
)

echo "=== Dynamo 2p6d 补跑 x7 === $(date)"
FAILED=(); CUR=0; LAST_SCRIPT=""

for cfg in "${CONFIGS[@]}"; do
  CUR=$((CUR+1))
  IFS='|' read -r model ds script dirname <<< "$cfg"

  if [[ "$script" != "$LAST_SCRIPT" ]]; then
    LAST_SCRIPT="$script"
    cleanup; sleep 2
    cd "$LAUNCH_DIR" || continue
    nohup bash "$script" > "${RESULT_BASE}/dynamo_$(echo $script|sed 's/.sh//')_${T}.log" 2>&1 &
    sleep 2
    if ! wait_ready; then
      echo "SKIP: Dynamo 启动失败"
      FAILED+=("Dynamo:$(basename $script)")
      continue
    fi
  fi

  run "$model" "$ds" "${RESULT_BASE}/${dirname}"
  sleep 30
done

cleanup
echo "=== 完成: OK=$((7-${#FAILED[@]}))/7  FAILED: ${FAILED[@]}  ==="
echo "结果: ${RESULT_BASE}/*_${T}/"
