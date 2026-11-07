#!/bin/bash
BASE=/root/predict-schedule/experiment/zrecent_use/0630
cd "$BASE" || exit 1
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export QWEN_MODEL_PATH="/root/myshare/models/Qwen-2.5-7b-Instruct"
export LLAMA_MODEL_PATH="/root/myshare/models/Meta-Llama-3-8B-Instruct"
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_SIMPLIFY_PREDICT=1
export WT_LIVE_CSV="$BASE/live_latency.csv"
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
while true; do
  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000' | wc -l)
  echo "[$(date '+%F %T')] free_gpus=$free/8"
  if [ "$free" -ge 8 ]; then
    echo "[$(date '+%F %T')] GPU 全空，启动 4P4D"
    rm -f "$BASE/live_latency.csv"
    DATASET_BASE_DIR="$BASE/admission_ab_base" \
    TRACE_NAME=qwen_o1500_x2 \
    MODEL_TYPE=qwen \
    COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf" \
    PD_RATIO_LIST="4:4" \
    TPOT_MS_LIST="30" \
    RUN_REPEAT_PER_CONFIG=1 \
    BASE_RESULT_DIR="$BASE/wl_4p4d_gmu08_1500x2" \
    nohup bash test_evaluation_scheme4_rr_compare.sh > "$BASE/wl_4p4d_gmu08_1500x2.log" 2>&1 &
    sleep 5
    nohup python3 -u wt_live_monitor.py "$BASE/live_latency.csv" --interval 5 > "$BASE/live_monitor.log" 2>&1 &
    echo "launched eval PID=$!"
    break
  fi
  sleep 180
done
