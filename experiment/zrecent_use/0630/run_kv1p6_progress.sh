#!/usr/bin/env bash
# kv1p6 上跑 progress-conservative 调度器（est = prompt + output*1.5）
# 对照：native(FCFS) 134.2 / GWF-oracle 135.1（已有）
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000_kv1p6
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_progress"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_o2000kv1p6_progress"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
# 路由用实际 BERT 预测（β=0.30，Presage 口径），不是 oracle
export WT_PREDICT_OUTPUT_SCALE=0.30
export WT_PROGRESS_MARGIN=1.5
export WT_PROGRESS_LOG_INTERVAL_MS=1000

echo "=== progress-conservative (margin=1.5) @ kv1p6 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
