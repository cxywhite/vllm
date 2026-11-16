#!/usr/bin/env bash
# cap500 + timestamp÷3（短输出 + 高并发）trace，β=0.0246（Σ-correct），
# 跑 GWF(margin=100) vs FCFS，验证"调长度拿优势 + β 对齐"。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000_kv1p6_cap500_ts3
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf scheme4"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_o2000kv1p6_cap500_ts3"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_SIMPLIFY_PREDICT=0
# cap500 后 Σactual/Σpred = 0.0246（预测对齐新输出）
export WT_PREDICT_OUTPUT_SCALE=0.0246
export WT_GWF_TTFT_MARGIN_MS=100
export WT_GWF_LOG_INTERVAL_MS=1000

echo "=== cap500_ts3 + beta=0.0246: GWF(margin=100) vs FCFS ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
