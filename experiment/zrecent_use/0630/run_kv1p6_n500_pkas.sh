#!/usr/bin/env bash
# PKAS 小批量验证（500 条 trace，1p3d，GPU 4-7）：验证 raw PKAS（无 β）能正确跑完、无请求丢失。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000_kv1p6_n500
export PD_RATIO_LIST="1:3"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_pkas"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_o2000kv1p6_n500_pkas"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_1P3D_PREFILL_GPUS=4
export PD_1P3D_DECODE_GPUS=5,6,7
export WT_SIMPLIFY_PREDICT=0
export WT_PREDICT_OUTPUT_SCALE=0.30
export WT_PKAS_LOG_INTERVAL_MS=1000

echo "=== scheme4_pkas (raw, 无 β) @ kv1p6 n500 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
