#!/usr/bin/env bash
# 通用 1p1d GWF 单臂：<TRACE_NAME> <RESULT_DIR>
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630
TRACE="${1:?}"; RDIR="${2:?}"
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME="$TRACE"
export PD_RATIO_LIST="1:1"
export PD_1P1D_PREFILL_GPUS=4 PD_1P1D_DECODE_GPUS=5
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/$RDIR"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PREFILL_GPU_MEMORY_UTILIZATION=0.8 DECODE_GPU_MEMORY_UTILIZATION=0.8
export WT_SIMPLIFY_PREDICT=1
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_GWF_WINDOW_TPOT_TARGET_MS=120
export WT_GWF_TTFT_MARGIN_MS=250
bash test_evaluation_scheme4_rr_compare.sh
