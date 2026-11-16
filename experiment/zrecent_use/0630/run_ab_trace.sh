#!/usr/bin/env bash
# 通用单臂 runner：<TRACE_NAME> <VARIANT> <RESULT_DIR> [WT_UNIBOOST_GAMMA]
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630
TRACE="${1:?}"; VAR="${2:?}"; RDIR="${3:?}"; G="${4:-1e-2}"
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME="$TRACE"
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="$VAR"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/$RDIR"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_SIMPLIFY_PREDICT=1
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_UNIBOOST_GAMMA="$G" WT_UNIBOOST_LOG_INTERVAL_MS=1000 WT_UNIBOOST_ADMIT_PREEMPT=0
export WT_GWF_WINDOW_TPOT_TARGET_MS=120
export WT_GWF_TTFT_MARGIN_MS=250
bash test_evaluation_scheme4_rr_compare.sh
