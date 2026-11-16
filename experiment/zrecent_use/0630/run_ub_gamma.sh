#!/usr/bin/env bash
# 逐一跑 UniBoost 的 gamma 网格 {1e-4,1e-3,1e-2}，o2000 + 4P4D + util0.8
# 用法: bash run_ub_gamma.sh <GAMMA>
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630
G="${1:?gamma}"
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_uniboost"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_ub_o2000_g${G}"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_SIMPLIFY_PREDICT=1
export WT_UNIBOOST_GAMMA="$G" WT_UNIBOOST_LOG_INTERVAL_MS=1000
bash test_evaluation_scheme4_rr_compare.sh
