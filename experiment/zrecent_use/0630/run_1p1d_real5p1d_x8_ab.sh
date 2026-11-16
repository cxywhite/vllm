#!/usr/bin/env bash
# 阶段B: 1p1d + 真实 BERT 预测 + GWF vs UniBoost 双臂对比（实例内隔离）
# trace = qwen_real5p1d_1p1d_x8（从 5p1d 真实数据按模板分布+预测准优选 4080 条，rate0.852/s）
# 与阶段A唯一区别：去掉 WT_SIMPLIFY_PREDICT（oracle），改 WT_PREDICT_OUTPUT_SCALE=0.30（β校准）
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_real5p1d_1p1d_x8
export PD_RATIO_LIST="1:1"
export PD_1P1D_PREFILL_GPUS=0 PD_1P1D_DECODE_GPUS=4
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf scheme4_uniboost"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_1p1d_real5p1d_x8_ab"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
# 真实 BERT 预测：不设 WT_SIMPLIFY_PREDICT，β 校准 0.30
export WT_PREDICT_OUTPUT_SCALE=0.30
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_UNIBOOST_CLOCK=wall WT_UNIBOOST_GAMMA0_S=0.1 WT_UNIBOOST_MAX_JUMP_S=60 WT_UNIBOOST_LOG_INTERVAL_MS=1000 WT_UNIBOOST_ADMIT_PREEMPT=0
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250
bash test_evaluation_scheme4_rr_compare.sh
