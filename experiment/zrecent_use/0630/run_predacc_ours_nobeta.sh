#!/usr/bin/env bash
# 消融：ours (Presage) 预测器，无 β（WT_PREDICT_OUTPUT_SCALE=1.0），重跑以便与
# µ-Serve / TRAIL++ 在同等"无 β"条件下三方对比。其余与 4 臂完全一致。
# 4p4d，o1600(GWF+UniBoost) -> o2000(GWF+UniBoost)。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=1
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_PREDICT_OUTPUT_SCALE=1.0
# 不设 WT_PREDICTOR_KIND -> 默认 "ours"
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_UNIBOOST_CLOCK=wall WT_UNIBOOST_GAMMA0_S=0.1 WT_UNIBOOST_MAX_JUMP_S=60 WT_UNIBOOST_LOG_INTERVAL_MS=1000 WT_UNIBOOST_ADMIT_PREEMPT=0
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250

echo "=== [ours no-beta] o1600 ==="
export TRACE_NAME=predacc_lmsys_o1600
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_predacc_ours_nobeta_o1600"
bash test_evaluation_scheme4_rr_compare.sh

echo "=== [ours no-beta] o2000 ==="
export TRACE_NAME=predacc_lmsys_o2000
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_predacc_ours_nobeta_o2000"
bash test_evaluation_scheme4_rr_compare.sh

echo "=== ours no-beta done ==="
