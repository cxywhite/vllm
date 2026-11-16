#!/usr/bin/env bash
# 1p1d x1.2 怪物 + timestamp x8 (rate 0.85/s) 甜点负载上，补 GWF vs FCFS vs KV-threshold 三臂。
# 之前只跑过 GWF vs UniBoost（GWF 144.17 > UniBoost 133.51）；现在补 FCFS + KV-threshold 完成 rebuttal 对比。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000_kv1p2_1p1d_x8
export PD_RATIO_LIST="1:1"
export PD_1P1D_PREFILL_GPUS=0 PD_1P1D_DECODE_GPUS=4
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_kvthreshold scheme4"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_1p1d_x1p2_x8_three"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export WT_SIMPLIFY_PREDICT=1
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250
export WT_GWF_LOG_INTERVAL_MS=1000
export WT_KV_ADMIT_THRESHOLD=0.8 WT_KV_ADMIT_LOG_INTERVAL_MS=1000
bash test_evaluation_scheme4_rr_compare.sh
