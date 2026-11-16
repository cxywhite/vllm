#!/usr/bin/env bash
# 1p1d ×1.2 ×8 native vLLM (scheme4 = vLLM 原生 FCFS 贪心准入) 臂 —— 补齐三臂对比的第三臂
# 与已完成的 scheme4_gwf(144.17) / scheme4_uniboost(133.51) 同 trace、同 oracle(WT_SIMPLIFY_PREDICT=1)
# GPU: 原 GWF/UniBoost 用 prefill0+decode4；现在 GPU0 被 co-tenant 占，prefill 改 5，decode 仍 4（保持 decode 一致）
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
export PD_1P1D_PREFILL_GPUS=5 PD_1P1D_DECODE_GPUS=4
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_1p1d_x1p2_x8_ab"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export WT_SIMPLIFY_PREDICT=1
export WT_NATIVE_LOG_INTERVAL_MS=1000
bash test_evaluation_scheme4_rr_compare.sh
