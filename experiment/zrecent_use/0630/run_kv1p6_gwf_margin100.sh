#!/usr/bin/env bash
# kv1p6 上跑「真实 BERT + scheme4 + GWF」，调 TTFT margin：250ms → 100ms。
# 目标：花 TTFT 富余（median 733→~900ms）换更低并发 → 更好 TPOT（当前绑定约束）。
# 其余旋钮与 run_kv1p6_gwf_real.sh 完全一致（WT_SIMPLIFY_PREDICT=0 + β=0.30）。
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
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_o2000kv1p6_gwf_m100"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_SIMPLIFY_PREDICT=0
export WT_PREDICT_OUTPUT_SCALE=0.30
# [WT] 唯一改动：TTFT margin 250ms -> 100ms（强制准入阈值 750ms -> 900ms）
export WT_GWF_TTFT_MARGIN_MS=100
export WT_GWF_LOG_INTERVAL_MS=1000

echo "=== scheme4_gwf (real BERT, beta=0.30, margin=100ms) @ kv1p6 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
