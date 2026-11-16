#!/usr/bin/env bash
# predacc_lmsys_o1600: FCFS(native greedy) + conservative(static buffer) 对比
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=predacc_lmsys_o1600
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export RUN_REPEAT_PER_CONFIG=1
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250

echo "=== [1/2] FCFS (native greedy) o1600 ==="
export COMPARE_EXPERIMENT_VARIANTS="scheme4"
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_predacc_fcfs_o1600"
unset WT_CONSERVATIVE_PREDICT WT_CONSERVATIVE_MAX_MODEL_LEN
bash test_evaluation_scheme4_rr_compare.sh

echo "=== [2/2] conservative (static buffer=7000) o1600 ==="
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_predacc_conservative_o1600"
export WT_CONSERVATIVE_PREDICT=1
export WT_CONSERVATIVE_MAX_MODEL_LEN=7000
bash test_evaluation_scheme4_rr_compare.sh

echo "=== done ==="
