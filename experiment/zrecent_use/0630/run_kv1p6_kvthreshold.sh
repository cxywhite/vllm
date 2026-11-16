#!/usr/bin/env bash
# kv1p6 上跑「真实 BERT + scheme4 + KV 阈值准入（keep a buffer）」。
# 审稿人描述的准入：track 实际 KV usage，达到阈值(0.8=80%→留 20% buffer)就停止从 waiting 加请求。
# 预测无关、无模拟、无自定义抢占。
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
export COMPARE_EXPERIMENT_VARIANTS="scheme4_kvthreshold"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_o2000kv1p6_kvthreshold"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_SIMPLIFY_PREDICT=0
export WT_PREDICT_OUTPUT_SCALE=0.30
# 阈值：0.8 = KV 用到 80% 就停止准入（留 20% buffer）。可扫 0.7/0.9。
export WT_KV_ADMIT_THRESHOLD=0.8
export WT_KV_ADMIT_LOG_INTERVAL_MS=1000

echo "=== scheme4_kvthreshold (threshold=0.8) @ kv1p6 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
