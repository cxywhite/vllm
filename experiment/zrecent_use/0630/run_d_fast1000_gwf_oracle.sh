#!/usr/bin/env bash
# trace D (fast1000, 403 长 + 600 短, span~28.6min, demand 1990) —— GWF oracle 臂。
# 验证「甜点 demand + 缩短时长」下 GWF 优势是否仍在；oracle(WT_SIMPLIFY_PREDICT=1) 精确路由。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000_kv1p2_1p1d_x8_fast1000
export PD_RATIO_LIST="1:1"
export PD_1P1D_PREFILL_GPUS=0 PD_1P1D_DECODE_GPUS=4
# 端口 offset 0（默认）：instance 22001/22011/22020/22032, PROXY 28002, BENCH 22007, predictor 32323, routing 32324
export WT_PREDICTOR_ZMQ_ADDR="tcp://127.0.0.1:32323"
export WT_PREDICT_PROXY_ZMQ_ADDR="tcp://127.0.0.1:32323"
export WT_ROUTING_PUB_ADDR="tcp://127.0.0.1:32324"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_d_fast1000_gwf_oracle"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export WT_SIMPLIFY_PREDICT=1
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250
export WT_GWF_LOG_INTERVAL_MS=1000

echo "=== scheme4_gwf (oracle) @ trace D fast1000 1p1d GPU0+4 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
