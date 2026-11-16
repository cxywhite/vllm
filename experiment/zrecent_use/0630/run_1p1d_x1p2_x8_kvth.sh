#!/usr/bin/env bash
# 1p1d ×1.2×8 KV-threshold（审稿人 keep-a-buffer）臂，独立重跑。
# GPU 1+5，端口 23xxx + BENCH 23007（避开 22xxx FCFS / 24xxx GWF）。
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
export PD_1P1D_PREFILL_GPUS=1 PD_1P1D_DECODE_GPUS=5
export PD_1P1D_PREFILL_PORTS=23001 PD_1P1D_PREFILL_KV_PORTS=23011
export PD_1P1D_DECODE_PORTS=23020 PD_1P1D_DECODE_KV_PORTS=23032
export PROXY_PORT=29002
export BENCH_PORT=23007
# 预测相关端口也要错开（否则三个 proxy 都绑默认 32323/32324 会撞）
export WT_PREDICTOR_ZMQ_ADDR="tcp://127.0.0.1:33323"
export WT_PREDICT_PROXY_ZMQ_ADDR="tcp://127.0.0.1:33323"
export WT_ROUTING_PUB_ADDR="tcp://127.0.0.1:33324"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_kvthreshold"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_1p1d_x1p2_x8_kvth"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export WT_SIMPLIFY_PREDICT=1
export WT_KV_ADMIT_THRESHOLD=0.8 WT_KV_ADMIT_LOG_INTERVAL_MS=1000

echo "=== scheme4_kvthreshold (oracle, threshold=0.8) @ ×1.2×8 1p1d GPU1+5 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
