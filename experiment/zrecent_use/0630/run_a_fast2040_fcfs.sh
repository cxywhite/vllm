#!/usr/bin/env bash
# trace A (fast2040, 等比例减半, rate 0.852 不变, span~40min) —— FCFS (scheme4 贪心) 臂。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=qwen_o2000_kv1p2_1p1d_x8_fast2040
export PD_RATIO_LIST="1:1"
export PD_1P1D_PREFILL_GPUS=2 PD_1P1D_DECODE_GPUS=3
export PD_1P1D_PREFILL_PORTS=24001 PD_1P1D_PREFILL_KV_PORTS=24011
export PD_1P1D_DECODE_PORTS=24020 PD_1P1D_DECODE_KV_PORTS=24032
export PROXY_PORT=30002
export BENCH_PORT=24007
export WT_PREDICTOR_ZMQ_ADDR="tcp://127.0.0.1:34323"
export WT_PREDICT_PROXY_ZMQ_ADDR="tcp://127.0.0.1:34323"
export WT_ROUTING_PUB_ADDR="tcp://127.0.0.1:34324"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_a_fast2040_fcfs"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export WT_SIMPLIFY_PREDICT=1
export WT_NATIVE_LOG_INTERVAL_MS=1000

echo "=== scheme4 native FCFS (oracle) @ trace A fast2040 1p1d GPU2+3 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
