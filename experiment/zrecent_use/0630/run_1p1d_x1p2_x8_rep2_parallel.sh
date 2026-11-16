#!/usr/bin/env bash
# 阶段A甜点 ×1.2×8 第2次重复（oracle），并行跑在 GPU 1+5（阶段B占用 GPU 0+4）
# 与 run_1p1d_x1p2_x8_ab.sh 唯一区别：GPU 1+5 + 端口整体偏移(避免与阶段B冲突)
# 端口偏移 +100；ZMQ 端口 32323/32324 -> 32423/32424（已补丁支持 env）
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
# 端口整体 +100 偏移（阶段B用 22001/22011/22020/22032/22007/28002）
export PD_1P1D_PREFILL_PORTS=22101 PD_1P1D_PREFILL_KV_PORTS=22111
export PD_1P1D_DECODE_PORTS=22120 PD_1P1D_DECODE_KV_PORTS=22132
export BENCH_PORT=22107 PROXY_PORT=28102
# ZMQ 端口偏移（阶段B用 32323/32324）
export WT_PREDICTOR_ZMQ_ADDR="tcp://127.0.0.1:32423"
export WT_ROUTING_PUB_ADDR="tcp://127.0.0.1:32424"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf scheme4_uniboost"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_1p1d_x1p2_x8_ab_rep2"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export WT_SIMPLIFY_PREDICT=1
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_UNIBOOST_CLOCK=wall WT_UNIBOOST_GAMMA0_S=0.1 WT_UNIBOOST_MAX_JUMP_S=60 WT_UNIBOOST_LOG_INTERVAL_MS=1000 WT_UNIBOOST_ADMIT_PREEMPT=0
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250
bash test_evaluation_scheme4_rr_compare.sh
