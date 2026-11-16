#!/usr/bin/env bash
# 1p1d ×1.2×8 基于预测 GWF（真实 BERT + β 校准），第三臂。
# 与 GWF oracle 144.17 同 trace、同 θ=120/margin=250；区别：真实 BERT 预测 + β 校准。
# GPU 2+3，端口全部 +2000（避开 22xxx FCFS / 23xxx KV-threshold），不碰另外两臂。
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
export PD_1P1D_PREFILL_GPUS=2 PD_1P1D_DECODE_GPUS=3
export PD_1P1D_PREFILL_PORTS=24001 PD_1P1D_PREFILL_KV_PORTS=24011
export PD_1P1D_DECODE_PORTS=24020 PD_1P1D_DECODE_KV_PORTS=24032
export PROXY_PORT=30002
export BENCH_PORT=24007
# 预测相关端口也要错开（否则三个 proxy 都绑默认 32323/32324 会撞）
export WT_PREDICTOR_ZMQ_ADDR="tcp://127.0.0.1:34323"
export WT_PREDICT_PROXY_ZMQ_ADDR="tcp://127.0.0.1:34323"
export WT_ROUTING_PUB_ADDR="tcp://127.0.0.1:34324"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_1p1d_x1p2_x8_gwf_real"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
# 真实 BERT 预测 + β 校准（0.04 ≈ 1/24，抵消 o2000 合成 trace 的 ~24× 过预测）。
# proxy 会把 β 校准后的 predict_output_mean 注入，decode 侧直接读它，无需 WT_PRED_CORRECTION
#（后者是在线逐桶修正，之前叠加导致挂死）。
export WT_SIMPLIFY_PREDICT=0
export WT_PREDICT_OUTPUT_SCALE=0.04
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250
export WT_GWF_LOG_INTERVAL_MS=1000

echo "=== scheme4_gwf (real BERT, β=0.04, margin=250) @ ×1.2×8 1p1d GPU2+3 ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== done ==="
