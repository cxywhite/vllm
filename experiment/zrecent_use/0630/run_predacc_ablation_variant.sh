#!/usr/bin/env bash
# 精度→goodput 消融：同一预测器(GWF) 用不同采样参数档，β=1.0。
# 用法: bash run_predacc_ablation_variant.sh <tag> <variant>
#   tag: o1600|o2000
#   variant: natural|vB|global
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630
TAG="${1:?tag}"; VARIANT="${2:?variant}"

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1
export QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export PD_RATIO_LIST="4:4"
export TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=1
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_PREDICT_OUTPUT_SCALE=1.0
export WT_NATIVE_LOG_INTERVAL_MS=1000
export WT_GWF_WINDOW_TPOT_TARGET_MS=120 WT_GWF_TTFT_MARGIN_MS=250

case "$VARIANT" in
  natural) TRACE_NAME="predacc_lmsys_${TAG}" ;;
  vB)      TRACE_NAME="predacc_lmsys_${TAG}_vB" ;;
  global)  TRACE_NAME="predacc_lmsys_${TAG}_global" ;;
  *) echo "unknown variant: $VARIANT"; exit 1 ;;
esac
export TRACE_NAME
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_predacc_${VARIANT}_${TAG}"

echo "=== [${VARIANT}] ${TAG} (β=1.0, scheme4_gwf, 4p4d) trace=${TRACE_NAME} ==="
bash test_evaluation_scheme4_rr_compare.sh
echo "=== [${VARIANT}] ${TAG} done ==="
