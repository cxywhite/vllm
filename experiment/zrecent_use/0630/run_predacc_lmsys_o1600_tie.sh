#!/usr/bin/env bash
# predacc_lmsys_o1600：TIE 基线（scheme4_tie），4p4d Qwen。
# 模式由 WT_TIE_MODE 决定：
#   oracle_len = 用 request.max_tokens(=真实长度) 排序，即 SSJF-with-perfect-length，
#                零预测器依赖、零崩溃风险，先用来验证 pipeline 不丢请求。
#   pred       = 用 DeBERTa 预测 (mu,sigma) 算完整 TIE 分数（需 WT_TIE_CKPT/STATS/ENCODER）。
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
export COMPARE_EXPERIMENT_VARIANTS="scheme4_tie"
export RUN_REPEAT_PER_CONFIG=1
export BASE_RESULT_DIR="/root/predict-schedule/experiment/zrecent_use/0630/wl_predacc_lmsys_o1600_tie"
export PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4
export PD_4P4D_PREFILL_GPUS=0,1,2,3 PD_4P4D_DECODE_GPUS=4,5,6,7
export WT_PREDICT_OUTPUT_SCALE=0.30
export WT_NATIVE_LOG_INTERVAL_MS=1000

# TIE 调度器（decode 侧注入）
export WT_TIE_MODE=pred
# result suffix distinguishes this predictor run
export TIE_RESULT_SUFFIX=tie_qwen10k
export WT_TIE_CKPT=/root/myshare/tie_train/qwen10k_deberta/qwen10k_logt_labels_deberta-v3-base/best_model.pth
export WT_TIE_STATS=/root/myshare/tie_train/qwen10k_deberta/qwen10k_logt_labels_deberta-v3-base/normalize_stats.json
export WT_TIE_ENCODER=/root/myshare/models/deberta-v3-base
export WT_TIE_DEVICE=cpu
export WT_TIE_BETA=0.1
export WT_TIE_MAX_TOKENS=8192
export WT_TIE_LOG_INTERVAL_MS=1000

bash test_evaluation_scheme4_rr_compare.sh
