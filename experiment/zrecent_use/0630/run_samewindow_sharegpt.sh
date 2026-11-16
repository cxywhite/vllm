#!/usr/bin/env bash
# [WT] "同窗口"启动器：Ours(scheme4_gwf) + RR 背靠背跑，只用 mysharegpt。
#
# 目的（2026-09-24）：Dynamo 在 14:37-15:21 跑完，要在**紧邻的同一窗口**补跑
# Ours 与 RR，构成三方同窗口对比。因为已证实跨窗口比 goodput 无效
# （宿主被抢占时 mean TPOT 从 42→70ms，goodput 掉 3~10 倍）。
#
# 与 run_ours_sharegpt.sh 的唯一区别：
#   COMPARE_EXPERIMENT_VARIANTS = "scheme4_gwf roundrobin"（脚本内部串行，不并行）。
#   顺序把 scheme4_gwf 放前面 —— 离 Dynamo 那轮时间最近，配对最紧。
#   trace 仍用 tmp_dataset3_ours_sharegpt（只含 mysharegpt 一条符号链接），
#   避免 resolve_dataset_paths 把 lmsys 也跑进去。
#
# 环境指纹：实验前后各记一次，用来证明窗口可比（见 wt_env_fingerprint.sh）。
#
# 用法：
#   cd /root/predict-schedule/experiment/zrecent_use/0630
#   nohup ./run_samewindow_sharegpt.sh \
#     > /root/myshare/wty_tmp_result/retry_qwen_4p4d/nohup_samewindow_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")"

: "${RESULT_ROOT:=/root/myshare/wty_tmp_result/retry_qwen_4p4d}"

# [WT] 环境指纹：实验前
./wt_env_fingerprint.sh pre_samewindow || true

# [WT] 环境指纹：实验后（无论成功失败都记，EXIT 里调用；cleanup 不 exit）
trap './wt_env_fingerprint.sh post_samewindow || true' EXIT

# 显式落地 GWF 开关（同 run_ours_sharegpt.sh，保证两轮口径一致）
: "${WT_GWF_WINDOW_FLOOR:=32}"
: "${WT_GWF_FORCE_IDLE:=0}"
export WT_GWF_WINDOW_FLOOR WT_GWF_FORCE_IDLE

echo "[WT][LAUNCHER] trace=tmp_dataset3_ours_sharegpt (仅 mysharegpt)"
echo "[WT][LAUNCHER] variants=scheme4_gwf + roundrobin（串行，同窗口配对）"
echo "[WT][LAUNCHER] result_root=${RESULT_ROOT}"
echo "[WT][LAUNCHER] WT_GWF_WINDOW_FLOOR=${WT_GWF_WINDOW_FLOOR} (代码默认 32)"
echo "[WT][LAUNCHER] WT_GWF_FORCE_IDLE=${WT_GWF_FORCE_IDLE}"
echo "[WT][LAUNCHER] tensor pool prefill=32 decode=32"
echo "[WT][LAUNCHER] 本轮改动：① GWF cap=256(WT_GWF_WINDOW_CAP) ② fastpath 门=120ms(WT_ROUTE_IDLE_TPOT_MS)"

# ⚠️ 这些注释必须在下面 `VAR=val \` 续行链【之外】：在链中间插 `#` 会把 env
#    前缀断开，导致 GPU_IDLE_MAX_MEMORY_MIB 等被丢掉（已踩过一次）。
MODEL_TYPE=qwen \
QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct \
DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets \
TRACE_NAME=tmp_dataset3_ours_sharegpt \
PD_RATIO_LIST=4:4 \
TPOT_MS_LIST=30 \
BENCH_GOODPUT_TTFT_MS=1000 \
GPU_IDLE_MAX_MEMORY_MIB=10 \
WT_ROUTE_DECISION_LOG_ENABLED=1 \
COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf roundrobin" \
RUN_REPEAT_PER_CONFIG=1 \
WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE= \
SKIP_EXISTING_RESULT_DIRS= \
PREFILL_TENSOR_POOL_MEMORY=32 \
DECODE_TENSOR_POOL_MEMORY=32 \
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
BASE_RESULT_DIR="${RESULT_ROOT}" \
bash ./test_evaluation_strategyA_oracle_gwf.sh

# [WT] 启动前检查项：
#   1) pgrep -af 'vllm|EngineCore|test_disagg_proxy'  → 必须为空
#   2) nvidia-smi --query-compute-apps=...            → 必须为空（[Not Found] = 别的租户占卡）
#   3) cat /proc/loadavg                              → 08-27 良好窗口 <10；09-24 拥挤窗口 28~31
