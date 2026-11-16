#!/usr/bin/env bash
# [WT] 只跑 Ours(scheme4_gwf) + mysharegpt(ds3) 的调参启动器。
#
# 背景（已核实，2026-09-24）：
#   - 原启动器 run_rr_ours_pool32.sh 用 TRACE_NAME=tmp_dataset3，而该目录下同时有
#     lmsys 与 mysharegpt 两条 CSV；resolve_dataset_paths 用
#     `find $dir -maxdepth 1 \( -type f -o -type l \) -name "qwen_*_600s_strategyA.csv"`
#     会把两条都跑（所以上一批 4 个配置里前两个是 lmsys）。
#   - 本脚本改指向 derived_datasets/tmp_dataset3_ours_sharegpt/，该目录内只有一条
#     mysharegpt 的符号链接 → 只会跑 mysharegpt，且不复制数据。
#   - COMPARE_EXPERIMENT_VARIANTS 只留 scheme4_gwf → 不再跑 roundrobin。
#   - scheme4_gwf 变体会自动带 `--geometric-window-scheduler`（主脚本 434/1633-1634 行），
#     所以 WT_GWF_* 开关对 decode 实例生效。
#
# 可覆盖的环境变量：
#   WT_GWF_WINDOW_FLOOR   GWF 准入窗口下限，默认保持历史值 32；
#                          仅显式设置时才测试其它下限。
#   WT_GWF_FORCE_IDLE     1 = 关掉准入门控，回到 stock FCFS（"worst case == baseline" 消融）。
#   KV pool                固定 prefill/decode 均为 32GiB，与本轮基线相同。
#   RESULT_ROOT            结果根目录，默认与 RR/历史结果同根，便于对照。
#
# 用法（务必先确认没有残留进程，见文件末尾检查项）：
#   cd /root/predict-schedule/experiment/zrecent_use/0630
#   nohup ./run_ours_sharegpt.sh \
#     > /root/myshare/wty_tmp_result/retry_qwen_4p4d/nohup_ours_sharegpt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 消融对照（不改评价口径，只换调度逻辑）：
#   WT_GWF_FORCE_IDLE=1 nohup ./run_ours_sharegpt.sh > ... &
#   WT_GWF_WINDOW_FLOOR=512 nohup ./run_ours_sharegpt.sh > ... &
set -euo pipefail

cd "$(dirname "$0")"

: "${RESULT_ROOT:=/root/myshare/wty_tmp_result/retry_qwen_4p4d}"

# 显式落地 GWF 开关。
# 这两项必须传到最深处的 decode EngineCore（vllm serve → 子进程），所以用显式
# export，而不是只做命令前缀赋值，避免"以为传了其实没传"。
# 代码和本脚本均默认 32；调高下限只作为显式消融，不改评价配置。
: "${WT_GWF_WINDOW_FLOOR:=32}"
: "${WT_GWF_FORCE_IDLE:=0}"
export WT_GWF_WINDOW_FLOOR WT_GWF_FORCE_IDLE

echo "[WT][LAUNCHER] trace=tmp_dataset3_ours_sharegpt (仅 mysharegpt)"
echo "[WT][LAUNCHER] variants=scheme4_gwf only, result_root=${RESULT_ROOT}"
echo "[WT][LAUNCHER] WT_GWF_WINDOW_FLOOR=${WT_GWF_WINDOW_FLOOR} (代码默认 32)"
echo "[WT][LAUNCHER] WT_GWF_FORCE_IDLE=${WT_GWF_FORCE_IDLE}"
echo "[WT][LAUNCHER] tensor pool prefill=32 decode=32"

# [WT] 打开路由决策日志：本次改动动了 scheme4 fastpath 的门槛，必须能事后验证
# fastpath 是否真的命中、每个请求路由到哪个 decode 实例（失衡 1.75x 的排查依据）。
# 4080 请求约 <2MB，注意 /root/myshare 空间。
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
COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf" \
RUN_REPEAT_PER_CONFIG=1 \
WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE= \
SKIP_EXISTING_RESULT_DIRS= \
PREFILL_TENSOR_POOL_MEMORY=32 \
DECODE_TENSOR_POOL_MEMORY=32 \
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
BASE_RESULT_DIR="${RESULT_ROOT}" \
bash ./test_evaluation_strategyA_oracle_gwf.sh

# [WT] 启动前检查项（上一批第 4 个配置连挂 3 次就是因为没满足）：
#   1) ps -eo pid,stat,args | grep -E 'vllm|EngineCore|test_disagg_proxy' | grep -v grep  → 必须为空
#      （有 Zl [VLLM::EngineCor] <defunct> = 子进程已被 OOM kill，等不回来，要 kill -KILL 父进程）
#   2) nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
#      ⚠️ 关键是 util 必须为 0。memory.used=3MiB 是驱动残留（无害），
#         但 "memory_used_mib=3, util_percent=100" = 上一轮进程在 NCCL 自旋，
#         此时 GPU_IDLE_MAX_MEMORY_MIB=10 会放行 → 新实例起来就撞死。
#   3) free -g  → available 要 ≥ 300，否则 pool=32 会被 OOM kill
