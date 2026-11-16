#!/usr/bin/env bash
# [WT] 串行跑 3 组配置，每组都跑 scheme4_gwf + roundrobin。
#
#   第 1 组：4:4  + dataset3-lmsys      ← 与"跑出 172 那次"完全相同的配置，只换数据集
#   第 2 组：2:6  + dataset3-sharegpt
#   第 3 组：2:6  + dataset3-lmsys
#
# 设计要点：
#   1. **完全不修改** test_evaluation_strategyA_oracle_gwf.sh —— 全部靠环境变量传参。
#      已核实：2p6d 的历史配置与 4p4d 逐项相同，**只有 pd_ratio 不同**
#      （见 1tmp/qwen_qwen-lmsys-chat-strategyA_aggressive_2p6d_..._20260726_023858 的 config）。
#   2. 数据集用"只含一条符号链接"的目录（tmp_dataset3_ours_lmsys / _sharegpt），
#      避免 resolve_dataset_paths 一次把 lmsys + sharegpt 都跑掉。
#   3. 代码指纹守卫：启动前校验调度器两个文件与"跑出 172 那次"字节一致。
#   4. 每组前后各记一次环境指纹（wt_env_fingerprint.sh），便于事后判断窗口可比性。
#   5. 一组失败不影响后面几组（不用 set -e），最后打印汇总。
#
# ETA：6 次运行 x 约 40~50 分钟 ≈ 4~5 小时。
#
# 用法：
#   cd /root/predict-schedule/experiment/zrecent_use/0630
#   nohup ./run_dataset3_lmsys_and_2p6d.sh \
#     > /root/myshare/wty_tmp_result/retry_qwen_4p4d/nohup_3rounds_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")"

: "${RESULT_ROOT:=/root/myshare/wty_tmp_result/retry_qwen_4p4d}"

# [WT] GWF 开关：与跑出 172 那次一致（代码默认 32 / 0）
: "${WT_GWF_WINDOW_FLOOR:=32}"
: "${WT_GWF_FORCE_IDLE:=0}"
export WT_GWF_WINDOW_FLOOR WT_GWF_FORCE_IDLE

# ---------------------------------------------------------------------------
# [WT] 代码指纹守卫
# ---------------------------------------------------------------------------
# 必须与 "2026-09-24 12:32 跑出 goodput 172.23" 那次的调度器代码一致：
#   vllm/v1/core/sched/geometric_window_scheduler.py  <- WT_GWF_WINDOW_CAP=256 + TPOT 目标 120ms
#   disagg_serving_suite/test_disagg_proxy.py <- fastpath 门 WT_ROUTE_IDLE_TPOT_MS(120)
EXPECT_SCHED="ecc3a294f6ad9d88f4a8c75a56864660"
EXPECT_PROXY="b01c644ae0c8bb47d6fd468fd28d063f"
SCHED_FILE=/root/predict-schedule/vllm/vllm/v1/core/sched/geometric_window_scheduler.py
PROXY_FILE=/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py
ACT_SCHED="$(sha256sum "$SCHED_FILE" 2>/dev/null | cut -c1-32)"
ACT_PROXY="$(sha256sum "$PROXY_FILE" 2>/dev/null | cut -c1-32)"

echo "============================================================"
echo "[WT][GUARD] geometric_window_scheduler.py sha256 = ${ACT_SCHED}  (期望 ${EXPECT_SCHED})"
echo "[WT][GUARD] test_disagg_proxy.py sha256 = ${ACT_PROXY}  (期望 ${EXPECT_PROXY})"
if [[ "$ACT_SCHED" != "$EXPECT_SCHED" || "$ACT_PROXY" != "$EXPECT_PROXY" ]]; then
    echo "[WT][GUARD] ❌ 调度器代码与跑出 172 那次不一致，中止（避免跑出不可比的数）"
    exit 2
fi
echo "[WT][GUARD] ✅ 调度器代码与 172 那轮字节一致"
echo "============================================================"

# ---------------------------------------------------------------------------
# [WT] GPU 空闲守卫
# ---------------------------------------------------------------------------
# 这台机器有其他租户会**间歇性**占用 GPU0（实测 8.9~9.5 GB、util 36~77%，
# nvidia-smi 里显示为 [Not Found] = 不属于任何容器）。4p4d 与 2p6d 都需要 8 张卡全空。
# 该作业的模式：常在我们实验结束后约 20 分钟启动，持续时长不定。
# 所以这里**主动等待**，而不是让评测脚本 120s 超时后把整轮判为 skipped_gpu_busy。
# 超时可用 WT_GPU_WAIT_TIMEOUT_S 调整（默认 6 小时）。
wait_for_gpu_idle() {
    local label="$1"
    local timeout="${WT_GPU_WAIT_TIMEOUT_S:-21600}"
    local waited=0
    local n
    while (( waited < timeout )); do
        n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
             | sed '/^[[:space:]]*$/d' | wc -l)"
        if (( n == 0 )); then
            echo "[WT][WAIT] ${label}: GPU 已全部空闲（等待 ${waited}s）"
            return 0
        fi
        if (( waited % 300 == 0 )); then
            echo "[WT][WAIT] ${label}: 仍有 ${n} 个 GPU 计算进程，已等 ${waited}s / ${timeout}s"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
                --format=csv,noheader 2>/dev/null | sed 's/^/      /'
        fi
        sleep 30
        waited=$(( waited + 30 ))
    done
    echo "[WT][WAIT] ${label}: ❌ 等待 ${timeout}s 后 GPU 仍被占用"
    return 1
}

# ---------------------------------------------------------------------------
# 三轮配置：pd_ratio | TRACE_NAME | 标签
# ---------------------------------------------------------------------------
ROUNDS=(
    "4:4|tmp_dataset3_ours_lmsys|dataset3-lmsys-4p4d"
    "2:6|tmp_dataset3_ours_sharegpt|dataset3-sharegpt-2p6d"
    "2:6|tmp_dataset3_ours_lmsys|dataset3-lmsys-2p6d"
)

FAILED=()

for entry in "${ROUNDS[@]}"; do
    IFS='|' read -r pd_ratio trace_name label <<< "$entry"

    echo
    echo "############################################################"
    echo "[WT] ROUND: ${label}   pd_ratio=${pd_ratio}   trace=${trace_name}"
    echo "     variants = scheme4_gwf + roundrobin（脚本内部串行）"
    echo "############################################################"

    if ! wait_for_gpu_idle "round ${label}"; then
        FAILED+=("${label}:gpu_busy")
        continue
    fi

    ./wt_env_fingerprint.sh "pre_${label}" || true

    # ⚠️ 注释必须在下面 `VAR=val \` 续行链【之外】：链中插 `#` 会断开 env 前缀。
    #    以下 env 与 run_ours_sharegpt.sh（跑出 172 那次）逐项一致，仅 pd/trace 不同。
    MODEL_TYPE=qwen \
    QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct \
    DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets \
    TRACE_NAME="${trace_name}" \
    PD_RATIO_LIST="${pd_ratio}" \
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
    rc=$?

    ./wt_env_fingerprint.sh "post_${label}" || true

    if [[ $rc -ne 0 ]]; then
        echo "[WT] ❌ ROUND ${label} 退出码 ${rc}"
        FAILED+=("${label}:rc=${rc}")
    else
        echo "[WT] ✅ ROUND ${label} 完成"
    fi
done

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "[WT] 三轮汇总（从各 run 的 bench 日志读取）"
echo "============================================================"
for entry in "${ROUNDS[@]}"; do
    IFS='|' read -r pd_ratio trace_name label <<< "$entry"
    pd_tag="$(printf '%s' "$pd_ratio" | tr ':' 'p')d"   # 4:4 -> 4p4d
    echo
    echo "--- ${label} (${pd_tag}) ---"
    for variant in scheme4_gwf roundrobin; do
        # 实际目录名形如：
        #   qwen_qwen-mysharegpt-<trace>_n4080_pd<pd_tag>_tpot30ms_trace<trace>_rep1_<ts>_<variant>
        # 注意 pd<pd_tag> 出现在 trace<trace> 之前，glob 顺序不能反。
        d="$(ls -dt "${RESULT_ROOT}"/*"pd${pd_tag}"*"trace${trace_name}"*"${variant}"* 2>/dev/null | head -1)"
        if [[ -n "$d" && -d "$d" ]]; then
            printf '  %-14s %s\n' "$variant" "$(basename "$d" | tail -c 40)"
            grep -ahE '^Successful requests|^Benchmark duration|^Total generated tokens|^Output token goodput|^Output token throughput|^Mean TPOT|^Median TPOT' \
                "$d"/log/bench_*.log 2>/dev/null | sed 's/^/      /'
        else
            echo "  ${variant}: 未找到结果目录"
        fi
    done
done

echo
if (( ${#FAILED[@]} == 0 )); then
    echo "[WT] 全部完成 ✅"
else
    echo "[WT] 有失败项（${#FAILED[@]}）: ${FAILED[*]}"
fi
echo "结果根目录: ${RESULT_ROOT}"
