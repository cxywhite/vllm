#!/usr/bin/env bash
# [WT] 等 Dynamo 释放 8 卡后，立刻在 qwentrace(dataset3) 上补跑 RR + Ours。
#
# 背景（为什么必须现在跑）：
#   2026-09-25 发现 Dynamo 在**同一批次、同一配置**下，8 小时内 goodput 差异极大：
#     lmsys    4p4d:  64.69 (03:08) -> 853.53  (12:40)   13.2x
#     lmsys    2p6d:  71.32 (04:26) -> 2852.45 (13:30)   40.0x
#     sharegpt 4p4d: 101.72 (03:48) -> 498.13  (13:11)    4.9x
#   while throughput 只变 1.2~1.4x、mean TPOT 只变 1.5~1.7x。
#   原因：TPOT-SLO=30ms 低于可达 TPOT 下限，达标率由分布左尾决定，弹性 ~6.9。
#   => 现在这个窗口的 Dynamo lmsys 2p6d 已超过历史任何一次，是**唯一能拿到有效三方对比的时机**。
#      我们 09-24 夜那批（Ours 312 / RR 300 @ lmsys 4p4d）明显是在坏窗口里跑的（约为安静期 0.49x）。
#
# 本脚本做 4 轮，每轮跑 scheme4_gwf + roundrobin：
#   1. 4:4 + qwentrace-lmsys
#   2. 4:4 + qwentrace-sharegpt
#   3. 2:6 + qwentrace-lmsys
#   4. 2:6 + qwentrace-sharegpt
# 合计 8 次运行，预计 4~5 小时。
#
# 设计要点：
#   1. **完全不修改** test_evaluation_strategyA_oracle_gwf.sh —— 全部靠环境变量传参。
#   2. 数据集用"只含一条符号链接"的目录（tmp_dataset3_ours_lmsys / _sharegpt），
#      避免 resolve_dataset_paths 一次把 lmsys + sharegpt 都跑掉。
#   3. 代码指纹守卫：启动前校验调度器两个文件与"跑出 172 那次"字节一致。
#   4. GPU 空闲守卫：**会自动等到 Dynamo 跑完**（外部占用时会一直轮询）。
#   5. 每组前后各记一次环境指纹（wt_env_fingerprint.sh），便于事后判断窗口可比性。
#   6. 一组失败不影响后面几组（不用 set -e），最后打印汇总 + 与 Dynamo 并列。
#
# 用法：
#   cd /root/predict-schedule/experiment/zrecent_use/0630
#   nohup ./run_qwentrace_rr_ours_after_dynamo.sh \
#     > /root/myshare/wty_tmp_result/qwentrace_goodwin/nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 只跑部分轮次（例如赶时间只跑 2p6d 两组）：
#   WT_ROUNDS="2:6|tmp_dataset3_ours_lmsys|trace3-lmsys-2p6d;2:6|tmp_dataset3_ours_sharegpt|trace3-sharegpt-2p6d" \
#     nohup ./run_qwentrace_rr_ours_after_dynamo.sh > sh.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")"

: "${RESULT_ROOT:=/root/myshare/wty_tmp_result/qwentrace_goodwin}"
mkdir -p "$RESULT_ROOT"

# [WT] GWF 开关：与跑出 172 那次一致（代码默认 32 / 0）
: "${WT_GWF_WINDOW_FLOOR:=32}"
: "${WT_GWF_FORCE_IDLE:=0}"
export WT_GWF_WINDOW_FLOOR WT_GWF_FORCE_IDLE

# ---------------------------------------------------------------------------
# [WT] 代码指纹守卫
# ---------------------------------------------------------------------------
# 必须与 "2026-09-24 12:32 跑出 goodput 172.23" 那次一致：
#   vllm/v1/core/sched/geometric_window_scheduler.py     <- WT_GWF_WINDOW_CAP=256 + TPOT 目标 120ms
#   disagg_serving_suite/test_disagg_proxy.py <- fastpath 门 WT_ROUTE_IDLE_TPOT_MS(120)
EXPECT_SCHED="ecc3a294f6ad9d88f4a8c75a56864660"
EXPECT_PROXY="b01c644ae0c8bb47d6fd468fd28d063f"
SCHED_FILE=/root/predict-schedule/vllm/vllm/v1/core/sched/geometric_window_scheduler.py
PROXY_FILE=/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py
ACT_SCHED="$(sha256sum "$SCHED_FILE" 2>/dev/null | cut -c1-32)"
ACT_PROXY="$(sha256sum "$PROXY_FILE" 2>/dev/null | cut -c1-32)"

echo "============================================================"
echo "[WT][GUARD] geometric_window_scheduler.py        sha256 = ${ACT_SCHED}  (期望 ${EXPECT_SCHED})"
echo "[WT][GUARD] test_disagg_proxy.py     sha256 = ${ACT_PROXY}  (期望 ${EXPECT_PROXY})"
if [[ "$ACT_SCHED" != "$EXPECT_SCHED" || "$ACT_PROXY" != "$EXPECT_PROXY" ]]; then
    echo "[WT][GUARD] ❌ 调度器代码与跑出 172 那次不一致，中止（避免跑出不可比的数）"
    exit 2
fi
echo "[WT][GUARD] ✅ 调度器代码与 172 那轮字节一致"
echo "============================================================"

# ---------------------------------------------------------------------------
# [WT] GPU 空闲守卫 —— 会等到 Dynamo 跑完
# ---------------------------------------------------------------------------
# 这台机器有其他租户/Dynamo 容器会占用全部 8 卡。4p4d 与 2p6d 都需要 8 张卡全空。
# 这里**主动等待**，而不是让评测脚本超时后把整轮判为 skipped_gpu_busy。
wait_for_gpu_idle() {
    local label="$1"
    local timeout="${WT_GPU_WAIT_TIMEOUT_S:-28800}"   # 默认 8 小时
    local waited=0
    local n
    while (( waited < timeout )); do
        n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
             | sed '/^[[:space:]]*$/d' | wc -l)"
        if (( n == 0 )); then
            echo "[WT][WAIT] ${label}: GPU 已全部空闲（已等 ${waited}s）"
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
# 轮次配置：pd_ratio | TRACE_NAME | 标签
# ---------------------------------------------------------------------------
if [[ -n "${WT_ROUNDS:-}" ]]; then
    IFS=';' read -ra ROUNDS <<< "$WT_ROUNDS"
else
    ROUNDS=(
        "4:4|tmp_dataset3_ours_lmsys|trace3-lmsys-4p4d"
        "4:4|tmp_dataset3_ours_sharegpt|trace3-sharegpt-4p4d"
        "2:6|tmp_dataset3_ours_lmsys|trace3-lmsys-2p6d"
        "2:6|tmp_dataset3_ours_sharegpt|trace3-sharegpt-2p6d"
    )
fi

echo
echo "============================================================"
echo "[WT] 计划轮次（每轮 = scheme4_gwf + roundrobin 两次运行）"
printf '      %s\n' "${ROUNDS[@]}"
echo "[WT] 结果根目录: ${RESULT_ROOT}"
echo "============================================================"

FAILED=()

for entry in "${ROUNDS[@]}"; do
    IFS='|' read -r pd_ratio trace_name label <<< "$entry"

    echo
    echo "############################################################"
    echo "[WT] ROUND: ${label}   pd_ratio=${pd_ratio}   trace=${trace_name}"
    echo "     variants = scheme4_gwf + roundrobin（脚本内部串行）"
    echo "     $(date --iso-8601=seconds)"
    echo "############################################################"

    if ! wait_for_gpu_idle "round ${label}"; then
        FAILED+=("${label}:gpu_busy")
        continue
    fi

    ./wt_env_fingerprint.sh "pre_${label}" || true

    # ⚠️ 注释必须在下面 `VAR=val \` 续行链【之外】：链中插 `#` 会断开 env 前缀。
    #    以下 env 与 run_dataset3_lmsys_and_2p6d.sh（09-24 那批）逐项一致，仅 result root 不同。
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
# 汇总：本批结果 + Dynamo 同窗口参考
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "[WT] 汇总（从各 run 的 bench 日志读取）"
echo "============================================================"
for entry in "${ROUNDS[@]}"; do
    IFS='|' read -r pd_ratio trace_name label <<< "$entry"
    pd_tag="$(printf '%s' "$pd_ratio" | tr ':' 'p')d"   # 4:4 -> 4p4d
    echo
    echo "--- ${label} (${pd_tag}) ---"
    for variant in scheme4_gwf roundrobin; do
        # 目录名形如 qwen_qwen-lmsys-chat-<trace>_n4080_pd<pd_tag>_tpot30ms_trace<trace>_rep1_<ts>_<variant>
        # 注意 pd<pd_tag> 出现在 trace<trace> 之前，glob 顺序不能反。
        d="$(ls -dt "${RESULT_ROOT}"/*"pd${pd_tag}"*"trace${trace_name}"*"${variant}"* 2>/dev/null | head -1)"
        if [[ -n "$d" && -d "$d" ]]; then
            printf '  %-14s %s\n' "$variant" "$(basename "$d" | tail -c 40)"
            grep -ahE '^Successful requests|^Benchmark duration|^Total generated tokens|^Output token goodput|^Output token throughput|^Mean TTFT|^Mean TPOT' \
                "$d"/log/bench_*.log 2>/dev/null | sed 's/^/      /'
            # 同时给出 token 加权的达标率拆解（表头的 mean/median 看不出左尾）
            python3 ./wt_analyze_run.py "$d" 2>/dev/null | sed -n '/达标/,$p' | sed 's/^/      /'
        else
            echo "  ${variant}: 未找到结果目录"
        fi
    done
done

echo
echo "============================================================"
echo "[WT] Dynamo 同窗口参考值（2026-09-25 12:38 批次，retry3）"
echo "============================================================"
DYNAMO_ROOT=/root/myshare/tmp_dynamo_result/retry3/run_20260925_123801
for d in "$DYNAMO_ROOT"/*_dynamo/; do
    [[ -d "$d" ]] || continue
    n="$(basename "$d")"
    src=$([[ "${n#*lmsys}" != "$n" ]] && echo lmsys || echo sharegpt)
    pd="$(printf '%s' "$n" | grep -oE 'pd[0-9]p[0-9]d')"
    gp="$(grep -m1 '^Output token goodput (tok/s):' "$d/benchmark.log" 2>/dev/null | awk '{print $NF}')"
    th="$(grep -m1 '^Output token throughput (tok/s):' "$d/benchmark.log" 2>/dev/null | awk '{print $NF}')"
    tp="$(grep -m1 '^Mean TPOT (ms):' "$d/benchmark.log" 2>/dev/null | awk '{print $NF}')"
    if [[ -n "$gp" ]]; then
        printf '  %-10s %-7s goodput=%-9s throughput=%-9s TPOT均=%s\n' "$src" "$pd" "$gp" "$th" "$tp"
    else
        printf '  %-10s %-7s (未完成)\n' "$src" "$pd"
    fi
done

echo
if (( ${#FAILED[@]} == 0 )); then
    echo "[WT] 全部完成 ✅"
else
    echo "[WT] 有失败项（${#FAILED[@]}）: ${FAILED[*]}"
fi
echo "结果根目录: ${RESULT_ROOT}"
echo "结束时间: $(date --iso-8601=seconds)"
