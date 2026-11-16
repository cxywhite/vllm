#!/usr/bin/env bash
# [WT] 环境指纹：记录宿主竞争状态，用于证明"两轮实验是否处于同一窗口"。
#
# 为什么需要它（2026-09-24 已证实）：
#   同一台机器、同一份代码、同一份配置、同一数据集，在宿主被其他租户抢占时，
#   mean TPOT 会从 42~49ms 涨到 68~70ms，goodput 掉 3~10 倍。
#   因此**跨窗口比 goodput 是无效的**，必须能证明两轮在可比窗口内。
#
# 判据（越靠前越重要）：
#   compute_apps 为空      -> 没有其他租户占卡
#   loadavg 1min 越小越好  -> 08-27 良好窗口 < 10，09-24 拥挤窗口 28~31
#   mean TPOT (跑完后)     -> 最直接的"机器速度"代理指标，差 >10% 就不能直接比
#
# 用法（两个容器都能调，都写进同一共享卷）：
#   ./wt_env_fingerprint.sh <label>
# 例：
#   ./wt_env_fingerprint.sh dynamo_pre
#   ./wt_env_fingerprint.sh ours_post
set -uo pipefail

LABEL="${1:-unlabeled}"
OUTDIR="${2:-/root/myshare/wty_tmp_result/env_fingerprint}"
mkdir -p "$OUTDIR" 2>/dev/null

TS="$(date +%Y%m%d_%H%M%S)"
F="${OUTDIR}/${TS}_${LABEL}.txt"

{
    echo "=========== WT environment fingerprint ==========="
    echo "label        : ${LABEL}"
    echo "timestamp    : $(date '+%F %T %z')"
    echo "hostname     : $(hostname 2>/dev/null)"
    echo
    echo "--- 宿主负载（容器内 /proc/loadavg 即宿主；52 核） ---"
    cat /proc/loadavg
    echo "cpus_visible : $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo '?')"
    echo
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,clocks.sm,power.draw \
        --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    echo "compute_apps（不属于本容器的进程会显示 [Not Found]）:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null
    echo
    echo "--- /dev/shm（宿主机全局共享，可反映其他租户活动） ---"
    df -h /dev/shm 2>/dev/null | tail -1
    echo "最近写入的 9 个条目（uid 1007 等非本租户内容 = 别人在跑）:"
    ls -lt --time-style=+%F_%T /dev/shm 2>/dev/null | head -10
    echo
    echo "--- 内存 ---"
    grep -E 'MemTotal|MemAvailable|Shmem' /proc/meminfo 2>/dev/null
} > "$F" 2>&1

echo "[WT] fingerprint -> ${F}"
cat "$F"
