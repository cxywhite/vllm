#!/usr/bin/env bash
# [WT] 资源采样器：在实验运行期间每 N 秒记录一次宿主/GPU 状态，用来回答
#      "为什么今天比 08-27/08-29 慢 1.5~1.8 倍"。
#
# 背景：已证实同代码、同配置、同数据集下，2026-09-24 比 08-27/08-29 慢约 1.5~1.8×
#   （RR median TPOT 48.14→65.35ms；Dynamo mean TPOT 42.44→70.27ms）。
#   已排除：我们的 proxy / GWF 代码 / vllm 代码 / 配置 / CPU 配额 / /dev/shm 泄漏。
#   仍需在**真实负载下**判定（空闲时采样不可信）：
#     H1 GPU 降频：空闲时 SM=1410MHz=最大、throttle 全 Not Active，但满载时可能 SW Power Cap
#     H2 外部进程：占用我们 GPU 的 SM（注意 490MiB+util 0% 的那种不偷算力，9.4GB+util 64% 的才偷）
#     H3 宿主 CPU/DRAM：影响 --enforce-eager 下每步 CPU 开销
#
# 用法（实验前启动，实验后 kill）：
#   ./wt_resource_watch.sh <label> &
#   ...
#   pkill -f wt_resource_watch.sh
#
# 输出：/root/myshare/wty_tmp_result/resource_watch/<label>_<ts>.csv
#   列：ts, load1, load5, mem_avail_gb, gpu{0..7}_mem, gpu{0..7}_util,
#       gpu0_sm_mhz, gpu0_power_w, gpu0_temp, gpu0_throttle, foreign_apps
set -uo pipefail

LABEL="${1:-run}"
INTERVAL="${2:-10}"
OUTDIR="${3:-/root/myshare/wty_tmp_result/resource_watch}"
mkdir -p "$OUTDIR" 2>/dev/null

TS="$(date +%Y%m%d_%H%M%S)"
CSV="${OUTDIR}/${LABEL}_${TS}.csv"

{
    echo -n "ts"
    echo -n ",load1,load5,mem_avail_gb"
    echo -n ",gpu0_sm_mhz,gpu0_power_w,gpu0_temp_c,gpu0_util,gpu0_mem_mib"
    echo -n ",gpu0_throttle,gpu1_util,gpu2_util,gpu3_util,gpu4_util,gpu5_util,gpu6_util,gpu7_util"
    echo ",applications"
} > "$CSV"

echo "[WT] resource watcher -> ${CSV} (interval=${INTERVAL}s)"

while true; do
    {
        date '+%F_%T'
    } | tr -d '\n' >> "$CSV"

    LOAD="$(cut -d' ' -f1,2 /proc/loadavg)"
    printf ',%s,%s,%s' \
        "$(echo "$LOAD" | cut -d' ' -f1)" \
        "$(echo "$LOAD" | cut -d' ' -f2)" \
        "$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)" >> "$CSV"

    # GPU0 详细（clock/power/temp/util/mem）+ 降频原因
    nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' | sed 's/^/,/' >> "$CSV"
    THR="$(nvidia-smi -q -d PERFORMANCE -i 0 2>/dev/null \
        | awk '/Clocks Event Reasons/{f=1} f&&/Active/{gsub(/^[ \t]+/,"");sub(/ *: *Active/,"");printf "%s;",$0} f&&/^$/{exit}')"
    printf ',%s' "${THR:--}" >> "$CSV"

    # 其余 7 卡 util
    for i in 1 2 3 4 5 6 7; do
        U="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$i" 2>/dev/null)"
        printf ',%s' "${U:-?}" >> "$CSV"
    done

    # 计算进程：[Not Found] = 不属于本容器 = 外部租户
    APPS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
        | tr '\n' ';' | tr -d ',')"
    printf ',"%s"\n' "${APPS:--}" >> "$CSV"

    sleep "$INTERVAL"
done
