#!/usr/bin/env bash
# 提取一次运行的最终指标，供 AI 自动分析使用。
# 用法: bash analyze_run.sh <result_dir>
# 输出: benchmark 末尾的 goodput / SLO / 吞吐 汇总 + recompute / stale_drop 计数。
set -uo pipefail
dir="${1:?用法: bash analyze_run.sh <result_dir>}"
bench=$(find "$dir" -name 'bench_trace*.log' 2>/dev/null | head -1)
if [[ -z "$bench" ]]; then
  echo "[analyze] 无 bench_trace 日志: $dir"
  exit 1
fi

echo "=== RESULT_DIR: $dir ==="
# 最终汇总（benchmark 末尾打印的关键指标）
tr '\r' '\n' < "$bench" 2>/dev/null | grep -E \
  "Successful requests|Failed requests|Request throughput|Request goodput|Output token goodput|Output token throughput|Total token goodput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|P99 ITL" \
  | tail -25

echo "--- 其它计数 ---"
echo "stale_drop 文件数: $(grep -rl 'Dropping stale deferred KV tensor' "$dir" 2>/dev/null | wc -l)"
# recompute：从各实例日志里抓最大 recompute 计数（GWF/UniBoost 打印格式不一，尽力而为）
rc_max=$(grep -rhoE 'recompute[:= ]+[0-9]+' "$dir"/log/*.log 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
echo "recompute 最大计数: ${rc_max:-未找到}"
