#!/usr/bin/env bash
# [WT] 安全停止 test_evaluation_strategyA_oracle_gwf.sh 批次。
#
# 用法：
#   ./stop_batch.sh                 # 停止所有 test_evaluation_strategyA_oracle_gwf.sh 批次
#   ./stop_batch.sh <脚本名或关键字>  # 停止匹配其它脚本名的批次，例如 test_evaluation_scheme4_rr_compare.sh
#   ./stop_batch.sh --dry-run       # 只列出将要终止的进程，不做任何终止
#
# 为什么不能只 kill 父进程：
#   主脚本里是 `trap cleanup INT TERM EXIT`（第 1839 行），但 cleanup() 不会 exit ——
#   发 TERM 只会触发一次清理，脚本随后会继续跑下一个配置并重新拉起 8 个服务。
#   所以必须先 SIGKILL 父进程（不给它继续执行的机会），再终止它派生的子进程组。
#
# 子进程（proxy / 8 个 vllm / benchmark）都是用 setsid 启动的，各自独立进程组（PGID == PID），
# 因此对子进程要按“进程组”终止：kill -TERM -- -<PGID>。

set -uo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi
PATTERN="${1:-test_evaluation_strategyA_oracle_gwf.sh}"

# 只挑真正在跑该脚本的 bash，避免匹配到 grep/编辑器等
mapfile -t PARENTS < <(
    pgrep -f "$PATTERN" 2>/dev/null | while read -r p; do
        cmd="$(ps -o args= -p "$p" 2>/dev/null)"
        [ -n "$cmd" ] || continue
        case "$cmd" in
            *bash*"$PATTERN"*) echo "$p" ;;
        esac
    done
)

if [ "${#PARENTS[@]}" -eq 0 ]; then
    echo "[stop_batch] 未找到运行中的批次（pattern=$PATTERN）"
    exit 0
fi

declare -a ALL_KIDS=()
for pid in "${PARENTS[@]}"; do
    echo "[stop_batch] 父进程 $pid : $(ps -o args= -p "$pid" | cut -c1-100)"
    mapfile -t KIDS < <(pgrep -P "$pid" 2>/dev/null)
    for k in "${KIDS[@]}"; do
        ALL_KIDS+=("$k")
        echo "            子进程 $k : $(ps -o args= -p "$k" 2>/dev/null | cut -c1-90)"
    done
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[stop_batch] --dry-run：未终止任何进程"
    exit 0
fi

# 1) 先 SIGKILL 父进程，杜绝“清理后继续下一个配置 / 重新自启动”
for pid in "${PARENTS[@]}"; do
    kill -KILL "$pid" 2>/dev/null
done
echo "[stop_batch] 已 SIGKILL 父进程：${PARENTS[*]}"

# 2) 终止子进程组（graceful），记录 PGID 以便后续强杀
declare -a PGIDS=()
for k in "${ALL_KIDS[@]}"; do
    g="$(ps -o pgid= -p "$k" 2>/dev/null | tr -d ' ')"
    [ -n "${g:-}" ] || continue
    PGIDS+=("$g")
    kill -TERM -- "-$g" 2>/dev/null
done
echo "[stop_batch] 已 TERM 子进程组：${PGIDS[*]:-无}"

# 3) 最多等 20s，仍未退出的子进程组强杀
for _ in $(seq 1 20); do
    alive=0
    for k in "${ALL_KIDS[@]}"; do
        kill -0 "$k" 2>/dev/null && alive=$((alive + 1))
    done
    [ "$alive" -eq 0 ] && break
    sleep 1
done

for g in "${PGIDS[@]}"; do
    kill -KILL -- "-$g" 2>/dev/null
done

# 4) 复核
echo "[stop_batch] 复核："
left=0
for k in "${ALL_KIDS[@]}"; do
    if kill -0 "$k" 2>/dev/null; then
        echo "  仍存活: $k $(ps -o args= -p "$k" | cut -c1-80)"
        left=$((left + 1))
    fi
done
for pid in "${PARENTS[@]}"; do
    kill -0 "$pid" 2>/dev/null && left=$((left + 1))
done
if [ "$left" -eq 0 ]; then
    echo "  已全部退出"
else
    echo "  仍有 $left 个进程存活，请检查：ps -o pid,stat,args -p ${ALL_KIDS[*]}"
fi
echo "[stop_batch] 当前 GPU："
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
