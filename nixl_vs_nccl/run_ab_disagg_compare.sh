#!/bin/bash

set -u
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_nixl_disagg.sh"
DISAGG_SCRIPT="$SCRIPT_DIR/disagg_example_p2p_nccl_xpyd.sh"

STAGE_SWITCH_TIMEOUT_SECONDS=${STAGE_SWITCH_TIMEOUT_SECONDS:-60}

if [ ! -f "$LAUNCH_SCRIPT" ]; then
    echo "❌ 未找到脚本: $LAUNCH_SCRIPT"
    exit 1
fi

if [ ! -f "$DISAGG_SCRIPT" ]; then
    echo "❌ 未找到脚本: $DISAGG_SCRIPT"
    exit 1
fi

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

# 默认使用完全独立的两套日志路径。
# 如果设置了 BASE_LOG_DIR，会回退到旧行为（同一父目录下分 la/dis 子目录）。
BASE_LOG_DIR=${BASE_LOG_DIR:-}
if [ -n "$BASE_LOG_DIR" ]; then
    LA_LOG_DIR=${LA_LOG_DIR:-$BASE_LOG_DIR/la_nixl}
    DIS_LOG_DIR=${DIS_LOG_DIR:-$BASE_LOG_DIR/dis_p2p_nccl}
else
    LA_LOG_ROOT=${LA_LOG_ROOT:-$SCRIPT_DIR/logs/la_runs}
    DIS_LOG_ROOT=${DIS_LOG_ROOT:-$SCRIPT_DIR/logs/dis_runs}
    LA_LOG_DIR=${LA_LOG_DIR:-$LA_LOG_ROOT/$RUN_TAG}
    DIS_LOG_DIR=${DIS_LOG_DIR:-$DIS_LOG_ROOT/$RUN_TAG}
fi

LA_LOG_SUFFIX=${LA_LOG_SUFFIX:-la_nixl}
DIS_LOG_SUFFIX=${DIS_LOG_SUFFIX:-dis_p2p_nccl}
LA_PID_TRACK_FILE="$LA_LOG_DIR/pids_${LA_LOG_SUFFIX}.txt"
DIS_PID_TRACK_FILE="$DIS_LOG_DIR/pids_${DIS_LOG_SUFFIX}.txt"

echo "============================================"
echo "A/B 对比测试开始"
echo "RUN_TAG: $RUN_TAG"
echo "LA 脚本日志目录:  $LA_LOG_DIR"
echo "DIS 脚本日志目录: $DIS_LOG_DIR"
echo "LA 日志后缀:     $LA_LOG_SUFFIX"
echo "DIS 日志后缀:    $DIS_LOG_SUFFIX"
echo "============================================"

is_port_busy() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"
        return $?
    fi

    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi

    return 1
}

wait_ports_free() {
    local timeout="$1"
    shift
    local ports=("$@")
    local start_ts
    start_ts=$(date +%s)

    while true; do
        local busy_ports=()
        for port in "${ports[@]}"; do
            if is_port_busy "$port"; then
                busy_ports+=("$port")
            fi
        done

        if [ "${#busy_ports[@]}" -eq 0 ]; then
            return 0
        fi

        local now_ts
        now_ts=$(date +%s)
        if (( now_ts - start_ts >= timeout )); then
            echo "⚠️ 端口在超时后仍被占用: ${busy_ports[*]}"
            return 1
        fi

        sleep 2
    done
}

kill_listeners_on_ports() {
    local ports=("$@")
    for port in "${ports[@]}"; do
        if command -v lsof >/dev/null 2>&1; then
            local pids
            pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
            if [ -n "$pids" ]; then
                kill -9 $pids 2>/dev/null || true
            fi
            continue
        fi

        if command -v fuser >/dev/null 2>&1; then
            local pids
            pids=$(fuser -n tcp "$port" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                kill -9 $pids 2>/dev/null || true
            fi
        fi
    done
}

kill_from_pid_files() {
    local pid_files=("$@")
    for pid_file in "${pid_files[@]}"; do
        if [ -f "$pid_file" ]; then
            while IFS= read -r pid; do
                if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done < "$pid_file"
        fi
    done
}

stop_all_instances_between_stages() {
    echo ""
    echo "--------------------------------------------"
    echo "阶段切换: 停止现有实例并等待端口释放"
    echo "--------------------------------------------"

    pkill -9 -f "toy_proxy_server.py" 2>/dev/null || true
    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true

    # 按阶段 PID 文件做精确补杀。
    kill_from_pid_files "$LA_PID_TRACK_FILE" "$DIS_PID_TRACK_FILE"

    # 涉及的关键端口：prefill/decode/proxy/benchmark/side-channel
    local critical_ports=(20003 20005 30001 10001 5559 5659)

    # 先按端口补杀一轮，再等待端口完全释放。
    kill_listeners_on_ports "${critical_ports[@]}"
    if ! wait_ports_free "$STAGE_SWITCH_TIMEOUT_SECONDS" "${critical_ports[@]}"; then
        echo "❌ 阶段切换失败：实例未完全停止，端口未释放。"
        return 1
    fi

    echo "阶段切换清理完成。"
    return 0
}

run_stage() {
    local stage_name="$1"
    local script_path="$2"
    local log_dir="$3"
    local log_suffix="$4"
    local pid_track_file="$5"

    mkdir -p "$log_dir"

    echo ""
    echo "--------------------------------------------"
    echo "[$stage_name] 开始"
    echo "脚本: $script_path"
    echo "LOG_DIR: $log_dir"
    echo "LOG_SUFFIX: $log_suffix"
    echo "PID_TRACK_FILE: $pid_track_file"
    echo "--------------------------------------------"

    LOG_DIR="$log_dir" LOG_SUFFIX="$log_suffix" PID_TRACK_FILE="$pid_track_file" bash "$script_path"
    local stage_exit_code=$?

    echo "[$stage_name] 结束，退出码: $stage_exit_code"
    return "$stage_exit_code"
}

LA_EXIT_CODE=0
DIS_EXIT_CODE=0

# 先做一次预清理，确保起跑环境干净
if ! stop_all_instances_between_stages; then
    exit 2
fi

run_stage "LAUNCH_NIXL" "$LAUNCH_SCRIPT" "$LA_LOG_DIR" "$LA_LOG_SUFFIX" "$LA_PID_TRACK_FILE"
LA_EXIT_CODE=$?

if [ "$LA_EXIT_CODE" -eq 0 ]; then
    # LA 完整跑完后，显式停实例再启动 DIS
    if ! stop_all_instances_between_stages; then
        exit 2
    fi

    run_stage "DISAGG_P2P_NCCL" "$DISAGG_SCRIPT" "$DIS_LOG_DIR" "$DIS_LOG_SUFFIX" "$DIS_PID_TRACK_FILE"
    DIS_EXIT_CODE=$?
else
    echo ""
    echo "⚠️ LA 阶段未成功完成，跳过 DIS 阶段。"
    DIS_EXIT_CODE=99
fi

# 全部结束后再清一次，避免遗留进程
if ! stop_all_instances_between_stages; then
    exit 2
fi

echo ""
echo "============================================"
echo "A/B 对比测试完成"
echo "LAUNCH_NIXL 退出码:    $LA_EXIT_CODE"
echo "DISAGG_P2P_NCCL 退出码: $DIS_EXIT_CODE"
echo "LA 脚本日志目录:  $LA_LOG_DIR"
echo "DIS 脚本日志目录: $DIS_LOG_DIR"
echo "============================================"

if [ "$LA_EXIT_CODE" -ne 0 ] || [ "$DIS_EXIT_CODE" -ne 0 ]; then
    exit 1
fi

exit 0
