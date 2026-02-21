#!/usr/bin/env bash
# 双端口循环压力测试 vLLM / OpenAI 兼容接口
# 使用方法：保存为 test_dual.sh → chmod +x test_dual.sh → ./test_dual.sh

set -u

# ---------------- 配置区 ----------------
HOST="localhost"
PORTS=("10190" "10191" "10192" "10193" "10194")           # 要测试的两个端口
MODEL="/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct"   # 模型名称或路径（仅用于请求参数，不影响实际测试）
ENDPOINT="/v1/chat/completions"

COUNT=100000                       # 总共要发送多少条请求
SLEEP_BETWEEN=0.01                 # 每条请求之间的间隔（秒），可调小来增加压力
# ---------------------------------------

echo "开始双端口测试"
echo "端口列表     : ${PORTS[*]}"
echo "模型         : ${MODEL}"
echo "总请求数     : ${COUNT}"
echo "请求间隔     : ${SLEEP_BETWEEN} 秒"
echo "----------------------------------------"

success=0
fail=0
total_duration=0.0

for ((i=1; i<=COUNT; i++)); do
    # 轮询端口
    port_idx=$(( (i-1) % ${#PORTS[@]} ))
    port=${PORTS[$port_idx]}

    prompt="第 ${i} 条测试：1 + ${i} = ? 请直接回答数字"

    echo -n "[${i}/${COUNT}] → :${port}   发送中... "

    start_time=$(date +%s.%N)

    response=$(curl -s -X POST "http://${HOST}:${port}${ENDPOINT}" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "messages": [
                {"role": "user", "content": "'"${prompt}"'"}
            ],
            "temperature": 0.7,
            "max_tokens": 5000,
            "ignore_eos": true,
            "stream": false
        }' 2>/dev/null)

    end_time=$(date +%s.%N)
    duration=$(awk "BEGIN {print ${end_time} - ${start_time}}")

    total_duration=$(awk "BEGIN {print ${total_duration} + ${duration}}")

    # 提取结果
    content=$(echo "${response}" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
    error=$(echo "${response}" | jq -r '.error.message // empty' 2>/dev/null)

    if [[ -n "${content}" ]]; then
        # 只显示前 30 个字符，避免输出太长
        trimmed_content=$(echo "${content}" | head -c 30 | tr -d '\n')
        echo -e "\033[32m成功\033[0m (${duration}s) → ${trimmed_content}..."
        ((success++))
    elif [[ -n "${error}" ]]; then
        echo -e "\033[31m失败\033[0m (${duration}s) → ${error}"
        ((fail++))
    else
        echo -e "\033[33m异常\033[0m (${duration}s) → 响应格式异常"
        ((fail++))
    fi

    sleep "${SLEEP_BETWEEN}"
done

# 汇总统计
avg_duration=$(awk "BEGIN {printf \"%.2f\", ${total_duration} / ${COUNT}}")
qps=$(awk "BEGIN {printf \"%.2f\", ${COUNT} / ${total_duration}}")

echo "----------------------------------------"
echo "测试完成！"
echo "成功请求    : ${success}"
echo "失败/异常    : ${fail}"
echo "总耗时       : ${total_duration} 秒"
echo "平均每请求耗时 : ${avg_duration} 秒"
echo "平均 QPS     : ${qps}"