#!/usr/bin/env bash
# 三臂序列器：等当前 GWF-real 跑完 → conservative(real) → FCFS(real)。
# 等待判据：无 vllm/bench/proxy/run-gwf 进程 + 8 卡显存空闲。
set -uo pipefail
cd /root/predict-schedule/experiment/zrecent_use/0630

echo "=== sequencer start $(date) ==="

# 1) 等 GWF-real 结束
while true; do
  n=$(ps -eo pid,args | grep -E 'vllm serve|test_disagg_proxy|benchmark_serving_trace|run_kv1p6_gwf_real' | grep -v grep | wc -l)
  if [ "$n" -eq 0 ]; then
    sleep 10
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
    if [ "$m" -lt 200 ]; then
      echo "GWF-real done + GPUs idle (${m} MiB total). -> conservative"
      break
    fi
  fi
  sleep 30
done

# 2) arm2: conservative
bash run_kv1p6_conservative_real.sh
echo "=== conservative done $(date) ==="

# 3) arm3: FCFS
bash run_kv1p6_fcfs_real.sh
echo "=== ALL seq done $(date) ==="
