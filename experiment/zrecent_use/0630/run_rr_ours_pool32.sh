#!/usr/bin/env bash
# [WT] 启动 RR + Ours(scheme4_gwf) 对比实验，KV tensor pool 大小由 POOL_GB 控制。
#
# 背景（2026-09-24 用 dmesg 证实，不再是推测）：
#   - 该池是 pinned HOST memory（vllm/.../p2p/tensor_memory_pool.py:
#     torch.empty(..., pin_memory=True)），每实例占 mem_pool_size_gb 字节主机内存，
#     且 pinned 页不可换出，只能实打实占物理内存。
#   - 池=32 时 8 实例 x 32GiB = 256GiB 不可换出；本机 MemTotal 377GiB。曾经由此被内核
#     OOM killer SIGKILL：
#         Out of memory: Killed process (VLLM::EngineCor) shmem-rss:33564428kB
#     EngineCore 死后变 Zl <defunct> 僵尸（父进程 vllm serve 卡在 wait_for_engine_startup
#     不收尸），nvidia-smi 里变 [NOT FOUND] 残留条目，实验反复 start_servers_failed。
#   - 直接原因是 /dev/shm 里 81GB 陈旧 vllm_offload_*.mmap 残留：强杀进程导致
#     cleanup() 没跑、文件没被 unlink。创建者 vllm/v1/kv_offload/cpu/shared_offload_region.py
#     已不在当前工作区（当前代码不再产生它）。
#     2026-09-24 已安全清理，实测前后：
#       /dev/shm  91.6GB → 11GB      Shmem     97.57 → 7.01 GiB
#       MemAvailable 209.87 → 326.56 GiB
#     清理时逐文件四重复查（顶层路径 + 文件名 + root 属主 + 无进程持有 fd/mmap），
#     25/25 删净；nccl-*/d2_*/q16h*/sglang_loads_*/uid 1007 等其它内容复核后全部未变。
#
# 池大小估算（8 实例，单位 GiB）：需要 ≈ 8*POOL_GB + 56
#   （8*4 为每实例 APIServer+engine 开销，24 为权重页缓存 + 安全余量）
#   ⚠️ POOL_GB 只能是 2 的幂！ TensorMemoryPool 会把 max_block_size 向上取整到 2 的幂：
#      5~8 → 实际 8GiB；9~16 → 实际 16GiB；17~32 → 实际 32GiB
#      （tensor_memory_pool.py 的 _round_to_power_of_two）
#      所以 POOL_GB=10/12/14 都等于 16，别以为写小一点就能省内存。
#   POOL_GB=8  → 实际  8GiB，需要≈120GiB  余量≈207GiB  ✅ 稳（9/1 该配置拿到史上最好 goodput 0.37）
#   POOL_GB=16 → 实际 16GiB，需要≈184GiB  余量≈143GiB  ✅ 稳
#   POOL_GB=32 → 实际 32GiB，需要≈285GiB  余量≈ 42GiB  ⚠️ 可一搏，但余量只 13%
#   （余量按 2026-09-24 清理后 MemAvailable≈326.6GiB 计）
#   ⚠️ 本机为多租户共享，内存波动可达上百 GiB（当天实测 MemAvailable 走过
#      16.6 / 95 / 210 / 326 GiB），swap 一度为 0。跑长 benchmark 中途被打死会前功尽弃，
#      故稳妥值仍是 8 或 16。
# 启动前务必确认：
#   grep MemAvailable /proc/meminfo          # 单位 kB，除以 1048576 得 GiB
#   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
#
# 用法：
#   cd /root/predict-schedule/experiment/zrecent_use/0630
#   POOL_GB=8 ./run_rr_ours_pool32.sh
#   nohup env POOL_GB=8 ./run_rr_ours_pool32.sh > /root/myshare/wty_tmp_result/retry_qwen_4p4d/nohup_pool8_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")"

: "${RESULT_ROOT:=/root/myshare/wty_tmp_result/retry_qwen_4p4d}"
# [WT] KV tensor pool 大小（GiB）。默认 8；改大前务必先确认主机可用内存够。
: "${POOL_GB:=8}"

MODEL_TYPE=qwen \
QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct \
DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets \
TRACE_NAME=tmp_dataset3 \
PD_RATIO_LIST=4:4 \
TPOT_MS_LIST=30 \
BENCH_GOODPUT_TTFT_MS=1000 \
GPU_IDLE_MAX_MEMORY_MIB=10 \
COMPARE_EXPERIMENT_VARIANTS="roundrobin scheme4_gwf" \
RUN_REPEAT_PER_CONFIG=1 \
WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE= \
SKIP_EXISTING_RESULT_DIRS= \
PREFILL_TENSOR_POOL_MEMORY="${POOL_GB}" \
DECODE_TENSOR_POOL_MEMORY="${POOL_GB}" \
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
BASE_RESULT_DIR="${RESULT_ROOT}" \
bash ./test_evaluation_strategyA_oracle_gwf.sh
