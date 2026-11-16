#!/usr/bin/env python3
# [WT] 宿主内存「延迟」基准（指针追逐）——而不是带宽。
#
# 为什么测延迟：
#   已把所有"静态"因素排除（代码/配置92键/数据集/编译产物torch包/环境变量/绑核），
#   硬件也健康（GPU 1410MHz+HBM87%峰值、CPU 单线程111.7GFLOPS、无调度饿死 0.6%）。
#   但引擎每步多花固定时间，是**均匀等比平移** —— 这正是「每次访存都变慢一点」的签名，
#   而带宽测试测不到这个。DRAM 延迟是典型的"被其他租户共享但看不见"的资源。
#
# 参考值（Xeon Gold 5320，DDR4-3200 2 socket）：
#   本地 NUMA 访存延迟约 90~110 ns；跨 socket 约 140~170 ns
#   若实测远高于此 -> 内存子系统被占用/延迟劣化
#
# 用法：python3 wt_mem_latency.py
import os
import time

import numpy as np

SIZE = 512 * 1024 * 1024 // 8          # 512 MB / 8B = 64M 个节点，远超 cache
nodes = np.arange(SIZE, dtype=np.int64)
rng = np.random.default_rng(42)
nodes = rng.permutation(nodes)          # 随机化，破坏预取
ptr = nodes.copy()                      # ptr[i] = 下一个下标

STEPS = 20_000_000


def chase():
    i = 0
    t0 = time.perf_counter()
    for _ in range(STEPS):
        i = ptr[i]
    dt = time.perf_counter() - t0
    return dt / STEPS * 1e9, i


chase()                                   # warmup
best = min(chase()[0] for _ in range(3))
print(f"  随机指针追逐延迟: {best:.1f} ns   ({STEPS:,} 次/轮, 取 3 轮最小值)")
print(f"  参考: 本地 NUMA 90~110 ns，跨 socket 140~170 ns")
print(f"  → {'⚠️ 明显偏高，内存延迟被劣化' if best > 140 else '✅ 在正常范围'}")

# NUMA 相关信息
try:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(("Cpus_allowed_list", "Mems_allowed_list")):
                print("  " + line.strip())
except OSError:
    pass
print(f"  当前线程 CPU: {os.sched_getcpu()}")
