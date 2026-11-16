#!/usr/bin/env python3
# [WT] 宿主内存带宽微基准：验证「其他租户占用 DRAM 带宽 → 每步 CPU/传输变慢」。
#
# 背景：已排除 GPU 降频(1410MHz 74.8%/零违规)、外部占卡、CPU 调度饿死(schedstat 等待 0.6%)、
#       代码/配置/数据集差异。剩下唯一未被测的共享资源是宿主 DRAM 带宽 ——
#       我们的 256GiB pinned tensor pool 在宿主内存里，KV 传输与 CPU 每步都要过它。
#
# 用法：
#   OMP_NUM_THREADS=1  python3 wt_mem_bw.py     # 单线程（延迟敏感）
#   OMP_NUM_THREADS=26 python3 wt_mem_bw.py     # 单 socket 满带宽
# 参考：Xeon Gold 5320 每 socket 8 通道 DDR4-3200 -> 理论 204 GB/s/socket
#       实测单线程流式约 10~15 GB/s；26 线程约 120~190 GB/s
import os
import time

import numpy as np

TH = os.environ.get("OMP_NUM_THREADS", "?")
N = 64 * 1024 * 1024          # 64M float64 = 512 MB
A = np.ones(N, dtype=np.float64)
B = np.ones(N, dtype=np.float64)
C = np.empty(N, dtype=np.float64)

for _ in range(2):            # warmup
    np.add(A, B, out=C)

ITS = 6
t0 = time.perf_counter()
for _ in range(ITS):
    np.add(A, B, out=C)
dt = time.perf_counter() - t0

# triad：读 A + 读 B + 写 C = 每元素 24 字节
gb = 3 * N * 8 * ITS / dt / 1e9
print(f"OMP_NUM_THREADS={TH:<3}  triad {gb:8.1f} GB/s   ({dt/ITS*1000:7.1f} ms/iter)")
