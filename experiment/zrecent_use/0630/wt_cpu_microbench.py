#!/usr/bin/env python3
# [WT] CPU 微基准：判断「系统性每步变慢」是否来自宿主 CPU 被过订阅。
#
# 用法：
#   OMP_NUM_THREADS=1  python3 wt_cpu_microbench.py
#   OMP_NUM_THREADS=52 python3 wt_cpu_microbench.py
# 对比两者的 GFLOPS 与扩展效率：
#   - 单线程接近该 CPU 应有水平、且 52 线程扩展良好 -> CPU 没被抢
#   - 单线程正常、52 线程扩展差         -> 被过订阅（其他租户在抢核）
#   - 单线程也明显偏低                   -> CPU 降频/被限
#
# 参考：Xeon Gold 5320 @ 2.2GHz(全核 turbo≈2.8GHz), AVX-512 -> 单核理论 ~90 GFLOPS(fp32)
#       numpy/cuBLAS 类实现通常能到理论值的 40~70%
import os
import subprocess
import time

import numpy as np

TH = os.environ.get("OMP_NUM_THREADS", "?")
N = 3000
ITS = 8

a = np.random.rand(N, N).astype(np.float32)
b = np.random.rand(N, N).astype(np.float32)

for _ in range(2):
    c = a @ b

t0 = time.perf_counter()
for _ in range(ITS):
    c = a @ b
dt = time.perf_counter() - t0

gflops = 2.0 * N**3 * ITS / dt / 1e9
with open("/proc/cpuinfo") as fh:
    mhz = [float(l.split(":")[1]) for l in fh if l.startswith("cpu MHz")]
print(f"OMP_NUM_THREADS={TH:<3}  {dt/ITS*1000:8.1f} ms/GEMM  {gflops:8.1f} GFLOPS "
      f" (cpu MHz min={min(mhz):.0f} avg={sum(mhz)/len(mhz):.0f} max={max(mhz):.0f})")
