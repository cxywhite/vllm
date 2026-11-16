#!/usr/bin/env python3
# [WT] GPU HBM 带宽微基准：decode 是 HBM 带宽受限的（每步都要把权重从 HBM 读一遍）。
#      若 HBM 实际带宽下降，就能解释「SM 时钟满速 1410MHz 但每 token 时间仍高 1.48×」。
#
# A800-SXM4-80GB 规格：HBM2e 2039 GB/s (理论峰值，实测 copy 通常 1400~1700 GB/s)
#
# 用法：python3 wt_gpu_hbm_bw.py [device] [size_mib]
import sys
import time

import torch

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SIZE_MIB = int(sys.argv[2]) if len(sys.argv) > 2 else 4096   # 每个张量 4 GiB

torch.cuda.set_device(DEV)
n = SIZE_MIB * 1024 * 1024 // 4          # float32 元素数
dev = torch.device(f"cuda:{DEV}")
print(f"device cuda:{DEV} ({torch.cuda.get_device_name(DEV)})  每张量 {SIZE_MIB} MiB  fp32")

a = torch.ones(n, dtype=torch.float32, device=dev)
b = torch.ones(n, dtype=torch.float32, device=dev)
c = torch.empty(n, dtype=torch.float32, device=dev)


def bench(name, fn, streams, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(DEV)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(DEV)
    dt = time.perf_counter() - t0
    gbs = streams * n * 4 * iters / dt / 1e9
    print(f"  {name:<26} {gbs:8.1f} GB/s   ({dt/iters*1000:7.3f} ms/op, {streams} 路流量)")
    return gbs


print()
print('=== HBM 带宽实测 ===')
copy_bw = bench("copy_ (读C写A)", lambda: c.copy_(a), 2)
add_bw = bench("add(读a读b写c)", lambda: torch.add(a, b, out=c), 3)

print()
print(f"规格峰值 2039 GB/s；copy 通常能到 1400~1700 GB/s")
print(f"  实测 copy = {copy_bw:.0f} GB/s = 峰值的 {copy_bw/2039*100:.0f}%")
