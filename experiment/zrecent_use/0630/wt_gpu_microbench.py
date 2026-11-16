#!/usr/bin/env python3
# [WT] GPU 纯算力微基准：判断「今天比 08-27 慢 1.35x」是否来自 GPU 硬件本身
#      （降频 / 供电 / 显存带宽），还是来自 CPU 侧（--enforce-eager 下每步的固定开销）。
#
# 用法：python3 wt_gpu_microbench.py [device_index] [size] [iters]
#   默认用 cuda:1（避开可能被外部租户占用的 GPU0）。
#
# [WT] 修正 2026-09-25：这里原来写的 156 TFLOPS 是 **TF32** 的峰值，写错了。
#   A800-SXM4-80GB (= A100 80GB SXM 同算力，只砍了 NVLink/FP64)：
#     FP16/BF16 Tensor dense = 108 SM x 1024 FMA/SM x 2 x 1410MHz = 311.9 TFLOPS
#     TF32      Tensor dense = 155.9 TFLOPS
#     FP32      CUDA core    = 108 x 128 FLOP x 1410MHz     =  19.5 TFLOPS
#
#   ⚠️ 判据必须换算，不能直接拿 1410MHz 的峰值比：
#     A800 是 400W 功耗墙，持续 tensor 负载**必然**降频到 ~1200MHz（这是产品行为）。
#     正确做法：实测 TFLOPS / (312 x 实际时钟/1410)。
#     2026-09-25 实测：253 TFLOPS @1214MHz -> 按实际时钟折算 = 94.3% of peak ✅
#     对照组：FP32 GEMM 跑到 19.0 TFLOPS @1410MHz = 97.6% of 19.5 ✅（CUDA core 路径不受功耗墙约束）
#
# 判据：
#   - 按实际时钟折算后仍 <85% -> GPU 侧有问题
#   - 时钟在**轻负载**下仍上不了 1410MHz -> GPU 侧有问题
#   - 持续 tensor 负载下掉到 ~1200MHz 是 400W 功耗墙的正常表现，不是故障
import subprocess
import sys
import threading
import time

import torch

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40

print(f"torch {torch.__version__}  cuda {torch.version.cuda}  devices={torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES={torch.cuda._parse_visible_devices() if hasattr(torch.cuda,'_parse_visible_devices') else 'n/a'}")
print(f"device cuda:{DEV} -> {torch.cuda.get_device_name(DEV)}")
print(f"matrix {N}x{N} fp16, iters={ITERS}")

# 必须显式设置当前设备，否则 torch.cuda.synchronize() 同步的是 device 0，
# 计时会立即返回（第一次跑就踩了这个坑，得到荒谬的 31948 TFLOPS）
torch.cuda.set_device(DEV)
print(f"current_device={torch.cuda.current_device()}")
dev = torch.device(f"cuda:{DEV}")
a = torch.randn(N, N, dtype=torch.float16, device=dev)
b = torch.randn(N, N, dtype=torch.float16, device=dev)
print(f"a.shape={tuple(a.shape)} dtype={a.dtype} device={a.device} bytes={a.numel()*a.element_size()/2**20:.0f}MB")

# warmup
for _ in range(3):
    c = a @ b
torch.cuda.synchronize(DEV)
print(f"warmup 完成, c[0,0]={c[0,0].item():.3f} (非零=确实算了)")

samples = []
stop = False


def sampler():
    while not stop:
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits", "-i", str(DEV)],
                capture_output=True, text=True, timeout=5)
            if out.stdout.strip():
                samples.append(out.stdout.strip())
        except Exception:
            pass
        time.sleep(0.25)


th = threading.Thread(target=sampler, daemon=True)
th.start()

torch.cuda.synchronize(DEV)
t0 = time.perf_counter()
for _ in range(ITERS):
    c = a @ b
torch.cuda.synchronize(DEV)
dt = time.perf_counter() - t0
stop = True
time.sleep(0.5)

print(f"总用时 {dt:.3f} s  ->  单次 GEMM {dt/ITERS*1000:.2f} ms")
flops = 2.0 * N**3 * ITERS
tflops = flops / dt / 1e12
PEAK_FP16_1410 = 311.9   # [WT] 108 SM x 1024 FMA x 2 x 1410MHz，不是 156（156 是 TF32）
print(f"实测 {tflops:.1f} TFLOPS   (1410MHz 峰值 {PEAK_FP16_1410})   "
      f"= {tflops/PEAK_FP16_1410*100:.0f}% of 1410MHz peak")
print()
print(f"负载中 GPU{DEV} 采样 ({len(samples)} 个, clock/power/temp/util):")
for s in samples[:10]:
    print("   ", s)
clocks = []
for s in samples:
    try:
        clocks.append(int(s.split(",")[0]))
    except (ValueError, IndexError):
        pass
if clocks:
    print(f"  clock: min={min(clocks)} max={max(clocks)} 平均={sum(clocks)/len(clocks):.0f} MHz  (max 1410)")
