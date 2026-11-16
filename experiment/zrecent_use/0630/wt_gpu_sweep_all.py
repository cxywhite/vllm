#!/usr/bin/env python3
# [WT] 逐卡体检：8 张 A800 逐一跑 HBM 带宽 + FP16 GEMM，确认没有单卡降级。
#
# 动机：decode3 曾承担 66.5% 的请求并撞到 465 并发崩溃点。若某张卡本身略慢，
#       路由/流水线会表现出「某实例特别容易堆积」的假象，需要先排除硬件差异。
#
# 用法：python3 wt_gpu_sweep_all.py [--gemm-seconds 0.8] [--bw-mib 2048]

import argparse
import subprocess
import threading
import time

import torch


def sample(dev):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", str(dev)],
            capture_output=True, text=True, timeout=5).stdout.strip()
        p = [x.strip() for x in out.split(",")]
        return int(p[0]), float(p[1]), float(p[2])
    except Exception:
        return None


class S:
    def __init__(self, dev):
        self.dev, self.buf, self.stop = dev, [], False
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop:
            v = sample(self.dev)
            if v:
                self.buf.append(v)
            time.sleep(0.15)

    def __enter__(self):
        self.t.start(); return self

    def __exit__(self, *a):
        self.stop = True; self.t.join(timeout=2)

    def avg(self):
        if not self.buf:
            return (0, 0, 0)
        n = len(self.buf)
        return (sum(x[0] for x in self.buf) / n,
                sum(x[1] for x in self.buf) / n,
                sum(x[2] for x in self.buf) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm-seconds", type=float, default=0.8)
    ap.add_argument("--bw-mib", type=int, default=2048)
    a = ap.parse_args()

    n_gpu = torch.cuda.device_count()
    print("=" * 92)
    print(f"[WT][SWEEP] 逐卡体检：{n_gpu} 张卡，各测 HBM 带宽(copy) + FP16 GEMM")
    print("=" * 92)
    print(f"{'GPU':>4}{'copy GB/s':>11}{'占理论':>8}{'FP16 TFLOPS':>13}"
          f"{'GEMM时钟':>10}{'能耗':>8}{'温度':>7}{'按时钟折算效率':>16}")

    rows = []
    for d in range(n_gpu):
        torch.cuda.set_device(d)
        dev = torch.device(f"cuda:{d}")

        # --- HBM copy ---
        n = a.bw_mib * 1024 * 1024 // 4
        x = torch.ones(n, dtype=torch.float32, device=dev)
        y = torch.empty(n, dtype=torch.float32, device=dev)
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(15):
            y.copy_(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        bw = 2 * n * 4 * 15 / dt / 1e9
        del x, y
        torch.cuda.empty_cache()

        # --- FP16 GEMM ---
        N = 8192
        p = torch.randn(N, N, dtype=torch.float16, device=dev)
        q = torch.randn(N, N, dtype=torch.float16, device=dev)
        for _ in range(3):
            r = p @ q
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = p @ q
        torch.cuda.synchronize()
        one = time.perf_counter() - t0
        iters = max(3, int(a.gemm_seconds / one))
        with S(d) as s:
            t0 = time.perf_counter()
            for _ in range(iters):
                r = p @ q
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        tf = 2.0 * N ** 3 * iters / dt / 1e12
        ck, pw, tp = s.avg()
        # 1410 MHz 下的 FP16 峰值 = 108 SM x 1024 FMA x 2 x 1410MHz = 311.9 TFLOPS
        peak_now = 311.9 * ck / 1410 if ck else 0
        eff = tf / peak_now * 100 if peak_now else 0
        print(f"{d:>4}{bw:>11.1f}{bw/2039*100:>7.1f}%{tf:>13.1f}"
              f"{ck:>9.0f}MHz{pw:>7.0f}W{tp:>6.0f}C{eff:>15.1f}%")
        rows.append({"gpu": d, "copy_gbs": bw, "fp16_tflops": tf,
                     "sm_clock": ck, "power_w": pw, "temp_c": tp, "eff_pct": eff})
        del p, q, r
        torch.cuda.empty_cache()

    print("\n--- 离散度（越大越可疑）---")
    for key, unit in [("copy_gbs", "GB/s"), ("fp16_tflops", "TFLOPS")]:
        v = [r[key] for r in rows]
        mn, mx = min(v), max(v)
        print(f"  {key:<14} min={mn:8.1f} max={mx:8.1f} {unit}"
              f"   极差/均值 = {(mx-mn)/(sum(v)/len(v))*100:5.2f}%")


if __name__ == "__main__":
    main()
