#!/usr/bin/env python3
# [WT] GPU 硬件审计：从 A800 规格推导理论峰值，逐项实测对比，判断硬件是否降级。
#
# 为什么需要它：
#   「09-24 比 08-27 慢 ~1.35x」的排查中，CPU / cgroup / 数据集 / 配置 / 代码全部被排除，
#   唯一还没被"自洽地"证伪的就是 GPU 本身。本脚本把 GPU 的每一项能力都换成
#   "实测 / 理论峰值" 的比值，任何一项明显偏低才说明硬件有问题。
#
# 理论峰值公式（Ampere GA100，全部由本机实际参数推导，不硬编码结论）：
#   FP16/BF16 Tensor dense = SM数 × 1024 FMA/cycle/SM × 2 FLOP × f_sm
#   TF32      Tensor dense = 上式 / 2
#   FP32      CUDA core    = SM数 ×  128 FLOP/cycle/SM × f_sm
#   FP64      (A100 SXM 为 FP32 的 1/2)
#   HBM       带宽          = f_mem × 2 (DDR) × 总线字节数
#
# 关键判据（避免误判）：
#   - 400W 功耗墙下**持续** tensor 负载会主动降频，这是产品行为不是故障。
#     所以必须把实测 TFLOPS 与「按当时实际 SM 时钟折算的峰值」比，而不是与 1410MHz 峰值比。
#   - 因此脚本同时跑 burst（~0.8s，反映峰值能力）和 sustain（~30s，反映稳态能力）。
#
# 用法：
#   python3 wt_gpu_hw_audit.py [device] [--out results.json]
#   默认 device=1（避开可能被外部租户占用的 GPU0）

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

import torch


# ---------------------------------------------------------------------------
# 采样器：负载期间持续记录 sm 时钟 / 功耗 / 温度 / 利用率 / 显存时钟
# ---------------------------------------------------------------------------
class Sampler:
    def __init__(self, dev, interval=0.2):
        self.dev = dev
        self.interval = interval
        self.samples = []
        self._stop = False
        self._th = None

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        if self._th:
            self._th.join(timeout=3)

    def _loop(self):
        while not self._stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                     "utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits", "-i", str(self.dev)],
                    capture_output=True, text=True, timeout=5)
                line = out.stdout.strip()
                if line:
                    self.samples.append(line)
            except Exception:
                pass
            time.sleep(self.interval)

    def stats(self):
        sm, mem, pw, tp = [], [], [], []
        for s in self.samples:
            p = [x.strip() for x in s.split(",")]
            try:
                sm.append(int(p[0])); mem.append(int(p[1]))
                pw.append(float(p[2])); tp.append(float(p[3]))
            except (ValueError, IndexError):
                pass

        def f(a):
            return {"n": len(a), "min": min(a), "max": max(a), "avg": sum(a) / len(a)} if a else {}
        return {"sm_clock_mhz": f(sm), "mem_clock_mhz": f(mem),
                "power_w": f(pw), "temp_c": f(tp)}


def throttle_reasons(dev):
    """读 nvidia-smi 的降频原因位，只有 Active 的才算数。"""
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE", "-i", str(dev)],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    res = {}
    m = re.search(r"Clocks Throttle Reasons(.*?)(?:\n\s*\n|$)", out, re.S)
    if not m:
        return res
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*([A-Za-z][A-Za-z0-9 ()/_-]*?)\s*:\s*(Active|Not Active)\s*$", line)
        if mm:
            res[mm.group(1).strip()] = mm.group(2)
    return res


def gpu_query(dev, fields):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                          "--format=csv,noheader,nounits", "-i", str(dev)],
                         capture_output=True, text=True, timeout=10)
    return [x.strip() for x in out.stdout.strip().split(",")]


def num(s):
    """nvidia-smi 对个别字段（如 power.limit）即使加了 nounits 仍会带 ' W' 后缀。"""
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else float("nan")


# ---------------------------------------------------------------------------
# 计时工具：正确同步目标设备（踩过的坑：同步 device0 会让计时立刻返回）
# ---------------------------------------------------------------------------
def timeit(fn, seconds, warmup=3, min_iters=1):
    """跑 fn 直到累计 >= seconds（或达到 min_iters），返回 (秒/次, 次数)。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 先测 1 次估算单次耗时，用于决定迭代数，避免固定 iters 导致时长失控
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-6)
    iters = max(min_iters, int(seconds / one))

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / iters, iters


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device", nargs="?", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--burst-seconds", type=float, default=0.8)
    ap.add_argument("--sustain-seconds", type=float, default=30.0)
    ap.add_argument("--skip-sustain", action="store_true")
    ap.add_argument("--skip-nvlink", action="store_true")
    args = ap.parse_args()

    DEV = args.device
    torch.cuda.set_device(DEV)
    dev = torch.device(f"cuda:{DEV}")
    props = torch.cuda.get_device_properties(DEV)
    R = {"device_index": DEV, "device_name": props.name,
         "sm_count": props.multi_processor_count,
         "torch": torch.__version__, "cuda": torch.version.cuda}

    print("=" * 78)
    print(f"[WT][HW] GPU 硬件审计  device=cuda:{DEV}  {props.name}")
    print("=" * 78)

    # ---------------- S0: 读取实际规格并推导理论峰值 --------------------------
    (f_sm_max, f_mem, plimit, pmax, mem_total) = gpu_query(
        DEV, "clocks.max.sm,clocks.max.memory,power.limit,power.max_limit,memory.total")
    f_sm_max = num(f_sm_max); f_mem = num(f_mem)
    plimit = num(plimit); mem_total = num(mem_total)

    SM = props.multi_processor_count
    # HBM2e: 5 栈 x 1024 bit = 5120 bit = 640 B；DDR 所以 x2
    BUS_BYTES = 640.0
    bw_theory = f_mem * 1e6 * 2 * BUS_BYTES / 1e9          # GB/s
    fp16_peak_max = SM * 1024 * 2 * f_sm_max * 1e6 / 1e12  # TFLOPS @ f_sm_max
    fp32_peak_max = SM * 128 * f_sm_max * 1e6 / 1e12

    R["spec"] = {"sm_max_mhz": f_sm_max, "mem_mhz": f_mem,
                 "power_limit_w": plimit, "power_max_w": num(pmax),
                 "mem_total_mib": mem_total, "bus_bytes": BUS_BYTES,
                 "hbm_theory_gbs": bw_theory,
                 "fp16_peak_tflops_at_max_clock": fp16_peak_max,
                 "fp32_peak_tflops_at_max_clock": fp32_peak_max}

    print(f"\n--- S0 规格与理论峰值 ---")
    print(f"  SM 数                : {SM}")
    print(f"  SM 最大时钟          : {f_sm_max:.0f} MHz")
    print(f"  显存时钟 / 总线      : {f_mem:.0f} MHz x2 x {BUS_BYTES:.0f} B")
    print(f"  => HBM 理论带宽      : {bw_theory:.0f} GB/s   (A800 规格 2039 GB/s)")
    print(f"  功耗墙               : {plimit:.0f} W (max {num(pmax):.0f} W)")
    print(f"  => FP16/BF16 Tensor  : {fp16_peak_max:.1f} TFLOPS @ {f_sm_max:.0f}MHz"
          f"   (= SM x 1024FMA x 2 x f)")
    print(f"  => TF32      Tensor  : {fp16_peak_max/2:.1f} TFLOPS")
    print(f"  => FP32      CUDA    : {fp32_peak_max:.1f} TFLOPS")

    R["throttle_before"] = throttle_reasons(DEV)
    print(f"  空闲时降频原因(Active): "
          f"{[k for k,v in R['throttle_before'].items() if v=='Active'] or '无'}")

    # ---------------- S1: 算力（GEMM）-------------------------------------
    print(f"\n--- S1 算力实测：矩阵乘 (TFLOPS) ---")
    print(f"{'dtype':<8}{'N':>7}{'模式':>8}{'秒/次':>11}{'TFLOPS':>10}"
          f"{'峰值@maxf':>11}{'占峰值':>9}{'实际时钟':>10}{'功耗':>8}")
    gemm = []

    def run_gemm(dtype, N, mode, seconds, peak):
        a = torch.randn(N, N, dtype=dtype, device=dev)
        b = torch.randn(N, N, dtype=dtype, device=dev)
        def fn():
            return a @ b
        with Sampler(DEV) as s:
            per, iters = timeit(fn, seconds)
        fl = 2.0 * N ** 3
        tf = fl / per / 1e12
        st = s.stats()
        ck = st.get("sm_clock_mhz", {}).get("avg", 0)
        pw = st.get("power_w", {}).get("avg", 0)
        # 按当时实际时钟折算的理论峰值 —— 这是唯一公平的对比基准
        peak_now = peak * (ck / f_sm_max) if ck else peak
        pct_now = tf / peak_now * 100 if peak_now else 0
        print(f"{str(dtype).replace('torch.',''):<8}{N:>7}{mode:>8}{per*1000:>10.3f}ms"
              f"{tf:>10.1f}{peak:>11.0f}{tf/peak*100:>8.1f}%"
              f"{ck:>9.0f}MHz{pw:>7.0f}W")
        gemm.append({"dtype": str(dtype), "N": N, "mode": mode,
                     "ms_per_op": per * 1000, "tflops": tf,
                     "peak_at_max_clock": peak, "pct_of_max_peak": tf / peak * 100,
                     "achieved_sm_clock_mhz": ck, "avg_power_w": pw,
                     "pct_of_clock_adjusted_peak": pct_now,
                     "iters": iters, "sampler": st})
        del a, b
        torch.cuda.empty_cache()

    N_BIG = 8192
    run_gemm(torch.float16, N_BIG, "burst", args.burst_seconds, fp16_peak_max)
    run_gemm(torch.bfloat16, N_BIG, "burst", args.burst_seconds, fp16_peak_max)
    if not args.skip_sustain:
        run_gemm(torch.float16, N_BIG, "sustain", args.sustain_seconds, fp16_peak_max)

    # TF32 / FP32
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        run_gemm(torch.float32, 4096, "tf32", args.burst_seconds, fp16_peak_max / 2)
        torch.backends.cuda.matmul.allow_tf32 = False
        run_gemm(torch.float32, 4096, "fp32", args.burst_seconds, fp32_peak_max)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # 小矩阵：反映 batch 小 / 形状不友好时的实际效率（decode 常见形态）
    run_gemm(torch.float16, 1024, "burst", args.burst_seconds, fp16_peak_max)
    R["gemm"] = gemm

    # ---------------- S2: HBM 带宽 ---------------------------------------
    print(f"\n--- S2 HBM 带宽实测 (GB/s, 理论 {bw_theory:.0f}) ---")
    print(f"{'大小':>8}{'copy(2路)':>12}{'add(3路)':>12}{'fill(只写)':>12}"
          f"{'read(只读)':>12}{'最好/理论':>11}")
    bw_rows = []

    for mib in [512, 2048, 8192]:
        n = mib * 1024 * 1024 // 4
        a = torch.ones(n, dtype=torch.float32, device=dev)
        b = torch.ones(n, dtype=torch.float32, device=dev)
        c = torch.empty(n, dtype=torch.float32, device=dev)

        per_copy, _ = timeit(lambda: c.copy_(a), 0.6)
        per_add, _ = timeit(lambda: torch.add(a, b, out=c), 0.6)
        per_fill, _ = timeit(lambda: c.fill_(1.5), 0.6)
        # 只读：sum 走 reduction，基本只产生读流量
        per_read, _ = timeit(lambda: torch.sum(a), 1.0)

        nb = n * 4
        r = {"size_mib": mib,
             "copy_gbs": 2 * nb / per_copy / 1e9,
             "add_gbs": 3 * nb / per_add / 1e9,
             "fill_gbs": 1 * nb / per_fill / 1e9,
             "read_gbs": 1 * nb / per_read / 1e9}
        best = max(r["copy_gbs"], r["add_gbs"])
        r["best_gbs"] = best
        r["pct_of_theory"] = best / bw_theory * 100
        print(f"{mib:>6}MiB{r['copy_gbs']:>12.1f}{r['add_gbs']:>12.1f}"
              f"{r['fill_gbs']:>12.1f}{r['read_gbs']:>12.1f}{r['pct_of_theory']:>10.1f}%")
        bw_rows.append(r)
        del a, b, c
        torch.cuda.empty_cache()
    R["hbm_bandwidth"] = bw_rows
    bw_best = max(r["best_gbs"] for r in bw_rows)
    R["hbm_best_gbs"] = bw_best

    # ---------------- S3: decode 形态（GEMV/小 batch）带宽 -----------------
    # decode 每步 = 把全部权重从 HBM 读一遍 -> 纯带宽受限。
    # 用 [out,in] @ [in,batch] 模拟一层 Linear，batch 取 1/8 反映真实 decode。
    print(f"\n--- S3 decode 形态带宽（权重读取，GB/s，理论 {bw_theory:.0f}）---")
    decode_rows = []
    for (out_f, in_f) in [(8192, 8192), (16384, 16384)]:
        w = torch.randn(out_f, in_f, dtype=torch.bfloat16, device=dev)
        wb = w.numel() * 2
        for bs in [1, 8, 64]:
            x = torch.randn(in_f, bs, dtype=torch.bfloat16, device=dev)
            per, _ = timeit(lambda: w @ x, 0.8)
            gbs = wb / per / 1e9          # 只算权重读取流量（x 很小可忽略）
            print(f"  {out_f}x{in_f} bf16, batch={bs:<3} {per*1000:8.3f} ms/步 "
                  f"-> 权重带宽 {gbs:7.1f} GB/s  = 理论 {gbs/bw_theory*100:5.1f}%")
            decode_rows.append({"shape": [out_f, in_f], "batch": bs,
                                "ms": per * 1000, "weight_gbs": gbs,
                                "pct_of_theory": gbs / bw_theory * 100})
        del w, x
        torch.cuda.empty_cache()
    R["decode_gemv"] = decode_rows

    # ---------------- S4: NVLink P2P（PD 分离的 KV 传输通道）---------------
    if not args.skip_nvlink and torch.cuda.device_count() > 1:
        print(f"\n--- S4 NVLink P2P 带宽（A800 规格：总计 400 GB/s = 单向 200 GB/s）---")
        peer = 0 if DEV != 0 else 1
        torch.cuda.set_device(peer)
        n = 4096 * 1024 * 1024 // 4
        src = torch.ones(n, dtype=torch.float32, device=torch.device(f"cuda:{peer}"))
        torch.cuda.set_device(DEV)
        dst = torch.empty(n, dtype=torch.float32, device=dev)
        try:
            per, _ = timeit(lambda: dst.copy_(src), 1.0)
            one_way = n * 4 / per / 1e9
            print(f"  cuda:{peer} -> cuda:{DEV}  张量 {n*4/2**30:.1f} GiB  "
                  f"{per*1000:.2f} ms  -> 单向 {one_way:.1f} GB/s "
                  f"({one_way/200*100:.0f}% of 200)")
            R["nvlink"] = {"peer": peer, "ms": per * 1000, "one_way_gbs": one_way,
                           "pct_of_200": one_way / 200 * 100}
        except Exception as e:
            print(f"  P2P 测试失败: {e}")
            R["nvlink"] = {"error": str(e)}
        finally:
            del src, dst
            torch.cuda.empty_cache()

    R["throttle_after"] = throttle_reasons(DEV)
    print(f"\n--- 负载后降频原因(Active) ---")
    for k, v in R["throttle_after"].items():
        mark = "  <== ACTIVE" if v == "Active" else ""
        print(f"  {k:<34} {v}{mark}")

    # ---------------- 结论汇总 -------------------------------------------
    print("\n" + "=" * 78)
    print("[WT][HW] 汇总：实测 / 理论 比值")
    print("=" * 78)
    print(f"  HBM 带宽            : {bw_best:7.1f} / {bw_theory:.0f} GB/s"
          f"  = {bw_best/bw_theory*100:5.1f}%")
    for g in gemm:
        if g["mode"] == "sustain":
            eff = g["pct_of_clock_adjusted_peak"]
            print(f"  FP16 GEMM (持续)    : {g['tflops']:7.1f} TFLOPS"
                  f"  @{g['achieved_sm_clock_mhz']:.0f}MHz"
                  f"  按时钟折算占峰值 {eff:5.1f}%")
    for g in gemm:
        if g["mode"] == "burst" and g["dtype"] == "torch.float16" and g["N"] == N_BIG:
            print(f"  FP16 GEMM (短时)    : {g['tflops']:7.1f} TFLOPS"
                  f"  @{g['achieved_sm_clock_mhz']:.0f}MHz"
                  f"  = 1410MHz 峰值的 {g['pct_of_max_peak']:5.1f}%")

    out = args.out or f"/root/myshare/wty_tmp_result/env_fingerprint/gpu_hw_audit_dev{DEV}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"\n原始结果: {out}")


if __name__ == "__main__":
    main()
