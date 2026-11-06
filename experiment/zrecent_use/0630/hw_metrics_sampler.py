#!/usr/bin/env python3
"""
Lightweight hardware sampler for the disagg A/B runs.

Snapshots (GPU util/mem per GPU, CPU loadavg, host memory) every INTERVAL_S into a
CSV, so workload characterization can correlate the admission-controller state with
hardware pressure.  Sampling is low-frequency and read-only: nvidia-smi + /proc reads,
so it does not perturb the benchmark.

Usage:
    python3 hw_metrics_sampler.py --out <csv> --interval <s> [--duration <s>]

Column order (unit-suffixed, per AGENTS.md):
    ts_s, load1, load5, load15, mem_avail_mib, mem_total_mib,
    gpu{0..7}_util_pct, gpu{0..7}_mem_mib
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys, time


def _read_loadavg() -> tuple[float, float, float]:
    a, b, c = open("/proc/loadavg").read().split()[:3]
    return float(a), float(b), float(c)


def _read_meminfo_mib() -> tuple[float, float]:
    d: dict[str, float] = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable"):
            d[k] = float(v.strip().split()[0]) / 1024.0  # kB -> MiB
    return d.get("MemAvailable", 0.0), d.get("MemTotal", 0.0)


def _read_gpu() -> list[tuple[float, float]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append((float(parts[1]), float(parts[2])))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0)  # 0 = forever
    args = ap.parse_args()

    header = ["ts_s", "load1", "load5", "load15",
              "mem_avail_mib", "mem_total_mib"]
    for i in range(8):
        header += [f"gpu{i}_util_pct", f"gpu{i}_mem_mib"]

    start = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        f.flush()
        while True:
            try:
                l1, l5, l15 = _read_loadavg()
                mavail, mtotal = _read_meminfo_mib()
                gpus = _read_gpu()
            except Exception:
                gpus = []
            row = [round(time.time() - start, 1), l1, l5, l15,
                   round(mavail, 1), round(mtotal, 1)]
            for i in range(8):
                if i < len(gpus):
                    row += [round(gpus[i][0], 1), round(gpus[i][1], 1)]
                else:
                    row += ["", ""]
            w.writerow(row)
            f.flush()
            if args.duration > 0 and (time.time() - start) >= args.duration:
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
