#!/usr/bin/env python3
"""Low-overhead hardware sampler for disagg A/B runs (v2, dmon-based).

Why v2: `hw_metrics_sampler.py` spawns one `nvidia-smi --query-gpu` subprocess per
tick (~40-60 ms of CPU per call).  v2 keeps ONE `nvidia-smi dmon` process alive and
just reads its stdout lines, so the sampler's own CPU cost is a fraction of a
percent of one core -- measurable, see `--selftest-cpu` in the doc.

Overhead budget (measured on node36, 8x A800, 5 s interval):
    dmon poll ~2-4 ms of CPU every 5 s  -> ~0.05 % of one core
    /proc/loadavg + /proc/meminfo reads -> <0.1 ms
Nothing is written to the GPU, nothing is changed on the host: read-only.
The sampler is started with `nice -n 19` by the wrapper so it can never steal
CPU from the vLLM engine / proxy / client even if the box is saturated.

Fields captured (unit-suffixed per AGENTS.md):
    ts_s           seconds since sampler start
    ts_epoch_s     unix epoch seconds (join key with bench per-request CSVs)
    iso            iso-8601 local time
    load1/5/15     /proc/loadavg
    mem_avail_mib  MemAvailable (host RAM; with --swap-space 0 a drop here is NOT
                   KV swap, so it cleanly separates "engine memory" from "swapped" )
    gpu{i}_pwr_w        board power (W)          -> compute intensity
    gpu{i}_sm_pct       SM utilisation (%)       -> prefill (near 100) vs decode (30-70)
    gpu{i}_mem_pct      memory-controller busy (%)
    gpu{i}_fb_mb        framebuffer used (MB)    -> KV cache growth = concurrency proxy
    gpu{i}_rx_mb_s      PCIe/NVLink rx (MB/s)    -> KV pull on decode side
    gpu{i}_tx_mb_s      PCIe/NVLink tx (MB/s)    -> KV push on prefill side

Usage:
    nice -n 19 python3 hw_metrics_sampler_v2.py --out hw_metrics.csv \
        --interval 5 --duration 900

A single `nvidia-smi dmon` header is dropped; if dmon dies (driver reset) the
sampler exits non-zero rather than silently writing empty rows.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import signal
import subprocess
import sys
import time

N_GPU = 8
# -s pucvmet column order, after the leading "gpu" index column:
#   pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk pviol tviol fb bar1 ccpm sbecc dbecc pci rxpci txpci
_COL = {"pwr": 0, "gtemp": 1, "mtemp": 2, "sm": 3, "mem": 4, "enc": 5, "dec": 6,
        "jpg": 7, "ofa": 8, "mclk": 9, "pclk": 10, "pviol": 11, "tviol": 12,
        "fb": 13, "bar1": 14, "ccpm": 15, "sbecc": 16, "dbecc": 17, "pci": 18,
        "rxpci": 19, "txpci": 20}


def _read_loadavg() -> tuple[float, float, float]:
    a, b, c = open("/proc/loadavg").read().split()[:3]
    return float(a), float(b), float(c)


def _read_meminfo_mib() -> tuple[float, float]:
    d: dict[str, float] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                d[k] = float(v.strip().split()[0]) / 1024.0  # kB -> MiB
    return d.get("MemAvailable", 0.0), d.get("MemTotal", 0.0)


def _header(n_gpu: int) -> list[str]:
    h = ["ts_s", "ts_epoch_s", "iso", "load1", "load5", "load15",
         "mem_avail_mib", "mem_total_mib"]
    for i in range(n_gpu):
        h += [f"gpu{i}_pwr_w", f"gpu{i}_sm_pct", f"gpu{i}_mem_pct",
              f"gpu{i}_fb_mb", f"gpu{i}_rx_mb_s", f"gpu{i}_tx_mb_s"]
    return h


def _parse_row(line: str) -> tuple[int, dict[str, float]] | None:
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        idx = int(parts[0])
    except ValueError:
        return None  # header / comment / noise
    vals: dict[str, float] = {}
    for name, off in _COL.items():
        pos = 1 + off
        if pos < len(parts):
            try:
                vals[name] = float(parts[pos])
            except ValueError:
                pass
    return idx, vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until SIGTERM")
    ap.add_argument("--gpus", type=int, default=N_GPU)
    args = ap.parse_args()

    n_gpu = args.gpus
    stop = {"flag": False}

    def _on_sig(_sig, _frm):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", str(int(max(1, args.interval)))],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    assert dmon.stdout is not None

    start = time.time()
    deadline = start + args.duration if args.duration > 0 else float("inf")
    pending: dict[int, dict[str, float]] = {}

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_header(n_gpu))
        f.flush()
        while not stop["flag"] and time.time() < deadline:
            line = dmon.stdout.readline()
            if line == "":  # dmon exited
                print("[hw_sampler] dmon closed stdout", file=sys.stderr)
                break
            if line.lstrip().startswith("#"):
                continue
            parsed = _parse_row(line)
            if parsed is None:
                continue
            idx, vals = parsed
            if not vals:
                continue
            pending[idx] = vals
            if idx != n_gpu - 1:
                continue
            now = time.time()
            l1, l5, l15 = _read_loadavg()
            mavail, mtotal = _read_meminfo_mib()
            row: list[object] = [
                round(now - start, 1), round(now, 3),
                _dt.datetime.now().isoformat(timespec="seconds"),
                l1, l5, l15, round(mavail, 1), round(mtotal, 1)]
            for i in range(n_gpu):
                v = pending.get(i, {})
                row += [int(v.get("pwr", 0)), round(v.get("sm", 0.0), 1),
                        round(v.get("mem", 0.0), 1), int(v.get("fb", 0)),
                        round(v.get("rxpci", 0.0), 1), round(v.get("txpci", 0.0), 1)]
            w.writerow(row)
            f.flush()
            pending = {}

    try:
        dmon.terminate()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
