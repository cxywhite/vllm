#!/usr/bin/env python3
# [WT] 资源仪表（精确版）：在实验运行期间采样，并**把宿主 CPU 占用拆成"我们自己"和"其他租户"**。
#
# 为什么要拆：
#   已证实同代码/同配置/同数据集/同模型下，2026-09-24 比 08-27 慢 1.35~1.48×（连单请求解码都慢 35%）。
#   已排除：vllm 代码、引擎初始化配置(33/33 字段相同)、config JSON、数据集文件(span_s 相同)、
#           /dev/shm 泄漏、容器 CPU 配额、CPU 降频(52核均匀2800MHz/performance)、GPU 降频(全 Not Active)。
#   用户确认 GPU 独享 → 剩下只能是宿主 CPU / 内存带宽 / 供电域被共享。
#
#   关键：容器内 /proc 只包含我们自己的进程，所以
#       我们的 CPU 用量 = sum(/proc/*/stat 的 utime+stime)
#       宿主总占用   = /proc/stat 的 busy 增量
#       其他租户     = 宿主总占用 - 我们的
#   于是可以在**同一次运行内**做因果检验：TPOT 是否随"其他租户 CPU 占用"同步恶化。
#   （不需要跨天比较，避免了窗口不同的争议）
#
# 用法：
#   nohup python3 wt_resource_watch.py <label> [interval_s] &
#   ... 跑实验 ...
#   pkill -f wt_resource_watch.py
#
# 输出：/root/myshare/wty_tmp_result/resource_watch/<label>_<ts>.csv

import csv
import os
import re
import subprocess
import sys
import time

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
OUTDIR = "/root/myshare/wty_tmp_result/resource_watch"
os.makedirs(OUTDIR, exist_ok=True)

HZ = os.sysconf("SC_CLK_TCK")
NCPU = len(os.sched_getaffinity(0))
TS = time.strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(OUTDIR, f"{LABEL}_{TS}.csv")

FIELDS = [
    "ts", "load1", "load5", "mem_avail_gb",
    "host_busy_pct", "our_cpu_pct", "other_cpu_pct", "n_our_procs",
    "gpu0_sm_mhz", "gpu0_max_sm_mhz", "gpu0_power_w", "gpu0_temp_c",
    "gpu0_util", "gpu0_mem_mib", "gpu0_throttle",
    "foreign_apps",
]


def read_host_cpu():
    """返回宿主累计 busy jiffies 和 total jiffies。"""
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu "):
                v = [int(x) for x in line.split()[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                total = sum(v)
                return total - idle, total
    return 0, 0


def read_our_cpu():
    """容器内可见的进程即我们自己的；返回 (累计 CPU 秒数, 进程数)。"""
    own = 0.0
    n = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                data = fh.read()
            close = data.rfind(b")")
            parts = data[close + 2:].split()
            own += (int(parts[11]) + int(parts[12])) / HZ
            n += 1
        except (OSError, IndexError, ValueError):
            continue
    return own, n


def read_gpu0():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,"
             "utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10)
        vals = [x.strip() for x in out.stdout.strip().split(",")]
    except Exception:
        vals = ["?"] * 6
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE", "-i", "0"],
                             capture_output=True, text=True, timeout=10)
        thr = []
        inblock = False
        for line in out.stdout.splitlines():
            if "Clocks Event Reasons" in line:
                inblock = True
                continue
            if inblock:
                if not line.strip():
                    break
                if "Active" in line and "Not Active" not in line:
                    thr.append(line.strip().split(":")[0].strip())
        vals.append(";".join(thr) if thr else "-")
    except Exception:
        vals.append("?")
    return vals


def read_apps():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        rows = []
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            rows.append(line.replace(",", " "))
        return "; ".join(rows) if rows else "-"
    except Exception:
        return "?"


def mem_avail_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return -1.0


def loadavg():
    try:
        with open("/proc/loadavg") as fh:
            f = fh.read().split()
        return float(f[0]), float(f[1])
    except Exception:
        return -1.0, -1.0


with open(CSV_PATH, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(FIELDS)

    hb, ht = read_host_cpu()
    oc, _ = read_our_cpu()

    while True:
        time.sleep(INTERVAL)

        hb2, ht2 = read_host_cpu()
        oc2, nproc_own = read_our_cpu()

        d_total = max(1, ht2 - ht)
        # host_pct: 宿主全部核的忙碌比例（100% = 52 核全满）
        host_pct = 100.0 * (hb2 - hb) / d_total
        # our_pct: 我们容器占宿主总容量的比例
        #   oc 单位是"CPU 秒"，d_total/HZ 是"wall 秒 x NCPU"= 总可用 CPU 秒
        our_pct = 100.0 * (oc2 - oc) * HZ / d_total
        other_pct = max(0.0, host_pct - our_pct)
        hb, ht, oc = hb2, ht2, oc2

        l1, l5 = loadavg()
        g = read_gpu0()
        row = [time.strftime("%F_%T"), f"{l1:.2f}", f"{l5:.2f}", f"{mem_avail_gb():.1f}",
               f"{host_pct:.1f}", f"{our_pct:.1f}", f"{other_pct:.1f}", nproc_own]
        row.extend(g[:6])
        row.append(g[6] if len(g) > 6 else "?")
        row.append(read_apps())
        writer.writerow(row)
        fh.flush()
