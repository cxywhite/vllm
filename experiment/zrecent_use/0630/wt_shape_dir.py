#!/usr/bin/env python3
# [WT] 只读：对任意 run 目录输出逐请求 TTFT/TPOT 分布 + 达标率。
# 用法: python3 wt_shape_dir.py <run_dir> [run_dir ...]
# 注意: test_results_*.csv 里 latency/ttft/tpot 单位是【秒】，且含换行字段。
import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def stats(d):
    f = glob.glob(d + "/dataset_result/test_results_*.csv")
    if not f:
        return None
    tp, tt, ot = [], [], []
    with open(f[0], newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            try:
                o = int(float(r["output_tokens"]))
                a = float(r["tpot"]) * 1000.0
                b = float(r["ttft"]) * 1000.0
            except (ValueError, KeyError, TypeError):
                continue
            if o >= 2 and a == a:
                tp.append(a)
            if b == b:
                tt.append(b)
            ot.append(o)
    return tp, tt, ot


def show(d):
    s = stats(d)
    tag = os.path.basename(d.rstrip("/"))[-8:]
    if not s:
        print(f"{tag}: 无结果 CSV")
        return
    tp, tt, ot = s
    n = len(ot)
    print(f"{tag}  n={n}")
    print(f"   TPOT  p50={q(tp,.5):7.2f} p90={q(tp,.9):8.2f} p99={q(tp,.99):9.2f} "
          f" | <=30ms {sum(1 for x in tp if x<=30)/len(tp)*100:5.1f}%  <=60ms {sum(1 for x in tp if x<=60)/len(tp)*100:5.1f}%")
    print(f"   TTFT  p50={q(tt,.5):7.1f} p90={q(tt,.9):8.1f} p99={q(tt,.99):9.1f} "
          f" | <=1000ms {sum(1 for x in tt if x<=1000)/len(tt)*100:5.1f}%")
    # 单指标拆解（bench 的 goodput 是双达标 -> 分开看）
    ok_t = sum(1 for x in tt if x <= 1000)
    ok_p = sum(1 for x in tp if x <= 30)
    print(f"   达标数 TTFT={ok_t}/{len(tt)}  TPOT={ok_p}/{len(tp)}")
    # 输出长度分位（确认两边 workload 一致）
    print(f"   输出长度 p50={q(ot,.5):6.0f} p90={q(ot,.9):7.0f} p99={q(ot,.99):7.0f} "
          f"sum={sum(ot):,}")


for d in sys.argv[1:]:
    show(d)
    print()
