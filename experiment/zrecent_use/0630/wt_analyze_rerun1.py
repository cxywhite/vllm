#!/usr/bin/env python3
# [WT] 只读分析：对比 qwentrace_rerun1 批次里 Ours vs RR 的
#   1) bench 指标配对表（含 token 总量核对）
#   2) decode 并发分布（解释「吞吐更高但 TPOT 更差」）
# 用法: python3 wt_analyze_rerun1.py
import glob
import os
import re
import statistics

ROOT = "/root/myshare/wty_tmp_result/qwentrace_rerun1"
CONC = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s, "
                  r"Running: (\d+) reqs, Waiting: (\d+) reqs")


def bench(d):
    for L in glob.glob(d + "/log/bench_*.log"):
        t = open(L, errors="ignore").read()
        if "Output token goodput" not in t:
            continue
        def g(p):
            m = re.search(p + r"\s*:?\s*([\d.]+)", t)
            return float(m.group(1)) if m else None
        return dict(
            good=g(r"Output token goodput \(tok/s\)"),
            thr=g(r"Output token throughput \(tok/s\)"),
            tok=g(r"Total generated tokens"),
            ttft=g(r"Mean TTFT \(ms\)"), tpot=g(r"Mean TPOT \(ms\)"),
            dur=g(r"Benchmark duration \(s\)"),
            succ=g(r"Successful requests"))
    return None


def conc(d):
    per = {}
    for f in sorted(glob.glob(d + "/log/decode*.log")):
        inst = os.path.basename(f).split("_")[0]
        rows = [int(m.group(2)) for m in
                (CONC.search(l) for l in open(f, errors="ignore")) if m]
        if rows:
            per[inst] = rows
    if not per:
        return None
    n = min(len(v) for v in per.values())
    tot = [sum(v[i] for v in per.values()) for i in range(n)]
    return per, tot


def main():
    runs = {}
    for d in glob.glob(ROOT + "/*/"):
        name = os.path.basename(d.rstrip("/"))
        m = re.search(r"pd(\dp\dd)_tpot30ms_trace\S*?ours_(lmsys|sharegpt)_rep1_(\d{8})_(\d{6})_(scheme4_aimd|roundrobin)", name)
        if not m:
            continue
        b = bench(d)
        if b:
            runs[(m.group(2), m.group(1), m.group(5))] = (d, b)

    print("=" * 100)
    print("【1】bench 配对表（token 总量核对 + 指标）")
    print("=" * 100)
    hdr = f"{'round':14} {'variant':5} {'goodput':>8} {'thru':>7} {'tokens':>10} {'TTFT':>8} {'TPOT':>7} {'dur':>6}"
    print(hdr)
    for tr in ("lmsys", "sharegpt"):
        for pd in ("4p4d", "2p6d"):
            for var in ("scheme4_aimd", "roundrobin"):
                if (tr, pd, var) not in runs:
                    continue
                _, b = runs[(tr, pd, var)]
                print(f"{tr + '-' + pd:14} {'Ours' if var == 'scheme4_aimd' else 'RR':5} "
                      f"{b['good']:>8.1f} {b['thr']:>7.0f} {b['tok']:>10,.0f} "
                      f"{b['ttft']:>8.1f} {b['tpot']:>7.2f} {b['dur']:>6.0f}")
    print()
    print("配对比值（Ours / RR）：")
    print(f"{'round':14} {'Δtok%':>8} {'goodput比':>10} {'thru比':>8} {'TTFT比':>8} {'TPOT比':>8} {'dur比':>7}")
    for tr in ("lmsys", "sharegpt"):
        for pd in ("4p4d", "2p6d"):
            o = runs.get((tr, pd, "scheme4_aimd"))
            q = runs.get((tr, pd, "roundrobin"))
            if not o or not q:
                continue
            _, bo = o
            _, bq = q
            print(f"{tr + '-' + pd:14} {(bo['tok']/bq['tok']-1)*100:>+7.3f}% "
                  f"{bo['good']/bq['good']:>10.3f} {bo['thr']/bq['thr']:>8.3f} "
                  f"{bo['ttft']/bq['ttft']:>8.3f} {bo['tpot']/bq['tpot']:>8.3f} "
                  f"{bo['dur']/bq['dur']:>7.3f}")

    print()
    print("=" * 100)
    print("【2】decode 并发分布（每 10s 采样，含各实例均值以看均衡度）")
    print("=" * 100)
    print(f"{'round':14} {'variant':5} {'并发mean':>9} {'p50':>5} {'p95':>6} {'max':>6} "
          f"{'合计mean':>9} {'合计max':>8}  各实例均值")
    for tr in ("lmsys", "sharegpt"):
        for pd in ("4p4d", "2p6d"):
            for var in ("scheme4_aimd", "roundrobin"):
                if (tr, pd, var) not in runs:
                    continue
                d = runs[(tr, pd, var)][0]
                c = conc(d)
                if not c:
                    print(f"{tr + '-' + pd:14} {var:5} 无并发日志")
                    continue
                per, tot = c
                a = [r for v in per.values() for r in v]
                a_sorted = sorted(a)
                means = [round(statistics.mean(v), 1) for v in per.values()]
                print(f"{tr + '-' + pd:14} {'Ours' if var == 'scheme4_aimd' else 'RR':5} "
                      f"{statistics.mean(a):>9.1f} {statistics.median(a):>5.0f} "
                      f"{a_sorted[int(len(a_sorted)*.95)]:>6} {max(a):>6} "
                      f"{statistics.mean(tot):>9.1f} {max(tot):>8}  {means}")


if __name__ == "__main__":
    main()
