#!/usr/bin/env python3
# [WT] 只读：对比 Ours vs RR 的逐请求 TPOT / TTFT 分布，判别是
#   (a) 「长尾恶化」型（p50 接近、p99 差）-> 调度/饥饿，策略问题
#   (b) 「整体抬高」型（p50 就相差很多）-> 每步更慢，实现/数据通路问题
# 注意：CSV 里 latency/ttft/tpot 单位是【秒】。文件含换行字段，必须用 csv.reader。
# 用法: python3 wt_tpot_shape.py
import csv
import glob
import os
import re
import statistics
import sys

csv.field_size_limit(sys.maxsize)
ROOT = "/root/myshare/wty_tmp_result/qwentrace_rerun1"


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, int(len(xs) * p))
    return xs[i]


def load(d):
    f = glob.glob(d + "/dataset_result/test_results_*.csv")
    if not f:
        return None
    rows = []
    with open(f[0], newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            try:
                ot = int(float(r["output_tokens"]))
                tpot = float(r["tpot"]) * 1000.0
                ttft = float(r["ttft"]) * 1000.0
                el = float(r["expect_output_len"])
            except (ValueError, KeyError, TypeError):
                continue
            rows.append((ot, tpot, ttft, el))
    return rows


def desc(rows, label):
    tp = [t for ot, t, _, _ in rows if ot >= 2 and t == t]
    tt = [t for _, _, t, _ in rows if t == t]
    # 只看「中等长度」请求（<4096 tokens），排除必然慢的超长请求
    tp_mid = [t for ot, t, _, _ in rows if 2 <= ot < 4096 and t == t]
    # 只看长请求（>=4096）
    tp_long = [t for ot, t, _, _ in rows if ot >= 4096 and t == t]
    print(f"  {label:6} n={len(rows):5}  "
          f"TPOT p50={q(tp,.5):6.2f} p90={q(tp,.9):7.2f} p99={q(tp,.99):8.2f} "
          f"mean={statistics.mean(tp):7.2f}  "
          f"| TPOT<30ms 占比 {sum(1 for t in tp if t<=30)/len(tp)*100:5.1f}%")
    print(f"  {'':6}       TTFT p50={q(tt,.5):7.1f} p90={q(tt,.9):8.1f} p99={q(tt,.99):9.1f} "
          f"mean={statistics.mean(tt):8.1f}")
    print(f"  {'':6}       TPOT(<4096tok) n={len(tp_mid):5} p50={q(tp_mid,.5):6.2f} "
          f"p90={q(tp_mid,.9):7.2f} mean={statistics.mean(tp_mid):7.2f}"
          f"   | TPOT(>=4096tok) n={len(tp_long):4} p50={q(tp_long,.5):7.2f}"
          if tp_long else "")
    return tp, tp_mid


def main():
    for tr in ("lmsys", "sharegpt"):
        for pd in ("4p4d", "2p6d"):
            pair = {}
            for var, tag in (("scheme4_aimd", "Ours"), ("roundrobin", "RR")):
                ds = [d for d in glob.glob(ROOT + "/*/")
                      if re.search(rf"pd{pd}_tpot30ms_trace\S*?ours_{tr}_", d)
                      and d.rstrip("/").endswith(var)]
                if not ds:
                    continue
                rows = load(ds[0].rstrip("/"))
                if rows:
                    pair[tag] = (rows, ds[0].rstrip("/"))
            if len(pair) < 2:
                continue
            print(f"=== {tr} {pd}  (Ours vs RR) ===")
            a = desc(pair["Ours"][0], "Ours")
            b = desc(pair["RR"][0], "RR")
            if a and b:
                print(f"  -> TPOT p50 比 Ours/RR = {q(a[0],.5)/q(b[0],.5):.3f}   "
                      f"<30ms 占比比 = {sum(1 for t in a[0] if t<=30)/len(a[0])/max(sum(1 for t in b[0] if t<=30)/len(b[0]),1e-9):.3f}")
            print()


if __name__ == "__main__":
    main()
