#!/usr/bin/env python3
# [WT] 只读：查「长尾请求」在时间上的分布，判断它是否与 🔴Remove 事件重合。
#   - 从 test_results_*.csv 取每条请求的 send_timestamp_ms 与 ttft
#   - 找出 ttft 超阈值的请求，看它们集中在哪个时间点
#   - 从 proxy 日志取 🔴/🔵 事件的上下文，推测发生时刻
# 用法: python3 wt_tail_vs_remove.py <run_dir>
import csv
import glob
import os
import re
import sys

csv.field_size_limit(sys.maxsize)


def main(d):
    d = d.rstrip("/")
    tag = os.path.basename(d)[-8:]
    print(f"=== {tag} ===")
    f = glob.glob(d + "/dataset_result/test_results_*.csv")
    if not f:
        print("  无结果 CSV")
        return
    rows = []
    with open(f[0], newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((float(r["send_timestamp_ms"]),
                             float(r["ttft"]) * 1000.0,
                             float(r["tpot"]) * 1000.0,
                             int(float(r["output_tokens"]))))
            except (ValueError, KeyError, TypeError):
                continue
    if not rows:
        print("  无有效行")
        return
    t0 = min(x[0] for x in rows)
    tot = max(x[0] for x in rows) - t0
    print(f"  发送窗口 {tot/1000:.0f}s，共 {len(rows)} 条")

    for thr, name in ((5000, "TTFT>5s"), (1000, "TTFT>1s"), (3000, "TPOT>3s")):
        idx = 1 if name.startswith("TTFT") else 2
        bad = [x for x in rows if x[idx] > thr]
        if not bad:
            print(f"  {name}: 0 条")
            continue
        # 按 10s 分桶
        buckets = {}
        for x in bad:
            buckets[int((x[0] - t0) // 10000)] = buckets.get(int((x[0] - t0) // 10000), 0) + 1
        top = sorted(buckets.items(), key=lambda kv: -kv[1])[:6]
        print(f"  {name}: {len(bad)} 条 ({len(bad)/len(rows)*100:.1f}%)，时间集中在 " +
              "  ".join(f"{b*10}s:{n}" for b, n in sorted(top)))

    # proxy 事件
    P = sorted(glob.glob(d + "/log/proxy_*.log"))
    if P:
        lines = open(P[0], errors="ignore").read().splitlines()
        for kw in ("🔴Remove", "🔵Add"):
            pos = [i for i, l in enumerate(lines) if kw in l]
            if pos:
                pct = [f"{i/len(lines)*100:.0f}%" for i in pos[:8]]
                print(f"  {kw}: {len(pos)} 次，位于日志进度 {', '.join(pct)}")
                for i in pos[:3]:
                    print(f"      {lines[i][:150]}")


for a in sys.argv[1:]:
    main(a)
    print()
