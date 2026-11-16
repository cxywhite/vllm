#!/usr/bin/env python3
# [WT] 只读：把「同一数据集」的所有运行（任何方法/variant）排在一起比 goodput。
#   同数据集判据 = Total generated tokens 匹配
#     lmsys    -> 8,648,897
#     sharegpt -> 9,590,221
#   token 明显不足的（被中断/截断）会被单独标出来，避免污染排名。
# 用法: python3 wt_scan_all_methods.py
import glob
import os
import re

BASE = {8648897: "lmsys", 9590221: "sharegpt"}
ROOTS = ["/root/myshare/wty_tmp_result", "/root/myshare/tmp_dynamo_result"]


def g(t, p):
    m = re.search(p + r"\s*:?\s*([\d.]+)", t)
    return float(m.group(1)) if m else None


rows = []
for R in ROOTS:
    for d in glob.glob(R + "/**/", recursive=True):
        name = os.path.basename(d.rstrip("/"))
        # 只认 run 目录：必须含 pd 标记
        if not re.search(r"pd\dp\dd", name):
            continue
        logs = glob.glob(d + "log/bench*.log") + glob.glob(d + "**/bench*.log", recursive=True)
        t = None
        for L in logs:
            s = open(L, errors="ignore").read()
            if "Output token goodput" in s:
                t = s
                break
        if t is None:
            continue
        tok = g(t, r"Total generated tokens")
        if tok is None:
            continue
        ds, dev = None, None
        for base, nm in BASE.items():
            if abs(tok - base) / base <= 0.01:      # 1% 容差（个位数请求返 0 token）
                ds, dev = nm, tok / base - 1
        pd = (re.search(r"pd(\dp\dd)", name) or [None, "?"])[1]
        # 方法名 = 目录名最后一段
        m = re.search(r"_rep1_\d{8}_\d{6}(_.*)$", name)
        method = m.group(1).lstrip("_") if m else "?"
        if method == "?" or "dynamo" in name:
            run = os.path.basename(os.path.dirname(d.rstrip("/")))
            fam = "lmsys" if "lmsys" in name else "sharegpt" if "sharegpt" in name else "?"
            method = f"dynamo {fam}"
        rows.append(dict(ds=ds, tok=tok, pd=pd, method=method, dev=dev or 0.0,
                         good=g(t, r"Output token goodput \(tok/s\)"),
                         thr=g(t, r"Output token throughput \(tok/s\)"),
                         ttft=g(t, r"Mean TTFT \(ms\)"), tpot=g(t, r"Mean TPOT \(ms\)"),
                         dir=os.path.basename(d.rstrip("/")),
                         src="dynamo" if "dynamo" in d else "ours"))

print("=== 同一数据集的所有运行（goodput@30ms 降序）===")
for ds in ("lmsys", "sharegpt"):
    for pd in ("4p4d", "2p6d"):
        sub = [r for r in rows if r["ds"] == ds and r["pd"] == pd]
        if not sub:
            continue
        print(f"\n--- {ds} {pd} ---")
        print(f"{'goodput':>8} {'thru':>7} {'TTFT':>9} {'TPOT':>7} {'Δtok':>7}  {'src':6} method")
        for r in sorted(sub, key=lambda r: -(r["good"] or 0)):
            print(f"{r['good']:>8.1f} {r['thr']:>7.0f} {r['ttft']:>9.1f} {r['tpot']:>7.2f} "
                  f"{r['dev']*100:>+6.2f}%  {r['src']:6} {r['method'][:40]}")
# 被排除的（token 不匹配）
bad = [r for r in rows if r["ds"] is None]
if bad:
    print(f"\n=== 被排除（token 总量不匹配，非同一数据集或已截断）: {len(bad)} 个 ===")
    for r in sorted(bad, key=lambda r: -(r["tok"] or 0))[:12]:
        print(f"  tokens={r['tok']:>10,.0f}  {r['pd']}  {r['method'][:40]}")
