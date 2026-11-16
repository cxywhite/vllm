#!/usr/bin/env python3
# [WT] 只读：把所有 Dynamo 历史运行按「同配置」过滤后对比。
#   同配置判据：Total generated tokens 必须等于 qwentrace 基准
#     lmsys   -> 8,648,897
#     sharegpt-> 9,590,221
#   （不同 token 总量 = 不同数据集/不同 trace 长度，不可比）
import glob
import os
import re

BASE = {"lmsys": 8648897, "sharegpt": 9590221, "ds5": 8648897}
ROOTS = ["/root/myshare/tmp_dynamo_result", "/root/myshare/dynamo",
         "/root/myshare/paper_figures/fig12_goodput_comparison/results/dynamo_0630",
         "/root/myshare/0802_wty_result/4080_result_llama/dynamo",
         "/root/myshare/models/wt_predictor/0630/dynamo_0630",
         "/root/myshare/0802_wty_result"]


def g(t, p):
    m = re.search(p + r"\s*:?\s*([\d.]+)", t)
    return float(m.group(1)) if m else None


def find_run_label(d):
    p = os.path.dirname(d.rstrip("/"))
    for _ in range(4):
        b = os.path.basename(p)
        if re.match(r"(run_|qwen_dynamo)", b):
            return b
        p = os.path.dirname(p)
    return "?"


rows, seen = [], set()
for R in ROOTS:
    for d in glob.glob(R + "/**/", recursive=True):
        name = os.path.basename(d.rstrip("/"))
        if "dynamo" not in name:
            continue
        key = os.path.realpath(d)
        if key in seen:
            continue
        t = None
        for L in glob.glob(d + "log/bench*.log") + glob.glob(d + "**/bench*.log", recursive=True):
            s = open(L, errors="ignore").read()
            if "Output token goodput" in s:
                t = s
                break
        if t is None:
            continue
        seen.add(key)
        tok = g(t, r"Total generated tokens")
        tr = ("lmsys" if "lmsys" in name else
              "sharegpt" if "sharegpt" in name else
              "ds5" if ("dataset5" in d or "predictor-friendly" in name) else "?")
        base = BASE.get(tr)
        if base is None or tok is None or abs(tok - base) > 1:
            continue
        pd = (re.search(r"pd(\dp\dd)", name) or [None, "?"])[1]
        rows.append(dict(run=find_run_label(d), tr=tr, pd=pd,
                         good=g(t, r"Output token goodput \(tok/s\)"),
                         thr=g(t, r"Output token throughput \(tok/s\)"),
                         tok=tok, ttft=g(t, r"Mean TTFT \(ms\)"),
                         tpot=g(t, r"Mean TPOT \(ms\)"),
                         dur=g(t, r"Benchmark duration \(s\)"), dir=os.path.realpath(d)))

rows.sort(key=lambda r: (r["tr"], r["pd"], r["run"]))
print("=== 同配置 Dynamo 运行（token 总量严格匹配）===")
print(f"{'trace':10} {'pd':5} {'run':24} {'goodput':>8} {'thru':>6} {'TPOT':>7} {'TTFT':>8} {'dur':>6}")
for r in rows:
    print(f"{r['tr']:10} {r['pd']:5} {r['run'][:24]:24} {r['good']:>8.1f} {r['thr']:>6.0f} "
          f"{r['tpot']:>7.2f} {r['ttft']:>8.1f} {r['dur']:>6.0f}")
print()
print("=== 同配置离散度（goodput@30ms）===")
for tr in ("lmsys", "sharegpt", "ds5"):
    for pd in ("2p6d", "4p4d"):
        v = [r for r in rows if r["tr"] == tr and r["pd"] == pd]
        if len(v) < 2:
            continue
        gs = [r["good"] for r in v]
        print(f"  {tr:10} {pd:5} n={len(v)}  {min(gs):8.1f} ~ {max(gs):8.1f}   "
              f"max/min = {max(gs)/max(min(gs), 1e-9):8.2f}x")
        for r in v:
            print(f"        {r['run'][:24]:24} goodput={r['good']:>8.1f}  TPOT={r['tpot']:>6.2f}  "
                  f"thru={r['thr']:>6.0f}")
