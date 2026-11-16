#!/usr/bin/env python3
# [WT] 全盘扫描 burstgpt 结果，按 模型/数据集/PD/方法 汇总
import glob
import os
import re

roots = ["/root/myshare/wty_tmp_result", "/root/myshare/paper_figures",
         "/root/myshare/models/wt_predictor", "/root/predict-schedule/experiment"]
dirs = set()
for r in roots:
    for p in glob.glob(os.path.join(r, "**", "*burstgpt*"), recursive=True):
        dirs.add(p if os.path.isdir(p) else os.path.dirname(p))

rows = []
for d in sorted(dirs):
    logs = glob.glob(os.path.join(d, "log", "bench_*.log"))
    logs += glob.glob(os.path.join(d, "benchmark.log"))
    if not logs:
        continue
    txt = open(logs[0], errors="ignore").read()

    def g(pat):
        m = re.search(pat + r":\s*([\d.]+)", txt)
        return float(m.group(1)) if m else None

    gp, req = g(r"Output token goodput \(tok/s\)"), g(r"Successful requests")
    thr, ttft, tpot = (g(r"Output token throughput \(tok/s\)"),
                       g(r"Mean TTFT \(ms\)"), g(r"Mean TPOT \(ms\)"))
    n = os.path.basename(d)
    mdl = n.split("_")[0]
    src = "lmsys" if "lmsys" in n else ("sharegpt" if "sharegpt" in n else "?")
    m_pd = re.search(r"pd(\dp\dd)", n)
    pd = m_pd.group(1) if m_pd else "?"
    meth = ("Dynamo" if "dynamo" in n else "RR" if "roundrobin" in n
            else "Ours" if "scheme4_gwf" in n else "?")
    m_ts = re.search(r"(20\d{6})_\d{6}", n)
    date = m_ts.group(1) if m_ts else "?"
    m_tp = re.search(r"tpot(\d+)ms", n)
    tpot_slo = m_tp.group(1) if m_tp else "?"
    rows.append({"date": date, "model": mdl, "src": src, "pd": pd, "meth": meth,
                 "tpot_slo": tpot_slo, "gp": gp, "req": req, "thr": thr,
                 "ttft": ttft, "tpot": tpot, "path": n})

print(f"扫描到 {len(rows)} 条有 bench 日志的 burstgpt 结果\n")
hdr = (f"{'日期':<10}{'模型':<7}{'数据':<10}{'PD':<7}{'SLO':<5}{'方法':<8}"
       f"{'req':>5}{'goodput':>9}{'thru':>7}{'TTFT':>8}{'TPOT':>7}")
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x["date"], x["model"], x["src"], x["pd"],
                                     x["tpot_slo"], x["meth"])):
    def fm(v, w, d=1):
        return f"{v:>{w}.{d}f}" if isinstance(v, float) else f"{'-':>{w}}"
    print(f"{r['date']:<10}{r['model']:<7}{r['src']:<10}{r['pd']:<7}"
          f"{r['tpot_slo']:<5}{r['meth']:<8}"
          f"{(int(r['req']) if r['req'] else 0):>5}"
          f"{fm(r['gp'],9,2)}{fm(r['thr'],7,0)}{fm(r['ttft'],8,1)}{fm(r['tpot'],7,2)}"
          f"  {r['tpot_slo'] and ''}")

print("\n--- 覆盖矩阵（goodput，仅 tpot=30）---")
cover = {}
for r in rows:
    if r["tpot_slo"] != "30":
        continue
    k = (r["model"], r["src"], r["pd"])
    cover.setdefault(k, {})[r["meth"]] = r["gp"]
print(f"{'模型':<7}{'数据':<10}{'PD':<7}{'RR':>10}{'Ours':>10}{'Dynamo':>10}")
for k in sorted(cover):
    c = cover[k]

    def f(v):
        return f"{v:>10.2f}" if isinstance(v, float) else f"{'—':>10}"
    print(f"{k[0]:<7}{k[1]:<10}{k[2]:<7}{f(c.get('RR'))}{f(c.get('Ours'))}{f(c.get('Dynamo'))}")
