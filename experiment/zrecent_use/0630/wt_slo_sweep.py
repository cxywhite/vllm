#!/usr/bin/env python3
# [WT] 离线 SLO 扫描：用已有的 per-request CSV 重算不同 TPOT-SLO 下的 goodput。
#
# 动机（2026-09-25 发现）：
#   Dynamo 在**同一批次、同一配置**下，8 小时内 goodput 从 64.69 变到 853.53（13.2×），
#   而 throughput 只变了 1.17×、mean TPOT 只变了 1.47×。
#   原因：TPOT-SLO=30ms 远低于这台机器的可达 TPOT 下限（≈38ms），
#   达标率完全由 TPOT 分布的**左尾**决定，机器速度小幅波动会让左尾面积成倍变化。
#   => goodput@30ms 是一个极端脆弱的指标，跨窗口不可比。
#
# 本脚本量化这个脆弱性：把 SLO 从 20ms 扫到 100ms，看
#   (a) 达标率/goodput 对 SLO 的弹性
#   (b) RR vs Ours 的排名在哪个 SLO 区间才稳定
#
# ⚠️ CSV 里 ttft/tpot/latency 单位是**秒**；必须用 csv.reader（有内嵌换行）。
#
# 用法：python3 wt_slo_sweep.py [--slos 20,25,30,40,50,60,80] [--ttft-slo 1000]

import argparse
import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)

# 同数据集(dataset3/qwentrace n4080) 的 09-24 背靠背配对
ROOT = "/root/myshare/wty_tmp_result/retry_qwen_4p4d"
RUNS = [
    ("lmsys",    "4p4d", "RR",   "20260924_193238_roundrobin"),
    ("lmsys",    "4p4d", "Ours", "20260924_184744_scheme4_gwf"),
    ("sharegpt", "4p4d", "Ours", "20260924_123238_scheme4_gwf"),
    ("sharegpt", "4p4d", "Ours", "20260924_160128_scheme4_gwf"),
    ("sharegpt", "4p4d", "RR",   "20260924_165303_roundrobin"),
    ("lmsys",    "2p6d", "Ours", "20260924_213634_scheme4_gwf"),
    ("lmsys",    "2p6d", "RR",   "20260924_221135_roundrobin"),
    ("sharegpt", "2p6d", "Ours", "20260924_202549_scheme4_gwf"),
    ("sharegpt", "2p6d", "RR",   "20260924_205952_roundrobin"),
]


def find_run(src, pd, ts):
    pat = os.path.join(ROOT, f"qwen_qwen-{'lmsys' if src=='lmsys' else 'mysharegpt'}*"
                            f"n4080_pd{pd}_tpot30ms*{ts}")
    hits = glob.glob(pat)
    return hits[0] if hits else None


def load(d):
    fs = glob.glob(os.path.join(d, "dataset_result", "test_results_*.csv"))
    if not fs:
        return None
    with open(sorted(fs)[-1], newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        ix = {k: hdr.index(k) for k in
              ("output_tokens", "ttft", "tpot", "send_timestamp_ms", "latency")}
        out = []
        t0 = t1 = None
        for row in r:
            try:
                ot = float(row[ix["output_tokens"]])
                tt = float(row[ix["ttft"]]) * 1000
                tp = row[ix["tpot"]]
                tp = float(tp) * 1000 if tp not in ("", None) else None
                st = float(row[ix["send_timestamp_ms"]])
                la = float(row[ix["latency"]])
            except (ValueError, IndexError):
                continue
            out.append((ot, tt, tp))
            t0 = st if t0 is None else min(t0, st)
            t1 = max(t1 or 0, st + la * 1000)
    return out, (t1 - t0) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slos", default="20,25,30,40,50,60,80,100")
    ap.add_argument("--ttft-slo", type=float, default=1000.0)
    a = ap.parse_args()
    slos = [float(x) for x in a.slos.split(",")]

    data = {}
    for src, pd, meth, ts in RUNS:
        d = find_run(src, pd, ts)
        if not d:
            print(f"[跳过] 找不到 {src} {pd} {meth} {ts}")
            continue
        res = load(d)
        if not res:
            print(f"[跳过] 无结果 CSV: {os.path.basename(d)}")
            continue
        data[(src, pd, meth, ts)] = res
        print(f"[载入] {src:<9}{pd:<6}{meth:<5}{ts}  {len(res[0])} 条, "
              f"时长 {res[1]:.0f}s")

    print("\n" + "=" * 100)
    print("[WT] goodput (tok/s) 随 TPOT-SLO 变化  ——  同一 run 只换 SLO 阈值")
    print("=" * 100)
    hdr = f"{'数据集':<10}{'PD':<6}{'方法':<6}{'启动':<16}"
    for s in slos:
        hdr += f"{'SLO'+str(int(s)):>9}"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(data, key=lambda x: (x[0], x[1], x[3])):
        src, pd, meth, ts = k
        recs, dur = data[k]
        line = f"{src:<10}{pd:<6}{meth:<6}{ts[:15]:<16}"
        for s in slos:
            gp = sum(ot for ot, tt, tp in recs
                     if tt <= a.ttft_slo and (tp is None or tp <= s))
            line += f"{gp/dur:>9.1f}"
        print(line)

    print("\n" + "=" * 100)
    print("[WT] 达标率 (%) 随 TPOT-SLO 变化")
    print("=" * 100)
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(data, key=lambda x: (x[0], x[1], x[3])):
        src, pd, meth, ts = k
        recs, dur = data[k]
        line = f"{src:<10}{pd:<6}{meth:<6}{ts[:15]:<16}"
        for s in slos:
            ok = sum(1 for ot, tt, tp in recs
                     if tt <= a.ttft_slo and (tp is None or tp <= s))
            line += f"{ok/len(recs)*100:>8.1f}%"
        print(line)

    print("\n" + "=" * 100)
    print("[WT] Ours / RR 比值 随 TPOT-SLO 变化  ——  看排名在哪个 SLO 区间稳定")
    print("=" * 100)
    print(f"{'数据集':<10}{'PD':<6}" + "".join(f"{'SLO'+str(int(s)):>9}" for s in slos))
    print("-" * 80)
    for src in ("lmsys", "sharegpt"):
        for pd in ("4p4d", "2p6d"):
            ours = [v for k, v in data.items()
                    if k[0] == src and k[1] == pd and k[2] == "Ours"]
            rr = [v for k, v in data.items()
                  if k[0] == src and k[1] == pd and k[2] == "RR"]
            if not ours or not rr:
                continue
            o, od = ours[0]
            r, rd = rr[0]
            line = f"{src:<10}{pd:<6}"
            for s in slos:
                go = sum(x for x, tt, tp in o if tt <= a.ttft_slo and (tp is None or tp <= s)) / od
                gr = sum(x for x, tt, tp in r if tt <= a.ttft_slo and (tp is None or tp <= s)) / rd
                line += f"{go/gr:>8.1f}x" if gr else f"{'n/a':>9}"
            print(line)


if __name__ == "__main__":
    main()
