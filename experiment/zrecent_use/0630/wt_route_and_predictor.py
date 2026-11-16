#!/usr/bin/env python3
# [WT] 只读：查 Ours(scheme4) 的「路由分布」与「预测器精度」。
#   思路：RR 是严格的 1/k 均匀分配，所以任何显著偏离均匀的负载分布都是 Ours 自己造成的。
#   负载用 sum(actual_output_len) 衡量（decode 的真实工作量），不是请求条数。
#   同时评估预测器误差：predict_output_len vs actual_output_len。
# 用法: python3 wt_route_and_predictor.py
import csv
import glob
import os
import re
import statistics
import sys

csv.field_size_limit(sys.maxsize)
ROOT = "/root/myshare/wty_tmp_result/qwentrace_rerun1"


def load_instance(f):
    out = []
    with open(f, newline="", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            try:
                out.append((int(float(r["actual_output_len"])),
                            float(r["predict_output_len"]),
                            float(r["predict_output_risk"]) if r.get("predict_output_risk") else None,
                            float(r["actual_minus_predict_mean"]) if r.get("actual_minus_predict_mean") else None))
            except (ValueError, TypeError, KeyError):
                continue
    return out


def analyze(d, label):
    files = sorted(glob.glob(d + "/log/predict_request_decode*.csv"))
    if not files:
        print(f"  [{label}] 无 predict_request CSV")
        return
    per = {}
    for f in files:
        inst = re.search(r"decode(\d+)", os.path.basename(f)).group(1)
        rows = load_instance(f)
        if rows:
            per[inst] = rows
    k = len(per)
    cnt = {i: len(v) for i, v in per.items()}
    tok = {i: sum(x[0] for x in v) for i, v in per.items()}
    allr = [x for v in per.values() for x in v]
    T = sum(tok.values())
    N = sum(cnt.values())
    print(f"  [{label}] 实例 {k} 个，共 {N} 请求 / {T:,} 输出 token")
    print(f"    请求条数: " + "  ".join(f"{i}:{cnt[i]}" for i in sorted(cnt))
          + f"   | 均匀应为 {N//k}  max/min={max(cnt.values())/min(cnt.values()):.3f}")
    print(f"    输出token: " + "  ".join(f"{i}:{tok[i]/1000:.0f}k" for i in sorted(tok))
          + f"   | 均匀应为 {T//k/1000:.0f}k  max/min={max(tok.values())/min(tok.values()):.3f}"
          + f"  最大偏离 {(max(tok.values())/(T/k)-1)*100:+.1f}%")
    # 预测器精度
    a = [x[0] for x in allr]
    p = [x[1] for x in allr]
    mae = statistics.mean(abs(x[0] - x[1]) for x in allr)
    bias = statistics.mean(x[0] - x[1] for x in allr)
    def corr(x, y):
        mx, my = statistics.mean(x), statistics.mean(y)
        num = sum((i - mx) * (j - my) for i, j in zip(x, y))
        den = (sum((i - mx) ** 2 for i in x) * sum((j - my) ** 2 for j in y)) ** .5
        return num / den if den else float("nan")
    print(f"    预测器: actual_mean={statistics.mean(a):.0f}  predict_mean={statistics.mean(p):.0f}  "
          f"MAE={mae:.0f}  bias={bias:+.0f}  corr={corr(p, a):.3f}")
    # 逐实例的平均输出长度（看是否把长请求集中到某一台）
    print(f"    各实例平均输出长度: " + "  ".join(
        f"{i}:{statistics.mean(x[0] for x in per[i]):.0f}" for i in sorted(per)))


def main():
    for tr in ("lmsys", "sharegpt"):
        for pd in ("4p4d", "2p6d"):
            ds = [d for d in glob.glob(ROOT + "/*/")
                  if re.search(rf"pd{pd}_tpot30ms_trace\S*?ours_{tr}_", d)
                  and d.rstrip("/").endswith("scheme4_aimd")]
            if ds:
                print(f"=== {tr} {pd} ===")
                analyze(ds[0].rstrip("/"), "Ours")
                print()


if __name__ == "__main__":
    main()
