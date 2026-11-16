#!/usr/bin/env python3
# [WT] 只读：量化预测器的真实精度（用真正使用了预测的那轮，fp0）。
#   关键区分：
#     - bias（系统性偏移）不致命：常数倍偏差不改变排序，路由仍可用
#     - 相关系数（尤其秩相关）才决定排序质量，排序错了路由就是错的
#   所以同时给 Pearson r 和 Spearman rho，以及与「预测常数(全局均值)」的基线对比。
# 用法: python3 wt_predictor_quality.py <run_dir> [<run_dir> ...]
import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    return pearson(rx, ry)


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5
    return num / den if den else float("nan")


def main(d):
    d = d.rstrip("/")
    tag = os.path.basename(d)[-8:]
    files = sorted(glob.glob(d + "/log/predict_request_decode*.csv"))
    act, prd = [], []
    for f in files:
        with open(f, newline="", errors="ignore") as fh:
            for r in csv.DictReader(fh):
                try:
                    a = float(r["actual_output_len"])
                    p = float(r["predict_output_len"])
                except (ValueError, TypeError, KeyError):
                    continue
                act.append(a)
                prd.append(p)
    n = len(act)
    if not n:
        print(f"{tag}: 无数据")
        return
    mean_a = sum(act) / n
    mae = sum(abs(a - p) for a, p in zip(act, prd)) / n
    bias = sum(a - p for a, p in zip(act, prd)) / n
    mae_base = sum(abs(a - mean_a) for a in act) / n
    print(f"{tag}  n={n}")
    print(f"   actual   mean={mean_a:8.0f}  p50={sorted(act)[n//2]:7.0f}  "
          f"p90={sorted(act)[int(n*.9)]:7.0f}  p99={sorted(act)[int(n*.99)]:8.0f}")
    print(f"   predict  mean={sum(prd)/n:8.0f}  p50={sorted(prd)[n//2]:7.0f}  "
          f"p90={sorted(prd)[int(n*.9)]:7.0f}  p99={sorted(prd)[int(n*.99)]:8.0f}")
    print(f"   倍数 predict/actual (mean) = {sum(prd)/n/mean_a:.2f}x")
    print(f"   MAE(预测)={mae:8.0f}   MAE(只报全局均值)={mae_base:8.0f}   "
          f"→ 预测比基线{'好' if mae < mae_base else '差'} {abs(1-mae/mae_base)*100:.1f}%")
    print(f"   bias(actual-predict)={bias:+8.0f}")
    print(f"   Pearson r = {pearson(prd, act):.3f}    Spearman rho = {spearman(prd, act):.3f}")
    # 分桶看：预测高/中/低的组，实际均值差多少（单调性 = 排序能力）
    idx = sorted(range(n), key=lambda i: prd[i])
    for k in (5, 10):
        print(f"   按预测值分 {k} 档，各档 actual 均值：", end="")
        out = []
        for q in range(k):
            seg = idx[q * n // k:(q + 1) * n // k]
            out.append(f"{sum(act[i] for i in seg)/len(seg):.0f}")
        print(" ".join(out))


for a in sys.argv[1:]:
    main(a)
    print()
