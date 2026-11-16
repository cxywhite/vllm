#!/usr/bin/env python3
# [WT] 只盘点 2026-09-22 起新跑的结果，按存放根目录分组。
import glob
import os
import re
import time
from collections import defaultdict

SINCE = time.mktime(time.strptime("2026-09-22", "%Y-%m-%d"))

ROOTS = [
    "/root/myshare/wty_tmp_result",
    "/root/myshare/tmp_dynamo_result",
    "/root/predict-schedule/experiment/zrecent_use/0630",
    "/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite",
    "/root/predict-schedule/experiment/wt_experiment",
    "/root/predict-schedule/experiment",
    "/root/myshare/models/wt_predictor/0630",
    "/root/myshare/paper_figures",
    "/root/predict-schedule/baseline_experiment",
    "/root/predict-schedule/predict_experiment",
    "/root/predict-schedule/batchsize_experiment",
]

RUN_NAME = re.compile(r"pd\d+p\d+d|_dynamo$")


def status_of(d):
    """返回 (status, goodput, 大小) ；status ∈ done/partial/empty"""
    logs = glob.glob(os.path.join(d, "log", "bench_*.log"))
    logs += glob.glob(os.path.join(d, "benchmark.log"))
    if not logs:
        if os.path.isdir(os.path.join(d, "log")) or os.path.isdir(
                os.path.join(d, "dataset_result")):
            return "empty", None
        return None, None
    gp = None
    for f in logs:
        try:
            with open(f, errors="ignore") as fh:
                for line in fh:
                    if line.startswith("Output token goodput"):
                        gp = line.split(":")[-1].strip()
                        break
        except OSError:
            pass
        if gp:
            break
    return ("done" if gp else "partial"), gp


rows = []
seen_dirs = set()
for root in ROOTS:
    if not os.path.isdir(root):
        continue
    for d in glob.glob(os.path.join(root, "**"), recursive=True):
        if not os.path.isdir(d) or d in seen_dirs:
            continue
        base = os.path.basename(d)
        if not RUN_NAME.search(base):
            continue
        # 只看最近几天动过的
        try:
            mt = max(os.path.getmtime(d),
                     max((os.path.getmtime(f) for f in glob.glob(os.path.join(d, "*"))
                          if os.path.isfile(f)), default=0))
        except OSError:
            continue
        if mt < SINCE:
            continue
        st, gp = status_of(d)
        if st is None:
            continue
        seen_dirs.add(d)
        # run 名里的时间戳
        m = re.search(r"(20\d{6}_\d{6})", base)
        ts = m.group(1) if m else "?"
        rows.append((root, ts, st, gp, d))

# 按根分组
agg = defaultdict(list)
for root, ts, st, gp, d in rows:
    agg[root].append((ts, st, gp, d))

print("=" * 100)
print("[WT] 2026-09-22 起新跑的结果，按存放根目录分组")
print("=" * 100)
for root in ROOTS:
    if root not in agg:
        continue
    items = sorted(agg[root])
    nd = sum(1 for _, s, _, _ in items if s == "done")
    np_ = sum(1 for _, s, _, _ in items if s == "partial")
    ne = sum(1 for _, s, _, _ in items if s == "empty")
    tag = "【Dynamo】" if "dynamo" in root else "【我们】"
    print(f"\n{tag} {root}")
    print(f"      完成 {nd} / 部分 {np_} / 空壳 {ne}   （共 {len(items)}）")
    for ts, st, gp, d in items:
        name = os.path.basename(d)
        short = re.sub(r"^(qwen|llama)_", "", name)
        short = short.replace("qwen-lmsys-chat-", "L|").replace("mysharegpt-chat-", "S|")
        short = short.replace("qwen-mysharegpt-", "S|").replace("lmsys-chat-", "L|")
        gp_s = f"goodput={gp}" if gp else ""
        print(f"        [{st:<7}] {ts}  {gp_s:<22} {short[:88]}")

nd_all = sum(1 for _, _, s, _, _ in rows if s == "done")
print(f"\n{'='*100}")
print(f"[WT] 合计最近几天有 bench 日志的 run: {len(rows)} 个，其中完成 {nd_all} 个")
print("[WT] 注: 未列出 = 空壳/无 bench 日志")
