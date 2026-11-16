#!/usr/bin/env python3
# [WT] 全盘盘点实验结果的存放位置：哪些在 wty_tmp_result，哪些散落别处。
import glob
import os
import re

CANDIDATE_ROOTS = [
    "/root/myshare/wty_tmp_result",
    "/root/myshare/tmp_dynamo_result",
    "/root/myshare/models/wt_predictor/0630",
    "/root/myshare/paper_figures",
    "/root/predict-schedule/experiment/zrecent_use/0630",
    "/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite",
    "/root/predict-schedule/experiment/wt_experiment",
    "/root/predict-schedule/experiment",
    "/root/predict-schedule/baseline_experiment",
    "/root/predict-schedule/batchsize_experiment",
    "/root/predict-schedule/predict_experiment",
    "/root/predict-schedule/nixl_vs_nccl",
    "/root/.cache/huggingface/hub/wt_predictor",
]

RUN_RE = re.compile(r"(pd\d+p\d+d|_dynamo$|dynamo_)")


def is_run_dir(d):
    """判定是不是一个 run 目录：名字里有 pdXpYd 或 dynamo，且带 log/bench 或 benchmark.log"""
    n = os.path.basename(d)
    if not RUN_RE.search(n):
        return None
    logs = glob.glob(os.path.join(d, "log", "bench_*.log"))
    logs += glob.glob(os.path.join(d, "benchmark.log"))
    if not logs:
        # 可能是空壳 run 目录，仍算一个条目
        if os.path.isdir(os.path.join(d, "log")) or os.path.isdir(
                os.path.join(d, "dataset_result")):
            return ("empty", None)
        return None
    txt_size = sum(os.path.getsize(f) for f in logs)
    done = False
    for f in logs:
        try:
            with open(f, errors="ignore") as fh:
                if "Output token goodput" in fh.read():
                    done = True
                    break
        except OSError:
            pass
    return ("done" if done else "partial", txt_size)


rows = []
for root in CANDIDATE_ROOTS:
    if not os.path.isdir(root):
        continue
    depth = root.count("/")
    for d in glob.glob(os.path.join(root, "**"), recursive=True):
        if not os.path.isdir(d):
            continue
        if d.count("/") - depth > 4:      # 限制深度
            continue
        r = is_run_dir(d)
        if r is None:
            continue
        if "dynamo" in os.path.basename(d).lower() or "dynamo" in d.lower():
            kind = "dynamo"
        else:
            kind = "ours/rr"
        rows.append((root, kind, r[0], d))

# 按根汇总
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(int))
for root, kind, status, d in rows:
    agg[root][(kind, status)] += 1

print("=" * 108)
print("[WT] 结果存放位置盘点")
print("=" * 108)
tot = defaultdict(int)
for root in CANDIDATE_ROOTS:
    if root not in agg:
        continue
    c = agg[root]
    o_done = c[("ours/rr", "done")]
    o_part = c[("ours/rr", "partial")] + c[("ours/rr", "empty")]
    d_done = c[("dynamo", "done")]
    d_part = c[("dynamo", "partial")] + c[("dynamo", "empty")]
    n_all = o_done + o_part + d_done + d_part
    tot["ours_done"] += o_done
    tot["ours_part"] += o_part
    tot["dynamo_done"] += d_done
    tot["dynamo_part"] += d_part
    print(f"{root}")
    print(f"    我们(RR/Ours): 完成 {o_done:>4}  未完成/空壳 {o_part:>4}"
          f"   |  Dynamo: 完成 {d_done:>4}  未完成/空壳 {d_part:>4}"
          f"   |  合计 {n_all}")
print("-" * 108)
print(f"总计: 我们 完成 {tot['ours_done']} / 未完成 {tot['ours_part']}"
      f"   |  Dynamo 完成 {tot['dynamo_done']} / 未完成 {tot['dynamo_part']}")

print("\n" + "=" * 108)
print("[WT] wty_tmp_result 内部细分")
print("=" * 108)
sub = defaultdict(lambda: defaultdict(int))
for root, kind, status, d in rows:
    if not root.endswith("wty_tmp_result"):
        continue
    rel = os.path.relpath(d, root).split("/")[0]
    sub[rel][(kind, status)] += 1
for k in sorted(sub):
    c = sub[k]
    od = c[("ours/rr", "done")]
    op = c[("ours/rr", "partial")] + c[("ours/rr", "empty")]
    dd = c[("dynamo", "done")]
    dp = c[("dynamo", "partial")] + c[("dynamo", "empty")]
    print(f"  {k:<26} 我们完成 {od:>4} 未完成 {op:>4}   Dynamo 完成 {dd:>3} 未完成 {dp:>3}")

print("\n" + "=" * 108)
print("[WT] 不在 wty_tmp_result 的『我们』的结果目录（按根聚合并给出示例）")
print("=" * 108)
by_root = defaultdict(list)
for root, kind, status, d in rows:
    if root.endswith("wty_tmp_result") or kind == "dynamo":
        continue
    by_root[root].append((status, d))
for root in sorted(by_root):
    items = by_root[root]
    nd = sum(1 for s, _ in items if s == "done")
    print(f"\n  {root}    完成 {nd} / 共 {len(items)}")
    for s, d in sorted(items)[:6]:
        print(f"      [{s:<7}] {d}")
    if len(items) > 6:
        print(f"      ... 其余 {len(items)-6} 个")
