#!/usr/bin/env python3
# [WT] 校验一轮运行是否被「心跳超时 → 实例重注册 → 配对错位」污染。
#
# 背景（2026-09-25 事故）：
#   proxy 用同一个计数器 count 索引 prefill / decode 两个 dict：
#       prefill_list[count % len]  ,  decode_list[count % len]
#   两个 dict 的顺序 = 实例 ZMQ 注册顺序。心跳间隔 3s、TTL 只有 5s，
#   一旦某实例心跳迟到 >2s 就被 🔴Remove 并在重注册时排到末尾 → 顺序变化
#   → 配对从「1:1 置换」变成「全互连」→ 单实例可能被无限投喂 → 并发塌陷。
#
# 判据：
#   干净 → 配对组合数 == 实例数（4p4d 为 4），各组合占比 ~25%，且 🔴Remove == 0
#   可疑 → 出现 🔴Remove，或组合数 > 实例数，或分布明显不均
#
# 用法：python3 wt_validate_proxy_routing.py <run_dir>
#   退出码 0 = CLEAN，1 = SUSPECT，2 = 无法判定

import glob
import os
import re
import sys
from collections import Counter

ADD_RE = re.compile(r"🔵Add \[HTTP:[\d.]+:(\d+)")
PAIR_RE = re.compile(
    r"handle_request count: (\d+), \[HTTP:[\d.]+:(\d+).*?\] 👉 \[HTTP:[\d.]+:(\d+)")


def main():
    if len(sys.argv) < 2:
        print("用法: wt_validate_proxy_routing.py <run_dir>")
        return 2
    d = sys.argv[1].rstrip("/")
    logs = glob.glob(os.path.join(d, "log", "proxy_*.log"))
    if not logs:
        print(f"[校验] ❌ 找不到 proxy 日志: {d}/log/proxy_*.log")
        return 2

    log = max(logs, key=os.path.getsize)
    n_remove, pairs, n_prefill, n_decode, add_ports = 0, [], set(), set(), []
    with open(log, errors="ignore") as f:
        for line in f:
            if "🔴Remove" in line:
                n_remove += 1
            ma = ADD_RE.search(line)
            if ma:
                add_ports.append(ma.group(1))
            m = PAIR_RE.search(line)
            if m:
                pairs.append((m.group(2), m.group(3)))
                n_prefill.add(m.group(2))
                n_decode.add(m.group(3))

    name = os.path.basename(d.rstrip("/"))
    uniq_add = set(add_ports)
    print(f"[校验] {name}")
    print(f"       日志 {os.path.basename(log)} ({os.path.getsize(log)/1e6:.1f} MB)")
    print(f"       🔵Add = {len(add_ports)}（去重端口 {len(uniq_add)}）   🔴Remove = {n_remove}")

    reasons = []

    # --- 通用判据 A：没有任何实例掉线后重注册 ---
    if n_remove > 0:
        reasons.append(f"出现 {n_remove} 次 🔴Remove（实例心跳超时掉线）")
    if len(add_ports) > len(uniq_add):
        reasons.append(f"🔵Add {len(add_ports)} 次 > 去重端口 {len(uniq_add)}"
                       " → 有实例掉线后重注册，字典顺序已变")
    if not add_ports:
        print("       ⚠️ 没有任何 🔵Add 记录（proxy 可能还没起来），无法判定")
        return 2

    # --- 判据 B：RR proxy 专属，prefill↔decode 配对拓扑 ---
    if pairs:
        combos = Counter(pairs)
        total = len(pairs)
        expect_inst = max(len(n_prefill), len(n_decode))
        print(f"       配对记录 {total} 条   prefill 端口 {len(n_prefill)} / decode 端口 {len(n_decode)}")
        print(f"       配对组合种类 = {len(combos)}   期望 = {expect_inst}")
        for (p, dd), n in combos.most_common(6):
            mark = "  ← 非基准组合" if n < total / expect_inst * 0.5 else ""
            print(f"         {p} -> {dd} : {n:>5} 次 ({n/total*100:5.1f}%){mark}")
        if len(combos) > expect_inst:
            reasons.append(f"配对组合 {len(combos)} > 实例数 {expect_inst}"
                           "（拓扑已从 1:1 置换变成全互连）")
        top = combos.most_common(expect_inst)
        if len(top) == expect_inst:
            lo = min(n for _, n in top) / total
            if lo < 0.15:
                reasons.append(f"主组合最低占比仅 {lo*100:.1f}%（负载不均）")
    else:
        print("       （无 handle_request 配对行 → 该 proxy 不打这种格式，跳过拓扑检查）")

    if reasons:
        print("\n       ❌ 判定: SUSPECT")
        for r in reasons:
            print(f"          - {r}")
        return 1
    print("\n       ✅ 判定: CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
