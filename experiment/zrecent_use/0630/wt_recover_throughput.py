#!/usr/bin/env python3
# [WT] 从 decode 实例日志恢复「吞吐/并发时间序列」——用于事故复盘。
#   vLLM 每 10s 打一行：Avg prompt throughput / Avg generation throughput /
#                        Running: N reqs, Waiting: M reqs / GPU KV cache usage: X%
#   对这些行积分可以近似还原总生成 token 数与并发演化。
import glob
import os
import re
import sys

LINE = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?"
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%")


def series(run_dir):
    out = {}          # inst -> list[(t, tok_s, running)]
    for f in sorted(glob.glob(os.path.join(run_dir, "log", "decode*.log"))):
        inst = os.path.basename(f).split("_")[0]
        rows = []
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = LINE.search(line)
                if m:
                    rows.append((m.group(1), float(m.group(2)), int(m.group(3)),
                                 float(m.group(5))))
        if rows:
            out[inst] = rows
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/myshare/wty_tmp_result/qwentrace_goodwin"
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        name = os.path.basename(d.rstrip("/"))
        if not name:
            continue
        s = series(d)
        if not s:
            continue
        print("=" * 96)
        print(f"[WT] {name.split('_rep1_')[-1]}")
        print("=" * 96)
        print(f"{'实例':<9}{'日志段':<10}{'峰值tok/s':>11}{'末段tok/s':>11}"
              f"{'Running峰值':>12}{'Running末值':>12}{'积分token数':>13}")
        grand = 0.0
        for inst in sorted(s):
            rows = s[inst]
            peak = max(r[1] for r in rows)
            tail = rows[-1][1]
            rmax = max(r[2] for r in rows)
            rlast = rows[-1][2]
            # 每两行间隔约 10s，积分近似总生成 token
            tok = sum(r[1] * 10.0 for r in rows)
            grand += tok
            print(f"{inst:<9}{rows[0][0][-8:]:<10}{peak:>11.1f}{tail:>11.1f}"
                  f"{rmax:>12}{rlast:>12}{tok:>13,.0f}")
        print(f"{'合计':<9}{'':<10}{'':>11}{'':>11}{'':>12}{'':>12}{grand:>13,.0f}")
        # 全局并发峰值
        allrows = [r for rows in s.values() for r in rows]
        print(f"\n  全局并发峰值 = {max(r[2] for r in allrows)}   "
              f"全局吞吐峰值 = {sum(max(r[1] for r in rows) for rows in s.values()):.0f} tok/s")
        # 吞吐塌陷时刻
        print("\n  吞吐塌陷过程（每实例最后 8 个采样点）:")
        for inst in sorted(s):
            rows = s[inst][-8:]
            print(f"    {inst:<9}" + " ".join(f"{r[1]:6.0f}" for r in rows)
                  + "   tok/s")
        print()


if __name__ == "__main__":
    main()
