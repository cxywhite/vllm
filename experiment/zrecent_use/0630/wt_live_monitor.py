#!/usr/bin/env python3
"""实时监控每个请求的 TTFT / TPOT（不阻塞 benchmark 主进程）。

用法：
    WT_LIVE_CSV=/path/live.csv 启动 benchmark（benchmark 侧会异步写这个文件）
    python3 wt_live_monitor.py /path/live.csv --interval 2 --show-each

输出（每 interval 秒）：
    [LIVE] n=1234 +12 | TTFT p50/p90/p99 = .. | TPOT p50/p90/p99 = .. | good=../..
    （加 --show-each 还会逐条打印新完成的请求）

设计：benchmark 侧用后台线程 + 队列写 CSV（不阻塞事件循环）；本脚本只读，
每次重新解析整个小文件（几千行），因此不会和写入方抢锁。
"""
import argparse
import csv
import os
import time


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, int(p * (len(s) - 1)))
    return s[k]


def read_rows(path):
    rows = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                # 跳过被截断的半行（写入线程可能正在 flush）
                if not row.get("ttft_ms") or not row.get("tpot_ms"):
                    continue
                rows.append(row)
    except (FileNotFoundError, PermissionError):
        pass
    except Exception:
        pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=30.0)
    ap.add_argument("--show-each", action="store_true")
    args = ap.parse_args()

    last = 0
    print(f"[LIVE] watching {args.csv_path} every {args.interval}s "
          f"(SLO ttft<={args.ttft_slo_ms:.0f}ms tpot<={args.tpot_slo_ms:.0f}ms)")
    while True:
        rows = read_rows(args.csv_path)
        new = rows[last:]
        last = len(rows)
        if args.show_each and new:
            for r in new:
                print(f"[LIVE] {r.get('req_id','?')}  "
                      f"ttft={r.get('ttft_ms')}ms  tpot={r.get('tpot_ms')}ms  "
                      f"lat={r.get('latency_ms')}ms  ok={r.get('success')}")
        if rows:
            ttfts, tpots = [], []
            good = 0
            for r in rows:
                if r.get("success") != "1":
                    continue
                try:
                    t = float(r["ttft_ms"])
                    p = float(r["tpot_ms"])
                except (TypeError, ValueError):
                    continue
                ttfts.append(t)
                tpots.append(p)
                if t <= args.ttft_slo_ms and p <= args.tpot_slo_ms:
                    good += 1
            n = len(rows)
            print(f"[LIVE] n={n} +{len(new)} | "
                  f"TTFT p50/p90/p99 = {pct(ttfts,.5):.0f}/{pct(ttfts,.9):.0f}/{pct(ttfts,.99):.0f} ms | "
                  f"TPOT p50/p90/p99 = {pct(tpots,.5):.1f}/{pct(tpots,.9):.1f}/{pct(tpots,.99):.1f} ms | "
                  f"good={good}/{n} ({good/n*100:.1f}%)")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
