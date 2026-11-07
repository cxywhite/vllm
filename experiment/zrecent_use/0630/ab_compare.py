#!/usr/bin/env python3
"""Admission A/B comparison table for one result root (scheme4 vs scheme4_gwf).

Reads only what a finished run writes:
  <run>/log/bench_trace_*.log   -> goodput, TTFT/TPOT percentiles
  <run>/log/decode*.log         -> [WT] counters: running, kv_usage_pct,
                                   preempted_recompute_tokens, demoted, stop_window
  <run>/dataset_result/test_results_*.csv (optional) -> good-request ratio

Usage:
    python3 ab_compare.py --root admission_ab_mooncake_cap [--out ab_table.md]

The point of the per-instance view is that `running` and `kv_usage_pct` are
per decode instance: averaging them across the 4 instances hides 2-3x
differences in request-length mix (see hw_guided_workload_and_sim.md §2.4).

Read-only.
"""
from __future__ import annotations

import argparse
import csv as _csvmod
import glob
import os
import re
import statistics as st
import sys

_csvmod.field_size_limit(1 << 30)   # prompts are long; default 128 KiB limit trips
KV_CAP_TOKENS = 881_616


def _bench(path: str) -> dict:
    out: dict = {}
    for line in open(path, errors="ignore"):
        m = re.match(r"^(Request goodput \(req/s\)|Request throughput \(req/s\)|"
                     r"Output token goodput \(tok/s\)|Successful requests|"
                     r"Mean TTFT \(ms\)|Median TTFT \(ms\)|P99 TTFT \(ms\)|"
                     r"Mean TPOT \(ms\)|Median TPOT \(ms\)|P99 TPOT \(ms\)|"
                     r"Mean E2EL \(ms\)|P99 E2EL \(ms\))\s*:\s*([0-9.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def _wt(run_dir: str) -> dict:
    per: dict[str, dict] = {}
    for path in glob.glob(os.path.join(run_dir, "log", "decode*.log")):
        inst = os.path.basename(path).split("_")[0]
        rec = per.setdefault(inst, {"run": [], "kv": [], "rc": 0, "demoted": 0,
                                    "stop_window": 0, "preempted": 0, "win": []})
        for line in open(path, errors="ignore"):
            if "[WT][" not in line:
                continue
            kv = {}
            for tok in line[line.index("[WT]["):].split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            try:
                rec["run"].append(int(kv.get("running", 0)))
                if kv.get("kv_usage_pct") is not None:
                    rec["kv"].append(float(kv["kv_usage_pct"]))
                rec["rc"] = max(rec["rc"], int(kv.get("preempted_recompute_tokens", 0)))
                rec["demoted"] += int(kv.get("demoted", 0))
                rec["stop_window"] += int(kv.get("stop_window", 0))
                rec["preempted"] += int(kv.get("preempted", 0))
                if int(kv.get("window", 0)) > 0:
                    rec["win"].append(int(kv["window"]))
            except ValueError:
                continue
    return per


def _good_ratio(run_dir: str) -> tuple[int, int]:
    n = good = 0
    for path in glob.glob(os.path.join(run_dir, "dataset_result", "test_results_*.csv")):
        import csv as _csv
        for r in _csv.DictReader(open(path, errors="ignore")):
            try:
                ttft = float(r["ttft"]) * 1000.0     # CSV stores SECONDS
                tpot = float(r["tpot"]) * 1000.0
            except (KeyError, ValueError):
                continue
            n += 1
            if ttft <= 1000.0 and tpot <= 30.0:
                good += 1
    return good, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(d):
            continue
        if d.endswith("_scheme4_gwf"):
            variant = "gwf"
        elif d.endswith("_scheme4"):
            variant = "greedy"
        else:
            continue
        bench = glob.glob(os.path.join(d, "log", "bench_trace_*.log"))
        if not bench:
            continue
        runs.setdefault(variant, []).append({
            "dir": d, "bench": _bench(bench[0]), "wt": _wt(d), "good": _good_ratio(d)})

    if not runs:
        print(f"no finished runs under {args.root}")
        return 1

    lines: list[str] = []
    p = lines.append
    p(f"# Admission A/B — `{args.root}`")
    for variant, label in (("greedy", "scheme4 (greedy)"), ("gwf", "scheme4_gwf (GWF)")):
        for r in runs.get(variant, []):
            b = r["bench"]
            tot_rc = sum(v["rc"] for v in r["wt"].values())
            kv = [x for v in r["wt"].values() for x in v["kv"]]
            run = [x for v in r["wt"].values() for x in v["run"]]
            dem = sum(v["demoted"] for v in r["wt"].values())
            sw = sum(v["stop_window"] for v in r["wt"].values())
            win = [x for v in r["wt"].values() for x in v["win"]]
            good, n = r["good"]
            p(f"\n## {label}")
            p(f"- goodput **{b.get('Request goodput (req/s)', float('nan')):.2f} req/s** "
              f"(throughput {b.get('Request throughput (req/s)', float('nan')):.2f} req/s)")
            p(f"- TTFT p50/p99 {b.get('Median TTFT (ms)', float('nan')):.0f} / "
              f"{b.get('P99 TTFT (ms)', float('nan')):.0f} ms")
            p(f"- TPOT p50/p99 {b.get('Median TPOT (ms)', float('nan')):.1f} / "
              f"{b.get('P99 TPOT (ms)', float('nan')):.1f} ms")
            p(f"- good requests {good}/{n} = {100.0*good/max(1,n):.1f} %")
            p(f"- running max {max(run) if run else 0}, kv_pct max {max(kv) if kv else float('nan'):.1f}")
            p(f"- **preempted_recompute_tokens total {tot_rc:,}** "
              f"(per instance: " + ", ".join(f"{k}={v['rc']:,}" for k, v in sorted(r["wt"].items())) + ")")
            p(f"- demoted Σ {dem}, stop_window Σ {sw}, window min {min(win) if win else 'n/a'}")

    if "greedy" in runs and "gwf" in runs:
        g = runs["greedy"][0]["bench"]
        w = runs["gwf"][0]["bench"]
        gp, wp = g.get("Request goodput (req/s)", 0.0), w.get("Request goodput (req/s)", 0.0)
        rc_g = sum(v["rc"] for v in runs["greedy"][0]["wt"].values())
        rc_w = sum(v["rc"] for v in runs["gwf"][0]["wt"].values())
        p("\n## Δ (GWF vs greedy)")
        if gp:
            p(f"- goodput {wp:.2f} vs {gp:.2f} req/s → **{(wp-gp)/gp*100:+.1f} %**")
        p(f"- P99 TTFT {w.get('P99 TTFT (ms)', float('nan')) - g.get('P99 TTFT (ms)', float('nan')):+.0f} ms, "
          f"P99 TPOT {w.get('P99 TPOT (ms)', float('nan')) - g.get('P99 TPOT (ms)', float('nan')):+.1f} ms")
        p(f"- recompute tokens {rc_w:,} vs {rc_g:,} → "
          f"{(rc_w - rc_g) / rc_g * 100:+.1f} %" if rc_g else
          f"- recompute tokens {rc_w:,} vs {rc_g:,} (greedy had none)")

    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
