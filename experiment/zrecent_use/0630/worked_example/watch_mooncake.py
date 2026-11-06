#!/usr/bin/env python3
"""Watch the mooncake A/B run and print a compact greedy-vs-GWF report when done.

Read-only: only reads result dirs / logs; never touches running processes or files.
Exits when both variants have per-request results, or when the runner process is gone,
or after a hard timeout.
"""
import csv, glob, os, subprocess, time

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--base", default="/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_mooncake")
_ap.add_argument("--pid", type=int, default=0)
_ap.add_argument("--label", default="MOONCAKE")
_args = _ap.parse_args()
BASE = _args.base
RUNNER_PID = _args.pid
csv.field_size_limit(1 << 30)


def results_for(suffix_exact):
    for d in glob.glob(BASE + "/qwen_*"):
        if not os.path.isdir(d):
            continue
        base = d.rstrip("/")
        if not base.endswith(suffix_exact):
            continue
        fs = glob.glob(d + "/dataset_result/test_results_*.csv")
        if fs:
            return d, fs[0]
    return None, None


def runner_alive():
    if not RUNNER_PID:
        return False
    try:
        out = subprocess.run(["ps", "-p", str(RUNNER_PID)],
                             capture_output=True, text=True, timeout=5)
        return "test_evaluation" in out.stdout or str(RUNNER_PID) in out.stdout
    except Exception:
        return False


def summ(D):
    rows = list(csv.DictReader(open(D, encoding="utf-8", errors="ignore")))
    tp = lambda r: float(r["tpot"]) * 1000.0
    tf = lambda r: float(r["ttft"]) * 1000.0
    n = len(rows)

    def q(a, p):
        a = sorted(a)
        return a[min(n - 1, int(n * p))]

    tps = [tp(r) for r in rows]
    tfs = [tf(r) for r in rows]
    good = [r for r in rows if tp(r) <= 30 and tf(r) <= 1000]
    gt = sum(int(float(r["output_tokens"])) for r in good)
    try:
        ts0 = min(float(r["send_timestamp_ms"]) for r in rows)
        ts1 = max(float(r["send_timestamp_ms"]) + float(r["latency"]) * 1000 for r in rows)
        span = (ts1 - ts0) / 1000.0
    except Exception:
        span = float("nan")
    return dict(n=n, good=len(good), gp_pct=100.0 * len(good) / max(1, n),
                goodput=gt / span if span else 0.0,
                t50=q(tps, .5), t90=q(tps, .9), t99=q(tps, .99), tmax=max(tps),
                f50=q(tfs, .5), f90=q(tfs, .9), f99=q(tfs, .99), fmax=max(tfs))


def gwf_counters(D):
    """Sum the last cumulative values per instance from the [WT][GWF] log lines."""
    last = {}
    win_min = None
    for f in glob.glob(D + "/log/decode*.log"):
        inst = os.path.basename(f)
        for line in open(f, encoding="utf-8", errors="ignore"):
            if "[WT][GWF]" not in line:
                continue
            kv = dict(__import__("re").findall(r"(\w+)=([-\d.]+)", line))
            try:
                win = int(float(kv.get("window", 256)))
            except Exception:
                continue
            win_min = win if win_min is None else min(win_min, win)
            last[inst] = kv
    agg = {}
    for kv in last.values():
        for k in ("admitted", "preempted", "recompute_tokens", "demoted",
                  "stop_window", "evicted_hopeless", "hopeless_forced"):
            agg[k] = agg.get(k, 0.0) + float(kv.get(k, 0.0))
    return agg, win_min


def main():
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        # exact matches (note: "_scheme4" also endswith "_scheme4_gwf", so check exact)
        gdir, gfile = None, None
        for d in glob.glob(BASE + "/qwen_*"):
            b = d.rstrip("/")
            if b.endswith("_scheme4") and os.path.isdir(d):
                fs = glob.glob(d + "/dataset_result/test_results_*.csv")
                if fs:
                    gdir, gfile = d, fs[0]
        wdir, wfile = None, None
        for d in glob.glob(BASE + "/qwen_*"):
            b = d.rstrip("/")
            if b.endswith("_scheme4_gwf") and os.path.isdir(d):
                fs = glob.glob(d + "/dataset_result/test_results_*.csv")
                if fs:
                    wdir, wfile = d, fs[0]
        if gfile and wfile:
            break
        if not runner_alive() and not (gfile and wfile):
            # runner gone: give it one more grace round
            time.sleep(30)
            if not (gfile and wfile):
                print("RUNNER_GONE without both results; producing partial report")
                break
        time.sleep(30)

    if not gfile:
        print("NO greedy result found")
        return
    s = summ(gfile)
    print(f"=== {_args.label} A/B ===")
    print(f"greedy(scheme4): n={s['n']} good={s['good']}({s['gp_pct']:.1f}%) "
          f"goodput={s['goodput']:.1f} tok/s | tpot p50/p90/p99/max="
          f"{s['t50']:.1f}/{s['t90']:.1f}/{s['t99']:.1f}/{s['tmax']:.0f}ms | "
          f"ttft p50/p90/p99/max={s['f50']:.0f}/{s['f90']:.0f}/{s['f99']:.0f}/{s['fmax']:.0f}ms")
    if wfile:
        w = summ(wfile)
        print(f"GWF(scheme4_gwf): n={w['n']} good={w['good']}({w['gp_pct']:.1f}%) "
              f"goodput={w['goodput']:.1f} tok/s | tpot p50/p90/p99/max="
              f"{w['t50']:.1f}/{w['t90']:.1f}/{w['t99']:.1f}/{w['tmax']:.0f}ms | "
              f"ttft p50/p90/p99/max={w['f50']:.0f}/{w['f90']:.0f}/{w['f99']:.0f}/{w['fmax']:.0f}ms")
        print(f"GOODPUT Δ = {(w['goodput']-s['goodput'])/s['goodput']*100:+.2f}%")
    else:
        print("GWF result not available yet")
    if wdir:
        agg, win_min = gwf_counters(wdir)
        print(f"GWF counters: {agg}  window_min={win_min}")


if __name__ == "__main__":
    main()
