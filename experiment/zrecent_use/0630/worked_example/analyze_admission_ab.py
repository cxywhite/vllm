#!/usr/bin/env python3
"""
Parse the admission-isolation A/B results (scheme4 vs scheme4_gwf) and compute
the admission-stage (second-stage) gain.

The A/B holds the scheme4 predictor-aware router FIXED and toggles only the
intra-instance admission controller:
  * scheme4      -> stock vLLM greedy admission (no GWF)
  * scheme4_gwf  -> GeometricWindowScheduler (GWF admission: theta window +
                    hopeless demote + eV eviction)

Outputs, per method, the median (over reps) of: output token goodput, 30ms
TPOT attainment, P99 TPOT, successful requests, and (for scheme4_gwf) the
scheduler's own preempt/demote/evict counters.
"""
from __future__ import annotations
import csv, re, sys, glob, statistics
from pathlib import Path

RESULT_ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_result3"
)

BENCH_PATS = {
    "succ_reqs": r"Successful requests:\s+(\d+)",
    "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
    "total_input_tok": r"Total input tokens:\s+(\d+)",
    "total_gen_tok": r"Total generated tokens:\s+(\d+)",
    "output_goodput": r"Output token goodput \(tok/s\):\s+([\d.]+)",
    "request_goodput": r"Request goodput \(req/s\):\s+([\d.]+)",
    "total_goodput": r"Total token goodput \(tok/s\):\s+([\d.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
}

GWF_RE = re.compile(
    r"\[WT\]\[GWF\] steps=(\d+) admitted=(\d+) preempted=(\d+) "
    r"evicted_hopeless=(\d+) demoted=(\d+) hopeless_forced=(\d+) "
    r"stop_window=(\d+) meas_tpot_ms=([-\d.]+) budget_ms=([\d.]+) "
    r"window=(\d+) running=(\d+) waiting=(\d+)")


def parse_benchmark_log(path: Path) -> dict:
    m = {}
    if not path.exists():
        return m
    text = path.read_text(errors="replace")
    for k, p in BENCH_PATS.items():
        mm = re.search(p, text)
        if mm:
            v = mm.group(1)
            m[k] = float(v) if "." in v else int(v)
    return m


def parse_gwf_counters(decode_logs: list[Path]) -> dict:
    """Sum the per-second GWF counters across a run's decode logs."""
    tot = {"steps": 0, "admitted": 0, "preempted": 0, "evicted_hopeless": 0,
           "demoted": 0, "hopeless_forced": 0, "stop_window": 0}
    meas_tpots: list[float] = []
    for lp in decode_logs:
        if not lp.exists():
            continue
        for line in lp.read_text(errors="replace").splitlines():
            mm = GWF_RE.search(line)
            if not mm:
                continue
            tot["steps"] += int(mm.group(1))
            tot["admitted"] += int(mm.group(2))
            tot["preempted"] += int(mm.group(3))
            tot["evicted_hopeless"] += int(mm.group(4))
            tot["demoted"] += int(mm.group(5))
            tot["hopeless_forced"] += int(mm.group(6))
            tot["stop_window"] += int(mm.group(7))
            mt = float(mm.group(8))
            if mt >= 0:
                meas_tpots.append(mt)
    tot["meas_tpot_p50_ms"] = statistics.median(meas_tpots) if meas_tpots else None
    tot["meas_tpot_p90_ms"] = (sorted(meas_tpots)[int(0.90 * len(meas_tpots))]
                               if meas_tpots else None)
    return tot


def main() -> int:
    runs: list[dict] = []
    for d in sorted(RESULT_ROOT.glob("*/")):
        name = d.name
        m = re.search(r"_rep(\d+)_", name)
        if not m:
            continue
        rep = int(m.group(1))
        if name.endswith("_scheme4_gwf"):
            method = "scheme4_gwf"
        elif name.endswith("_scheme4"):
            method = "scheme4"
        else:
            continue
        bench_logs = list(d.glob("log/bench_trace_*.log")) + list(d.glob("benchmark*.log"))
        bm = {}
        for bl in bench_logs:
            bm.update(parse_benchmark_log(bl))
        # also try dataset_result dir for benchmark.log
        for bl in d.glob("dataset_result/*.log"):
            bm.update(parse_benchmark_log(bl))
        decode_logs = list(d.glob("log/decode*.log"))
        gwf = parse_gwf_counters(decode_logs)
        runs.append({"method": method, "rep": rep, "bench": bm, "gwf": gwf, "dir": name})

    print(f"found {len(runs)} completed runs\n")

    for method in ("scheme4", "scheme4_gwf"):
        rs = [r for r in runs if r["method"] == method]
        if not rs:
            print(f"== {method}: no completed runs yet ==")
            continue
        print(f"== {method}: {len(rs)} rep(s) ==")
        for r in sorted(rs, key=lambda x: x["rep"]):
            b = r["bench"]
            print(f"  rep{r['rep']}: goodput={b.get('output_goodput','-')} tok/s  "
                  f"p99_tpot={b.get('p99_tpot_ms','-')}ms  succ={b.get('succ_reqs','-')}")
        # median over reps
        def med(key):
            vals = [r["bench"][key] for r in rs if key in r["bench"]]
            return statistics.median(vals) if vals else None
        print(f"  MEDIAN: goodput={med('output_goodput')}  p99_tpot={med('p99_tpot_ms')}  "
              f"succ={med('succ_reqs')}  total_gen_tok={med('total_gen_tok')}")
        # GWF counters (only scheme4_gwf has them)
        if method == "scheme4_gwf":
            for r in sorted(rs, key=lambda x: x["rep"]):
                g = r["gwf"]
                print(f"  rep{r['rep']} GWF: preempted={g['preempted']} demoted={g['demoted']} "
                      f"evicted_hopeless={g['evicted_hopeless']} stop_window={g['stop_window']} "
                      f"meas_tpot_p50={g.get('meas_tpot_p50_ms')}")
        print()

    # gain = scheme4_gwf vs scheme4 (median)
    s4 = [r for r in runs if r["method"] == "scheme4"]
    g4 = [r for r in runs if r["method"] == "scheme4_gwf"]
    if s4 and g4:
        def med(key, rs):
            vals = [r["bench"][key] for r in rs if key in r["bench"]]
            return statistics.median(vals) if vals else None
        gp4 = med("output_goodput", s4)
        gg4 = med("output_goodput", g4)
        if gp4 and gg4:
            delta = (gg4 - gp4) / gp4 * 100
            print(f"== ADMISSION GAIN (scheme4_gwf vs scheme4, median goodput) ==")
            print(f"  scheme4      = {gp4:.1f} tok/s")
            print(f"  scheme4_gwf  = {gg4:.1f} tok/s")
            print(f"  Δ = {delta:+.1f}%")
            tp4 = med("p99_tpot_ms", s4)
            tg4 = med("p99_tpot_ms", g4)
            if tp4 and tg4:
                print(f"  P99 TPOT: scheme4={tp4:.1f}ms  scheme4_gwf={tg4:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
