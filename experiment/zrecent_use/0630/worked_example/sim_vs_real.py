#!/usr/bin/env python3
"""Close the loop: does the offline simulator reproduce a REAL A/B run?

The sim is only useful for workload search if its prediction of the
*admission-controller delta* is right.  This script runs the sim twice (greedy,
GWF) on the same trace and puts its numbers next to the two real run dirs:

    python3 sim_vs_real.py \
        --real-greedy admission_ab_dense_short/..._scheme4 \
        --real-gwf    admission_ab_dense_short/..._scheme4_gwf \
        --trace       admission_ab_base/bimodal_burst/qwen_lmsys_bimodal_burst_600s.csv \
        --curve-json  service_models/dense_short_gwf.json --curve-scale 0.7 \
        --imbalance 1.0 --out sim_vs_real_dense_short.md

What it checks (and what each mismatch means):
  * goodput Δ% (GWF vs greedy)   -> the number the paper claims; sign must match
  * max_running                  -> must match greedy (no gate) to be believable
  * demoted / stop_window totals -> sim counters are cumulative, the real logs are
                                    1 s deltas, so the real column is the SUM of
                                    deltas; a 5x+ ratio means the sim's hopeless
                                    predicate fires far too often
  * TTFT/TPOT percentiles        -> the sim does not model them yet (marked n/a)

Read-only: writes only --out.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import statistics as st
import sys


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def _bench(log_dir: str) -> dict:
    out: dict = {}
    for path in glob.glob(os.path.join(log_dir, "bench_trace_*.log")):
        for line in open(path, errors="ignore"):
            m = re.match(r"^(Request goodput \(req/s\)|Output token goodput \(tok/s\)|"
                         r"P99 TTFT \(ms\)|P50 TTFT \(ms\)|Median TTFT \(ms\)|"
                         r"Median TPOT \(ms\)|P99 TPOT \(ms\)|Mean TPOT \(ms\)|"
                         r"Request throughput \(req/s\))\s*:\s*([0-9.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def _wt_sums(log_dir: str) -> dict:
    tot = {"demoted": 0, "stop_window": 0, "preempted": 0}
    peak = {"running": 0, "waiting": 0}
    for path in glob.glob(os.path.join(log_dir, "decode*.log")):
        for line in open(path, errors="ignore"):
            if "[WT][" not in line:
                continue
            kv = {}
            for tok in line[line.index("[WT]["):].split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            try:
                for k in tot:
                    tot[k] += int(kv.get(k, 0))
                peak["running"] = max(peak["running"], int(kv.get("running", 0)))
                peak["waiting"] = max(peak["waiting"], int(kv.get("waiting", 0)))
            except ValueError:
                continue
    return {**tot, **peak}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-greedy", required=True)
    ap.add_argument("--real-gwf", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--curve-json", default=os.path.join(here, "..", "service_models", "dense_short_gwf.json"))
    ap.add_argument("--curve-scale", type=float, default=1.0)
    ap.add_argument("--imbalance", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sim = _load(os.path.join(here, "sim_admission.py"), "sim_admission")
    if args.curve_json and os.path.exists(args.curve_json):
        model = json.load(open(args.curve_json))
        curve = [(int(u), float(v) * args.curve_scale) for u, v in model["tpot_curve"]]
        if curve:
            sim.TPOT_CURVES["measured"] = curve
            curve_name = "measured"
        else:
            curve_name = "real"
    else:
        model, curve_name = {}, "real"

    reqs = sim.load_trace(args.trace)
    sim_out = {}
    for policy in ("greedy", "gwf"):
        sim_out[policy] = sim.simulate(reqs, policy, curve_name, args.imbalance)

    real = {}
    for tag, d in (("greedy", args.real_greedy), ("gwf", args.real_gwf)):
        real[tag] = {"bench": _bench(os.path.join(d, "log")),
                     "wt": _wt_sums(os.path.join(d, "log"))}

    rg = real["greedy"]["bench"].get("Request goodput (req/s)", 0.0)
    rw = real["gwf"]["bench"].get("Request goodput (req/s)", 0.0)
    sg = sim_out["greedy"]["goodput_req_s"]
    sw = sim_out["gwf"]["goodput_req_s"]
    d_real = (rw - rg) / rg * 100.0 if rg else float("nan")
    d_sim = (sw - sg) / sg * 100.0 if sg else float("nan")

    # goodput totals (sim is per decode instance -> x4 to compare with the cluster)
    n_inst = 4
    rows = [
        ("goodput (req/s, cluster)", f"{rg:.2f}", f"{rw:.2f}",
         f"{sg * n_inst:.2f}", f"{sw * n_inst:.2f}"),
        ("Δ goodput GWF vs greedy", "—", f"{d_real:+.1f}%", "—", f"{d_sim:+.1f}%"),
        ("max_running (per instance)",
         str(real["greedy"]["wt"]["running"]), str(real["gwf"]["wt"]["running"]),
         str(sim_out["greedy"]["max_running"]), str(sim_out["gwf"]["max_running"])),
        ("max_waiting", str(real["greedy"]["wt"]["waiting"]), str(real["gwf"]["wt"]["waiting"]),
         str(sim_out["greedy"]["max_waiting"]), str(sim_out["gwf"]["max_waiting"])),
        ("demoted (total)", str(real["greedy"]["wt"]["demoted"]), str(real["gwf"]["wt"]["demoted"]),
         str(sim_out["greedy"]["counters"]["demoted"]), str(sim_out["gwf"]["counters"]["demoted"])),
        ("stop_window (total)", str(real["greedy"]["wt"]["stop_window"]), str(real["gwf"]["wt"]["stop_window"]),
         str(sim_out["greedy"]["counters"]["stop_window"]), str(sim_out["gwf"]["counters"]["stop_window"])),
        ("preempted (total)", str(real["greedy"]["wt"]["preempted"]), str(real["gwf"]["wt"]["preempted"]),
         str(sim_out["greedy"]["counters"]["preempted"]), str(sim_out["gwf"]["counters"]["preempted"])),
        ("P50 TTFT (ms)", f"{real['greedy']['bench'].get('Median TTFT (ms)', float('nan')):.0f}",
         f"{real['gwf']['bench'].get('Median TTFT (ms)', float('nan')):.0f}", "n/a", "n/a"),
        ("P99 TTFT (ms)", f"{real['greedy']['bench'].get('P99 TTFT (ms)', float('nan')):.0f}",
         f"{real['gwf']['bench'].get('P99 TTFT (ms)', float('nan')):.0f}", "n/a", "n/a"),
        ("median TPOT (ms)", f"{real['greedy']['bench'].get('Median TPOT (ms)', float('nan')):.1f}",
         f"{real['gwf']['bench'].get('Median TPOT (ms)', float('nan')):.1f}", "n/a", "n/a"),
    ]

    lines = []
    p = lines.append
    p(f"# Simulator vs real — `{os.path.basename(args.trace)}`")
    p(f"\ncurve: `{curve_name}` (scale {args.curve_scale}), imbalance {args.imbalance}, "
      f"kv cap {model.get('kv_cap_tokens', 0):,} tok")
    p("\n| metric | real greedy | real GWF | sim greedy | sim GWF |")
    p("|---|---|---|---|---|")
    for r in rows:
        p("| " + " | ".join(str(x) for x in r) + " |")

    p("\n## Verdict")
    verdict = []
    if d_real == d_real and d_sim == d_sim:
        if d_real * d_sim < 0:
            verdict.append(f"**SIGN MISMATCH**: real {d_real:+.1f}% vs sim {d_sim:+.1f}% — "
                           "the sim must not be used to rank workloads until this is fixed.")
        else:
            verdict.append(f"sign agrees (real {d_real:+.1f}%, sim {d_sim:+.1f}%); "
                           f"magnitude error {abs(d_sim - d_real):.1f} pp.")
    if real["gwf"]["wt"]["demoted"] and sim_out["gwf"]["counters"]["demoted"]:
        ratio = sim_out["gwf"]["counters"]["demoted"] / real["gwf"]["wt"]["demoted"]
        verdict.append(f"demoted count ratio sim/real = {ratio:.1f}x "
                       f"({sim_out['gwf']['counters']['demoted']} vs {real['gwf']['wt']['demoted']}); "
                       "target 0.7-1.5x — the `is_hopeless()` predicate is too eager when >2x.")
    if real["gwf"]["wt"]["stop_window"] and sim_out["gwf"]["counters"]["stop_window"]:
        ratio = sim_out["gwf"]["counters"]["stop_window"] / real["gwf"]["wt"]["stop_window"]
        verdict.append(f"stop_window count ratio sim/real = {ratio:.1f}x "
                       f"({sim_out['gwf']['counters']['stop_window']} vs {real['gwf']['wt']['stop_window']}).")
    verdict.append("the sim does not emit TTFT/TPOT percentiles yet — add them before using it "
                   "to rank workloads on SLO attainment.")
    for v in verdict:
        p(f"- {v}")

    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
