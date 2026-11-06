#!/usr/bin/env python3
"""Extract a service model from ONE real A/B run, for the offline simulator.

The offline simulator (`worked_example/sim_admission.py`) needs four things that are
only knowable from a real run:

  1. `tpot(running)`     -- decode cost as a function of per-instance concurrency.
                            Read from the `[WT][NATIVE]`/`[WT][GWF]` 1 s lines
                            (`running=`, `meas_tpot_ms=`), bucketed by running, p50.
  2. KV capacity         -- `GPU KV cache size: N tokens` in the decode engine log.
  3. real goodput        -- `Request goodput (req/s)` in the bench log (greedy and GWF),
                            plus p50/p99 TTFT/TPOT from the per-request CSV.
  4. routing imbalance   -- peak running ratio across the 4 decode instances; the sim's
                            `--imbalance` knob. Measured, not guessed.

Unit trap (cost me one wrong conclusion): the per-request CSV stores `ttft`/`tpot` in
**seconds** (0.0222 = 22.2 ms). Everything emitted here is in ms / tokens explicitly.

Usage:
    python3 extract_service_model.py --run-dir <result dir> --out model.json
    # then feed it to the simulator:
    python3 worked_example/sim_admission.py --trace <trace.csv> \
        --curve-json model.json --imbalance <measured>

Read-only: writes only --out.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

MiB = 1024.0 * 1024.0
QVWEN_KV_BYTES_PER_TOKEN = 2 * 28 * 4 * 128 * 2   # 57,344 B/token
BUCKET = 32                                        # running bucket width for the curve


def _wt_records(log_dir: str) -> list[dict]:
    recs: list[dict] = []
    for path in glob.glob(os.path.join(log_dir, "decode*.log")):
        for line in open(path, errors="ignore"):
            if "[WT][" not in line:
                continue
            kv: dict[str, str] = {}
            for tok in line[line.index("[WT]["):].split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            if "running" not in kv:
                continue
            try:
                recs.append({
                    "t_ms": int(kv.get("t_ms", -1)),
                    "running": int(kv["running"]),
                    "tpot_ms": float(kv.get("meas_tpot_ms", "nan")),
                    "kv_pct": float(kv.get("kv_usage_pct", "nan")),
                    "window": int(kv.get("window", -1)),
                    "demoted": int(kv.get("demoted", 0)),
                    "stop_window": int(kv.get("stop_window", 0)),
                    "preempted": int(kv.get("preempted", 0)),
                    "src": "gwf" if "[WT][GWF]" in line else "native",
                    "instance": os.path.basename(path).split("_")[0],
                })
            except ValueError:
                continue
    return recs


def _kv_cap_tokens(log_dir: str) -> int:
    for path in glob.glob(os.path.join(log_dir, "decode*.log")):
        for line in open(path, errors="ignore"):
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if m:
                return int(m.group(1).replace(",", ""))
    return 0


def _bench_metrics(log_dir: str) -> dict:
    out: dict = {}
    for path in glob.glob(os.path.join(log_dir, "bench_trace_*.log")):
        for line in open(path, errors="ignore"):
            m = re.match(r"^(Request goodput \(req/s\)|Output token goodput \(tok/s\)|"
                         r"Total token goodput \(tok/s\)|Request throughput \(req/s\)|"
                         r"Mean TTFT \(ms\)|Median TTFT \(ms\)|P99 TTFT \(ms\)|"
                         r"Mean TPOT \(ms\)|Median TPOT \(ms\)|P99 TPOT \(ms\)|"
                         r"Successful requests)\s*:\s*([0-9.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def _per_req(run_dir: str) -> dict:
    paths = glob.glob(os.path.join(run_dir, "dataset_result", "test_results_*.csv"))
    if not paths:
        return {}
    rows = []
    for path in paths:
        for r in csv.DictReader(open(path, errors="ignore")):
            try:
                rows.append({
                    "prompt_len": int(float(r["prompt_len"])),
                    "out": int(float(r["output_tokens"])),
                    "ttft_ms": float(r["ttft"]) * 1000.0,   # CSV stores SECONDS
                    "tpot_ms": float(r["tpot"]) * 1000.0,
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        return {}
    ttft = sorted(x["ttft_ms"] for x in rows)
    tpot = sorted(x["tpot_ms"] for x in rows)
    outs = sorted(x["out"] for x in rows)
    ins = sorted(x["prompt_len"] for x in rows)

    def q(a, f):
        return a[min(len(a) - 1, int(len(a) * f))]

    return {
        "n": len(rows),
        "ttft_ms": {"p50": q(ttft, 0.5), "p90": q(ttft, 0.9), "p99": q(ttft, 0.99), "max": ttft[-1]},
        "tpot_ms": {"p50": q(tpot, 0.5), "p90": q(tpot, 0.9), "p99": q(tpot, 0.99), "max": tpot[-1]},
        "output_tokens": {"p50": q(outs, 0.5), "p90": q(outs, 0.9), "max": outs[-1],
                          "mean": st.mean(outs)},
        "prompt_len": {"p50": q(ins, 0.5), "p90": q(ins, 0.9), "max": ins[-1]},
    }


def _curve(recs: list[dict]) -> tuple[list[list[float]], dict]:
    """tpot_p50 per running bucket -> [[upper_bound, tpot_ms], ...]."""
    buckets: dict[int, list[float]] = {}
    for r in recs:
        if r["tpot_ms"] != r["tpot_ms"] or r["tpot_ms"] <= 0:
            continue
        b = r["running"] // BUCKET
        buckets.setdefault(b, []).append(r["tpot_ms"])
    curve, detail = [], {}
    for b in sorted(buckets):
        vals = sorted(buckets[b])
        ub = (b + 1) * BUCKET
        p50 = vals[len(vals) // 2]
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.9))]
        curve.append([ub, round(p50, 1)])
        detail[str(ub)] = {"n": len(vals), "p50": round(p50, 1), "p90": round(p90, 1)}
    return curve, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="one *_scheme4 or *_scheme4_gwf result dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curve-from", default="",
                    help="reuse tpot_curve from another model json (e.g. the GWF run of the same A/B) "
                         "when this scheduler did not log meas_tpot (native greedy has no meas_tpot)")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")
    log_dir = os.path.join(run_dir, "log")
    recs = _wt_records(log_dir)
    src = "gwf" if any(r["src"] == "gwf" for r in recs) else "native"
    my = [r for r in recs if r["src"] == src]
    curve, detail = _curve(my)
    curve_src = "this run"
    if not curve and args.curve_from:
        other = json.load(open(args.curve_from))
        curve = other.get("tpot_curve", [])
        detail = other.get("tpot_curve_detail", {})
        curve_src = f"borrowed from {other.get('source_run', args.curve_from)}"
    kv_cap = _kv_cap_tokens(log_dir)
    bench = _bench_metrics(log_dir)
    per_req = _per_req(run_dir)

    # routing imbalance: peak per-instance running / mean of peaks
    peaks: dict[str, int] = {}
    for r in my:
        peaks[r["instance"]] = max(peaks.get(r["instance"], 0), r["running"])
    imbalance = (max(peaks.values()) / st.mean(peaks.values())) if len(peaks) > 1 else 1.0

    span_s = 0.0
    if per_req:
        span_s = per_req["n"] / max(1e-9, bench.get("Request throughput (req/s)", 1.0))

    model = {
        "source_run": run_dir,
        "scheduler": src,
        "kv_cap_tokens": kv_cap,
        "kv_bytes_per_token": QVWEN_KV_BYTES_PER_TOKEN,
        "kv_cap_gib": round(kv_cap * QVWEN_KV_BYTES_PER_TOKEN / 1024**3, 1) if kv_cap else 0,
        "max_num_seqs": 1024,
        "tpot_curve": curve,
        "tpot_curve_source": curve_src,
        "tpot_curve_detail": detail,
        "tpot_curve_bucket_width": BUCKET,
        "curve_scale_suggested": None,   # filled below when per-request TPOT exists
        "measured": {
            "max_running": max((r["running"] for r in my), default=0),
            "peak_running_per_instance": peaks,
            "routing_imbalance": round(imbalance, 2),
            # per-interval deltas in the log -> sums are the true totals
            "total_demoted": sum(r["demoted"] for r in my),
            "total_stop_window": sum(r["stop_window"] for r in my),
            "total_preempted": sum(r["preempted"] for r in my),
            "max_demoted": max((r["demoted"] for r in my), default=0),
            "max_stop_window": max((r["stop_window"] for r in my), default=0),
            "max_preempted": max((r["preempted"] for r in my), default=0),
            "max_kv_pct": max((r["kv_pct"] for r in my if r["kv_pct"] == r["kv_pct"]), default=None),
            "min_window": min((r["window"] for r in my if r["window"] > 0), default=None),
            "bench": bench,
        },
        "per_request": per_req,
        "arrival": {
            "n": per_req.get("n", 0),
            "span_s": round(span_s, 1),
            "req_s": bench.get("Request throughput (req/s)", 0.0),
        },
        "slo": {"ttft_ms": 1000.0, "tpot_ms": 30.0},
        "notes": [
            "tpot_curve is p50 meas_tpot per running bucket, from the scheduler's own 1 s lines",
            "per-request ttft/tpot in the CSV are SECONDS and were multiplied by 1000 here",
            "kv cap comes from the engine log 'GPU KV cache size: N tokens'",
        ],
    }
    # The bucket curve is a LIFETIME-average per request (scheduler
    # _observed_tpot_s = elapsed/gen_delta), so during a ramp it lags and
    # over-estimates the instantaneous step time.  The sim needs the scale that
    # reproduces the client-observed mean TPOT.
    if curve and per_req:
        wsum = sum(v["p50"] for v in detail.values()) if detail else 0.0
        if wsum:
            model["curve_scale_suggested"] = round(
                per_req["tpot_ms"]["p50"] / (wsum / len(detail)), 2)
    json.dump(model, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    print(f"  scheduler={src}  samples={len(my)}  kv_cap={kv_cap:,} tok "
          f"({model['kv_cap_gib']} GiB)  imbalance={imbalance:.2f}")
    print(f"  peak running={model['measured']['max_running']}  "
          f"demoted={model['measured']['max_demoted']} stop_window={model['measured']['max_stop_window']} "
          f"preempted={model['measured']['max_preempted']}")
    print(f"  goodput={bench.get('Request goodput (req/s)', 'n/a')} req/s")
    if curve:
        print(f"  tpot curve source: {curve_src}")
        print("  tpot curve (running_ub -> p50 ms): " +
              ", ".join(f"{int(u)}:{v}" for u, v in curve))
    else:
        print("  WARNING: no (running, meas_tpot) samples -> curve empty", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
