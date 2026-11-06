#!/usr/bin/env python3
"""Live resumable simulation: initialise from a running experiment's live state and
replay the REMAINING trace through greedy/GWF, so one real run can be simulated and
re-adjusted many times before it finishes.

Incremental inputs (verified to grow during a run):
  * log/predict_request_decode*.csv -- one row per request (dataset_req_id, event_ts_s,
    actual_output_len) => tells which trace requests are already done.
  * log/decode*.log  -- [WT][NATIVE] or [WT][GWF] every 1 s: running / waiting /
    kv_usage_pct / preempted_recompute_tokens / (GWF: window, meas_tpot_ms).
  * log/bench_trace*.log -- tqdm progress: completed/total + ETA.

Read-only; never touches the running processes. Usage:
  python3 live_sim.py --dir <result_dir> --trace <trace.csv> [--greedy-kv 31.0]
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_admission as S  # reuse Req / tpot curve / constants

csv.field_size_limit(1 << 30)


def read_trace(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="ignore")))
    out = []
    t0 = None
    for r in rows:
        ts = float(r.get("timestamp", 0))
        if t0 is None:
            t0 = ts
        out.append((r.get("req_id", ""), (ts - t0) / 1000.0,
                    int(float(r.get("prompt_len", 0))),
                    int(float(r.get("output_tokens", 0)))))
    return out, (t0 or 0.0)


def read_done(req_dir):
    done = set()
    for f in glob.glob(req_dir + "/log/predict_request_decode*.csv"):
        try:
            for r in csv.DictReader(open(f, encoding="utf-8", errors="ignore")):
                rid = r.get("dataset_req_id")
                if rid:
                    done.add(str(rid))
        except Exception:
            pass
    return done


def read_live_state(req_dir):
    """Per-instance latest snapshot + which variant (NATIVE=greedy, GWF)."""
    state = {}
    variant = None
    for f in sorted(glob.glob(req_dir + "/log/decode*.log")):
        inst = os.path.basename(f)
        last = None
        for line in open(f, encoding="utf-8", errors="ignore"):
            if "[WT][GWF]" in line:
                variant = "gwf"
                kv = dict(re.findall(r"(\w+)=([-\d.]+)", line))
                last = {"running": int(float(kv.get("running", 0))),
                        "waiting": int(float(kv.get("waiting", 0))),
                        "window": int(float(kv.get("window", 256))),
                        "meas_tpot_ms": float(kv.get("meas_tpot_ms", -1)),
                        "kv_usage_pct": None,
                        "preempt_tokens": float(kv.get("recompute_tokens", 0))}
            elif "[WT][NATIVE]" in line:
                variant = "greedy"
                kv = dict(re.findall(r"(\w+)=([-\d.]+)", line))
                last = {"running": int(float(kv.get("running", 0))),
                        "waiting": int(float(kv.get("waiting", 0))),
                        "window": 256,
                        "meas_tpot_ms": -1.0,
                        "kv_usage_pct": float(kv.get("kv_usage_pct", 0)),
                        "preempt_tokens": float(kv.get("preempted_recompute_tokens", 0))}
        if last:
            state[inst] = last
    return state, variant


def read_progress(req_dir):
    f = None
    for f in glob.glob(req_dir + "/log/bench_trace*.log"):
        pass
    if not f:
        return None
    txt = open(f, encoding="utf-8", errors="ignore").read().split("\r")
    pat = re.compile(r"(\d+)/(\d+)\s*\[(\d+):(\d+)<([^,]+),\s*([\d.]+)s/it\]")
    for chunk in reversed(txt):
        m = pat.search(chunk)
        if m:
            return {"done": int(m.group(1)), "total": int(m.group(2)),
                    "elapsed_s": int(m.group(3)) * 60 + int(m.group(4)),
                    "eta": m.group(5), "s_per_it": float(m.group(6))}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="experiment result dir (with log/)")
    ap.add_argument("--trace", required=True, help="trace CSV replayed by the run")
    ap.add_argument("--imbalance", type=float, default=2.0)
    ap.add_argument("--curve", default="real")
    ap.add_argument("--thrash-ms", type=float, default=S.THRASH_TPOT_MS)
    a = ap.parse_args()

    trace, _ = read_trace(a.trace)
    done = read_done(a.dir)
    progress = read_progress(a.dir)
    state, variant = read_live_state(a.dir)

    n_total = len(trace)
    n_done = len(done & {t[0] for t in trace})
    elapsed = (progress or {}).get("elapsed_s", 0)

    # live aggregate running / kv
    runs = [s["running"] for s in state.values()] or [0]
    kvs = [s["kv_usage_pct"] for s in state.values() if s["kv_usage_pct"] is not None]
    meas = [s["meas_tpot_ms"] for s in state.values() if s["meas_tpot_ms"] > 0]

    print(f"=== LIVE STATE ({os.path.basename(a.dir.rstrip('/'))}) ===")
    print(f"  variant={variant}  elapsed≈{elapsed}s  done={n_done}/{n_total} "
          f"({100.0*n_done/max(1,n_total):.1f}%)  ETA={ (progress or {}).get('ETA') }")
    print(f"  per-instance running={runs} (max={max(runs)}) waiting={[s['waiting'] for s in state.values()]}")
    if kvs:
        print(f"  kv_usage_pct={[round(x,1) for x in kvs]} (mean={sum(kvs)/len(kvs):.1f}%)")
    if meas:
        print(f"  decode meas_tpot_ms={[round(x,1) for x in meas]}  (window="
              f"{[s['window'] for s in state.values()]})")

    if n_done == 0 or elapsed == 0:
        print("  (too early to forecast)")
        return

    # ---- forecast the remaining trace ----
    # remaining requests are those not in `done`, at their original timestamps.
    remaining = [t for t in trace if t[0] not in done]
    S.THRASH_TPOT_MS = a.thrash_ms
    curve = S.TPOT_CURVES[a.curve]

    # arrival rate observed so far (per second, over the whole trace window)
    lam = n_done / elapsed
    # service rate implied by completion
    print(f"\n  observed: λ={lam:.2f} req/s, μ={n_done/max(1,elapsed):.2f} req/s")

    # ---- replay the remaining requests from the live state ----
    # per-instance replay: feed share = imbalance/4 of remaining, init running/kv.
    share = a.imbalance / S.N_DECODE_INSTANCES
    reqs = [S.Req(t_arr_s=max(0.0, t[1] - elapsed), prompt_len=t[2], out_tokens=t[3])
            for t in remaining]
    reqs.sort(key=lambda r: r.t_arr_s)
    sim = S.simulate(reqs, "gwf", a.curve, a.imbalance) if reqs else None
    print("  ---- forecast (remaining trace replayed from live state, GWF modelled) ----")
    if sim:
        print(f"  simulated remaining: max_running={sim['max_running']} "
              f"max_waiting={sim['max_waiting']} stop_window={sim['counters']['stop_window']} "
              f"demoted={sim['counters']['demoted']} window_min={sim['final_window']}")
        pred = sim["max_running"] >= 256 or sim["counters"]["stop_window"] > 0
        print(f"  => window would {'SHRINK/ENFORCE' if pred else 'stay at 256'} "
              f"({'cliff/cap engaged' if pred else 'no cap -> GWF ~ no-op'})")
    else:
        print("  no remaining requests")

    # ---- live KV-based extrapolation (more reliable than the sim's absolute running) ----
    if kvs and max(runs) > 0:
        run_now = max(runs)
        kv_now = sum(kvs) / len(kvs)
        avg_kv = (kv_now / 100.0) * S.KV_CAP_TOKENS_PER_INSTANCE / run_now
        run_kv_full = S.KV_CAP_TOKENS_PER_INSTANCE / max(1.0, avg_kv)
        kv_at_cap = 256.0 * avg_kv / S.KV_CAP_TOKENS_PER_INSTANCE * 100.0
        print(f"\n  ---- live KV extrapolation (avg {avg_kv:.0f} KV tokens/request) ----")
        print(f"  greedy (no window): KV hits 100% at running≈{run_kv_full:.0f}"
              f" -> {'CLIFF / thrash' if run_kv_full < 1024 else 'may OOM first'}")
        print(f"  GWF (window cap 256): KV only ≈{kv_at_cap:.0f}% at 256"
              f" -> {'avoids cliff' if kv_at_cap < 90 else 'still near cliff'}")
        if kv_at_cap < 90 < 100 and run_kv_full > 256:
            print("  => clean separation: GWF caps before KV exhaustion, greedy does not.")

    # ---- recommendation (uses the KV extrapolation, not just the current snapshot) ----
    run_max = max(runs)
    kv_max = max(kvs) if kvs else 0.0
    print("\n  ---- adjustment hint ----")
    if kvs and max(runs) > 0:
        run_now = max(runs)
        kv_now = sum(kvs) / len(kvs)
        avg_kv = (kv_now / 100.0) * S.KV_CAP_TOKENS_PER_INSTANCE / run_now
        run_kv_full = S.KV_CAP_TOKENS_PER_INSTANCE / max(1.0, avg_kv)
        kv_at_cap = 256.0 * avg_kv / S.KV_CAP_TOKENS_PER_INSTANCE * 100.0
        if run_kv_full < 1024 and kv_at_cap < 90:
            print(f"  ON TRACK (clean separation): greedy exhausts KV at running≈{run_kv_full:.0f} (cliff),"
                  f" GWF caps at 256 with KV≈{kv_at_cap:.0f}% (no cliff).")
            print("           Keep this trace/cap -- it exercises the theta window as designed.")
        elif kv_at_cap >= 90:
            print(f"  GWF ALSO NEAR CLIFF: KV≈{kv_at_cap:.0f}% at the 256 cap -> less separation.")
            print("           Consider a lower cap or fewer long requests.")
        else:
            print(f"  TOO LIGHT: greedy KV-full at running≈{run_kv_full:.0f} >= 1024 (or unreachable);")
            print("             GWF gate no-op -> raise cap / add long requests.")
    else:
        print(f"  running={run_max}, KV={kv_max:.0f}% -> waiting for more live data.")


if __name__ == "__main__":
    main()
