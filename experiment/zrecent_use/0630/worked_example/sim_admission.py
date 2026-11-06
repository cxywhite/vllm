#!/usr/bin/env python3
"""Offline discrete-event simulator: greedy (stock) vs GWF admission on a trace.

Purpose (the "simulation" tool in the workload-search loop):
  Instead of spending ~20 min per real A/B run, replay a candidate trace through
  BOTH admission controllers in seconds and predict:
    * max running concurrency per decode instance  (the GWF trigger condition)
    * max waiting queue length                       (overload indicator)
    * whether GWF's gate counters fire (stop_window / demoted / preempted)
    * goodput (SLO-satisfied tokens / second) for greedy vs GWF

Calibration (measured 2026-11-06, Qwen-2.5-7B-Instruct, 4P4D, enforce_eager):
  * GPU KV cache = 881,616 tokens per decode instance (decode0 log: kv_cache_utils)
  * decode-side meas_tpot ~23 ms at running <= 96 (bimodal8000, clean load)
  * client-side tpot p50: short(200 tok)=28.3 ms, long(8000 tok)=34.8 ms
  * the "97 ms @ 129-256 / 5264 ms @ 257+" cliff is from the OLD 4080 run that
    contained 30k-token extremes; the TRUE clean-load cliff must be re-probed,
    so it is a tunable curve here (see TPOT_CURVE).

Units are suffix-labelled per AGENTS.md (ms / s / tokens). Read-only: this script
never deletes or moves anything; it only writes its own report under --out.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field

csv.field_size_limit(1 << 30)

# ---- measured constants (see docstring) ------------------------------------
KV_CAP_TOKENS_PER_INSTANCE = 881_616
N_DECODE_INSTANCES = 4
MAX_NUM_SEQS = 1024
PREFILL_RATE_TOK_S = 25_000.0
CLIENT_OVERHEAD_MS = 10.0      # decode-side meas_tpot -> client-side tpot gap
TTFT_SLO_MS = 1000.0
TPOT_SLO_MS = 30.0
GWF_WINDOW_INIT = 256
GWF_WINDOW_FLOOR = 32
GWF_WINDOW_TPOT_TARGET_MS = 120.0   # window shrinks when meas_tpot > this
GWF_DEMOTE_MIN_RUNNING_RATIO = 0.5   # demote only when running >= A/2

# decode per-token time (ms) as a function of running concurrency.
# curve name -> list of (running_upper_bound_inclusive, tpot_ms). Linear interp.
TPOT_CURVES: dict[str, list[tuple[int, float]]] = {
    # clean-load estimate: ~20 ms at low concurrency (matches bimodal8000 23 ms),
    # old-run cliff values kept for the 129+/257+ regimes until re-probed.
    "clean": [(64, 20.0), (128, 32.0), (256, 97.0), (10_000, 5264.0)],
    # the exact numbers from the code comment (2026-09-24, 4080 trace w/ extremes)
    "old4080": [(8, 74.0), (32, 84.0), (128, 98.0), (256, 97.0), (10_000, 5264.0)],
}


def tpot_ms(running: int, curve: list[tuple[int, float]]) -> float:
    for ub, v in curve:
        if running <= ub:
            return v
    return curve[-1][1]


@dataclass
class Req:
    t_arr_s: float
    prompt_len: int
    out_tokens: int
    # runtime
    t_admit_s: float = 0.0
    t_first_s: float = 0.0
    first_done: bool = False
    tokens_done: int = 0
    decode_time_s: float = 0.0
    preemptions: int = 0
    finished: bool = False
    good: bool = False

    @property
    def kv_tokens(self) -> int:
        return self.prompt_len + self.out_tokens


def tpot_of(r: Req) -> float:
    if r.tokens_done <= 0:
        return float("nan")
    return r.decode_time_s / r.tokens_done * 1000.0


def simulate(requests: list[Req], policy: str, curve_name: str,
             imbalance: float) -> dict:
    """Single decode instance fed (1/N_DECODE_INSTANCES * imbalance) of the load.

    `imbalance` models the observed uneven proxy routing (measured per-instance
    running 18..96 with mean ~50 on bimodal8000, i.e. ~1.9x the balanced share
    on the hottest instance). Default 1.0 = balanced lower bound.
    """
    curve = TPOT_CURVES[curve_name]
    span_s = max(1.0, requests[-1].t_arr_s + 60.0)
    dt_s = 0.1
    running: list[Req] = []
    waiting: list[Req] = []
    kv_used = 0
    window = GWF_WINDOW_INIT
    meas_tpot: float | None = None
    last_window_update_s = -1.0

    counters = {"admitted": 0, "stop_window": 0, "demoted": 0, "preempted": 0,
                "recompute_tokens": 0, "hopeless_forced": 0}
    max_running = 0
    max_waiting = 0
    good_tokens = 0

    # request feed: take a fraction of arrivals (balanced) and scale by imbalance
    # for the "hottest instance" estimate; imbalance only multiplies arrival rate.
    arrivals = sorted(requests, key=lambda r: r.t_arr_s)
    i_arr = 0
    n_total = len(arrivals)
    # each instance gets ~1/N of requests; hottest gets `imbalance` x that.
    share = imbalance / N_DECODE_INSTANCES
    feed_acc = 0.0

    t = 0.0
    # The modeled instance receives `share = imbalance / N` of the arrivals.
    # Observed hot-instance running (96) was ~3x the balanced share (31), i.e.
    # the predictor router concentrates long requests onto one decode instance.
    my_idx = [j for j in range(n_total) if j % N_DECODE_INSTANCES == 0]

    def is_hopeless(r: Req, now: float) -> bool:
        # TTFT blow: waited > 1000ms before first token
        if r.first_done:
            ttft_ms = (r.t_first_s - r.t_arr_s) * 1000.0
        else:
            ttft_ms = (now - r.t_arr_s) * 1000.0
        if ttft_ms > TTFT_SLO_MS:
            return True
        # TPOT blow: >=4 samples and observed > SLO
        if r.tokens_done >= 4 and tpot_of(r) > TPOT_SLO_MS:
            return True
        return False

    def eviction_value(r: Req) -> float:
        gen_ratio = min(1.0, r.tokens_done / max(1, r.out_tokens))
        rem_ratio = min(1.0, (r.out_tokens - r.tokens_done) / 32768.0)
        slack = 1.0 - min(1.0, tpot_of(r) / TPOT_SLO_MS) if r.tokens_done else 1.0
        return gen_ratio - slack - 0.5 * rem_ratio

    n_steps = int(span_s / dt_s)
    for _ in range(n_steps):
        demoted_this_step = 0  # mirror the real per-step cap (16)
        # 1. arrivals (fractional feed at rate `share` models the hot instance)
        while i_arr < n_total and arrivals[i_arr].t_arr_s <= t:
            feed_acc += share
            if feed_acc >= 1.0:
                waiting.append(arrivals[i_arr])
                feed_acc -= 1.0
            i_arr += 1

        # 2. decode: advance running requests
        rn = len(running)
        step_tpot_ms = tpot_ms(rn, curve)
        per_step_adv = (dt_s / (step_tpot_ms / 1000.0)) if step_tpot_ms > 0 else 1.0
        for r in running:
            r.tokens_done += per_step_adv
            r.decode_time_s += dt_s
            if not r.first_done and r.tokens_done >= 1.0:
                r.first_done = True
                r.t_first_s = t
        # finish
        still = []
        for r in running:
            if r.tokens_done >= r.out_tokens:
                r.finished = True
                kv_used -= r.kv_tokens
                ttft_ms = (r.t_first_s - r.t_arr_s) * 1000.0
                r.good = (ttft_ms <= TTFT_SLO_MS and tpot_of(r) <= TPOT_SLO_MS)
                if r.good:
                    good_tokens += r.out_tokens
            else:
                still.append(r)
        running = still

        # 3. GWF window update (once per second)
        if policy == "gwf" and t - last_window_update_s >= 1.0:
            # EMA of per-request tpot over requests with >=4 samples
            samples = [tpot_of(r) for r in running if r.tokens_done >= 4]
            if samples:
                mean = sum(samples) / len(samples)
                meas_tpot = mean if meas_tpot is None else 0.5 * meas_tpot + 0.5 * mean
                last_window_update_s = t
                if meas_tpot > GWF_WINDOW_TPOT_TARGET_MS and rn >= window:
                    window = max(GWF_WINDOW_FLOOR, window // 2)
                elif meas_tpot <= GWF_WINDOW_TPOT_TARGET_MS:
                    window = min(256, window + max(1, window // 8))

        # 4. admission (waiting -> running)
        admitted = 0
        guard = 0
        while waiting:
            guard += 1
            if guard > 10_000:  # safety valve: should never trigger
                break
            if policy == "gwf" and rn >= window:
                counters["stop_window"] += 1
                break
            # demote: running >= A/2, head is hopeless, bounded per step (16)
            if (policy == "gwf" and rn >= window * GWF_DEMOTE_MIN_RUNNING_RATIO
                    and demoted_this_step < 16 and is_hopeless(waiting[0], t)):
                waiting.append(waiting.pop(0))
                counters["demoted"] += 1
                demoted_this_step += 1
                continue
            r = waiting[0]
            if kv_used + r.kv_tokens > KV_CAP_TOKENS_PER_INSTANCE:
                # KV full: the real vLLM scheduler does NOT preempt running
                # requests just to admit a waiting one -- the new request simply
                # waits (the waiting loop `break`s on allocate_slots()==None).
                break
            # admit
            r = waiting.pop(0)
            r.t_admit_s = t
            kv_used += r.kv_tokens
            running.append(r)
            counters["admitted"] += 1
            rn = len(running)
            admitted += 1

        max_running = max(max_running, len(running))
        max_waiting = max(max_waiting, len(waiting))
        t += dt_s

    # client-side tpot for finished requests is already per-request; goodput
    # = good tokens / span_s (per instance)
    goodput = good_tokens / span_s
    return {
        "policy": policy, "max_running": max_running, "max_waiting": max_waiting,
        "goodput_tok_s": goodput, "good_tokens": good_tokens, "span_s": span_s,
        "counters": counters, "final_meas_tpot_ms": (meas_tpot or 0.0),
        "final_window": window,
        "n_total_fed": len(my_idx), "share": share,
    }


def load_trace(path: str) -> list[Req]:
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="ignore")))
    reqs = []
    t0 = None
    for r in rows:
        ts_ms = float(r.get("timestamp", 0))
        if t0 is None:
            t0 = ts_ms
        reqs.append(Req(
            t_arr_s=(ts_ms - t0) / 1000.0,
            prompt_len=int(float(r.get("prompt_len", 0))),
            out_tokens=int(float(r.get("output_tokens", 0))),
        ))
    return reqs


def build_bimodal(n_bursts, reqs_per_burst, long_frac, out_short, out_long,
                  prompt_len, burst_period_s, burst_duration_s, front_load=False):
    reqs = []
    idx = 0
    for b in range(n_bursts):
        for i in range(reqs_per_burst):
            is_long = i < int(reqs_per_burst * long_frac)
            if front_load:
                ts_s = (i / max(1, reqs_per_burst - 1)) * burst_duration_s
            else:
                ts_s = b * burst_period_s + (i / max(1, reqs_per_burst - 1)) * burst_duration_s
            reqs.append(Req(t_arr_s=ts_s, prompt_len=prompt_len,
                            out_tokens=out_long if is_long else out_short))
            idx += 1
    reqs.sort(key=lambda r: r.t_arr_s)
    return reqs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="trace CSV to simulate")
    ap.add_argument("--curve", default="clean", choices=list(TPOT_CURVES))
    ap.add_argument("--imbalance", type=float, default=1.6,
                    help="hottest-instance arrival multiplier vs balanced (obs ~1.9)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep workload params instead of one trace")
    ap.add_argument("--out", help="write JSON report")
    # workload builder params
    ap.add_argument("--n-bursts", type=int, default=10)
    ap.add_argument("--reqs-per-burst", type=int, default=60)
    ap.add_argument("--long-frac", type=float, default=0.5)
    ap.add_argument("--out-short", type=int, default=200)
    ap.add_argument("--out-long", type=int, default=8000)
    ap.add_argument("--prompt-len", type=int, default=56)
    ap.add_argument("--burst-period-s", type=float, default=60.0)
    ap.add_argument("--burst-duration-s", type=float, default=6.0)
    ap.add_argument("--front-load", action="store_true")
    a = ap.parse_args()

    def run_one(reqs, tag):
        rows = []
        for policy in ("greedy", "gwf"):
            r = simulate(reqs, policy, a.curve, a.imbalance)
            r["tag"] = tag
            rows.append(r)
        return rows

    if a.sweep:
        out = []
        grid = [
            # (tag, n_bursts, reqs_per_burst, long_frac, out_short, out_long, front_load)
            ("bim8000", 10, 60, 0.5, 200, 8000, False),
            ("bim20000", 10, 60, 0.5, 200, 20000, False),
            ("dense3000_b800", 1, 800, 0.0, 3000, 3000, True),
            ("dense2500_b700", 1, 700, 0.0, 2500, 2500, True),
            ("bim3000_b90", 10, 90, 0.5, 200, 3000, False),
            ("bim3000_b90_fl", 1, 540, 0.5, 200, 3000, True),
        ]
        for tag, nb, rpb, lf, os_, ol, fl in grid:
            reqs = build_bimodal(nb, rpb, lf, os_, ol, a.prompt_len,
                                 a.burst_period_s, a.burst_duration_s, fl)
            out.extend(run_one(reqs, tag))
        if a.out:
            json.dump(out, open(a.out, "w"), indent=2)
            print(f"wrote {a.out}")
        hdr = f"{'workload':<16}{'policy':<8}{'max_run':>8}{'max_wait':>9}{'goodput':>9}{'stopwin':>8}{'demote':>7}{'preempt':>8}{'meas_tpot':>10}{'window':>7}"
        print(hdr)
        for r in out:
            c = r["counters"]
            print(f"{r['tag']:<16}{r['policy']:<8}{r['max_running']:>8}{r['max_waiting']:>9}"
                  f"{r['goodput_tok_s']:>9.1f}{c['stop_window']:>8}{c['demoted']:>7}"
                  f"{c['preempted']:>8}{r['final_meas_tpot_ms']:>10.1f}{r['final_window']:>7}")
        return 0

    reqs = load_trace(a.trace) if a.trace else build_bimodal(
        a.n_bursts, a.reqs_per_burst, a.long_frac, a.out_short, a.out_long,
        a.prompt_len, a.burst_period_s, a.burst_duration_s, a.front_load)
    rows = run_one(reqs, os.path.basename(a.trace) if a.trace else "workload")
    for r in rows:
        c = r["counters"]
        print(f"[{r['policy']}] max_running={r['max_running']} max_waiting={r['max_waiting']} "
              f"goodput={r['goodput_tok_s']:.1f} tok/s  stop_window={c['stop_window']} "
              f"demoted={c['demoted']} preempted={c['preempted']} "
              f"recompute_tokens={c['recompute_tokens']} meas_tpot={r['final_meas_tpot_ms']:.1f}ms "
              f"window={r['final_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
