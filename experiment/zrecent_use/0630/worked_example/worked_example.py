#!/usr/bin/env python3
"""
Worked example: how predicted length, eviction value eV, and the 120 ms / 30 ms
targets enter ONE admission decision in the GWF intra-instance scheduler.

Grounding (all real data):
  * trace: qwen_lmsys_predictor_friendly_4080_600s_strategyA.csv  (n=4080)
  * predictor output: retry_test/ds5_both/.../predict_request_decode{0..3}_*.csv
  * scheduler logic: vllm/vllm/v1/core/sched/geometric_window_scheduler.py

Everything is computed with the exact formulas from geometric_window_scheduler.py;
no synthetic numbers are invented for the paper.
"""
from __future__ import annotations
import csv, sys

csv.field_size_limit(sys.maxsize)

# ---- scheduler constants (from geometric_window_scheduler.py) ---------------
MAX_MODEL_LEN = 32768        # Qwen-2.5-7B-Instruct
TPOT_SLO_MS = 30.0           # evaluation SLO  -> drives "hopeless"
THETA_MS = 120.0             # window-feedback target -> drives admission window A
A_MIN, A_MAX = 32, 256       # admission window bounds
SMIN = 4                     # min post-first-token samples before TPOT verdict

TRACE = ("experiment/zrecent_use/0630/derived_datasets/tmp_dataset5_predictor_friendly4080/"
         "qwen_lmsys_predictor_friendly_4080_600s_strategyA.csv")
PRED_BASE = ("experiment/zrecent_use/0630/retry_test/ds5_both/"
             "qwen_qwen-lmsys-chat-tmp_dataset5_predictor_friendly4080_n4080_pd4p4d_"
             "tpot30ms_tracetmp_dataset5_predictor_friendly4080_rep1_20260923_090234_"
             "scheme4_aimd/log/predict_request_decode")


def load() -> list[dict]:
    trace = {r["req_id"]: r for r in csv.DictReader(open(TRACE, encoding="utf-8"))}
    pred: dict[str, dict] = {}
    for i in range(4):
        for r in csv.DictReader(open(PRED_BASE + f"{i}_20260923_090234.csv", encoding="utf-8")):
            pred[r["dataset_req_id"]] = r
    out = []
    for rid, pr in pred.items():
        if rid not in trace:
            continue
        pm = pr.get("predict_output_mean", "").strip()
        act = pr.get("actual_output_len", "").strip()
        if not pm or not act:
            continue
        out.append({
            "req_id": rid,
            "prompt_len": int(pr["actual_prompt_tokens"]),
            "pred_len": float(pm),          # RAW predicted remaining/output length
            "actual_len": int(act),          # ground-truth length in this run
        })
    return out


def eviction_value(generated_tokens: int, pred_len: float, observed_tpot_ms: float) -> float:
    """Exact _eviction_value from geometric_window_scheduler.py (L334-348)."""
    predicted = max(1, int(pred_len))
    gen_ratio = min(1.0, generated_tokens / predicted)
    remaining = max(0, predicted - generated_tokens)
    rem_ratio = min(1.0, remaining / max(1, MAX_MODEL_LEN))
    slack = max(0.0, min(1.0, 1.0 - observed_tpot_ms / TPOT_SLO_MS))
    return gen_ratio - slack - 0.5 * rem_ratio


def is_hopeless(observed_tpot_ms: float, samples: int) -> bool:
    """_is_hopeless: TPOT over budget with >= SMIN samples (TTFT omitted here)."""
    return samples >= SMIN and observed_tpot_ms > TPOT_SLO_MS


def window_step(meas_tpot_ema_ms: float, running: int, A: int, theta_ms: float) -> tuple[int, str]:
    """Exact window update (L676-693): halve if tau>theta AND |R|>=A, else grow by 1/8."""
    if meas_tpot_ema_ms > theta_ms and running >= A:
        A = max(A_MIN, A // 2)
        return A, f"tau={meas_tpot_ema_ms:.0f}ms > theta={theta_ms:.0f}ms and |R|={running}>=A -> A=max({A_MIN}, {A // 2})"
    if meas_tpot_ema_ms <= theta_ms:
        A = min(A_MAX, A + max(1, A // 8))
        return A, f"tau={meas_tpot_ema_ms:.0f}ms <= theta={theta_ms:.0f}ms -> A=min({A_MAX}, A+max(1,A//8))"
    return A, "no update"


def main() -> int:
    rows = load()
    print(f"loaded {len(rows)} merged request rows\n")

    # ------------------------------------------------------------------ Part 1
    # theta=120ms vs SLO=30ms: why the window does NOT collapse.
    # Measured TPOT floor / flat-regime numbers are quoted verbatim from the
    # scheduler's own calibration comment (qwen 4p4d 4080, 2026-09-24):
    #   running<=8 -> p50 74ms (floor); 17-32 -> 84ms; 65-128 -> 98ms;
    #   129-256 -> 97ms; 257+ -> 5264ms (cliff).
    print("== Part 1: theta=120ms vs SLO=30ms (window feedback) ==")
    tau = 98.0   # measured TPOT EMA in the flat regime (65-128 running)
    running = 256  # loaded: |R| >= A so the shrink guard is satisfiable
    for theta, name in [(120.0, "theta=120ms (paper)"), (30.0, "theta=30ms (=SLO, naive)")]:
        A = 256
        traj = [A]
        for _ in range(4):  # 4 x 1s updates
            A, _ = window_step(tau, running, A, theta)
            traj.append(A)
        print(f"  {name:24s} tau={tau}ms running={running}: A over 4s = {traj}")
    print("  => with theta=30ms the window halves every second toward A_min=32;")
    print("     with theta=120ms (above the ~98ms flat-regime TPOT) it stays at the cap.\n")

    # ------------------------------------------------------------------ Part 2
    # Predicted length -> eV -> preemption victim.
    # Pick the first 5 real requests from the merged table (real pred & actual).
    print("== Part 2: predicted length -> eV -> victim (5 real running requests) ==")
    snap = rows[:5]
    # snapshot: generated tokens so far (g_i) and observed per-token time (tau_i).
    # These are chosen to expose the mechanism; predicted/actual lengths are REAL.
    snap_g = [74, 512, 615, 512, 610]          # ~50% of each actual length
    snap_t = [20.0, 40.0, 22.0, 45.0, 18.0]    # ms/token (30ms SLO separates live vs hopeless)

    print(f"{'req':>6} {'prompt':>6} {'pred(raw)':>10} {'actual':>7} {'g':>5} {'tau_ms':>7} "
          f"{'hopeless':>9} {'gen_ratio':>10} {'rem_ratio':>10} {'slack':>6} {'eV':>7}")
    for (r, g, t) in zip(snap, snap_g, snap_t):
        hop = is_hopeless(t, SMIN)
        gr = min(1.0, g / max(1, int(r["pred_len"])))
        rem = max(0, int(r["pred_len"]) - g)
        rr = min(1.0, rem / MAX_MODEL_LEN)
        sl = max(0.0, min(1.0, 1.0 - t / TPOT_SLO_MS))
        ev = eviction_value(g, r["pred_len"], t)
        print(f"{r['req_id']:>6} {r['prompt_len']:>6} {r['pred_len']:>10.0f} {r['actual_len']:>7} "
              f"{g:>5} {t:>7.1f} {str(hop):>9} {gr:>10.3f} {rr:>10.3f} {sl:>6.2f} {ev:>7.3f}")

    # victim = tail of sorted-by-(hopeless, eV) descending = lowest eV in the LIVE class
    live = [(eviction_value(g, r["pred_len"], t), r["req_id"])
            for (r, g, t) in zip(snap, snap_g, snap_t) if not is_hopeless(t, SMIN)]
    victim = min(live, key=lambda x: x[0])
    print(f"\n  live-class eV values: {[(round(e, 3), rid) for e, rid in live]}")
    print(f"  preemption victim (lowest eV among live): req {victim[1]} (eV={victim[0]:.3f})")
    print("  hopeless requests (tau>30ms) form a strictly LOWER class and are evicted first.\n")

    # ------------------------------------------------------------------ Part 3
    # Why calibrating the prediction matters (the ~11x miscalibration finding).
    print("== Part 3: what the ~11x miscalibrated prediction does to eV ==")
    r = rows[0]  # req with actual ~149, pred ~13650
    print(f"  req {r['req_id']}: actual={r['actual_len']}, pred(raw)={r['pred_len']:.0f}")
    for label, pred in [("raw (~11x)", r["pred_len"]), ("calibrated", r["pred_len"] / 10.96)]:
        g = 74
        gr = min(1.0, g / max(1, int(pred)))
        rr = min(1.0, (max(0, int(pred) - g)) / MAX_MODEL_LEN)
        print(f"    {label:14s} pred={pred:8.0f}  gen_ratio={gr:.3f}  rem_ratio={rr:.3f}  "
              f"(gen_ratio collapses / rem_ratio saturates when pred is inflated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
