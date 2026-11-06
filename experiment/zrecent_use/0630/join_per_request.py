#!/usr/bin/env python3
"""
Join per-request data from one A/B run into a single analysis CSV.

Inputs (per rep, under <result_dir>/):
  * dataset_result/test_results_*.csv  -> per-request timing (latency/ttft/tpot/itl)
      and lengths (prompt_len, output_tokens, expect_output_len)
  * log/predict_request_decode*.csv    -> per-request predicted length
      (predict_output_mean) + actual_output_len

Join key: test_results.req_id == predict.dataset_req_id.

Outputs:
  <result_dir>/per_request_joined.csv  -> one row per request with is_good flag
  stdout summary -> bins by (actual output len) and (pred/actual) with goodput share

Usage:
  python3 join_per_request.py <result_dir>
"""
from __future__ import annotations
import csv, sys, statistics
from pathlib import Path

csv.field_size_limit(sys.maxsize)

TTFT_SLO_MS = 1000.0
TPOT_SLO_MS = 30.0


def load_test_results(d: Path) -> dict[str, dict]:
    out = {}
    for p in d.glob("dataset_result/test_results_*.csv"):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out[r["req_id"]] = r
    return out


def load_predictions(d: Path) -> dict[str, dict]:
    out = {}
    for p in d.glob("log/predict_request_decode*.csv"):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out[r.get("dataset_req_id", "")] = r
    return out


def _f(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def main() -> int:
    d = Path(sys.argv[1])
    tr = load_test_results(d)
    pr = load_predictions(d)
    print(f"test_results={len(tr)}  predictions={len(pr)}")

    joined = []
    for rid, r in tr.items():
        p = pr.get(rid, {})
        # NOTE: test_results stores latency/ttft/tpot/itl in SECONDS -> *1000 to ms
        ttft_ms = _f(r.get("ttft", "-1")) * 1000.0
        tpot_ms = _f(r.get("tpot", "-1")) * 1000.0
        prompt_len = int(r.get("prompt_len", "0") or "0")
        out_tokens = int(r.get("output_tokens", "0") or "0")
        pred_mean = _f(p.get("predict_output_mean", "-1"))
        is_good = (ttft_ms >= 0 and ttft_ms <= TTFT_SLO_MS
                   and tpot_ms >= 0 and tpot_ms <= TPOT_SLO_MS)
        joined.append({
            "req_id": rid,
            "prompt_len": prompt_len,
            "output_tokens": out_tokens,
            "predict_output_mean": pred_mean,
            "pred_actual_ratio": (pred_mean / out_tokens if out_tokens > 0 else -1.0),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "itl_ms": _f(r.get("itl", "-1")) * 1000.0,
            "is_good": 1 if is_good else 0,
        })

    # write joined CSV
    fn = list(joined[0].keys()) if joined else []
    outp = d / "per_request_joined.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(joined)
    print(f"wrote {outp} ({len(joined)} rows)")

    if not joined:
        return 0

    # summary: goodput share by actual output length bin
    def bins(key, edges):
        out = {}
        for r in joined:
            v = r[key]
            b = next((e for e in edges if v <= e), edges[-1])
            out.setdefault(b, []).append(r)
        return out

    print("\n== goodput share by actual output length ==")
    for b, rows in sorted(bins("output_tokens", [200, 8000]).items()):
        n = len(rows)
        good = sum(r["is_good"] for r in rows)
        good_tok = sum(r["output_tokens"] for r in rows if r["is_good"])
        tot_tok = sum(r["output_tokens"] for r in rows)
        med_tpot = statistics.median([r["tpot_ms"] for r in rows if r["tpot_ms"] >= 0])
        print(f"  out<={b:5d}: n={n:3d}  good_req={good}/{n}  good_tok={good_tok}/{tot_tok} "
              f"  med_tpot={med_tpot:.1f}ms")

    print("\n== goodput share by pred/actual ratio ==")
    for b, rows in sorted(bins("pred_actual_ratio", [1.0, 2.0, 10.0]).items()):
        n = len(rows)
        good = sum(r["is_good"] for r in rows)
        print(f"  ratio<={b:5.1f}: n={n:3d}  good_req={good}/{n}")

    overall_good = sum(r["is_good"] for r in joined)
    print(f"\nOVERALL: good_req={overall_good}/{len(joined)}  "
          f"good_tok={sum(r['output_tokens'] for r in joined if r['is_good'])}/"
          f"{sum(r['output_tokens'] for r in joined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
