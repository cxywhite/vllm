#!/usr/bin/env python3
"""
Build a "bimodal + bursty" trace that should make GWF beat greedy admission.

Workload hypothesis (derived from the measurements so far):
  * short input  -> KV pressure comes from OUTPUT length, not input;
  * bimodal output (short "live" + long "hopeless") -> long requests miss the 30ms
    TPOT SLO and become "hopeless"; greedy serves them FIFO and lets them starve the
    short ones, GWF demotes them and keeps the short ones alive;
  * bursty arrival -> high concurrency bursts push the decode instance past the
    TPOT cliff (~257 running); greedy overfills into the cliff, GWF caps at A=256.

The trace keeps REAL prompts (from the burstgpt/qwen trace pool) and only rewrites
output_tokens (bimodal) and timestamp (bursty), so the benchmark replays realistic
requests.  All knobs are env-overridable.

Output CSV columns match benchmark_serving_trace.py:
  req_id, prompt, prompt_len, output_tokens, temperature, top_p, top_k,
  repetition_penalty, timestamp
"""
from __future__ import annotations
import csv, os, sys, random
from pathlib import Path

# ---- knobs (unit-suffixed, per AGENTS.md) ----------------------------------
TRACE_SPAN_S = 600.0            # total trace duration
BURST_PERIOD_S = 60.0           # one burst every this many seconds
BURST_DURATION_S = 6.0          # requests inside a burst are spread over this window
REQS_PER_BURST = 60             # requests per burst
LONG_FRACTION = 0.5             # fraction of long ("hopeless") requests per burst
OUTPUT_SHORT_TOKENS = 200       # short "live" output length
OUTPUT_LONG_TOKENS = 8000       # long "hopeless" output length
MAX_PROMPT_LEN = 2000           # keep input short-to-moderate
SEED = 42

SRC_TRACE = Path(
    "/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/"
    "0630_evaluation_dataset/burstgpt/qwen_lmsys_burstgpt_673.csv")
OUT_PATH = Path(
    "/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base/"
    "bimodal_burst/qwen_lmsys_bimodal_burst_600s.csv")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def main() -> int:
    for k, v in [("TRACE_SPAN_S", TRACE_SPAN_S), ("BURST_PERIOD_S", BURST_PERIOD_S),
                 ("BURST_DURATION_S", BURST_DURATION_S), ("REQS_PER_BURST", REQS_PER_BURST),
                 ("LONG_FRACTION", LONG_FRACTION), ("OUTPUT_SHORT_TOKENS", OUTPUT_SHORT_TOKENS),
                 ("OUTPUT_LONG_TOKENS", OUTPUT_LONG_TOKENS), ("MAX_PROMPT_LEN", MAX_PROMPT_LEN),
                 ("SEED", SEED)]:
        pass  # documented knobs; parse below explicitly

    span_s = float(os.environ.get("TRACE_SPAN_S", str(TRACE_SPAN_S)))
    burst_period_s = float(os.environ.get("BURST_PERIOD_S", str(BURST_PERIOD_S)))
    burst_duration_s = float(os.environ.get("BURST_DURATION_S", str(BURST_DURATION_S)))
    reqs_per_burst = _env_int("REQS_PER_BURST", REQS_PER_BURST)
    long_frac = float(os.environ.get("LONG_FRACTION", str(LONG_FRACTION)))
    out_short = _env_int("OUTPUT_SHORT_TOKENS", OUTPUT_SHORT_TOKENS)
    out_long = _env_int("OUTPUT_LONG_TOKENS", OUTPUT_LONG_TOKENS)
    max_prompt_len = _env_int("MAX_PROMPT_LEN", MAX_PROMPT_LEN)
    seed = _env_int("SEED", SEED)

    rng = random.Random(seed)

    # 1. load real prompts (short-to-moderate input length)
    rows = list(csv.DictReader(open(SRC_TRACE, encoding="utf-8")))
    pool = [r for r in rows if int(r["prompt_len"]) <= max_prompt_len and int(r["prompt_len"]) >= 8]
    if len(pool) < reqs_per_burst * 10:
        print(f"WARNING: prompt pool {len(pool)} is small for {reqs_per_burst}/burst")
    print(f"prompt pool (prompt_len<= {max_prompt_len}): {len(pool)} of {len(rows)}")

    # 2. assign output lengths (bimodal) and bursty timestamps
    n_bursts = int(span_s / burst_period_s)
    total = n_bursts * reqs_per_burst
    n_long_per_burst = int(reqs_per_burst * long_frac)
    print(f"bursts={n_bursts}  total_reqs={total}  long/burst={n_long_per_burst} "
          f"(out_long={out_long} tokens) short/burst={reqs_per_burst - n_long_per_burst} "
          f"(out_short={out_short} tokens)")

    out_rows: list[dict] = []
    used_prompts = 0
    for b in range(n_bursts):
        burst_start_s = b * burst_period_s
        for i in range(reqs_per_burst):
            is_long = i < n_long_per_burst
            src = pool[used_prompts % len(pool)]
            used_prompts += 1
            # spread requests inside the burst window, then a small jitter
            ts_ms = int((burst_start_s + rng.uniform(0, burst_duration_s)) * 1000)
            out_rows.append({
                "req_id": f"bb{b:03d}_{i:03d}",
                "prompt": src["prompt"],
                "prompt_len": src["prompt_len"],
                "output_tokens": str(out_long if is_long else out_short),
                "temperature": src.get("temperature", "0.8"),
                "top_p": src.get("top_p", "0.7"),
                "top_k": src.get("top_k", "50"),
                "repetition_penalty": src.get("repetition_penalty", "1.0"),
                "timestamp": str(ts_ms),
            })
    # sort by timestamp so the benchmark replays in arrival order
    out_rows.sort(key=lambda r: int(r["timestamp"]))

    out_path = Path(os.environ.get("OUT_PATH", str(OUT_PATH)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn = list(out_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(out_rows)

    n_long = sum(1 for r in out_rows if int(r["output_tokens"]) == out_long)
    print(f"wrote {out_path}")
    print(f"  total={len(out_rows)}  long={n_long}  short={len(out_rows) - n_long}  "
          f"total_out_tokens={sum(int(r['output_tokens']) for r in out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
