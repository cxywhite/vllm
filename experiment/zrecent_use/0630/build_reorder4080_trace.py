#!/usr/bin/env python3
"""Rebuild the tmp_dataset4 intent correctly: keep ALL 4080 requests, only reorder timestamps.

No filtering, no deduplication, no modification of prompts/output labels/predicted values.
Only the arrival order (timestamp) is changed so that long requests arrive earlier
and more evenly, reducing the artificial drain tail.

Placement strategy (same idea as build_lowmae_longsafe_trace.py, but on the full 4080):
  - normalize timestamps to a strict 600s window, monotonically increasing
  - >16K requests are placed evenly within the first 480s
  - 8K-16K requests are placed evenly within the first 540s
  - remaining requests are shuffled (seed 42) into the leftover slots
"""

import argparse
import csv
import random
import sys
from pathlib import Path

DEFAULT_SOURCE = Path(
    "derived_datasets/tmp_dataset3/qwen_lmsys_qwentrace_4080_600s_strategyA.csv"
)
DEFAULT_OUTPUT = Path(
    "derived_datasets/tmp_dataset4_reorder4080/"
    "qwen_lmsys_qwentrace_4080_reorder_600s_strategyA.csv"
)

VERYLONG_DEADLINE_S = 480.0
LONG_DEADLINE_S = 540.0
DURATION_S = 600.0
EXPECTED_COUNT = 4080


def band_for(tokens: int) -> str | None:
    if 1 <= tokens <= 4096:
        return "short"
    if 4097 <= tokens <= 8192:
        return "medium"
    if 8193 <= tokens <= 16384:
        return "long"
    if 16385 <= tokens <= 30000:
        return "verylong"
    if 30001 <= tokens <= 32768:
        return "extreme"
    return None


def build_slots_from_source(source_rows: list[dict], duration_ms: int) -> list[int]:
    """Sample all source timestamps and normalize onto a strict target duration."""
    source_ts = sorted(int(r["timestamp"]) for r in source_rows)
    source_start = source_ts[0]
    source_span = source_ts[-1] - source_start
    if source_span <= 0:
        raise ValueError("source timestamp span must be positive")
    n = len(source_ts)
    sampled = [
        source_ts[round(index * (n - 1) / (n - 1))]
        for index in range(n)
    ]
    slots = [
        source_start + round((ts - source_start) * duration_ms / source_span)
        for ts in sampled
    ]
    for i in range(1, len(slots)):
        slots[i] = max(slots[i], slots[i - 1] + 1)
    if slots[-1] - source_start > duration_ms:
        raise ValueError("timestamp normalization produced a slot beyond the target duration")
    return slots


def evenly_spaced_positions(available: list[int], count: int) -> list[int]:
    if count == 0:
        return []
    if count == 1:
        return [available[len(available) // 2]]
    return [
        available[round(index * (len(available) - 1) / (count - 1))]
        for index in range(count)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-seconds", type=float, default=DURATION_S)
    parser.add_argument(
        "--verylong-deadline-seconds", type=float, default=VERYLONG_DEADLINE_S
    )
    parser.add_argument("--long-deadline-seconds", type=float, default=LONG_DEADLINE_S)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output} (use --overwrite)")
    if not 0 < args.verylong_deadline_seconds < args.long_deadline_seconds < args.duration_seconds:
        raise ValueError("deadlines must satisfy 0 < verylong < long < duration")

    csv.field_size_limit(sys.maxsize)
    with args.source.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    total = len(rows)
    if total != EXPECTED_COUNT:
        raise ValueError(f"expected {EXPECTED_COUNT} rows, got {total}")
    for field in ("req_id", "prompt", "prompt_len", "output_tokens", "timestamp"):
        if field not in fieldnames:
            raise ValueError(f"source missing field: {field}")

    for row in rows:
        if band_for(int(float(row["output_tokens"]))) is None:
            raise ValueError(
                f"req_id={row['req_id']} has out-of-range output_tokens={row['output_tokens']}"
            )

    duration_ms = round(args.duration_seconds * 1000)
    slots = build_slots_from_source(rows, duration_ms)
    start = slots[0]

    verylong_rows = [
        r for r in rows
        if band_for(int(float(r["output_tokens"]))) in ("verylong", "extreme")
    ]
    long_rows = [
        r for r in rows
        if band_for(int(float(r["output_tokens"]))) == "long"
    ]
    rest_rows = [
        r for r in rows
        if band_for(int(float(r["output_tokens"]))) in ("short", "medium")
    ]
    if len(verylong_rows) + len(long_rows) + len(rest_rows) != total:
        raise RuntimeError("band partition lost rows")

    assignments: dict[int, dict] = {}

    eligible = [
        i for i, t in enumerate(slots)
        if (t - start) / 1000 <= args.verylong_deadline_seconds
    ]
    for pos, row in zip(
        evenly_spaced_positions(eligible, len(verylong_rows)), verylong_rows
    ):
        assignments[pos] = row

    eligible = [
        i for i, t in enumerate(slots)
        if i not in assignments and (t - start) / 1000 <= args.long_deadline_seconds
    ]
    for pos, row in zip(evenly_spaced_positions(eligible, len(long_rows)), long_rows):
        assignments[pos] = row

    random.Random(args.seed).shuffle(rest_rows)
    remaining_positions = [i for i in range(total) if i not in assignments]
    if len(rest_rows) != len(remaining_positions):
        raise RuntimeError(
            f"slot mismatch: {len(rest_rows)} rows vs {len(remaining_positions)} positions"
        )
    assignments.update(zip(remaining_positions, rest_rows))

    output_rows = []
    for i, t in enumerate(slots):
        row = assignments[i]
        out = {field: row[field] for field in fieldnames if field != "timestamp"}
        out["timestamp"] = str(t)
        output_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    out_ts = [int(r["timestamp"]) for r in output_rows]
    out_ids = [r["req_id"] for r in output_rows]
    out_tokens = [int(float(r["output_tokens"])) for r in output_rows]
    n_gt16k = sum(1 for o in out_tokens if o > 16384)
    n_8_16k = sum(1 for o in out_tokens if 8193 <= o <= 16384)
    max_gt16k_ts = max(
        (int(r["timestamp"]) for r in output_rows if int(float(r["output_tokens"])) > 16384),
        default=None,
    )
    max_8_16k_ts = max(
        (
            int(r["timestamp"])
            for r in output_rows
            if 8193 <= int(float(r["output_tokens"])) <= 16384
        ),
        default=None,
    )
    monotonic = all(out_ts[i] < out_ts[i + 1] for i in range(total - 1))
    span_s = (out_ts[-1] - out_ts[0]) / 1000
    last_60s_gt8k = sum(
        1
        for r in output_rows
        if int(float(r["output_tokens"])) > 8192
        and (int(r["timestamp"]) - out_ts[0]) / 1000 > 540
    )

    print(f"output: {args.output}")
    print(f"requests: {len(output_rows)}")
    print(f"unique req_id: {len(set(out_ids))}")
    print(f"total output tokens: {sum(out_tokens)}")
    print(f"timestamp strictly monotonic: {monotonic}")
    print(f"span_s: {span_s:.3f}")
    print(f">16K requests: {n_gt16k}, latest ts {max_gt16k_ts} ms")
    print(f"8-16K requests: {n_8_16k}, latest ts {max_8_16k_ts} ms")
    print(f"last 60s (>8K) count: {last_60s_gt8k}")


if __name__ == "__main__":
    main()
