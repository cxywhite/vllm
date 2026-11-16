#!/usr/bin/env python3
"""
Further dampen output lengths in dense time windows of a StrategyA trace.
Preserves longest-K requests unchanged.
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dampen output lengths in dense trace windows."
    )
    parser.add_argument("input", type=Path, help="Existing StrategyA CSV")
    parser.add_argument("output", type=Path, help="Derived CSV")
    parser.add_argument(
        "--dense-cap",
        type=int,
        default=2048,
        help="Max output_tokens in dense windows (default: 2048)",
    )
    parser.add_argument(
        "--global-cap",
        type=int,
        default=0,
        help="Global max output_tokens cap, 0=disabled (default: 0)",
    )
    parser.add_argument(
        "--dense-scale",
        type=float,
        default=1.0,
        help="Scale applied to non-capped outputs in dense windows (default: 1.0=no extra scaling)",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=30,
        help="Sliding window size in seconds (default: 30)",
    )
    parser.add_argument(
        "--top-dense",
        type=int,
        default=6,
        help="Number of densest windows to dampen (default: 6)",
    )
    parser.add_argument(
        "--keep-longest",
        type=int,
        default=50,
        help="Preserve top-K longest requests unchanged (default: 50)",
    )
    parser.add_argument(
        "--stretch-dense",
        type=float,
        default=1.0,
        help="Stretch inter-arrival gaps inside dense windows by this factor (default: 1.0=no stretch)",
    )
    parser.add_argument(
        "--offpeak-prompt-min",
        type=int,
        default=0,
        help="Move requests with prompt_len >= this value out of dense windows (0=disabled)",
    )
    parser.add_argument(
        "--preserve-above",
        type=int,
        default=0,
        help="Keep output_tokens >= this value unchanged and move them out of dense windows (0=disabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dense_cap <= 0:
        raise ValueError("--dense-cap must be positive")
    if args.stretch_dense < 1.0:
        raise ValueError("--stretch-dense must be >= 1.0")
    if args.global_cap < 0:
        raise ValueError("--global-cap must be non-negative")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must differ")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    csv.field_size_limit(sys.maxsize)
    with args.input.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        if "output_tokens" not in reader.fieldnames:
            raise ValueError("missing output_tokens column")
        rows = list(reader)

    if not rows:
        raise ValueError("empty input")

    # identify longest-K to preserve
    lengths = [int(row["output_tokens"]) for row in rows]
    preserved = set(
        sorted(range(len(rows)), key=lambda i: (-lengths[i], i))[
            : min(args.keep_longest, len(rows))
        ]
    )

    # normalize timestamps
    ts_list = [int(row["timestamp"]) for row in rows]
    t0 = ts_list[0]
    norm_t = [(t - t0) / 1000 for t in ts_list]
    trace_end = max(norm_t)

    # sliding window density
    w = args.window_seconds
    step = max(1, w // 3)
    win_reqs: dict[tuple[float, float], list[int]] = {}
    for start in range(0, int(trace_end) + 1, step):
        end = start + w
        idxs = [i for i, t in enumerate(norm_t) if start <= t < end]
        if idxs:
            win_reqs[(start, end)] = idxs

    # top-N densest windows
    top_windows = sorted(win_reqs.items(), key=lambda x: -len(x[1]))[
        : args.top_dense
    ]

    # merge overlapping windows into intervals
    merged_intervals: list[tuple[float, float]] = []
    for (s, e), _ in sorted(top_windows, key=lambda x: x[0][0]):
        if merged_intervals and s <= merged_intervals[-1][1] + w * 0.5:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], e))
        else:
            merged_intervals.append((s, e))

    print(f"input: {len(rows)} requests, trace {trace_end:.0f}s")
    print(f"top-{args.top_dense} {w}s windows merged into {len(merged_intervals)} interval(s):")
    for s, e in merged_intervals:
        cnt = sum(1 for t in norm_t if s <= t < e)
        print(f"  [{s:.0f}s - {e:.0f}s] {cnt} requests")

    dense_idx: set[int] = set()
    for s, e in merged_intervals:
        for i, t in enumerate(norm_t):
            if s <= t < e:
                dense_idx.add(i)

    print(f"requests in dense intervals: {len(dense_idx)}")

    # Optional: move long-prompt requests out of dense windows (big KV payloads
    # stall the P2P NCCL transfer path when they collide with dense bursts).
    offpeak_idx: set[int] = set()
    if args.offpeak_prompt_min > 0 or args.preserve_above > 0:
        if args.offpeak_prompt_min > 0 and "prompt_len" not in reader.fieldnames:
            raise ValueError("missing prompt_len column for --offpeak-prompt-min")
        moved_by_interval: dict[tuple[float, float], list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            hit = False
            if args.offpeak_prompt_min > 0:
                try:
                    pl = int(row["prompt_len"])
                except ValueError:
                    pl = 0
                hit = pl >= args.offpeak_prompt_min
            if not hit and args.preserve_above > 0:
                hit = int(row["output_tokens"]) >= args.preserve_above
            if not hit:
                continue
            t = norm_t[i]
            for s, e in merged_intervals:
                if s <= t < e:
                    moved_by_interval[(s, e)].append(i)
                    offpeak_idx.add(i)
                    break
        if offpeak_idx:
            for (s, e), idxs in moved_by_interval.items():
                idxs.sort(key=lambda i: ts_list[i])
                base = int(e + 1)  # seconds after the dense interval end
                for rank, i in enumerate(idxs):
                    rows[i]["timestamp"] = str(t0 + (base * 1000) + rank * 10)
            print(f"offpeak: {len(offpeak_idx)} requests moved out of dense windows (long prompt / preserved long output)")

    # requests that remain inside dense windows are subject to length capping
    dense_cap_idx = dense_idx - offpeak_idx

    original_total = sum(lengths)
    changed = 0
    capped = 0

    for i, row in enumerate(rows):
        if i in preserved:
            continue
        orig = lengths[i]
        new = orig
        if i in dense_cap_idx:
            if args.dense_scale < 1.0:
                new = max(1, round(new * args.dense_scale))
            new = min(new, args.dense_cap)
        if args.global_cap > 0:
            new = min(new, args.global_cap)
        new = max(1, new)
        row["output_tokens"] = str(new)
        if new != orig:
            changed += 1
            if new < orig:
                capped += 1

    derived_total = sum(int(row["output_tokens"]) for row in rows)
    reduction = 100 * (original_total - derived_total) / max(original_total, 1)

    # Optional: stretch inter-arrival gaps inside dense intervals (keeps
    # request count and length distribution, lowers instantaneous arrival rate).
    if args.stretch_dense > 1.0:
        stretch = args.stretch_dense
        intervals_sorted = sorted(merged_intervals)

        def in_interval(t: float) -> bool:
            return any(s <= t < e for s, e in intervals_sorted)

        raw_ts = [int(row["timestamp"]) for row in rows]
        order = sorted(range(len(rows)), key=lambda i: raw_ts[i])
        new_ts = dict.fromkeys(order)
        shift = 0.0
        prev_t = None
        prev_in = False
        stretched_gaps = 0
        for pos, i in enumerate(order):
            t = (raw_ts[i] - t0) / 1000
            if prev_t is None:
                pass
            else:
                cur_in = in_interval(t)
                gap = t - prev_t
                if prev_in and cur_in and gap > 0:
                    shift += gap * (stretch - 1.0)
                    stretched_gaps += 1
                prev_in = cur_in
            new_ts[i] = raw_ts[i] + round(shift * 1000)
            prev_t = t
        for i in order:
            rows[i]["timestamp"] = str(new_ts[i])
        new_span = (max(new_ts.values()) - min(new_ts.values())) / 1000
        print(f"stretch-dense={stretch}: {stretched_gaps} gaps stretched, trace span {trace_end:.0f}s -> {new_span:.0f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"changed: {changed} (capped down: {capped})")
    print(f"output tokens: {original_total} -> {derived_total} ({reduction:.1f}% reduction)")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
