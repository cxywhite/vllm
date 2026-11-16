#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a derived trace CSV with capped output lengths."
    )
    parser.add_argument("input", type=Path, help="Source trace CSV")
    parser.add_argument("output", type=Path, help="Derived trace CSV")
    parser.add_argument(
        "--cap",
        type=int,
        required=True,
        help="Maximum output_tokens for requests in the selected window",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Inclusive normalized trace window start (default: 0)",
    )
    parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Exclusive normalized trace window end (default: end of trace)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing derived CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cap <= 0:
        raise ValueError("--cap must be positive")
    if args.start_seconds < 0:
        raise ValueError("--start-seconds must be non-negative")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        raise ValueError("--end-seconds must be greater than --start-seconds")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different; preserve the source trace")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    csv.field_size_limit(sys.maxsize)
    with args.input.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        required = {"timestamp", "output_tokens"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("input CSV has no requests")

    first_timestamp = int(rows[0]["timestamp"])
    original_total = 0
    derived_total = 0
    changed = 0

    for row in rows:
        original_length = int(row["output_tokens"])
        trace_seconds = (int(row["timestamp"]) - first_timestamp) / 1000
        in_window = trace_seconds >= args.start_seconds and (
            args.end_seconds is None or trace_seconds < args.end_seconds
        )
        derived_length = min(original_length, args.cap) if in_window else original_length
        row["output_tokens"] = str(derived_length)
        original_total += original_length
        derived_total += derived_length
        changed += derived_length != original_length

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reduction = 100 * (original_total - derived_total) / original_total
    window_end = "end" if args.end_seconds is None else f"{args.end_seconds:g}s"
    print(f"requests: {len(rows)}")
    print(f"changed requests: {changed}")
    print(f"window: {args.start_seconds:g}s-{window_end}, cap: {args.cap}")
    print(f"output tokens: {original_total} -> {derived_total} ({reduction:.1f}% reduction)")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()