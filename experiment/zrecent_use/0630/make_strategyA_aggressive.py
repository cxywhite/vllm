#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Further shorten a StrategyA trace while preserving its longest requests."
    )
    parser.add_argument("input", type=Path, help="Existing StrategyA CSV")
    parser.add_argument("output", type=Path, help="Derived CSV outside the source directory")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="Scale applied to non-preserved output lengths (default: 0.75)",
    )
    parser.add_argument(
        "--keep-longest",
        type=int,
        default=50,
        help="Number of longest requests to preserve unchanged (default: 50)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing derived CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be in (0, 1]")
    if args.keep_longest < 0:
        raise ValueError("--keep-longest must be non-negative")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different; preserve StrategyA")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    csv.field_size_limit(sys.maxsize)
    with args.input.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        if "output_tokens" not in reader.fieldnames:
            raise ValueError("input CSV is missing output_tokens")
        rows = list(reader)

    if not rows:
        raise ValueError("input CSV has no requests")
    if args.keep_longest > len(rows):
        raise ValueError("--keep-longest exceeds the request count")

    lengths = [int(row["output_tokens"]) for row in rows]
    preserved = set(
        sorted(range(len(rows)), key=lambda index: (-lengths[index], index))[
            : args.keep_longest
        ]
    )
    original_total = sum(lengths)
    changed = 0
    for index, row in enumerate(rows):
        if index in preserved:
            continue
        derived_length = max(1, round(lengths[index] * args.scale))
        row["output_tokens"] = str(derived_length)
        changed += derived_length != lengths[index]

    derived_total = sum(int(row["output_tokens"]) for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reduction = 100 * (original_total - derived_total) / original_total
    print(f"requests: {len(rows)}")
    print(f"preserved longest requests: {len(preserved)}")
    print(f"changed requests: {changed}")
    print(f"output tokens: {original_total} -> {derived_total} ({reduction:.1f}% reduction)")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()