#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


DEFAULT_SOURCE = Path(
    "derived_datasets/tmp_dataset3/qwen_lmsys_qwentrace_4080_600s_strategyA.csv"
)
DEFAULT_PREDICTION_GLOB = (
    "retry_test/tmp_dataset4_reorder4080_rr_ours/"
    "*scheme4_gwf/log/predict_request_decode*.csv"
)
DEFAULT_OUTPUT = Path(
    "derived_datasets/tmp_dataset5_predictor_friendly4080/"
    "qwen_lmsys_predictor_friendly_4080_600s_strategyA.csv"
)
EXPECTED_REQUESTS = 4080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep all requests and the exact timestamp multiset, while arranging "
            "long requests in four-request groups and smoothing predicted load."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prediction-glob", default=DEFAULT_PREDICTION_GLOB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extreme-deadline-seconds", type=float, default=180.0)
    parser.add_argument("--long-deadline-seconds", type=float, default=430.0)
    parser.add_argument("--density-window-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def read_predictions(pattern: str) -> dict[str, float]:
    predictions: dict[str, float] = {}
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no prediction CSVs matched: {pattern}")
    for path in paths:
        _, rows = read_rows(Path(path))
        for row in rows:
            request_id = row.get("dataset_req_id", "").strip()
            value = row.get("predict_output_mean", "").strip()
            if request_id and value:
                predictions[request_id] = float(value)
    return predictions


def reserve_grouped_positions(
    available: set[int], eligible: list[int], count: int, group_size: int = 4
) -> list[int]:
    if count == 0:
        return []
    group_count = (count + group_size - 1) // group_size
    targets = [
        eligible[round(group * (len(eligible) - 1) / max(group_count - 1, 1))]
        for group in range(group_count)
    ]
    selected: list[int] = []
    for target in targets:
        group: list[int] = []
        for position in range(target, eligible[-1] + 1):
            if position in available:
                group.append(position)
                if len(group) == min(group_size, count - len(selected)):
                    break
        if len(group) < min(group_size, count - len(selected)):
            for position in reversed(eligible):
                if position in available and position not in group:
                    group.append(position)
                    if len(group) == min(group_size, count - len(selected)):
                        break
        if not group:
            raise RuntimeError("unable to reserve grouped positions")
        for position in group:
            available.remove(position)
        selected.extend(sorted(group))
    return selected[:count]


def local_density(timestamps: list[int], position: int, window_ms: int) -> int:
    timestamp = timestamps[position]
    return sum(abs(other - timestamp) <= window_ms for other in timestamps)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output} (use --overwrite)")
    if not 0 < args.extreme_deadline_seconds < args.long_deadline_seconds:
        raise ValueError("deadlines must satisfy 0 < extreme < long")

    csv.field_size_limit(sys.maxsize)
    fieldnames, rows = read_rows(args.source)
    if len(rows) != EXPECTED_REQUESTS:
        raise ValueError(f"expected {EXPECTED_REQUESTS} rows, got {len(rows)}")
    predictions = read_predictions(args.prediction_glob)

    timestamps = sorted(int(row["timestamp"]) for row in rows)
    start_timestamp = timestamps[0]
    available = set(range(len(timestamps)))
    assignments: dict[int, dict[str, str]] = {}

    extreme = [row for row in rows if int(row["output_tokens"]) > 30000]
    long = [row for row in rows if 8192 < int(row["output_tokens"]) <= 30000]
    remaining = [row for row in rows if int(row["output_tokens"]) <= 8192]

    extreme_eligible = [
        position
        for position, timestamp in enumerate(timestamps)
        if (timestamp - start_timestamp) / 1000 <= args.extreme_deadline_seconds
    ]
    extreme_positions = reserve_grouped_positions(
        available, extreme_eligible, len(extreme)
    )
    extreme.sort(
        key=lambda row: (
            -predictions.get(row["req_id"], 16384.0),
            -int(row["output_tokens"]),
            int(row["req_id"]),
        )
    )
    assignments.update(zip(extreme_positions, extreme))

    long_eligible = [
        position
        for position, timestamp in enumerate(timestamps)
        if position in available
        and (timestamp - start_timestamp) / 1000 <= args.long_deadline_seconds
    ]
    long_positions = reserve_grouped_positions(available, long_eligible, len(long))
    long.sort(
        key=lambda row: (
            -predictions.get(row["req_id"], 16384.0),
            -int(row["output_tokens"]),
            int(row["req_id"]),
        )
    )
    assignments.update(zip(long_positions, long))

    remaining_positions = sorted(available)
    window_ms = round(args.density_window_seconds * 1000)
    remaining_positions.sort(
        key=lambda position: (
            local_density(timestamps, position, window_ms),
            timestamps[position],
        )
    )
    remaining.sort(
        key=lambda row: (
            -predictions.get(row["req_id"], 16384.0),
            -int(row["output_tokens"]),
            int(row["req_id"]),
        )
    )
    if len(remaining_positions) != len(remaining):
        raise RuntimeError("remaining request and timestamp counts differ")
    assignments.update(zip(remaining_positions, remaining))

    output_rows = []
    for position, timestamp in enumerate(timestamps):
        source_row = assignments[position]
        output_row = dict(source_row)
        output_row["timestamp"] = str(timestamp)
        output_rows.append(output_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "requests": len(output_rows),
        "unique_req_ids": len({row["req_id"] for row in output_rows}),
        "total_output_tokens": sum(int(row["output_tokens"]) for row in output_rows),
        "timestamp_multiset_preserved": sorted(
            int(row["timestamp"]) for row in rows
        ) == sorted(int(row["timestamp"]) for row in output_rows),
        "timestamp_span_seconds": (timestamps[-1] - timestamps[0]) / 1000,
        "predictions_available": len(predictions),
        "extreme_requests": len(extreme),
        "long_requests": len(long),
        "extreme_latest_seconds": max(
            (int(row["timestamp"]) - start_timestamp) / 1000
            for row in output_rows
            if int(row["output_tokens"]) > 30000
        ),
        "long_latest_seconds": max(
            (int(row["timestamp"]) - start_timestamp) / 1000
            for row in output_rows
            if 8192 < int(row["output_tokens"]) <= 30000
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()