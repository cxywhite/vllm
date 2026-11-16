#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET = Path(
    "/root/predict-schedule/experiment/disagg_serving_suite/"
    "wt_experiment/dataset/dataset/qwen-lmsys-chat-updated.csv"
)
DEFAULT_PREDICTIONS = Path(
    "/root/predict-schedule/experiment/wt_experiment/dataset/"
    "new_select_dataset_3/tmp/"
    "qwen_lmsys_predict_request_decode0_20260501_174106_"
    "selected_expected_len_top2025_plus496_joint_low_error_sorted.csv"
)
DEFAULT_TIMESTAMP_SOURCE = Path(
    "derived_datasets/tmp_dataset3/"
    "qwen_lmsys_qwentrace_4080_600s_strategyA.csv"
)
DEFAULT_OUTPUT = Path(
    "derived_datasets/tmp_dataset4_lowmae_longsafe/"
    "qwen_lmsys_lowmae_longsafe_seed42_1000_600s_strategyA.csv"
)

BAND_LIMITS = {
    "short": (1, 4096),
    "medium": (4097, 8192),
    "long": (8193, 16384),
    "verylong": (16385, 30000),
    "extreme": (30001, 32768),
}
DEFAULT_QUOTAS = {
    "short": 533,
    "medium": 120,
    "long": 150,
    "verylong": 150,
    "extreme": 47,
}
OUTPUT_FIELDS = [
    "req_id",
    "prompt",
    "prompt_len",
    "output_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "timestamp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Qwen LMSYS trace from real low-error requests, then place "
            "long requests early enough to avoid an artificial drain tail."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--timestamp-source", type=Path, default=DEFAULT_TIMESTAMP_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--verylong-deadline-seconds", type=float, default=480.0)
    parser.add_argument("--long-deadline-seconds", type=float, default=540.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return reader.fieldnames, list(reader)


def band_for(output_tokens: int) -> str | None:
    for name, (lower, upper) in BAND_LIMITS.items():
        if lower <= output_tokens <= upper:
            return name
    return None


def evenly_spaced_positions(available: list[int], count: int) -> list[int]:
    if count > len(available):
        raise ValueError(f"need {count} positions, only {len(available)} available")
    if count == 0:
        return []
    if count == 1:
        return [available[len(available) // 2]]
    indices = [round(index * (len(available) - 1) / (count - 1)) for index in range(count)]
    return [available[index] for index in indices]


def build_timestamp_slots(
    source_rows: list[dict[str, str]], count: int, duration_seconds: float
) -> list[int]:
    source_timestamps = sorted(int(row["timestamp"]) for row in source_rows)
    if len(source_timestamps) < count:
        raise ValueError("timestamp source has fewer rows than the requested trace")
    source_start = source_timestamps[0]
    source_span = source_timestamps[-1] - source_start
    if source_span <= 0:
        raise ValueError("timestamp source must have a positive duration")

    sampled = [
        source_timestamps[round(index * (len(source_timestamps) - 1) / (count - 1))]
        for index in range(count)
    ]
    target_span_ms = round(duration_seconds * 1000)
    slots = [
        source_start + round((timestamp - source_start) * target_span_ms / source_span)
        for timestamp in sampled
    ]
    for index in range(1, len(slots)):
        slots[index] = max(slots[index], slots[index - 1] + 1)
    if slots[-1] - source_start > target_span_ms:
        raise ValueError("timestamp normalization produced a slot beyond the target duration")
    return slots


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if not 0 < args.verylong_deadline_seconds < args.long_deadline_seconds < args.duration_seconds:
        raise ValueError("deadlines must satisfy 0 < verylong < long < duration")

    csv.field_size_limit(sys.maxsize)
    dataset_fields, dataset_rows = read_csv(args.dataset)
    prediction_fields, prediction_rows = read_csv(args.predictions)
    _, timestamp_rows = read_csv(args.timestamp_source)
    required_dataset_fields = set(OUTPUT_FIELDS) - {"timestamp"}
    missing = required_dataset_fields.difference(dataset_fields)
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")
    required_prediction_fields = {
        "dataset_req_id",
        "qwen_dataset_output_tokens",
        "qwen_predicted_expected_output_len",
    }
    missing = required_prediction_fields.difference(prediction_fields)
    if missing:
        raise ValueError(f"prediction CSV is missing fields: {sorted(missing)}")

    dataset_by_id = {str(row["req_id"]).strip(): row for row in dataset_rows}
    pools: dict[str, list[tuple[float, float, int, dict[str, str]]]] = defaultdict(list)
    prediction_by_id: dict[str, dict[str, str]] = {}
    for prediction in prediction_rows:
        request_id = str(prediction["dataset_req_id"]).strip()
        dataset_row = dataset_by_id.get(request_id)
        if dataset_row is None:
            continue
        actual = int(float(dataset_row["output_tokens"]))
        recorded_actual = int(float(prediction["qwen_dataset_output_tokens"]))
        if actual != recorded_actual:
            raise ValueError(f"output length mismatch for req_id={request_id}")
        predicted = float(prediction["qwen_predicted_expected_output_len"])
        band = band_for(actual)
        if band is None:
            continue
        log_error = abs(math.log1p(predicted) - math.log1p(actual))
        absolute_error = abs(predicted - actual)
        pools[band].append((log_error, absolute_error, int(request_id), dataset_row))
        prediction_by_id[request_id] = prediction

    selected_by_band: dict[str, list[dict[str, str]]] = {}
    for band, quota in DEFAULT_QUOTAS.items():
        candidates = sorted(pools[band], key=lambda item: (item[0], item[1], item[2]))
        if len(candidates) < quota:
            raise ValueError(f"band {band} has {len(candidates)} candidates, needs {quota}")
        selected_by_band[band] = [item[3] for item in candidates[:quota]]

    total_count = sum(DEFAULT_QUOTAS.values())
    slots = build_timestamp_slots(timestamp_rows, total_count, args.duration_seconds)
    start_timestamp = slots[0]
    assignments: dict[int, dict[str, str]] = {}

    verylong_rows = selected_by_band["verylong"] + selected_by_band["extreme"]
    eligible = [
        index
        for index, timestamp in enumerate(slots)
        if (timestamp - start_timestamp) / 1000 <= args.verylong_deadline_seconds
    ]
    for position, row in zip(evenly_spaced_positions(eligible, len(verylong_rows)), verylong_rows):
        assignments[position] = row

    eligible = [
        index
        for index, timestamp in enumerate(slots)
        if index not in assignments
        and (timestamp - start_timestamp) / 1000 <= args.long_deadline_seconds
    ]
    for position, row in zip(
        evenly_spaced_positions(eligible, len(selected_by_band["long"])),
        selected_by_band["long"],
    ):
        assignments[position] = row

    remaining_rows = selected_by_band["short"] + selected_by_band["medium"]
    random.Random(args.seed).shuffle(remaining_rows)
    remaining_positions = [index for index in range(total_count) if index not in assignments]
    if len(remaining_rows) != len(remaining_positions):
        raise RuntimeError("request and timestamp slot counts differ")
    assignments.update(zip(remaining_positions, remaining_rows))

    output_rows: list[dict[str, str]] = []
    selected_prediction_rows: list[dict[str, str]] = []
    for index, timestamp in enumerate(slots):
        source_row = assignments[index]
        output_row = {field: source_row[field] for field in OUTPUT_FIELDS if field != "timestamp"}
        output_row["timestamp"] = str(timestamp)
        output_rows.append(output_row)
        selected_prediction_rows.append(prediction_by_id[str(source_row["req_id"]).strip()])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    sidecar = args.output.with_name(f"{args.output.stem}_predictions.csv")
    with sidecar.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(selected_prediction_rows)

    actual_total = sum(int(row["output_tokens"]) for row in output_rows)
    predicted_total = sum(
        float(row["qwen_predicted_expected_output_len"])
        for row in selected_prediction_rows
    )
    prediction_ratios = sorted(
        max(
            int(dataset_by_id[str(row["dataset_req_id"]).strip()]["output_tokens"]),
            float(row["qwen_predicted_expected_output_len"]),
        )
        / max(
            1,
            min(
                int(dataset_by_id[str(row["dataset_req_id"]).strip()]["output_tokens"]),
                float(row["qwen_predicted_expected_output_len"]),
            ),
        )
        for row in selected_prediction_rows
    )
    summary = {
        "dataset": str(args.output),
        "prediction_sidecar": str(sidecar),
        "seed": args.seed,
        "requests": len(output_rows),
        "duration_seconds": (int(output_rows[-1]["timestamp"]) - int(output_rows[0]["timestamp"])) / 1000,
        "band_quotas": DEFAULT_QUOTAS,
        "actual_output_tokens": actual_total,
        "predicted_output_tokens": predicted_total,
        "prediction_total_bias_percent": 100 * (predicted_total / actual_total - 1),
        "prediction_ratio_median": prediction_ratios[len(prediction_ratios) // 2],
        "prediction_ratio_p90": prediction_ratios[math.ceil(0.9 * len(prediction_ratios)) - 1],
        "max_output_tokens": max(int(row["output_tokens"]) for row in output_rows),
        "verylong_latest_arrival_seconds": max(
            (int(row["timestamp"]) - start_timestamp) / 1000
            for row in output_rows
            if int(row["output_tokens"]) > 16384
        ),
        "long_latest_arrival_seconds": max(
            (int(row["timestamp"]) - start_timestamp) / 1000
            for row in output_rows
            if 8192 < int(row["output_tokens"]) <= 16384
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()