import argparse
from pathlib import Path

from trace_utils import (
    add_trace_type_argument,
    format_timestamp_label,
    load_trace_records,
    parse_timestamp_bound,
    write_trace_records,
)


def build_output_path(
    input_path: Path,
    output_dir: Path,
    trace_type: str,
    start_timestamp_ms: float,
    end_timestamp_ms: float,
) -> Path:
    start_label = format_timestamp_label(trace_type, start_timestamp_ms)
    end_label = format_timestamp_label(trace_type, end_timestamp_ms)
    return output_dir / f"{input_path.stem}_ts{start_label}_{end_label}{input_path.suffix}"


def filter_trace_by_timestamp(
    loaded,
    output_path: Path,
    trace_type: str,
    start_timestamp_ms: float,
    end_timestamp_ms: float,
) -> None:
    kept_records = [
        record
        for record in loaded.records
        if record.timestamp_ms is not None and start_timestamp_ms <= record.timestamp_ms <= end_timestamp_ms
    ]

    write_trace_records(output_path, trace_type, kept_records, loaded.fieldnames)

    print(f"Done. total={loaded.total_rows}, kept={len(kept_records)}, removed={loaded.total_rows - len(kept_records)}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JSONL trace by keeping rows whose timestamp is within the given range"
    )
    add_trace_type_argument(parser)
    parser.add_argument(
        "--input",
        default="/root/.cache/huggingface/hub/datasets/wt_predictor/filter_trace_8192/qwen_trace/qwen_thinking_blksz_16_th8192.jsonl",
        help="Input trace file path (.jsonl or .csv depending on trace type)",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/tmp_dataset",
        help="Directory to save filtered trace",
    )
    parser.add_argument(
        "--start-timestamp",
        default=2000000,
        help="Keep rows whose timestamp is greater than or equal to this value; use ISO-8601 for Azure and numeric values for mooncake/qwen/BurstGPT",
    )
    parser.add_argument(
        "--end-timestamp",
        default=5000000,
        help="Keep rows whose timestamp is less than or equal to this value; use ISO-8601 for Azure and numeric values for mooncake/qwen/BurstGPT",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    loaded = load_trace_records(
        input_path,
        trace_type=args.trace_type,
        require_timestamp=True,
        skip_invalid=True,
    )

    timestamps = [record.timestamp_ms for record in loaded.records if record.timestamp_ms is not None]
    start_timestamp_ms = parse_timestamp_bound(args.trace_type, args.start_timestamp) if args.start_timestamp is not None else min(timestamps)
    end_timestamp_ms = parse_timestamp_bound(args.trace_type, args.end_timestamp) if args.end_timestamp is not None else max(timestamps)

    if start_timestamp_ms > end_timestamp_ms:
        raise ValueError("start_timestamp cannot be greater than end_timestamp")

    output_path = build_output_path(
        input_path,
        output_dir,
        args.trace_type,
        start_timestamp_ms,
        end_timestamp_ms,
    )
    filter_trace_by_timestamp(
        loaded,
        output_path,
        args.trace_type,
        start_timestamp_ms,
        end_timestamp_ms,
    )


if __name__ == "__main__":
    main()