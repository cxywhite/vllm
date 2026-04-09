import argparse
from pathlib import Path

from trace_utils import (
    add_trace_type_argument,
    compute_average_qps_from_records,
    load_trace_records,
    scale_trace_records,
    write_trace_records,
)


def format_qps_for_filename(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def build_output_path(input_path: Path, output_dir: Path, target_avg_qps: float) -> Path:
    target_label = format_qps_for_filename(target_avg_qps)
    return output_dir / f"{input_path.stem}_avg{target_label}{input_path.suffix}"


def scale_trace_to_target_qps(
    input_path: Path,
    output_path: Path,
    trace_type: str,
    target_avg_qps: float,
    current_avg_qps: float | None,
) -> None:
    if target_avg_qps <= 0:
        raise ValueError("target_avg_qps must be positive")

    loaded = load_trace_records(
        input_path,
        trace_type=trace_type,
        require_timestamp=True,
        skip_invalid=False,
    )
    source_avg_qps = current_avg_qps if current_avg_qps is not None else compute_average_qps_from_records(loaded.records)

    if source_avg_qps <= 0:
        raise ValueError("current_avg_qps must be positive")

    scale_factor = source_avg_qps / target_avg_qps
    scaled_records = scale_trace_records(loaded.records, trace_type, scale_factor)
    write_trace_records(output_path, trace_type, scaled_records, loaded.fieldnames)

    scaled_avg_qps = compute_average_qps_from_records(scaled_records)
    print(
        f"{input_path.name}: current_avg_qps={source_avg_qps:.4f}, "
        f"target_avg_qps={target_avg_qps:.4f}, scale_factor={scale_factor:.6f}"
    )
    print(f"Output: {output_path}")
    print(f"Scaled trace average qps (recomputed): {scaled_avg_qps:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale trace timestamps proportionally so the average QPS approaches the target average QPS"
    )
    add_trace_type_argument(parser)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input trace file paths (.jsonl or .csv depending on trace type)",
    )
    parser.add_argument(
        "--current-avg-qps",
        nargs="*",
        type=float,
        help="Current average QPS for each input trace, in the same order as --input. If omitted, compute from file.",
    )
    parser.add_argument(
        "--target-avg-qps",
        type=float,
        default=8.0,
        help="Target average QPS after scaling",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/tmp_dataset",
        help="Directory to save scaled traces",
    )

    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input]
    output_dir = Path(args.output_dir)

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

    provided_qps = args.current_avg_qps
    if provided_qps is not None and len(provided_qps) not in (0, len(input_paths)):
        raise ValueError("--current-avg-qps must provide either 0 values or exactly one value per input trace")

    if not provided_qps:
        provided_qps = [None] * len(input_paths)

    for input_path, current_avg_qps in zip(input_paths, provided_qps):
        output_path = build_output_path(input_path, output_dir, args.target_avg_qps)
        scale_trace_to_target_qps(
            input_path=input_path,
            output_path=output_path,
            trace_type=args.trace_type,
            target_avg_qps=args.target_avg_qps,
            current_avg_qps=current_avg_qps,
        )


if __name__ == "__main__":
    main()