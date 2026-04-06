import argparse
from pathlib import Path

from trace_utils import TraceRecord, add_trace_type_argument, load_trace_records, write_trace_records


def _format_float_for_filename(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _mean(values: list[int]) -> float:
    if not values:
        raise ValueError("trace has no valid length values")
    return sum(values) / len(values)


def _get_length_keys(trace_type: str) -> tuple[str, str]:
    if trace_type in ("mooncake", "qwen"):
        return "input_length", "output_length"
    if trace_type == "azure":
        return "ContextTokens", "GeneratedTokens"
    if trace_type == "burstgpt":
        return "Request tokens", "Response tokens"
    if trace_type == "test_results":
        return "prompt_len", "output_tokens"
    raise ValueError(f"unsupported trace type: {trace_type}")


def _get_block_size(trace_type: str) -> int:
    """Get block size for a trace type"""
    if trace_type == "qwen":
        return 16
    return 512  # default for mooncake, azure, burstgpt, test_results


def _recalculate_hash_ids(original_hash_ids: list, original_input_length: int, new_input_length: int, block_size: int) -> list:
    """
    Recalculate hash_ids based on new input_length.
    For prefix reuse, we need to generate new hash_ids that match the new input_length.
    """
    if not original_hash_ids:
        # No hash_ids, generate new ones based on new_input_length
        new_count = max(1, (new_input_length + block_size - 1) // block_size)
        base_id = original_hash_ids[0] if original_hash_ids else 0
        return [base_id + i for i in range(new_count)]
    
    # Calculate how many blocks we need for the new input
    new_block_count = max(1, (new_input_length + block_size - 1) // block_size)
    
    # Use the base from original hash_ids but with correct count
    base_id = original_hash_ids[0] if original_hash_ids else 0
    return [base_id + i for i in range(new_block_count)]


def _scale_length(value: int, scale_factor: float, min_length: int) -> int:
    scaled = int(round(value * scale_factor))
    return max(min_length, scaled)


def _build_output_path(input_path: Path, output_dir: Path, input_label: str, output_label: str) -> Path:
    return output_dir / f"{input_path.stem}_{input_label}_{output_label}{input_path.suffix}"


def _build_length_label(prefix: str, target_value: float | None, scale_value: float | None) -> str:
    if scale_value is not None:
        return f"{prefix}X{_format_float_for_filename(scale_value)}"
    if target_value is not None:
        return f"{prefix}T{_format_float_for_filename(target_value)}"
    return f"{prefix}X1"


def _resolve_scale_factors(
    loaded_records: list[TraceRecord],
    target_avg_input_length: float | None,
    target_avg_output_length: float | None,
    input_scale_factor: float | None,
    output_scale_factor: float | None,
    use_same_scale: bool = False,
) -> tuple[float, float, float, float]:
    input_vals = [record.input_length for record in loaded_records if record.input_length is not None]
    output_vals = [record.output_length for record in loaded_records if record.output_length is not None]
    source_input_avg = _mean([int(v) for v in input_vals])
    source_output_avg = _mean([int(v) for v in output_vals])

    if use_same_scale:
        # 使用单一比例，同时缩放input和output
        if input_scale_factor is not None:
            # 优先使用input_scale_factor作为统一比例
            unified_scale = input_scale_factor
        elif output_scale_factor is not None:
            unified_scale = output_scale_factor
        elif target_avg_input_length is not None:
            unified_scale = target_avg_input_length / source_input_avg
        elif target_avg_output_length is not None:
            unified_scale = target_avg_output_length / source_output_avg
        else:
            unified_scale = 1.0
        
        if unified_scale <= 0:
            raise ValueError("scale factor must be positive")
        
        return source_input_avg, source_output_avg, unified_scale, unified_scale
    else:
        # 原有的独立缩放逻辑
        if input_scale_factor is None:
            if target_avg_input_length is None:
                input_scale_factor = 1.0
            else:
                if target_avg_input_length <= 0:
                    raise ValueError("target_avg_input_length must be positive")
                input_scale_factor = target_avg_input_length / source_input_avg

        if output_scale_factor is None:
            if target_avg_output_length is None:
                output_scale_factor = 1.0
            else:
                if target_avg_output_length <= 0:
                    raise ValueError("target_avg_output_length must be positive")
                output_scale_factor = target_avg_output_length / source_output_avg

        if input_scale_factor <= 0:
            raise ValueError("input_scale_factor must be positive")
        if output_scale_factor <= 0:
            raise ValueError("output_scale_factor must be positive")

        return source_input_avg, source_output_avg, input_scale_factor, output_scale_factor


def scale_trace_lengths(
    input_path: Path,
    output_path: Path,
    trace_type: str,
    target_avg_input_length: float | None,
    target_avg_output_length: float | None,
    input_scale_factor: float | None,
    output_scale_factor: float | None,
    min_length: int,
    use_same_scale: bool = False,
    max_context_length: int | None = None,
) -> None:
    loaded = load_trace_records(
        input_path,
        trace_type=trace_type,
        require_lengths=True,
        skip_invalid=False,
    )

    source_input_avg, source_output_avg, input_scale_factor, output_scale_factor = _resolve_scale_factors(
        loaded.records,
        target_avg_input_length,
        target_avg_output_length,
        input_scale_factor,
        output_scale_factor,
        use_same_scale=use_same_scale,
    )

    input_key, output_key = _get_length_keys(trace_type)
    block_size = _get_block_size(trace_type)
    scaled_records: list[TraceRecord] = []

    for record in loaded.records:
        if record.input_length is None or record.output_length is None:
            raise ValueError("record has missing input_length/output_length")

        new_input = _scale_length(record.input_length, input_scale_factor, min_length)
        new_output = _scale_length(record.output_length, output_scale_factor, min_length)

        # Filter out records that exceed max_context_length
        if max_context_length is not None and new_input + new_output > max_context_length:
            continue  # Skip this record

        # Recalculate hash_ids for mooncake/qwen traces when scaling
        if trace_type in ("mooncake", "qwen"):
            new_hash_ids = _recalculate_hash_ids(
                list(record.hash_ids) if record.hash_ids else [],
                record.input_length,
                new_input,
                block_size
            )
        else:
            new_hash_ids = list(record.hash_ids) if record.hash_ids else []

        scaled_record = TraceRecord(
            raw=dict(record.raw),
            timestamp_ms=record.timestamp_ms,
            input_length=new_input,
            output_length=new_output,
            hash_ids=new_hash_ids,
        )
        scaled_record.raw[input_key] = new_input
        scaled_record.raw[output_key] = new_output
        scaled_record.raw['hash_ids'] = new_hash_ids  # Also update raw for mooncake/qwen
        scaled_records.append(scaled_record)

    # Calculate filtered count
    filtered_count = len(loaded.records) - len(scaled_records)
    
    scaled_input_avg = _mean([record.input_length for record in scaled_records if record.input_length is not None]) if scaled_records else 0
    scaled_output_avg = _mean([record.output_length for record in scaled_records if record.output_length is not None]) if scaled_records else 0

    write_trace_records(output_path, trace_type, scaled_records, loaded.fieldnames)

    print(
        f"{input_path.name}: input_avg {source_input_avg:.4f} -> {scaled_input_avg:.4f} "
        f"(scale={input_scale_factor:.6f}), "
        f"output_avg {source_output_avg:.4f} -> {scaled_output_avg:.4f} "
        f"(scale={output_scale_factor:.6f})"
    )
    if filtered_count > 0:
        print(f"  Filtered {filtered_count} records exceeding max_context_length={max_context_length}")
    print(f"  Output: {output_path} ({len(scaled_records)} records)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale input_length and output_length in trace records proportionally"
    )
    add_trace_type_argument(parser)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input trace file paths (.jsonl or .csv depending on trace type)",
    )
    parser.add_argument(
        "--target-avg-input-length",
        type=float,
        default=None,
        help="Target average input_length after scaling. If omitted, use --input-scale-factor or keep unchanged.",
    )
    parser.add_argument(
        "--target-avg-output-length",
        type=float,
        default=None,
        help="Target average output_length after scaling. If omitted, use --output-scale-factor or keep unchanged.",
    )
    parser.add_argument(
        "--input-scale-factor",
        type=float,
        default=None,
        help="Direct scale factor for input_length. Overrides --target-avg-input-length.",
    )
    parser.add_argument(
        "--output-scale-factor",
        type=float,
        default=None,
        help="Direct scale factor for output_length. Overrides --target-avg-output-length.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum value after scaling for input/output lengths",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/workspace/vllm_v0.11.0_2/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/selected_trace_for_test",
        help="Directory to save scaled traces",
    )
    parser.add_argument(
        "--no-same-scale",
        action="store_false",
        dest="same_scale",
        help="Disable uniform scaling mode (same scale for input and output). Use this if you want different scale factors for input and output.",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Maximum allowed context length (input + output). Records exceeding this will be filtered out.",
    )

    args = parser.parse_args()

    if args.min_length < 0:
        raise ValueError("min_length must be >= 0")

    if (
        args.target_avg_input_length is None
        and args.target_avg_output_length is None
        and args.input_scale_factor is None
        and args.output_scale_factor is None
        and not args.same_scale
    ):
        raise ValueError(
            "At least one scaling option is required: --target-avg-input-length/--target-avg-output-length/"
            "--input-scale-factor/--output-scale-factor or use --no-same-scale to disable uniform scaling"
        )

    input_paths = [Path(path) for path in args.input]
    output_dir = Path(args.output_dir)

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

    for input_path in input_paths:
        input_label = _build_length_label("in", args.target_avg_input_length, args.input_scale_factor)
        output_label = _build_length_label("out", args.target_avg_output_length, args.output_scale_factor)
        output_path = _build_output_path(
            input_path,
            output_dir,
            input_label,
            output_label,
        )
        scale_trace_lengths(
            input_path=input_path,
            output_path=output_path,
            trace_type=args.trace_type,
            target_avg_input_length=args.target_avg_input_length,
            target_avg_output_length=args.target_avg_output_length,
            input_scale_factor=args.input_scale_factor,
            output_scale_factor=args.output_scale_factor,
            min_length=args.min_length,
            use_same_scale=args.same_scale,
            max_context_length=args.max_context_length,
        )


if __name__ == "__main__":
    main()