import argparse
import csv
import math
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

from trace_utils import add_trace_type_argument, load_trace_records, write_trace_records


def build_output_path(input_path: Path, output_dir: Path, threshold: int) -> Path:
    return output_dir / f"{input_path.stem}_th{threshold}{input_path.suffix}"


def _parse_int_like(value: str | None) -> int | None:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        pass

    try:
        parsed = float(text)
    except ValueError:
        return None

    if not parsed.is_integer():
        return None

    return int(parsed)


def _get_length_keys(trace_type: str) -> tuple[str, str]:
    if trace_type == "azure":
        return "ContextTokens", "GeneratedTokens"
    if trace_type == "burstgpt":
        return "Request tokens", "Response tokens"
    raise ValueError(f"unsupported csv trace type: {trace_type}")


def _filter_csv_trace_fast(input_path: Path, output_path: Path, threshold: int, trace_type: str) -> None:
    input_key, output_key = _get_length_keys(trace_type)

    total = 0
    kept = 0
    invalid = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as fin, output_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"input file is missing CSV header: {input_path}")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if row is None:
                continue

            if not any((value or "").strip() for value in row.values()):
                continue

            total += 1
            input_len = _parse_int_like(row.get(input_key))
            output_len = _parse_int_like(row.get(output_key))

            if input_len is None or output_len is None:
                invalid += 1
                continue

            if input_len + output_len < threshold:
                writer.writerow(row)
                kept += 1

    if invalid:
        print(f"[WARN] skipped invalid length rows: {invalid}")
    print(f"Done. total={total}, kept={kept}, removed={total - kept}")
    print(f"Output: {output_path}")


def _process_csv_chunk(task):
    (
        input_path_str,
        tmp_output_str,
        start_offset,
        end_offset,
        header_end,
        threshold,
        input_idx,
        output_idx,
        expected_cols,
    ) = task

    input_path = Path(input_path_str)
    tmp_output = Path(tmp_output_str)

    total = 0
    kept = 0
    invalid = 0

    with input_path.open("rb") as fin, tmp_output.open("wb") as fout:
        fin.seek(start_offset)

        # Non-first chunks may start in the middle of a line; drop it.
        if start_offset != header_end:
            fin.readline()

        while True:
            line_start = fin.tell()
            if line_start >= end_offset:
                break

            raw_line = fin.readline()
            if not raw_line:
                break

            if not raw_line.strip():
                continue

            total += 1

            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                invalid += 1
                continue

            try:
                row = next(csv.reader([decoded]))
            except Exception:
                invalid += 1
                continue

            if len(row) != expected_cols:
                invalid += 1
                continue

            input_len = _parse_int_like(row[input_idx])
            output_len = _parse_int_like(row[output_idx])
            if input_len is None or output_len is None:
                invalid += 1
                continue

            if input_len + output_len < threshold:
                fout.write(raw_line)
                kept += 1

    return total, kept, invalid, str(tmp_output)


def _filter_csv_trace_parallel(
    input_path: Path,
    output_path: Path,
    threshold: int,
    trace_type: str,
    workers: int,
) -> None:
    input_key, output_key = _get_length_keys(trace_type)

    with input_path.open("rb") as fin:
        header_bytes = fin.readline()
        if not header_bytes:
            raise ValueError(f"input file is missing CSV header: {input_path}")
        header_end = fin.tell()
        file_size = fin.seek(0, os.SEEK_END)

    header_text = header_bytes.decode("utf-8").rstrip("\r\n")
    fieldnames = next(csv.reader([header_text]))
    if not fieldnames:
        raise ValueError(f"input file is missing CSV header: {input_path}")

    if input_key not in fieldnames or output_key not in fieldnames:
        raise ValueError(f"CSV header missing required columns: {input_key}, {output_key}")

    input_idx = fieldnames.index(input_key)
    output_idx = fieldnames.index(output_key)
    expected_cols = len(fieldnames)

    data_size = max(0, file_size - header_end)
    if data_size == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(fieldnames)
        print("Done. total=0, kept=0, removed=0")
        print(f"Output: {output_path}")
        return

    actual_workers = max(1, min(workers, mp.cpu_count(), data_size))
    chunk_size = int(math.ceil(data_size / actual_workers))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="trace_filter_", dir=str(output_path.parent))

    try:
        tasks = []
        for idx in range(actual_workers):
            start = header_end + idx * chunk_size
            end = min(file_size, start + chunk_size)
            tmp_path = Path(tmp_dir_obj.name) / f"part_{idx:04d}.csvpart"
            tasks.append(
                (
                    str(input_path),
                    str(tmp_path),
                    start,
                    end,
                    header_end,
                    threshold,
                    input_idx,
                    output_idx,
                    expected_cols,
                )
            )

        with mp.Pool(processes=actual_workers) as pool:
            results = pool.map(_process_csv_chunk, tasks)

        total = sum(item[0] for item in results)
        kept = sum(item[1] for item in results)
        invalid = sum(item[2] for item in results)

        with output_path.open("wb") as fout:
            fout.write(header_bytes)
            for _, _, _, tmp_output in results:
                with Path(tmp_output).open("rb") as fin_part:
                    while True:
                        block = fin_part.read(1024 * 1024)
                        if not block:
                            break
                        fout.write(block)

        if invalid:
            print(f"[WARN] skipped invalid length rows: {invalid}")
        print(f"Done. total={total}, kept={kept}, removed={total - kept}")
        print(f"Output: {output_path}")
        print(f"workers: {actual_workers}")
    finally:
        tmp_dir_obj.cleanup()


def filter_trace(input_path: Path, output_path: Path, threshold: int, trace_type: str, workers: int) -> None:
    if trace_type in ("azure", "burstgpt"):
        if workers > 1:
            _filter_csv_trace_parallel(input_path, output_path, threshold, trace_type, workers)
        else:
            _filter_csv_trace_fast(input_path, output_path, threshold, trace_type)
        return

    loaded = load_trace_records(
        input_path,
        trace_type=trace_type,
        require_lengths=True,
        skip_invalid=True,
    )

    kept_records = [
        record
        for record in loaded.records
        if record.input_length is not None
        and record.output_length is not None
        and record.input_length + record.output_length < threshold
    ]

    write_trace_records(output_path, trace_type, kept_records, loaded.fieldnames)

    print(f"Done. total={loaded.total_rows}, kept={len(kept_records)}, removed={loaded.total_rows - len(kept_records)}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JSONL trace by keeping requests where input_length + output_length < threshold"
    )
    add_trace_type_argument(parser)
    # Keep defaults consistent with the current default --input (Azure CSV).
    parser.set_defaults(trace_type="qwen")
    parser.add_argument(
        "--input",
        default="/root/workspace/vllm_v0.11.0_2/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/raw_trace/qwen/qwen_thinking_blksz_16.jsonl",
        help="Input trace file path (.jsonl or .csv depending on trace type)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8192,
        help="Filter threshold: keep rows with input_length + output_length < threshold",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/workspace/vllm_v0.11.0_2/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/filter_trace_8192/qwen_trace",
        help="Directory to save filtered trace",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for CSV traces (azure/burstgpt). Use 1 for single-process streaming.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.threshold <= 0:
        raise ValueError("threshold must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv" and args.trace_type in ("mooncake", "qwen"):
        raise ValueError("CSV trace requires --trace-type azure or burstgpt (mooncake/qwen expect JSONL)")
    if suffix in (".jsonl", ".json") and args.trace_type not in ("mooncake", "qwen"):
        raise ValueError("JSONL/JSON trace requires --trace-type mooncake or qwen")

    output_path = build_output_path(input_path, output_dir, args.threshold)
    filter_trace(input_path, output_path, args.threshold, args.trace_type, args.workers)


if __name__ == "__main__":
    main()
