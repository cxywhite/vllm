import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TRACE_TYPES = ("mooncake", "qwen", "azure", "burstgpt", "test_results")


@dataclass
class TraceRecord:
    raw: dict
    timestamp_ms: float | None
    input_length: int | None
    output_length: int | None
    hash_ids: list[int]


@dataclass
class LoadedTrace:
    records: list[TraceRecord]
    fieldnames: list[str] | None
    total_rows: int


def add_trace_type_argument(parser) -> None:
    parser.add_argument(
        "--trace-type",
        choices=TRACE_TYPES,
        default="mooncake",
        help="Trace format to process: mooncake/qwen JSONL(auto-normalized to ms), AzureLLMInferenceTrace CSV, BurstGPT CSV, or test_results CSV",
    )


def _infer_numeric_timestamp_scale_to_ms(values: list[float]) -> float:
    finite_nonzero = [abs(v) for v in values if math.isfinite(v) and v != 0.0]
    if not finite_nonzero:
        return 1.0

    finite_nonzero.sort()
    probe = finite_nonzero[len(finite_nonzero) // 2]

    if probe >= 1e17:
        # nanoseconds -> milliseconds
        return 1.0 / 1_000_000.0
    if probe >= 1e14:
        # microseconds -> milliseconds
        return 1.0 / 1_000.0
    if probe >= 1e11:
        # already milliseconds
        return 1.0
    if probe >= 1e8:
        # epoch seconds -> milliseconds
        return 1_000.0

    # Relative traces (start around zero): detect once per file.
    # Typical second-based traces have median in thousands, ms-based traces in >=1e4.
    return 1_000.0 if probe < 1e4 else 1.0


def _parse_int_like(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        try:
            parsed = float(text)
        except ValueError:
            return None

        return int(parsed) if parsed.is_integer() else None

    return None


def _parse_float_like(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        try:
            return float(text)
        except ValueError:
            return None

    return None


def _coerce_hash_ids(value) -> list[int]:
    if not isinstance(value, list):
        return []

    out = []
    for item in value:
        parsed = _parse_int_like(item)
        if parsed is None:
            return []
        out.append(parsed)

    return out


def parse_timestamp_value(trace_type: str, value):
    if trace_type == "mooncake":
        return _parse_float_like(value)

    if trace_type == "qwen":
        return _parse_float_like(value)

    if trace_type == "azure":
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.timestamp() * 1000.0

    if trace_type == "burstgpt":
        seconds = _parse_float_like(value)
        if seconds is None:
            return None
        return seconds * 1000.0

    if trace_type == "test_results":
        return _parse_float_like(value)

    raise ValueError(f"unsupported trace type: {trace_type}")


def parse_timestamp_bound(trace_type: str, value):
    parsed = parse_timestamp_value(trace_type, value)
    if parsed is None:
        raise ValueError(f"invalid timestamp bound for trace_type={trace_type}: {value}")

    if trace_type in ("mooncake", "qwen"):
        # For manual bounds, keep convention simple and explicit: numeric values are milliseconds.
        return parsed

    return parsed


def format_timestamp_label(trace_type: str, timestamp_ms: float) -> str:
    if trace_type == "azure":
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y%m%dT%H%M%S%fZ")

    value = timestamp_ms if trace_type in ("mooncake", "qwen", "test_results") else timestamp_ms / 1000.0
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))

    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _build_record(trace_type: str, raw: dict) -> TraceRecord:
    if trace_type in ("mooncake", "qwen"):
        timestamp_ms = parse_timestamp_value(trace_type, raw.get("timestamp"))
        input_length = _parse_int_like(raw.get("input_length"))
        output_length = _parse_int_like(raw.get("output_length"))
        hash_ids = _coerce_hash_ids(raw.get("hash_ids"))
    elif trace_type == "azure":
        timestamp_ms = parse_timestamp_value(trace_type, raw.get("TIMESTAMP"))
        input_length = _parse_int_like(raw.get("ContextTokens"))
        output_length = _parse_int_like(raw.get("GeneratedTokens"))
        hash_ids = []
    elif trace_type == "burstgpt":
        timestamp_ms = parse_timestamp_value(trace_type, raw.get("Timestamp"))
        input_length = _parse_int_like(raw.get("Request tokens"))
        output_length = _parse_int_like(raw.get("Response tokens"))
        hash_ids = []
    elif trace_type == "test_results":
        timestamp_ms = parse_timestamp_value(trace_type, raw.get("send_timestamp_ms"))
        input_length = _parse_int_like(raw.get("prompt_len"))
        output_length = _parse_int_like(raw.get("output_tokens"))
        hash_ids = []
    else:
        raise ValueError(f"unsupported trace type: {trace_type}")

    return TraceRecord(
        raw=raw,
        timestamp_ms=timestamp_ms,
        input_length=input_length,
        output_length=output_length,
        hash_ids=hash_ids,
    )


def load_trace_records(
    input_path: Path,
    trace_type: str,
    require_timestamp: bool = False,
    require_lengths: bool = False,
    skip_invalid: bool = False,
) -> LoadedTrace:
    records: list[TraceRecord] = []
    total_rows = 0
    fieldnames = None

    def handle_invalid(line_no: int, reason: str):
        if skip_invalid:
            print(f"[WARN] line {line_no}: {reason}, skipped")
            return
        raise ValueError(f"line {line_no}: {reason}")

    if trace_type in ("mooncake", "qwen"):
        buffered_rows: list[tuple[int, dict]] = []
        raw_timestamps: list[float] = []

        with input_path.open("r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue

                total_rows += 1
                try:
                    raw_obj = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    if skip_invalid:
                        print(f"[WARN] line {line_no}: invalid JSON, skipped")
                        continue
                    raise ValueError(f"line {line_no}: invalid JSON") from exc

                buffered_rows.append((line_no, raw_obj))
                parsed_ts = _parse_float_like(raw_obj.get("timestamp"))
                if parsed_ts is not None:
                    raw_timestamps.append(parsed_ts)

        timestamp_scale = _infer_numeric_timestamp_scale_to_ms(raw_timestamps)

        for line_no, raw_obj in buffered_rows:
            raw_ts = _parse_float_like(raw_obj.get("timestamp"))
            timestamp_ms = None if raw_ts is None else raw_ts * timestamp_scale

            record = TraceRecord(
                raw=raw_obj,
                timestamp_ms=timestamp_ms,
                input_length=_parse_int_like(raw_obj.get("input_length")),
                output_length=_parse_int_like(raw_obj.get("output_length")),
                hash_ids=_coerce_hash_ids(raw_obj.get("hash_ids")),
            )

            if require_timestamp and record.timestamp_ms is None:
                handle_invalid(line_no, "missing/invalid timestamp")
                continue
            if require_lengths and (record.input_length is None or record.output_length is None):
                handle_invalid(line_no, "missing/invalid input_length or output_length")
                continue

            records.append(record)
    else:
        with input_path.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            fieldnames = list(reader.fieldnames or [])

            if not fieldnames:
                raise ValueError(f"input file is missing CSV header: {input_path}")

            for row_idx, row in enumerate(reader, start=2):
                if row is None:
                    continue

                if not any((value or "").strip() for value in row.values()):
                    continue

                total_rows += 1
                record = _build_record(trace_type, dict(row))

                if require_timestamp and record.timestamp_ms is None:
                    handle_invalid(row_idx, "missing/invalid timestamp")
                    continue

                if require_lengths and (record.input_length is None or record.output_length is None):
                    handle_invalid(row_idx, "missing/invalid input_length or output_length")
                    continue

                records.append(record)

    if not records:
        raise ValueError(f"input file has no valid records: {input_path}")

    return LoadedTrace(records=records, fieldnames=fieldnames, total_rows=total_rows)


def _format_scaled_timestamp(trace_type: str, timestamp_ms: float):
    if trace_type in ("mooncake", "qwen"):
        return int(round(timestamp_ms))

    if trace_type == "azure":
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    if trace_type == "burstgpt":
        seconds = timestamp_ms / 1000.0
        rounded = round(seconds)
        if abs(seconds - rounded) < 1e-9:
            return str(int(rounded))
        return f"{seconds:.6f}".rstrip("0").rstrip(".")

    if trace_type == "test_results":
        return int(round(timestamp_ms))

    raise ValueError(f"unsupported trace type: {trace_type}")


def set_record_timestamp(trace_type: str, record: TraceRecord, timestamp_ms: float) -> None:
    record.timestamp_ms = timestamp_ms

    if trace_type in ("mooncake", "qwen"):
        record.raw["timestamp"] = _format_scaled_timestamp(trace_type, timestamp_ms)
    elif trace_type == "azure":
        record.raw["TIMESTAMP"] = _format_scaled_timestamp(trace_type, timestamp_ms)
    elif trace_type == "burstgpt":
        record.raw["Timestamp"] = _format_scaled_timestamp(trace_type, timestamp_ms)
    elif trace_type == "test_results":
        record.raw["send_timestamp_ms"] = _format_scaled_timestamp(trace_type, timestamp_ms)
    else:
        raise ValueError(f"unsupported trace type: {trace_type}")


def write_trace_records(
    output_path: Path,
    trace_type: str,
    records: list[TraceRecord],
    fieldnames: list[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trace_type in ("mooncake", "qwen"):
        with output_path.open("w", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record.raw, ensure_ascii=False) + "\n")
        return

    final_fieldnames = list(fieldnames or [])
    if not final_fieldnames and records:
        final_fieldnames = list(records[0].raw.keys())

    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=final_fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.raw.get(name, "") for name in final_fieldnames})


def compute_average_qps_from_records(records: list[TraceRecord]) -> float:
    timestamps = [record.timestamp_ms for record in records if record.timestamp_ms is not None]
    if not timestamps:
        raise ValueError("trace has no valid timestamps")

    start_ts = min(timestamps)
    end_ts = max(timestamps)
    start_sec = int(start_ts // 1000)
    end_sec = int(end_ts // 1000)
    total_seconds = end_sec - start_sec + 1

    if total_seconds <= 0:
        raise ValueError("invalid trace duration")

    counts_by_second: dict[int, int] = {}
    for timestamp in timestamps:
        second_bin = int(timestamp // 1000) - start_sec
        counts_by_second[second_bin] = counts_by_second.get(second_bin, 0) + 1

    total_requests = sum(counts_by_second.get(second_idx, 0) for second_idx in range(total_seconds))
    return total_requests / total_seconds


def scale_trace_records(records: list[TraceRecord], trace_type: str, scale_factor: float) -> list[TraceRecord]:
    timestamps = [record.timestamp_ms for record in records if record.timestamp_ms is not None]
    if not timestamps:
        raise ValueError("trace has no valid timestamps")

    base_timestamp = min(timestamps)
    scaled_records: list[TraceRecord] = []

    for record in records:
        scaled_record = TraceRecord(
            raw=dict(record.raw),
            timestamp_ms=record.timestamp_ms,
            input_length=record.input_length,
            output_length=record.output_length,
            hash_ids=list(record.hash_ids),
        )

        if record.timestamp_ms is None:
            raise ValueError("cannot scale record without timestamp")

        relative_timestamp = record.timestamp_ms - base_timestamp
        scaled_timestamp = base_timestamp + relative_timestamp * scale_factor
        set_record_timestamp(trace_type, scaled_record, scaled_timestamp)
        scaled_records.append(scaled_record)

    return scaled_records