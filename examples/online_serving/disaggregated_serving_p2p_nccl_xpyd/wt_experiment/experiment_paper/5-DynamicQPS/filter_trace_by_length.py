import argparse
import json
from pathlib import Path


def build_output_path(input_path: Path, output_dir: Path, threshold: int) -> Path:
    return output_dir / f"{input_path.stem}_th{threshold}{input_path.suffix}"


def filter_trace(input_path: Path, output_path: Path, threshold: int) -> None:
    kept = 0
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue

            total += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_no}: invalid JSON, skipped")
                continue

            input_len = obj.get("input_length")
            output_len = obj.get("output_length")

            if not isinstance(input_len, int) or not isinstance(output_len, int):
                print(f"[WARN] line {line_no}: missing/invalid input_length or output_length, skipped")
                continue

            if input_len + output_len < threshold:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Done. total={total}, kept={kept}, removed={total - kept}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JSONL trace by keeping requests where input_length + output_length < threshold"
    )
    parser.add_argument(
        "--input",
        default="/root/Mooncake/FAST25-release/traces/conversation_trace.jsonl",
        help="Input trace file path (.jsonl)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help="Filter threshold: keep rows with input_length + output_length < threshold",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/Mooncake/FAST25-release/traces",
        help="Directory to save filtered trace",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.threshold <= 0:
        raise ValueError("threshold must be positive")

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    output_path = build_output_path(input_path, output_dir, args.threshold)
    filter_trace(input_path, output_path, args.threshold)


if __name__ == "__main__":
    main()
