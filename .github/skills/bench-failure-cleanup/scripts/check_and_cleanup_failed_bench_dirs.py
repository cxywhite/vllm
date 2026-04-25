#!/usr/bin/env python3
"""Scan benchmark result directories and optionally delete failed ones.

A directory is considered PASS only when:
1) Its name contains `_n<expected>_`.
2) A bench log exists under `<dir>/log/` matching a glob pattern.
3) The selected bench log contains `Successful requests: <num>`.
4) <num> equals expected.

Otherwise it is FAIL with a reason.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

N_PATTERN = re.compile(r"_n(?P<n>\d+)_")
SUCCESS_PATTERN = re.compile(r"Successful requests:\s*(\d+)")
PROTECTED_PATHS = [Path("/root/.cache/huggingface/hub").resolve()]


@dataclass
class ScanRow:
    dir_name: str
    expected_n: str
    successful: str
    status: str
    reason: str
    log_path: str


def is_in_path(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_protected_path(path: Path) -> bool:
    return any(path == protected or is_in_path(path, protected) for protected in PROTECTED_PATHS)


def parse_expected_n(dir_name: str) -> int | None:
    match = N_PATTERN.search(dir_name)
    if not match:
        return None
    return int(match.group("n"))


def choose_log_file(log_dir: Path, log_glob: str) -> Path | None:
    candidates = [p for p in log_dir.glob(log_glob) if p.is_file()]
    if not candidates:
        return None
    # Choose the newest candidate by mtime for deterministic behavior.
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def parse_successful_requests(log_path: Path) -> int | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    matches = SUCCESS_PATTERN.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def scan_dirs(base_dir: Path, log_glob: str) -> list[ScanRow]:
    rows: list[ScanRow] = []
    for run_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        expected_n = parse_expected_n(run_dir.name)
        if expected_n is None:
            rows.append(
                ScanRow(
                    dir_name=run_dir.name,
                    expected_n="",
                    successful="",
                    status="FAIL",
                    reason="dir_name_n_not_found",
                    log_path="",
                )
            )
            continue

        log_dir = run_dir / "log"
        log_path = choose_log_file(log_dir, log_glob) if log_dir.is_dir() else None
        if log_path is None:
            rows.append(
                ScanRow(
                    dir_name=run_dir.name,
                    expected_n=str(expected_n),
                    successful="",
                    status="FAIL",
                    reason="bench_log_not_found",
                    log_path="",
                )
            )
            continue

        successful = parse_successful_requests(log_path)
        if successful is None:
            rows.append(
                ScanRow(
                    dir_name=run_dir.name,
                    expected_n=str(expected_n),
                    successful="",
                    status="FAIL",
                    reason="successful_requests_not_found",
                    log_path=str(log_path),
                )
            )
            continue

        if successful == expected_n:
            rows.append(
                ScanRow(
                    dir_name=run_dir.name,
                    expected_n=str(expected_n),
                    successful=str(successful),
                    status="PASS",
                    reason="",
                    log_path=str(log_path),
                )
            )
        else:
            rows.append(
                ScanRow(
                    dir_name=run_dir.name,
                    expected_n=str(expected_n),
                    successful=str(successful),
                    status="FAIL",
                    reason="successful_not_equal_expected_n",
                    log_path=str(log_path),
                )
            )

    return rows


def write_tsv(rows: Iterable[ScanRow], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["dir_name", "expected_n", "successful", "status", "reason", "log_path"])
        for row in rows:
            writer.writerow(
                [
                    row.dir_name,
                    row.expected_n,
                    row.successful,
                    row.status,
                    row.reason,
                    row.log_path,
                ]
            )


def parse_reason_filter(value: str) -> set[str]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    return set(parts)


def safe_delete_failed_dirs(
    rows: Iterable[ScanRow],
    base_dir: Path,
    reason_filter: set[str],
    dry_run: bool,
) -> tuple[list[str], list[str], list[str]]:
    deleted: list[str] = []
    not_found: list[str] = []
    skipped: list[str] = []

    base_real = base_dir.resolve()

    if is_protected_path(base_real):
        raise SystemExit(
            "ERROR: delete operation is forbidden for protected path "
            f"and subdirectories: {base_real}"
        )

    for row in rows:
        if row.status != "FAIL":
            continue
        if reason_filter and row.reason not in reason_filter:
            continue

        target = (base_dir / row.dir_name).resolve()

        if is_protected_path(target):
            skipped.append(f"{row.dir_name} (protected_path_blocked)")
            continue

        if not is_in_path(target, base_real):
            skipped.append(f"{row.dir_name} (outside_base_dir)")
            continue

        if not target.exists():
            not_found.append(row.dir_name)
            continue

        if dry_run:
            skipped.append(f"{row.dir_name} (dry_run)")
            continue

        shutil.rmtree(target)
        deleted.append(row.dir_name)

    return deleted, not_found, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Check benchmark success counts and optionally delete failed directories.")
    parser.add_argument("--base-dir", required=True, help="Result base directory that contains run directories.")
    parser.add_argument("--log-glob", default="bench*.log", help="Glob for bench logs under each run's log directory.")
    parser.add_argument(
        "--out-tsv",
        default="/tmp/bench_check.tsv",
        help="Output TSV path. Default: /tmp/bench_check.tsv",
    )
    parser.add_argument(
        "--delete-fail",
        action="store_true",
        help="Delete directories with FAIL status after scanning.",
    )
    parser.add_argument(
        "--fail-reasons",
        default="",
        help="Comma-separated FAIL reasons to delete. Empty means all FAIL reasons.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview deletion targets without deleting.")

    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_tsv = Path(args.out_tsv).resolve()

    if not base_dir.is_dir():
        raise SystemExit(f"ERROR: base dir not found: {base_dir}")

    if args.delete_fail and is_protected_path(base_dir):
        raise SystemExit(
            "ERROR: delete operation is forbidden for /root/.cache/huggingface/hub "
            "and all its subdirectories."
        )

    rows = scan_dirs(base_dir, args.log_glob)
    write_tsv(rows, out_tsv)

    total = len(rows)
    passed = sum(1 for r in rows if r.status == "PASS")
    failed = sum(1 for r in rows if r.status == "FAIL")

    print(f"base_dir={base_dir}")
    print(f"total={total}")
    print(f"pass={passed}")
    print(f"fail={failed}")
    print(f"out_tsv={out_tsv}")

    fail_rows = [r for r in rows if r.status == "FAIL"]
    if fail_rows:
        print("failed_dirs:")
        for r in fail_rows:
            print(
                f"- {r.dir_name} | expected_n={r.expected_n or 'na'} "
                f"| successful={r.successful or 'na'} | reason={r.reason}"
            )

    if args.delete_fail:
        reasons = parse_reason_filter(args.fail_reasons)
        deleted, not_found, skipped = safe_delete_failed_dirs(
            rows=rows,
            base_dir=base_dir,
            reason_filter=reasons,
            dry_run=args.dry_run,
        )
        print("delete_summary:")
        print(f"- delete_fail={args.delete_fail}")
        print(f"- dry_run={args.dry_run}")
        print(f"- reason_filter={','.join(sorted(reasons)) if reasons else 'ALL'}")
        print(f"- deleted={len(deleted)}")
        print(f"- not_found={len(not_found)}")
        print(f"- skipped={len(skipped)}")

        if deleted:
            print("deleted_dirs:")
            for name in deleted:
                print(f"- {name}")

        if not_found:
            print("not_found_dirs:")
            for name in not_found:
                print(f"- {name}")

        if skipped:
            print("skipped_dirs:")
            for item in skipped:
                print(f"- {item}")


if __name__ == "__main__":
    main()
