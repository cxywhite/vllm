#!/usr/bin/env python3
"""Sample 1k rows from a CSV, compute mean_len/sigma_len stats, and save results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INPUT_CSV = Path(
    "/root/myshare/mean_and_sigma/"
    "llama_4_sample_range_50_lens_compute_means_and_probabilities_20250603_121645_with_stats_v2.csv"
)
OUTPUT_DIR = Path(
    "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/"
    "wt_experiment/experiment_paper/Mean_and_sigma/dataset"
)
SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def numeric_describe(series: pd.Series) -> dict:
    """Return key descriptive stats for a numeric series."""
    desc = series.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(desc.get("count", 0)),
        "mean": float(desc.get("mean", float("nan"))),
        "std": float(desc.get("std", float("nan"))),
        "min": float(desc.get("min", float("nan"))),
        "p25": float(desc.get("25%", float("nan"))),
        "p50": float(desc.get("50%", float("nan"))),
        "p75": float(desc.get("75%", float("nan"))),
        "p90": float(desc.get("90%", float("nan"))),
        "p95": float(desc.get("95%", float("nan"))),
        "p99": float(desc.get("99%", float("nan"))),
        "max": float(desc.get("max", float("nan"))),
    }


def format_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column cleanup/rename rules for exported CSV files."""
    out = df.reset_index(drop=True).copy()

    # Add sequential request id based on row order.
    out.insert(0, "req_id", range(1, len(out) + 1))

    out = out.drop(columns=["input_tokens", "output", "output_tokens"], errors="ignore")
    out = out.rename(
        columns={
            "input_length": "prompt_len",
            "output_length": "output_tokens",
            "topp": "top_p",
            "topk": "top_k",
        }
    )
    return out


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = {"id", "mean_len", "sigma_len"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Convert target columns to numeric for reliable statistics.
    df["mean_len"] = pd.to_numeric(df["mean_len"], errors="coerce")
    df["sigma_len"] = pd.to_numeric(df["sigma_len"], errors="coerce")

    # Shuffle first, then pick up to 1k rows.
    shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    sample_n = min(SAMPLE_SIZE, len(shuffled))
    sampled = shuffled.head(sample_n).copy()

    id_counts = sampled["id"].value_counts(dropna=False)
    singleton_ids = set(id_counts[id_counts == 1].index.tolist())
    singleton_rows = sampled[sampled["id"].isin(singleton_ids)].copy()

    stats = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUTPUT_DIR),
        "total_rows_in_input": int(len(df)),
        "sample_size_requested": SAMPLE_SIZE,
        "sample_size_actual": int(len(sampled)),
        "unique_ids_in_sample": int(sampled["id"].nunique(dropna=False)),
        "ids_appearing_once_count": int((id_counts == 1).sum()),
        "rows_with_singleton_ids_count": int(len(singleton_rows)),
        "mean_len_stats": numeric_describe(sampled["mean_len"].dropna()),
        "sigma_len_stats": numeric_describe(sampled["sigma_len"].dropna()),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sampled_csv = OUTPUT_DIR / "sampled_1k_shuffled.csv"
    singleton_csv = OUTPUT_DIR / "sampled_1k_singleton_id_rows.csv"
    stats_json = OUTPUT_DIR / "sampled_1k_stats.json"
    stats_txt = OUTPUT_DIR / "sampled_1k_stats.txt"

    sampled_export = format_export_df(sampled)
    singleton_export = format_export_df(singleton_rows)

    sampled_export.to_csv(sampled_csv, index=False)
    singleton_export.to_csv(singleton_csv, index=False)

    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with stats_txt.open("w", encoding="utf-8") as f:
        f.write("Sampled 1k Statistics\n")
        f.write("=" * 24 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    print("Done.")
    print(f"Sampled data: {sampled_csv}")
    print(f"Singleton-id rows: {singleton_csv}")
    print(f"Stats JSON: {stats_json}")
    print(f"Stats TXT: {stats_txt}")


if __name__ == "__main__":
    main()
