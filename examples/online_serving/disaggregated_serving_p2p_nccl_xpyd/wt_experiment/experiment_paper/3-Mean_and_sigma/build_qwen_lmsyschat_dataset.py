#!/usr/bin/env python3
"""Build qwen_lmsyschat dataset by matching id and sampling params."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

FILTER_CSV = Path(
    "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/"
    "wt_experiment/experiment_paper/Mean_and_sigma/dataset/sampled_1k_id_temp_topp_topk_rp.csv"
)
SOURCE_CSV = Path(
    "/root/myshare/mean_and_sigma/"
    "qwen_4_sample_range_50_lens_compute_means_and_probabilities_20260221_131530_with_stats_v2.csv"
)
OUTPUT_DIR = Path(
    "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/"
    "wt_experiment/experiment_paper/Mean_and_sigma/dataset"
)


def normalize_key_columns(df: pd.DataFrame, *, from_filter: bool) -> pd.DataFrame:
    out = df.copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")

    if from_filter:
        out["temperature_key"] = pd.to_numeric(out["temperature"], errors="coerce").round(2)
        out["top_p_key"] = pd.to_numeric(out["topp"], errors="coerce").round(2)
        out["top_k_key"] = pd.to_numeric(out["topk"], errors="coerce").fillna(0).astype(int)
        out["repetition_penalty_key"] = pd.to_numeric(out["repetition_penalty"], errors="coerce").round(2)
    else:
        out["temperature_key"] = pd.to_numeric(out["temperature"], errors="coerce").round(2)
        out["top_p_key"] = pd.to_numeric(out["topp"], errors="coerce").round(2)
        out["top_k_key"] = pd.to_numeric(out["topk"], errors="coerce").fillna(0).astype(int)
        out["repetition_penalty_key"] = pd.to_numeric(out["repetition_penalty"], errors="coerce").round(2)

    return out


def main() -> None:
    if not FILTER_CSV.exists():
        raise FileNotFoundError(f"Filter CSV not found: {FILTER_CSV}")
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    filter_df = pd.read_csv(FILTER_CSV)
    source_df = pd.read_csv(SOURCE_CSV)

    required_filter = {"id", "temperature", "topp", "topk", "repetition_penalty"}
    required_source = {
        "prompt",
        "input_length",
        "id",
        "temperature",
        "topk",
        "repetition_penalty",
        "topp",
        "lens",
        "probabilities",
        "mean",
        "mean_len",
        "sigma_len",
        "confidence_95",
    }

    missing_filter = sorted(required_filter - set(filter_df.columns))
    missing_source = sorted(required_source - set(source_df.columns))
    if missing_filter:
        raise ValueError(f"Missing columns in filter CSV: {missing_filter}")
    if missing_source:
        raise ValueError(f"Missing columns in source CSV: {missing_source}")

    filter_df = normalize_key_columns(filter_df, from_filter=True)
    source_df = normalize_key_columns(source_df, from_filter=False)

    key_cols = ["id", "temperature_key", "top_p_key", "top_k_key", "repetition_penalty_key"]

    # Keep one source row for each key to avoid ambiguous merge expansion.
    source_unique = source_df.drop_duplicates(subset=key_cols, keep="first").copy()

    merged = filter_df[key_cols].merge(
        source_unique,
        on=key_cols,
        how="left",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched) > 0:
        print(f"Warning: {len(unmatched)} rows from filter CSV did not match source CSV.")

    matched = merged[merged["_merge"] == "both"].copy().reset_index(drop=True)

    out_df = pd.DataFrame(
        {
            "req_id": range(1, len(matched) + 1),
            "prompt": matched["prompt"],
            "id": matched["id"].astype(int),
            "prompt_len": pd.to_numeric(matched["input_length"], errors="coerce").fillna(0).astype(int),
            # Source CSV does not provide per-request output_tokens; use mean_len as token-length proxy.
            "output_tokens": pd.to_numeric(matched["mean_len"], errors="coerce").fillna(0).round().astype(int),
            "temperature": pd.to_numeric(matched["temperature"], errors="coerce").map(lambda x: f"{x:.2f}"),
            "top_p": pd.to_numeric(matched["topp"], errors="coerce").map(lambda x: f"{x:.2f}"),
            "top_k": pd.to_numeric(matched["topk"], errors="coerce").fillna(0).round().astype(int),
            "repetition_penalty": pd.to_numeric(matched["repetition_penalty"], errors="coerce").map(lambda x: f"{x:.2f}"),
            "lens": matched["lens"],
            "probabilities": matched["probabilities"],
            "mean": pd.to_numeric(matched["mean"], errors="coerce").fillna(0).astype(int),
            "mean_len": pd.to_numeric(matched["mean_len"], errors="coerce").fillna(0).astype(int),
            "sigma_len": pd.to_numeric(matched["sigma_len"], errors="coerce").fillna(0).astype(int),
            "confidence_95": matched["confidence_95"],
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = OUTPUT_DIR / f"qwen_lmsyschat_sampled_1k_shuffled_{ts}.csv"
    out_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(f"Matched rows: {len(out_df)}")


if __name__ == "__main__":
    main()
