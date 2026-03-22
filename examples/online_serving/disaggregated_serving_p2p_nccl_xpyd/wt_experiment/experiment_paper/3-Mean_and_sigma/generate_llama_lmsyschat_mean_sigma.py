#!/usr/bin/env python3
"""Export rows from loadbalance CSV that are missing in stats CSV by sampling keys."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_LOADBALANCE = Path(
    "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/"
    "wt_experiment/experiment_paper/Loadbalance/dataset/"
    "llama-lmsys-chat-loadbalance_1p3d_good_2short1long_3.csv"
)
CSV_STATS = Path(
    "/root/myshare/mean_and_sigma/"
    "llama_4_sample_range_50_lens_compute_means_and_probabilities_20250603_121645_with_stats_v2.csv"
)
OUT_DIR = Path(
    "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/"
    "wt_experiment/experiment_paper/Mean_and_sigma/dataset"
)
OUT_PREFIX = "llama_lmsyschat_mean_and_sigma"

JOIN_KEYS = ["prompt", "temperature", "top_p", "top_k", "repetition_penalty"]
NUMERIC_JOIN_KEYS = ["temperature", "top_p", "top_k", "repetition_penalty"]
ALIAS_COLS = {"topp": "top_p", "topk": "top_k"}


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def canonicalize_sampling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify sampling column names so both topp/topk and top_p/top_k are supported."""
    out = df.copy()
    for old_col, new_col in ALIAS_COLS.items():
        if new_col not in out.columns and old_col in out.columns:
            out = out.rename(columns={old_col: new_col})
    return out


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Normalize prompt text to avoid mismatches caused by whitespace/newline variants.
    out["prompt"] = (
        out["prompt"]
        .astype(str)
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.strip()
    )
    # Match rule: keep 2 decimals for temperature/top_p/repetition_penalty, and integer for top_k.
    for c in ["temperature", "top_p", "repetition_penalty"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out["top_k"] = pd.to_numeric(out["top_k"], errors="coerce").round(0).astype("Int64")
    return out


def main() -> None:
    if not CSV_LOADBALANCE.exists():
        raise FileNotFoundError(f"File not found: {CSV_LOADBALANCE}")
    if not CSV_STATS.exists():
        raise FileNotFoundError(f"File not found: {CSV_STATS}")

    lb = pd.read_csv(CSV_LOADBALANCE)
    st = pd.read_csv(CSV_STATS)

    lb = canonicalize_sampling_columns(lb)
    st = canonicalize_sampling_columns(st)

    require_columns(lb, JOIN_KEYS, "loadbalance csv")
    require_columns(st, JOIN_KEYS, "stats csv")

    lb = normalize_keys(lb)
    st = normalize_keys(st)

    merged = lb.merge(
        st[JOIN_KEYS].drop_duplicates(),
        on=JOIN_KEYS,
        how="left",
        indicator=True,
    )
    unmatched = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUT_DIR / f"{OUT_PREFIX}_{ts}.csv"
    unmatched.to_csv(out_csv, index=False)

    # Requirement: print only the number of rows not found in stats CSV.
    print(len(unmatched))


if __name__ == "__main__":
    main()
