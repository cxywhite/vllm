#!/usr/bin/env python3
"""Extract selected columns from sampled_1k_shuffled.csv and rename top_p/top_k."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path(
    "/root/predict-schedule/vllm/examples/online_serving/"
    "disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/"
    "Mean_and_sigma/dataset/sampled_1k_shuffled.csv"
)
OUTPUT_CSV = Path(
    "/root/predict-schedule/vllm/examples/online_serving/"
    "disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/"
    "Mean_and_sigma/dataset/sampled_1k_id_temp_topp_topk_rp.csv"
)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["id", "temperature", "top_p", "top_k", "repetition_penalty"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    out_df = df[required_cols].rename(columns={"top_p": "topp", "top_k": "topk"})

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
