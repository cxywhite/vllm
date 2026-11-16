#!/usr/bin/env python3
"""Llama-3-8B + LMSys, 2P6D, TPOT SLO=30ms.

Plot per-request TPOT CDF for the three traces (Qwen / Mooncake / BurstGPT)
and the three methods (RR / Dynamo / Ours), to show TPOT compliance-rate
differences and that BurstGPT is ~100% within the 30 ms SLO.

Per-request TPOT definition (matches the paper's goodput SLO):
    tpot_ms = (latency - ttft) / (output_tokens - 1) * 1000   for output_tokens > 1
              = 0                                             otherwise
Source data: <run_dir>/dataset_result/test_results_*.csv  (RR / Ours).
Dynamo has no per-request latency dump; we annotate its aggregate mean/p99 instead.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLO_TPOT_MS = 30.0

ROOT = Path("/root/.cache/huggingface/hub/wt_predictor/0630")

# (trace, method) -> test_results CSV (RR / Ours) or None
RUNS: dict[tuple[str, str], Path | None] = {
    ("Qwen", "RR"): ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-qwen_strategyA_n4080_pd2p6d_tpot30ms_traceqwen_strategyA_rep1_20260720_145304_roundrobin/dataset_result/test_results_20260720_151538.csv",
    ("Qwen", "Ours"): ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-qwen_strategyA_n4080_pd2p6d_tpot30ms_traceqwen_strategyA_rep1_20260720_151604_scheme4_aimd/dataset_result/test_results_20260720_152908.csv",
    ("Qwen", "Dynamo"): None,  # no Dynamo run for qwen trace 2p6d
    ("Mooncake", "RR"): ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-mooncake_n1790_pd2p6d_tpot30ms_tracemooncake_rep1_20260710_185910_roundrobin/dataset_result/test_results_20260710_191551.csv",
    ("Mooncake", "Ours"): ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-mooncake_n1790_pd2p6d_tpot30ms_tracemooncake_rep1_20260710_191606_scheme4_aimd/dataset_result/test_results_20260710_192427.csv",
    ("Mooncake", "Dynamo"): None,  # Dynamo result has no per-request latency
    ("BurstGPT", "RR"): ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260709_174420_roundrobin/dataset_result/test_results_20260709_175227.csv",
    ("BurstGPT", "Ours"): ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260709_175243_scheme4_aimd/dataset_result/test_results_20260709_180114.csv",
    ("BurstGPT", "Dynamo"): None,  # Dynamo result has no per-request latency
}

# Aggregate tpot (mean_ms, p99_ms) for Dynamo, from openai-chat-trace JSON.
DYNAMO_AGG = {
    "BurstGPT": (16.55, 20.05),
    "Mooncake": (32.50, 42.35),
    "Qwen": None,
}

TRACE_ORDER = ["Qwen", "Mooncake", "BurstGPT"]
METHOD_ORDER = ["RR", "Dynamo", "Ours"]
COLORS = {"RR": "#73C58C", "Dynamo": "#F0A35E", "Ours": "#5B8FB9"}
LABELS = {"RR": "RR", "Dynamo": "Dynamo", "Ours": "Ours (Presage)"}


def load_tpot_ms(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, encoding="utf-8")
    lat = pd.to_numeric(df["latency"], errors="coerce")
    ttft = pd.to_numeric(df["ttft"], errors="coerce")
    out_tokens = pd.to_numeric(df["output_tokens"], errors="coerce")

    valid = (out_tokens > 1) & lat.notna() & ttft.notna() & (lat >= ttft)
    tpot_ms = (lat - ttft) / (out_tokens - 1) * 1000.0
    return tpot_ms[valid].to_numpy(dtype=float)


def cdf_curve(values: np.ndarray):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    print("=" * 70)
    print("Llama-3-8B + LMSys, 2P6D, TPOT SLO = 30 ms")
    print("=" * 70)

    for col, trace in enumerate(TRACE_ORDER):
        ax = axes[col]
        for method in METHOD_ORDER:
            csv_path = RUNS[(trace, method)]
            if csv_path is not None:
                tpot = load_tpot_ms(csv_path)
                x, y = cdf_curve(tpot)
                ax.plot(x, y, color=COLORS[method], lw=2.0, label=LABELS[method])
                comply = float((tpot <= SLO_TPOT_MS).mean())
                print(f"[{trace:9s} | {method:7s}] n={len(tpot):5d} "
                      f"mean={np.mean(tpot):6.2f} p50={np.median(tpot):6.2f} "
                      f"p99={np.percentile(tpot, 99):6.2f} ms  "
                      f"compliance(<=30ms)={comply*100:5.2f}%")
            else:
                agg = DYNAMO_AGG[trace]
                if agg is not None:
                    mean_ms, p99_ms = agg
                    ax.scatter([mean_ms], [0.5], color=COLORS[method],
                               marker="o", s=40, zorder=5)
                    ax.scatter([p99_ms], [0.99], color=COLORS[method],
                               marker="x", s=60, zorder=5)
                    print(f"[{trace:9s} | {method:7s}] per-request N/A; "
                          f"aggregate mean={mean_ms} p99={p99_ms} ms")

        ax.axvline(SLO_TPOT_MS, color="#CC3333", linestyle="--", lw=1.6,
                   label="SLO 30 ms")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.02)
        ax.set_title(trace, fontsize=12, fontweight="bold")
        ax.set_xlabel("Per-request TPOT (ms)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        if col == 0:
            ax.set_ylabel("CDF", fontsize=11)

    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle(
        "Llama-3-8B-Instruct + LMSYS, 2P6D — per-request TPOT CDF "
        "(SLO: TPOT ≤ 30 ms)",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_png = Path(__file__).with_suffix(".png")
    out_pdf = Path(__file__).with_suffix(".pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
