#!/usr/bin/env python3
"""Plot prefill(16->32) and predict timeline summary from timing CSV.

Input CSV columns expected:
- req_id
- predict_start_ts_ns
- prefill_end_ts_ns
- predict_end_ts_ns

The figure uses two lanes:
- Upper lane: prefill_16_32 intervals [predict_start_ts_ns, prefill_end_ts_ns]
- Lower lane: predict intervals [predict_start_ts_ns, predict_end_ts_ns]

Each lane is drawn with:
- a whisker-like |-|
- a colored concentration rectangle based on quantiles
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class IntervalData:
    starts: np.ndarray
    ends: np.ndarray


def _to_float_or_none(value: str) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_intervals(csv_path: str) -> Tuple[IntervalData, IntervalData, int, int]:
    prefill_starts: List[float] = []
    prefill_ends: List[float] = []
    predict_starts: List[float] = []
    predict_ends: List[float] = []

    total_rows = 0
    dropped_rows = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "predict_start_ts_ns",
            "prefill_end_ts_ns",
            "predict_end_ts_ns",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            ps = _to_float_or_none(row.get("predict_start_ts_ns", ""))
            pe = _to_float_or_none(row.get("prefill_end_ts_ns", ""))
            pd = _to_float_or_none(row.get("predict_end_ts_ns", ""))

            if ps is None or pe is None or pd is None:
                dropped_rows += 1
                continue

            # Guard against broken timestamps.
            if pe < ps or pd < ps:
                dropped_rows += 1
                continue

            prefill_starts.append(ps)
            prefill_ends.append(pe)
            predict_starts.append(ps)
            predict_ends.append(pd)

    if not prefill_starts:
        raise ValueError("No complete rows available after filtering.")

    prefill = IntervalData(np.array(prefill_starts), np.array(prefill_ends))
    predict = IntervalData(np.array(predict_starts), np.array(predict_ends))
    return prefill, predict, total_rows, dropped_rows


def _draw_lane(
    ax: plt.Axes,
    label: str,
    y: float,
    interval: IntervalData,
    color: str,
    lane_height: float = 0.28,
) -> Tuple[float, float, float, float]:
    # Global whisker.
    global_start = float(np.min(interval.starts))
    global_end = float(np.max(interval.ends))

    ax.hlines(y, global_start, global_end, color=color, linewidth=2.0, alpha=0.9)
    ax.vlines([global_start, global_end], y - lane_height / 2, y + lane_height / 2,
              color=color, linewidth=2.0, alpha=0.9)

    # Concentrated region via quantiles (middle 50%).
    q_start = float(np.quantile(interval.starts, 0.25))
    q_end = float(np.quantile(interval.ends, 0.75))
    if q_end < q_start:
        q_start, q_end = q_end, q_start

    rect = patches.Rectangle(
        (q_start, y - lane_height / 2),
        max(q_end - q_start, 1e-9),
        lane_height,
        facecolor=color,
        edgecolor="none",
        alpha=0.28,
        label=label,
    )
    ax.add_patch(rect)

    return global_start, global_end, q_start, q_end


def plot_timeline(
    csv_path: str,
    output_path: str,
    title: str,
    dpi: int,
) -> None:
    prefill, predict, total_rows, dropped_rows = load_intervals(csv_path)

    # Use ns as base and convert axis to ms relative time for readability.
    t0 = float(min(np.min(prefill.starts), np.min(predict.starts)))
    prefill_ms = IntervalData((prefill.starts - t0) / 1e6, (prefill.ends - t0) / 1e6)
    predict_ms = IntervalData((predict.starts - t0) / 1e6, (predict.ends - t0) / 1e6)

    fig, ax = plt.subplots(figsize=(13, 5.8))

    p_gs, p_ge, p_qs, p_qe = _draw_lane(
        ax,
        label="prefill_16_32 concentration",
        y=1.0,
        interval=prefill_ms,
        color="#1f77b4",
    )
    d_gs, d_ge, d_qs, d_qe = _draw_lane(
        ax,
        label="predict concentration",
        y=0.0,
        interval=predict_ms,
        color="#d62728",
    )

    # Coverage ratio: predict fully hidden in prefill_16_32.
    covered = np.sum(predict.ends <= prefill.ends)
    valid = len(prefill.starts)
    cover_ratio = covered / valid if valid > 0 else 0.0

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["predict", "prefill_16_32"], fontsize=12)
    ax.set_xlabel("Time (ms, relative to first request start)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    x_min = min(p_gs, d_gs)
    x_max = max(p_ge, d_ge)
    margin = (x_max - x_min) * 0.03 if x_max > x_min else 1.0
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(-0.6, 1.6)

    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    info = (
        f"valid rows: {valid}/{total_rows}  (dropped: {dropped_rows})\n"
        f"predict hidden by prefill_16_32: {cover_ratio:.1%}\n"
        f"prefill concentration: [{p_qs:.2f}, {p_qe:.2f}] ms\n"
        f"predict concentration: [{d_qs:.2f}, {d_qe:.2f}] ms"
    )
    ax.text(
        0.01,
        0.02,
        info,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prefill_16_32 vs predict timeline summary from timing CSV.")
    parser.add_argument("--input", required=True, help="Path to timing CSV file.")
    parser.add_argument("--output", required=True, help="Path to save output figure (png/pdf).")
    parser.add_argument(
        "--title",
        default="Prefill(16->32) vs Predict Timeline",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_timeline(args.input, args.output, args.title, args.dpi)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
