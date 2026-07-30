#!/usr/bin/env python3
"""Figure: Predictor latency hiding analysis (predict_latency_hiding.pdf).

Source: zrecent_use/0630/plot_predict_latency.py
Data:   experiment/wt_experiment/experiment_paper/4-Predict_latency/tmp/
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    "/root/predict-schedule/experiment/wt_experiment/experiment_paper/"
    "4-Predict_latency/tmp/predict_timing_2026-04-28_01-28-35.csv"
)

df = pd.read_csv(DATA_PATH)
metrics = ["predict_ms", "prefill_tail_ms"]
labels = ["Predictor Duration", "Prefill Duration"]
colors = ["#F0A35E", "#5B8FB9"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7.5,
    "axes.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

fig, ax = plt.subplots(figsize=(3.5, 2.0), facecolor="white")
y_positions = [0, 1]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    data = df[metric].dropna()
    p05 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)

    y = y_positions[i]
    ax.plot([p05, p95], [y, y], color="#333333", linewidth=1.2, zorder=1,
            solid_capstyle="butt")
    cap_h = 0.18
    ax.plot([p05, p05], [y - cap_h, y + cap_h], color="#333333",
            linewidth=1.2, zorder=1, solid_capstyle="butt")
    ax.plot([p95, p95], [y - cap_h, y + cap_h], color="#333333",
            linewidth=1.2, zorder=1, solid_capstyle="butt")
    box_h = 0.42
    rect = patches.FancyBboxPatch(
        (p25, y - box_h / 2), p75 - p25, box_h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="none", alpha=0.82, zorder=2,
    )
    ax.add_patch(rect)
    ax.plot([p50, p50], [y - box_h / 2, y + box_h / 2],
            color="white", linewidth=1.6, zorder=3, solid_capstyle="butt")

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Latency (ms)", fontsize=8, labelpad=3)
ax.set_xlim(8, 50)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.tick_params(axis="x", labelsize=7.5, length=2.5, width=0.5,
               direction="out", top=False)
ax.tick_params(axis="y", labelsize=8, length=0, pad=4)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#6A6A6A")
    ax.spines[spine].set_linewidth(0.55)
ax.grid(axis="x", color="#D4D4D4", linestyle="--", linewidth=0.35, alpha=0.6)
ax.set_axisbelow(True)
ax.invert_yaxis()

for i, metric in enumerate(metrics):
    data = df[metric].dropna()
    p50 = np.percentile(data, 50)
    y = y_positions[i]
    ax.annotate(
        f"{p50:.1f} ms",
        xy=(p50, y), xytext=(p50 + 3.5, y + 0.28),
        fontsize=7, color="#333333", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.5,
                        connectionstyle="arc3,rad=0.15"),
    )

for fmt, dpi_val in [("pdf", None), ("png", 300)]:
    path = SCRIPT_DIR / f"predict_latency_hiding.{fmt}"
    kwargs = dict(bbox_inches="tight", pad_inches=0.03)
    if dpi_val:
        kwargs["dpi"] = dpi_val
    fig.savefig(str(path), **kwargs)
    print(f"Saved {path}")

plt.close(fig)
