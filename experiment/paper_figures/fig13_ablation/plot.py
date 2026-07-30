#!/usr/bin/env python3
"""Figure: Inter-instance scheduling ablation (inter_scheduling_goodput.pdf).

Source: zrecent_use/0630/plot_ablation.py
Data:   hardcoded in script
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

methods = ["Round-Robin", "Current-Capacity", "Current-3DLoad", "Predict-3DLoad"]
goodputs = [1861.77, 2146.95, 2154.81, 2861.0]
short_names = ["RR", "Curr-Cap", "Curr-3DL", "Pred-3DL"]
bar_colors = ["#73C58C", "#F0A35E", "#5B8FB9", "#2ca02c"]

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

fig, ax = plt.subplots(figsize=(3.5, 2.2))

bars = ax.bar(short_names, goodputs, color=bar_colors, edgecolor="white",
              linewidth=0.8, width=0.52, zorder=3)

for bar, val in zip(bars, goodputs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7.5, color="#111111")

ax.set_ylabel("Output Token Goodput (tok/s)", fontsize=8.5, labelpad=4)
ax.grid(True, alpha=0.2, axis="y", linestyle="--", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#AAAAAA")
ax.spines["bottom"].set_color("#AAAAAA")
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(color="#666666", labelsize=7.5)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, max(goodputs) * 1.25)

plt.tight_layout(pad=0.8)

from pathlib import Path
out = Path(__file__).resolve().parent / "inter_scheduling_goodput.pdf"
plt.savefig(str(out), bbox_inches="tight", pad_inches=0.02)
print(f"Saved: {out}")
plt.close()
