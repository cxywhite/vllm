#!/usr/bin/env python3
"""Figure: Cross-instance load imbalance under round-robin (motivation_figure3.pdf).

Source: paper_motivation_experiment/image4/plot_decode_load2.py
Data:   image4/qwen_qwen-mysharegpt_n1683_pd2p6d_tpot50ms_rep1_20260424_221312/
"""

import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = str(
    SCRIPT_DIR
    / "data"
    / "qwen_qwen-mysharegpt_n1683_pd2p6d_tpot50ms_rep1_20260424_221312"
    / "log"
    / "proxy_20260424_221312.log"
)

MAX_POINTS = 420
FIG_SIZE = (7.2, 2.35)
DPI = 300
MIN_Y_MAX = 1.05
Y_PAD_RATIO = 0.04

DECODE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

NUMBER_PATTERN = r"[+-]?(?:inf|\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
TIME_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
MONITOR_PATTERN = re.compile(
    r"decode=(?P<decode>[\d\.]+:\d+).*?"
    rf"(?:curr_bottle|bottle)=(?P<bottle>{NUMBER_PATTERN}).*?"
    rf"(?:curr_compute|compute)=(?P<compute>{NUMBER_PATTERN}).*?"
    rf"(?:curr_mem|memory)=(?P<memory>{NUMBER_PATTERN}).*?"
    rf"(?:curr_cap|capacity)=(?P<capacity>{NUMBER_PATTERN})",
    re.IGNORECASE,
)


def parse_log(log_path):
    data = defaultdict(lambda: {"time": [], "bottle": [], "compute": [], "memory": [], "capacity": []})
    last_time = None
    synthetic_index = 0
    with open(log_path) as f:
        for line in f:
            t = TIME_PATTERN.search(line)
            if t:
                last_time = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")
            if "[WT][MONITOR]" not in line:
                continue
            m = MONITOR_PATTERN.search(line)
            if not m:
                continue
            d = m.group("decode")
            if last_time is None:
                last_time = datetime.fromtimestamp(synthetic_index)
            data[d]["time"].append(last_time)
            data[d]["bottle"].append(float(m.group("bottle")))
            data[d]["compute"].append(float(m.group("compute")))
            data[d]["memory"].append(float(m.group("memory")))
            data[d]["capacity"].append(float(m.group("capacity")))
            synthetic_index += 1
    return data


def downsample(x, y, max_points):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return [x[i] for i in idx], [y[i] for i in idx]


def auto_ymax(values, min_ymax=MIN_Y_MAX, pad_ratio=Y_PAD_RATIO):
    if not values:
        return min_ymax
    return max(min_ymax, max(values) * (1 + pad_ratio))


def main():
    data = parse_log(LOG_PATH)
    decodes = sorted(data.keys(), key=lambda x: int(x.split(":")[-1]))
    if not decodes:
        raise SystemExit(f"No [WT][MONITOR] samples in {LOG_PATH}")
    decode_map = {d: f"decode-{i}" for i, d in enumerate(decodes)}

    fig, axis = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)
    formatter = mdates.DateFormatter("%M:%S")

    for i, d in enumerate(decodes):
        t, v = downsample(data[d]["time"], data[d]["bottle"], MAX_POINTS)
        axis.plot(t, v, linewidth=1.8, color=DECODE_COLORS[i % len(DECODE_COLORS)],
                  label=decode_map[d])

    axis.set_ylabel("Bottleneck load", fontsize=9)
    axis.set_xlabel("Time", fontsize=9)
    bottle_values = []
    for d in decodes:
        bottle_values.extend(data[d]["bottle"])
    axis.set_ylim(0, auto_ymax(bottle_values))
    axis.xaxis.set_major_formatter(formatter)
    axis.grid(True, color="#A0A0A0", alpha=0.22, linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4D4D4D")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                ncol=6, fontsize=8)

    fig.tight_layout(pad=0.25)
    out = SCRIPT_DIR / "motivation_figure3.pdf"
    fig.savefig(str(out), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
