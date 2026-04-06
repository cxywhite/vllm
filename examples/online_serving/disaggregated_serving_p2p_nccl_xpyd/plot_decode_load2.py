import os
import re
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse


MAX_POINTS = 300
FIG_SIZE = (12, 6)
DPI = 150
MIN_Y_MAX = 1.05
Y_PAD_RATIO = 0.08

LINESTYLES = {
    "compute": "-",
    "memory": "--",
    "capacity": ":",
}

TIME_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
MONITOR_PATTERN = re.compile(
    r"decode=(?P<decode>[\d\.]+:\d+).*?"
    r"bottle=(?P<bottle>[\d\.]+).*?"
    r"compute=(?P<compute>[\d\.]+).*?"
    r"memory=(?P<memory>[\d\.]+).*?"
    r"capacity=(?P<capacity>[\d\.]+)"
)


def parse_log(log_path):
    data = defaultdict(lambda: {
        "time": [],
        "bottle": [],
        "compute": [],
        "memory": [],
        "capacity": [],
    })

    last_time = None
    with open(log_path) as f:
        for line in f:
            t = TIME_PATTERN.search(line)
            if t:
                last_time = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")

            if "[WT][MONITOR]" not in line or last_time is None:
                continue

            m = MONITOR_PATTERN.search(line)
            if not m:
                continue

            d = m.group("decode")
            data[d]["time"].append(last_time)
            data[d]["bottle"].append(float(m.group("bottle")))
            data[d]["compute"].append(float(m.group("compute")))
            data[d]["memory"].append(float(m.group("memory")))
            data[d]["capacity"].append(float(m.group("capacity")))

    return data


def downsample(x, y, max_points):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return [x[i] for i in idx], [y[i] for i in idx]


def auto_ymax(values, min_ymax=MIN_Y_MAX, pad_ratio=Y_PAD_RATIO):
    if not values:
        return min_ymax

    vmax = max(values)
    return max(min_ymax, vmax * (1 + pad_ratio))


# =======================
# 图 1
# =======================
def plot_figure_1(data, decode_map, decodes, log_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE, dpi=DPI, sharex=True)
    formatter = mdates.DateFormatter("%M:%S")

    target = decodes[0]

    for metric, style in LINESTYLES.items():
        t, v = downsample(data[target]["time"], data[target][metric], MAX_POINTS)
        ax1.plot(t, v, linestyle=style, linewidth=2, label=metric)

    ax1.set_ylabel("Load", fontsize=9)
    load_values = []
    for metric in LINESTYLES:
        load_values.extend(data[target][metric])
    ax1.set_ylim(0, auto_ymax(load_values))
    ax1.grid(True, alpha=0.4)
    ax1.legend(frameon=False)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.text(
        0.5, -0.32,
        f"(a) Load Dynamics of {decode_map[target]}",
        transform=ax1.transAxes,
        fontsize=10,
        ha="center"
    )

    for d in decodes:
        t, v = downsample(data[d]["time"], data[d]["bottle"], MAX_POINTS)
        ax2.plot(t, v, linewidth=2, label=decode_map[d])

    ax2.set_ylabel("Bottle", fontsize=9)
    bottle_values = []
    for d in decodes:
        bottle_values.extend(data[d]["bottle"])
    ax2.set_ylim(0, auto_ymax(bottle_values))
    ax2.grid(True, alpha=0.4)
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_xlabel("Time", fontsize=9)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.text(
        0.5, -0.32,
        "(b) Bottle Load Across Decode Instances",
        transform=ax2.transAxes,
        fontsize=10,
        ha="center"
    )

    plt.tight_layout()
    plt.savefig(log_path.replace(".log", "_fig1.png"))
    plt.close()


# =======================
# 图 2
# =======================
def plot_figure_2(data, decode_map, decodes, log_path):
    n = len(decodes)
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 3 * (n + 1)), dpi=DPI, sharex=True)
    formatter = mdates.DateFormatter("%M:%S")

    for i, d in enumerate(decodes):
        ax = axes[i]
        for metric, style in LINESTYLES.items():
            t, v = downsample(data[d]["time"], data[d][metric], MAX_POINTS)
            ax.plot(t, v, linestyle=style, linewidth=2, label=metric)

        ax.set_ylabel("Load", fontsize=9)
        load_values = []
        for metric in LINESTYLES:
            load_values.extend(data[d][metric])
        ax.set_ylim(0, auto_ymax(load_values))
        ax.grid(True, alpha=0.4)
        ax.legend(frameon=False)
        ax.tick_params(axis="x", labelbottom=False)
        ax.text(
            0.5, -0.32,
            f"({chr(97+i)}) Load Dynamics of {decode_map[d]}",
            transform=ax.transAxes,
            fontsize=10,
            ha="center"
        )

    ax = axes[-1]
    for d in decodes:
        t, v = downsample(data[d]["time"], data[d]["bottle"], MAX_POINTS)
        ax.plot(t, v, linewidth=2, label=decode_map[d])

    ax.set_ylabel("Bottle", fontsize=9)
    bottle_values = []
    for d in decodes:
        bottle_values.extend(data[d]["bottle"])
    ax.set_ylim(0, auto_ymax(bottle_values))
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("Time", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_formatter(formatter)
    ax.text(
        0.5, -0.32,
        f"({chr(97+n)}) Bottle Load Across Decode Instances",
        transform=ax.transAxes,
        fontsize=10,
        ha="center"
    )

    plt.tight_layout()
    plt.savefig(log_path.replace(".log", "_fig2.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True)
    args = parser.parse_args()

    data = parse_log(args.log_path)
    decodes = sorted(data.keys(), key=lambda x: int(x.split(":")[-1]))
    decode_map = {d: f"decode-{i}" for i, d in enumerate(decodes)}

    plot_figure_1(data, decode_map, decodes, args.log_path)
    plot_figure_2(data, decode_map, decodes, args.log_path)


if __name__ == "__main__":
    main()
