#!/usr/bin/env python3
"""Figure: Token composition goodput/throughput (motivation_figure6 b-d, f-h).

Source: paper_motivation_experiment/image11-16/draw.py
Data:   image11-16/160k_llama/
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data" / "160k_llama"
DPI = 300
FIG_SIZE = (3.15, 1.88)

SCENARIOS = [
    ("gpu_input", "Input-heavy"),
    ("gpu_output", "Output-heavy"),
    ("gpu_equal", "Balanced"),
]
UTILIZATIONS = ["0.3", "0.8"]

METRIC_PATTERNS = {
    "batchsize": re.compile(r"Successful requests:\s+([0-9.]+)"),
    "output_goodput": re.compile(r"Output token goodput \(tok/s\):\s+([0-9.]+)"),
    "total_goodput": re.compile(r"Total token goodput \(tok/s\):\s+([0-9.]+)"),
    "output_throughput": re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
    "total_throughput": re.compile(r"Total Token throughput \(tok/s\):\s+([0-9.]+)"),
}

# Map (scenario_key, util) to motivation_figure6 subfigure letter
PANEL_MAP = {
    ("gpu_input", "0.3"):  "b",  # input-heavy, 0.3
    ("gpu_equal", "0.3"):  "c",  # balanced,    0.3
    ("gpu_output", "0.3"): "d",  # output-heavy, 0.3
    ("gpu_input", "0.8"):  "f",  # input-heavy, 0.8
    ("gpu_equal", "0.8"):  "g",  # balanced,    0.8
    ("gpu_output", "0.8"): "h",  # output-heavy, 0.8
}


def parse_metric(content, metric_name):
    match = METRIC_PATTERNS[metric_name].search(content)
    return float(match.group(1)) if match else np.nan


def parse_benchmark_log(log_path):
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    batchsize = parse_metric(content, "batchsize")
    if np.isnan(batchsize):
        file_match = re.search(r"benchmark_np(\d+)_", log_path.name)
        batchsize = float(file_match.group(1)) if file_match else np.nan
    return {
        "batchsize": int(batchsize),
        "output_goodput": parse_metric(content, "output_goodput"),
        "total_goodput": parse_metric(content, "total_goodput"),
        "output_throughput": parse_metric(content, "output_throughput"),
        "total_throughput": parse_metric(content, "total_throughput"),
    }


def load_case(case_dir):
    rows = []
    for log_path in sorted(case_dir.glob("benchmark_np*.log")):
        row = parse_benchmark_log(log_path)
        if not np.isnan(row["batchsize"]) and row["batchsize"] != 480:
            rows.append(row)
    return sorted(rows, key=lambda row: row["batchsize"])


def draw_case(rows, title, output_path, y_limit):
    batchsizes = np.array([row["batchsize"] for row in rows])
    output_goodput = np.array([row["output_goodput"] for row in rows])
    output_throughput = np.array([row["output_throughput"] for row in rows])

    x_positions = np.arange(len(batchsizes))
    fig, axis = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    axis.bar(x_positions, output_goodput, width=0.68, color="#4E79A7",
             edgecolor="#2F4B66", linewidth=0.45, label="Goodput", zorder=2)
    axis.plot(x_positions, output_throughput, color="#D55E00", marker="o",
              markersize=2.4, linewidth=1.15, label="Throughput", zorder=3)

    axis.set_title(title, fontsize=8.0, pad=2.0)
    axis.set_xlabel("Batch size", fontsize=7.1, labelpad=1.0)
    axis.set_ylabel("Output tok/s", fontsize=7.1, labelpad=1.0)
    axis.set_ylim(0, y_limit)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(v) for v in batchsizes], rotation=30, ha="right")

    axis.grid(axis="y", color="#A0A0A0", alpha=0.22, linewidth=0.7)
    axis.tick_params(axis="both", labelsize=6.4, width=0.8, length=2.4)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4D4D4D")

    legend = axis.legend(frameon=True, loc="upper right", fontsize=5.8,
                         handlelength=1.25, borderaxespad=0.2, borderpad=0.2,
                         labelspacing=0.25)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.78)

    fig.tight_layout(pad=0.2)
    fig.savefig(str(output_path), dpi=DPI, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    cases_data = {}
    for util in UTILIZATIONS:
        for scenario_key, scenario_label in SCENARIOS:
            case_dir = DATA_ROOT / scenario_key / util
            rows = load_case(case_dir)
            if not rows:
                raise RuntimeError(f"No benchmark logs under {case_dir}")
            cases_data[(scenario_key, util)] = (scenario_label, rows)

    max_value = 0.0
    for _, rows in cases_data.values():
        for row in rows:
            max_value = max(max_value, row["output_goodput"], row["output_throughput"])
    y_limit = max_value * 1.18

    for (scenario_key, util), (scenario_label, rows) in cases_data.items():
        panel = PANEL_MAP[(scenario_key, util)]
        title = f"({panel}) {scenario_label}, mem util={util}"
        out_name = f"motivation_figure6{panel}.png"
        draw_case(rows, title, SCRIPT_DIR / out_name, y_limit)
        print(f"Saved {SCRIPT_DIR / out_name}")


if __name__ == "__main__":
    main()
