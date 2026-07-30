#!/usr/bin/env python3
"""Figure: Oracle-to-baseline goodput ratio heatmap (motivation_figure5.pdf).

Source: paper_motivation_experiment/image5/2-Loadbalance/plotloadbalance2.py
  (mean-ratio version using rep-aggregated pooling)
Data:   image5/2-Loadbalance/paper/1p3d/

Note: The original image6 script (plotloadbalance_mean_sigma_avg.py) required
baseline_307 data which is no longer available. This script uses the image5 data
(same benchmark experiments, different aggregation method) to produce an
equivalent ratio heatmap.
"""

import os
import re
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_PATH = SCRIPT_DIR / "data"

DATASETS = {"Lmsys-chat-1M": "lmsyschat", "Sharegpt": "mysharegpt"}
MODELS = {"Llama-3-8B-Instruct": "llama", "Qwen-2.5-7B-Instruct": "qwen"}
QPS_LIST = [6, 8, 10, 12]
METRICS = ["Output token goodput", "Total token goodput"]
V_MIN, V_MAX = 0.95, 2.48


def extract_val(log_path, metric):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = rf"{metric}.*?:\s+([\d.]+)"
        match = re.search(pattern, content)
        return float(match.group(1)) if match else None
    except Exception:
        return None


def get_avg(dirs_list, metric):
    """Pool all log values across dirs and return mean."""
    vals = []
    for d in dirs_list:
        log_files = glob.glob(os.path.join(d, "log", "*.log"))
        for log_f in log_files:
            v = extract_val(log_f, metric)
            if v is not None:
                vals.append(v)
    return np.mean(vals) if vals else None


def get_ratio_data():
    final_rows = []
    for model_label, model_dir_key in MODELS.items():
        for metric in METRICS:
            row_data = {"Model": model_label, "Metric": metric}
            for ds_label, ds_dir_key in DATASETS.items():
                target_dir = os.path.join(ROOT_PATH, f"1p3d_{model_dir_key}_{ds_dir_key}")
                for qps in QPS_LIST:
                    col_name = f"{ds_label}_{qps}"
                    rr_dirs = glob.glob(os.path.join(target_dir, f"*_rr{qps}_*_rr"))
                    opt_dirs = glob.glob(os.path.join(target_dir, f"*_rr{qps}_*_optimal"))
                    avg_rr = get_avg(rr_dirs, metric)
                    avg_opt = get_avg(opt_dirs, metric)
                    row_data[col_name] = avg_opt / avg_rr if avg_rr and avg_opt else np.nan
            final_rows.append(row_data)
    return pd.DataFrame(final_rows)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.8,
    })
    df = get_ratio_data()
    plot_cols = [f"{ds}_{qps}" for ds in DATASETS for qps in QPS_LIST]
    data_matrix = df[plot_cols].values
    n_rows, n_cols = data_matrix.shape

    p_1_0 = (1.0 - 0.95) / (2.48 - 0.95)
    p_1_5 = (1.5 - 0.95) / (2.48 - 0.95)
    nodes = [0.0, p_1_0, p_1_0 + 0.01, p_1_5, p_1_5 + 0.01, 1.0]
    colors = ["#e06666", "#f4cccc", "#fff2cc", "#f6b26b", "#d9ead3", "#38761d"]
    cmap = LinearSegmentedColormap.from_list("custom_split", list(zip(nodes, colors)))
    norm = Normalize(vmin=V_MIN, vmax=V_MAX)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    LABEL_WIDTH, CELL_H, CELL_W = 4.5, 0.8, 1.2

    ax.set_xlim(0, LABEL_WIDTH + n_cols * CELL_W)
    ax.set_ylim(-0.5, (n_rows + 2.5) * CELL_H)

    for r in range(n_rows):
        for c in range(n_cols):
            val = data_matrix[r, c]
            x = LABEL_WIDTH + c * CELL_W
            y = (n_rows - 1 - r) * CELL_H
            face_color = "#EFEFEF" if np.isnan(val) else cmap(norm(val))
            rect = plt.Rectangle((x, y), CELL_W, CELL_H, facecolor=face_color,
                                 edgecolor="white", lw=0.5)
            ax.add_patch(rect)
            if not np.isnan(val):
                ax.text(x + CELL_W / 2, y + CELL_H / 2, f"{val:.2f}",
                        ha="center", va="center", fontsize=10, fontweight="bold")

    line_x_end = LABEL_WIDTH + n_cols * CELL_W
    ax.hlines((n_rows + 2) * CELL_H, 0.2, line_x_end, color="black", lw=1.5)
    ax.hlines(n_rows * CELL_H, 0.2, line_x_end, color="black", lw=0.8)
    ax.hlines(0, 0.2, line_x_end, color="black", lw=1.5)

    header_y1, header_y2 = (n_rows + 1.1) * CELL_H, (n_rows + 0.3) * CELL_H
    ax.text(LABEL_WIDTH / 2, header_y1, "Dataset", ha="center", va="center", fontsize=10)
    ax.text(LABEL_WIDTH / 2, header_y2, "Req Per Sec (req/s)", ha="center", va="center", fontsize=10)

    for i, ds in enumerate(DATASETS.keys()):
        ds_center_x = LABEL_WIDTH + (i * 4 + 2) * CELL_W
        ax.text(ds_center_x, header_y1, ds, ha="center", va="center", fontsize=11)
        for j, qps in enumerate(QPS_LIST):
            ax.text(LABEL_WIDTH + (i * 4 + j + 0.5) * CELL_W, header_y2,
                    str(qps), ha="center", va="center", fontsize=10)

    for i, model in enumerate(MODELS.keys()):
        model_center_y = (n_rows - (i * 2 + 1)) * CELL_H
        ax.text(0.3, model_center_y, model, ha="left", va="center", fontsize=10)
        for j, metric in enumerate(METRICS):
            row_y = (n_rows - 1 - (i * 2 + j)) * CELL_H
            label = "Output Goodput" if "Output" in metric else "Total Goodput"
            ax.text(LABEL_WIDTH - 0.2, row_y + CELL_H / 2, label, ha="right",
                    va="center", fontsize=9)

    ax.axis("off")
    out = SCRIPT_DIR / "motivation_figure5.pdf"
    plt.savefig(str(out), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
