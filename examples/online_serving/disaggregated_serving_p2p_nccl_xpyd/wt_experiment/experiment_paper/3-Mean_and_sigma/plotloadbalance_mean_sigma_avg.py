#!/usr/bin/env python3

import os
import re
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


# ================= Config =================
ROOT_PATH = (
    "/root/predict-schedule/vllm/examples/online_serving/"
    "disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/"
    "tmp/baseline_307"
)

DATASETS = {"Lmsys-chat-1M": "lmsyschat", "Sharegpt": "mysharegpt"}
MODELS = {"Llama-3-8b-Instruct": "llama", "Qwen-2.5-7b-Instruct": "qwen"}

# Use paired rep-to-rep matching and mean(opt/rr) for each request rate.
QPS_LIST = [6, 8, 10, 12]
METRICS = ["Output token goodput", "Total token goodput"]

# Color map range
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


def avg_metric_from_benchmark_dir(benchmark_dir, metric):
    log_files = glob.glob(os.path.join(benchmark_dir, "log", "*.log"))
    vals = []
    for log_f in log_files:
        v = extract_val(log_f, metric)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def extract_rep_id(path_or_name):
    name = os.path.basename(path_or_name.rstrip("/"))
    m = re.search(r"_rep(\d+)_", name)
    return m.group(1) if m else None


def paired_mean_ratio_by_qps(metric, qps, rr_dir, opt_dir):
    rr_dirs = glob.glob(os.path.join(rr_dir, f"*_rr{qps}_*_rr"))
    opt_dirs = glob.glob(os.path.join(opt_dir, f"*_rr{qps}_*_optimal"))

    rr_by_rep = {}
    opt_by_rep = {}

    for d in rr_dirs:
        rep_id = extract_rep_id(d)
        if rep_id is not None:
            rr_by_rep[rep_id] = d

    for d in opt_dirs:
        rep_id = extract_rep_id(d)
        if rep_id is not None:
            opt_by_rep[rep_id] = d

    ratios = []
    for rep_id in sorted(set(rr_by_rep.keys()) & set(opt_by_rep.keys())):
        rr_val = avg_metric_from_benchmark_dir(rr_by_rep[rep_id], metric)
        opt_val = avg_metric_from_benchmark_dir(opt_by_rep[rep_id], metric)
        if rr_val and opt_val:
            ratios.append(opt_val / rr_val)

    # Expect 3 reps; if fewer/more are available, average over all matched reps.
    return float(np.mean(ratios)) if ratios else np.nan


def get_combo_dirs(model_dir_key, ds_dir_key):
    base_dir = os.path.join(ROOT_PATH, f"1p3d_{model_dir_key}_{ds_dir_key}_mean_and_sigma")
    rr_dir = os.path.join(base_dir, "rr")
    opt_dir = os.path.join(base_dir, "optimal")
    if not os.path.isdir(rr_dir) or not os.path.isdir(opt_dir):
        return None, None
    return rr_dir, opt_dir


def get_ratio_data():
    final_rows = []

    for model_label, model_dir_key in MODELS.items():
        for metric in METRICS:
            row_data = {"Model": model_label, "Metric": metric}

            for ds_label, ds_dir_key in DATASETS.items():
                rr_dir, opt_dir = get_combo_dirs(model_dir_key, ds_dir_key)
                for qps in QPS_LIST:
                    col_name = f"{ds_label}_{qps}"
                    if rr_dir is None or opt_dir is None:
                        row_data[col_name] = np.nan
                    else:
                        row_data[col_name] = paired_mean_ratio_by_qps(metric, qps, rr_dir, opt_dir)

            final_rows.append(row_data)

    return pd.DataFrame(final_rows)


def draw_tight_academic_table(df):
    plot_cols = [f"{ds}_{qps}" for ds in DATASETS for qps in QPS_LIST]
    data_matrix = df[plot_cols].values
    n_rows, n_cols = data_matrix.shape

    p_1_0 = (1.0 - 0.95) / (2.48 - 0.95)
    p_1_5 = (1.5 - 0.95) / (2.48 - 0.95)

    nodes = [0.0, p_1_0, p_1_0 + 0.01, p_1_5, p_1_5 + 0.01, 1.0]
    colors = [
        "#e06666",
        "#f4cccc",
        "#fff2cc",
        "#f6b26b",
        "#d9ead3",
        "#38761d",
    ]
    cmap = LinearSegmentedColormap.from_list("custom_split", list(zip(nodes, colors)))
    norm = Normalize(vmin=V_MIN, vmax=V_MAX)

    fig, ax = plt.subplots(figsize=(12, 4))
    label_width = 4.5
    cell_h, cell_w = 0.8, 1.2

    ax.set_xlim(0, label_width + n_cols * cell_w)
    ax.set_ylim(-0.5, (n_rows + 2.5) * cell_h)

    for r in range(n_rows):
        for c in range(n_cols):
            val = data_matrix[r, c]
            x = label_width + c * cell_w
            y = (n_rows - 1 - r) * cell_h
            face_color = "#EFEFEF" if np.isnan(val) else cmap(norm(val))
            rect = plt.Rectangle((x, y), cell_w, cell_h, facecolor=face_color, edgecolor="white", lw=0.5)
            ax.add_patch(rect)
            if not np.isnan(val):
                ax.text(
                    x + cell_w / 2,
                    y + cell_h / 2,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    line_x_end = label_width + n_cols * cell_w
    ax.hlines((n_rows + 2) * cell_h, 0.2, line_x_end, color="black", lw=1.5)
    ax.hlines(n_rows * cell_h, 0.2, line_x_end, color="black", lw=0.8)
    ax.hlines(0, 0.2, line_x_end, color="black", lw=1.5)

    header_y1, header_y2 = (n_rows + 1.1) * cell_h, (n_rows + 0.3) * cell_h
    ax.text(label_width / 2, header_y1, "Dataset", ha="center", va="center", fontsize=10)
    ax.text(label_width / 2, header_y2, "Req Per Sec (req/s)", ha="center", va="center", fontsize=10)

    for i, ds in enumerate(DATASETS.keys()):
        ds_center_x = label_width + (i * len(QPS_LIST) + len(QPS_LIST) / 2) * cell_w
        ax.text(ds_center_x, header_y1, ds, ha="center", va="center", fontsize=11)
        for j, qps in enumerate(QPS_LIST):
            ax.text(
                label_width + (i * len(QPS_LIST) + j + 0.5) * cell_w,
                header_y2,
                str(qps),
                ha="center",
                va="center",
                fontsize=10,
            )

    for i, model in enumerate(MODELS.keys()):
        model_center_y = (n_rows - (i * len(METRICS) + 1)) * cell_h
        ax.text(0.3, model_center_y, model, ha="left", va="center", fontsize=10)
        for j, metric in enumerate(METRICS):
            row_y = (n_rows - 1 - (i * len(METRICS) + j)) * cell_h
            label = "Output Goodput" if "Output" in metric else "Total Goodput"
            ax.text(label_width - 0.2, row_y + cell_h / 2, label, ha="right", va="center", fontsize=9)

    ax.axis("off")
    save_dir = (
        "/root/predict-schedule/vllm/examples/online_serving/"
        "disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/Mean_and_sigma/fig"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"table_three_color_repmean_{datetime.now().strftime('%H%M%S')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Table generated: {save_path}")


if __name__ == "__main__":
    df = get_ratio_data()
    draw_tight_academic_table(df)
