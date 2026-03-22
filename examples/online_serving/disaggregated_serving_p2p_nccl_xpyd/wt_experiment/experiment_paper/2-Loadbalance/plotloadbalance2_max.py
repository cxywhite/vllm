import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ================= 配置区 =================
ROOT_PATH = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/Loadbalance/paper/1p3d"
DATASETS = {"Lmsys-chat-1M": "lmsyschat", "Sharegpt": "mysharegpt"}
MODELS = {"Llama-3-8b-Instruct": "llama", "Qwen-2.5-7b-Instruct": "qwen"}
QPS_LIST = [6, 8, 10, 12]
METRICS = ["Output token goodput", "Total token goodput"]

# 颜色映射范围
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


def extract_rep_id(path_or_name):
    name = os.path.basename(path_or_name.rstrip("/"))
    m = re.search(r"_rep(\d+)_", name)
    return m.group(1) if m else None


def avg_metric_from_one_dir(target_dir, metric):
    vals = []
    log_files = glob.glob(os.path.join(target_dir, "log", "*.log"))
    for log_f in log_files:
        v = extract_val(log_f, metric)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def max_ratio_by_qps(metric, qps, target_dir):
    rr_dirs = glob.glob(os.path.join(target_dir, f"*_rr{qps}_*_rr"))
    opt_dirs = glob.glob(os.path.join(target_dir, f"*_rr{qps}_*_optimal"))

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
    matched_reps = sorted(set(rr_by_rep.keys()) & set(opt_by_rep.keys()))

    for rep_id in matched_reps:
        rr_val = avg_metric_from_one_dir(rr_by_rep[rep_id], metric)
        opt_val = avg_metric_from_one_dir(opt_by_rep[rep_id], metric)
        if rr_val and opt_val:
            ratios.append(opt_val / rr_val)

    # 回退逻辑：若目录名没有rep编号，使用原脚本的聚合方式做每次目录配对比值
    if not ratios and rr_dirs and opt_dirs:
        rr_vals = []
        opt_vals = []
        for d in rr_dirs:
            rr_val = avg_metric_from_one_dir(d, metric)
            if rr_val:
                rr_vals.append(rr_val)
        for d in opt_dirs:
            opt_val = avg_metric_from_one_dir(d, metric)
            if opt_val:
                opt_vals.append(opt_val)
        for rr_val, opt_val in zip(sorted(rr_vals), sorted(opt_vals)):
            ratios.append(opt_val / rr_val)

    return max(ratios) if ratios else np.nan


def get_ratio_data():
    final_rows = []
    for model_label, model_dir_key in MODELS.items():
        for metric in METRICS:
            row_data = {"Model": model_label, "Metric": metric}
            for ds_label, ds_dir_key in DATASETS.items():
                target_dir = os.path.join(ROOT_PATH, f"1p3d_{model_dir_key}_{ds_dir_key}")
                for qps in QPS_LIST:
                    col_name = f"{ds_label}_{qps}"
                    row_data[col_name] = max_ratio_by_qps(metric, qps, target_dir)
            final_rows.append(row_data)
    return pd.DataFrame(final_rows)


def draw_tight_academic_table(df):
    plot_cols = [f"{ds}_{qps}" for ds in DATASETS for qps in QPS_LIST]
    data_matrix = df[plot_cols].values
    n_rows, n_cols = data_matrix.shape

    # --- 核心修改：三段式颜色映射 ---
    # 计算关键位置在 0.95-2.48 归一化轴上的点
    # pos = (val - v_min) / (v_max - v_min)
    p_1_0 = (1.0 - 0.95) / (2.48 - 0.95)  # 约 0.0326
    p_1_5 = (1.5 - 0.95) / (2.48 - 0.95)  # 约 0.3595

    nodes = [0.0, p_1_0, p_1_0 + 0.01, p_1_5, p_1_5 + 0.01, 1.0]
    colors = [
        "#e06666",  # 0.95: 深红
        "#f4cccc",  # 1.0直前: 浅红
        "#fff2cc",  # 1.0直后: 浅橘
        "#f6b26b",  # 1.5直前: 深橘
        "#d9ead3",  # 1.5直后: 浅绿
        "#38761d",  # 2.48: 中深绿 (比之前最深的 #004400 浅一些)
    ]
    cmap = LinearSegmentedColormap.from_list("custom_split", list(zip(nodes, colors)))
    norm = Normalize(vmin=V_MIN, vmax=V_MAX)

    fig, ax = plt.subplots(figsize=(12, 4))
    LABEL_WIDTH = 4.5
    CELL_H, CELL_W = 0.8, 1.2

    ax.set_xlim(0, LABEL_WIDTH + n_cols * CELL_W)
    ax.set_ylim(-0.5, (n_rows + 2.5) * CELL_H)

    # 1. 绘制单元格
    for r in range(n_rows):
        for c in range(n_cols):
            val = data_matrix[r, c]
            x = LABEL_WIDTH + c * CELL_W
            y = (n_rows - 1 - r) * CELL_H
            face_color = "#EFEFEF" if np.isnan(val) else cmap(norm(val))
            rect = plt.Rectangle((x, y), CELL_W, CELL_H, facecolor=face_color, edgecolor="white", lw=0.5)
            ax.add_patch(rect)
            if not np.isnan(val):
                ax.text(
                    x + CELL_W / 2,
                    y + CELL_H / 2,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    # 2. 绘制三线表线条
    line_x_end = LABEL_WIDTH + n_cols * CELL_W
    ax.hlines((n_rows + 2) * CELL_H, 0.2, line_x_end, color="black", lw=1.5)
    ax.hlines(n_rows * CELL_H, 0.2, line_x_end, color="black", lw=0.8)
    ax.hlines(0, 0.2, line_x_end, color="black", lw=1.5)

    # 3. 填写表头
    header_y1, header_y2 = (n_rows + 1.1) * CELL_H, (n_rows + 0.3) * CELL_H
    ax.text(LABEL_WIDTH / 2, header_y1, "Dataset", ha="center", va="center", fontsize=10)
    ax.text(LABEL_WIDTH / 2, header_y2, "Req Per Sec (req/s)", ha="center", va="center", fontsize=10)

    for i, ds in enumerate(DATASETS.keys()):
        ds_center_x = LABEL_WIDTH + (i * 4 + 2) * CELL_W
        ax.text(ds_center_x, header_y1, ds, ha="center", va="center", fontsize=11)
        for j, qps in enumerate(QPS_LIST):
            ax.text(LABEL_WIDTH + (i * 4 + j + 0.5) * CELL_W, header_y2, str(qps), ha="center", va="center", fontsize=10)

    # 4. 填写模型名称与指标 (修复垂直居中)
    for i, model in enumerate(MODELS.keys()):
        model_center_y = (n_rows - (i * 2 + 1)) * CELL_H
        ax.text(0.3, model_center_y, model, ha="left", va="center", fontsize=10)
        for j, metric in enumerate(METRICS):
            row_y = (n_rows - 1 - (i * 2 + j)) * CELL_H
            label = "Output Goodput" if "Output" in metric else "Total Goodput"
            ax.text(LABEL_WIDTH - 0.2, row_y + CELL_H / 2, label, ha="right", va="center", fontsize=9)

    ax.axis("off")
    save_dir = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/Loadbalance/paper/fig"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"table_three_color_max_{datetime.now().strftime('%H%M%S')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"表格已生成: {save_path}")


if __name__ == "__main__":
    df = get_ratio_data()
    draw_tight_academic_table(df)
