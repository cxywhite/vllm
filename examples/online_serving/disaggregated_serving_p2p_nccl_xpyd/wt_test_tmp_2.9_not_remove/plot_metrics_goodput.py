import os
import re
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 全局配置
# =========================

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_PNG = f"goodput_comparison_{TIMESTAMP}.png"

METHODS = [
    ("vllm", "baseline_vllm"),
    ("dynamo", "baseline_dynamo"),
    ("ours", "e2e_auto"),
]

METHOD_COLORS = {
    "vllm": "#8ECFC9",
    "dynamo": "#FFBE7A",
    "ours": "#82B0D2",
}

MODEL_DISPLAY = {
    "llama": "llama3-8b-Instruct",
    "qwen": "qwen2.5-7b-Instruct",
}

DATASET_DISPLAY = {
    "sharegpt": "sharegpt",
    "lmsys": "lmsys-chat-1m",
}

QPS_2P4D = [4, 8, 12, 16, 20]
QPS_2P6D = [8, 12, 18, 24, 30]

SUBPLOT_DIRS = {
    ("llama", "sharegpt", "2p4d"):
        "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/llama_sharegpt_2p4d",
    ("qwen",  "sharegpt", "2p4d"): None,
    ("llama", "lmsys",    "2p4d"): None,
    ("qwen",  "lmsys",    "2p4d"): None,

    ("llama", "sharegpt", "2p6d"):
        "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/llama_sharegpt_2p6d",
    ("qwen",  "sharegpt", "2p6d"): None,
    ("llama", "lmsys",    "2p6d"): None,
    ("qwen",  "lmsys",    "2p6d"): None,
}

# =========================
# 正则
# =========================

OUTPUT_RE = re.compile(r"Output token goodput \(tok/s\):\s+([\d.]+)")
TOTAL_RE  = re.compile(r"Total token goodput \(tok/s\):\s+([\d.]+)")
RR_RE     = re.compile(r"_rr(\d+)_")

# =========================
# 日志解析
# =========================

def parse_log(log_path: Path):
    out_gp = tot_gp = None
    with open(log_path) as f:
        for line in f:
            if out_gp is None:
                m = OUTPUT_RE.search(line)
                if m:
                    out_gp = float(m.group(1))
            if tot_gp is None:
                m = TOTAL_RE.search(line)
                if m:
                    tot_gp = float(m.group(1))
    return out_gp, tot_gp


def load_subplot_data(base_dir: str):
    data = {m[0]: {} for m in METHODS}
    if base_dir is None:
        return data

    base_dir = Path(base_dir)
    if not base_dir.exists():
        return data

    for method_name, method_prefix in METHODS:
        method_dirs = [
            d for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith(method_prefix)
        ]
        if not method_dirs:
            continue

        for bench_dir in method_dirs[0].iterdir():
            if not bench_dir.is_dir():
                continue

            m = RR_RE.search(bench_dir.name)
            if not m:
                continue
            qps = int(m.group(1))

            log_dir = bench_dir / "log"
            logs = list(log_dir.glob("bench*.log"))
            if not logs:
                continue

            out_gp, tot_gp = parse_log(logs[0])
            if out_gp is not None:
                data[method_name][qps] = (out_gp, tot_gp)

    return data

# =========================
# 画图
# =========================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
})

fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey="row")

bar_width = 0.22
x_2p4d = np.arange(len(QPS_2P4D))
x_2p6d = np.arange(len(QPS_2P6D))

subplot_keys = [
    ("llama", "sharegpt"),
    ("qwen",  "sharegpt"),
    ("llama", "lmsys"),
    ("qwen",  "lmsys"),
]

for col, (model, dataset) in enumerate(subplot_keys):
    for row, pd_ratio in enumerate(["2p4d", "2p6d"]):
        ax = axes[row, col]

        base_dir = SUBPLOT_DIRS[(model, dataset, pd_ratio)]
        data = load_subplot_data(base_dir)

        qps_list = QPS_2P4D if pd_ratio == "2p4d" else QPS_2P6D
        x_base = x_2p4d if pd_ratio == "2p4d" else x_2p6d

        for i, (method, _) in enumerate(METHODS):
            out_vals, tot_vals = [], []
            for q in qps_list:
                v = data[method].get(q, (0, 0))
                out_vals.append(v[0])
                tot_vals.append(v[1])

            offset = (i - 1) * bar_width
            ax.bar(
                x_base + offset,
                out_vals,
                width=bar_width,
                color=METHOD_COLORS[method],
                label=method if (row == 0 and col == 0) else None,
            )
            ax.bar(
                x_base + offset,
                tot_vals,
                width=bar_width,
                fill=False,
                edgecolor=METHOD_COLORS[method],
                linestyle="--",
                linewidth=2,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(qps_list)

        if col == 0:
            ax.set_ylabel("Goodput (tokens/s)")

        if row == 1:
            ax.set_xlabel("QPS (req/s)")

        if row == 0:
            ax.set_title(
                f"Dataset: {DATASET_DISPLAY[dataset]}\n{MODEL_DISPLAY[model]}"
            )

# 行标签
axes[0, 0].text(-0.45, 0.5, "2P4D", transform=axes[0, 0].transAxes,
                rotation=90, va="center", fontsize=15)
axes[1, 0].text(-0.45, 0.5, "2P6D", transform=axes[1, 0].transAxes,
                rotation=90, va="center", fontsize=15)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print(f"Saved figure to {OUTPUT_PNG}")
