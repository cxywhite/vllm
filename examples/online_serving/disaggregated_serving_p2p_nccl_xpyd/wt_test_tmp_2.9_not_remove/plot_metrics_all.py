import os
import re
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# 全局配置
# ==========================================================

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
# 修改输出目录名称
OUT_DIR = Path(f"metric_all_{TIMESTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("vllm", "baseline_vllm"),
    ("dynamo", "baseline_dynamo"),
    ("ours", "e2e_auto"),
]

# 修改颜色配置
# vllm 改为翡翠绿，与 ours 的蓝色区分开
# dynamo 保持橙色
# ours 保持蓝色
METHOD_COLORS = {
    "vllm": "#50C878",   # Emerald Green (更绿，区分度更高)
    "dynamo": "#FFBE7A", # Orange
    "ours": "#82B0D2",   # Blue
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

# 注意：请根据实际情况替换这里的路径
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

# ==========================================================
# 正则表达式 (保持不变，匹配您提供的日志格式)
# ==========================================================

RR_RE = re.compile(r"rr(\d+)")

RE_MAP = {
    "out_gp": re.compile(r"Output token goodput \(tok/s\):\s+([\d.]+)"),
    "tot_gp": re.compile(r"Total token goodput \(tok/s\):\s+([\d.]+)"),
    "mean_ttft": re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)"),
    "p99_ttft": re.compile(r"P99 TTFT \(ms\):\s+([\d.]+)"),
    "mean_tpot": re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)"),
    "p99_tpot": re.compile(r"P99 TPOT \(ms\):\s+([\d.]+)"),
    "mean_e2e": re.compile(r"Mean E2EL \(ms\):\s+([\d.]+)"),
    "p99_e2e": re.compile(r"P99 E2EL \(ms\):\s+([\d.]+)"),
}

# ==========================================================
# 日志解析逻辑
# ==========================================================

def parse_log(log_path: Path):
    metrics = {k: None for k in RE_MAP}
    with open(log_path) as f:
        for line in f:
            for k, reg in RE_MAP.items():
                if metrics[k] is None:
                    m = reg.search(line)
                    if m:
                        metrics[k] = float(m.group(1))
    return metrics


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

        for mdir in method_dirs:
            for bench_dir in mdir.iterdir():
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

                metrics = parse_log(logs[0])
                data[method_name][qps] = metrics

    return data

# ==========================================================
# 绘图逻辑：单指标 2x4 柱状图
# ==========================================================

def plot_single_metric_2x4(metric_key,
                           ylabel,
                           filename,
                           slo=None):
    """
    绘制单个指标的 2x4 子图。
    所有方法都使用柱状图表示。
    """
    print(f"Generating plot for: {metric_key} -> {filename}")

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
    })

    fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharey="row")

    # 柱状图宽度设置
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

            # 加载数据
            base_dir = SUBPLOT_DIRS[(model, dataset, pd_ratio)]
            data = load_subplot_data(base_dir)

            qps_list = QPS_2P4D if pd_ratio == "2p4d" else QPS_2P6D
            x_base = x_2p4d if pd_ratio == "2p4d" else x_2p6d

            # 遍历三种方法绘制柱状图
            for i, (method, _) in enumerate(METHODS):
                vals = []
                for q in qps_list:
                    m = data[method].get(q, {})
                    val = m.get(metric_key, 0)
                    if val is None: val = 0
                    vals.append(val)
                
                # 计算偏移量，使柱子并排显示
                # i=0 (vllm) -> offset -0.22
                # i=1 (dynamo)-> offset 0
                # i=2 (ours) -> offset +0.22
                offset = (i - 1) * bar_width
                
                ax.bar(
                    x_base + offset,
                    vals,
                    width=bar_width,
                    color=METHOD_COLORS[method],
                    label=method if (row == 0 and col == 0) else None,
                    edgecolor='white', # 增加一点白色边框让柱子更清晰
                    linewidth=0.5
                )

            # 绘制 SLO 线 (红色实线)
            if slo is not None:
                ax.axhline(slo, color="red", linestyle="-", linewidth=2, label="SLO" if (row==0 and col==0) else None)

            # 设置 X 轴标签
            ax.set_xticks(x_base)
            ax.set_xticklabels(qps_list)

            # 仅在第一列设置 Y 轴标签
            if col == 0:
                ax.set_ylabel(ylabel)

            # 仅在第二行设置 X 轴标题
            if row == 1:
                ax.set_xlabel("QPS (req/s)")

            # 仅在第一行设置子图标题
            if row == 0:
                ax.set_title(
                    f"Dataset: {DATASET_DISPLAY[dataset]}\n{MODEL_DISPLAY[model]}"
                )
            
            # 添加网格以便于阅读
            ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 在左侧添加大标题表示 PD Ratio
    axes[0, 0].text(-0.45, 0.5, "2P4D", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", fontsize=16, fontweight='bold')
    axes[1, 0].text(-0.45, 0.5, "2P6D", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", fontsize=16, fontweight='bold')

    # 图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # 调整图例顺序，把 SLO 放到最后或者按照您的喜好
    fig.legend(handles, labels, loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_path = OUT_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================================
# 主执行流程：生成 7 张图
# ==========================================================

if __name__ == "__main__":
    
    # 1. Output Token Goodput (无 SLO)
    plot_single_metric_2x4(
        metric_key="out_gp",
        ylabel="Goodput (tokens/s)",
        filename="goodput.png",
        slo=None
    )

    # 2. Mean TTFT (SLO = 1000ms)
    plot_single_metric_2x4(
        metric_key="mean_ttft",
        ylabel="Mean TTFT (ms)",
        filename="ttft_mean.png",
        slo=1000
    )

    # 3. P99 TTFT (SLO = 1000ms)
    plot_single_metric_2x4(
        metric_key="p99_ttft",
        ylabel="P99 TTFT (ms)",
        filename="ttft_p99.png",
        slo=1000
    )

    # 4. Mean TPOT (SLO = 50ms)
    plot_single_metric_2x4(
        metric_key="mean_tpot",
        ylabel="Mean TPOT (ms)",
        filename="tpot_mean.png",
        slo=50
    )

    # 5. P99 TPOT (SLO = 50ms)
    plot_single_metric_2x4(
        metric_key="p99_tpot",
        ylabel="P99 TPOT (ms)",
        filename="tpot_p99.png",
        slo=50
    )

    # 6. Mean E2E (无 SLO)
    plot_single_metric_2x4(
        metric_key="mean_e2e",
        ylabel="Mean E2E Latency (ms)",
        filename="e2e_mean.png",
        slo=None
    )

    # 7. P99 E2E (无 SLO)
    plot_single_metric_2x4(
        metric_key="p99_e2e",
        ylabel="P99 E2E Latency (ms)",
        filename="e2e_p99.png",
        slo=None
    )

    print(f"\nAll 7 figures have been saved to directory: {OUT_DIR}")