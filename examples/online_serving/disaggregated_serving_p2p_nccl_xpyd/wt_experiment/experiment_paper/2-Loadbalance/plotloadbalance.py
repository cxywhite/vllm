import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- 配置路径 ---
base_dir = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/tmp/baseline_302/1p3d_qwen_lmsyschat_qwen-lmsys-chat-loadbalance_1p3d_good"
save_dir = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/fig' 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
test_model = "qwen"  # 可选 "llama" 或 "qwen"
test_dataset = "lmsys-chat"  # 可选 "lmsys-chat"
# parser = argparse.ArgumentParser(description="Parse benchmark logs and plot load balance figures.")
# parser.add_argument("--test_model", choices=["llama", "qwen"], required=True,
#                     help="Test model type: llama or qwen")
# parser.add_argument("--test_dataset", choices=["lmsys-chat", "mysharegpt"], required=True,
#                     help="Test dataset: lmsys-chat or mysharegpt")
# args = parser.parse_args()

# test_model = args.test_model
# test_dataset = args.test_dataset
# --- 1. 数据解析 ---
def parse_log(file_path):
    metrics = {}
    patterns = {
        'output_goodput': r"Output token goodput \(tok/s\):\s+([\d.]+)",
        'total_goodput': r"Total token goodput \(tok/s\):\s+([\d.]+)",
        'mean_ttft': r"Mean TTFT \(ms\):\s+([\d.]+)",
        'p99_ttft': r"P99 TTFT \(ms\):\s+([\d.]+)",
        'mean_tpot': r"Mean TPOT \(ms\):\s+([\d.]+)",
        'p99_tpot': r"P99 TPOT \(ms\):\s+([\d.]+)",
        'mean_e2el': r"Mean E2EL \(ms\):\s+([\d.]+)",
        'p99_e2el': r"P99 E2EL \(ms\):\s+([\d.]+)"
    }
    
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))
    return metrics

# --- 2. 收集与聚合数据 ---
data_list = []
if os.path.exists(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("benchmark_")]
    for folder in subdirs:
        parts = folder.split('_')
        try:
            qps_str = [p for p in parts if p.startswith('rr') and p[2:].isdigit()][0]
            qps = int(qps_str[2:])
            method = parts[-1] 
            rep = [p for p in parts if p.startswith('rep')][0]

            if test_model == "llama":
                log_file_name = f"bench_np1000_rr{qps}_mt8192_{rep}.log"
            else:
                log_file_name = f"bench_np1000_rr{qps}_mt32768_{rep}.log"
            log_path = os.path.join(base_dir, folder, "log", log_file_name)
            
            metrics = parse_log(log_path)
            if metrics:
                metrics.update({'qps': qps, 'method': method, 'rep': rep})
                data_list.append(metrics)
        except:
            continue

df = pd.DataFrame(data_list)
if not df.empty:
    df_avg = df.groupby(['method', 'qps']).mean(numeric_only=True).reset_index()
    qps_list = sorted(df_avg['qps'].unique())
    methods = ['rr', 'optimal']
    colors = {'rr': '#99ccff', 'optimal': "#daa67b"} 
else:
    df_avg = pd.DataFrame()

# --- 3. 绘制 Goodput 图 (图一) ---
def plot_goodput():
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.3
    x = np.arange(len(qps_list))

    for i, method in enumerate(methods):
        method_data = df_avg[df_avg['method'] == method].set_index('qps').reindex(qps_list)
        pos = x + (i - 0.5) * bar_width
        
        # 实心柱体，边线颜色设为柱体颜色
        ax.bar(pos, method_data['output_goodput'], width=bar_width, 
                label=f'{method} Output', color=colors[method], 
                edgecolor=colors[method], linewidth=1)
        
        # 空心柱体，边线颜色设为柱体颜色
        ax.bar(pos, method_data['total_goodput'], width=bar_width, 
                label=f'{method} Total', color='none', 
                edgecolor=colors[method], linewidth=1.5, linestyle='--')

    ax.set_xlabel('QPS (req/s)', fontsize=12)
    ax.set_ylabel('Goodput (tokens/s)', fontsize=12)
    ax.set_title('Goodput Comparison: RR vs Optimal', fontsize=14, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(qps_list)
    
    # 增加顶部留白并放置图例
    ax.margins(y=0.3)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"goodput_analysis_{timestamp}_{test_model}_{test_dataset}.png"), dpi=300)

# --- 4. 绘制 Latency 六子图 (图二) ---
def plot_latency_metrics():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_config = [
        ('p99_ttft', 'P99 TTFT (ms)', 0, 0, 1000, 'TTFT SLO'),
        ('p99_tpot', 'P99 TPOT (ms)', 0, 1, 50, 'TPOT SLO'),
        ('p99_e2el', 'P99 E2EL (ms)', 0, 2, None, None),
        ('mean_ttft', 'Mean TTFT (ms)', 1, 0, 1000, 'TTFT SLO'),
        ('mean_tpot', 'Mean TPOT (ms)', 1, 1, 50, 'TPOT SLO'),
        ('mean_e2el', 'Mean E2EL (ms)', 1, 2, None, None)
    ]

    bar_width = 0.3
    x = np.arange(len(qps_list))

    for m_key, m_label, row, col, threshold, slo_name in metrics_config:
        ax = axes[row, col]
        for i, method in enumerate(methods):
            method_data = df_avg[df_avg['method'] == method].set_index('qps').reindex(qps_list)
            pos = x + (i - 0.5) * bar_width
            # 关键修改：edgecolor 改为 colors[method]
            ax.bar(pos, method_data[m_key], width=bar_width, 
                   label=method, color=colors[method], 
                   edgecolor=colors[method], alpha=0.9)
        
        if threshold:
            ax.axhline(y=threshold, color='red', linestyle='-', linewidth=1.5, label=slo_name)
            
        ax.set_title(m_label, fontsize=14, fontweight='normal')
        ax.set_xticks(x)
        ax.set_xticklabels(qps_list)
        ax.set_xlabel('QPS (req/s)', fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        # 增加 y 轴顶部边缘，防止图例遮挡柱子或 SLO 线
        ax.margins(y=0.4)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"latency_analysis_{timestamp}_{test_model}_{test_dataset}.png"), dpi=300)

# --- 执行 ---
if not df_avg.empty:
    plot_goodput()
    plot_latency_metrics()
    print(f"Done! Final versions saved to {save_dir}")
else:
    print("No data processed. Check your directory structure.")