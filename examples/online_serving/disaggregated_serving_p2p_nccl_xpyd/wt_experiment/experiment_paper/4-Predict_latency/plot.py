import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 读取数据
df = pd.read_csv('/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/4-Predict_latency/tmp/predict_timing_2026-03-19_19-14-56.csv')

# 2. 准备绘图数据和配置
metrics = ['predict_ms', 'prefill_tail_ms']
labels = ['Predictor Duration', 'Prefill (L16-32) Duration']
colors = ['#FF9F40', '#36A2EB'] # 橙色表示 Predictor，蓝色表示 Prefill

fig, ax = plt.subplots(figsize=(10, 4))
y_positions = [0, 1]

# 3. 循环计算分位数并绘制图形
for i, (metric, color) in enumerate(zip(metrics, colors)):
    # 提取非空数据
    data = df[metric].dropna()
    
    # 计算分位数: 5%, 25% (集中区域起点), 50% (中位数), 75% (集中区域终点), 95%
    p05 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)
    
    y = y_positions[i]
    
    # --- 绘制 |-| (5% 到 95% 的线段和两端竖线) ---
    # 横线 (-)
    ax.plot([p05, p95], [y, y], color='black', linewidth=1.5, zorder=1)
    
    # 两端竖线 (|)
    cap_height = 0.2
    ax.plot([p05, p05], [y - cap_height/2, y + cap_height/2], color='black', linewidth=1.5, zorder=1)
    ax.plot([p95, p95], [y - cap_height/2, y + cap_height/2], color='black', linewidth=1.5, zorder=1)
    
    # --- 绘制带颜色的矩形 (25% 到 75% 的集中数据区) ---
    rect_height = 0.4
    rect = patches.Rectangle((p25, y - rect_height/2), width=(p75 - p25), height=rect_height, 
                             facecolor=color, edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    
    # --- (可选) 绘制中位数白线，帮助看出集中区域的重心 ---
    ax.plot([p50, p50], [y - rect_height/2, y + rect_height/2], color='white', linewidth=2, zorder=3)

# 4. 图表修饰
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Latency / Duration (ms)', fontsize=12)
ax.set_title('Latency Distribution: 5th-95th Percentile with IQR Box', fontsize=14)

# 自动调整 X 轴范围，留出一点边距
all_data = pd.concat([df['predict_ms'], df['prefill_tail_ms']]).dropna()
ax.set_xlim(max(0, np.percentile(all_data, 1) - 5), np.percentile(all_data, 99) + 5)

ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
# 保存图表
plt.savefig('/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/4-Predict_latency/tmp/predict_latency_distribution.png', dpi=300)
plt.show()