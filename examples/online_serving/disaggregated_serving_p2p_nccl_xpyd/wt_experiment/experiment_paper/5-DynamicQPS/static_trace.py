import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_qps_from_trace(file_path, start_sec=None, end_sec=None):
    # 1. 加载 JSONL 数据
    timestamps = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                timestamps.append(data['timestamp'])

    if not timestamps:
        print("输入文件中没有可用的 timestamp 数据。")
        return

    # 2. 数据处理
    # 将时间戳从毫秒转换为秒 (假设单位是 ms)
    df = pd.DataFrame({'timestamp_ms': timestamps})
    df['time_s_abs'] = df['timestamp_ms'] / 1000.0

    # 转换为相对时间（从首条请求开始计时），便于按时间区间截取
    base_time = df['time_s_abs'].min()
    df['time_s'] = df['time_s_abs'] - base_time

    # 基于整份 trace 统计平均 QPS，包含没有请求的秒数
    df_all = df.copy()
    df_all['second_bin'] = df_all['time_s'].astype(int)
    full_range_all = range(int(df_all['second_bin'].min()), int(df_all['second_bin'].max()) + 1)
    qps_series_all = df_all.groupby('second_bin').size().reindex(full_range_all, fill_value=0)
    average_qps = float(qps_series_all.mean())

    # 处理时间区间参数
    if start_sec is None:
        start_sec = float(df['time_s'].min())
    if end_sec is None:
        end_sec = float(df['time_s'].max())

    if start_sec < 0 or end_sec < 0:
        raise ValueError("start_sec 和 end_sec 必须为非负数。")
    if start_sec > end_sec:
        raise ValueError("start_sec 不能大于 end_sec。")

    # 按给定时间区间截取
    df = df[(df['time_s'] >= start_sec) & (df['time_s'] <= end_sec)].copy()
    if df.empty:
        print(f"指定区间 [{start_sec}, {end_sec}] 秒内没有数据，无法绘图。")
        return
    
    # 向下取整到最近的一秒，用于统计每秒的请求数
    df['second_bin'] = df['time_s'].astype(int)
    
    # 统计每秒出现的频率，即 QPS
    qps_series = df.groupby('second_bin').size()
    
    # 补齐指定区间内缺失的秒数（如果某些秒内没有请求，QPS 应为 0）
    range_start = int(start_sec)
    range_end = int(end_sec)
    full_range = range(range_start, range_end + 1)
    qps_series = qps_series.reindex(full_range, fill_value=0)

    # 3. 绘图
    plt.figure(figsize=(10, 5))
    
    # 使用平滑后的波浪线风格曲线，避免离散秒级数据看起来像条形/阶梯
    x = qps_series.index.to_numpy(dtype=float)
    y = qps_series.values.astype(float)

    if len(x) >= 4:
        # 先做滚动均值平滑，再插值到更密集点，形成更连续的波形
        y_smooth = pd.Series(y).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
        x_dense = np.linspace(x.min(), x.max(), num=max(len(x) * 10, 200))
        y_dense = np.interp(x_dense, x, y_smooth)
        plt.plot(x_dense, y_dense, color='#1f77b4', linewidth=2.0)
    else:
        plt.plot(x, y, color='#1f77b4', linewidth=2.0)

    plt.axhline(
        y=average_qps,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Average QPS = {average_qps:.2f}'
    )
    
    # 设置标题和标签
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('QPS (req/s)', fontsize=12)
    # plt.title(f'QPS Over Time ({start_sec:.2f}s - {end_sec:.2f}s)', fontsize=14)
    
    # 优化网格线
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 调整坐标轴范围到指定区间
    plt.xlim(left=range_start, right=range_end)
    plt.ylim(bottom=0)

    ax = plt.gca()
    current_yticks = ax.get_yticks()
    yticks = np.append(current_yticks, average_qps)
    yticks = np.unique(np.round(yticks, 2))
    ax.set_yticks(yticks)

    ytick_labels = []
    for tick in yticks:
        if np.isclose(tick, average_qps):
            ytick_labels.append(f'{average_qps:.2f}')
        elif float(tick).is_integer():
            ytick_labels.append(str(int(tick)))
        else:
            ytick_labels.append(f'{tick:.2f}')
    ax.set_yticklabels(ytick_labels)

    plt.legend()
    
    plt.tight_layout()
    
    # 保存或展示
    # 保存文件后缀加时间戳以避免覆盖
    output_filename = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/fig/qps_plot_{args.start_sec}_{args.end_sec}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"图表已保存为: {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSONL trace 中按时间区间绘制 QPS 曲线")
    parser.add_argument(
        "--file",
        default='/root/Mooncake/FAST25-release/traces/conversation_trace.jsonl',
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="截取起始秒（相对首条请求时间），默认自动取最小值"
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="截取结束秒（相对首条请求时间），默认自动取最大值"
    )

    args = parser.parse_args()
    plot_qps_from_trace(args.file, args.start_sec, args.end_sec)