import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

from trace_utils import add_trace_type_argument, load_trace_records


BASE_DIR = '/root/workspace/vllm_v0.11.0_2/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS'


def _default_fig_dir(trace_type: str) -> str:
    return os.path.join(BASE_DIR, 'fig', trace_type)


def _default_tmp_dir(trace_type: str) -> str:
    return os.path.join(BASE_DIR, 'tmp', trace_type)


def _build_stats_lines(values, title):
    arr = np.asarray(values, dtype=float)
    quantiles = [0.5, 0.9, 0.95, 0.99]
    lines = [
        f"[{title}]",
        f"count: {arr.size}",
        f"min: {arr.min():.4f}",
        f"max: {arr.max():.4f}",
        f"mean: {arr.mean():.4f}",
    ]
    for q in quantiles:
        lines.append(f"p{int(q * 100)}: {np.quantile(arr, q):.4f}")
    lines.append("")
    return lines


def _build_cdf_curve(values):
    sorted_vals = np.sort(np.asarray(values, dtype=float))
    n = sorted_vals.size
    if n == 1:
        return sorted_vals, np.array([1.0])

    empirical_cdf = np.arange(1, n + 1, dtype=float) / n
    x_dense = np.linspace(sorted_vals[0], sorted_vals[-1], num=min(max(200, n * 10), 5000))
    y_dense = np.interp(x_dense, sorted_vals, empirical_cdf, left=0.0, right=1.0)
    return x_dense, y_dense


def _build_smoothed_curve(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size >= 4:
        kernel = np.ones(5, dtype=float)
        numerator = np.convolve(y, kernel, mode='same')
        denominator = np.convolve(np.ones_like(y), kernel, mode='same')
        y_smooth = numerator / denominator
        # Cap interpolation points to avoid plotting millions of points on long traces.
        dense_points = min(max(x.size * 4, 200), 20000)
        x_dense = np.linspace(x.min(), x.max(), num=dense_points)
        y_dense = np.interp(x_dense, x, y_smooth)
        return x_dense, y_dense
    return x, y


def _load_csv_arrays_fast(file_path: Path, trace_type: str, chunksize: int):
    if trace_type == 'azure':
        ts_col = 'TIMESTAMP'
        input_col = 'ContextTokens'
        output_col = 'GeneratedTokens'
    elif trace_type == 'burstgpt':
        ts_col = 'Timestamp'
        input_col = 'Request tokens'
        output_col = 'Response tokens'
    elif trace_type == 'test_results':
        ts_col = 'send_timestamp_ms'
        input_col = 'prompt_len'
        output_col = 'output_tokens'
    else:
        raise ValueError(f'unsupported csv trace type: {trace_type}')

    ts_parts = []
    input_parts = []
    output_parts = []

    def _normalize_epoch_to_ms(series: pd.Series) -> pd.Series:
        # Pandas backends can expose epoch integers in s/us/ns; normalize to ms.
        numeric = pd.to_numeric(series, errors='coerce').astype(float)
        non_na = numeric.dropna()
        if non_na.empty:
            return numeric

        scale_probe = float(non_na.abs().median())
        if scale_probe >= 1e17:
            # nanoseconds -> milliseconds
            return numeric / 1_000_000.0
        if scale_probe >= 1e14:
            # microseconds -> milliseconds
            return numeric / 1_000.0
        if scale_probe >= 1e11:
            # already milliseconds
            return numeric
        if scale_probe >= 1e8:
            # seconds -> milliseconds
            return numeric * 1_000.0

        raise ValueError(f'unexpected epoch scale in timestamp column: median={scale_probe}')

    for chunk in pd.read_csv(file_path, usecols=[ts_col, input_col, output_col], chunksize=chunksize):
        input_vals = pd.to_numeric(chunk[input_col], errors='coerce')
        output_vals = pd.to_numeric(chunk[output_col], errors='coerce')

        if trace_type == 'azure':
            # Parse ISO timestamps with timezone and convert to milliseconds.
            ts_raw = pd.to_datetime(chunk[ts_col], errors='coerce', utc=True).astype('int64')
            ts_ms = _normalize_epoch_to_ms(ts_raw)
            ts_ms = pd.Series(ts_ms, index=chunk.index)
            ts_ms = ts_ms.where(~pd.isna(chunk[ts_col]), np.nan)
        elif trace_type == 'burstgpt':
            ts_ms = pd.to_numeric(chunk[ts_col], errors='coerce') * 1000.0
        else:
            # test_results timestamps are already epoch milliseconds.
            ts_ms = pd.to_numeric(chunk[ts_col], errors='coerce')

        valid = (~ts_ms.isna()) & (~input_vals.isna()) & (~output_vals.isna())
        if not valid.any():
            continue

        ts_parts.append(ts_ms[valid].to_numpy(dtype=float, copy=False))
        input_parts.append(input_vals[valid].to_numpy(dtype=float, copy=False))
        output_parts.append(output_vals[valid].to_numpy(dtype=float, copy=False))

    if not ts_parts:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    return (
        np.concatenate(ts_parts),
        np.concatenate(input_parts),
        np.concatenate(output_parts),
    )


def plot_qps_from_trace(
    file_path,
    output_dir,
    trace_type,
    start_sec=None,
    end_sec=None,
    stats_file=None,
    cdf_file=None,
    csv_chunksize=1_000_000,
):
    # 1. 加载 trace 数据
    input_path = Path(file_path)

    if trace_type in ('azure', 'burstgpt', 'test_results'):
        timestamps_ms, input_vals, output_vals = _load_csv_arrays_fast(input_path, trace_type, csv_chunksize)
    else:
        records = []
        input_lengths = []
        output_lengths = []
        loaded = load_trace_records(
            input_path,
            trace_type=trace_type,
            require_timestamp=True,
            require_lengths=True,
            skip_invalid=True,
        )

        for record in loaded.records:
            if record.timestamp_ms is None:
                continue

            input_len = float(record.input_length) if record.input_length is not None else None
            output_len = float(record.output_length) if record.output_length is not None else None

            records.append({
                'timestamp_ms': record.timestamp_ms,
                'input_len': 0.0 if input_len is None else input_len,
                'output_len': 0.0 if output_len is None else output_len,
            })
            if input_len is not None:
                input_lengths.append(input_len)
            if output_len is not None:
                output_lengths.append(output_len)

        if not records:
            print("输入文件中没有可用的 timestamp 数据。")
            return

        timestamps_ms = np.asarray([record['timestamp_ms'] for record in records], dtype=float)
        input_vals = np.asarray([record['input_len'] for record in records], dtype=float)
        output_vals = np.asarray([record['output_len'] for record in records], dtype=float)

    if timestamps_ms.size == 0:
        print("输入文件中没有可用的 timestamp 数据。")
        return

    if input_vals.size == 0 or output_vals.size == 0:
        raise ValueError('trace 中缺少 input/output 长度字段，请确认包含 input_length/output_length 或 input_len/output_len。')

    input_lengths = input_vals
    output_lengths = output_vals

    tmp_dir = _default_tmp_dir(trace_type)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = input_path.stem

    if stats_file is None:
        stats_file = os.path.join(tmp_dir, f'length_stats_{now_tag}_{input_stem}.txt')
    if cdf_file is None:
        cdf_file = os.path.join(output_dir, f'trace_{input_stem}_{start_sec}_{end_sec}_{now_tag}.png')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    os.makedirs(os.path.dirname(cdf_file), exist_ok=True)

    # 2. 统计并写入文件
    stats_lines = []
    stats_lines.extend(_build_stats_lines(input_lengths, 'input_len'))
    stats_lines.extend(_build_stats_lines(output_lengths, 'output_len'))
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))

    # 3. 准备 CDF 曲线
    x_in, y_in = _build_cdf_curve(input_lengths)
    x_out, y_out = _build_cdf_curve(output_lengths)

    # 4. QPS 数据处理
    # 将时间戳从毫秒转换为秒 (假设单位是 ms)
    time_s_abs = timestamps_ms / 1000.0

    # 转换为相对时间（从首条请求开始计时），便于按时间区间截取
    base_time = time_s_abs.min()
    time_s = time_s_abs - base_time

    # 处理时间区间参数
    if start_sec is None:
        start_sec = float(time_s.min())
    if end_sec is None:
        end_sec = float(time_s.max())

    if start_sec < 0 or end_sec < 0:
        raise ValueError("start_sec 和 end_sec 必须为非负数。")
    if start_sec > end_sec:
        raise ValueError("start_sec 不能大于 end_sec。")

    # 按给定时间区间截取
    mask = (time_s >= start_sec) & (time_s <= end_sec)
    if not np.any(mask):
        print(f"指定区间 [{start_sec}, {end_sec}] 秒内没有数据，无法绘图。")
        return

    filtered_time_s = time_s[mask]
    filtered_input = input_vals[mask]
    filtered_output = output_vals[mask]

    # 向下取整到最近的一秒，用于统计每秒的请求数
    second_bins = filtered_time_s.astype(int)

    # 补齐指定区间内缺失的秒数（如果某些秒内没有请求，QPS 应为 0）
    range_start = int(start_sec)
    range_end = int(end_sec)
    full_range = np.arange(range_start, range_end + 1, dtype=int)
    qps_values = np.zeros(full_range.shape[0], dtype=float)
    input_token_values = np.zeros(full_range.shape[0], dtype=float)
    output_token_values = np.zeros(full_range.shape[0], dtype=float)

    for second_bin, input_len, output_len in zip(second_bins, filtered_input, filtered_output):
        idx = second_bin - range_start
        if 0 <= idx < full_range.shape[0]:
            qps_values[idx] += 1
            input_token_values[idx] += input_len
            output_token_values[idx] += output_len

    avg_qps = float(qps_values.mean())

    # 5. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    x = full_range.astype(float)
    qps_dense_x, qps_dense_y = _build_smoothed_curve(x, qps_values)
    axes[0].plot(qps_dense_x, qps_dense_y, color='#1f77b4', linewidth=2.0, label='QPS')
    axes[0].axhline(
        y=avg_qps,
        color='red',
        linestyle='--',
        linewidth=1.8,
        label=f'Average QPS: {avg_qps:.2f}'
    )
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('QPS (req/s)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_xlim(left=range_start, right=range_end)
    axes[0].set_ylim(bottom=0)
    existing_ticks = axes[0].get_yticks()
    y_ticks = np.unique(np.append(existing_ticks, avg_qps))
    axes[0].set_yticks(np.sort(y_ticks))
    axes[0].set_yticklabels([
        f'avg={tick:.2f}' if np.isclose(tick, avg_qps) else f'{tick:g}'
        for tick in axes[0].get_yticks()
    ])
    axes[0].legend(loc='upper right')

    input_dense_x, input_dense_y = _build_smoothed_curve(x, input_token_values)
    output_dense_x, output_dense_y = _build_smoothed_curve(x, output_token_values)
    axes[1].plot(input_dense_x, input_dense_y, color='#1f77b4', linewidth=2.0, label='input_len')
    axes[1].plot(output_dense_x, output_dense_y, color='#ff7f0e', linewidth=2.0, label='output_len')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Tokens', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_xlim(left=range_start, right=range_end)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='upper right')

    axes[2].plot(x_in, y_in, label='input_len', color='#1f77b4', linewidth=2.2)
    axes[2].plot(x_out, y_out, label='output_len', color='#ff7f0e', linewidth=2.2)
    axes[2].set_xlabel('Length', fontsize=12)
    axes[2].set_ylabel('CDF', fontsize=12)
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper right')

    fig.tight_layout()

    # 保存或展示
    output_filename = cdf_file
    fig.savefig(output_filename, dpi=300)
    print(f"平均 QPS: {avg_qps:.2f}")
    print(f"长度统计已保存为: {stats_file}")
    print(f"图表已保存为: {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSONL trace 绘制 QPS，并导出输入/输出长度统计与 CDF")
    add_trace_type_argument(parser)
    # Keep default trace type aligned with default input file.
    parser.set_defaults(trace_type="qwen")
    parser.add_argument(
        "--file",
        default='/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/tmp_dataset/qwen_thinking_blksz_16_th8192_avg8.jsonl',
        help="输入 trace 文件路径"
    )
    # 添加输出目录前缀
    parser.add_argument(
        "--output-dir",
        default=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/tmp_dataset/fig',
        help="输出图表的目录路径；不传则使用 BASE_DIR/fig/<trace_type>"
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
    parser.add_argument(
        "--stats-file",
        default=None,
        help="长度统计 txt 输出路径；不传则自动写入 tmp 目录"
    )
    parser.add_argument(
        "--cdf-file",
        default=None,
        help="三联图输出路径；不传则自动写入 output-dir"
    )
    parser.add_argument(
        "--csv-chunksize",
        type=int,
        default=1000000,
        help="CSV trace 的分块读取行数（仅 azure/burstgpt 生效）",
    )

    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else _default_fig_dir(args.trace_type)
    plot_qps_from_trace(
        args.file,
        output_dir,
        args.trace_type,
        args.start_sec,
        args.end_sec,
        args.stats_file,
        args.cdf_file,
        args.csv_chunksize,
    )