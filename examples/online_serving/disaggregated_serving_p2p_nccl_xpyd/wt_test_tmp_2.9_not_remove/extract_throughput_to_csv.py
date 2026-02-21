#!/usr/bin/env python3
"""
extract_throughput_to_csv.py

功能:
  从日志文件中提取 throughput 信息，保存为 CSV 文件。
  输出字段：
    avg_prompt_throughput_tokens_per_s,
    avg_generation_throughput_tokens_per_s,
    running_reqs,
    waiting_reqs,
    gpu_kv_cache_usage_percent,
    prefix_cache_hit_rate_percent

  另外：
    - 如果日志文件名包含 "prefill"（不区分大小写）则画图：纵轴=Avg prompt throughput，横轴=running_reqs
    - 如果日志文件名包含 "decode"（不区分大小写）则画图：纵轴=Avg generation throughput，横轴=running_reqs

用法:
  # 保存脚本并加可执行权限
  chmod +x extract_throughput_to_csv.py

  # 运行脚本（默认在同目录产出 CSV & PNG）
  python extract_throughput_to_csv.py /path/to/prefill1.log

  # 或者指定输出目录
  python extract_throughput_to_csv.py ./experiment_result/log/prefill1.log -o ./experiment_result/fig
"""

import re
import csv
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# 去掉 ANSI 转义序列（例如 \x1b[1;36m）
_ansi_re = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# 匹配 throughput 日志行
_line_re = re.compile(
    r'Avg prompt throughput:\s*(?P<prompt>[0-9.]+)\s*tokens/s,\s*'
    r'Avg generation throughput:\s*(?P<gen>[0-9.]+)\s*tokens/s,\s*'
    r'Running:\s*(?P<running>\d+)\s*reqs,\s*'
    r'Waiting:\s*(?P<waiting>\d+)\s*reqs,\s*'
    r'GPU KV cache usage:\s*(?P<gpu_kv>[0-9.]+)%\s*,\s*'
    r'Prefix cache hit rate:\s*(?P<prefix_hit>[0-9.]+)%'
)

def strip_ansi(s: str) -> str:
    return _ansi_re.sub('', s)

def parse_line(line: str):
    clean = strip_ansi(line)
    m = _line_re.search(clean)
    if not m:
        return None
    gd = m.groupdict()
    return {
        'avg_prompt_throughput_tokens_per_s': float(gd['prompt']),
        'avg_generation_throughput_tokens_per_s': float(gd['gen']),
        'running_reqs': int(gd['running']),
        'waiting_reqs': int(gd['waiting']),
        'gpu_kv_cache_usage_percent': float(gd['gpu_kv']),
        'prefix_cache_hit_rate_percent': float(gd['prefix_hit'])
    }

def plot_metric(rows, x_key, y_key, outpath, title=None):
    # rows: list of dicts
    # build x, y lists and sort by x for nicer line
    xy = [(r[x_key], r[y_key]) for r in rows]
    xy_sorted = sorted(xy, key=lambda t: (t[0],))  # sort by x (running_reqs)
    x = [t[0] for t in xy_sorted]
    y = [t[1] for t in xy_sorted]

    plt.figure()
    # single line plot (no explicit color set)
    plt.plot(x, y, marker='o')
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Extract throughput metrics from log into CSV (and optionally plot).")
    p.add_argument('logfile', help="Path to the log file to parse")
    p.add_argument('--outdir', '-o', default=None, help="Optional output directory for CSV/PNG (default: same dir as logfile)")
    args = p.parse_args()

    logfile = Path(args.logfile)
    if not logfile.exists():
        print(f"Error: logfile not found: {logfile}")
        return

    rows = []
    with logfile.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parsed = parse_line(line)
            if parsed:
                rows.append(parsed)

    if not rows:
        print("No matching throughput lines found in the log.")
        return

    # write CSV
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else logfile.parent
    outdir.mkdir(parents=True, exist_ok=True)
    csv_outpath = outdir / f"{logfile.stem}_{now}.csv"

    fieldnames = [
        'avg_prompt_throughput_tokens_per_s',
        'avg_generation_throughput_tokens_per_s',
        'running_reqs',
        'waiting_reqs',
        'gpu_kv_cache_usage_percent',
        'prefix_cache_hit_rate_percent'
    ]

    with csv_outpath.open('w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} record(s) to {csv_outpath}")

    # decide whether to plot and what to plot
    name_lower = logfile.name.lower()
    plot_made = False
    png_outpath = outdir / f"{logfile.stem}_{now}.png"

    if 'prefill' in name_lower:
        # x = running_reqs, y = avg_prompt_throughput_tokens_per_s
        title = f"{logfile.name} - Avg prompt throughput vs running_reqs ({now})"
        plot_metric(rows, 'running_reqs', 'avg_prompt_throughput_tokens_per_s', png_outpath, title=title)
        print(f"Saved plot to {png_outpath}")
        plot_made = True
    elif 'decode' in name_lower:
        # x = running_reqs, y = avg_generation_throughput_tokens_per_s
        title = f"{logfile.name} - Avg generation throughput vs running_reqs ({now})"
        plot_metric(rows, 'running_reqs', 'avg_generation_throughput_tokens_per_s', png_outpath, title=title)
        print(f"Saved plot to {png_outpath}")
        plot_made = True
    else:
        print("Log filename does not contain 'prefill' or 'decode' - skipping plotting.")

    if plot_made:
        print("Plotting finished.")

if __name__ == '__main__':
    main()
