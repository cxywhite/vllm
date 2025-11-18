#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_parser_and_plot.py

Usage example:
  python log_parser_and_plot2.py \
    --log-dirs /root/predict/batchsize_experiment/log/gpu_0.3 /root/predict/batchsize_experiment/log/gpu_0.95 \
    --labels gpu_0.3 gpu_0.95 \
    --out-dir /root/predict/batchsize_experiment/result

作用:
- 分别读取两个目录下的 .log 文件并解析指标（与之前版本保持一致）
- 对相同指标（例如 Output token goodput (tok/s)）把两条曲线（来自两个目录）画在同一张图中，曲线为平滑连接
- 保存 PNG，每张图内含两个系列并带图例；保存 CSV（每个目录各自一个 CSV），并额外保存合并 CSV（带 source 列）
"""
from __future__ import annotations
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
# Regex patterns to extract values from logs
PATTERNS = {
    "Successful requests": re.compile(r"Successful requests:\s*([\d,]+)"),
    "Request throughput (req/s)": re.compile(r"Request throughput \(req/s\):\s*([\d\.]+)"),
    "Request goodput (req/s)": re.compile(r"Request goodput \(req/s\):\s*([\d\.]+)"),
    "Output token goodput (tok/s)": re.compile(r"Output token goodput \(tok/s\):\s*([\d\.]+)"),
    "Total token goodput (tok/s)": re.compile(r"Total token goodput \(tok/s\):\s*([\d\.]+)"),
    "Output token throughput (tok/s)": re.compile(r"Output token throughput \(tok/s\):\s*([\d\.]+)"),
    "Total Token throughput (tok/s)": re.compile(r"Total Token throughput \(tok/s\):\s*([\d\.]+)"),
}

# try to find batchsize in filename like client_batchsize32_... or batchsize-32
BATCHNAME_RE = re.compile(r"batchsize[_-]?(\d+)", flags=re.IGNORECASE)

# metrics to plot (column names must match parsed names)
PLOT_METRICS = [
    "Output token goodput (tok/s)",
    "Total token goodput (tok/s)",
    "Output token throughput (tok/s)",
    "Total Token throughput (tok/s)",
]

def tokyo_timestamp_now() -> str:
    now = datetime.now(timezone.utc) + timedelta(hours=9)
    return now.strftime("%Y%m%d_%H%M%S")

def parse_file(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    data = {}
    m = BATCHNAME_RE.search(path.name)
    data["batchsize"] = int(m.group(1)) if m else None
    for key, pat in PATTERNS.items():
        mm = pat.search(text)
        if mm:
            val = mm.group(1).replace(",", "")
            if "." in val:
                try:
                    data[key] = float(val)
                except:
                    data[key] = None
            else:
                try:
                    data[key] = int(val)
                except:
                    data[key] = None
        else:
            data[key] = None
    if data["batchsize"] is None and data.get("Successful requests") is not None:
        try:
            data["batchsize"] = int(data["Successful requests"])
        except:
            data["batchsize"] = None
    data["_source_file"] = str(path)
    return data

def collect_metrics(log_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(log_dir.rglob("*.log")):
        try:
            rows.append(parse_file(p))
        except Exception as e:
            print(f"Warning: failed to parse {p}: {e}", file=sys.stderr)
    if not rows:
        raise SystemExit(f"No .log files found under {log_dir}")
    df = pd.DataFrame(rows)
    df["batchsize"] = pd.to_numeric(df["batchsize"], errors="coerce")
    df = df[df["batchsize"].notna()].copy()
    if df.empty:
        raise SystemExit(f"No log entries with batchsize found under {log_dir}")
    for col in list(PATTERNS.keys()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Successful requests" in df.columns:
        df = df.drop(columns=["Successful requests"])
    df = df.sort_values("batchsize").reset_index(drop=True)
    return df

def save_csv(df: pd.DataFrame, out_dir: Path, prefix: str = "metrics") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = tokyo_timestamp_now()
    fn = out_dir / f"{prefix}_{ts}.csv"
    df.to_csv(fn, index=False)
    return fn

def smooth_connect_curve(x: np.ndarray, y: np.ndarray, num_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    xs_raw = x[mask]
    ys_raw = y[mask]
    if xs_raw.size == 0:
        return np.array([]), np.array([])
    if xs_raw.size == 1:
        return xs_raw.copy(), ys_raw.copy()
    order = np.argsort(xs_raw)
    xs_sorted = xs_raw[order]
    ys_sorted = ys_raw[order]
    xs_dense = np.linspace(xs_sorted.min(), xs_sorted.max(), num_points)
    ys_dense = np.interp(xs_dense, xs_sorted, ys_sorted)
    w = max(3, int(len(xs_dense) / 20))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    pad = w // 2
    ys_padded = np.pad(ys_dense, pad_width=pad, mode='edge')   # edge padding
    ys_smooth = np.convolve(ys_padded, kernel, mode='valid')   # valid -> same length as ys_dense
    return xs_dense, ys_smooth

def plot_two_series(df_a: pd.DataFrame, label_a: str, df_b: pd.DataFrame, label_b: str,
                    metric: str, out_path: Path):
    
        
    # prepare arrays
    xa = df_a["batchsize"].to_numpy(dtype=float)
    ya = df_a[metric].to_numpy(dtype=float) if metric in df_a.columns else np.array([])

    xb = df_b["batchsize"].to_numpy(dtype=float)
    yb = df_b[metric].to_numpy(dtype=float) if metric in df_b.columns else np.array([])

    if (np.isfinite(ya).sum() == 0) and (np.isfinite(yb).sum() == 0):
        raise RuntimeError(f"No valid points for metric '{metric}' in both sources.")

    # figure: 更紧凑的尺寸 (宽, 高)，dpi 高以保证清晰
    plt.figure(figsize=(5, 3), dpi=300)

    # 颜色设置（可自行替换为你喜欢的 hex 颜色或 'C0','C1'）
    col_a = 'C0'  # 或 '#1f77b4'
    col_b = 'C1'  # 或 '#ff7f0e'

    # series A 无散点，仅平滑实线
    if np.isfinite(ya).sum() > 0:
        if xa.size == 1 or np.isfinite(ya).sum() == 1:
            # 单点时也用一个小圆点标示（可删）
            plt.plot(xa, ya, marker='o', color=col_a, markersize=4, linestyle='', label=f"{label_a}")
        else:
            xs_a, ys_a = smooth_connect_curve(xa, ya, num_points=800)
            if xs_a.size > 0:
                # 画出原始数据点
                # plt.scatter(xa, ya, color=col_a, s=15, alpha=0.6, label=f"{label_a} data", zorder=2)
                plt.plot(xs_a, ys_a, linestyle='-', color=col_a, linewidth=1.2, label=f"{label_a}")

    # series B 无散点，仅平滑实线（原来的虚线改为实线并用不同颜色）
    if np.isfinite(yb).sum() > 0:
        if xb.size == 1 or np.isfinite(yb).sum() == 1:
            plt.plot(xb, yb, marker='s', color=col_b, markersize=4, linestyle='', label=f"{label_b}")
        else:
            xs_b, ys_b = smooth_connect_curve(xb, yb, num_points=800)
            if xs_b.size > 0:
                # 画出原始数据点
                # plt.scatter(xa, ya, color=col_a, s=15, alpha=0.6, label=f"{label_a} data", zorder=2)
                plt.plot(xs_b, ys_b, linestyle='-', color=col_b, linewidth=1.2, label=f"{label_b}")

    # 轴标签、标题、网格、图例（字体大小已由顶部 rcParams 控制）
    plt.xlabel("batchsize")
    plt.ylabel(metric)
    plt.title(f"{metric}")
    plt.grid(True, linestyle=':', linewidth=0.4)

    # 调整图例位置（右上、可改 'lower right' 等）
    plt.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    # 保存时使用 bbox_inches='tight' 剔除多余边距，确保紧凑
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    return out_path


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\d]+", "_", s).strip("_")

def main():
    parser = argparse.ArgumentParser(description="Collect metrics from two log directories and plot comparisons.")
    parser.add_argument("--log-dirs", nargs=2, type=Path,
                        default=[Path("/root/predict/batchsize_experiment/log/gpu_0.3"),
                                 Path("/root/predict/batchsize_experiment/log/gpu_0.95")],
                        help="Two log directories (recursive).")
    parser.add_argument("--labels", nargs=2, type=str, default=["gpu_0.3", "gpu_0.95"],
                        help="Labels for the two directories (used in legends and filenames).")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for CSV and PNGs.")
    parser.add_argument("--csv-prefix", default="metrics", help="CSV filename prefix for per-source CSVs.")
    args = parser.parse_args()

    dir_a, dir_b = [p.expanduser().resolve() for p in args.log_dirs]
    label_a, label_b = args.labels
    out_dir: Path = args.out_dir.expanduser().resolve()
    if not dir_a.exists():
        raise SystemExit(f"Log directory not found: {dir_a}")
    if not dir_b.exists():
        raise SystemExit(f"Log directory not found: {dir_b}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect
    print(f"Collecting logs from {dir_a} ({label_a}) ...")
    df_a = collect_metrics(dir_a)
    print(f"Collected {len(df_a)} rows from {dir_a}")
    print(f"Collecting logs from {dir_b} ({label_b}) ...")
    df_b = collect_metrics(dir_b)
    print(f"Collected {len(df_b)} rows from {dir_b}")

    # Save per-source CSVs
    csv_a = save_csv(df_a, out_dir, prefix=f"{args.csv_prefix}_{safe_name(label_a)}")
    csv_b = save_csv(df_b, out_dir, prefix=f"{args.csv_prefix}_{safe_name(label_b)}")
    print(f"Saved CSV -> {csv_a}")
    print(f"Saved CSV -> {csv_b}")

    # Save combined CSV with a 'source' column
    ts = tokyo_timestamp_now()
    df_a_copy = df_a.copy()
    df_b_copy = df_b.copy()
    df_a_copy["source"] = label_a
    df_b_copy["source"] = label_b
    df_combined = pd.concat([df_a_copy, df_b_copy], axis=0, ignore_index=True)
    combined_csv = out_dir / f"{args.csv_prefix}_combined_{ts}.csv"
    df_combined.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV -> {combined_csv}")

    # For each metric, plot two series on one figure
    for metric in PLOT_METRICS:
        try:
            if (metric not in df_a.columns) and (metric not in df_b.columns):
                print(f"Warning: metric '{metric}' not present in either source. Skipping.")
                continue
            fname = f"{safe_name(metric)}_{safe_name(label_a)}_vs_{safe_name(label_b)}_{ts}.png"
            out_path = out_dir / fname
            plot_two_series(df_a, label_a, df_b, label_b, metric, out_path)
            print(f"Saved comparison plot -> {out_path}")
        except Exception as e:
            print(f"Failed to create comparison plot for {metric}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
