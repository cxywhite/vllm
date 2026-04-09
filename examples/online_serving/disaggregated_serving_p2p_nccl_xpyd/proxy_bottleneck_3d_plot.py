#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, to_rgb


CONFIG_PATTERN = re.compile(
    r"proxy_\d{8}_\d{6}_(?:recover_)?in(?P<input_len>\d+)_out(?P<output_len>\d+)_rr(?P<qps>\d+)\.log$"
)

MONITOR_PATTERN = re.compile(
    r"\[WT\]\[MONITOR\].*?"
    r"compute=(?P<compute>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?"
    r"memory=(?P<memory>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?"
    r"capacity=(?P<capacity>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

METRICS = ("compute", "memory", "capacity")
METRIC_ORDER = {"compute": 0, "memory": 1, "capacity": 2}
METRIC_COLORS = {
    "compute": "#1f77b4",
    "memory": "#d62728",
    "capacity": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse proxy logs and plot dominant bottleneck by (input_len, output_len, qps)."
    )
    parser.add_argument("--log-dir", required=True, help="Directory containing proxy log files")
    parser.add_argument(
        "--out-png",
        default=None,
        help=(
            "Output PNG path. Default: <log-dir>/proxy_bottleneck_2d.png for facets "
            "or <log-dir>/proxy_bottleneck_heatmap.png for heatmap."
        ),
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Default: <log-dir>/proxy_bottleneck_summary.csv",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Floating-point tolerance for zero and tie checks.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("facets", "heatmap", "heatmap-qps-split"),
        default="facets",
        help=(
            "facets: one panel per qps; "
            "heatmap: single heatmap aggregated across all qps; "
            "heatmap-qps-split: single 2D grid with in-cell qps slices."
        ),
    )
    return parser.parse_args()


def parse_log_file(log_path: str, eps: float):
    votes = defaultdict(float)
    nonzero_samples = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[WT][MONITOR]" not in line:
                continue

            m = MONITOR_PATTERN.search(line)
            if not m:
                continue

            values = {
                "compute": float(m.group("compute")),
                "memory": float(m.group("memory")),
                "capacity": float(m.group("capacity")),
            }

            # Only keep samples where the 3 load dimensions are not all zero.
            if all(abs(values[k]) <= eps for k in METRICS):
                continue

            nonzero_samples += 1
            max_value = max(values.values())
            winners = [k for k, v in values.items() if abs(v - max_value) <= eps]
            weight = 1.0 / len(winners)
            for w in winners:
                votes[w] += weight

    return votes, nonzero_samples


def aggregate_by_config(log_dir: str, eps: float):
    cfg_votes = defaultdict(lambda: defaultdict(float))
    cfg_samples = defaultdict(int)
    cfg_files = defaultdict(int)

    for file_name in sorted(os.listdir(log_dir)):
        if not file_name.endswith(".log"):
            continue
        if "bootstrap" in file_name:
            continue

        cfg_match = CONFIG_PATTERN.match(file_name)
        if not cfg_match:
            continue

        cfg = (
            int(cfg_match.group("input_len")),
            int(cfg_match.group("output_len")),
            int(cfg_match.group("qps")),
        )

        log_path = os.path.join(log_dir, file_name)
        votes, nonzero_samples = parse_log_file(log_path, eps)
        if nonzero_samples <= 0:
            continue

        cfg_files[cfg] += 1
        cfg_samples[cfg] += nonzero_samples
        for metric in METRICS:
            cfg_votes[cfg][metric] += votes.get(metric, 0.0)

    rows = []
    for cfg in sorted(cfg_votes.keys()):
        input_len, output_len, qps = cfg
        total_votes = sum(cfg_votes[cfg][m] for m in METRICS)
        if total_votes <= 0:
            continue

        freqs = {m: cfg_votes[cfg][m] / total_votes for m in METRICS}
        dominant_metric = max(
            METRICS,
            key=lambda m: (freqs[m], -METRIC_ORDER[m]),
        )
        dominant_ratio = freqs[dominant_metric]

        rows.append(
            {
                "input_len": input_len,
                "output_len": output_len,
                "qps": qps,
                "nonzero_samples": cfg_samples[cfg],
                "files": cfg_files[cfg],
                "compute_freq": freqs["compute"],
                "memory_freq": freqs["memory"],
                "capacity_freq": freqs["capacity"],
                "dominant_metric": dominant_metric,
                "dominant_ratio": dominant_ratio,
            }
        )

    return rows


def save_csv(rows, out_csv: str):
    fieldnames = [
        "input_len",
        "output_len",
        "qps",
        "nonzero_samples",
        "files",
        "compute_freq",
        "memory_freq",
        "capacity_freq",
        "dominant_metric",
        "dominant_ratio",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_2d_facets(rows, out_png: str):
    qps_values = sorted({row["qps"] for row in rows})
    qps_to_rows = {qps: [r for r in rows if r["qps"] == qps] for qps in qps_values}

    n_panels = len(qps_values)
    ncols = min(4, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        dpi=160,
        sharex=True,
        sharey=True,
    )

    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, qps in enumerate(qps_values):
        ax = axes[idx]
        panel_rows = qps_to_rows[qps]

        for row in panel_rows:
            dominant_metric = row["dominant_metric"]
            dominant_ratio = row["dominant_ratio"]
            size = 40 + 280 * dominant_ratio
            alpha = 0.45 + 0.5 * dominant_ratio

            ax.scatter(
                row["input_len"],
                row["output_len"],
                s=size,
                color=METRIC_COLORS[dominant_metric],
                alpha=alpha,
                edgecolors="black",
                linewidths=0.4,
            )

        ax.set_title(f"qps={qps} (configs={len(panel_rows)})", fontsize=10)
        ax.grid(True, alpha=0.28)

    for ax in axes[n_panels:]:
        ax.axis("off")

    for i, ax in enumerate(axes[:n_panels]):
        if i % ncols == 0:
            ax.set_ylabel("Output Length")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Input Length")

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=METRIC_COLORS[m],
            markeredgecolor="black",
            markersize=8,
            label=m,
        )
        for m in METRICS
    ]

    size_levels = [0.4, 0.6, 0.8, 1.0]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="black",
            markersize=math.sqrt(40 + 280 * lv),
            label=f"ratio={lv:.1f}",
        )
        for lv in size_levels
    ]

    fig.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995),
        title="Dominant metric",
    )
    fig.legend(
        handles=size_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.995),
        title="Dominance ratio",
    )

    fig.suptitle(
        "Dominant Bottleneck per Config (2D by QPS)\n"
        "Each panel: x=input_len, y=output_len; color=dominant metric; size=dominance ratio",
        y=1.03,
        fontsize=13,
    )
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    plt.savefig(out_png)
    plt.close(fig)


def aggregate_rows_across_qps(rows):
    agg_votes = defaultdict(lambda: defaultdict(float))
    agg_samples = defaultdict(int)

    for row in rows:
        key = (row["input_len"], row["output_len"])
        samples = int(row["nonzero_samples"])
        agg_samples[key] += samples

        # Recover weighted votes from frequency columns so we can merge qps fairly.
        agg_votes[key]["compute"] += float(row["compute_freq"]) * samples
        agg_votes[key]["memory"] += float(row["memory_freq"]) * samples
        agg_votes[key]["capacity"] += float(row["capacity_freq"]) * samples

    agg_rows = []
    for key in sorted(agg_votes.keys()):
        input_len, output_len = key
        total_votes = sum(agg_votes[key][m] for m in METRICS)
        if total_votes <= 0:
            continue

        freqs = {m: agg_votes[key][m] / total_votes for m in METRICS}
        dominant_metric = max(
            METRICS,
            key=lambda m: (freqs[m], -METRIC_ORDER[m]),
        )
        dominant_ratio = freqs[dominant_metric]

        agg_rows.append(
            {
                "input_len": input_len,
                "output_len": output_len,
                "nonzero_samples": agg_samples[key],
                "compute_freq": freqs["compute"],
                "memory_freq": freqs["memory"],
                "capacity_freq": freqs["capacity"],
                "dominant_metric": dominant_metric,
                "dominant_ratio": dominant_ratio,
            }
        )

    return agg_rows


def plot_single_heatmap(rows, out_png: str):
    agg_rows = aggregate_rows_across_qps(rows)
    if not agg_rows:
        raise RuntimeError("No rows available for heatmap plotting.")

    x_values = sorted({row["input_len"] for row in agg_rows})
    y_values = sorted({row["output_len"] for row in agg_rows})
    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}

    # 0=compute,1=memory,2=capacity,3=missing
    metric_to_code = {"compute": 0, "memory": 1, "capacity": 2}
    missing_code = 3

    code_grid = [
        [missing_code for _ in range(len(x_values))]
        for _ in range(len(y_values))
    ]
    ratio_grid = [[None for _ in range(len(x_values))] for _ in range(len(y_values))]

    for row in agg_rows:
        xi = x_index[row["input_len"]]
        yi = y_index[row["output_len"]]
        code_grid[yi][xi] = metric_to_code[row["dominant_metric"]]
        ratio_grid[yi][xi] = row["dominant_ratio"]

    cmap = ListedColormap([
        METRIC_COLORS["compute"],
        METRIC_COLORS["memory"],
        METRIC_COLORS["capacity"],
        "#e9e9e9",
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(1.1 * len(x_values) + 4, 1.0 * len(y_values) + 3), dpi=180)
    ax.imshow(code_grid, cmap=cmap, norm=norm, origin="lower", aspect="auto")

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(x) for x in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(y) for y in y_values])
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Output Length")
    ax.set_title(
        "Single Heatmap (Aggregated Across All QPS)\n"
        "Cell color=dominant metric, text=dominance ratio"
    )

    for yi in range(len(y_values)):
        for xi in range(len(x_values)):
            ratio = ratio_grid[yi][xi]
            if ratio is not None:
                ax.text(
                    xi,
                    yi,
                    f"{ratio:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    # Draw minor-grid lines to emphasize heatmap cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(x_values))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(y_values))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor=METRIC_COLORS["compute"], edgecolor="black", label="compute"),
        Patch(facecolor=METRIC_COLORS["memory"], edgecolor="black", label="memory"),
        Patch(facecolor=METRIC_COLORS["capacity"], edgecolor="black", label="capacity"),
    ]
    ax.legend(handles=legend_handles, title="Dominant metric", loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def plot_single_heatmap_qps_split(rows, out_png: str):
    x_values = sorted({row["input_len"] for row in rows})
    y_values = sorted({row["output_len"] for row in rows})
    qps_values = sorted({row["qps"] for row in rows})

    # (input_len, output_len, qps) -> row
    row_map = {
        (row["input_len"], row["output_len"], row["qps"]): row
        for row in rows
    }

    n_x = len(x_values)
    n_y = len(y_values)
    n_qps = len(qps_values)
    total_x_slots = n_x * n_qps

    fig_width = max(10.5, min(18.0, 0.38 * total_x_slots + 4.2))
    fig_height = max(6.8, min(10.5, 0.78 * n_y + 2.7))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    # Keep whitespace compact while reserving a narrow right area for color bars.
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.13, top=0.90)

    def blend_with_white(hex_color: str, ratio: float):
        ratio = max(0.0, min(1.0, float(ratio)))
        base = to_rgb(hex_color)
        return tuple((1.0 - ratio) * 1.0 + ratio * c for c in base)

    # Fill each qps sub-cell and encode ratio in color depth.
    for yi, output_len in enumerate(y_values):
        for xi, input_len in enumerate(x_values):
            group_x0 = xi * n_qps
            for qi, qps in enumerate(qps_values):
                key = (input_len, output_len, qps)
                row = row_map.get(key)
                x0 = group_x0 + qi
                y0 = yi

                if row is None:
                    fill_color = "#efefef"
                else:
                    metric_color = METRIC_COLORS[row["dominant_metric"]]
                    fill_color = blend_with_white(metric_color, row["dominant_ratio"])

                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        1.0,
                        1.0,
                        facecolor=fill_color,
                        edgecolor="white",
                        linewidth=0.35,
                    )
                )

    # Draw boundaries of big cells (one big cell = one input_len x output_len group).
    for xi in range(n_x + 1):
        ax.axvline(x=xi * n_qps, color="#bcbcbc", linewidth=1.1)
    for yi in range(n_y + 1):
        ax.axhline(y=yi, color="#d9d9d9", linewidth=1.0)

    ax.set_xlim(0, total_x_slots)
    ax.set_ylim(0, n_y)
    # Show only input_len on the single x-axis.
    input_tick_positions = [i * n_qps + n_qps / 2 for i in range(n_x)]
    ax.set_xticks(input_tick_positions)
    ax.set_xticklabels([str(x) for x in x_values], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(n_y)])
    ax.set_yticklabels([str(y) for y in y_values])
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Output Length")

    ax.set_title(
        "Single 2D Grid with In-Cell QPS Slices\n"
        "Color hue=dominant metric, color depth=dominance ratio"
    )

    # Right-side ratio bars: equal size, vertically stacked, touching each other,
    # and aligned to main axis top/bottom.
    ax_pos = ax.get_position()
    bar_x = ax_pos.x1 + 0.012
    bar_w = 0.028
    bar_h = ax_pos.height / 3.0

    def add_metric_gradient(bar_bottom, metric_name, metric_color, tick_mode="none"):
        cax = fig.add_axes([bar_x, bar_bottom, bar_w, bar_h])
        cmap = LinearSegmentedColormap.from_list(
            f"ratio_{metric_name}",
            ["#ffffff", metric_color],
        )
        gradient = [[i / 255.0] for i in range(256)]
        cax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1], origin="lower")
        cax.set_xticks([])
        cax.yaxis.tick_right()
        if tick_mode == "top":
            cax.set_yticks([1.0])
            cax.set_yticklabels(["1"], fontsize=8)
        elif tick_mode == "bottom":
            cax.set_yticks([0.0])
            cax.set_yticklabels(["0"], fontsize=8)
        else:
            cax.set_yticks([])
        cax.tick_params(axis="y", pad=1)
        for spine in cax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#666666")
        fig.text(
            bar_x + bar_w + 0.008,
            bar_bottom + bar_h / 2,
            metric_name,
            fontsize=9,
            ha="left",
            va="center",
        )

    add_metric_gradient(ax_pos.y1 - bar_h, "compute", METRIC_COLORS["compute"], tick_mode="top")
    add_metric_gradient(ax_pos.y1 - 2 * bar_h, "memory", METRIC_COLORS["memory"], tick_mode="none")
    add_metric_gradient(ax_pos.y1 - 3 * bar_h, "capacity", METRIC_COLORS["capacity"], tick_mode="bottom")

    plt.savefig(out_png)
    plt.close(fig)


def main():
    args = parse_args()

    log_dir = os.path.abspath(args.log_dir)
    if args.out_png:
        out_png = args.out_png
    elif args.plot_mode == "heatmap-qps-split":
        out_png = os.path.join(log_dir, "proxy_bottleneck_heatmap_qps_split.png")
    elif args.plot_mode == "heatmap":
        out_png = os.path.join(log_dir, "proxy_bottleneck_heatmap.png")
    else:
        out_png = os.path.join(log_dir, "proxy_bottleneck_2d.png")
    out_csv = args.out_csv or os.path.join(log_dir, "proxy_bottleneck_summary.csv")

    rows = aggregate_by_config(log_dir, eps=args.eps)
    if not rows:
        raise RuntimeError(
            "No valid monitor samples found. Please check log directory and filename pattern."
        )

    save_csv(rows, out_csv)
    if args.plot_mode == "heatmap":
        plot_single_heatmap(rows, out_png)
    elif args.plot_mode == "heatmap-qps-split":
        plot_single_heatmap_qps_split(rows, out_png)
    else:
        plot_2d_facets(rows, out_png)

    print(f"Parsed configs: {len(rows)}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved PNG: {out_png}")


if __name__ == "__main__":
    main()
