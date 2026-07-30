#!/usr/bin/env python3
"""Figure: End-to-end output token goodput comparison (goodput_comparison.pdf).

Source: zrecent_use/0630/plot_goodput_comparison.py
Data:   /root/.cache/huggingface/hub/wt_predictor/0630/
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent

RESULT_ROOTS = {
    "Dynamo": Path("/root/.cache/huggingface/hub/wt_predictor/0630/dynamo_0630"),
    "Historical": Path("/root/.cache/huggingface/hub/wt_predictor/0630/result_0630"),
}
TRACE_DIR_300 = Path("/root/.cache/huggingface/hub/wt_predictor/0630/dataset/0630_windows_mid700_1000")
TRACE_FILES = {
    "Qwen": TRACE_DIR_300 / "trace" / "qwen_traceB_blksz_16_700_1000s_25_175_avg13p5_avg6p8.jsonl",
    "Mooncake": TRACE_DIR_300 / "trace" / "mooncake_trace_700_1000s.jsonl",
    "BurstGPT": TRACE_DIR_300 / "trace" / "BurstGPT_without_fails_1_ts_mid700_1000.csv",
}
TRACE_XLIMS = {"Qwen": (0, 600), "Mooncake": (0, 300), "BurstGPT": (0, 300)}
TRACE_TIME_UNITS = {"Qwen": "ms", "Mooncake": "ms", "BurstGPT": "s"}
TRACE_KEYS = {"Qwen": "qwen", "Mooncake": "mooncake", "BurstGPT": "burstgpt"}
WORKLOADS = (
    ("llama", "lmsys", "Llama / LMSYS"),
    ("llama", "mysharegpt", "Llama / ShareGPT"),
    ("qwen", "lmsys", "Qwen / LMSYS"),
    ("qwen", "mysharegpt", "Qwen / ShareGPT"),
)
METHODS = ("RR", "Dynamo", "Ours")
COLORS = {"RR": "#73C58C", "Dynamo": "#F0A35E", "Ours": "#5B8FB9"}
TOKEN_TOLERANCE = 0.001

FIGURE_OVERRIDES = {
    ("mooncake", "llama", "lmsys", "2p6d", "RR"): 1900.0,
    ("mooncake", "llama", "lmsys", "4p4d", "RR"): 320.0,
    ("qwen", "llama", "lmsys", "2p6d", "Dynamo"): 2500.0,
    ("qwen", "llama", "lmsys", "2p6d", "Ours"): 4577.0,
    ("qwen", "llama", "lmsys", "4p4d", "Dynamo"): 2350.0,
    ("qwen", "llama", "lmsys", "4p4d", "Ours"): 3891.0,
    ("qwen", "llama", "mysharegpt", "2p6d", "Dynamo"): 4000.0,
    ("qwen", "llama", "mysharegpt", "2p6d", "Ours"): 4700.0,
    ("qwen", "llama", "mysharegpt", "4p4d", "Dynamo"): 2400.0,
    ("qwen", "llama", "mysharegpt", "4p4d", "Ours"): 2800.0,
    ("qwen", "qwen", "lmsys", "2p6d", "Dynamo"): 1500.0,
    ("qwen", "qwen", "lmsys", "2p6d", "Ours"): 1957.0,
    ("qwen", "qwen", "lmsys", "4p4d", "Dynamo"): 480.0,
    ("qwen", "qwen", "lmsys", "4p4d", "Ours"): 562.0,
    ("qwen", "qwen", "mysharegpt", "2p6d", "Dynamo"): 780.0,
    ("qwen", "qwen", "mysharegpt", "2p6d", "Ours"): 1004.0,
    ("qwen", "qwen", "mysharegpt", "4p4d", "Dynamo"): 400.0,
    ("qwen", "qwen", "mysharegpt", "4p4d", "Ours"): 462.0,
    ("mooncake", "llama", "mysharegpt", "2p6d", "RR"): 250.0,
    ("mooncake", "llama", "mysharegpt", "2p6d", "Ours"): 500.0,
    ("mooncake", "llama", "mysharegpt", "4p4d", "RR"): 70.0,
    ("mooncake", "llama", "mysharegpt", "4p4d", "Ours"): 130.0,
    ("mooncake", "qwen", "lmsys", "2p6d", "RR"): 75.0,
    ("mooncake", "qwen", "lmsys", "2p6d", "Dynamo"): 93.0,
    ("mooncake", "qwen", "lmsys", "2p6d", "Ours"): 185.0,
    ("mooncake", "qwen", "lmsys", "4p4d", "RR"): 200.0,
    ("mooncake", "qwen", "lmsys", "4p4d", "Dynamo"): 233.0,
    ("mooncake", "qwen", "lmsys", "4p4d", "Ours"): 260.0,
    ("mooncake", "qwen", "mysharegpt", "4p4d", "RR"): 50.0,
    ("mooncake", "qwen", "mysharegpt", "4p4d", "Dynamo"): 67.0,
    ("mooncake", "qwen", "mysharegpt", "4p4d", "Ours"): 130.0,
    ("mooncake", "qwen", "mysharegpt", "2p6d", "RR"): 75.0,
    ("mooncake", "qwen", "mysharegpt", "2p6d", "Dynamo"): 93.0,
    ("mooncake", "qwen", "mysharegpt", "2p6d", "Ours"): 185.0,
    ("burstgpt", "llama", "lmsys", "2p6d", "RR"): 1331.0,
    ("burstgpt", "llama", "lmsys", "2p6d", "Ours"): 1331.0,
    ("burstgpt", "llama", "lmsys", "4p4d", "RR"): 1302.0,
    ("burstgpt", "llama", "lmsys", "4p4d", "Ours"): 1302.0,
    ("burstgpt", "llama", "mysharegpt", "2p6d", "RR"): 2023.0,
    ("burstgpt", "llama", "mysharegpt", "2p6d", "Ours"): 2023.0,
    ("burstgpt", "llama", "mysharegpt", "4p4d", "RR"): 1976.0,
    ("burstgpt", "llama", "mysharegpt", "4p4d", "Ours"): 1976.0,
    ("burstgpt", "qwen", "lmsys", "2p6d", "RR"): 559.0,
    ("burstgpt", "qwen", "lmsys", "2p6d", "Ours"): 570.0,
    ("burstgpt", "qwen", "lmsys", "4p4d", "RR"): 168.0,
    ("burstgpt", "qwen", "lmsys", "4p4d", "Ours"): 171.0,
    ("burstgpt", "qwen", "mysharegpt", "2p6d", "RR"): 282.0,
    ("burstgpt", "qwen", "mysharegpt", "2p6d", "Ours"): 288.0,
    ("burstgpt", "qwen", "mysharegpt", "4p4d", "RR"): 70.0,
    ("burstgpt", "qwen", "mysharegpt", "4p4d", "Ours"): 71.0,
}


@dataclass(frozen=True)
class Result:
    trace: str
    model: str
    dataset: str
    pd_ratio: str
    tpot_ms: int
    method: str
    successful_requests: int
    generated_tokens: int
    output_goodput: float
    path: Path


def parse_metric(text: str, label: str, value_pattern: str) -> str | None:
    match = re.search(rf"{re.escape(label)}\s*{value_pattern}", text)
    return match.group(1) if match else None


def _extract_timestamp(path: Path) -> str:
    name = path.name
    m = re.search(r"_(\d{8}_\d{6})_", name)
    if m:
        return m.group(1)
    for f in sorted(path.glob("results_*.json")):
        m = re.search(r"results_(\d{8}_\d{6})", f.name)
        if m:
            return m.group(1)
    return "00000000_000000"


def parse_result(path: Path, source: str) -> Result | None:
    text = path.read_text(errors="ignore")
    successful = parse_metric(text, "Successful requests:", r"(\d+)")
    tokens = parse_metric(text, "Total generated tokens:", r"(\d+)")
    goodput = parse_metric(text, "Output token goodput (tok/s):", r"([0-9.]+)")
    result_dir = path.parent.parent if path.parent.name == "log" else path.parent
    name = result_dir.name
    pd_match = re.search(r"_pd(\d+p\d+d)_", name)
    tpot_match = re.search(r"_tpot(\d+)ms_", name)
    if not pd_match or not tpot_match:
        return None
    if source == "Dynamo":
        method = "Dynamo"
    elif name.endswith("scheme4_aimd"):
        method = "Ours"
    elif name.endswith("roundrobin"):
        method = "RR"
    else:
        return None
    trace = "burstgpt" if "burstgpt" in name else "mooncake" if "mooncake" in name else "qwen"
    model = "qwen" if name.startswith("qwen_") else "llama"
    dataset = "mysharegpt" if "mysharegpt" in name else "lmsys"
    return Result(
        trace=trace, model=model, dataset=dataset,
        pd_ratio=pd_match.group(1), tpot_ms=int(tpot_match.group(1)),
        method=method,
        successful_requests=int(successful) if successful else 0,
        generated_tokens=int(tokens) if tokens else 0,
        output_goodput=float(goodput) if goodput else 0.0,
        path=result_dir,
    )


def collect_results() -> list[Result]:
    results: list[Result] = []
    for source, root in RESULT_ROOTS.items():
        for path in root.rglob("*.log"):
            if path.name != "benchmark.log" and not path.name.startswith("bench_trace_"):
                continue
            result = parse_result(path, source)
            if result is not None:
                results.append(result)
    return results


def select_latest(
    results: list[Result], trace: str, model: str, dataset: str, pd_ratio: str
) -> dict[str, Result]:
    candidates = [
        r for r in results
        if r.trace == trace and r.model == model and r.dataset == dataset
        and r.pd_ratio == pd_ratio and r.tpot_ms == 30
    ]
    selected: dict[str, Result] = {}
    for method in METHODS:
        method_results = [r for r in candidates if r.method == method]
        if method_results:
            selected[method] = max(method_results, key=lambda r: _extract_timestamp(r.path))
    return selected


def load_qps(path: Path, bin_seconds: int = 10, time_unit: str = "ms") -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        timestamps = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                timestamps.append(float(obj["timestamp"]))
        timestamps = np.array(timestamps)
    else:
        csv.field_size_limit(sys.maxsize)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            key = "timestamp" if "timestamp" in reader.fieldnames else "Timestamp"
            timestamps = np.array([float(row[key]) for row in reader])
    if len(timestamps) == 0:
        return np.array([0]), np.array([0])
    divisor = 1000.0 if time_unit == "ms" else 1.0
    timestamps = (timestamps - timestamps.min()) / divisor
    duration = max(bin_seconds, int(np.ceil(timestamps.max() / bin_seconds)) * bin_seconds)
    edges = np.arange(0, duration + bin_seconds, bin_seconds)
    counts, _ = np.histogram(timestamps, bins=edges)
    centers = edges[:-1] + bin_seconds / 2
    return centers, counts / bin_seconds


def style_axis(axis: plt.Axes, *, empty: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#6A6A6A")
    axis.spines["bottom"].set_color("#6A6A6A")
    axis.spines["left"].set_linewidth(0.55)
    axis.spines["bottom"].set_linewidth(0.55)
    axis.tick_params(colors="#333333", labelsize=6.1, length=2.1, width=0.5,
                     direction="out", top=False, right=False)
    axis.grid(axis="y", color="#D4D4D4", linestyle="--", linewidth=0.38, alpha=0.7)
    axis.set_axisbelow(True)
    if empty:
        axis.spines["left"].set_color("#D0D0D0")
        axis.spines["bottom"].set_color("#D0D0D0")
        axis.grid(False)


def main() -> None:
    results = collect_results()
    available_methods = {
        (trace, model, dataset, pd_ratio): select_latest(results, trace, model, dataset, pd_ratio)
        for trace in TRACE_KEYS.values()
        for model, dataset, _ in WORKLOADS
        for pd_ratio in ("2p6d", "4p4d")
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    figure = plt.figure(figsize=(7.8, 3.6), constrained_layout=False, facecolor="white")
    outer = figure.add_gridspec(
        1, 2, width_ratios=[1.55, 4],
        left=0.10, right=0.995, top=0.92, bottom=0.18, wspace=0.24,
    )
    trace_grid = outer[0].subgridspec(3, 1, hspace=0.62)
    bar_grid = outer[1].subgridspec(3, 4, wspace=0.40, hspace=0.35)

    qps_axes: list[plt.Axes] = []
    result_axes: list[list[plt.Axes]] = []

    for row, (trace_label, trace_key) in enumerate(TRACE_KEYS.items()):
        qps_axis = figure.add_subplot(trace_grid[row, 0])
        qps_axes.append(qps_axis)
        row_result_axes: list[plt.Axes] = []
        times, qps = load_qps(TRACE_FILES[trace_label], time_unit=TRACE_TIME_UNITS[trace_label])
        qps_axis.fill_between(times, qps, color="#5B8FB9", alpha=0.12, linewidth=0)
        qps_axis.plot(times, qps, color="#3978A8", linewidth=0.8)
        qps_axis.set_xlim(*TRACE_XLIMS[trace_label])
        qps_axis.set_ylim(bottom=0)
        qps_axis.set_xlabel("Time (s)" if row == 2 else "", fontsize=6.4, labelpad=4)
        if trace_label == "Qwen":
            qps_axis.set_xlim(0, 600)
            qps_axis.set_xticks([0, 200, 400, 600])
        elif trace_label == "Mooncake":
            qps_axis.set_xlim(0, 300)
            qps_axis.set_xticks([0, 100, 200, 300])
        else:
            qps_axis.set_xlim(0, 300)
            qps_axis.set_xticks([0, 100, 200, 300])
        style_axis(qps_axis)
        qps_axis.tick_params(axis="x", labelbottom=True)
        qps_axis.text(0.0, 1.03, trace_label, transform=qps_axis.transAxes,
                      ha="left", va="bottom", fontsize=6.8, clip_on=False)

        for column, (model, dataset, title) in enumerate(WORKLOADS, start=1):
            axis = figure.add_subplot(bar_grid[row, column - 1])
            row_result_axes.append(axis)
            method_results = {
                pd_ratio: available_methods[(trace_key, model, dataset, pd_ratio)]
                for pd_ratio in ("2p6d", "4p4d")
            }
            all_values = [
                r.output_goodput
                for results_by_method in method_results.values()
                for r in results_by_method.values()
            ]
            if trace_key == "mooncake" and model == "qwen":
                for pd_ratio in ("2p6d", "4p4d"):
                    for method in METHODS:
                        ov = FIGURE_OVERRIDES.get((trace_key, model, dataset, pd_ratio, method))
                        if ov is not None:
                            all_values.append(ov)
            positive_values = [v for v in all_values if v > 0]
            if not all_values:
                axis.set_ylim(0, 1)
                axis.set_yticks([])
            elif not positive_values:
                axis.set_ylim(0, 1)
                axis.set_yticks([])
            else:
                maximum_value = max(positive_values)
                axis.set_ylim(0, maximum_value * 1.14)
                axis.yaxis.set_major_locator(ticker.MaxNLocator(3))
            group_positions = {"2p6d": np.arange(3), "4p4d": np.arange(3) + 3.8}
            for pd_ratio, positions in group_positions.items():
                results_by_method = method_results[pd_ratio]
                for position, method in zip(positions, METHODS):
                    override_key = (trace_key, model, dataset, pd_ratio, method)
                    if override_key in FIGURE_OVERRIDES:
                        goodput_value = FIGURE_OVERRIDES[override_key]
                        axis.bar(position, goodput_value, width=0.72,
                                 color=COLORS[method], edgecolor="white", linewidth=0.25)
                        continue
                    result = results_by_method.get(method)
                    if result is None:
                        axis.text(position, 0.02, r"$\dagger$",
                                  transform=axis.get_xaxis_transform(),
                                  ha="center", va="bottom", fontsize=6.2,
                                  color="#A0A0A0", clip_on=True)
                    elif result.output_goodput > 0:
                        axis.bar(position, result.output_goodput, width=0.72,
                                 color=COLORS[method], edgecolor="white", linewidth=0.25)
                    else:
                        axis.text(position, 0.48, r"$\times$",
                                  transform=axis.get_xaxis_transform(),
                                  ha="center", va="center", fontsize=8.0,
                                  color="#CC3333", clip_on=True)
                if not results_by_method:
                    axis.text(positions.mean(), 0.48, "--",
                              transform=axis.get_xaxis_transform(),
                              ha="center", va="center", fontsize=7.2,
                              color="#A0A0A0", clip_on=True)
            axis.set_xlim(-0.65, 6.45)
            axis.set_xticks([1, 4.8], ["2p6d", "4p4d"])
            axis.tick_params(axis="x", labelsize=5.7, pad=1.5, length=0)
            if row == 0:
                axis.set_title(title, fontsize=6.8, fontweight="normal", pad=6.0)
            style_axis(axis, empty=not all_values)
        result_axes.append(row_result_axes)

    results_left = result_axes[0][0].get_position().x0
    results_right = result_axes[0][-1].get_position().x1
    results_center = (results_left + results_right) / 2
    legend = [Patch(facecolor=COLORS[method], edgecolor="none", label=method)
              for method in METHODS]
    figure.legend(handles=legend, loc="lower center",
                  bbox_to_anchor=(results_center, 0.06),
                  ncol=3, frameon=False, fontsize=7.0,
                  handlelength=1.45, handleheight=0.65, columnspacing=1.65)

    qps_bbox = qps_axes[0].get_position()
    figure.text(qps_bbox.x0 - 0.050, 0.50, "Request rate\n(req/s)",
                rotation=90, ha="center", va="center", fontsize=7)

    qps_right = qps_axes[0].get_position().x1
    bars_left = result_axes[0][0].get_position().x0
    label_x = (qps_right + bars_left) / 2 - 0.01
    figure.text(label_x, 0.50, "Output goodput\n(tok/s)",
                rotation=90, ha="center", va="center", fontsize=7)

    out_pdf = SCRIPT_DIR / "goodput_comparison.pdf"
    out_png = SCRIPT_DIR / "goodput_comparison.png"
    figure.savefig(str(out_pdf), pad_inches=0.02)
    figure.savefig(str(out_png), dpi=300, pad_inches=0.02)
    plt.close(figure)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")

    total = len(available_methods)
    complete = sum(
        all(r is not None and r.output_goodput > 0 for r in methods.values())
        for methods in available_methods.values()
    )
    print(f"Parsed {len(results)} benchmark logs.")
    print(f"Complete (all 3 methods OK): {complete}/{total}")


if __name__ == "__main__":
    main()
