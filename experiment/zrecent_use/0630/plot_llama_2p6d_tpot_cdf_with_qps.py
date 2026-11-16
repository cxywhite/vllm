#!/usr/bin/env python3
"""Combined figure: fig9 leftmost QPS + Llama/LMSYS 2P6D goodput + TPOT CDF.

Layout (3 rows x 4 cols), one row per trace (Qwen / Mooncake / BurstGPT):
    [ request-rate QPS ] [ output goodput bars ] [ full TPOT CDF ] [ zoom @30 ms ]
Left col reproduces fig9's request-rate panels; 2nd col shows the Llama/LMSYS
2P6D output-goodput result (RR / Dynamo / Ours); 3rd/4th show the TPOT CDF and
its zoom at the 30 ms SLO.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

SLO_TPOT_MS = 30.0
Z99 = 2.3263478740408408
ROOT = Path("/root/.cache/huggingface/hub/wt_predictor/0630")

# ---- fig9 leftmost QPS panel data ----
TRACE_DIR = ROOT / "dataset/0630_windows_mid700_1000"
TRACE_FILES = {
    "Qwen": TRACE_DIR / "trace/qwen_traceB_blksz_16_700_1000s_25_175_avg13p5_avg6p8.jsonl",
    "Mooncake": TRACE_DIR / "trace/mooncake_trace_700_1000s.jsonl",
    "BurstGPT": TRACE_DIR / "trace/BurstGPT_without_fails_1_ts_mid700_1000.csv",
}
TRACE_XLIMS = {"Qwen": (0, 600), "Mooncake": (0, 300), "BurstGPT": (0, 300)}
TRACE_TIME_UNITS = {"Qwen": "ms", "Mooncake": "ms", "BurstGPT": "s"}

# ---- Llama/LMSYS 2P6D output goodput (tok/s), exactly fig9's column ----
GOODPUT = {
    "Qwen": {"RR": 2108.85, "Dynamo": 2500.0, "Ours": 2979.27},
    "Mooncake": {"RR": 1900.0, "Dynamo": 1906.61, "Ours": 2378.31},
    "BurstGPT": {"RR": 1331.0, "Dynamo": 1330.75, "Ours": 1331.0},
}

# ---- Llama/LMSYS 2P6D output throughput (tok/s), annotated above goodput bars ----
THROUGHPUT = {
    "Qwen": {"RR": 9900.0, "Dynamo": 10300.0, "Ours": 10700.0},
    "Mooncake": {"RR": 6100.0, "Dynamo": 6400.0, "Ours": 6700.0},
    "BurstGPT": {"RR": 1314.87, "Dynamo": 1339.52, "Ours": 1339.52},
}

# ---- TPOT CDF spec ----
SPECS = {
    ("Qwen", "RR"): ("goodput", 33.91, 2108.85, 9920.15),
    ("Qwen", "Dynamo"): ("goodput", 34.80, 2500.0, 9920.0),
    ("Qwen", "Ours"): ("goodput", 32.24, 2979.27, 10660.72),
    ("Mooncake", "RR"): ("goodput", 30.79, 1900.0, 6542.75),
    ("Mooncake", "Dynamo"): ("goodput", 32.50, 1906.61, 6213.31),
    ("Mooncake", "Ours"): ("goodput", 31.05, 2378.31, 6523.86),
    ("BurstGPT", "RR"): ("measured", ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260709_174420_roundrobin/dataset_result/test_results_20260709_175227.csv"),
    ("BurstGPT", "Ours"): ("measured", ROOT / "result_0630/tpot30/llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260709_175243_scheme4_aimd/dataset_result/test_results_20260709_180114.csv"),
    ("BurstGPT", "Dynamo"): ("p99", 16.55, 20.05),
}

TRACE_ORDER = ["Qwen", "Mooncake", "BurstGPT"]
METHODS = ["RR", "Dynamo", "Ours"]
COLORS = {"RR": "#73C58C", "Dynamo": "#F0A35E", "Ours": "#5B8FB9"}
SLO_COLOR = "#444444"
METHOD_LABELS = {"RR": "RR", "Dynamo": "Dynamo", "Ours": "Ours (Presage)"}
LINE_STYLES = {"RR": "-", "Dynamo": "--", "Ours": "-"}


def load_qps(path, bin_seconds=10, time_unit="ms"):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        timestamps = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                timestamps.append(float(json.loads(line)["timestamp"]))
        timestamps = np.array(timestamps)
    else:
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            key = "timestamp" if "timestamp" in reader.fieldnames else "Timestamp"
            timestamps = np.array([float(row[key]) for row in reader])
    divisor = 1000.0 if time_unit == "ms" else 1.0
    timestamps = (timestamps - timestamps.min()) / divisor
    duration = max(bin_seconds, int(np.ceil(timestamps.max() / bin_seconds)) * bin_seconds)
    edges = np.arange(0, duration + bin_seconds, bin_seconds)
    counts, _ = np.histogram(timestamps, bins=edges)
    centers = edges[:-1] + bin_seconds / 2
    return centers, counts / bin_seconds


def load_tpot_ms(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    lat = pd.to_numeric(df["latency"], errors="coerce")
    ttft = pd.to_numeric(df["ttft"], errors="coerce")
    out = pd.to_numeric(df["output_tokens"], errors="coerce")
    valid = (out > 1) & lat.notna() & ttft.notna() & (lat >= ttft)
    return ((lat - ttft) / (out - 1) * 1000.0)[valid].to_numpy()


def cdf_curve(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def fit_mean_p99(mean_ms, p99_ms):
    L = np.log(p99_ms / mean_ms)
    s = Z99 - np.sqrt(Z99 ** 2 - 2.0 * L)
    m = np.log(mean_ms) - 0.5 * s * s
    return m, s


def fit_mean_compliance(mean_ms, compliance, slo_ms=30.0):
    mu = float(mean_ms)
    c = float(np.clip(compliance, 1e-4, 1 - 1e-4))
    ln_slo = np.log(slo_ms)

    def f(sigma):
        m = np.log(mu) - 0.5 * sigma * sigma
        return norm.cdf((ln_slo - m) / sigma) - c

    lo, hi = 0.005, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    s = 0.5 * (lo + hi)
    m = np.log(mu) - 0.5 * s * s
    return m, s


def lognormal_curve(m, s, xmax=60.0):
    x = np.linspace(0.01, xmax, 600)
    y = norm.cdf((np.log(x) - m) / s)
    return x, y


def get_curve(spec):
    kind = spec[0]
    if kind == "measured":
        tpot = load_tpot_ms(spec[1])
        return cdf_curve(tpot)[0], cdf_curve(tpot)[1]
    if kind == "p99":
        m, s = fit_mean_p99(spec[1], spec[2])
        return lognormal_curve(m, s)
    m, s = fit_mean_compliance(spec[1], spec[2] / spec[3])
    return lognormal_curve(m, s)


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(colors="#333333", labelsize=8, length=2.5, width=0.6, direction="out")
    ax.grid(axis="y", color="#DDDDDD", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "Liberation Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(
        3, 4, figsize=(6.8, 5.5),
        gridspec_kw={"width_ratios": [1.0, 0.80, 1.20, 0.85],
                     "hspace": 0.55, "wspace": 0.68},
    )

    for row, trace in enumerate(TRACE_ORDER):
        ax_qps = axes[row, 0]
        ax_gp = axes[row, 1]
        ax_full = axes[row, 2]
        ax_zoom = axes[row, 3]

        # ---- col 1: request-rate QPS (fig9 leftmost) ----
        times, qps = load_qps(TRACE_FILES[trace], time_unit=TRACE_TIME_UNITS[trace])
        ax_qps.fill_between(times, qps, color="#5B8FB9", alpha=0.12, linewidth=0)
        ax_qps.plot(times, qps, color="#3978A8", linewidth=0.8)
        ax_qps.set_xlim(*TRACE_XLIMS[trace])
        ax_qps.set_ylim(bottom=0)
        ax_qps.set_xticks([0, 300, 600] if trace == "Qwen" else [0, 150, 300])
        ax_qps.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_qps.text(0.0, 1.04, trace, transform=ax_qps.transAxes,
                    ha="left", va="bottom", fontsize=9, fontweight="bold")
        style_axis(ax_qps)
        if row == 1:
            ax_qps.set_ylabel("Request rate (req/s)", fontsize=9, labelpad=3)
        if row == 2:
            ax_qps.set_xlabel("Time (s)", fontsize=9, labelpad=3)

        # ---- col 2: Llama/LMSYS 2P6D output goodput + throughput ----
        vals = [GOODPUT[trace][m] for m in METHODS]
        bars = ax_gp.bar(METHODS, vals, width=0.6, color=[COLORS[m] for m in METHODS],
                         edgecolor="white", linewidth=0.4)
        thr = [THROUGHPUT[trace][m] for m in METHODS]
        ax_gp.plot(range(len(METHODS)), thr, color="#8E44AD", marker="o", ms=4.0,
                   lw=1.1, linestyle="-", zorder=3, markeredgecolor="white",
                   markeredgewidth=0.5,
                   label="Throughput" if row == 0 else None)
        ax_gp.set_ylim(0, max(thr) * 1.15)
        ax_gp.set_xticks([0, 1, 2])
        ax_gp.set_xticklabels(["RR", "Dyn", "Ours"], fontsize=7)
        ax_gp.tick_params(axis="x", labelsize=7)
        style_axis(ax_gp)
        if row == 1:
            ax_gp.set_ylabel("Output (tok/s)", fontsize=9, labelpad=3)
        if row == 2:
            ax_gp.set_xlabel("Method", fontsize=9, labelpad=3)

        # ---- col 3: full TPOT CDF ----
        curves = {m: get_curve(SPECS[(trace, m)]) for m in METHODS}
        comp = {m: float(np.interp(SLO_TPOT_MS, curves[m][0], curves[m][1])) for m in METHODS}
        for method in METHODS:
            x, y = curves[method]
            ax_full.plot(x, y, color=COLORS[method], lw=1.6,
                         linestyle=LINE_STYLES[method],
                         label=METHOD_LABELS[method] if row == 0 else None,
                         solid_capstyle="round")
        ax_full.axvline(SLO_TPOT_MS, color=SLO_COLOR, linestyle="--", lw=1.0,
                        alpha=0.85, zorder=1, label="SLO (30 ms)" if row == 0 else None)
        ax_full.set_xlim(0, 60)
        ax_full.set_ylim(0, 1.02)
        ax_full.set_xticks([0, 15, 30, 45, 60])
        ax_full.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        style_axis(ax_full)
        if row == 1:
            ax_full.set_ylabel("CDF", fontsize=9, labelpad=3)
        if row == 2:
            ax_full.set_xlabel("Per-request TPOT (ms)", fontsize=9, labelpad=3)

        # ---- col 4: zoom around 30 ms SLO ----
        # curves are nearly coincident here, so draw them faint and mark each
        # method's compliance level with a horizontal guide + a dot at 30 ms.
        for method in METHODS:
            x, y = curves[method]
            ax_zoom.plot(x, y, color=COLORS[method], lw=1.3,
                         linestyle=LINE_STYLES[method], alpha=0.9, zorder=2)
        for method in METHODS:
            ax_zoom.axhline(comp[method], color=COLORS[method], linestyle="-",
                            lw=1.1, alpha=0.75, zorder=3)
            ax_zoom.plot([SLO_TPOT_MS], [comp[method]], marker="o", ms=6.5,
                         color=COLORS[method], zorder=5,
                         markeredgecolor="white", markeredgewidth=0.8)
        ax_zoom.axvline(SLO_TPOT_MS, color=SLO_COLOR, linestyle="--", lw=1.0,
                        alpha=0.85, zorder=1)
        ax_zoom.set_xlim(26, 34)
        ax_zoom.set_xticks([26, 30, 34])
        vals_c = list(comp.values())
        lo, hi = min(vals_c), max(vals_c)
        pad = 0.015
        ax_zoom.set_ylim(lo - pad, hi + pad)
        style_axis(ax_zoom)
        if row == 1:
            ax_zoom.set_ylabel("CDF (zoom)", fontsize=9, labelpad=3)
        if row == 2:
            ax_zoom.set_xlabel("TPOT (ms)", fontsize=9, labelpad=3)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.14, wspace=0.68)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    handles += h2
    labels += l2
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.56, 0.01), columnspacing=0.9,
               handlelength=2.0, handletextpad=0.4)

    out_png = Path(__file__).with_suffix(".png")
    out_pdf = Path(__file__).with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
