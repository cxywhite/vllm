import argparse
import os
import re
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import numpy as np

TIME_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
MONITOR_PATTERN = re.compile(
    r"decode=(?P<decode>[\d\.]+:\d+).*?"
    r"bottle=(?P<bottle>[\d\.]+).*?"
    r"compute=(?P<compute>[\d\.]+).*?"
    r"memory=(?P<memory>[\d\.]+).*?"
    r"capacity=(?P<capacity>[\d\.]+)"
)

LINESTYLES = {
    "compute": "-",
    "memory": "--",
    "capacity": ":",
}

COLORS = {
    "compute": "#1f77b4",
    "memory": "#ff7f0e",
    "capacity": "#2ca02c",
}

SUBCAPTIONS = [
    "(a) compute bottle",
    "(b) memory bottle",
    "(c) capacity bottle",
]


def parse_log(log_path):
    data = defaultdict(lambda: {
        "time": [],
        "bottle": [],
        "compute": [],
        "memory": [],
        "capacity": [],
    })

    last_time = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            t = TIME_PATTERN.search(line)
            if t:
                last_time = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S")

            if "[WT][MONITOR]" not in line or last_time is None:
                continue

            m = MONITOR_PATTERN.search(line)
            if not m:
                continue

            d = m.group("decode")
            data[d]["time"].append(last_time)
            data[d]["bottle"].append(float(m.group("bottle")))
            data[d]["compute"].append(float(m.group("compute")))
            data[d]["memory"].append(float(m.group("memory")))
            data[d]["capacity"].append(float(m.group("capacity")))

    return data


def find_proxy_log(experiment_dir):
    log_dir = os.path.join(experiment_dir, "log")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Missing log dir: {log_dir}")

    candidates = sorted(
        [name for name in os.listdir(log_dir) if name.startswith("proxy_") and name.endswith(".log")]
    )
    if not candidates:
        raise FileNotFoundError(f"No proxy_*.log found in {log_dir}")

    return os.path.join(log_dir, candidates[0])


def pick_primary_decode(parsed_data):
    if not parsed_data:
        raise ValueError("No monitor data parsed from log.")
    return max(parsed_data.keys(), key=lambda key: len(parsed_data[key]["time"]))


def first_positive_index(values):
    for idx, value in enumerate(values):
        if value > 0:
            return idx
    return 0


def find_end_by_duration(times, start_idx, seconds_limit):
    if not times:
        return 0
    start_t = times[start_idx]
    end_idx = start_idx
    for idx in range(start_idx, len(times)):
        if (times[idx] - start_t).total_seconds() <= seconds_limit:
            end_idx = idx
        else:
            break
    return end_idx


def find_end_by_abs_mmss(times, start_idx, target_minute, target_second):
    if not times:
        return 0

    target = target_minute * 60 + target_second
    end_idx = start_idx

    for idx in range(start_idx, len(times)):
        current = times[idx].minute * 60 + times[idx].second
        if current <= target:
            end_idx = idx
        else:
            break

    return end_idx


def build_window_indices(times, compute_values, mode, target_len=None):
    start = first_positive_index(compute_values)

    if mode == "to_2630":
        if start >= len(times):
            return 0, max(len(compute_values) - 1, 0)
        end = find_end_by_abs_mmss(times, start, target_minute=26, target_second=30)
        return start, end

    if mode == "middle_match_first":
        if target_len is None or target_len <= 0:
            raise ValueError("target_len must be positive for middle_match_first mode")
        valid_len = len(compute_values) - start
        if valid_len <= 0:
            return 0, max(len(compute_values) - 1, 0)

        half = valid_len // 2
        middle_start = start + half - target_len // 2
        lower_bound = start
        upper_bound = max(start, len(compute_values) - target_len)
        middle_start = max(lower_bound, min(middle_start, upper_bound))
        end = min(middle_start + target_len - 1, len(compute_values) - 1)
        return middle_start, end

    if mode == "quarter_match_first":
        if target_len is None or target_len <= 0:
            raise ValueError("target_len must be positive for quarter_match_first mode")
        valid_len = len(compute_values) - start
        if valid_len <= 0:
            return 0, max(len(compute_values) - 1, 0)

        quarter_start = start + valid_len // 4
        lower_bound = start
        upper_bound = max(start, len(compute_values) - target_len)
        quarter_start = max(lower_bound, min(quarter_start, upper_bound))
        end = min(quarter_start + target_len - 1, len(compute_values) - 1)
        return quarter_start, end

    if mode == "match_first":
        if target_len is None or target_len <= 0:
            raise ValueError("target_len must be positive for match_first mode")
        end = min(start + target_len - 1, len(compute_values) - 1)
        return start, end

    raise ValueError(f"Unknown mode: {mode}")


def slice_decode_series(series, start, end):
    return {
        "time": series["time"][start:end + 1],
        "compute": series["compute"][start:end + 1],
        "memory": series["memory"][start:end + 1],
        "capacity": series["capacity"][start:end + 1],
    }


def calc_ylim(series):
    values = np.array(series["compute"] + series["memory"] + series["capacity"])
    if values.size == 0:
        return 0.0, 1.05

    min_v = float(np.min(values))
    max_v = float(np.max(values))
    span = max(max_v - min_v, 0.05)
    lower = max(0.0, min_v - 0.15 * span)
    upper = min(1.05, max_v + 0.20 * span)
    if upper - lower < 0.12:
        upper = min(1.05, lower + 0.12)
    return lower, upper


def render_triptych(experiment_payloads, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=220, sharey=False)
    formatter = mdates.DateFormatter("%M:%S")

    for i, payload in enumerate(experiment_payloads):
        ax = axes[i]
        for metric in ("compute", "memory", "capacity"):
            ax.plot(
                payload["series"]["time"],
                payload["series"][metric],
                linestyle=LINESTYLES[metric],
                color=COLORS[metric],
                linewidth=2.4,
                label=metric,
            )

        y_low, y_high = calc_ylim(payload["series"])
        ax.set_ylim(y_low, y_high)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="x", labelsize=9, rotation=20)
        ax.tick_params(axis="y", labelsize=9)
        if i == 1:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Time", fontsize=10)
        if i == 0:
            ax.set_ylabel("Load", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.legend(frameon=False, fontsize=9, loc="upper right")

        ax.text(
            0.5,
            -0.19,
            payload["subcaption"],
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="top",
        )

        ax.text(
            0.5,
            -0.31,
            payload["subtitle"],
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            va="top",
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35, wspace=0.20)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot 1x3 3D load figure from three experiments.")
    parser.add_argument(
        "--paper_dir",
        default="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/3DLoad/3Dload/paper",
        help="Root paper directory containing benchmark_* folders",
    )
    parser.add_argument(
        "--save_dir",
        default="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/3DLoad/3Dload/paper/fig",
        help="Directory to save generated figure",
    )
    args = parser.parse_args()

    experiments = [
        {
            "folder": "benchmark_np5000_rr70_20260212_152440_1p1d",
            "subtitle": "Distribution1 14~481, Qps 70req/s",
            "mode": "to_2630",
        },
        {
            "folder": "benchmark_np5000_rr1_mt8192_20260222_073624_1p1d",
            "subtitle": "Distribution2 16~8k, Qps 1req/s",
            "mode": "quarter_match_first",
        },
        {
            "folder": "benchmark_np5000_rr2_mt8192_20260222_030423_1p1d",
            "subtitle": "Distribution2 16~8k, Qps 2req/s",
            "mode": "quarter_match_first",
        },
    ]

    prepared = []
    base_len = None

    for exp in experiments:
        exp_dir = os.path.join(args.paper_dir, exp["folder"])
        log_path = find_proxy_log(exp_dir)
        parsed = parse_log(log_path)
        primary_decode = pick_primary_decode(parsed)
        series = parsed[primary_decode]

        if exp["mode"] == "to_2630":
            start, end = build_window_indices(series["time"], series["compute"], mode="to_2630")
            clipped = slice_decode_series(series, start, end)
            base_len = len(clipped["time"])
        else:
            start, end = build_window_indices(
                series["time"],
                series["compute"],
                mode=exp["mode"],
                target_len=base_len,
            )
            clipped = slice_decode_series(series, start, end)

        prepared.append(
            {
                "title": exp["folder"],
                "subcaption": SUBCAPTIONS[len(prepared)],
                "subtitle": exp["subtitle"],
                "series": clipped,
                "log_path": log_path,
                "decode": primary_decode,
                "window": (start, end),
            }
        )

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.save_dir, f"three_d_load_triptych_{timestamp}.png")

    render_triptych(prepared, output_path)

    print("Saved:", output_path)
    for p in prepared:
        print(
            f"{p['title']} | decode={p['decode']} | log={p['log_path']} | window={p['window']} | points={len(p['series']['time'])}"
        )


if __name__ == "__main__":
    main()
