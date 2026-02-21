# import os
# import re
# from datetime import datetime
# from collections import defaultdict

# import matplotlib.pyplot as plt
# import numpy as np
# import argparse


# # =======================
# # 可调参数
# # =======================
# MAX_POINTS = 300          # 每个 decode 实例最多绘制的点数
# FIG_SIZE = (12, 6)
# DPI = 150

# MARKERS = {
#     "compute": "o",
#     "memory": "^",
#     "capacity": "s",
# }

# COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# # =======================
# # 日志解析
# # =======================
# TIME_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

# # ✅ 修复：不依赖字段顺序
# MONITOR_PATTERN = re.compile(
#     r"decode=(?P<decode>[\d\.]+:\d+).*?"
#     r"compute=(?P<compute>[\d\.]+).*?"
#     r"memory=(?P<memory>[\d\.]+).*?"
#     r"capacity=(?P<capacity>[\d\.]+)"
# )


import os
import re
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import argparse


MAX_POINTS = 300
FIG_SIZE = (13, 6)
DPI = 150

LINESTYLES = {
    "compute": "-",
    "memory": "--",
    "capacity": ":",
}

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

TIME_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
MONITOR_PATTERN = re.compile(
    r"decode=(?P<decode>[\d\.]+:\d+).*?"
    r"compute=(?P<compute>[\d\.]+).*?"
    r"memory=(?P<memory>[\d\.]+).*?"
    r"capacity=(?P<capacity>[\d\.]+)"
)


def parse_log(log_path):
    data = defaultdict(lambda: {
        "time": [],
        "compute": [],
        "memory": [],
        "capacity": [],
    })

    last_time = None

    with open(log_path) as f:
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
            data[d]["compute"].append(float(m.group("compute")))
            data[d]["memory"].append(float(m.group("memory")))
            data[d]["capacity"].append(float(m.group("capacity")))

    if not data:
        raise RuntimeError("No MONITOR data parsed")

    return data


def downsample(x, y, max_points):
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return [x[i] for i in idx], [y[i] for i in idx]


def plot_decode_load(log_path):
    data = parse_log(log_path)

    # 按端口号排序，映射 decode-0,1,2...
    decodes = sorted(data.keys(), key=lambda x: int(x.split(":")[-1]))
    decode_map = {d: f"decode-{i}" for i, d in enumerate(decodes)}

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    for i, d in enumerate(decodes):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

        for metric in ["compute", "memory", "capacity"]:
            t, v = downsample(data[d]["time"], data[d][metric], MAX_POINTS)

            ax.plot(
                t,
                v,
                color=color,
                linestyle=LINESTYLES[metric],
                linewidth=1.6,
                alpha=0.9,
                label=f"{decode_map[d]}-{metric}"
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Load Value")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)

    ax.legend(
        fontsize=8,
        ncol=3,
        bbox_to_anchor=(0.5, 1.18),
        loc="upper center",
        frameon=False
    )

    fig.autofmt_xdate()
    plt.tight_layout()

    out = os.path.join(os.path.dirname(log_path),
                       os.path.basename(log_path).replace(".log", "_all.png"))
    plt.savefig(out)
    plt.close()

    print(f"[OK] Saved {out}")



# =======================
# 主入口
# =======================
def main():
    parser = argparse.ArgumentParser(
        description="Plot decode load from WT MONITOR logs"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to proxy_timestamp.log"
    )

    args = parser.parse_args()
    plot_decode_load(args.log_path)


if __name__ == "__main__":
    main()
