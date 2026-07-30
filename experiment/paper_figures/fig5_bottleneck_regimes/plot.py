#!/usr/bin/env python3
"""
Figure 5: Dominant bottleneck regimes (motivation_figure2.pdf).

WARNING: The original plotting script and benchmark data for this figure
are LOST.

This script was reconstructed by extracting grid data directly from the
published PDF. All bottleneck values are hardcoded below.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


# ==========================================================
# Sweep axes
# ==========================================================

INPUT_LENS = [
    50, 100, 200, 500,
    1000, 2000, 4000, 6000
]

OUTPUT_LENS = [
    50, 100, 200, 500,
    1000, 2000, 4000, 6000
]


# ==========================================================
# Grid data  (extracted from the published PDF)
# 0 compute
# 1 memory
# 2 capacity
# ==========================================================

GRID_50_QPS1 = [
    [1,2,1,2,2,2,1,1],
    [1,2,1,2,2,2,1,1],
    [1,2,1,2,2,2,2,1],
    [1,2,1,2,2,2,2,2],
    [1,2,1,0,0,2,2,2],
    [1,2,1,0,0,2,2,2],
    [1,1,1,0,0,2,2,2],
    [1,1,1,0,0,2,2,2],
]


GRID_50_QPS2 = [
    [1,2,2,2,1,1,1,2],
    [1,2,2,2,2,1,1,2],
    [1,2,2,2,2,2,1,2],
    [1,2,2,2,2,2,2,2],
    [1,2,2,2,2,2,2,2],
    [1,0,0,2,2,2,2,2],
    [1,0,0,0,2,2,2,2],
    [1,0,0,0,2,2,2,2],
]


GRID_50_QPS4 = [
    [2,2,2,2,1,1,1,0],
    [2,2,2,2,2,1,1,2],
    [2,2,2,2,2,2,1,2],
    [2,2,2,2,2,2,1,2],
    [2,2,2,2,2,2,1,2],
    [0,2,2,2,2,2,1,2],
    [0,2,2,2,2,2,1,2],
    [0,2,0,2,2,2,1,1],
]


GRID_30_QPS1 = [
    [1,1,1,1,1,2,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [0,2,0,2,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
]


GRID_30_QPS2 = [
    [1,1,1,1,1,2,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
]


GRID_30_QPS4 = [
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,2],
    [1,1,1,1,1,1,1,2],
    [0,1,1,1,1,1,1,2],
    [1,1,1,1,1,1,1,2],
    [1,1,1,1,1,1,1,1],
]


GRID_DATA = {

    (50,1): GRID_50_QPS1,
    (50,2): GRID_50_QPS2,
    (50,4): GRID_50_QPS4,

    (30,1): GRID_30_QPS1,
    (30,2): GRID_30_QPS2,
    (30,4): GRID_30_QPS4,

}



# ==========================================================
# Plot
# ==========================================================


def main():

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    TPOT_VALUES = [50,30]
    QPS_VALUES = [1,2,4]


    cmap = colors.LinearSegmentedColormap.from_list(
        "bottleneck",
        [
            "#2F6BAE",
            "#DB7370",
            "#71B47F",
        ],
        N=256
    )


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2,5.6),
        dpi=300
    )


    X, Y = np.meshgrid(
        INPUT_LENS,
        OUTPUT_LENS
    )


    for r,tpot in enumerate(TPOT_VALUES):

        for c,qps in enumerate(QPS_VALUES):

            ax = axes[r,c]


            grid = np.array(
                GRID_DATA[(tpot,qps)]
            )


            im = ax.pcolormesh(
                X,
                Y,
                grid,
                cmap=cmap,
                vmin=0,
                vmax=2,
                shading="nearest"
            )


            ax.set_xscale("log")
            ax.set_yscale("log")


            ax.set_xticks(INPUT_LENS)
            ax.set_yticks(OUTPUT_LENS)


            if r == 1:
                ax.set_xticklabels(
                    INPUT_LENS,
                    fontsize=7,
                    rotation=45,
                    ha="right"
                )
            else:
                ax.set_xticklabels([])


            if c == 0:
                ax.set_yticklabels(
                    OUTPUT_LENS,
                    fontsize=7
                )
            else:
                ax.set_yticklabels([])


            ax.set_title(
                f"QPS = {qps}",
                fontsize=9
            )


            ax.tick_params(
                width=0.5,
                length=3
            )


    axes[0,0].set_ylabel(
        "Output Length",
        fontsize=9
    )

    axes[1,0].set_ylabel(
        "Output Length",
        fontsize=9
    )


    for ax in axes[1]:
        ax.set_xlabel(
            "Input Length",
            fontsize=9
        )


    # colorbar

    cbar = fig.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.03
    )

    cbar.set_ticks(
        [0,1,2]
    )

    cbar.set_ticklabels(
        [
            "compute",
            "memory",
            "capacity"
        ]
    )


    fig.tight_layout(
        rect=[0,0,0.93,1]
    )


    out = SCRIPT_DIR / "motivation_figure2.pdf"


    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved {out}"
    )



if __name__ == "__main__":
    main()
