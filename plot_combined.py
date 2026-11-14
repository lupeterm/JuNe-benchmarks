"""NeoN's and JuNe's GPU assembly throughput next to the icoFoam baseline on 64 MPI ranks, over the
number of cells. Colours follow the strategy kind as in the NeoFOAM plot (face-based blue, cell-based
orange, batched green, global face-based red); NeoN is drawn with hollow circles, fused NeoN with
filled circles, JuNe with filled squares and icoFoam with black triangles. The executor markers and
the strategy colours are explained by two legends inside the axes, top left.

    python3 plot_combined.py [results_dir]

reads june_neon/h200/cavity_throughput.csv (NeoFOAM benchmark) and june_neon/cavity_icofoam.csv
(icofoam/parse_results.py) and writes june_neon/cavity_combined_gbs.pdf.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter, FixedLocator

TAB10 = plt.get_cmap("tab10").colors
KINDS = {  # strategy kind -> colour
    "Face-Based": TAB10[0],
    "Global Face-Based": TAB10[3],
    "Cell-Based": TAB10[1],
    "Batched Face-Based": TAB10[2],
}
EXECUTORS = {  # executor -> (marker, filled)
    "NeoN": ("o", False),
    "NeoN Fused": ("o", True),
    "JuNe": ("s", True),
    "icoFoam": ("^", True),
}
SERIES = {  # strategy in the CSVs -> (kind, executor)
    "NeoN face-based": ("Face-Based", "NeoN"),
    "NeoN global face-based": ("Global Face-Based", "NeoN"),
    "NeoN cell-based": ("Cell-Based", "NeoN"),
    "NeoN batched": ("Batched Face-Based", "NeoN"),
    "NeoN fused face-based": ("Face-Based", "NeoN Fused"),
    "NeoN fused global face-based": ("Global Face-Based", "NeoN Fused"),
    "NeoN fused cell-based": ("Cell-Based", "NeoN Fused"),
    "NeoN fused batched": ("Batched Face-Based", "NeoN Fused"),
    "JuNe face-based": ("Face-Based", "JuNe"),
    "JuNe global face-based": ("Global Face-Based", "JuNe"),
    "JuNe cell-based": ("Cell-Based", "JuNe"),
    "JuNe batched": ("Batched Face-Based", "JuNe"),
    "icoFoam 64 ranks": (None, "icoFoam"),  # no kind: drawn in dark grey
}
ICOFOAM_COLOR = "0.15"
CELL_TICKS = [1e3, 1e4, 1e5, 1e6, 1e7, 6.4e7]  # 64 M = 400^3, the largest mesh, instead of 100 M


def style(strategy: str):
    kind, executor = SERIES[strategy]
    m, filled = EXECUTORS[executor]
    return KINDS[kind] if kind else ICOFOAM_COLOR, m, filled


def executor_handles():
    return [Line2D([], [], color="0.25", marker=m, linestyle="none", markersize=9,
                   markerfacecolor="0.25" if filled else "none", label=e)
            for e, (m, filled) in EXECUTORS.items()]


def kind_handles():
    return [Patch(facecolor=c, label=k) for k, c in KINDS.items()]


def plot(data: pd.DataFrame, column: str, ylabel: str, out: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    for strategy in (s for s in SERIES if s in set(data["strategy"])):
        col, m, filled = style(strategy)
        points = data.loc[data["strategy"].eq(strategy)].sort_values("cells")
        ax.plot(points["cells"], points[column], color=col, alpha=0.35, linewidth=1)
        ax.scatter(points["cells"], points[column], s=50, marker=m, edgecolors=col,
                   facecolors=col if filled else "none", linewidths=1.3, alpha=0.85)
    ax.set(xscale="log", xlabel="Number of cells", ylabel=ylabel)
    ax.xaxis.set_major_locator(FixedLocator(CELL_TICKS))
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.set_ylim(bottom=0)
    ax.grid(True, which="minor", alpha=0.15)
    ax.add_artist(ax.legend(handles=executor_handles(), loc="upper left", bbox_to_anchor=(0.01, 0.99),
                            frameon=False, labelspacing=0.8))
    ax.legend(handles=kind_handles(), loc="upper left", bbox_to_anchor=(0.01, 0.70), frameon=False,
              labelspacing=0.8)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out.with_suffix(".pdf"))


def main(results: Path):
    gpu = pd.read_csv(results / "h200" / "cavity_throughput.csv")
    if "matrix" in gpu:  # the scalar matrix of the segregated momentum equation, as icoFoam's
        gpu = gpu.loc[gpu["matrix"].eq("scalar")]
    cpu = pd.read_csv(results / "cavity_icofoam.csv")
    data = pd.concat([gpu, cpu], ignore_index=True)
    data = data.loc[data["strategy"].isin(SERIES)]
    plot(data, "throughput_gbs", "Matrix assembly throughput [GB/s]", results / "cavity_combined_gbs")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
