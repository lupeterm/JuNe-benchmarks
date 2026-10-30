"""NeoN's and JuNe's GPU assembly throughput next to the icoFoam baseline on 64 MPI ranks, over the
number of cells. Colours follow the strategy kind as in the NeoFOAM plot (face-based blue, cell-based
orange, batched green, global face-based red); NeoN is drawn with hollow circles, fused NeoN with
filled circles, JuNe with filled squares and icoFoam with black triangles. The legend sits right of the axes, ordered by the throughput on the largest case.

    python3 plot_combined.py [results_dir]

reads results/h200/cavity_throughput.csv (NeoFOAM benchmark) and results/cavity_icofoam.csv
(icofoam/parse_results.py) and writes results/cavity_combined_{gbs,mcells}.{svg,png}.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter

# strategy in the CSVs -> (legend label, colour, marker, filled)
TAB10 = plt.get_cmap("tab10").colors
SERIES = {
    "NeoN face-based": ("Face-Based", TAB10[0], "o", False),
    "NeoN cell-based": ("Cell-Based", TAB10[1], "o", False),
    "NeoN batched": ("Batched Face-Based", TAB10[2], "o", False),
    "NeoN fused face-based": ("Fused Face-Based", TAB10[0], "o", True),
    "NeoN fused cell-based": ("Fused Cell-Based", TAB10[1], "o", True),
    "NeoN fused batched": ("Fused Batched Face-Based", TAB10[2], "o", True),
    "JuNe face-based": ("Face-Based", TAB10[0], "s", True),
    "JuNe global face-based": ("Global Face-Based", TAB10[3], "s", True),
    "JuNe cell-based": ("Cell-Based", TAB10[1], "s", True),
    "JuNe batched": ("Batched Face-Based", TAB10[2], "s", True),
    "icoFoam 64 ranks": ("icoFoam", "0.15", "^", True),
}
GROUPS = {  # legend inside the axes -> (marker, filled)
    "NeoN, NVIDIA H200": ("o", False),
    "NeoN fused, NVIDIA H200": ("o", True),
    "JuNe, NVIDIA H200": ("s", True),
    "icoFoam, 64 MPI ranks (2x AMD EPYC 9555)": ("^", True),
}


def handle(label, col, m, filled) -> Line2D:
    return Line2D([], [], color=col, marker=m, linestyle="none", markersize=9,
                  markerfacecolor=col if filled else "none", label=label)


def plot(data: pd.DataFrame, column: str, ylabel: str, out: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    largest = data.loc[data["cells"].eq(data["cells"].max())].groupby("strategy")[column].mean()
    order = sorted((s for s in SERIES if s in set(data["strategy"])), key=lambda s: -largest.get(s, 0.0))
    for strategy in order:
        label, col, m, filled = SERIES[strategy]
        points = data.loc[data["strategy"].eq(strategy)].sort_values("cells")
        ax.plot(points["cells"], points[column], color=col, alpha=0.35, linewidth=1)
        ax.scatter(points["cells"], points[column], s=50, marker=m, edgecolors=col,
                   facecolors=col if filled else "none", linewidths=1.3, alpha=0.85)
    ax.set(xscale="log", xlabel="Number of cells", ylabel=ylabel)
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.set_ylim(bottom=0)
    ax.grid(True, which="minor", alpha=0.15)
    ax.add_artist(ax.legend(handles=[handle(g, "0.25", m, f) for g, (m, f) in GROUPS.items()],
                            loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False, labelspacing=0.8))
    legend = ax.legend(handles=[handle(SERIES[s][0], *SERIES[s][1:]) for s in order],
                       loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False,
                       labelspacing=1.6, handletextpad=0.8)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", bbox_extra_artists=[legend])
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=[legend])
    print("wrote", out.with_suffix(".pdf"), "and .svg")


def main(results: Path):
    gpu = pd.read_csv(results / "h200" / "cavity_throughput.csv")
    cpu = pd.read_csv(results / "cavity_icofoam.csv")
    data = pd.concat([gpu, cpu], ignore_index=True)
    data = data.loc[data["strategy"].isin(SERIES)]  # drops nothing today, guards against new strategies
    plot(data, "throughput_gbs", "Matrix assembly throughput [GB/s]", results / "cavity_combined_gbs")
    # plot(data, "throughput_mcells_s", "Matrix assembly throughput [million cells/s]", results / "cavity_combined_mcells")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
