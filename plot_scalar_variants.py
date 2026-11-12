"""Three versions of the scalar-matrix assembly throughput figure on the H200, stacked vertically so
they can be compared side by side:

  1. the current figure: all twelve strategies, one legend entry per strategy on the right, ordered
     by the throughput at the largest mesh;
  2. no NeoN (non-fused) curves: fused NeoN and JuNe as a speedup over the non-fused NeoN strategy
     of the same kind at the same mesh;
  3. all strategies, but the legend names each strategy kind once (by colour), below the marker
     legend for the executor; both inside the axes.

Colours follow the strategy kind (face-based blue, global face-based orange, cell-based green,
batched red); NeoN is drawn with hollow circles, fused NeoN with filled circles, JuNe with squares.

    python3 plot_scalar_variants.py [results_dir]

reads june_neon/h200/cavity_throughput.csv and writes
june_neon/h200/cavity_throughput_gbs_scalar_variants.pdf.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter, FixedLocator, FormatStrFormatter, NullLocator

TAB10 = plt.get_cmap("tab10").colors
KINDS = {  # strategy kind -> colour, as in the current scalar figure
    "Face-Based": TAB10[0],
    "Global Face-Based": TAB10[1],
    "Cell-Based": TAB10[2],
    "Batched Face-Based": TAB10[3],
}
EXECUTORS = {  # executor -> (marker, filled, legend prefix)
    "NeoN": ("o", False, ""),
    "NeoN Fused": ("o", True, "Fused "),
    "JuNe": ("s", True, ""),
}
SERIES = {  # strategy in the CSV -> (kind, executor)
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
}
CELL_TICKS = [1e3, 1e4, 1e5, 1e6, 1e7, 6.4e7]  # 64 M = 400^3, the largest mesh, instead of 100 M
BASELINE = {kind: s for s, (kind, e) in SERIES.items() if e == "NeoN"}  # kind -> non-fused NeoN


def style(strategy: str):
    kind, executor = SERIES[strategy]
    m, filled, prefix = EXECUTORS[executor]
    return KINDS[kind], m, filled, prefix + kind


def draw(ax, points: pd.DataFrame, strategy: str, column: str):
    col, m, filled, label = style(strategy)
    points = points.sort_values("cells")
    ax.plot(points["cells"], points[column], color=col, alpha=0.35, linewidth=1)
    return ax.scatter(points["cells"], points[column], s=50, marker=m, edgecolors=col,
                      facecolors=col if filled else "none", linewidths=1.3, alpha=0.85, label=label)


def executor_legend(ax, executors, anchor=(0.01, 0.88)):
    handles = [Line2D([], [], color="0.25", marker=EXECUTORS[e][0], linestyle="none", markersize=9,
                      markerfacecolor="0.25" if EXECUTORS[e][1] else "none", label=e)
               for e in executors]
    ax.add_artist(ax.legend(handles=handles, loc="upper left", bbox_to_anchor=anchor,
                            frameon=False, labelspacing=0.8))


def kind_legend(ax, anchor=(1.01, 1.0), labelspacing=1.2):
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in KINDS.items()], loc="upper left",
              bbox_to_anchor=anchor, frameon=False, labelspacing=labelspacing)


def title(ax, text: str):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", fontsize=17, fontweight="bold",
            color="0.2")


def current(ax, data: pd.DataFrame):
    """All strategies, one legend entry each, ordered by throughput at the largest mesh."""
    last = data.loc[data["n"].eq(data["n"].max())].set_index("strategy")["throughput_gbs"]
    handles = {s: draw(ax, data.loc[data["strategy"].eq(s)], s, "throughput_gbs") for s in SERIES}
    order = sorted(handles, key=lambda s: -last.get(s, 0))
    executor_legend(ax, EXECUTORS)
    ax.legend(handles=[handles[s] for s in order], loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, labelspacing=1.2)
    ax.set(ylabel="Matrix Assembly Throughput [GB/s]")
    ax.set_ylim(bottom=0)
    title(ax, "(a) current: all strategies")


def speedup(ax, data: pd.DataFrame):
    """Fused NeoN and JuNe over the non-fused NeoN strategy of the same kind."""
    base = data.set_index(["strategy", "n"])["throughput_gbs"]
    for s, (kind, executor) in SERIES.items():
        if executor == "NeoN":
            continue
        points = data.loc[data["strategy"].eq(s)].copy()
        points["speedup"] = points["throughput_gbs"].to_numpy() / base.loc[BASELINE[kind]].reindex(points["n"]).to_numpy()
        draw(ax, points, s, "speedup")
    ax.axhline(1, color="0.3", linewidth=1.2, linestyle="--", zorder=0)
    executor_legend(ax, ["NeoN Fused", "JuNe"], anchor=(1.01, 0.45))  # the small meshes reach the top left
    kind_legend(ax)
    ax.set(ylabel="Speedup over non-fused NeoN")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 9)  # headroom for the title
    ax.yaxis.set_major_locator(FixedLocator([0.75, 1, 1.5, 2, 3, 4, 6]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(NullLocator())
    title(ax, "(b) speedup over non-fused NeoN of the same kind")


def reduced(ax, data: pd.DataFrame):
    """All strategies, each kind named once in the legend; both legends inside the axes."""
    for s in SERIES:
        draw(ax, data.loc[data["strategy"].eq(s)], s, "throughput_gbs")
    executor_legend(ax, EXECUTORS)
    kind_legend(ax, anchor=(0.01, 0.66), labelspacing=0.8)  # inside, below the executors
    ax.set(ylabel="Matrix Assembly Throughput [GB/s]")
    ax.set_ylim(bottom=0)
    title(ax, "(c) all strategies, reduced legend")


def main(results: Path):
    data = pd.read_csv(results / "h200" / "cavity_throughput.csv")
    data = data.loc[data["matrix"].eq("scalar") & data["strategy"].isin(SERIES)]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    current(axes[0], data)
    speedup(axes[1], data)
    reduced(axes[2], data)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(CELL_TICKS))
        ax.xaxis.set_major_formatter(EngFormatter())
        ax.grid(True, which="minor", alpha=0.15)
        ax.tick_params(labelbottom=True)
    axes[-1].set_xlabel("Cells")
    fig.suptitle("NVIDIA H200, scalar matrix", fontsize=19, fontweight="bold", color="0.2")
    out = results / "h200" / "cavity_throughput_gbs_scalar_variants"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
