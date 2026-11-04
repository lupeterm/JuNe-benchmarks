"""CPU thread-scaling figures of the cavity assembly benchmark (NeoN, NeoN fused and JuNe on the
H200 node's CPUs, 1 to 64 threads), in the style of the GPU figures: colours by strategy kind,
hollow circles for NeoN, filled circles for fused NeoN, filled squares for JuNe.

  cpu_h200_threads<t>_{gbs,mcells}   throughput over the number of cells, one figure per thread count
  cpu_h200_speedup_n<n>              speedup over the thread count, one subplot per executor, at the
                                     cache-resident size where the serial throughput peaks and at
                                     the largest mesh

    python3 plot_cpu.py [results_dir]

reads june_neon/cavity_throughput_cpu.csv and writes svg and pdf next to it.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter

TAB10 = plt.get_cmap("tab10").colors
KINDS = {  # strategy kind -> colour
    "Face-Based": TAB10[0],
    "Global Face-Based": TAB10[3],
    "Cell-Based": TAB10[1],
    "Batched Face-Based": TAB10[2],
}
EXECUTORS = {  # executor -> (marker, filled)
    "NeoN": ("o", False),
    "NeoN fused": ("o", True),
    "JuNe": ("s", True),
}
SERIES = {  # strategy in the CSV -> (kind, executor)
    "NeoN face-based": ("Face-Based", "NeoN"),
    "NeoN global face-based": ("Global Face-Based", "NeoN"),
    "NeoN cell-based": ("Cell-Based", "NeoN"),
    "NeoN batched": ("Batched Face-Based", "NeoN"),
    "NeoN fused face-based": ("Face-Based", "NeoN fused"),
    "NeoN fused global face-based": ("Global Face-Based", "NeoN fused"),
    "NeoN fused cell-based": ("Cell-Based", "NeoN fused"),
    "NeoN fused batched": ("Batched Face-Based", "NeoN fused"),
    "JuNe face-based": ("Face-Based", "JuNe"),
    "JuNe global face-based": ("Global Face-Based", "JuNe"),
    "JuNe cell-based": ("Cell-Based", "JuNe"),
    "JuNe batched": ("Batched Face-Based", "JuNe"),
}


def style(strategy: str):
    kind, executor = SERIES[strategy]
    m, filled = EXECUTORS[executor]
    return KINDS[kind], m, filled


def legends(ax, extra=()):
    """Executor markers and strategy colours inside the axes, top left, below optional extra handles."""
    executors = [Line2D([], [], color="0.25", marker=m, linestyle="none", markersize=9,
                        markerfacecolor="0.25" if f else "none", label=e) for e, (m, f) in EXECUTORS.items()]
    ax.add_artist(ax.legend(handles=list(extra) + executors, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                            frameon=False, labelspacing=0.8))
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in KINDS.items()], loc="upper left",
              bbox_to_anchor=(0.01, 0.99 - 0.075 * (len(executors) + len(extra)) - 0.03), frameon=False,
              labelspacing=0.8)


def setup():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    return plt.subplots(figsize=(15, 6), constrained_layout=True)


def save(fig, out: Path):
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".svg"), "and .pdf")


def plot_throughput(data: pd.DataFrame, threads: int, column: str, ylabel: str, out: Path):
    fig, ax = setup()
    for strategy in (s for s in SERIES if s in set(data["strategy"])):
        col, m, filled = style(strategy)
        points = data.loc[data["strategy"].eq(strategy)].sort_values("cells")
        ax.plot(points["cells"], points[column], color=col, alpha=0.35, linewidth=1)
        ax.scatter(points["cells"], points[column], s=50, marker=m, edgecolors=col,
                   facecolors=col if filled else "none", linewidths=1.3, alpha=0.85)
    ax.set(xscale="log", xlabel="Number of cells", ylabel=ylabel)
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.set_ylim(bottom=0)
    ax.grid(True, which="minor", alpha=0.15)
    ax.text(0.98, 0.97, f"{threads} thread{'s' if threads > 1 else ''}", transform=ax.transAxes,
            ha="right", va="top", fontsize=18, fontweight="bold", color="0.2")
    legends(ax)
    save(fig, out)


def plot_speedup(data: pd.DataFrame, n: int, out: Path):
    """Speedup over the thread count for one mesh size, one subplot per executor:
    serial time / time with t threads."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 17, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    fig, axes = plt.subplots(1, len(EXECUTORS), figsize=(15, 5.5), sharey=True, constrained_layout=True)
    size = data.loc[data["n"].eq(n)]
    threads = sorted(size["threads"].unique())
    for ax, executor in zip(axes, EXECUTORS):
        ax.plot(threads, threads, color="0.5", linestyle="--", linewidth=1.2, label="ideal")
        for strategy in (s for s in SERIES if s in set(size["strategy"]) and SERIES[s][1] == executor):
            col, m, filled = style(strategy)
            points = size.loc[size["strategy"].eq(strategy)].set_index("threads")["time_mean_ms"].sort_index()
            speedup = points.loc[1] / points
            ax.plot(speedup.index, speedup.values, color=col, alpha=0.35, linewidth=1)
            ax.scatter(speedup.index, speedup.values, s=50, edgecolors=col,
                       facecolors=col, linewidths=1.3, alpha=0.85)
        ax.set(xscale="log", yscale="log", xlabel="Threads", title=executor)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(t) for t in threads])
        ax.set_yticks(threads)
        ax.set_yticklabels([str(t) for t in threads])
        ax.minorticks_off()
    axes[0].set_ylabel("Speedup over 1 thread")
    axes[0].legend(handles=[Line2D([], [], color="0.5", linestyle="--", linewidth=1.2, label="ideal")]
                   + [Patch(facecolor=c, label=k) for k, c in KINDS.items()],
                   loc="upper left", frameon=False, labelspacing=0.8)
    # axes[1].text(0.04, 0.96, f"{n}\u00b3 = {n**3:,} cells", transform=axes[1].transAxes, ha="left",
                #  va="top", fontsize=16, fontweight="bold", color="0.2")
    save(fig, out)


def peak_sizes(data: pd.DataFrame) -> list[int]:
    """The mesh sizes that show the two throughput peaks: the cache-resident size where the serial
    median throughput over the strategies peaks, and the largest mesh, where the threaded runs reach
    their plateau."""
    serial = data.loc[data["threads"].eq(1)].groupby("n")["throughput_gbs"].median()
    cache = int(serial.idxmax())
    largest = int(data["n"].max())
    print(f"serial throughput peaks at n = {cache} ({serial[cache]:.1f} GB/s median), largest n = {largest}")
    return sorted({cache, largest})


def main(results: Path):
    data = pd.read_csv(results / "cavity_throughput_cpu.csv")
    data = data.loc[data["strategy"].isin(SERIES)]
    for threads in sorted(data["threads"].unique()):
        sub = data.loc[data["threads"].eq(threads)]
        plot_throughput(sub, threads, "throughput_gbs", "Matrix assembly throughput [GB/s]",
                        results / f"cpu_h200_threads{threads}_gbs")
        plot_throughput(sub, threads, "throughput_mcells_s", "Matrix assembly throughput [million cells/s]",
                        results / f"cpu_h200_threads{threads}_mcells")
    for n in peak_sizes(data):
        plot_speedup(data, n, results / f"cpu_h200_speedup_n{n}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
