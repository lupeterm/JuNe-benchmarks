"""CPU thread-scaling figures of the cavity assembly benchmark (NeoN, NeoN fused and JuNe on the
H200 node's CPUs, 1 to 64 threads), in the style of the GPU figures: colours by strategy kind,
hollow circles for NeoN, filled circles for fused NeoN, filled squares for JuNe.

  cpu_h200_threads<t>_gbs_neon        throughput over the number of cells, NeoN fused against NeoN
  cpu_h200_threads<t>_gbs_fused_june  throughput over the number of cells, NeoN fused against JuNe
  cpu_h200_threads<t>_june_speedup    speedup of JuNe over fused NeoN (fused time / JuNe time) per
                                      strategy kind, over the number of cells
                                      (one set per thread count)
  cpu_h200_speedup_n<n>              speedup over the thread count, one subplot per executor, at the
                                     cache-resident size where the serial throughput peaks and at
                                     the largest mesh

    python3 plot_cpu.py [results_dir]

reads june_neon/cavity_throughput_cpu.csv and writes pdf next to it.
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


def legends(ax, extra=(), top=0.99, shown=tuple(EXECUTORS)):
    """Executor markers (of the executors in shown) and strategy colours inside the axes, top left from
    height top downwards, below optional extra handles."""
    executors = [Line2D([], [], color="0.25", marker=m, linestyle="none", markersize=9,
                        markerfacecolor="0.25" if f else "none", label=e)
                 for e, (m, f) in EXECUTORS.items() if e in shown]
    ax.add_artist(ax.legend(handles=list(extra) + executors, loc="upper left", bbox_to_anchor=(0.01, top),
                            frameon=False, labelspacing=0.8))
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in KINDS.items()], loc="upper left",
              bbox_to_anchor=(0.01, top - 0.075 * (len(executors) + len(extra)) - 0.03), frameon=False,
              labelspacing=0.8)


def setup():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    return plt.subplots(figsize=(15, 6), constrained_layout=True)


def save(fig, out: Path):
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"))


def plot_throughput(data: pd.DataFrame, threads: int, column: str, ylabel: str, out: Path,
                    matrix: str = "Vec3", shown=tuple(EXECUTORS)):
    """Throughput over the number of cells of the strategies of the executors in shown."""
    fig, ax = setup()
    for strategy in (s for s in SERIES if s in set(data["strategy"]) and SERIES[s][1] in shown):
        col, m, filled = style(strategy)
        points = data.loc[data["strategy"].eq(strategy)].sort_values("cells")
        ax.plot(points["cells"], points[column], color=col, alpha=0.35, linewidth=1)
        ax.scatter(points["cells"], points[column], s=50, marker=m, edgecolors=col,
                   facecolors=col if filled else "none", linewidths=1.3, alpha=0.85)
    ax.set(xscale="log", xlabel="Number of cells", ylabel=ylabel)
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.set_ylim(bottom=0)
    ax.grid(True, which="minor", alpha=0.15)
    # the title sits top left, the legends below it
    ax.text(0.02, 0.97, f"{' vs '.join(shown)}, {threads} thread{'s' if threads > 1 else ''}, {matrix} matrix",
            transform=ax.transAxes, ha="left", va="top", fontsize=18, fontweight="bold", color="0.2")
    legends(ax, top=0.90, shown=shown)
    save(fig, out)


def plot_june_speedup(data: pd.DataFrame, threads: int, out: Path, matrix: str = "Vec3"):
    """Speedup of JuNe over fused NeoN per strategy kind: fused NeoN time / JuNe time, above 1 JuNe is
    faster."""
    fig, ax = setup()
    column = "time_median_ms" if "time_median_ms" in data else "time_mean_ms"
    times = data.set_index(["strategy", "n"])[column]
    cells = data.groupby("n")["cells"].first()
    lo, hi = 1.0, 1.0
    for kind, col in KINDS.items():
        fused = next(s for s, (k, e) in SERIES.items() if k == kind and e == "NeoN fused")
        june = next((s for s, (k, e) in SERIES.items() if k == kind and e == "JuNe"), None)
        if june is None or fused not in times.index.get_level_values(0) or june not in times.index.get_level_values(0):
            continue
        ratio = (times[fused] / times[june]).dropna().sort_index()
        lo, hi = min(lo, ratio.min()), max(hi, ratio.max())
        ax.plot(cells[ratio.index], ratio.values, color=col, alpha=0.35, linewidth=1)
        ax.scatter(cells[ratio.index], ratio.values, s=50, marker="s", color=col, alpha=0.85)
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.2)
    ax.set(xscale="log", yscale="log", xlabel="Number of cells", ylabel="Speedup of JuNe over fused NeoN")
    ax.set_ylim(lo / 1.5, hi * 4)  # headroom on top for the title
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, which="minor", alpha=0.15)
    ax.text(0.02, 0.97, f"JuNe vs NeoN fused, {threads} thread{'s' if threads > 1 else ''}, {matrix} matrix",
            transform=ax.transAxes, ha="left", va="top", fontsize=18, fontweight="bold", color="0.2")
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in KINDS.items()]
              + [Line2D([], [], color="0.4", linestyle="--", label="equal time")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False,
              labelspacing=0.8)  # right of the axes: the ratios cover every corner of the plot
    save(fig, out)


def plot_speedup(data: pd.DataFrame, n: int, out: Path, matrix: str = "Vec3"):
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
            column = "time_median_ms" if "time_median_ms" in size else "time_mean_ms"
            points = size.loc[size["strategy"].eq(strategy)].set_index("threads")[column].sort_index()
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
    axes[-1].text(0.96, 0.04, f"{n}\u00b3 cells, {matrix} matrix", transform=axes[-1].transAxes,
                  ha="right", va="bottom", fontsize=14, color="0.3")
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
    # the cache peak is searched among the small meshes: with the scalar matrix the serial throughput
    # keeps rising slowly with the mesh size, so the global maximum would sit next to the largest mesh
    cache = int(serial.loc[serial.index <= 100].idxmax())
    largest = int(data["n"].max())
    print(f"serial throughput peaks at n = {cache} ({serial[cache]:.1f} GB/s median), largest n = {largest}")
    return sorted({cache, largest})


def main(results: Path):
    all_data = pd.read_csv(results / "cavity_throughput_cpu.csv")
    all_data = all_data.loc[all_data["strategy"].isin(SERIES)]
    if "matrix" not in all_data:
        all_data["matrix"] = "Vec3"
    # the speedup sizes are shared by the matrix types; they come from the first one present
    sizes = peak_sizes(all_data.loc[all_data["matrix"].eq(sorted(all_data["matrix"].unique())[-1])])
    # one set of figures per matrix type: the Vec3 ones keep their names, scalar gets a suffix
    for matrix, data in all_data.groupby("matrix"):
        suffix = "" if matrix == "Vec3" else f"_{matrix}"
        for threads in sorted(data["threads"].unique()):
            sub = data.loc[data["threads"].eq(threads)]
            ylabel = "Matrix assembly throughput [GB/s]"
            plot_throughput(sub, threads, "throughput_gbs", ylabel,
                            results / f"cpu_h200_threads{threads}_gbs_neon{suffix}", matrix, ("NeoN", "NeoN fused"))
            plot_throughput(sub, threads, "throughput_gbs", ylabel,
                            results / f"cpu_h200_threads{threads}_gbs_fused_june{suffix}", matrix, ("NeoN fused", "JuNe"))
            plot_june_speedup(sub, threads, results / f"cpu_h200_threads{threads}_june_speedup{suffix}", matrix)
        for n in sizes:
            plot_speedup(data, n, results / f"cpu_h200_speedup_n{n}{suffix}", matrix)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
