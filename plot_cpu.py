"""CPU figures of the cavity assembly benchmark: NeoN, NeoN Fused and JuNe on the H200 node's AMD EPYC
9555 CPUs, 1 to 64 threads. Colours, markers and the per-thread plots are those of the GPU scalar
figures (plot_scalar_variants.py, plot_scalar_pairs.py): colours by strategy kind, hollow circles for
NeoN, filled circles for NeoN Fused, squares for JuNe; the title top left inside the axes with the
legend below it, a column for the throughputs and a row for the speedup, and the y range raised until
no point or line reaches them.

  cpu_h200_threads<t>_gbs_neon        throughput over the number of cells, NeoN against NeoN Fused
  cpu_h200_threads<t>_gbs_fused_june  throughput over the number of cells, NeoN Fused against JuNe
  cpu_h200_threads<t>_june_speedup    speedup of JuNe over NeoN Fused per strategy kind
  cpu_h200_threads<t>_pairs           the three above stacked
                                      (one set per thread count)
  cpu_h200_speedup_n<n>               speedup over the thread count, one subplot per executor, at the
                                      cache-resident size where the serial throughput peaks and at
                                      the largest mesh
  cpu_h200_efficiency_n<n>            parallel efficiency (speedup / threads) at the same sizes

    python3 plot_cpu.py [results_dir]

reads june_neon/cavity_throughput_cpu.csv and writes pdf next to it, with the matrix type as a
suffix for any matrix but Vec3.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_scalar_pairs import figures, save, style_setup
from plot_scalar_variants import EXECUTORS, KINDS, SERIES, style

CPU = "AMD EPYC 9555"  # the CPUs of the H200 node
PAIRS = {  # plot of plot_scalar_pairs.figures -> file name part
    "neon_vs_fused": "gbs_neon",
    "fused_vs_june": "gbs_fused_june",
    "june_over_fused": "june_speedup",
    "pairs": "pairs",
}


def plot_speedup(data: pd.DataFrame, n: int, out: Path, matrix: str = "Vec3", efficiency: bool = False):
    """Speedup over the thread count for one mesh size, one subplot per executor:
    serial time / time with t threads. With efficiency, the parallel efficiency instead:
    speedup / t, on a linear axis with the ideal at 1."""
    style_setup()
    plt.rcParams.update({"axes.titlesize": 17})
    fig, axes = plt.subplots(1, len(EXECUTORS), figsize=(15, 5.5), sharey=True, constrained_layout=True)
    size = data.loc[data["n"].eq(n)]
    threads = sorted(size["threads"].unique())
    column = "time_median_ms" if "time_median_ms" in size else "time_mean_ms"
    bottom, top = 1, 0
    for ax, executor in zip(axes, EXECUTORS):
        ax.plot(threads, [1] * len(threads) if efficiency else threads, color="0.5", linestyle="--",
                linewidth=1.2, label="ideal")
        for strategy in (s for s in SERIES if s in set(size["strategy"]) and SERIES[s][1] == executor):
            col = style(strategy)[0]
            points = size.loc[size["strategy"].eq(strategy)].set_index("threads")[column].sort_index()
            speedup = points.loc[1] / points
            if efficiency:
                speedup /= speedup.index
            bottom, top = min(bottom, speedup.min()), max(top, speedup.max())
            ax.plot(speedup.index, speedup.values, color=col, alpha=0.35, linewidth=1)
            ax.scatter(speedup.index, speedup.values, s=50, edgecolors=col,
                       facecolors=col, linewidths=1.3, alpha=0.85)
        ax.set(xscale="log", xlabel="Threads", title=executor)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(t) for t in threads])
        if not efficiency:
            ax.set_yscale("log")
            ax.set_yticks(threads)
            ax.set_yticklabels([str(t) for t in threads])
        ax.minorticks_off()
    handles = ([Line2D([], [], color="0.5", linestyle="--", linewidth=1.2, label="ideal")]
               + [Patch(facecolor=c, label=k) for k, c in KINDS.items()])
    if efficiency:  # every curve starts at 1 in the top left: legend in a row below the subplots
        axes[0].set_ylim(0, max(1.0, top) * 1.08)
        axes[0].set_ylabel("Parallel efficiency")
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    else:
        # one doubling above the ideal at the most threads, for the legend, and room below the slowest
        axes[0].set_ylim(min(bottom, 1) / 1.4, max(top, threads[-1]) * 2)
        axes[0].set_ylabel("Speedup over 1 thread")
        axes[0].legend(handles=handles, loc="upper left", frameon=False, labelspacing=0.8)
    fig.suptitle(f"{CPU}, {n}³ cells, {matrix} matrix", fontsize=18, fontweight="bold", color="0.2")
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
    # one set of figures per matrix type: the Vec3 ones keep their names, the others get a suffix
    for matrix, data in all_data.groupby("matrix"):
        suffix = "" if matrix == "Vec3" else f"_{matrix}"
        for threads in sorted(data["threads"].unique()):
            style_setup()
            figures(data.loc[data["threads"].eq(threads)],
                    f"{CPU}, {threads} thread{'s' if threads > 1 else ''}, {matrix} matrix",
                    {plot: results / f"cpu_h200_threads{threads}_{part}{suffix}" for plot, part in PAIRS.items()},
                    fixed=True)
        for n in sizes:
            plot_speedup(data, n, results / f"cpu_h200_speedup_n{n}{suffix}", matrix)
            plot_speedup(data, n, results / f"cpu_h200_efficiency_n{n}{suffix}", matrix, efficiency=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
