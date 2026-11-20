"""div + laplacian (no ddt) of the momentum equation on the lid-driven cavity, scalar matrix, fp64, H200:
every strategy of NeoN, NeoN fused and JuNe, JuNe with relaxed atomics that compile to REDG (JuNe.jl
src/types.jl relaxed_add!, like Kokkos' atomic_add/sub) instead of Atomix's acquire-release atomics.
NeoFOAM benchmarks/june/red/cavity_red.job: test_cavity.cpp, five fenced assemblies per strategy
including the reset, the first dropped, median reported; throughput = nnz * 8 B / median time.

    python3 plot_red.py [log_dir]   reads <log_dir>/cavity_<n>.log (default ./h200), writes
                                    h200/cavity_red.csv and
      h200/cavity_red_gbs.pdf       throughput of NeoN fused and JuNe against the number of cells
      h200/cavity_red_speedup.pdf   speedup of JuNe over NeoN fused of the same kind
(non-fused NeoN and global face-based are measured and in the csv, but not plotted)
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter, FixedLocator, FormatStrFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_scalar_variants import EXECUTORS, KINDS, SERIES, draw  # noqa: E402

TIME = re.compile(r"^Assembly time \((.+) (\d+)\) = (\d+) \[us\]", re.M)


def parse(logs: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(logs.glob("cavity_*.log")):
        text = log.read_text()
        n = int(log.stem.split("_")[1])
        nnz = int(re.search(r"^nnz: (\d+)", text, re.M).group(1))
        for strategy, run, us in TIME.findall(text):
            if int(run) > 0:  # run 0 compiles JuNe's kernels
                rows.append({"n": n, "cells": n**3, "nnz": nnz, "strategy": strategy, "ms": int(us) / 1e3})
    raw = pd.DataFrame(rows)
    data = raw.groupby(["n", "cells", "nnz", "strategy"])["ms"].agg(time_median_ms="median", runs="count").reset_index()
    data["throughput_gbs"] = data["nnz"] * 8 / (data["time_median_ms"] / 1e3) / 1e9
    return data


def axes_cells(ax, cells):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(cells))
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Cells")


SHOWN = {k: c for k, c in KINDS.items() if k != "Global Face-Based"}  # strategy kinds in the figures
COMPARED = [s for s, (k, e) in SERIES.items() if e in ("NeoN Fused", "JuNe") and k in SHOWN]
BLANK = Line2D([], [], linestyle="none", label=" ")  # pads the first legend column to the kinds


def legend(ax, executors):
    handles = [Line2D([], [], color="0.25", marker=EXECUTORS[e][0], linestyle="none", markersize=9,
                      markerfacecolor="0.25" if EXECUTORS[e][1] else "none", label=e) for e in executors]
    handles += [BLANK] * (len(SHOWN) - len(handles))
    handles += [Patch(facecolor=c, label=k) for k, c in SHOWN.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncols=2)


def plot_gbs(data, here):
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for strategy in COMPARED:
        draw(ax, data[data["strategy"].eq(strategy)], strategy, "throughput_gbs")
    axes_cells(ax, sorted(data["cells"].unique()))
    ax.set_ylabel("Throughput [GB/s]")
    ax.set_ylim(0, data.loc[data["strategy"].isin(COMPARED), "throughput_gbs"].max() * 1.35)
    legend(ax, ["NeoN Fused", "JuNe"])
    ax.set_title("div + laplacian, scalar matrix, H200, JuNe with REDG atomics")
    fig.savefig(here / "cavity_red_gbs.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", here / "cavity_red_gbs.pdf")


def plot_speedup(data, here):
    t = data.set_index(["n", "strategy"])["time_median_ms"]
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for strategy, (kind, executor) in SERIES.items():
        if executor != "JuNe" or kind not in SHOWN:
            continue
        name = strategy.removeprefix("JuNe ")
        pts = [(n**3, t[(n, "NeoN fused " + name)] / t[(n, strategy)]) for n in sorted(data["n"].unique())
               if (n, "NeoN fused " + name) in t and (n, strategy) in t]
        ax.plot(*zip(*pts), color=KINDS[kind], marker="s", markersize=7)
    ax.axhline(1.0, color="0.3", linewidth=1)
    axes_cells(ax, sorted(data["cells"].unique()))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Speedup of JuNe over NeoN fused")
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in SHOWN.items()], loc="upper left",
              frameon=False, ncols=2)
    ax.set_ylim(0.5, 1.25)
    ax.set_title("div + laplacian, scalar matrix, H200, JuNe with REDG atomics")
    fig.savefig(here / "cavity_red_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", here / "cavity_red_speedup.pdf")


if __name__ == "__main__":
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "h200"
    out = Path(__file__).resolve().parent / "h200"
    out.mkdir(exist_ok=True)
    data = parse(logs)
    data.to_csv(out / "cavity_red.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11})
    plot_gbs(data, out)
    plot_speedup(data, out)
    print(data.pivot(index="strategy", columns="n", values="time_median_ms").round(2).to_string())
