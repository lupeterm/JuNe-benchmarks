"""Mixed-precision outlook: JuNe's GPU assembly of the momentum equation ddt + div + laplacian
(scalar matrix) on the lid-driven cavity, every JuNe GPU strategy, in the precision configurations of
JuNe.jl src/precision.jl (NeoFOAM test_cavity.cpp with CAVITY_PRECISION=1 CAVITY_DDT=1):

    whole    all operators and the matrix in fp64, fp32, fp16, bf16        labels fp64, fp32, ...
    reduce   one operator in fp32 or bf16, the rest and the matrix in fp64  labels ddt-fp32, div-bf16, ...
    alone    one operator alone, it and the matrix in each precision        labels only-ddt-fp64, ...

Five repetitions of every measurement, the first dropped, median reported. Speedups: whole and reduce
against the fp64 assembly of the whole equation, alone against the same operator alone in fp64.

    python3 plot_precision.py [log_dir] [--no-global]
                                          reads <log_dir>/cavity_<n>.log (default ./h200), writes
                                          precision.csv, precision_deviation.csv and the figures
      precision_whole_speedup.pdf     whole assembly: speedup over fp64 against the number of cells
      precision_whole_gbs.pdf         whole assembly: throughput (nnz * sizeof(T) / time)
      precision_whole_time.pdf        whole assembly: median assembly time
      precision_reduce_speedup.pdf    one operator reduced
      precision_alone_speedup.pdf     each operator alone
      precision_overheads.pdf         allocation, conversion and back-conversion against the assembly

With --no-global the figures leave out the global face-based strategy and are written with a _noglobal
suffix next to the full ones; the CSVs are not rewritten.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter, FixedLocator, FuncFormatter, NullFormatter

BYTES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2}
PRECISIONS = list(BYTES)
OPERATORS = {"ddt": "ddt", "div": "div", "lap": "laplacian"}
STRATEGIES = ["face-based", "global face-based", "cell-based", "batched"]
MARKERS = dict(zip(STRATEGIES, "osD^"))  # per strategy, so that leaving one out keeps the others'
TAG = ""  # appended to every figure name: "_noglobal" without the global face-based strategy
LABELS = {"face-based": "Face-Based", "global face-based": "Global Face-Based", "cell-based": "Cell-Based",
          "batched": "Batched Face-Based"}
COLOURS = {"fp64": "#08306b", "fp32": "#2171b5", "fp16": "#e6550d", "bf16": "#fdae6b"}
OP_COLOURS = {"ddt": "#2ca02c", "div": "#1f77b4", "lap": "#d62728"}
LARGE = {"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18, "legend.fontsize": 16,
         "xtick.labelsize": 15, "ytick.labelsize": 15}  # the figures that go into the paper
TIMES = re.compile(r"^(Assembly|Allocation|Conversion|Back-conversion) time \(JuNe (.+?) (\S+) (\d+)\) = (\d+) \[us\]", re.M)
DEV = re.compile(r"Precision deviation \(JuNe (.+?) (\S+)\): matrix ([0-9.e+-]+) rhs ([0-9.e+-]+)")


def describe(label):
    """configuration label -> (mode, operator, precision)"""
    parts = label.split("-")
    if parts[0] == "only":
        return "alone", parts[1], parts[2]
    if len(parts) == 2:
        return "reduce", parts[0], parts[1]
    return "whole", None, parts[0]


def parse(logs: Path):
    rows, devs = [], []
    for log in sorted(logs.glob("cavity_*.log")):
        text = log.read_text()
        nnz = re.search(r"^nnz: (\d+)", text, re.M)
        if not nnz:
            continue
        n, nnz = int(log.stem.split("_")[1]), int(nnz.group(1))
        for kind, strategy, label, run, us in TIMES.findall(text):
            if int(run) > 0:  # run 0 compiles the kernels of the configuration
                rows.append({"n": n, "cells": n**3, "nnz": nnz, "strategy": strategy, "config": label,
                             "kind": kind.lower().replace("-", "_"), "ms": int(us) / 1e3})
        for strategy, label, mat, rhs in DEV.findall(text):
            devs.append({"n": n, "strategy": strategy, "config": label, "matrix": float(mat), "rhs": float(rhs)})
    raw = pd.DataFrame(rows)
    data = (raw.groupby(["n", "cells", "nnz", "strategy", "config", "kind"])["ms"].median()
            .unstack("kind").add_suffix("_ms").reset_index())
    data[["mode", "operator", "precision"]] = pd.DataFrame([describe(c) for c in data["config"]], index=data.index)
    data["throughput_gbs"] = data["nnz"] * data["precision"].map(BYTES) / (data["assembly_ms"] / 1e3) / 1e9
    base = data.set_index(["n", "strategy", "config"])["assembly_ms"]
    def reference(r):
        return base[(r.n, r.strategy, f"only-{r.operator}-fp64" if r["mode"] == "alone" else "fp64")]
    data["speedup_over_fp64"] = [reference(r) / r.assembly_ms for _, r in data.iterrows()]
    return data.sort_values(["strategy", "config", "n"]), pd.DataFrame(devs)


def style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11})


def finish(ax, ylabel=None, one=False, largest=False):
    ax.set_xscale("log")
    if largest:  # label the largest mesh, 400^3 cells, too
        ax.xaxis.set_major_locator(FixedLocator([1e5, 1e6, 1e7, 6.4e7]))
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.grid(True, which="minor", alpha=0.15)
    if one:
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)
    if ylabel:
        ax.set_ylabel(ylabel)


def line(ax, sub, column, colour, marker="o", label=None, linestyle="-"):
    sub = sub.sort_values("cells")
    ax.plot(sub["cells"], sub[column], color=colour, marker=marker, markersize=6, linewidth=1.5,
            linestyle=linestyle, label=label)


def save(fig, path):
    path = path.with_name(path.stem + TAG + path.suffix)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_whole(data, here, column, ylabel, name, one, paper=False):
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4.5 * len(STRATEGIES), 4.4), sharey=True,
                             constrained_layout=True)
    logy = column == "assembly_ms"  # times span two orders of magnitude over the sweep
    whole = data[data["mode"].eq("whole")]
    for ax, strategy in zip(axes, STRATEGIES):
        for prec in PRECISIONS:
            line(ax, whole[whole["strategy"].eq(strategy) & whole["precision"].eq(prec)], column, COLOURS[prec], label=prec)
        finish(ax, ylabel if strategy == STRATEGIES[0] else None, one, paper)
        if logy:  # ticks at 1, 2, 5 times every power of ten, labelled with plain numbers
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([m * 10.0**e for e in range(-2, 3) for m in (1, 2, 5)]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(LABELS[strategy])
        ax.set_xlabel("Cells")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False,
               title=None if paper else "whole momentum assembly in one precision")
    save(fig, here / name)


def plot_reduce(data, here):
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4.5 * len(STRATEGIES), 4.4), sharey=True,
                             constrained_layout=True)
    red = data[data["mode"].eq("reduce")]
    for ax, strategy in zip(axes, STRATEGIES):
        for op in OPERATORS:
            for prec, ls in (("fp32", "-"), ("bf16", "--")):
                line(ax, red[red["strategy"].eq(strategy) & red["operator"].eq(op) & red["precision"].eq(prec)],
                     "speedup_over_fp64", OP_COLOURS[op], linestyle=ls)
        finish(ax, "Speedup over fp64" if strategy == STRATEGIES[0] else None, True, True)
        ax.set_title(LABELS[strategy])
        ax.set_xlabel("Cells")
    handles = [Line2D([], [], color=OP_COLOURS[o], marker="o", label=f"{OPERATORS[o]} reduced") for o in OPERATORS]
    handles += [Line2D([], [], color="0.3", linestyle="-", label="fp32"), Line2D([], [], color="0.3", linestyle="--", label="bf16")]
    fig.legend(handles=handles, loc="outside upper center", ncols=5, frameon=False)
    save(fig, here / "precision_reduce_speedup.pdf")


def plot_alone(data, here):
    fig, axes = plt.subplots(3, len(STRATEGIES), figsize=(4.5 * len(STRATEGIES), 11), sharey=True,
                             constrained_layout=True)
    alone = data[data["mode"].eq("alone")]
    for row, op in zip(axes, OPERATORS):
        for ax, strategy in zip(row, STRATEGIES):
            for prec in PRECISIONS:
                line(ax, alone[alone["strategy"].eq(strategy) & alone["operator"].eq(op) & alone["precision"].eq(prec)],
                     "speedup_over_fp64", COLOURS[prec], label=prec)
            finish(ax, f"{OPERATORS[op]} alone: speedup" if strategy == STRATEGIES[0] else None, True)
            if op == "ddt":
                ax.set_title(LABELS[strategy])
            if op == "lap":
                ax.set_xlabel("Cells")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False,
               title="each operator assembled alone, operator and matrix in one precision")
    save(fig, here / "precision_alone_speedup.pdf")


def plot_overheads(data, here):
    """allocation, conversion and back-conversion of the whole reduced assembly against its assembly
    time, for every strategy (they barely depend on the strategy)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, constrained_layout=True)
    whole = data[data["mode"].eq("whole") & ~data["precision"].eq("fp64")]
    for ax, prec in zip(axes, ["fp32", "fp16", "bf16"]):
        sub = whole[whole["precision"].eq(prec)]
        for strategy, marker in ((s, MARKERS[s]) for s in STRATEGIES):
            s = sub[sub["strategy"].eq(strategy)]
            line(ax, s, "assembly_ms", "0.2", marker=marker)
            line(ax, s, "allocation_ms", "#9467bd", marker=marker, linestyle=":")
            line(ax, s, "conversion_ms", "#8c564b", marker=marker, linestyle="--")
            line(ax, s, "back_conversion_ms", "#e377c2", marker=marker, linestyle="-.")
        finish(ax, "Time [ms]" if prec == "fp32" else None)
        ax.set_yscale("log")
        ax.set_title(f"whole assembly in {prec}")
        ax.set_xlabel("Cells")
    handles = [Line2D([], [], color=c, linestyle=ls, label=l) for c, ls, l in (
        ("0.2", "-", "assembly"), ("#9467bd", ":", "allocation"), ("#8c564b", "--", "conversion of the inputs"),
        ("#e377c2", "-.", "back-conversion to fp64"))]
    handles += [Line2D([], [], color="0.4", marker=m, linestyle="none", label=LABELS[s]) for s, m in ((s, MARKERS[s]) for s in STRATEGIES)]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False)
    save(fig, here / "precision_overheads.pdf")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    args = [a for a in sys.argv[1:] if a != "--no-global"]
    logs = Path(args[0]) if args else here / "h200"
    data, devs = parse(logs)
    if "--no-global" in sys.argv:
        STRATEGIES.remove("global face-based")
        TAG = "_noglobal"
    else:
        data.to_csv(here / "precision.csv", index=False)
        devs.to_csv(here / "precision_deviation.csv", index=False)
    style()
    plot_whole(data, here, "throughput_gbs", "Throughput [GB/s]", "precision_whole_gbs.pdf", False)
    with plt.rc_context(LARGE):
        plot_whole(data, here, "speedup_over_fp64", "Speedup over fp64", "precision_whole_speedup.pdf", True, True)
        plot_whole(data, here, "assembly_ms", "Assembly time [ms]", "precision_whole_time.pdf", False, True)
        plot_reduce(data, here)
    plot_alone(data, here)
    plot_overheads(data, here)
