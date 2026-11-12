"""Scalar-matrix assembly throughput on the H200 and on the H200 node's AMD EPYC 9555 CPUs (one set per thread
count), one pair of executors at a time, in the style of
variant (c) of plot_scalar_variants.py (each strategy kind named once by colour, the executor by
marker, in one legend inside the axes, placed where it needs the least room above the data: a column
in a top corner for the throughput, a row at the top left or centre for the speedup):

  1. throughput of NeoN against NeoN Fused;
  2. throughput of NeoN Fused against JuNe;
  3. speedup of JuNe over NeoN Fused of the same kind.

    python3 plot_scalar_pairs.py [results_dir]

reads june_neon/h200/cavity_throughput.csv and june_neon/cavity_throughput_cpu.csv and writes

  june_neon/h200/cavity_scalar_<plot>.pdf          GPU
  june_neon/cpu_h200_threads<t>_scalar_<plot>.pdf  CPU, one set per thread count

with <plot> one of neon_vs_fused, fused_vs_june, june_over_fused, and the three stacked in <plot> = pairs.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter, FixedLocator, FormatStrFormatter, NullLocator

from plot_scalar_variants import CELL_TICKS, EXECUTORS, KINDS, SERIES, draw

CPU = "AMD EPYC 9555"  # the CPUs of the H200 node
FUSED = {kind: s for s, (kind, e) in SERIES.items() if e == "NeoN Fused"}  # kind -> NeoN Fused
SPEEDUP_TICKS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50]


def legend(ax, executors=(), row: bool = False):
    """Executor markers above the strategy colours in one column, or all in one row, inside the axes;
    fit_legends moves it to the place that needs the least room (a top corner for the column, top left
    or top centre for the row, below the title where they would meet) and raises the y range so that
    it covers no point."""
    handles = [Line2D([], [], color="0.25", marker=EXECUTORS[e][0], linestyle="none", markersize=9,
                      markerfacecolor="0.25" if EXECUTORS[e][1] else "none", label=e) for e in executors]
    handles += [Patch(facecolor=c, label=k) for k, c in KINDS.items()]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False,
                    labelspacing=0.8, ncol=len(handles) if row else 1)
    leg.places = ("upper left", "upper center") if row else ("upper left", "upper right")


def need_top(ax, span: np.ndarray, below: float) -> float:
    """The top of the y range at which the highest value in span sits at the axes fraction below."""
    bottom, top = ax.get_ylim()
    if not len(span):
        return top
    if ax.get_yscale() == "log":
        return max(top, bottom * (span.max() / bottom) ** (1 / below))
    return max(top, bottom + (span.max() - bottom) / below)


def under(ax, left: float, right: float) -> np.ndarray:
    """The data y of the points and of the line crossings between the axes fractions left and right."""
    points = np.concatenate([c.get_offsets() for c in ax.collections])
    points = points[np.isfinite(points).all(axis=1)]
    x = ax.transAxes.inverted().transform(ax.transData.transform(points))[:, 0]
    return np.concatenate([points[(x > left) & (x < right), 1], crossings(ax, left), crossings(ax, right)])


def legend_room(ax) -> tuple[tuple[float, float], float]:
    """Where the legend goes (its top-left corner in axes fractions) and the top of the y range that
    keeps every point and line below both the panel title and the legend: of the legend's places, the
    one that needs the least room wins. A place that would overlap the title in x goes below it.
    Needs a drawn figure (the sizes)."""
    renderer = ax.figure.canvas.get_renderer()
    leg, text = ax.get_legend(), ax.panel_title
    box = leg.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    head = text.get_window_extent(renderer).transformed(ax.transAxes.inverted())
    width, height = box.width, box.height
    title_top = need_top(ax, under(ax, 0, head.x1 + 0.02), head.y0 - 0.03)
    room = {}
    for place in leg.places:
        left = {"upper left": 0.01, "upper center": 0.5 - width / 2, "upper right": 0.99 - width}[place]
        upper = head.y0 - 0.01 if left < head.x1 + 0.02 else 0.99  # below the title where they would meet
        need = need_top(ax, under(ax, left - 0.01, left + width + 0.01), upper - height - 0.03)
        room[(left, upper)] = max(title_top, need)
    anchor = min(room, key=room.get)  # ties: the first place, top left
    return anchor, room[anchor]


def crossings(ax, edge: float) -> np.ndarray:
    """The data y where the connecting lines cross x = edge (in axes fraction); with the points on either
    side, these bound how high a line reaches inside the legend's span of x."""
    ys = []
    for line in ax.get_lines():
        if line.get_transform() is not ax.transData:  # reference lines such as axhline
            continue
        xy = np.column_stack(line.get_data()).astype(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 2:
            continue
        display = ax.transData.transform(xy)  # the segments are straight in display space
        frac = ax.transAxes.inverted().transform(display)[:, 0]
        for i in np.flatnonzero((frac[:-1] - edge) * (frac[1:] - edge) < 0):
            t = (edge - frac[i]) / (frac[i + 1] - frac[i])
            ys.append(ax.transData.inverted().transform(display[i] + t * (display[i + 1] - display[i]))[1])
    return np.array(ys)


def fit_legends(axes, shared=()):
    """Move each legend to its place and raise the y ranges; the axes in shared get one common top."""
    axes[0].figure.canvas.draw()
    rooms = [legend_room(ax) for ax in axes]
    common = max((top for ax, (_, top) in zip(axes, rooms) if ax in shared), default=None)
    for ax, (anchor, top) in zip(axes, rooms):
        ax.get_legend().set_loc("upper left")
        ax.get_legend().set_bbox_to_anchor(anchor, transform=ax.transAxes)
        ax.set_ylim(top=common if ax in shared else top)


def title(ax, text: str):
    """The bold panel title inside the axes, top left, as in the GPU figures; fit_legends keeps the
    legend and the data clear of it."""
    ax.panel_title = ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=17,
                             fontweight="bold", color="0.2")


def throughput(ax, data: pd.DataFrame, executors: list[str], text: str):
    for s, (_, executor) in SERIES.items():
        if executor in executors:
            draw(ax, data.loc[data["strategy"].eq(s)], s, "throughput_gbs")
    legend(ax, executors)
    ax.set(ylabel="Matrix Assembly Throughput [GB/s]")
    ax.set_ylim(0, data["throughput_gbs"].max() * 1.05)  # same scale in both throughput plots
    title(ax, text)


def june_over_fused(ax, data: pd.DataFrame, text: str):
    fused = data.set_index(["strategy", "n"])["throughput_gbs"]
    lo, hi = 1, 1
    for s, (kind, executor) in SERIES.items():
        if executor != "JuNe":
            continue
        points = data.loc[data["strategy"].eq(s)].copy()
        points["speedup"] = points["throughput_gbs"].to_numpy() / fused.loc[FUSED[kind]].reindex(points["n"]).to_numpy()
        draw(ax, points, s, "speedup")
        lo, hi = min(lo, points["speedup"].min()), max(hi, points["speedup"].max())
    ax.axhline(1, color="0.3", linewidth=1.2, linestyle="--", zorder=0)
    legend(ax, row=True)
    ax.set(ylabel="Speedup of JuNe over NeoN Fused")
    ax.set_yscale("log")
    lo, hi = lo / 1.15, hi * 1.15
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(SPEEDUP_TICKS))  # fit_legends may raise the top
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(NullLocator())
    title(ax, text)


PLOTS = {  # output suffix -> (draw function, short title, title in the stacked figure)
    "neon_vs_fused": (lambda ax, d, t: throughput(ax, d, ["NeoN", "NeoN Fused"], t),
                                    "NeoN vs. NeoN Fused", "(1) NeoN vs. NeoN Fused"),
    "fused_vs_june": (lambda ax, d, t: throughput(ax, d, ["NeoN Fused", "JuNe"], t),
                                    "NeoN Fused vs. JuNe", "(2) NeoN Fused vs. JuNe"),
    "june_over_fused": (june_over_fused, "JuNe over NeoN Fused",
                                      "(3) speedup of JuNe over NeoN Fused of the same kind"),
}


def finish(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(CELL_TICKS))
    ax.xaxis.set_major_formatter(EngFormatter())
    ax.grid(True, which="minor", alpha=0.15)
    ax.tick_params(labelbottom=True)


def save(fig, out: Path):
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"))


THROUGHPUT = ("neon_vs_fused", "fused_vs_june")  # the plots that share one y scale


def figures(data: pd.DataFrame, label: str, prefix: Path):
    """The three plots as standalone figures <prefix>_<plot>.pdf and stacked in <prefix>_pairs.pdf."""
    standalone = {}
    for name, (plot, short, _) in PLOTS.items():
        fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
        plot(ax, data, f"{label}: {short}")
        finish(ax)
        ax.set_xlabel("Cells")
        standalone[name] = ax
    # the legend sizes are known once drawn; each figure is drawn on its own
    for name, ax in standalone.items():
        fit_legends([ax])
    common = max(standalone[name].get_ylim()[1] for name in THROUGHPUT)
    for name, ax in standalone.items():
        if name in THROUGHPUT:
            ax.set_ylim(top=common)
        save(ax.figure, prefix.with_name(f"{prefix.name}_{name}"))

    fig, axes = plt.subplots(len(PLOTS), 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    for ax, (plot, _, long) in zip(axes, PLOTS.values()):
        plot(ax, data, long)
        finish(ax)
    axes[-1].set_xlabel("Cells")
    fig.suptitle(label, fontsize=19, fontweight="bold", color="0.2")
    fit_legends(list(axes), shared=[ax for ax, name in zip(axes, PLOTS) if name in THROUGHPUT])
    save(fig, prefix.with_name(f"{prefix.name}_pairs"))


def scalar(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data.loc[data["matrix"].eq("scalar") & data["strategy"].isin(SERIES)]


def main(results: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 15})
    figures(scalar(results / "h200" / "cavity_throughput.csv"), "NVIDIA H200, scalar matrix",
            results / "h200" / "cavity_scalar")
    cpu = scalar(results / "cavity_throughput_cpu.csv")
    for threads, data in cpu.groupby("threads"):
        figures(data, f"{CPU}, {threads} thread{'s' if threads > 1 else ''}, scalar matrix",
                results / f"cpu_h200_threads{threads}_scalar")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "june_neon")
