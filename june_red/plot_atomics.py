"""JuNe's three implementations of the matrix atomics, measured one after another on the same node
(NeoFOAM benchmarks/june/red/cavity_atomics.job): div + laplacian (no ddt) of the momentum equation on
the lid-driven cavity, scalar matrix, fp64, H200, five fenced assemblies including the reset, the first
dropped, median reported; throughput = nnz * 8 B / median time. NeoN fused is measured in every
variant's run as a check that the node behaves the same, it is in the csv but not plotted.

    original  Atomix.@atomic, acquire-release on CUDA: MEMBAR.GPU + ATOMG + CCTL.IVALL per update
    atom      relaxed UnsafeAtomics.modify!: ATOMG without barriers
    red       relaxed, fp64 on the GPU as inline PTX red.relaxed.gpu.global.add.f64: REDG

    python3 plot_atomics.py [log_dir]   reads <log_dir>/<variant>/cavity_<n>.log (default
                                        ./h200_atomics), writes h200_atomics/cavity_atomics.csv and
      h200_atomics/cavity_atomics_gbs.pdf       throughput of the variants
      h200_atomics/cavity_atomics_speedup.pdf   speedup of atom and red over original
(the figures show face-based only; every JuNe strategy is measured and in the csv)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

from plot_red import axes_cells, parse

VARIANTS = {  # variant -> (label, colour, marker)
    "original": ("Atomix (acquire-release)", "0.55", "v"),
    "atom": ("relaxed, ATOMG", "#e6550d", "D"),
    "red": ("relaxed, REDG", "#08519c", "s"),
}
STRATEGIES = {"face-based": "Face-Based"}  # cell-based and batched barely use atomics, see the csv


def load(logs: Path) -> pd.DataFrame:
    frames = []
    for variant in VARIANTS:
        d = parse(logs / variant)
        d["variant"] = variant
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def panels():
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    return fig, [ax]


def plot_gbs(data, here):
    fig, axes = panels()
    for ax, (strategy, title) in zip(axes, STRATEGIES.items()):
        for variant, (label, colour, marker) in VARIANTS.items():
            s = data[data["variant"].eq(variant) & data["strategy"].eq("JuNe " + strategy)].sort_values("cells")
            ax.plot(s["cells"], s["throughput_gbs"], color=colour, marker=marker, markersize=7, label="JuNe " + label)
        axes_cells(ax, sorted(data["cells"].unique()))
        ax.set_title(title)
    axes[0].set_ylabel("Throughput [GB/s]")
    axes[0].set_ylim(0, None)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3, frameon=False,
               title="div + laplacian, scalar matrix, H200")
    fig.savefig(here / "cavity_atomics_gbs.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", here / "cavity_atomics_gbs.pdf")


def plot_speedup(data, here):
    t = data.set_index(["variant", "n", "strategy"])["time_median_ms"]
    fig, axes = panels()
    for ax, (strategy, title) in zip(axes, STRATEGIES.items()):
        name = "JuNe " + strategy
        for variant, (label, colour, marker) in VARIANTS.items():
            if variant == "original":
                continue
            pts = [(n**3, t[("original", n, name)] / t[(variant, n, name)]) for n in sorted(data["n"].unique())]
            ax.plot(*zip(*pts), color=colour, marker=marker, markersize=7, label="JuNe " + label)
        ax.axhline(1.0, color="0.3", linewidth=1)
        axes_cells(ax, sorted(data["cells"].unique()))
        ax.set_title(title)
    axes[0].set_ylabel("Speedup over Atomix (acquire-release)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False,
               title="div + laplacian, scalar matrix, H200")
    fig.savefig(here / "cavity_atomics_speedup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", here / "cavity_atomics_speedup.pdf")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "h200_atomics"
    out = here / "h200_atomics"
    out.mkdir(exist_ok=True)
    data = load(logs)
    data.to_csv(out / "cavity_atomics.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13, "legend.fontsize": 11})
    plot_gbs(data, out)
    plot_speedup(data, out)
    print(data.pivot_table(index=["strategy", "variant"], columns="n", values="time_median_ms").round(2).to_string())
