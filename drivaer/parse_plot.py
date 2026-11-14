"""Integration benchmark: assembly of the whole momentum equation ddt(U) + div(phi,U) - laplacian(nu,U)
as a scalar matrix on the 65M cell occDrivAer mesh (OHC-1 case, upwind convection), every NeoN,
fused NeoN and JuNe strategy on one GPU. Logs come from benchmarks/june/drivaer_assembly.job of
NeoFOAM (test/test_cavity.cpp with CAVITY_DDT=1 CAVITY_MATRIX=scalar, three timed assemblies per
strategy, the first one dropped).

    python3 parse_plot.py     reads <gpu>/drivaer.log next to this file, writes drivaer.csv and
                              drivaer_<gpu>.pdf
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MATRIX_BYTES = {"Vec3": 3 * 8, "scalar": 8}
TIME = re.compile(r"Assembly time \((.+) (\d+)\) = (\d+) \[us\]")
STRATEGIES = {  # print name in the test -> (label, group)
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
KINDS = ["Face-Based", "Global Face-Based", "Cell-Based", "Batched Face-Based"]
GROUPS = {"NeoN": ("#9ecae1", "none"), "NeoN fused": ("#1f77b4", "#1f77b4"), "JuNe": ("#d62728", "#d62728")}


def parse(here: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(here.glob("*/drivaer.log")):
        text = log.read_text()
        cells = int(re.search(r"^cells: (\d+)", text, re.M).group(1))
        nnz = int(re.search(r"^nnz: (\d+)", text, re.M).group(1))
        matrix = re.search(r"^matrix: (\w+)", text, re.M).group(1)
        runs = pd.DataFrame(TIME.findall(text), columns=["strategy", "run", "us"]).astype({"run": int, "us": float})
        runs = runs[runs["run"] > 0]
        for strategy, sub in runs.groupby("strategy"):
            ms = sub["us"] / 1e3
            rows.append({"gpu": log.parent.name.upper(), "cells": cells, "nnz": nnz, "matrix": matrix,
                         "strategy": strategy, "time_mean_ms": ms.mean(), "time_std_ms": ms.std(ddof=0),
                         "runs": len(ms), "throughput_gbs": nnz * MATRIX_BYTES[matrix] / (ms.mean() / 1e3) / 1e9})
    return pd.DataFrame(rows)


def plot(data: pd.DataFrame, out: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    data = data.set_index("strategy")
    width = 0.26
    top = 0.0
    for j, (group, (face, edge)) in enumerate(GROUPS.items()):
        x = [i + (j - 1) * (width + 0.03) for i in range(len(KINDS))]
        values = []
        for kind in KINDS:
            name = next((s for s, (k, g) in STRATEGIES.items() if k == kind and g == group), None)
            values.append(data["throughput_gbs"].get(name, float("nan")))
        ax.bar(x, values, width=width, color=face, edgecolor=edge if edge != "none" else "#1f77b4",
               linewidth=1.3, label=group)
        for xi, v, kind in zip(x, values, KINDS):
            if pd.notna(v):
                name = next(s for s, (k, g) in STRATEGIES.items() if k == kind and g == group)
                ax.text(xi, v * 1.01, f"{v:.0f}\n{data.loc[name, 'time_mean_ms']:.0f} ms", ha="center",
                        va="bottom", fontsize=11)
                top = max(top, v)
    ax.set_xticks(range(len(KINDS)))
    ax.set_xticklabels(KINDS)
    ax.set_ylabel("Assembly throughput [GB/s]")
    ax.set_ylim(top=top * 1.35)
    cells = int(data["cells"].iloc[0])
    ax.text(0.02, 0.97, f"occDrivAer, {cells:,} cells, {data['matrix'].iloc[0]} matrix, {data['gpu'].iloc[0]}",
            transform=ax.transAxes, ha="left", va="top", fontsize=15, fontweight="bold", color="0.2")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False)  # the right bars are the lowest
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data = parse(here)
    data.to_csv(here / "drivaer.csv", index=False)
    print(data.round(2).to_string(index=False))
    for gpu, sub in data.groupby("gpu"):
        plot(sub, here / f"drivaer_{gpu.lower()}")
