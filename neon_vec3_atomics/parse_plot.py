"""NeoN face-based Vec3 assembly: one 24-byte Kokkos atomic per diagonal/rhs update ("whole",
Kokkos' lock-based fallback for types that are not 4/8/16 bytes) against three scalar atomics
("component", hardware atomics). Logs come from benchmarks/june/cavity_vec3_atomics.job of NeoFOAM
(test/test_cavity.cpp, div + laplacian on the lid-driven cavity, three timed assemblies per
strategy, the first one dropped).

    python3 parse_plot.py            reads <gpu>/cavity_<n>_<variant>.log next to this file,
                                     writes cavity_vec3_atomics.csv and one figure per GPU
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VEC3_BYTES = 3 * 8  # the Vec3 matrix stores three doubles per non-zero
TIME = re.compile(r"Assembly time \((.+) (\d+)\) = (\d+) \[us\]")
VARIANTS = {"whole": "1× atomic_sub(Vec3)", "component": "3× atomic_sub(scalar)"}
STRATEGIES = {"NeoN face-based": "Face-Based", "NeoN fused face-based": "Fused Face-Based"}
COLOURS = {"whole": "#9ecae1", "component": "#1f77b4"}


def parse(here: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(here.glob("*/cavity_*_*.log")):
        text = log.read_text()
        _, n, variant = log.stem.split("_")
        cells = int(re.search(r"^cells: (\d+)", text, re.M).group(1))
        nnz = int(re.search(r"^nnz: (\d+)", text, re.M).group(1))
        runs = pd.DataFrame(TIME.findall(text), columns=["strategy", "run", "us"]).astype({"run": int, "us": float})
        runs = runs[runs["run"] > 0]  # the first run includes JIT and first-touch effects
        for strategy, sub in runs.groupby("strategy"):
            if strategy not in STRATEGIES:  # the test always times its JuNe reference as well
                continue
            ms = sub["us"] / 1e3
            rows.append({"gpu": log.parent.name.split("-")[0].upper(), "n": int(n), "cells": cells, "nnz": nnz,
                         "strategy": strategy, "atomics": variant, "time_mean_ms": ms.mean(),
                         "time_std_ms": ms.std(ddof=0), "runs": len(ms),
                         "throughput_gbs": nnz * VEC3_BYTES / (ms.mean() / 1e3) / 1e9})
    return pd.DataFrame(rows).sort_values(["gpu", "n", "strategy", "atomics"])


def plot(data: pd.DataFrame, out: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    strategies = [s for s in STRATEGIES if s in set(data["strategy"])]
    width = 0.36
    top = 0
    for j, (variant, label) in enumerate(VARIANTS.items()):
        sub = data[data["atomics"] == variant].set_index("strategy").reindex(strategies)
        x = [i + (j - 0.5) * (width + 0.04) for i in range(len(strategies))]
        ax.bar(x, sub["throughput_gbs"], width=width, color=COLOURS[variant], label=label)
        for xi, (gbs, ms) in zip(x, zip(sub["throughput_gbs"], sub["time_mean_ms"])):
            if pd.notna(gbs):
                ax.text(xi, gbs * 1.01, f"{gbs:.0f} GB/s\n{ms:.1f} ms", ha="center", va="bottom", fontsize=12)
                top = max(top, gbs)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGIES[s] for s in strategies])
    ax.set_ylabel("Assembly throughput [GB/s]")
    ax.set_ylim(top=top * 1.3)
    n = int(data["n"].iloc[0])
    ax.text(0.02, 0.97, f"NeoN Vec3 matrix, {n}³ = {n**3:,} cells, {data['gpu'].iloc[0]}",
            transform=ax.transAxes, ha="left", va="top", fontsize=15, fontweight="bold", color="0.2")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False, title="Diagonal and rhs update")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data = parse(here)
    data.to_csv(here / "cavity_vec3_atomics.csv", index=False)
    print(data.round(2).to_string(index=False))
    for (gpu, n), sub in data.groupby(["gpu", "n"]):
        plot(sub, here / f"cavity_vec3_atomics_{gpu.lower()}_n{n}")
