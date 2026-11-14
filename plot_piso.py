"""Composition of a neoIcoFoam time step on the cavity: momentum assembly, momentum solve, pressure
assembly and pressure solve per variant of the momentum assembly (NeoN face-based, NeoN fused
face-based, JuNe face-based), as stacked bars of the mean wall time per step.

    python3 plot_piso.py [csv] [out]

reads june_neon/cavity_piso.csv (parse_piso.py of the NeoFOAM benchmark) and writes pdf.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

VARIANTS = {"neon": "NeoN", "fused": "NeoN fused", "june": "JuNe", "icofoam": "icoFoam, 64 ranks"}
PARTS = {  # column -> (label, colour)
    "momentum_assembly_ms": ("Momentum assembly", "#1f77b4"),
    "momentum_solve_ms": ("Momentum solve", "#aec7e8"),
    "pressure_assembly_ms": ("Pressure assembly", "#ff7f0e"),
    "pressure_solve_ms": ("Pressure solve", "#ffbb78"),
    "other_ms": ("Rest of the step", "0.75"),
}


def plot(data: pd.DataFrame, out: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    # left: the whole step stacked, right: the momentum assembly alone, which the step hides
    fig, (ax, axA) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [3, 2]})
    data = data.set_index("variant").loc[[v for v in VARIANTS if v in set(data["variant"])]].copy()
    data["other_ms"] = data["step_ms"] - data[[c for c in PARTS if c != "other_ms"]].sum(axis=1)
    labels = [VARIANTS[v] for v in data.index]
    bottom = pd.Series(0.0, index=data.index)
    for column, (label, colour) in PARTS.items():
        ax.bar(range(len(data)), data[column], bottom=bottom, color=colour, label=label, width=0.6)
        bottom += data[column]
    for i, (variant, total) in enumerate(zip(data.index, bottom)):
        assembly = data.loc[variant, "momentum_assembly_ms"]
        ax.text(i, total * 1.01, f"{total / 1e3:,.1f} s\nassembly {100 * assembly / total:.2f} %",
                ha="center", va="bottom", fontsize=13)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean wall time per time step [ms]")
    ax.set_ylim(top=bottom.max() * 1.2)
    n = int(data["n"].iloc[0])
    ax.text(0.02, 0.97, f"{n}\u00b3 = {n**3:,} cells, {data['gpu'].iloc[0]}", transform=ax.transAxes,
            ha="left", va="top", fontsize=15, fontweight="bold", color="0.2")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False, labelspacing=0.8, fontsize=13)

    assembly = data["momentum_assembly_ms"]
    axA.bar(range(len(data)), assembly, color=PARTS["momentum_assembly_ms"][1], width=0.6)
    for i, value in enumerate(assembly):
        axA.text(i, value * 1.02, f"{value:.1f} ms", ha="center", va="bottom", fontsize=13)
    axA.set_xticks(range(len(data)))
    axA.set_xticklabels(labels)
    axA.set_ylabel("Momentum assembly per time step [ms]")
    axA.set_ylim(top=assembly.max() * 1.2)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out.with_suffix(".pdf"))
    return fig


def main(csv: Path, out: Path):
    data = pd.read_csv(csv)
    for (gpu, n), sub in data.groupby(["gpu", "n"]):
        plot(sub, out.with_name(f"{out.name}_{gpu.lower()}_n{n}"))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "june_neon"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else here / "cavity_piso.csv",
         Path(sys.argv[2]) if len(sys.argv) > 2 else here / "cavity_piso")
