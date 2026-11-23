"""Time share of every operator in JuNe's GPU assembly of the momentum equation at the largest mesh,
reverse-engineered from the precision study (precision.csv of plot_precision.py): for every strategy
and precision P the whole assembly t_whole (ddt + div + laplacian, matrix in P), every operator
assembled alone t_op (matrix in P) and the reset of matrix and rhs alone R (part of every assembly).
Assuming the parts add up:

  face-based, global face-based, batched: ddt is a kernel of its own, div and laplacian share the face
  loops (indices, face data, atomic updates, the boundary kernel), which ddt alone does not run:
      t_ddt = R + ddt,  t_div = R + L + div,  t_lap = R + L + lap,  t_whole = R + L + ddt + div + lap
  cell-based: one cell kernel for all operators, run for each of them alone too:
      t_op = R + L + op,  t_whole = R + L + ddt + div + lap

Solved for the loop part L and the operators; the call into Julia and the kernel launches end up in L
(and, for the face strategies, partly in ddt). By construction the stacked parts equal t_whole; a
negative part means the operators do not add up for that strategy and precision.

    python3 plot_shares.py [--no-global]   reads precision.csv, writes precision_shares.csv and
                                           precision_shares.pdf (_noglobal without global face-based)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from plot_precision import LABELS, LARGE, OP_COLOURS, OPERATORS, PRECISIONS, STRATEGIES

PARTS = {"reset": ("reset of matrix and rhs", "0.35"), "loop": ("shared loop", "0.72"),
         **{op: (name, OP_COLOURS[op]) for op, name in OPERATORS.items()}}


def shares(data: pd.DataFrame, n: int) -> pd.DataFrame:
    d = data[data["n"].eq(n)].set_index(["strategy", "config"])
    t, reset = d["assembly_ms"], d["reset_ms"]
    rows = []
    for strategy in sorted(data["strategy"].unique()):
        for prec in PRECISIONS:
            whole = t[(strategy, prec)]
            r = reset[(strategy, prec)]
            alone = {op: t[(strategy, f"only-{op}-{prec}")] - r for op in OPERATORS}  # without the reset
            if strategy == "cell-based":
                loop = (sum(alone.values()) - (whole - r)) / 2
                part = {op: alone[op] - loop for op in OPERATORS}
            else:
                ddt = alone["ddt"]
                loop = alone["div"] + alone["lap"] - (whole - r - ddt)
                part = {"ddt": ddt, "div": alone["div"] - loop, "lap": alone["lap"] - loop}
            row = {"n": n, "strategy": strategy, "precision": prec, "whole_ms": whole, "reset_ms": r,
                   "loop_ms": loop, **{f"{op}_ms": part[op] for op in OPERATORS}}
            rows.append(row)
    return pd.DataFrame(rows)


def plot(s: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(4.5 * len(STRATEGIES), 5), sharey=True,
                             constrained_layout=True)
    for ax, strategy in zip(axes, STRATEGIES):
        sub = s[s["strategy"].eq(strategy)].set_index("precision").loc[list(PRECISIONS)]
        bottom = pd.Series(0.0, index=sub.index)
        negative = {x: [] for x in sub.index}  # parts that do not add up: named above the bar instead
        for part, (name, colour) in PARTS.items():
            height = sub[f"{part}_ms"].clip(lower=0)
            ax.bar(sub.index, height, bottom=bottom, color=colour, width=0.7)
            bottom += height
            for x, v in sub[f"{part}_ms"].items():
                if v < 0:
                    negative[x].append(f"{name} {v:.1f}")
        for x, top, total in zip(sub.index, bottom, sub["whole_ms"]):
            ax.text(x, top * 1.01, "\n".join([*negative[x], f"{total:.1f}"]), ha="center", va="bottom",
                    fontsize=12)
        ax.set_title(LABELS[strategy])
        ax.set_xlabel("Precision")
    axes[0].set_ylabel("Assembly time [ms]")
    axes[0].set_ylim(0, s["whole_ms"].max() * 1.2)
    cells = int(s["n"].iloc[0]) ** 3
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in PARTS.values()], loc="outside upper center",
               ncols=len(PARTS), frameon=False, title=f"{cells / 1e6:.0f}M cells, whole momentum equation")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data = pd.read_csv(here / "precision.csv")
    tag = ""
    if "--no-global" in sys.argv:
        STRATEGIES.remove("global face-based")
        tag = "_noglobal"
    s = shares(data, int(data["n"].max()))
    if not tag:
        s.to_csv(here / "precision_shares.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    with plt.rc_context(LARGE):
        plot(s, here / f"precision_shares{tag}.pdf")
    cols = ["reset_ms", "loop_ms", *(f"{op}_ms" for op in OPERATORS), "whole_ms"]
    print(s.set_index(["strategy", "precision"])[cols].round(2).to_string())
