"""Replace JuNe's rows of h200/cavity_throughput.csv by the JuNe rerun with the relaxed REDG atomics
(JuNe.jl src/types.jl relaxed_add!): logs in h200/june_red/ from NeoFOAM benchmarks/june/
cavity_throughput.job with CAVITY_MATRIX=scalar CAVITY_ONLY="JuNe face-based,JuNe cell-based,JuNe
batched" (job 638985 on gpu-nvidia-h200-3), the same cavity series, operators (div + laplacian) and
timing as the NeoN rows, which stay. JuNe global face-based was not rerun and is dropped.

    python3 merge_june_red.py   (idempotent: the JuNe rows always come from h200/june_red/)

The aggregation is that of NeoFOAM benchmarks/june/parse_results.py: first run dropped, mean, std and
median of the rest, throughput = nnz * 8 B / median time.
"""
import re
from pathlib import Path

import pandas as pd

TIME = re.compile(r"Assembly time \((.+) (\d+)\) = (\d+) \[us\]")
here = Path(__file__).resolve().parent / "h200"

rows = []
for log in sorted(here.glob("june_red/cavity_*.log")):
    text = log.read_text()
    cells = int(re.search(r"^cells: (\d+)", text, re.M).group(1))
    nnz = int(re.search(r"^nnz: (\d+)", text, re.M).group(1))
    assert re.search(r"^matrix: scalar", text, re.M), log
    for strategy, run, us in TIME.findall(text):
        rows.append({"gpu": "H200", "n": int(log.stem.split("_")[1]), "cells": cells, "nnz": nnz,
                     "matrix": "scalar", "strategy": strategy, "run": int(run), "time_ms": int(us) / 1e3})
runs = pd.DataFrame(rows)
runs = runs[runs["run"] > 0]  # run 0 carries the JIT compilation
june = (runs.groupby(["gpu", "n", "cells", "nnz", "matrix", "strategy"], as_index=False)
            .agg(time_mean_ms=("time_ms", "mean"), time_std_ms=("time_ms", "std"),
                 time_median_ms=("time_ms", "median"), runs=("run", "count")))
june["throughput_gbs"] = june["nnz"] * 8 / (june["time_median_ms"] / 1e3) / 1e9

csv = here / "cavity_throughput.csv"
data = pd.read_csv(csv)
neon = data[~data["strategy"].str.startswith("JuNe")]
# 250^3 and 350^3 were meshed later for the precision study and have no NeoN rows in this series
june = june[june["n"].isin(neon["n"])]
for n, nnz in june.groupby("n")["nnz"].first().items():  # the same meshes as the NeoN rows
    assert (neon.loc[neon["n"].eq(n), "nnz"] == nnz).all(), n
data = pd.concat([neon, june], ignore_index=True).sort_values(["gpu", "matrix", "strategy", "cells"])
data.to_csv(csv, index=False)
print(f"{len(neon)} NeoN rows kept, {len(june)} JuNe rows from {june['n'].nunique()} meshes -> {csv}")
