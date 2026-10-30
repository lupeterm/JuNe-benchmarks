"""Mean assembly time and throughput of the icoFoam cavity runs.

Reads results/<node>-icofoam-<ranks>/cavity_<n>.log written by cavity_icofoam.job. Every run prints
one "Momentum <seconds>" line per rank after a barrier; the run's time is the slowest rank. All runs
count (no JIT), the matrix is Vec3 valued (24 bytes per entry) and the cavity mesh has
n^3 cells and 3 n^2 (n - 1) internal faces, i.e. nnz = cells + 2 internal faces as in NeoN.

    python3 parse_results.py [results_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

RUN = re.compile(r"^--- run (\d+)$", re.M)
MOMENTUM = re.compile(r"^Momentum ([0-9.eE+-]+)$", re.M)


def parse_log(log: Path, ranks: int) -> list[dict]:
    text = log.read_text()
    n = int(log.stem.split("_")[1])
    cells = n**3
    nnz = cells + 2 * 3 * n**2 * (n - 1)
    rows = []
    for chunk in RUN.split(text)[1:][1::2]:  # text after every "--- run k" header
        times = [float(t) for t in MOMENTUM.findall(chunk)]
        if len(times) != ranks:
            raise ValueError(f"{log}: {len(times)} Momentum lines, expected {ranks}")
        rows.append({"n": n, "cells": cells, "nnz": nnz, "time_ms": 1e3 * max(times)})
    return rows


def main(results: Path):
    rows = []
    for folder in sorted(results.glob("*-icofoam-*")):
        node, _, ranks = folder.name.split("-")
        for log in sorted(folder.glob("cavity_*.log")):
            try:
                parsed = parse_log(log, int(ranks))
            except ValueError as e:
                print(f"skipped: {e}")
                continue
            for r in parsed:
                rows.append({"gpu": node.upper(), "strategy": f"icoFoam {ranks} ranks", **r})
    raw = pd.DataFrame(rows)
    data = (raw.groupby(["gpu", "n", "cells", "nnz", "strategy"], as_index=False)
            .agg(time_mean_ms=("time_ms", "mean"), time_std_ms=("time_ms", "std"), runs=("time_ms", "size")))
    data["throughput_gbs"] = data["nnz"] * 24 / (data["time_mean_ms"] / 1e3) / 1e9
    data["throughput_mcells_s"] = data["cells"] / (data["time_mean_ms"] / 1e3) / 1e6
    data = data.sort_values(["gpu", "strategy", "cells"])
    out = results / "cavity_icofoam.csv"
    data.to_csv(out, index=False)
    print(f"{len(data)} rows from {raw['n'].nunique()} cases on {sorted(raw['gpu'].unique())} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")
