#!/usr/bin/env python3
"""Merge the benchmark CSVs and print markdown tables (median ms per kernel launch).

Usage: compare.py results/*.csv > results/summary.md
"""
import csv
import sys
from collections import defaultdict

rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        rows += list(csv.DictReader(f))

# the scatter pattern must be identical across implementations
idxsums = defaultdict(set)
for r in rows:
    idxsums[(r["N"], r["M"])].add(r["idxsum"])
mismatch = [k for k, v in idxsums.items() if len(v) > 1]
if mismatch:
    sys.exit(f"index patterns differ between implementations for (N, M) = {mismatch}")

failed = [r for r in rows if r["check"] == "FAIL"]
impls = sorted({r["impl"] for r in rows}, key=lambda s: (not s.startswith("kokkos"), s))
times = {(r["impl"], r["variant"], int(r["M"])): r["median_ms"] for r in rows}
n = int(rows[0]["N"])
ms = sorted({int(r["M"]) for r in rows})


def fmt(v):
    if v is None:
        return ""
    if v == "nan":
        return "skipped"
    v = float(v)
    return f"{v:.3f}" if v < 100 else f"{v:.0f}"


def table(title, variants):
    cols = [(i, v) for v in variants for i in impls if any((i, v, m) in times for m in ms)]
    print(f"### {title}\n")
    print("| M | N/M | " + " | ".join(f"{i}<br>{v}" for i, v in cols) + " |")
    print("|---:|---:|" + "---:|" * len(cols))
    for m in ms:
        cells = [fmt(times.get((i, v, m))) for i, v in cols]
        print(f"| {m} | {n // m} | " + " | ".join(cells) + " |")
    print()


def ratio_table(title, num, den):
    print(f"### {title}\n")
    print("| M | N/M | " + " | ".join(impls) + " |")
    print("|---:|---:|" + "---:|" * len(impls))
    for m in ms:
        cells = []
        for i in impls:
            a, b = times.get((i, num, m)), times.get((i, den, m))
            ok = a not in (None, "nan") and b not in (None, "nan")
            cells.append(f"{float(a) / float(b):.2f}x" if ok else "")
        print(f"| {m} | {n // m} | " + " | ".join(cells) + " |")
    print()


print(f"# Atomic-add contention benchmark (N = {n} updates, median ms per launch)\n")
if failed:
    print("**CHECK FAILED:** " + ", ".join(f"{r['impl']}/{r['variant']}/M={r['M']}" for r in failed) + "\n")
table("double", ["scalar_plain", "scalar_atomic"])
table("vec3", ["vec3_plain", "vec3_atomic_cmpt", "vec3_atomic_struct"])
ratio_table("Cost of atomics: scalar_atomic / scalar_plain", "scalar_atomic", "scalar_plain")
ratio_table("Cost of atomics: vec3_atomic_cmpt / vec3_plain", "vec3_atomic_cmpt", "vec3_plain")
ratio_table("Cost of the lock: vec3_atomic_struct / vec3_atomic_cmpt", "vec3_atomic_struct", "vec3_atomic_cmpt")
