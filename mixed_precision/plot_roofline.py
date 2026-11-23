"""Roofline of JuNe's whole-assembly precision configurations on the H200: the momentum equation
ddt + div + laplacian on the 400^3 cavity, scalar matrix, JuNe face-based, cell-based and batched in
fp64, fp32, fp16 and bf16, measured with Nsight Compute (NeoFOAM benchmarks/june/precision/
profile_roofline.job): every kernel inside the NVTX range "precision <strategy> <configuration>", i.e.
one reset of matrix and rhs plus the assembly, untimed and outside the timed runs. Only the assembly
kernels count: the reset is a fill kernel in fp64 and bf16 but a driver memset in fp32 and fp16, which
Nsight Compute does not see. Nsight Compute locks the clocks (SM at its base clock), so times, rates and
ceilings all refer to that clock.

    FLOPs   add + mul + 2 * fma over the fp64, fp32 and half-precision thread instructions (Nsight
            Compute's convention; bf16 arithmetic runs on the half-precision units, partly in fp32;
            the 16-bit compare-and-swap loops of the atomics add one half-precision add per retry)
    bytes   DRAM bytes read + written
    time    sum of the kernel durations (the gaps between the kernels are not included)

A whole assembly is the sum over its kernels, one panel per strategy. Ceilings:
Nsight Compute's peak sustained rates of the device (DRAM bytes per cycle times the DRAM clock, FMA
thread instructions per cycle times the SM clock, times 2; fp32 and half precision come out equal).

    python3 plot_roofline.py [csv]   reads h200/roofline_400.csv (the raw page with the NVTX ranges),
                                     writes roofline.csv and roofline.pdf
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_precision import COLOURS, LABELS, LARGE

STRATEGIES = ["face-based", "cell-based", "batched"]
OPS = {t: [f"sm__sass_thread_inst_executed_op_{t}{o}_pred_on.sum" for o in ("add", "mul", "fma")] for t in "dfh"}
RANGE = re.compile(r"precision (\S+) (fp64|fp32|fp16|bf16)")


UNITS = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0, "ns": 1e-9, "us": 1e-6, "ms": 1e-3,
         "s": 1.0, "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
         "byte/cycle": 1.0, "Kbyte/cycle": 1e3, "Mbyte/cycle": 1e6, "hz": 1.0, "Khz": 1e3, "Mhz": 1e6, "Ghz": 1e9,
         "inst": 1.0, "inst/cycle": 1.0}


def load(path: Path):
    """the kernels of every NVTX range (without the reset's fill kernels: fp32 and fp16 reset with a driver
    memset, which is no kernel, fp64 and bf16 with a fill kernel) and the device peaks"""
    raw = pd.read_csv(path, low_memory=False)
    units, raw = raw.iloc[0], raw.iloc[1:]
    value = lambda r, c: float(str(r[c]).replace(",", "")) * UNITS[str(units[c]).strip()]
    nvtx = next(c for c in raw.columns if "Push/Pop_Range" in c)
    kernels = []
    for _, r in raw.iterrows():
        m = RANGE.search(str(r[nvtx]))
        if not m or "fill_kernel" in str(r["Kernel Name"]):
            continue
        flops = {t: value(r, ops[0]) + value(r, ops[1]) + 2 * value(r, ops[2]) for t, ops in OPS.items()}
        kernels.append({"strategy": m.group(1), "precision": m.group(2), "kernel": str(r["Kernel Name"])[:60],
                        "time_s": value(r, "gpu__time_duration.sum"),
                        "bytes": value(r, "dram__bytes_read.sum") + value(r, "dram__bytes_write.sum"),
                        **{f"flop_{t}": v for t, v in flops.items()}})
    k = pd.DataFrame(kernels)
    k["flops"] = k[["flop_d", "flop_f", "flop_h"]].sum(axis=1)
    first = raw.iloc[0]
    sm_hz = value(first, "sm__cycles_elapsed.avg.per_second")
    peaks = {"DRAM": value(first, "dram__bytes.sum.peak_sustained") * value(first, "dram__cycles_elapsed.avg.per_second"),
             **{p: 2 * value(first, f"sm__sass_thread_inst_executed_op_{t}fma_pred_on.sum.peak_sustained") * sm_hz
                for p, t in (("fp64", "d"), ("fp32", "f"), ("fp16", "h"))}}
    return k, peaks, sm_hz


def plot(whole, peaks, path):
    """one panel per strategy: the whole assembly in every precision under the device's ceilings"""
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(5.2 * len(STRATEGIES), 5.4), sharey=True,
                             constrained_layout=True)
    ai = np.logspace(-2, 2, 400)
    ceilings = {}  # peak -> precisions with that peak, so that equal ceilings share one line and label
    for p in ("fp64", "fp32", "fp16"):
        ceilings.setdefault(round(peaks[p] / 1e12), []).append(p)
    for ax, s in zip(axes, STRATEGIES):
        for tflops, ps in ceilings.items():
            peak = peaks[ps[0]]
            ax.plot(ai, np.minimum(peaks["DRAM"] * ai, peak), color="0.35", linestyle="--" if len(ps) > 1 else "-",
                    linewidth=1.3)
            above = peak == max(peaks[q] for q in ("fp64", "fp32", "fp16"))  # the lower one goes below its line
            ax.text(ai[-1] * 0.8, peak * (1.12 if above else 1 / 1.12), f"{' / '.join(ps)} {tflops} TFLOP/s",
                    ha="right", va="bottom" if above else "top", fontsize=12, color="0.25")
        ax.text(1.4e-2, peaks["DRAM"] * 1.4e-2 * 1.7, f"DRAM {peaks['DRAM'] / 1e12:.1f} TB/s", fontsize=12,
                color="0.25", rotation=37, rotation_mode="anchor")
        for p in COLOURS:
            w = whole[whole["strategy"].eq(s) & whole["precision"].eq(p)]
            ax.scatter(w["ai"], w["flop_per_s"], s=110, marker="o", color=COLOURS[p], edgecolors="0.1",
                       linewidths=0.8, zorder=3, label=p)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ai[0], ai[-1])
        ax.set_ylim(peaks["DRAM"] * ai[0] * 0.5, max(peaks.values()) * 3)
        ax.set_title(LABELS[s])
        ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    axes[0].set_ylabel("Performance [FLOP/s]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=len(COLOURS), frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "h200" / "roofline_400.csv"
    kernels, peaks, sm_hz = load(csv)
    whole = kernels.groupby(["strategy", "precision"], as_index=False)[["time_s", "bytes", "flops", "flop_d", "flop_f", "flop_h"]].sum()
    whole["ai"] = whole["flops"] / whole["bytes"]
    whole["flop_per_s"] = whole["flops"] / whole["time_s"]
    whole["gbs"] = whole["bytes"] / whole["time_s"] / 1e9
    whole["dram_share"] = whole["gbs"] * 1e9 / peaks["DRAM"]
    whole.to_csv(here / "roofline.csv", index=False)
    print(f"SM clock under Nsight Compute {sm_hz / 1e9:.3f} GHz;", {k: f"{v:.3g}" for k, v in peaks.items()})
    print(whole[["strategy", "precision", "time_s", "bytes", "flops", "ai", "flop_per_s", "gbs", "dram_share"]].to_string(index=False))
    plt.style.use("seaborn-v0_8-whitegrid")
    with plt.rc_context(LARGE):
        plot(whole, peaks, here / "roofline.pdf")
