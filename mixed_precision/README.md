# Mixed-precision outlook: JuNe's GPU assembly of the momentum equation in reduced precision

The momentum equation ddt(U) + div(phi,U) - laplacian(nu,U) (Euler, upwind, uncorrected; scalar
matrix, Vec3 rhs) on the lid-driven cavity from 50^3 to 400^3 in steps of 50 (250^3 and 350^3 are
copies of the 260^3 case with the cell count changed), every JuNe GPU strategy, one H200 (job
641555 on gpu-nvidia-h200-3, logs in h200/). Driver: NeoFOAM test_cavity.cpp with CAVITY_PRECISION=1
CAVITY_DDT=1 calls NeoN's june::assemble with the Julia wrapper "benchmark_precision" (JuNe.jl
src/precision.jl). JuNe's matrix atomics are relaxed (JuNe.jl src/types.jl relaxed_add!): fp64 and
fp32 as PTX red (REDG), fp16 and bf16 as relaxed compare-and-swap loops (the H200 has no scalar 16-bit
atomic add). The earlier runs (job 633303 with Atomix's acquire-release atomics, job 638986 without
fp16 in the reduce family and without the reset timing) are in the git history.

Configurations per strategy and size:

| family | configurations | matrix |
|---|---|---|
| whole | all operators in fp64, fp32, fp16, bf16 | same type |
| reduce | one of ddt, div, laplacian in fp32, fp16 or bf16, the others in fp64 | fp64 |
| alone | one operator alone in fp64, fp32, fp16, bf16 | same type |

An operator in precision P computes in P from its own inputs stored in P (ddt: volumes, old velocity;
div: flux, cell-based face signs; laplacian: gamma, delta coefficients, face areas) and adds into the
matrix in its type. The boundary data div and laplacian share stays in fp64 when only one of them is
reduced and is converted when read (about 1 % of the input data at 400^3).

Measured, five times each, the first dropped, median reported, synchronised before and after:
- assembly: reset of matrix and rhs + the JuNe GPU entry;
- allocation: the device arrays of the reduced inputs and, if the matrix is reduced, of the outputs
  (CUDA.jl's pool trimmed before every allocation, so it is not served from its cache);
- conversion: fp64 -> P of the reduced inputs, on the device;
- back-conversion: matrix and rhs -> fp64, if the matrix is reduced;
- reset: the reset of matrix and rhs alone (part of the assembly time), for the operator shares;
- deviation: largest difference of matrix and rhs from the fp64 result of the same expression,
  relative to its largest entry. The old velocity is zero, so ddt adds nothing to the rhs, and ddt's
  diagonal is far below the largest matrix entry, so its error barely shows in this measure.

`python3 plot_precision.py` -> precision.csv, precision_deviation.csv and
precision_whole_speedup.pdf, precision_whole_gbs.pdf, precision_reduce_speedup.pdf,
precision_alone_speedup.pdf, precision_overheads.pdf (curves over the number of cells);
`--no-global` writes the same figures without global face-based (suffix _noglobal).
`python3 plot_shares.py [--no-global]` -> precision_shares.csv, precision_shares.pdf (see below).

Speedup over fp64 at 400^3 (64M cells; fp64: face 14.8 ms, global 15.9, cell 28.1, batched 28.9):

| configuration | face-based | global face-based | cell-based | batched |
|---|---|---|---|---|
| whole fp32 | 1.51 | 1.52 | 2.74 | 1.78 |
| whole fp16 | 1.13 | 1.17 | 6.00 | 2.55 |
| whole bf16 | 1.05 | 1.10 | 4.95 | 2.35 |
| ddt reduced, fp32 / fp16 / bf16 | 1.02 / 1.03 / 1.03 | 1.02 / 1.03 / 1.03 | 1.01 / 1.02 / 1.02 | 1.00 / 1.01 / 1.01 |
| div reduced, fp32 / fp16 / bf16 | 1.01 / 1.01 / 1.02 | 1.03 / 1.03 / 1.04 | 1.23 / 1.37 / 1.37 | 0.95 / 0.96 / 0.97 |
| laplacian reduced, fp32 / fp16 / bf16 | 1.03 / 1.05 / 1.05 | 1.05 / 1.05 / 1.06 | 1.21 / 1.34 / 1.34 | 1.00 / 1.03 / 1.03 |
| ddt alone, fp32 / fp16 / bf16 | 1.88 / 2.60 / 1.68 | 1.89 / 2.60 / 1.68 | 1.53 / 2.28 / 1.81 | 1.84 / 2.42 / 1.60 |
| div alone, fp32 / fp16 / bf16 | 1.39 / 0.97 / 0.89 | 1.41 / 1.00 / 0.92 | 2.55 / 4.27 / 3.42 | 1.95 / 2.77 / 2.50 |
| laplacian alone, fp32 / fp16 / bf16 | 1.42 / 0.96 / 0.89 | 1.45 / 0.98 / 0.91 | 2.20 / 3.79 / 3.06 | 1.95 / 2.87 / 2.58 |

Overheads at 400^3 (median over the strategies, ms): whole fp32 allocation 11.0, conversion 4.8,
back-conversion 2.9 (assembly 10.3); whole fp16/bf16 5.5, 4.5, 2.7 (assembly 12 to 13); one operator
in fp16 or bf16: allocation 0.7 to 2.2, conversion 0.9 to 2.6, no back-conversion. The reset of
matrix and rhs takes 1.33 ms in fp64, 0.61 in fp32, 0.31 in fp16, but 1.31 in bf16 (as slow as fp64;
cause not investigated).

Deviation at 400^3: fp32 1.3e-7 (rhs 2.5e-8), bf16 6.7e-3 (1.4e-3), fp16 0.67 (1.0: face areas and
fluxes underflow, unusable); alone in bf16: laplacian 1.4e-3, div 7.7e-3, ddt 3.2e-3; one operator in
fp16: ddt 6.9e-7, div 3.9e-3, laplacian 0.67 (rhs 1.0: the face areas underflow, unusable).

## Operator shares

plot_shares.py splits the 400^3 assembly of every strategy and precision into the reset, the loop
the operators share and ddt, div and laplacian, from the whole assembly, every operator alone and the
reset (equations in its docstring): in face-based, global face-based and batched ddt is a kernel of its
own and div and laplacian share the face loops (indices, face data, atomic updates, boundary kernel);
in cell-based one cell kernel serves all three. The parts add up to the whole assembly by
construction; batched fp64 gives div -1.1 ms (labelled in the figure): there the operators do not
add up exactly (not investigated). Face-based fp64 (14.8 ms): reset 1.3, shared face loop 8.7, ddt 3.5, div 0.3,
laplacian 1.0, i.e. the face loop with its index loads and atomics dominates, the arithmetic of the
two operators is small; cell-based fp64 (28.1 ms): reset 1.3, cell loop 5.6, ddt 1.9, div 10.4,
laplacian 8.8, which drop to 0.4 and 0.5 ms in fp16.

The earlier div + laplacian study (4 configurations, 100^3 to 400^3, job 633259) is in
archive_divlap/. atomics_test.* holds the 16-bit atomics test (exact, about 10x slower than 32/64 bit).

## Roofline

roofline.pdf places the whole assembly (ddt + div + laplacian, 400^3, scalar matrix) of JuNe
face-based, cell-based and batched in fp64, fp32, fp16 and bf16 under the H200's ceilings, one panel
per strategy. Measured with Nsight Compute (profile_roofline.job, job 641734 on gpu-nvidia-h200-3):
one untimed reset + assembly per configuration in the NVTX range "precision <strategy> <precision>"
of JuNe.jl src/precision.jl, its assembly kernels only (the reset is a fill kernel in fp64 and bf16
but a driver memset, invisible to Nsight Compute, in fp32 and fp16). FLOPs: add + mul + 2 fma over the
fp64, fp32 and half-precision instructions; bytes: DRAM read + write; time: the kernel durations at
the clock Nsight Compute locks (SM 1.36 GHz). Ceilings: DRAM 4.9 TB/s, fp64 23 TFLOP/s, fp32 and half
precision 46 TFLOP/s at that clock. `python3 plot_roofline.py` reads h200/roofline_400.csv and writes
roofline.csv and roofline.pdf.

Every configuration is memory-bound (0.05 to 0.65 FLOP/byte, ridge points at 4.7 and 9.4): reduced
precision moves the points to the right, i.e. the same FLOPs over fewer bytes. Share of the DRAM peak
reached: face-based 41 % (fp64), 34 % (fp32), 14 % (fp16, bf16: the 16-bit compare-and-swap loops of
the atomics, not the bandwidth, limit it); cell-based 56 %, 33 %, 42 %, 42 %; batched 63 %, 66 %,
57 %, 57 % (it moves about 4.7 times the interior-face data of face-based).
