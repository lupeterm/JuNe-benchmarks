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
