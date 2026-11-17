# Mixed-precision outlook: JuNe's GPU assembly of the momentum equation in reduced precision

The momentum equation ddt(U) + div(phi,U) - laplacian(nu,U) (Euler, upwind, uncorrected; scalar
matrix, Vec3 rhs) on the lid-driven cavity from 50^3 to 400^3 in steps of 50 (250^3 and 350^3 are
copies of the 260^3 case with the cell count changed), every JuNe GPU strategy, one H200 (job
633303, logs in h200/). Driver: NeoFOAM test_cavity.cpp with CAVITY_PRECISION=1 CAVITY_DDT=1 calls
NeoN's june::assemble with the Julia wrapper "benchmark_precision" (JuNe.jl src/precision.jl).

Configurations per strategy and size:

| family | configurations | matrix |
|---|---|---|
| whole | all operators in fp64, fp32, fp16, bf16 | same type |
| reduce | one of ddt, div, laplacian in fp32 or bf16, the others in fp64 | fp64 |
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
- deviation: largest difference of matrix and rhs from the fp64 result of the same expression,
  relative to its largest entry. The old velocity is zero, so ddt adds nothing to the rhs, and ddt's
  diagonal is far below the largest matrix entry, so its error barely shows in this measure.

`python3 plot_precision.py` -> precision.csv, precision_deviation.csv and
precision_whole_speedup.pdf, precision_whole_gbs.pdf, precision_reduce_speedup.pdf,
precision_alone_speedup.pdf, precision_overheads.pdf (curves over the number of cells).

Speedup over fp64 at 400^3 (64M cells; fp64: face 17.8 ms, global 19.6, cell 28.2, batched 28.8):

| configuration | face-based | global face-based | cell-based | batched |
|---|---|---|---|---|
| whole fp32 | 1.45 | 1.48 | 2.74 | 1.77 |
| whole fp16 | 0.83 | 0.76 | 5.93 | 2.60 |
| whole bf16 | 0.80 | 0.73 | 4.91 | 2.37 |
| ddt reduced, fp32 / bf16 | 1.02 / 1.02 | 1.01 / 1.02 | 1.01 / 1.02 | 1.00 / 1.01 |
| div reduced, fp32 / bf16 | 1.01 / 1.01 | 1.05 / 1.05 | 1.23 / 1.36 | 0.95 / 0.95 |
| laplacian reduced, fp32 / bf16 | 1.03 / 1.04 | 1.05 / 1.05 | 1.22 / 1.33 | 1.00 / 1.03 |
| ddt alone, fp32 / bf16 | 1.88 / 1.70 | 1.89 / 1.70 | 1.54 / 1.83 | 1.84 / 1.62 |
| div alone, fp32 / bf16 | 1.36 / 0.76 | 1.38 / 0.81 | 2.55 / 3.42 | 1.95 / 2.50 |
| laplacian alone, fp32 / bf16 | 1.38 / 0.67 | 1.40 / 0.70 | 2.16 / 2.97 | 1.94 / 2.60 |

Overheads at 400^3 (median over the strategies, ms): whole fp32 allocation 10.8, conversion 4.8,
back-conversion 2.9 (assembly 12.8); whole fp16/bf16 5.5, 4.4, 2.7 (assembly 16 to 17); one operator
in bf16: allocation 0.8 to 2.0, conversion 0.9 to 2.5, no back-conversion.

Deviation at 400^3: fp32 1.3e-7 (rhs 2.5e-8), bf16 6.7e-3 (1.4e-3), fp16 0.67 (1.0: face areas and
fluxes underflow, unusable); laplacian alone in bf16 1.3e-3, div 5.4e-5, ddt 2.2e-9.

The earlier div + laplacian study (4 configurations, 100^3 to 400^3, job 633259) is in
archive_divlap/. atomics_test.* holds the 16-bit atomics test (exact, about 10x slower than 32/64 bit).
