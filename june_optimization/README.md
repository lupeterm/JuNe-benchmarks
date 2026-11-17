# JuNe with Julia's default optimisation against -O3 on the GPU

Does a higher Julia optimisation level speed up JuNe's GPU assembly? The cavity benchmark (NeoFOAM
test/test_cavity.cpp: div + laplacian, scalar matrix, five fenced assemblies per strategy, the first
dropped) on the 300^3 lid-driven cavity (27,000,000 cells, 188,460,000 non-zeros) on one H200, with
the embedded Julia started either with its defaults (-O2) or with -O3 (JUNE_JULIA_ARGS="-O3", read by
test_cavity.cpp before jl_init and printed in every log). Every JuNe strategy plus fused NeoN
face-based as a reference that Julia's options cannot affect; each variant in three separate
processes, alternating (june_opt.job, job 632989 on gpu-nvidia-h200-3, logs in h200/).

Result: no difference. Every strategy has the same median with both options, to within 0.3 %:

| strategy | -O2 [ms] | -O3 [ms] |
|---|---|---|
| JuNe face-based | 7.01 | 7.02 |
| JuNe global face-based | 7.82 | 7.82 |
| JuNe cell-based | 11.98 | 11.98 |
| JuNe batched | 11.78 | 11.78 |
| NeoN fused face-based (reference) | 5.61 | 5.61 |

This is expected: on the GPU, Julia's -O only applies to JuNe's host code (argument wrapping, operator
parsing, kernel launch). CUDA.jl compiles the kernels with GPUCompiler at LLVM level 2 whatever Julia
is started with, and ptxas runs at its default -O3. The first assembly, which includes the
compilation, is about 1 to 3 % slower with -O3.

The stronger option --check-bounds=no could not be measured: it changes Julia's type inference, which
leaves JuNe's atomic updates (Atomix) with a non-inferable call, and CUDA.jl rejects the face-based,
global face-based and cell-based kernels with InvalidIRError (job 632923). Removing the bounds checks
per kernel with `@kernel inbounds=true` would avoid that, but needs a change to JuNe.
