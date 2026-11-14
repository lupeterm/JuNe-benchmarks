# Integration benchmark: momentum assembly on the 64M cell lid-driven cavity

The DrivAer integration benchmark (../drivaer) on the 400^3 lid-driven cavity: assembly of the whole
momentum equation ddt(U) + div(phi,U) - laplacian(nu,U) as the scalar matrix of the segregated
equation (Vec3 rhs), 64,000,000 cells, 447,040,000 non-zeros, every NeoN, fused NeoN and JuNe
strategy on one H200 GPU.

- Case: nobackup/julia/view/base/mesh/400/case of the cavity series, unchanged: Euler ddt, Gauss
  upwind div(phi,U), Gauss linear uncorrected laplacian(nu,U); nu = 1.507e-5 as for the DrivAer.
- Run: NeoFOAM benchmarks/june/drivaer_assembly.job with NAME=cavity400 and CASE_DIR set to the case,
  job 631300 on gpu-nvidia-h200-2, log in h200/.
- Timing: as for the DrivAer, five fenced assemblies per strategy including the reset, the first
  dropped, NeoN assembling the operators only; every strategy's matrix and rhs equal the JuNe batched
  reference. One NeoN cell-based run took twice its median, the median is unaffected.
- Throughput: nnz * 8 bytes / median time. `python3 plot_integration.py cavity400` from the
  repository root -> cavity400.csv, cavity400_h200.pdf.

Unlike the 400^3 point of the cavity series in ../june_neon, which assembles div + laplacian only,
this run includes the ddt term.
