# Integration benchmark: momentum assembly on the occDrivAer mesh

Assembly of the whole momentum equation ddt(U) + div(phi,U) - laplacian(nu,U) as the scalar matrix of
the segregated equation (Vec3 rhs) on the 65,334,765 cell occDrivAer mesh of the OpenFOAM HPC
Challenge (460,041,233 non-zeros), every NeoN, fused NeoN and JuNe strategy on one H200 GPU.

- Case: nobackup/driveAerLukas/occDrivAerStaticMesh with div(phi,U) switched to Gauss upwind (JuNe has
  no linearUpwind), Euler ddt, uncorrected laplacian, nu = 1.507e-5.
- Run: NeoFOAM benchmarks/june/drivaer_assembly.job (test/test_cavity.cpp with CAVITY_DDT=1
  CAVITY_MATRIX=scalar), job 630461 on gpu-nvidia-h200-3, log in h200/.
- Timing: five assemblies per strategy, the first dropped; each includes the reset of the system and is
  fenced before and after, NeoN assembles the operators only, i.e. both sides do the same work. Every
  strategy's matrix and rhs equal the JuNe batched reference.
- Throughput: nnz * 8 bytes / median time. `python3 parse_plot.py` -> drivaer.csv, drivaer_h200.pdf.
