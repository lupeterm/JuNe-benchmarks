# icoFoam cavity assembly benchmark

OpenFOAM baseline for the JuNe/NeoFOAM cavity benchmark: the instrumented `icoFoam` of
`openfoam-OpenFOAM-v2412` assembles `div(phi,U) - laplacian(nu,U)` (one warm-up, one timed
assembly, `Momentum <s>` printed by every rank) on 64 MPI ranks of the H200 node's CPUs.

1. `sbatch --nodelist=gpu-nvidia-h200-2 icofoam/cavity_icofoam.job` runs every cavity size three
   times and writes `results/h200-icofoam-64/cavity_<n>.log`. It uses the existing 64-rank
   decompositions in `nobackup/ldcDist` through view cases in `nobackup/june-icofoam` that take
   `system/` (upwind div, uncorrected laplacian) from the NeoFOAM benchmark cases.
2. `python3 icofoam/parse_results.py` takes the slowest rank per run, averages the runs and writes
   `results/cavity_icofoam.csv` with the columns of the NeoFOAM `cavity_throughput.csv`
   (`throughput_gbs` = nnz * 24 B / time).
