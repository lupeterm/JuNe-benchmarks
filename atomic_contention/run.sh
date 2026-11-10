#!/usr/bin/env bash
# Build and run both sides, then summarise.
# Usage: ./run.sh [N=16777216] [reps=20] [warmup=3] [maxchain=16384]
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-16777216}
REPS=${2:-20}
WARMUP=${3:-3}
MAXCHAIN=${4:-16384}

# nvcc 13.2 rejects GCC > 15 as host compiler
export NVCC_WRAPPER_DEFAULT_COMPILER=${NVCC_WRAPPER_DEFAULT_COMPILER:-g++-14}
KOKKOS_SRC=${KOKKOS_SRC:-$HOME/clones/neoninstall/NeoN/build/develop/_deps/kokkos-src}

cmake -S kokkos -B kokkos/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$KOKKOS_SRC/bin/nvcc_wrapper" -DKOKKOS_SRC="$KOKKOS_SRC" > /dev/null
cmake --build kokkos/build -j > /dev/null
julia --project=julia -e 'using Pkg; Pkg.instantiate()'

mkdir -p results
kokkos/build/bench "$N" "$REPS" "$WARMUP" "$MAXCHAIN" > results/kokkos.csv
for wg in 64 256; do
    julia --project=julia julia/bench.jl "$N" "$REPS" "$WARMUP" "$wg" > "results/julia_wg$wg.csv"
done
python3 compare.py results/*.csv | tee results/summary.md
python3 plot.py results
