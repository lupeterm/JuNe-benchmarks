# NeoN Vec3 assembly: one Vec3 atomic vs three scalar atomics

Every face of NeoN's face-based assembly updates the diagonal of its owner and neighbour row (and
the boundary kernels the rhs) with `Kokkos::atomic_sub`. For the coupled Vec3 matrix the operand is
a 24-byte `Vec3`. Kokkos' atomic layer (desul, bundled with Kokkos 5.0.2) only has lock-free atomics
for 4, 8 and 16-byte types (`desul/atomics/Compare_Exchange_CUDA.hpp`,
`device_atomic_always_lock_free`); every other type takes the lock-based fallback
(`desul/atomics/Lock_Based_Fetch_Op_CUDA.hpp`): hash the address into a global lock array, spin
until the lock is acquired, fenced read-modify-write, release, repeated in a warp-wide ballot loop
until every lane has succeeded.

This benchmark measures what that fallback costs in the real assembly by running the same kernels
with the Vec3 update done as

1. **whole**: `1x atomic_sub(Vec3)`, the lock-based fallback (NeoN as is), or
2. **component**: `3x atomic_sub(scalar)`, one hardware atomic per component.

The variant is a runtime switch in NeoN (`NEON_VEC3_COMPONENT_ATOMICS=1`, `vec3ComponentAtomics()`
and `atomicAdd`/`atomicSub` in `NeoN/core/parallelAlgorithms.hpp`), used by the face-based div,
laplacian and fused div+laplacian kernels (interior and boundary) and by the boundary contribution in
`linearSystem.hpp`. The cell-based, batched and global face-based strategies are untouched. The
synthetic counterpart, atomic adds of doubles and Vec3 under controlled contention in Julia and
Kokkos, is in `../atomic_contention`.

## Protocol

`benchmarks/june/cavity_vec3_atomics.job` of NeoFOAM: the cavity microbenchmark
(`test/test_cavity.cpp`, implicit div + laplacian of the momentum equation, Vec3 matrix) on the
200^3 lid-driven cavity (8,000,000 cells, 55,760,000 non-zeros), strategies "NeoN face-based" and
"NeoN fused face-based", three timed assemblies each, the first one dropped. Throughput is
nnz * 24 B / time, as in `../june_neon`. Both variants are checked against the test's reference
matrix (the JuNe batched assembly), which passed in every run. The log of every run states the
active variant (`Vec3 atomics: ...`).

```
python3 parse_plot.py     # <gpu>/cavity_<n>_<variant>.log -> cavity_vec3_atomics.csv, figures
```

## Results

| GPU | strategy | 1x atomic_sub(Vec3) | 3x atomic_sub(scalar) |
|---|---|---|---|
| L40S | face-based | 32.8 ms, 41 GB/s | 33.2 ms, 40 GB/s |
| L40S | fused face-based | 15.8 ms, 85 GB/s | 15.2 ms, 88 GB/s |

Figure: `cavity_vec3_atomics_l40s_n200.pdf`. The runtime switch and the job script were
temporary and have been removed from NeoN and NeoFOAM again after this run; the H200 job was
cancelled before it started.

On the L40S the two variants are within run-to-run noise (the face-based component runs scatter by
5 ms around the same mean). At the contention of a face-based assembly, about six updates per
diagonal entry, the lock-based fallback costs the same as three hardware atomics: both are a few
global-memory round trips per update. The Vec3 atomic implementation is therefore not what makes
the Vec3 assembly slower than the scalar one; the remaining factor is the matrix traffic itself.
