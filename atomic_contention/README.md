# Atomic-add contention benchmark

Cost of atomic adds under contention on the GPU: Julia (KernelAbstractions + Atomix, as in
JuNe.jl) vs C++/Kokkos (as in NeoN).

```bash
./run.sh            # N=2^24, 20 reps, 3 warmup -> results/*.csv, summary.md, plot.pdf
```

## Results

`results/` holds a run on an RTX 4070 workstation (CUDA 13.2, driver 610.57, Kokkos 5.0.2,
Julia 1.12.1 with KernelAbstractions 0.9.45, Atomix 1.5.0, CUDA.jl 6.4.2 as pinned in
`julia/Manifest.toml`): `summary.md` has the tables, `plot.pdf` the figure.

[median time per launch vs updates per target](results/plot.pdf)

- Julia/Atomix and Kokkos emit the same hardware atomics and are within a few percent at high
  contention; at moderate contention Kokkos is ~10-50% faster.
- The lock-based `atomic_add(NeoN::Vec3)` is ~2.5x slower than three component atomics while
  the targets fit in L2, 17x at 16k updates per target, and impractical beyond. Once the target
  array exceeds the 36 MB L2 everything is DRAM-bound and the lock costs nothing extra.
- In the mid range the atomics beat the racy non-atomic baseline: an atomic whose result is
  unused is presumably fire-and-forget at L2 (`red`), while a plain `+=` has to load before it
  stores (not verified in the SASS).

## What is measured

Each kernel performs `N` updates `t[idx[i]] += v[i]` scattered onto `M` targets, with
`idx[i] = splitmix64(i) % M` generated identically on both sides (the `idxsum` column
cross-checks this). `M` sweeps `1, 4, 16, …, N`, i.e. from every thread hitting one address
(`N/M = N` updates per target) to essentially no conflicts.

| variant              | Kokkos                                         | Julia                               |
|----------------------|------------------------------------------------|-------------------------------------|
| `scalar_plain`       | `t(j) += v(i)`                                 | `t[j] += v[i]`                      |
| `scalar_atomic`      | `Kokkos::atomic_add(&t(j), v(i))`              | `Atomix.@atomic t[j] += v[i]`       |
| `vec3_plain`         | `t(j) += v(i)` on `NeoN::Vec3`                 | 3× `t[3j-k] += v[3i-k]`             |
| `vec3_atomic_cmpt`   | 3× `Kokkos::atomic_add(&t(j)[k], v(i)[k])`     | 3× `Atomix.@atomic t[3j-k] += …`    |
| `vec3_atomic_struct` | `Kokkos::atomic_add(&t(j), v(i))` on `NeoN::Vec3` | — (Atomix has no struct atomic)  |

- `*_plain` are racy, non-atomic baselines (wrong results when `M < N`, timing only).
- `vec3_atomic_struct` is what NeoN does, e.g. `gaussGreenGrad.cpp`. `NeoN::Vec3` is
  24 bytes, which is too large for a hardware atomic, so Kokkos/desul falls back to a
  **lock-based atomic**: lock an entry of a global lock array hashed by address, do a
  non-atomic RMW, unlock. `vec3_atomic_cmpt` is the lock-free alternative that JuNe uses: three
  independent `atomicrmw fadd` on the doubles.
- Vec3 data is interleaved xyz on both sides (`View<Vec3*>` / `Vector{Float64}` of length 3N).
- Atomic results are checked exactly (all values are small integers, so the sums are
  order-independent); `check` is `ok`/`FAIL`.

## Setup details

- Kokkos 5.0.2 from the local NeoN checkout (`$KOKKOS_SRC`, default
  `~/clones/neoninstall/NeoN/build/develop/_deps/kokkos-src`), built here in Release with CUDA,
  `ADA89`, and the real `NeoN/core/primitives/vec3.hpp` with `NeoN_DP_SCALAR=1`.
- Host compiler `g++-14` through `nvcc_wrapper` (nvcc 13.2 rejects GCC 16).
- Julia package versions are bounded by JuNe.jl's `[compat]` (`julia/Project.toml`).
- Kokkos picks its own CUDA block size for a `RangePolicy` (as in NeoN); Julia runs with
  workgroup size 64 (as in JuNe) and 256.
- Timing: wall clock around launch + `fence`/`synchronize`, median (and min) over the reps,
  target array zeroed before each rep outside the timed region. Includes launch overhead
  (a few µs), which only matters for the smallest times.
- `vec3_atomic_struct` is skipped where `N/M > maxchain` (default 16384): at `M = 1` the lock
  serialises everything and 2^16 updates already take ~5 s on an RTX 4070, so N = 2^24 would
  take hours. Runs over 1 s are measured 3 times instead of `reps`.

Individual sides:

```bash
kokkos/build/bench [N] [reps] [warmup] [maxchain] > kokkos.csv
julia --project=julia julia/bench.jl [N] [reps] [warmup] [workgroupsize] > julia.csv
python3 compare.py *.csv
python3 plot.py results   # stdlib only; PNG via rsvg-convert if installed
```
