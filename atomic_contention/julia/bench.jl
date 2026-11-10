# Atomic-add contention benchmark, Julia side (KernelAbstractions + Atomix on CUDA).
#
# N updates are scattered onto M targets (idx[i] = splitmix64(i) % M, identical to the
# Kokkos side). Vec3s are interleaved xyz Float64 vectors, as in JuNe. Variants:
#   scalar_plain      t[j] += v[i]                (racy, no-atomic baseline)
#   scalar_atomic     Atomix.@atomic t[j] += v[i]
#   vec3_plain        3x t[3j-k] += v[3i-k]       (racy, no-atomic baseline)
#   vec3_atomic_cmpt  3x Atomix.@atomic on the components (lock-free; Atomix has no
#                     lock-based struct atomic, so there is no vec3_atomic_struct here)
#
# Usage: julia --project bench.jl [N=2^24] [reps=20] [warmup=3] [workgroupsize=64] > julia.csv
using KernelAbstractions, Atomix, CUDA

function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    z = x
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

@kernel function scalar_plain!(t, idx, v)
    i = @index(Global)
    @inbounds begin
        j = idx[i] + 1
        t[j] += v[i]
    end
end

@kernel function scalar_atomic!(t, idx, v)
    i = @index(Global)
    @inbounds begin
        j = idx[i] + 1
        Atomix.@atomic t[j] += v[i]
    end
end

@kernel function vec3_plain!(t, idx, v)
    i = @index(Global)
    @inbounds begin
        j = idx[i] + 1
        t[3j-2] += v[3i-2]
        t[3j-1] += v[3i-1]
        t[3j]   += v[3i]
    end
end

@kernel function vec3_atomic_cmpt!(t, idx, v)
    i = @index(Global)
    @inbounds begin
        j = idx[i] + 1
        Atomix.@atomic t[3j-2] += v[3i-2]
        Atomix.@atomic t[3j-1] += v[3i-1]
        Atomix.@atomic t[3j]   += v[3i]
    end
end

# Median/min over `reps` runs after `warmup` runs, wall clock around launch + synchronize
# (same as Kokkos::Timer + fence on the Kokkos side). A run longer than SLOW_MS cuts the
# measurement down to that run plus 2 more.
const SLOW_MS = 1000.0
function time_kernel(backend, kernel!, t, args, n; reps, warmup)
    ms = Float64[]
    for r in 1:(warmup + reps)
        fill!(t, 0.0)
        KernelAbstractions.synchronize(backend)
        t0 = time_ns()
        kernel!(t, args...; ndrange=n)
        KernelAbstractions.synchronize(backend)
        dt = (time_ns() - t0) / 1e6
        (r > warmup || dt > SLOW_MS) && push!(ms, dt)
        dt > SLOW_MS && length(ms) >= 3 && break
    end
    sort!(ms)
    return ms[cld(length(ms) + 1, 2)], ms[1]
end

function report(impl, variant, n, m, idxsum, (med, mn), check)
    println(join((impl, variant, n, m, idxsum, round(med; digits=6), round(mn; digits=6),
                  round(n / (med * 1e-3) / 1e9; digits=4), check), ','))
    flush(stdout)
end

function main(args)
    n      = length(args) >= 1 ? parse(Int, args[1]) : 2^24
    reps   = length(args) >= 2 ? parse(Int, args[2]) : 20
    warmup = length(args) >= 3 ? parse(Int, args[3]) : 3
    wg     = length(args) >= 4 ? parse(Int, args[4]) : 64
    impl = "julia_wg$wg"

    backend = CUDABackend()
    println(stderr, "KernelAbstractions ", pkgversion(KernelAbstractions), ", Atomix ",
            pkgversion(Atomix), ", CUDA.jl ", pkgversion(CUDA), ", Julia ", VERSION, ", ",
            CUDA.name(CUDA.device()), ", N=$n reps=$reps warmup=$warmup workgroupsize=$wg")
    println("impl,variant,N,M,idxsum,median_ms,min_ms,gupdates_per_s,check")

    vs = CUDA.fill(1.0, n)
    vv = CuArray(repeat([1.0, 2.0, 3.0], n))
    idx = CuArray{Int32}(undef, n)
    kernels = (scalar_plain  = scalar_plain!(backend, wg),  scalar_atomic    = scalar_atomic!(backend, wg),
               vec3_plain    = vec3_plain!(backend, wg),    vec3_atomic_cmpt = vec3_atomic_cmpt!(backend, wg))

    m = 1
    while m <= n
        idxH = Int32[splitmix64(UInt64(i)) % UInt64(m) for i in 0:n-1]
        idxsum = sum(UInt64, idxH)
        copyto!(idx, idxH)
        ts = CUDA.zeros(Float64, m)
        tv = CUDA.zeros(Float64, 3m)
        # atomics on exact small integers -> the sums are exact regardless of order
        checkS() = sum(Array(ts)) == n ? "ok" : "FAIL"
        function checkV()
            h = Array(tv)
            sum(@view h[1:3:end]) == n && sum(@view h[2:3:end]) == 2n &&
                sum(@view h[3:3:end]) == 3n ? "ok" : "FAIL"
        end
        kw = (; reps, warmup)

        report(impl, "scalar_plain", n, m, idxsum,
               time_kernel(backend, kernels.scalar_plain, ts, (idx, vs), n; kw...), "racy")
        report(impl, "scalar_atomic", n, m, idxsum,
               time_kernel(backend, kernels.scalar_atomic, ts, (idx, vs), n; kw...), checkS())
        report(impl, "vec3_plain", n, m, idxsum,
               time_kernel(backend, kernels.vec3_plain, tv, (idx, vv), n; kw...), "racy")
        report(impl, "vec3_atomic_cmpt", n, m, idxsum,
               time_kernel(backend, kernels.vec3_atomic_cmpt, tv, (idx, vv), n; kw...), checkV())
        m *= 4
    end
end

main(ARGS)
