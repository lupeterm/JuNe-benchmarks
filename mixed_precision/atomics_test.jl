# Small test: do atomic adds work on the GPU for Float64, Float32, Float16 and BFloat16, the way
# JuNe's kernels use them (KernelAbstractions kernel + Atomix.@atomic), and through CUDA.jl directly
# (CUDA.@atomic)? For every type: correctness under contention (exact sums), the instruction the
# compiler emits (native atom.add or a compare-and-swap loop) and the time of a contended kernel.
#
#   julia --project=/storage/home/lupeterm/JuNe.jl atomics_test.jl
using CUDA, KernelAbstractions, Atomix, Printf

const BFloat16 = Base.root_module(Base.PkgId(Base.UUID("ab4f0b2a-ad5b-11e8-123f-65d77653426b"), "BFloat16s")).BFloat16
const TYPES = (Float64, Float32, Float16, BFloat16)

# every thread adds `val` to target[targets[i]]; with JuNe's face-based assembly each diagonal entry
# receives about six such updates (one per face of a hexahedral cell)
@kernel function ka_atomic_add!(target, @Const(targets), val)
    i = @index(Global)
    Atomix.@atomic target[targets[i]] += val
end

function cuda_atomic_add!(target, targets, val)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(targets)
        CUDA.@atomic target[targets[i]] += val
    end
    return
end

function atom_lines(ptx::String)
    lines = filter(l -> occursin(r"\batom\.|\bred\.|atom\.cas|cas\.", l), split(ptx, '\n'))
    isempty(lines) ? "no atomic instruction found" : join(unique(strip.(replace.(lines, r"%\w+" => "%r", r"\[[^\]]*\]" => "[..]"))), " | ")
end

function test(T, which; nTargets = 1 << 20, updatesPerTarget = 6)
    n = nTargets * updatesPerTarget
    targets = CuArray(Int32.(repeat(1:nTargets, inner = updatesPerTarget)))   # contended: neighbours collide
    target = CUDA.zeros(T, nTargets)
    val = T(0.125)                                                             # exact in every type
    run! = if which == :atomix
        k = ka_atomic_add!(CUDABackend(), 256)
        () -> (k(target, targets, val; ndrange = n); KernelAbstractions.synchronize(CUDABackend()))
    else
        () -> (@cuda threads = 256 blocks = cld(n, 256) cuda_atomic_add!(target, targets, val); CUDA.synchronize())
    end
    ok, msg, ptx, ms = false, "", "", NaN
    try
        # the first run compiles the kernel; its PTX is captured on the way
        buf = IOBuffer()
        CUDA.@device_code_ptx io = buf run!()
        ptx = String(take!(buf))
        expected = T(updatesPerTarget * 0.125)
        ok = all(Array(target) .== expected)
        msg = ok ? "exact" : @sprintf("WRONG, e.g. %s instead of %s", Array(target)[1], expected)
        times = Float64[]
        for _ in 1:5
            fill!(target, zero(T)); CUDA.synchronize()
            t = time_ns(); run!(); push!(times, (time_ns() - t) / 1e6)
        end
        ms = minimum(times)
    catch err
        msg = "FAILED: " * first(split(sprint(showerror, err), '\n'))
    end
    @printf("%-9s %-7s %-60s %8.3f ms  %s\n", string(T), string(which), msg, ms, isempty(ptx) ? "" : atom_lines(ptx))
end

println(CUDA.name(CUDA.device()), ", capability ", CUDA.capability(CUDA.device()))
for which in (:atomix, :cuda), T in TYPES
    test(T, which)
end
