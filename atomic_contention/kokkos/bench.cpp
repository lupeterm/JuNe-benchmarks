// Atomic-add contention benchmark, Kokkos side.
//
// N updates are scattered onto M targets (idx[i] = splitmix64(i) % M, identical to the
// Julia side). Variants:
//   scalar  plain         t[j] += v[i]                       (racy, no-atomic baseline)
//   scalar  atomic        Kokkos::atomic_add(&t[j], v[i])
//   vec3    plain         t[j] += v[i]                       (racy, no-atomic baseline)
//   vec3    atomic_cmpt   3x Kokkos::atomic_add on the scalar components (lock-free)
//   vec3    atomic_struct Kokkos::atomic_add(&t[j], v[i]) on NeoN::Vec3 (24 bytes, so Kokkos
//                         takes its lock-based fallback -- this is what NeoN does)
//
// Usage: bench [N=2^24] [reps=20] [warmup=3] [maxchain=16384] > kokkos.csv
//   vec3_atomic_struct is skipped where N/M (updates per target) > maxchain: at M=1 it
//   serialises on one lock and 2^16 updates already take ~5 s on an RTX 4070.
#include <Kokkos_Core.hpp>
#include <NeoN/core/primitives/vec3.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using NeoN::Vec3;
using Exec = Kokkos::DefaultExecutionSpace;
using Policy = Kokkos::RangePolicy<Exec>;

static uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Stats
{
    double medianMs, minMs;
};

// Median/min over `reps` runs after `warmup` runs. A run longer than SLOW_MS (the lock-based
// atomics at high contention) cuts the measurement down to that run plus 2 more.
constexpr double SLOW_MS = 1000.0;

template<class Reset, class Run>
Stats time(int reps, int warmup, Reset reset, Run run)
{
    std::vector<double> ms;
    for (int r = 0; r < warmup + reps; ++r)
    {
        reset();
        Kokkos::fence();
        Kokkos::Timer timer;
        run();
        Kokkos::fence();
        double t = timer.seconds() * 1e3;
        if (r >= warmup || t > SLOW_MS) ms.push_back(t);
        if (t > SLOW_MS && ms.size() >= 3) break;
    }
    std::sort(ms.begin(), ms.end());
    return {ms[ms.size() / 2], ms.front()};
}

static void report(
    const char* variant, int64_t n, int64_t m, uint64_t idxSum, Stats s, const char* check
)
{
    std::printf(
        "kokkos,%s,%ld,%ld,%lu,%.6f,%.6f,%.4f,%s\n",
        variant,
        n,
        m,
        idxSum,
        s.medianMs,
        s.minMs,
        double(n) / (s.medianMs * 1e-3) / 1e9,
        check
    );
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard(argc, argv);
    const int64_t n = argc > 1 ? std::atol(argv[1]) : (int64_t(1) << 24);
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 3;
    const int64_t maxChain = argc > 4 ? std::atol(argv[4]) : 16384;

    std::fprintf(
        stderr,
        "Kokkos %d.%d.%d, exec space %s, N=%ld reps=%d warmup=%d maxchain=%ld\n",
        KOKKOS_VERSION / 10000,
        KOKKOS_VERSION / 100 % 100,
        KOKKOS_VERSION % 100,
        Exec::name(),
        n,
        reps,
        warmup,
        maxChain
    );
    std::printf("impl,variant,N,M,idxsum,median_ms,min_ms,gupdates_per_s,check\n");

    Kokkos::View<double*> vs("vs", n);
    Kokkos::View<Vec3*> vv("vv", n);
    Kokkos::deep_copy(vs, 1.0);
    Kokkos::deep_copy(vv, Vec3(1.0, 2.0, 3.0));

    Kokkos::View<int32_t*> idx("idx", n);
    auto idxH = Kokkos::create_mirror_view(idx);

    for (int64_t m = 1; m <= n; m *= 4)
    {
        uint64_t idxSum = 0;
        for (int64_t i = 0; i < n; ++i)
        {
            idxH(i) = int32_t(splitmix64(uint64_t(i)) % uint64_t(m));
            idxSum += uint64_t(idxH(i));
        }
        Kokkos::deep_copy(idx, idxH);

        Kokkos::View<double*> ts("ts", m);
        Kokkos::View<Vec3*> tv("tv", m);
        auto resetS = [&] { Kokkos::deep_copy(ts, 0.0); };
        auto resetV = [&] { Kokkos::deep_copy(tv, Vec3(0.0, 0.0, 0.0)); };

        // atomics on exact small integers -> the sums are exact regardless of order
        auto checkS = [&]
        {
            double sum = 0.0;
            Kokkos::parallel_reduce(
                Policy(0, m), KOKKOS_LAMBDA(int64_t j, double& acc) { acc += ts(j); }, sum
            );
            return sum == double(n) ? "ok" : "FAIL";
        };
        auto checkV = [&]
        {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            Kokkos::parallel_reduce(
                Policy(0, m),
                KOKKOS_LAMBDA(int64_t j, double& ax, double& ay, double& az) {
                    ax += tv(j)[0];
                    ay += tv(j)[1];
                    az += tv(j)[2];
                },
                sx,
                sy,
                sz
            );
            return sx == double(n) && sy == 2.0 * n && sz == 3.0 * n ? "ok" : "FAIL";
        };

        Stats s;

        s = time(reps, warmup, resetS, [&] {
            Kokkos::parallel_for(
                "scalar_plain", Policy(0, n), KOKKOS_LAMBDA(int64_t i) { ts(idx(i)) += vs(i); }
            );
        });
        report("scalar_plain", n, m, idxSum, s, "racy");

        s = time(reps, warmup, resetS, [&] {
            Kokkos::parallel_for(
                "scalar_atomic",
                Policy(0, n),
                KOKKOS_LAMBDA(int64_t i) { Kokkos::atomic_add(&ts(idx(i)), vs(i)); }
            );
        });
        report("scalar_atomic", n, m, idxSum, s, checkS());

        s = time(reps, warmup, resetV, [&] {
            Kokkos::parallel_for(
                "vec3_plain", Policy(0, n), KOKKOS_LAMBDA(int64_t i) { tv(idx(i)) += vv(i); }
            );
        });
        report("vec3_plain", n, m, idxSum, s, "racy");

        s = time(reps, warmup, resetV, [&] {
            Kokkos::parallel_for(
                "vec3_atomic_cmpt",
                Policy(0, n),
                KOKKOS_LAMBDA(int64_t i) {
                    Vec3& t = tv(idx(i));
                    const Vec3 v = vv(i);
                    Kokkos::atomic_add(&t[0], v[0]);
                    Kokkos::atomic_add(&t[1], v[1]);
                    Kokkos::atomic_add(&t[2], v[2]);
                }
            );
        });
        report("vec3_atomic_cmpt", n, m, idxSum, s, checkV());

        if (n / m > maxChain)
        {
            std::printf("kokkos,vec3_atomic_struct,%ld,%ld,%lu,nan,nan,nan,skipped\n", n, m, idxSum);
            continue;
        }
        s = time(reps, warmup, resetV, [&] {
            Kokkos::parallel_for(
                "vec3_atomic_struct",
                Policy(0, n),
                KOKKOS_LAMBDA(int64_t i) { Kokkos::atomic_add(&tv(idx(i)), vv(i)); }
            );
        });
        report("vec3_atomic_struct", n, m, idxSum, s, checkV());
    }
    return 0;
}
