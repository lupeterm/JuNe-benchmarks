cmake_minimum_required(VERSION 3.25)
project(atomic_bench_kokkos LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Same Kokkos source and NeoN headers as the local NeoN build
set(NEON_DIR "$ENV{HOME}/clones/neoninstall/NeoN" CACHE PATH "NeoN checkout (for Vec3 header)")
set(KOKKOS_SRC "${NEON_DIR}/build/develop/_deps/kokkos-src" CACHE PATH "Kokkos source tree")

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
set(Kokkos_ARCH_ADA89 ON CACHE BOOL "")
add_subdirectory(${KOKKOS_SRC} kokkos EXCLUDE_FROM_ALL)

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${NEON_DIR}/include)
# NeoN's default (NeoN_DEFINE_DP_SCALAR=ON): scalar = double
target_compile_definitions(bench PRIVATE NeoN_DP_SCALAR=1)
target_link_libraries(bench PRIVATE Kokkos::kokkos)
