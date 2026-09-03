# External QP solvers for the cross-solver benchmarks and tests: PIQP and
# ProxSuite (ProxQP), as pinned, hash-verified release tarballs consumed
# header-only. Nothing is vendored into this repo and nothing is built:
# SOURCE_SUBDIR points at a directory that does not exist, so
# FetchContent_MakeAvailable downloads and populates but never calls
# add_subdirectory (proxsuite's own CMake needs its jrl-cmakemodules
# machinery; PIQP's would work but adds nothing over an include path).
#
# Exposes: bench::piqp, bench::proxsuite (INTERFACE targets), and
# PROXSUITE_SOURCE_DIR (the tarball also ships the Maros-Meszaros .mat
# files used by bench_maros_meszaros and python/run_maros_meszaros.py).
# Appends "+simde" to ELASTIQP_ARCH_LABEL in the caller's scope when
# ProxQP is vectorized.

include(FetchContent)
FetchContent_Declare(piqp_src
  URL https://github.com/PREDICT-EPFL/piqp/archive/refs/tags/v0.6.3.tar.gz
  URL_HASH SHA256=05c110365256995eef72436ab79855ad33226d8df7aaa8592489bc34531a3d1e
  SOURCE_SUBDIR cmake_disabled)
FetchContent_Declare(proxsuite_src
  URL https://github.com/Simple-Robotics/proxsuite/archive/refs/tags/v0.7.3.tar.gz
  URL_HASH SHA256=e634babff534c8812c6dbc6b022b1d01f9a2a4a9a73cf87a6dc299a4de996904
  SOURCE_SUBDIR cmake_disabled)
FetchContent_MakeAvailable(piqp_src proxsuite_src)
set(PROXSUITE_SOURCE_DIR ${proxsuite_src_SOURCE_DIR})

add_library(bench_piqp INTERFACE)
add_library(bench::piqp ALIAS bench_piqp)
target_include_directories(bench_piqp INTERFACE ${piqp_src_SOURCE_DIR}/include)
target_link_libraries(bench_piqp INTERFACE Eigen3::Eigen)

# proxsuite header-only: its only generated header is proxsuite/config.hpp
# (version macros), synthesized here from the release's package.xml.
set(PROXSUITE_VERSION "0.0.0")
if(EXISTS ${proxsuite_src_SOURCE_DIR}/package.xml)
  file(READ ${proxsuite_src_SOURCE_DIR}/package.xml _proxsuite_pkg)
  if(_proxsuite_pkg MATCHES "<version>([0-9]+)\\.([0-9]+)\\.([0-9]+)</version>")
    set(PROXSUITE_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
    set(PROXSUITE_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(PROXSUITE_VERSION_MINOR ${CMAKE_MATCH_2})
    set(PROXSUITE_VERSION_PATCH ${CMAKE_MATCH_3})
  endif()
endif()
set(PROXSUITE_CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR}/proxsuite_config)
file(WRITE ${PROXSUITE_CONFIG_DIR}/proxsuite/config.hpp
  "#pragma once\n"
  "#define PROXSUITE_VERSION \"${PROXSUITE_VERSION}\"\n"
  "#define PROXSUITE_MAJOR_VERSION ${PROXSUITE_VERSION_MAJOR}\n"
  "#define PROXSUITE_MINOR_VERSION ${PROXSUITE_VERSION_MINOR}\n"
  "#define PROXSUITE_PATCH_VERSION ${PROXSUITE_VERSION_PATCH}\n"
  "#define PROXSUITE_DLLAPI\n")
add_library(bench_proxsuite INTERFACE)
add_library(bench::proxsuite ALIAS bench_proxsuite)
target_include_directories(bench_proxsuite INTERFACE
  ${proxsuite_src_SOURCE_DIR}/include ${PROXSUITE_CONFIG_DIR})
target_link_libraries(bench_proxsuite INTERFACE Eigen3::Eigen)

# Vectorization (simde) when a system simde is available, matching
# upstream's default configuration.
find_path(SIMDE_INCLUDE_DIR simde/x86/avx2.h)
if(SIMDE_INCLUDE_DIR)
  message(STATUS "elastiqp_benchmarks: simde found at ${SIMDE_INCLUDE_DIR} "
    "-- building proxsuite with PROXSUITE_VECTORIZE")
  target_include_directories(bench_proxsuite INTERFACE ${SIMDE_INCLUDE_DIR})
  target_compile_definitions(bench_proxsuite INTERFACE PROXSUITE_VECTORIZE)
  string(APPEND ELASTIQP_ARCH_LABEL "+simde")
else()
  message(STATUS "elastiqp_benchmarks: simde not found -- building "
    "proxsuite without vectorization (apt install libsimde-dev)")
endif()
