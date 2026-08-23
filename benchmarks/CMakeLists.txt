# ElastiQP benchmarks. Enable with -DELASTIQP_BUILD_BENCHMARKS=ON.
#
# The core benchmarks are self-contained: they need nothing beyond Eigen and
# this repo (robot benchmarks replay the committed data/robot_sequences.bin,
# so Pinocchio is only needed to REgenerate that file). Cross-solver
# comparisons against PIQP and ProxQP are opt-in via
# -DELASTIQP_BENCH_EXTERNAL_SOLVERS=ON, which downloads pinned release
# tarballs at configure time (header-only; nothing is vendored or built).

option(ELASTIQP_BENCH_EXTERNAL_SOLVERS
  "Fetch PIQP and ProxSuite (pinned release tarballs) and build the cross-solver benchmarks"
  OFF)

set(ELASTIQP_BENCH_DATA
  ${CMAKE_CURRENT_SOURCE_DIR}/data/robot_sequences.bin)

# Arch provenance for CSV output. For a codegen comparison, configure a
# second build tree with -DCMAKE_CXX_FLAGS=-march=native and diff the CSVs
# on the arch column.
if(CMAKE_CXX_FLAGS MATCHES "march=native")
  set(ELASTIQP_ARCH_LABEL "native")
else()
  set(ELASTIQP_ARCH_LABEL "generic")
endif()

function(elastiqp_bench name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/common          # problem_gen.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common) # qp_io.hpp
  target_link_libraries(${name} PRIVATE elastiqp::elastiqp)
  target_compile_definitions(${name} PRIVATE
    ELASTIQP_ARCH_LABEL="${ELASTIQP_ARCH_LABEL}"
    ELASTIQP_SEQUENCE_FILE="${ELASTIQP_BENCH_DATA}")
endfunction()

# --- core (Eigen-only) -------------------------------------------------------
elastiqp_bench(bench_random_qp     bench_random_qp.cc)
elastiqp_bench(bench_collision_2d  bench_collision_2d.cc)
elastiqp_bench(bench_robot_control bench_robot_control.cc)
elastiqp_bench(bench_relax_warm    bench_relax_warm.cc)
elastiqp_bench(bench_fwd_warm      bench_fwd_warm.cc)
elastiqp_bench(bench_bcl_strategies bench_bcl_strategies.cc)
elastiqp_bench(bench_diff_robot    bench_diff_robot.cc)
elastiqp_bench(bench_eq_elastic    bench_eq_elastic.cc)

# --- robotics generator + correctness tests (need Pinocchio) -----------------
# Need an installed Pinocchio (ROS/pip/conda binary or source build; point
# CMAKE_PREFIX_PATH at it). Only regeneration of the committed data file and
# the builder correctness tests live here -- the benchmarks above replay the
# committed file and never touch Pinocchio.
find_package(pinocchio QUIET)
if(pinocchio_FOUND)
  message(STATUS
    "elastiqp benchmarks: pinocchio ${pinocchio_VERSION} found -- building robotics targets")
  add_executable(gen_robot_sequences robotics/gen_robot_sequences.cc)
  add_executable(test_robot_control robotics/test_robot_control.cc)
  foreach(tgt gen_robot_sequences test_robot_control)
    target_include_directories(${tgt} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/robotics
      ${CMAKE_CURRENT_SOURCE_DIR}/common
      ${CMAKE_SOURCE_DIR}/common)
    target_link_libraries(${tgt} PRIVATE
      elastiqp::elastiqp pinocchio::pinocchio)
    # Binary pinocchio distributions (ROS, pip) are generic builds; compiling
    # code that constructs pinocchio types with -march=native widens Eigen's
    # fixed-size alignment (AVX) and shifts the joint-variant layout relative
    # to the precompiled library, crashing at runtime (boost::bad_get). Only
    # these targets touch pinocchio, so neutralize the flag for them (the
    # trailing -march wins). Turn OFF if your pinocchio itself was built with
    # -march=native.
    option(ELASTIQP_ROBOT_GENERIC_ARCH
      "Compile robotics targets with generic -march to match binary pinocchio builds" ON)
    if(ELASTIQP_ROBOT_GENERIC_ARCH
       AND CMAKE_CXX_FLAGS MATCHES "march=native"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      target_compile_options(${tgt} PRIVATE -march=x86-64 -mtune=generic)
    endif()
  endforeach()
  # The vendored PIQP subset used by the main test suite also cross-validates
  # the robotics problem builders on the expanded (n+p) formulation.
  if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/piqp/include)
    target_compile_definitions(test_robot_control PRIVATE ELASTIQP_HAVE_PIQP)
    target_include_directories(test_robot_control PRIVATE
      ${CMAKE_SOURCE_DIR}/third_party/piqp/include)
  endif()
  add_test(NAME elastiqp.robot_control COMMAND test_robot_control)

  # Explicit-only regeneration of the COMMITTED data file (writes into the
  # source tree on purpose; never part of the default build). See
  # benchmarks/README.md for the full regeneration workflow.
  add_custom_target(regen_robot_sequences
    COMMAND gen_robot_sequences ${ELASTIQP_BENCH_DATA}
    DEPENDS gen_robot_sequences
    COMMENT "Regenerating benchmarks/data/robot_sequences.bin (source tree)")
else()
  message(STATUS "elastiqp benchmarks: pinocchio not found -- generator/tests "
    "skipped; robot benchmarks replay the committed data file "
    "(hint: -DCMAKE_PREFIX_PATH=<pinocchio prefix>)")
endif()

# --- cross-solver benchmarks (opt-in; PIQP + ProxSuite via FetchContent) -----
# Pinned release tarballs, consumed header-only: SOURCE_SUBDIR points at a
# directory that does not exist, so FetchContent_MakeAvailable downloads and
# populates but never calls add_subdirectory (proxsuite's own CMake needs its
# jrl-cmakemodules machinery; PIQP's would work but adds nothing over an
# include path). PIQP is pinned to the same version as the vendored subset in
# third_party/piqp that the tests use, so reference and timing runs see the
# same solver.
if(ELASTIQP_BENCH_EXTERNAL_SOLVERS)
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

  add_library(bench_piqp INTERFACE)
  target_include_directories(bench_piqp INTERFACE
    ${piqp_src_SOURCE_DIR}/include)
  target_link_libraries(bench_piqp INTERFACE Eigen3::Eigen)

  # proxsuite header-only: its only generated header is proxsuite/config.hpp
  # (version macros), synthesized here from the release's package.xml.
  # Vectorization (simde) is enabled when a system simde is available,
  # matching upstream's default configuration.
  set(PROXSUITE_VERSION "0.0.0")
  if(EXISTS ${proxsuite_src_SOURCE_DIR}/package.xml)
    file(READ ${proxsuite_src_SOURCE_DIR}/package.xml _proxsuite_pkg)
    if(_proxsuite_pkg MATCHES "<version>([0-9]+)\\.([0-9]+)\\.([0-9]+)</version>")
      set(PROXSUITE_VERSION
        "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
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
  target_include_directories(bench_proxsuite INTERFACE
    ${proxsuite_src_SOURCE_DIR}/include ${PROXSUITE_CONFIG_DIR})
  target_link_libraries(bench_proxsuite INTERFACE Eigen3::Eigen)
  find_path(SIMDE_INCLUDE_DIR simde/x86/avx2.h)
  if(SIMDE_INCLUDE_DIR)
    message(STATUS "elastiqp benchmarks: simde found at ${SIMDE_INCLUDE_DIR} "
      "-- building proxsuite with PROXSUITE_VECTORIZE")
    target_include_directories(bench_proxsuite INTERFACE ${SIMDE_INCLUDE_DIR})
    target_compile_definitions(bench_proxsuite INTERFACE PROXSUITE_VECTORIZE)
    string(APPEND ELASTIQP_ARCH_LABEL "+simde")
  else()
    message(STATUS "elastiqp benchmarks: simde not found -- building "
      "proxsuite without vectorization (apt install libsimde-dev)")
  endif()

  # elastiqp_bench reads ELASTIQP_ARCH_LABEL at call time, so targets added
  # below automatically carry the "+simde" provenance appended above.
  function(elastiqp_bench_ext name)
    elastiqp_bench(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE bench_piqp bench_proxsuite)
  endfunction()

  elastiqp_bench_ext(bench_robot_multisolver
    external/bench_robot_multisolver.cc)
  elastiqp_bench_ext(bench_condensed_vs_expanded
    external/bench_condensed_vs_expanded.cc)
  elastiqp_bench_ext(bench_proxqp_closest
    external/bench_proxqp_closest.cc)
  # Paper experiment (examples/experiments): all routes, feasible + conflict.
  elastiqp_bench_ext(robot_solver_comparison
    ${CMAKE_SOURCE_DIR}/examples/experiments/robot_solver_comparison.cc)

  # Maros-Meszaros: the .mat files ship inside the proxsuite release; a
  # build-time Python script (needs scipy) packs the n <= 200 subset into a
  # flat binary so the C++ benchmark needs no .mat reader.
  set(MM_DATA_DIR ${proxsuite_src_SOURCE_DIR}/test/data/maros_meszaros_data)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  set(MM_SCIPY_OK FALSE)
  if(Python3_FOUND)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import scipy.io"
      RESULT_VARIABLE _mm_scipy_result OUTPUT_QUIET ERROR_QUIET)
    if(_mm_scipy_result EQUAL 0)
      set(MM_SCIPY_OK TRUE)
    endif()
  endif()
  if(MM_SCIPY_OK)
    set(MM_PACKED ${CMAKE_CURRENT_BINARY_DIR}/maros_meszaros_small.bin)
    add_custom_command(
      OUTPUT ${MM_PACKED}
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_maros_meszaros.py
        ${MM_DATA_DIR} ${MM_PACKED} --nmax 200
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_maros_meszaros.py
      COMMENT "Packing Maros-Meszaros dense subset")
    add_custom_target(maros_meszaros_data DEPENDS ${MM_PACKED})
    elastiqp_bench_ext(bench_maros_meszaros external/bench_maros_meszaros.cc)
    add_dependencies(bench_maros_meszaros maros_meszaros_data)
    target_compile_definitions(bench_maros_meszaros PRIVATE
      ELASTIQP_MM_DATA="${MM_PACKED}")
  else()
    message(STATUS "elastiqp benchmarks: python3 with scipy not found -- "
      "skipping bench_maros_meszaros")
  endif()
endif()
