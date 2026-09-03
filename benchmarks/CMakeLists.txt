cmake_minimum_required(VERSION 3.18)
project(elastiqp_benchmarks LANGUAGES CXX)

# ElastiQP benchmarks, cross-solver comparisons, and the tests that compare
# elastiqp against external solvers. Standalone project; see README.md.
#
# Dependencies, and where each is resolved:
#   Eigen3            find_package (system / conda / vcpkg)
#   elastiqp          cmake/elastiqp.cmake  (local checkout or git fetch)
#   PIQP, ProxSuite   cmake/external_solvers.cmake (pinned release tarballs,
#                     header-only; ELASTIQP_BENCH_EXTERNAL_SOLVERS)
#   Pinocchio         find_package, optional (regenerating robot data only)
#   Python + scipy    optional (packing the Maros-Meszaros set)

enable_testing()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ELASTIQP_BENCH_EXTERNAL_SOLVERS
  "Fetch PIQP and ProxSuite (pinned release tarballs) and build the cross-solver benchmarks and tests"
  ON)

find_package(Eigen3 REQUIRED NO_MODULE)
include(cmake/elastiqp.cmake)  # -> elastiqp::elastiqp, elastiqp::testing

set(ELASTIQP_BENCH_DATA ${CMAKE_CURRENT_SOURCE_DIR}/data/robot_sequences.bin)

# Arch provenance for CSV output. For a codegen comparison, configure a
# second build tree with -DCMAKE_CXX_FLAGS=-march=native and diff the CSVs
# on the arch column.
if(CMAKE_CXX_FLAGS MATCHES "march=native")
  set(ELASTIQP_ARCH_LABEL "native")
else()
  set(ELASTIQP_ARCH_LABEL "generic")
endif()

if(ELASTIQP_BENCH_EXTERNAL_SOLVERS)
  include(cmake/external_solvers.cmake)  # -> bench::piqp, bench::proxsuite
endif()

# Shared helpers: common/qp_io.hpp (EQPS robot-sequence reader) plus
# elastiqp::testing (problem_gen.hpp, drift_traj.hpp, ipm_reference.hpp).
add_library(bench_common INTERFACE)
target_include_directories(bench_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(bench_common INTERFACE elastiqp::testing)

# elastiqp_bench reads ELASTIQP_ARCH_LABEL at call time, so targets added
# after external_solvers.cmake carry its "+simde" provenance.
function(elastiqp_bench name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE bench_common)
  target_compile_definitions(${name} PRIVATE
    ELASTIQP_ARCH_LABEL="${ELASTIQP_ARCH_LABEL}"
    ELASTIQP_SEQUENCE_FILE="${ELASTIQP_BENCH_DATA}")
endfunction()
function(elastiqp_bench_ext name src)
  elastiqp_bench(${name} ${src})
  target_link_libraries(${name} PRIVATE bench::piqp bench::proxsuite)
endfunction()

# --- core benchmarks (Eigen-only) --------------------------------------------
# Robot benchmarks replay the committed data/robot_sequences.bin, so
# Pinocchio is only needed to REgenerate that file (below).
elastiqp_bench(bench_random_qp      core/bench_random_qp.cc)
elastiqp_bench(bench_collision_2d   core/bench_collision_2d.cc)
elastiqp_bench(bench_robot_control  core/bench_robot_control.cc)
elastiqp_bench(bench_relax_warm     core/bench_relax_warm.cc)
elastiqp_bench(bench_fwd_warm       core/bench_fwd_warm.cc)
elastiqp_bench(bench_bcl_strategies core/bench_bcl_strategies.cc)
elastiqp_bench(bench_diff_robot     core/bench_diff_robot.cc)
elastiqp_bench(bench_eq_elastic     core/bench_eq_elastic.cc)

# --- cross-solver benchmarks + tests (PIQP, ProxQP) --------------------------
if(ELASTIQP_BENCH_EXTERNAL_SOLVERS)
  elastiqp_bench_ext(bench_robot_multisolver     external/bench_robot_multisolver.cc)
  elastiqp_bench_ext(bench_condensed_vs_expanded external/bench_condensed_vs_expanded.cc)
  elastiqp_bench_ext(bench_proxqp_closest        external/bench_proxqp_closest.cc)
  # Paper experiment: all routes, feasible + conflict variants.
  elastiqp_bench_ext(robot_solver_comparison     experiments/robot_solver_comparison.cc)

  # elastiqp::Solver vs vanilla PIQP (strict problems + the expanded
  # (n+p) elastic formulation), and the same for the test-only IPM
  # reference that elastiqp's own test suite uses as its oracle.
  elastiqp_bench_ext(test_piqp_cross_validation tests/test_piqp_cross_validation.cc)
  add_test(NAME elastiqp_bench.piqp_cross_validation COMMAND test_piqp_cross_validation)
  elastiqp_bench_ext(test_ipm_reference tests/test_ipm_reference.cc)
  add_test(NAME elastiqp_bench.ipm_reference_validation COMMAND test_ipm_reference)

  # Maros-Meszaros: the .mat files ship inside the proxsuite release; a
  # build-time Python script (needs scipy) packs the n <= 200 subset into a
  # flat binary so the C++ benchmark needs no .mat reader.
  set(MM_DATA_DIR ${PROXSUITE_SOURCE_DIR}/test/data/maros_meszaros_data)
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
    message(STATUS "elastiqp_benchmarks: python3 with scipy not found -- "
      "skipping bench_maros_meszaros")
  endif()
endif()

# --- robotics generator + correctness tests (need Pinocchio) -----------------
# Only regeneration of the committed data file and the builder correctness
# tests live here; point CMAKE_PREFIX_PATH at an installed Pinocchio
# (ROS/pip/conda binary or source build).
find_package(pinocchio QUIET)
if(pinocchio_FOUND)
  message(STATUS
    "elastiqp_benchmarks: pinocchio ${pinocchio_VERSION} found -- building robotics targets")
  add_executable(gen_robot_sequences robotics/gen_robot_sequences.cc)
  add_executable(test_robot_control tests/test_robot_control.cc)
  foreach(tgt gen_robot_sequences test_robot_control)
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/robotics)
    target_link_libraries(${tgt} PRIVATE bench_common pinocchio::pinocchio)
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
  if(ELASTIQP_BENCH_EXTERNAL_SOLVERS)
    # Also cross-validate the robotics problem builders against PIQP on the
    # expanded (n+p) formulation.
    target_compile_definitions(test_robot_control PRIVATE ELASTIQP_HAVE_PIQP)
    target_link_libraries(test_robot_control PRIVATE bench::piqp)
  endif()
  add_test(NAME elastiqp_bench.robot_control COMMAND test_robot_control)

  # Explicit-only regeneration of the COMMITTED data file (writes into the
  # source tree on purpose; never part of the default build). See README.md
  # for the full regeneration workflow.
  add_custom_target(regen_robot_sequences
    COMMAND gen_robot_sequences ${ELASTIQP_BENCH_DATA}
    DEPENDS gen_robot_sequences
    COMMENT "Regenerating data/robot_sequences.bin (source tree)")
else()
  message(STATUS "elastiqp_benchmarks: pinocchio not found -- generator/tests "
    "skipped; robot benchmarks replay the committed data file "
    "(hint: -DCMAKE_PREFIX_PATH=<pinocchio prefix>)")
endif()
