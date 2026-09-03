cmake_minimum_required(VERSION 3.18)

# Determine version from package.xml
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/package.xml _elastiqp_package_xml)
string(REGEX MATCH "<version>([0-9]+\\.[0-9]+\\.[0-9]+)</version>" _
  "${_elastiqp_package_xml}")
project(elastiqp VERSION ${CMAKE_MATCH_1} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Tests and Python bindings are built when this is the top-level project.
# A consumer that pulls this tree in via add_subdirectory (the
# elastiqp_benchmarks repo, a robot stack) gets only the header-only target
# unless it opts in. The tests are self-contained (Eigen only; the oracle is
# the in-tree IPM reference). Cross-validation against external solvers
# (PIQP, ProxQP) and all benchmarks live in the separate elastiqp_benchmarks
# repo, which consumes this one.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_elastiqp_top_level ON)
else()
  set(_elastiqp_top_level OFF)
endif()
option(ELASTIQP_BUILD_TESTS "Build the elastiqp test suite" ${_elastiqp_top_level})
option(ELASTIQP_BUILD_PYTHON "Build the Python bindings and JAX FFI" ${_elastiqp_top_level})
if(SKBUILD)  # pip wheel build: bindings only, no install/export
  set(ELASTIQP_BUILD_TESTS OFF)
  set(ELASTIQP_BUILD_PYTHON ON)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

# --- the library (header-only) -----------------------------------------------
add_library(elastiqp INTERFACE)
add_library(elastiqp::elastiqp ALIAS elastiqp)
target_include_directories(elastiqp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(elastiqp INTERFACE Eigen3::Eigen)
target_compile_features(elastiqp INTERFACE cxx_std_17)

# Dev-only headers shared by the tests here and by the benchmarks repo:
# problem generators, the drifting-trajectory generator, and the test-only
# IPM reference solver (tests/support/). Never installed.
add_library(elastiqp_testing INTERFACE)
add_library(elastiqp::testing ALIAS elastiqp_testing)
target_include_directories(elastiqp_testing INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
target_link_libraries(elastiqp_testing INTERFACE elastiqp::elastiqp)

# --- tests -------------------------------------------------------------------
if(ELASTIQP_BUILD_TESTS)
  enable_testing()
  function(elastiqp_test name)
    add_executable(${name} tests/${name}.cc)
    target_link_libraries(${name} PRIVATE elastiqp::testing)
  endfunction()

  # Solver correctness, relax() and the KKT VJP against the IPM reference.
  elastiqp_test(test_pdal)
  add_test(NAME elastiqp.pdal COMMAND test_pdal)

  # Regression watch for the warm-start saturation-creep failure mode
  # (isolated by bench_bcl_strategies in the benchmarks repo).
  elastiqp_test(test_bcl_creep)
  add_test(NAME elastiqp.bcl_creep COMMAND test_bcl_creep)

  # Regression watch for the warm-start deactivation-creep failure mode
  # (pinch-release tick from examples/experiments/constraint_conflict_demo.py;
  # see tests/test_gap_creep.cc).
  elastiqp_test(test_gap_creep)
  add_test(NAME elastiqp.gap_creep COMMAND test_gap_creep)
endif()

# --- Python bindings + JAX FFI -----------------------------------------------
if(ELASTIQP_BUILD_PYTHON)
  add_library(elastiqp_jax_ffi SHARED bindings/jax_ffi.cc)
  target_include_directories(elastiqp_jax_ffi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xla_ffi)
  target_link_libraries(elastiqp_jax_ffi PRIVATE elastiqp::elastiqp)
  if(SKBUILD)
    install(TARGETS elastiqp_jax_ffi LIBRARY DESTINATION elastiqp)
  endif()

  find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
      OUTPUT_VARIABLE NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE NB_RESULT ERROR_QUIET)
    if(NB_RESULT EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
      find_package(nanobind CONFIG REQUIRED)
      nanobind_add_module(elastiqp_ext NB_STATIC bindings/bindings.cc)
      set_target_properties(elastiqp_ext PROPERTIES OUTPUT_NAME _core)
      target_link_libraries(elastiqp_ext PRIVATE elastiqp::elastiqp)
      if(SKBUILD)
        install(TARGETS elastiqp_ext LIBRARY DESTINATION elastiqp)
      endif()
      if(ELASTIQP_BUILD_TESTS)
        set(_elastiqp_test_env
          "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
          "ELASTIQP_BUILD_DIR=$<TARGET_FILE_DIR:elastiqp_ext>")
        add_test(NAME elastiqp.bindings
          COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py)
        set_tests_properties(elastiqp.bindings PROPERTIES
          ENVIRONMENT "${_elastiqp_test_env}")
        execute_process(
          COMMAND "${Python_EXECUTABLE}" -c "import jax"
          RESULT_VARIABLE JAX_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(JAX_IMPORT_RESULT EQUAL 0)
          add_test(NAME elastiqp.jax_ffi
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_jax_ffi.py)
          set_tests_properties(elastiqp.jax_ffi PROPERTIES
            ENVIRONMENT "${_elastiqp_test_env}")
        else()
          message(STATUS "elastiqp: jax not importable -- skipping the jax_ffi test")
        endif()
      endif()
    else()
      message(STATUS "elastiqp: Python found but nanobind not importable -- skipping bindings")
    endif()
  else()
    message(STATUS "elastiqp: Python not found -- skipping bindings")
  endif()
endif()

# --- install / find_package(elastiqp) export ---------------------------------
if(NOT SKBUILD)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/elastiqp_install.cmake)
endif()
