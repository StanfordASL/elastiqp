cmake_minimum_required(VERSION 3.18)

# Determine version from package.xml
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/package.xml _elastiqp_package_xml)
string(REGEX MATCH "<version>([0-9]+\\.[0-9]+\\.[0-9]+)</version>" _
  "${_elastiqp_package_xml}")
project(elastiqp VERSION ${CMAKE_MATCH_1} LANGUAGES CXX)

enable_testing()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(elastiqp INTERFACE)
add_library(elastiqp::elastiqp ALIAS elastiqp)
target_include_directories(elastiqp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(elastiqp INTERFACE Eigen3::Eigen)
target_compile_features(elastiqp INTERFACE cxx_std_17)

# Build tests. Skip inside a pip wheel build (scikit). The tests
# cross-validate the solver against vendored PIQP (third_party/piqp) and a
# test-only IPM reference implementation (tests/ipm_reference.hpp); neither
# is installed or distributed. common/ holds dev-only headers (problem
# generators) shared by the tests and benchmarks.
if(NOT SKBUILD)
  add_executable(test_pdal tests/test_pdal.cc)
  target_include_directories(test_pdal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/piqp/include)
  target_link_libraries(test_pdal PRIVATE elastiqp::elastiqp)
  add_test(NAME elastiqp.pdal_cross_validation COMMAND test_pdal)

  # Regression watch for the warm-start saturation-creep failure mode
  # (see benchmarks/bench_bcl_strategies.cc). Needs only Eigen; the
  # trajectory generator lives in benchmarks/common but is header-only.
  add_executable(test_bcl_creep tests/test_bcl_creep.cc)
  target_include_directories(test_bcl_creep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/common)
  target_link_libraries(test_bcl_creep PRIVATE elastiqp::elastiqp)
  add_test(NAME elastiqp.bcl_creep COMMAND test_bcl_creep)

  # Regression watch for the warm-start deactivation-creep failure mode
  # (pinch-release tick from examples/experiments/constraint_conflict_demo.py;
  # see tests/test_gap_creep.cc). Needs only Eigen.
  add_executable(test_gap_creep tests/test_gap_creep.cc)
  target_link_libraries(test_gap_creep PRIVATE elastiqp::elastiqp)
  add_test(NAME elastiqp.gap_creep COMMAND test_gap_creep)

  add_executable(test_ipm_reference tests/test_ipm_reference.cc)
  target_include_directories(test_ipm_reference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/piqp/include)
  target_link_libraries(test_ipm_reference PRIVATE elastiqp::elastiqp)
  add_test(NAME elastiqp.ipm_reference_validation COMMAND test_ipm_reference)
endif()

# Build benchmarks if flag provided
option(ELASTIQP_BUILD_BENCHMARKS "Build the benchmarks/ directory" OFF)
if(ELASTIQP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Build JAX FFI
add_library(elastiqp_jax_ffi SHARED bindings/jax_ffi.cc)
target_include_directories(elastiqp_jax_ffi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xla_ffi)
target_link_libraries(elastiqp_jax_ffi PRIVATE elastiqp::elastiqp)
if(SKBUILD)
  install(TARGETS elastiqp_jax_ffi LIBRARY DESTINATION elastiqp)
endif()

# Build python bindings
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
    if(SKBUILD)
      install(TARGETS elastiqp_ext LIBRARY DESTINATION elastiqp)
    endif()
  else()
    message(STATUS "elastiqp: Python found but nanobind not importable -- skipping bindings")
  endif()
else()
  message(STATUS "elastiqp: Python not found -- skipping bindings")
endif()

# Skip install/export if in a pip build
if(SKBUILD)
  return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS elastiqp EXPORT elastiqpTargets)
install(EXPORT elastiqpTargets
  NAMESPACE elastiqp::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elastiqp)

configure_package_config_file(
  cmake/elastiqpConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/elastiqpConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elastiqp)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/elastiqpConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/elastiqpConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/elastiqpConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/elastiqp)
