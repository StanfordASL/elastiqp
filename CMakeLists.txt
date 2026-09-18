cmake_minimum_required(VERSION 3.18)

# Determine version from package.xml
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/package.xml _elastiqp_package_xml)
string(REGEX MATCH "<version>([0-9]+\\.[0-9]+\\.[0-9]+)</version>" _
  "${_elastiqp_package_xml}")
project(elastiqp VERSION ${CMAKE_MATCH_1} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()


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

# Main library
add_library(elastiqp INTERFACE)
add_library(elastiqp::elastiqp ALIAS elastiqp)
target_include_directories(elastiqp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(elastiqp INTERFACE Eigen3::Eigen)
target_compile_features(elastiqp INTERFACE cxx_std_17)

# Headers for tests (never installed)
add_library(elastiqp_testing INTERFACE)
add_library(elastiqp::testing ALIAS elastiqp_testing)
target_include_directories(elastiqp_testing INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
target_link_libraries(elastiqp_testing INTERFACE elastiqp::elastiqp)

# Tests
if(ELASTIQP_BUILD_TESTS)
  enable_testing()
  function(elastiqp_test name)
    add_executable(${name} tests/${name}.cc)
    target_link_libraries(${name} PRIVATE elastiqp::testing)
  endfunction()

  elastiqp_test(test_solvers)
  add_test(NAME elastiqp.solvers COMMAND test_solvers)

  elastiqp_test(test_das)
  add_test(NAME elastiqp.das COMMAND test_das)

  elastiqp_test(test_pdal)
  add_test(NAME elastiqp.pdal COMMAND test_pdal)

  elastiqp_test(test_ipm)
  add_test(NAME elastiqp.ipm COMMAND test_ipm)

  elastiqp_test(test_bcl_creep)
  add_test(NAME elastiqp.bcl_creep COMMAND test_bcl_creep)

  elastiqp_test(test_gap_creep)
  add_test(NAME elastiqp.gap_creep COMMAND test_gap_creep)

  elastiqp_test(test_ruiz)
  add_test(NAME elastiqp.ruiz COMMAND test_ruiz)

  elastiqp_test(test_lp)
  add_test(NAME elastiqp.lp COMMAND test_lp)
endif()

# Python bindings and JAX FFI
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
      # NOTE: Use NOMINSIZE when working with nanobind here as its default is
      # -Os but we want -O3
      nanobind_add_module(elastiqp_ext NB_STATIC NOMINSIZE bindings/bindings.cc)
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
        execute_process(
          COMMAND "${Python_EXECUTABLE}" -c "import torch"
          RESULT_VARIABLE TORCH_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(TORCH_IMPORT_RESULT EQUAL 0)
          add_test(NAME elastiqp.torch
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_torch.py)
          set_tests_properties(elastiqp.torch PROPERTIES
            ENVIRONMENT "${_elastiqp_test_env}")
        else()
          message(STATUS "elastiqp: torch not importable -- skipping the torch test")
        endif()
      endif()
    else()
      message(STATUS "elastiqp: Python found but nanobind not importable -- skipping bindings")
    endif()
  else()
    message(STATUS "elastiqp: Python not found -- skipping bindings")
  endif()
endif()

# Intstall/export
if(NOT SKBUILD)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/elastiqp_install.cmake)
endif()
