# Locate the elastiqp source tree and add it as a subproject. This is the
# ONE place the benchmarks repo depends on elastiqp.
#
# Resolution order:
#   1. -DELASTIQP_SOURCE_DIR=<path>   explicit checkout
#   2. ../                            this repo nested inside the elastiqp tree
#   3. ../elastiqp                    sibling checkout
#   4. FetchContent (git, pinned to ELASTIQP_GIT_TAG)
#
# elastiqp is header-only, so consuming the source tree (rather than an
# installed package) costs nothing and gives access to the dev-only
# elastiqp::testing target (problem generators, drifting-trajectory
# generator, IPM reference) that the benchmarks and cross-solver tests use.
# The subproject builds only the library targets (its own tests and Python
# bindings are opt-in via ELASTIQP_BUILD_TESTS / ELASTIQP_BUILD_PYTHON).

set(ELASTIQP_SOURCE_DIR "" CACHE PATH
  "Path to an elastiqp source checkout (empty: auto-detect ../ or ../elastiqp, else fetch)")
set(ELASTIQP_GIT_REPOSITORY "https://github.com/StanfordASL/elastiqp"
  CACHE STRING "Where to fetch elastiqp from when no local checkout is found")
set(ELASTIQP_GIT_TAG "main" CACHE STRING "elastiqp git ref to fetch")

function(_elastiqp_is_source_tree dir out)
  if(EXISTS "${dir}/include/elastiqp/elastiqp.hpp" AND EXISTS "${dir}/CMakeLists.txt")
    set(${out} TRUE PARENT_SCOPE)
  else()
    set(${out} FALSE PARENT_SCOPE)
  endif()
endfunction()

set(_elastiqp_dir "${ELASTIQP_SOURCE_DIR}")
if(_elastiqp_dir STREQUAL "")
  foreach(cand "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_SOURCE_DIR}/../elastiqp")
    _elastiqp_is_source_tree("${cand}" _ok)
    if(_ok)
      get_filename_component(_elastiqp_dir "${cand}" ABSOLUTE)
      break()
    endif()
  endforeach()
else()
  _elastiqp_is_source_tree("${_elastiqp_dir}" _ok)
  if(NOT _ok)
    message(FATAL_ERROR
      "ELASTIQP_SOURCE_DIR=${_elastiqp_dir} does not look like an elastiqp checkout")
  endif()
endif()

if(_elastiqp_dir STREQUAL "")
  include(FetchContent)
  FetchContent_Declare(elastiqp
    GIT_REPOSITORY ${ELASTIQP_GIT_REPOSITORY}
    GIT_TAG ${ELASTIQP_GIT_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_GetProperties(elastiqp)
  if(NOT elastiqp_POPULATED)
    message(STATUS "elastiqp_benchmarks: fetching elastiqp ${ELASTIQP_GIT_TAG}")
    FetchContent_Populate(elastiqp)
  endif()
  set(_elastiqp_dir ${elastiqp_SOURCE_DIR})
endif()
message(STATUS "elastiqp_benchmarks: using elastiqp at ${_elastiqp_dir}")

set(ELASTIQP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ELASTIQP_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(${_elastiqp_dir} ${CMAKE_BINARY_DIR}/elastiqp EXCLUDE_FROM_ALL)
