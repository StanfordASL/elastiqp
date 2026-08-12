# Self-contained ElastiQP benchmarks (no dependencies beyond Eigen and the
# dev-only headers in common/). Enable with -DELASTIQP_BUILD_BENCHMARKS=ON.
foreach(bench bench_random_qp bench_collision_2d)
  add_executable(${bench} ${bench}.cc)
  target_include_directories(${bench} PRIVATE ${CMAKE_SOURCE_DIR}/common)
  target_link_libraries(${bench} PRIVATE elastiqp::elastiqp)
endforeach()
