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
