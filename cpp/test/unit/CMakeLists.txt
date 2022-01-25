cmake_minimum_required(VERSION 3.16)
project(dolfinx-tests)

# Find DOLFINx config file
find_package(DOLFINX REQUIRED)


add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM DEPENDS poisson.ufl COMMENT "Compile poisson.ufl using FFCx")

# Make test executable
set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/poisson.c
  ${CMAKE_CURRENT_SOURCE_DIR}/vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/sub_systems_manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/index_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/sort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/distributed_mesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/CIFailure.cpp
  )

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)


add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_executable(unittests ${TEST_SOURCES})
target_link_libraries(unittests PRIVATE Catch dolfinx)
target_compile_features(unittests PRIVATE cxx_std_17)

# Enable testing
enable_testing()

# Test target
add_test(unittests unittests)
