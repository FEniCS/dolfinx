cmake_minimum_required(VERSION 3.10)
project(dolfinx-tests)

if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

# Make test executable
set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/sub_systems_manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/index_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/distributed_mesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/CIFailure.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/mesh_function.cpp
  )

  # Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_executable(unittests ${TEST_SOURCES})
target_link_libraries(unittests PRIVATE Catch dolfinx)
target_compile_features(unittests PRIVATE cxx_std_17)

# Test target
add_test(unittests unittests)
