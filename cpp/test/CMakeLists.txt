cmake_minimum_required(VERSION 3.21)

find_package(Catch2 3)
if(NOT Catch2_FOUND)
  message(STATUS "Catch2 not found. Downloading.")
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
  )
  FetchContent_MakeAvailable(Catch2)
endif()

include(Catch)

# FFCx compile: expr.py -> expr.c
add_custom_command(
  OUTPUT expr.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/fem/expr.py
  VERBATIM
  DEPENDS fem/expr.py
  COMMENT "Compile expr.py using FFCx"
)

# FFCx compile: poisson.py -> poisson.c
add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

add_executable(
  unittests
  main.cpp
  graph.cpp
  vector.cpp
  matrix.cpp
  io.cpp
  common/CIFailure.cpp
  common/sub_systems_manager.cpp
  common/index_map.cpp
  common/sort.cpp
  fem/form.cpp
  fem/functionspace.cpp
  mesh/branching_manifold.cpp
  mesh/distributed_mesh.cpp
  mesh/generation.cpp
  mesh/read_named_meshtags.cpp
  mesh/refinement/interval.cpp
  mesh/refinement/option.cpp
  mesh/refinement/rectangle.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/expr.c
  ${CMAKE_CURRENT_BINARY_DIR}/poisson.c
)
target_link_libraries(unittests PRIVATE Catch2::Catch2WithMain dolfinx)

# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(unittests PRIVATE bcrypt)
endif()

target_include_directories(
  unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
# makes version.h accessible
target_include_directories(
  unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../>
)

set_target_properties(
  unittests
  PROPERTIES CMAKE_C_STANDARD 17
             CMAKE_CXX_STANDARD 20
             CMAKE_CXX_STANDARD_REQUIRED ON
             CMAKE_CXX_EXTENSIONS OFF
)

list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic;-g;-O2)
target_compile_options(
  unittests
  PRIVATE
    $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
)

# TODO: should not be set in test and dolfinx but in cpp once.
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    unittests
    PROPERTIES
      CXX_CLANG_TIDY
      "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy"
  )
endif()

# Nice for debugging tests one at a time - but extremely slow due to process
# spawning/tear down: catch_discover_tests(unittests)

foreach(np IN LISTS DOLFINX_TEST_MPI_PROCESSES)
  add_test(NAME unittests_np_${np} COMMAND mpiexec -np ${np} ./unittests)
endforeach()
