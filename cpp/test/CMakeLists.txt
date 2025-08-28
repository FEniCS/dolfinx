cmake_minimum_required(VERSION 3.21)

project(dolfinx-tests LANGUAGES C CXX)

if (PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing

  set(CMAKE_C_STANDARD 17) # For FFCx generated .c files.
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  find_package(DOLFINX REQUIRED)

  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic;-g;-O2)
endif()

find_package(Catch2 3 REQUIRED)

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
target_link_libraries(unittests PRIVATE Catch2::Catch2 dolfinx)

# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(unittests PRIVATE bcrypt)
endif()

# Required to find FFCx generated headers.
target_include_directories(
  unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if (NOT PROJECT_IS_TOP_LEVEL)
  # Required to find version.h
  target_include_directories(
    unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../>
  )
endif()

set_target_properties(
  unittests
  PROPERTIES CMAKE_C_STANDARD 17 # For FFCx generated .c files.
             CMAKE_CXX_STANDARD 20
             CMAKE_CXX_STANDARD_REQUIRED ON
             CMAKE_CXX_EXTENSIONS OFF
)

if (DEFINED DOLFINX_CXX_DEVELOPER_FLAGS)
  target_compile_options(
    unittests
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
  )
endif()

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    unittests
    PROPERTIES
      CXX_CLANG_TIDY
      "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy"
  )
endif()

# Uncommenting line allows for debugging individual tests, but slow.
#catch_discover_tests(unittests)

add_test(NAME unittests COMMAND $<TARGET_FILE:unittests>)
