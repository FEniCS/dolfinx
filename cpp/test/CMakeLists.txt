cmake_minimum_required(VERSION 3.16)

project(dolfinx-tests LANGUAGES C CXX)
set(CMAKE_C_STANDARD 17) # For FFCx generated .c files.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find DOLFINx config file
find_package(DOLFINX REQUIRED)

include(FeatureSummary)

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
       "Add paths to linker search and installed rpath." ON
)

# clang-tidy
option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(ENABLE_CLANG_TIDY ENABLE_CLANG_TIDY "Run clang-tidy while building")

add_custom_command(
  OUTPUT expr.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/fem/expr.py
  VERBATIM
  DEPENDS fem/expr.py
  COMMENT "Compile expr.py using FFCx"
)

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

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

# Add some strict compiler checks only on C++ part (Developer).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

target_compile_options(
  unittests
  PRIVATE
  $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(unittests PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy")
endif()

# Enable testing
enable_testing()

# Test target
add_test(unittests unittests)
