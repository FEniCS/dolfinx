cmake_minimum_required(VERSION 3.16)
project(dolfinx-tests)

project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_C_STANDARD 17) # For FFCx generated .c files.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find DOLFINx config file
find_package(DOLFINX REQUIRED)

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
  vector.cpp
  matrix.cpp
  io.cpp
  common/sub_systems_manager.cpp
  common/index_map.cpp
  common/sort.cpp
  fem/functionspace.cpp
  mesh/distributed_mesh.cpp
  mesh/generation.cpp
  common/CIFailure.cpp
  refinement/interval.cpp
  refinement/option.cpp
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

# Enable testing
enable_testing()

# Test target
add_test(unittests unittests)
