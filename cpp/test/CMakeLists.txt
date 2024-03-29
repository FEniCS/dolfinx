cmake_minimum_required(VERSION 3.16)
project(dolfinx-tests)

project(${PROJECT_NAME} LANGUAGES C CXX)
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
  mesh/distributed_mesh.cpp
  common/CIFailure.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/poisson.c
)
target_link_libraries(unittests PRIVATE Catch2::Catch2WithMain dolfinx)
target_include_directories(
  unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_compile_options(
  unittests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror -Wextra -pedantic>
)

# Enable testing
enable_testing()

# Test target
add_test(unittests unittests)
